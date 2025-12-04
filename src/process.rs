use std::{cmp::Ordering, iter::repeat_n};

use candle_core::{Context, Error, IndexOp, Result, Tensor, bail};
use either::Either;
use tokenizers::{Encoding, Tokenizer};

use crate::{
    DTYPE, ProvenceModel, ProvenceOutput,
    sentence_rounding::{
        sentence_rounding_tensor, split_and_round_sentences,
        split_sentences_and_track_from_encoding,
    },
};

pub type MultipleQuestions = Vec<String>;
pub type MultipleContexts = Vec<Vec<String>>;
pub type MultipleTitles = Vec<Vec<String>>;
pub type PreparedProcessParams = (MultipleQuestions, MultipleContexts, Option<MultipleTitles>);
pub type FlattenedQuestionsContexts = (Vec<String>, Vec<usize>, Vec<usize>, Vec<usize>);
pub type EncodedInput = (Encoding, Tensor, Tensor);

#[derive(Debug, Clone)]
pub struct ProcessedResults {
    pub pruned_context: MultipleContexts,
    pub reranking_score: Vec<Vec<f32>>,
    pub compression_rate: Vec<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct ProcessedResult {
    pub question: String,
    pub context: String,
    pub pruned_context: String,
    pub reranking_score: f32,
    pub compression_rate: f32,
}

pub mod config {
    // TODO: don't hardcode
    pub const SEPARATOR_TOKEN: &str = "[SEP]";
    pub const TITLE_PARAM_SPECIAL_VALUE: &str = "first_sentence";
    pub const MAX_LEN: usize = 512;
    // TODO: use tokens not chars
    pub const CHARS_PER_TOKEN: usize = 4;
    pub const MAX_LEN_CHARS: usize = MAX_LEN * CHARS_PER_TOKEN;
    pub const MAX_LEN_CHARS_PART: usize = (MAX_LEN_CHARS) / 2;
}

impl ProvenceModel {
    /// Process question / context with sentence-level rounding
    #[allow(clippy::too_many_arguments)]
    pub fn process(
        &self,
        tokenizer: &Tokenizer,
        // TODO: make slices?
        question: Either<MultipleQuestions, &str>,
        context: Either<MultipleContexts, &str>,
        title: Option<Either<MultipleTitles, &str>>,
        threshold: Option<f32>,
        always_select_first: Option<bool>,
        batch_size: Option<usize>,
        reorder: Option<bool>,
        top_k: Option<usize>,
        enable_warnings: Option<bool>,
    ) -> Result<ProcessedResults> {
        let (questions, contexts, titles) = Self::prepare_process_params(question, context, title)?;

        let pruned_context = Vec::with_capacity(questions.len());
        let reranking_score = Vec::with_capacity(questions.len());
        let compression_rate = Vec::with_capacity(questions.len());

        if questions.is_empty() {
            return Ok(ProcessedResults {
                pruned_context,
                reranking_score,
                compression_rate,
            });
        }

        let threshold = threshold.unwrap_or(0.1);
        let always_select_first = always_select_first.unwrap_or(true);
        let reorder = reorder.unwrap_or(false);
        let top_k = top_k.unwrap_or(5);

        // TODO implement
        let batch_size = batch_size.unwrap_or(32);
        let _enable_warnings = enable_warnings.unwrap_or(true);

        self.process_batched(
            tokenizer,
            &questions,
            &contexts,
            &titles,
            threshold,
            always_select_first,
            batch_size,
            reorder,
            top_k,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn process_batched(
        &self,
        tokenizer: &Tokenizer,
        questions: &[String],
        contexts: &[Vec<String>],
        titles: &Option<Vec<Vec<String>>>,
        threshold: f32,
        always_select_first: bool,

        batch_size: usize,
        reorder: bool,
        top_k: usize,
    ) -> Result<ProcessedResults> {
        if questions.is_empty() {
            return Ok(ProcessedResults {
                pruned_context: Vec::new(),
                reranking_score: Vec::new(),
                compression_rate: Vec::new(),
            });
        }

        let (
            pair_input_texts,
            pair_question_indices,
            pair_context_indices,
            pair_context_start_offsets,
        ) = Self::get_questions_contexts_pairs(questions, contexts, titles)?;

        if pair_input_texts.is_empty() {
            return Ok(ProcessedResults {
                pruned_context: vec![Vec::new(); questions.len()],
                reranking_score: vec![Vec::new(); questions.len()],
                compression_rate: vec![Vec::new(); questions.len()],
            });
        }

        let mut pruned_context_by_question = vec![Vec::new(); questions.len()];
        let mut reranking_by_question = vec![Vec::new(); questions.len()];
        let mut compression_by_question = vec![Vec::new(); questions.len()];

        let encodings = tokenizer
            .encode_batch(pair_input_texts, true)
            .map_err(|e| Error::msg(format!("encode_batch failed: {}", e)))?;

        let total_pair_count = encodings.len();
        let pad_id = tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);
        let separator_token_id = tokenizer
            .token_to_id(config::SEPARATOR_TOKEN)
            .context("Separator token not in tokenizer vocabulary")?;

        let mut pair_cursor = 0;

        while pair_cursor < total_pair_count {
            let chunk_end = std::cmp::min(pair_cursor + batch_size, total_pair_count);

            let encoding_chunk = encodings
                .get(pair_cursor..chunk_end)
                .context("Failed to get encoding chunk")?;

            let (input_ids, attention_mask, sequence_lens) =
                Self::batch_from_encodings(encoding_chunk, pad_id, &self.device)?;

            let output = self.forward(&input_ids, Some(attention_mask))?;

            let keep_probs_all = Self::calculate_keep_probs(&output.compression_logits)?;
            let ranking_scores = &output.ranking_scores;

            // ====================================================================
            // STEP 1: CPU-side metadata extraction (encodings only)
            // ====================================================================
            let mut batch_metadata = Vec::with_capacity(encoding_chunk.len());

            for chunk_local_index in 0..encoding_chunk.len() {
                let pair_global_index = pair_cursor + chunk_local_index;

                let sample_encoding = encoding_chunk
                    .get(chunk_local_index)
                    .context("Failed to get encoding for sample")?;

                let context_start_offset = *pair_context_start_offsets
                    .get(pair_global_index)
                    .context("Failed to get context_start_offset for sample")?;

                let source_context_text = {
                    let question_index = *pair_question_indices
                        .get(pair_global_index)
                        .context("Missing question mapping")?;

                    let context_index = *pair_context_indices
                        .get(pair_global_index)
                        .context("Missing context mapping")?;

                    contexts
                        .get(question_index)
                        .and_then(|v| v.get(context_index))
                        .context("Missing original context text")?
                        .to_owned()
                };

                let (_sentences, sentences_token_coords) = split_sentences_and_track_from_encoding(
                    &source_context_text,
                    sample_encoding.get_offsets(),
                    context_start_offset,
                )?;

                let question_index = *pair_question_indices
                    .get(pair_global_index)
                    .context("Missing question mapping")?;

                batch_metadata.push((sentences_token_coords, source_context_text, question_index));
            }

            // ====================================================================
            // STEP 2: GPU processing - build keep masks on GPU
            // NO GPU→CPU TRANSFERS in this section
            // ====================================================================
            let mut keep_mask_tensors = Vec::with_capacity(encoding_chunk.len());

            for chunk_local_index in 0..encoding_chunk.len() {
                let sample_sequence_len = *sequence_lens
                    .get(chunk_local_index)
                    .context("Failed to get sequence_len for sample")?;

                // GPU slice, no transfer
                let sample_keep_probs_tensor =
                    keep_probs_all
                        .i(chunk_local_index)?
                        .narrow(0, 0, sample_sequence_len)?;

                let (sentences_token_coords, _, _) = batch_metadata
                    .get(chunk_local_index)
                    .context("Failed to get batch metadata")?;

                // Build keep mask on GPU
                let keep_mask_tensor = sentence_rounding_tensor(
                    &sample_keep_probs_tensor,
                    sentences_token_coords,
                    threshold,
                    always_select_first,
                )?;

                keep_mask_tensors.push(keep_mask_tensor);
            }

            // ====================================================================
            // STEP 3: SINGLE BATCH GPU→CPU TRANSFER
            // Transfer all data at once to minimize sync overhead
            // ====================================================================

            // Transfer all ranking scores in one go
            let all_ranking_scores: Vec<f32> = ranking_scores.to_vec1()?;

            // Transfer all input IDs in one go
            let all_input_ids: Vec<Vec<u32>> = (0..encoding_chunk.len())
                .map(|i| input_ids.i(i)?.to_vec1())
                .collect::<Result<Vec<_>>>()?;

            // Transfer all keep masks in one go
            let all_keep_masks: Vec<Vec<bool>> = keep_mask_tensors
                .iter()
                .map(|mask| {
                    mask.to_vec1::<u8>()
                        .map(|v| v.into_iter().map(|x| x != 0).collect())
                })
                .collect::<Result<Vec<_>>>()?;

            // ====================================================================
            // STEP 4: CPU-side post-processing (decoding, etc.)
            // ====================================================================
            for chunk_local_index in 0..encoding_chunk.len() {
                let sample_reranking_score = *all_ranking_scores
                    .get(chunk_local_index)
                    .context("Failed to get ranking score")?;

                let token_ids_vec = all_input_ids
                    .get(chunk_local_index)
                    .context("Failed to get token IDs")?;

                let keep_mask = all_keep_masks
                    .get(chunk_local_index)
                    .context("Failed to get keep mask")?;

                // Find separator
                let separator_token_index = token_ids_vec
                    .iter()
                    .position(|&id| id == separator_token_id)
                    .context("separator token missing")?;

                let (_, source_context_text, question_index) = batch_metadata
                    .get(chunk_local_index)
                    .context("Failed to get batch metadata")?;

                let (kept_token_ids, _removed_token_ids) = Self::apply_keep_mask_after_skip(
                    token_ids_vec,
                    separator_token_index,
                    keep_mask,
                );

                let pruned_context = tokenizer
                    .decode(&kept_token_ids, true)
                    .map_err(|e| Error::msg(format!("Decoding failed: {}", e)))?;

                let compression_rate =
                    Self::calculate_compression_rate(source_context_text, &pruned_context);

                pruned_context_by_question
                    .get_mut(*question_index)
                    .context("Missing pruned_context bucket")?
                    .push(pruned_context);

                reranking_by_question
                    .get_mut(*question_index)
                    .context("Missing reranking bucket")?
                    .push(sample_reranking_score);

                compression_by_question
                    .get_mut(*question_index)
                    .context("Missing compression bucket")?
                    .push(compression_rate);
            }

            pair_cursor = chunk_end;
        }

        // Reorder logic
        if reorder {
            for question_i in 0..questions.len() {
                let contexts_for_question = pruned_context_by_question
                    .get_mut(question_i)
                    .context("Missing pruned_context bucket for reorder")?;

                let reranking_scores_for_question = reranking_by_question
                    .get_mut(question_i)
                    .context("Missing reranking bucket for reorder")?;

                let compression_rates_for_question = compression_by_question
                    .get_mut(question_i)
                    .context("Missing compression bucket for reorder")?;

                let mut sorted_indices: Vec<usize> =
                    (0..reranking_scores_for_question.len()).collect();

                sorted_indices.sort_by(|&a, &b| {
                    let a_val = *reranking_scores_for_question.get(a).unwrap_or(&0.0);
                    let b_val = *reranking_scores_for_question.get(b).unwrap_or(&0.0);

                    b_val.partial_cmp(&a_val).unwrap_or(Ordering::Equal)
                });

                let sorted_indices: Vec<usize> = sorted_indices.into_iter().take(top_k).collect();

                let top_contexts: Vec<String> = sorted_indices
                    .iter()
                    .filter_map(|&id| contexts_for_question.get(id).cloned())
                    .collect();

                let top_reranking_scores: Vec<f32> = sorted_indices
                    .iter()
                    .filter_map(|&id| reranking_scores_for_question.get(id).copied())
                    .collect();

                let top_compression_rates: Vec<f32> = sorted_indices
                    .iter()
                    .filter_map(|&id| compression_rates_for_question.get(id).copied())
                    .collect();

                *contexts_for_question = top_contexts;
                *reranking_scores_for_question = top_reranking_scores;
                *compression_rates_for_question = top_compression_rates;
            }
        }

        Ok(ProcessedResults {
            pruned_context: pruned_context_by_question,
            reranking_score: reranking_by_question,
            compression_rate: compression_by_question,
        })
    }

    pub fn process_question_context(
        &self,
        tokenizer: &Tokenizer,
        question: &str,
        context: &str,
        context_start_offset: usize,
        threshold: f32,
        always_select_first: bool,
    ) -> Result<ProcessedResult> {
        let mut input_text = Self::apply_template(question, context);
        Self::truncate_warn(&mut input_text, "Process input", config::MAX_LEN_CHARS);

        let (encoding, input_ids, attention_mask) = self.encode_input(tokenizer, &input_text)?;
        let tokens = encoding.get_ids();
        let separator_index =
            Self::get_separator_index(&encoding).context("separator token missing")?;

        let output = self.forward(&input_ids, Some(attention_mask))?;

        let ranking_scores = output.ranking_scores.to_vec1::<f32>()?;
        let reranking_score = ranking_scores
            .first()
            .copied()
            .context("ranking_scores was empty")?;

        let keep_probs = Self::get_keep_probabilities(&output)?;
        let keep_mask = split_and_round_sentences(
            context,
            encoding.get_offsets(),
            context_start_offset,
            &keep_probs,
            threshold,
            always_select_first,
        )?;

        let (kept_token_ids, _removed_token_ids) =
            Self::apply_keep_mask_after_skip(tokens, separator_index, &keep_mask);

        let pruned_context = tokenizer
            .decode(&kept_token_ids, true)
            .map_err(|e| Error::msg(format!("Decoding failed: {}", e)))?;

        let compression_rate = Self::calculate_compression_rate(context, &pruned_context);

        Ok(ProcessedResult {
            question: question.to_owned(),
            context: context.to_owned(),
            pruned_context,
            reranking_score,
            compression_rate,
        })
    }

    fn encode_input(&self, tokenizer: &Tokenizer, input_text: &str) -> Result<EncodedInput> {
        let encoding = tokenizer
            .encode(input_text, true)
            .map_err(|e| Error::msg(format!("Tokenization failed: {}", e)))?;

        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;

        let attention_mask =
            Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;

        Ok((encoding, input_ids, attention_mask))
    }

    pub fn apply_template(question: &str, context: &str) -> String {
        format!("{} {} {}", question, config::SEPARATOR_TOKEN, context)
    }

    fn prepare_process_params(
        question: Either<MultipleQuestions, &str>,
        context: Either<MultipleContexts, &str>,
        title: Option<Either<MultipleTitles, &str>>,
    ) -> Result<PreparedProcessParams> {
        let mut questions = match question {
            Either::Left(q) => q,
            Either::Right(q) => vec![q.to_owned()],
        };

        let mut contexts = match context {
            Either::Left(c) => c,
            Either::Right(c) => vec![vec![c.to_owned()]],
        };

        let mut titles = title.and_then(|t| match t {
            Either::Left(t) => Some(t),
            Either::Right(s) => {
                if s == config::TITLE_PARAM_SPECIAL_VALUE {
                    None
                } else {
                    Some(vec![vec![s.to_owned()]])
                }
            }
        });

        if questions.len() != contexts.len() {
            bail!("'questions' and 'contexts' must have same lengths")
        }

        for (question_i, question) in questions.iter_mut().enumerate() {
            if question.is_empty() {
                bail!("Question '{}' cannot be empty", question_i)
            }

            Self::truncate_warn(
                question,
                &format!("Question {}", question_i),
                config::MAX_LEN_CHARS_PART,
            );
        }

        match titles {
            Some(ref mut titles) => {
                if titles.len() != questions.len() {
                    bail!("'titles' must be a list of strings of the same length as 'questions'")
                }

                for (inner_contexts_index, (inner_contexts, inner_titles)) in
                    contexts.iter_mut().zip(titles.iter_mut()).enumerate()
                {
                    let mut new_inner_contexts = Vec::with_capacity(inner_contexts.len());
                    let mut new_inner_titles = Vec::with_capacity(inner_titles.len());

                    let inner_titles_len = inner_titles.len();
                    let inner_contexts_len = inner_contexts.len();

                    if inner_titles_len != inner_contexts_len {
                        bail!(
                            "Each list in 'titles' must have the same length as the corresponding list in 'context': context list {} has length {} but title list has length {}",
                            inner_contexts_index,
                            inner_titles_len,
                            inner_contexts_len
                        );
                    }

                    for (title_i, title) in inner_titles.iter_mut().enumerate() {
                        Self::truncate_warn(
                            title,
                            &format!("Title {}", title_i),
                            config::MAX_LEN_CHARS_PART,
                        );
                    }

                    for (context_index, (mut context, title)) in inner_contexts
                        .drain(..)
                        .zip(inner_titles.drain(..))
                        .enumerate()
                    {
                        let split_result = Self::split_warn(
                            &mut context,
                            &format!("Context [{}][{}]", inner_contexts_index, context_index),
                            config::MAX_LEN_CHARS_PART,
                        );

                        // Need to dupe titles to match the new split contexts
                        new_inner_contexts.push(context);
                        new_inner_titles.push(title.clone());

                        if let Some(splits) = split_result {
                            for split_chunk in splits {
                                new_inner_contexts.push(split_chunk);
                                new_inner_titles.push(title.clone());
                            }
                        }
                    }

                    *inner_contexts = new_inner_contexts;
                    *inner_titles = new_inner_titles;
                }
            }
            None => {
                for (inner_contexts_index, inner_contexts) in contexts.iter_mut().enumerate() {
                    let mut new_inner_contexts = Vec::with_capacity(inner_contexts.len());

                    for (context_index, mut context_str) in inner_contexts.drain(..).enumerate() {
                        let split_result = Self::split_warn(
                            &mut context_str,
                            &format!("Context [{}][{}]", inner_contexts_index, context_index),
                            config::MAX_LEN_CHARS_PART,
                        );

                        new_inner_contexts.push(context_str);

                        if let Some(splits) = split_result {
                            new_inner_contexts.extend(splits);
                        }
                    }

                    *inner_contexts = new_inner_contexts;
                }
            }
        }

        Ok((questions, contexts, titles))
    }

    fn split_warn(text: &mut String, label: &str, max_len: usize) -> Option<Vec<String>> {
        let split_byte_index = text.char_indices().nth(max_len).map(|(index, _)| index);

        match split_byte_index {
            None => None,
            Some(split_index) => {
                let preview_num_chars = 20;
                let preview: String = text.chars().take(preview_num_chars).collect();

                eprintln!(
                    "WARNING: {} '{}...' (len {}) exceeded max len {}. Splitting.",
                    label,
                    preview,
                    text.chars().count(),
                    max_len
                );

                let mut remainder = text.split_off(split_index);
                let mut chunks = Vec::new();

                loop {
                    match remainder.char_indices().nth(max_len) {
                        Some((next_split_index, _)) => {
                            let tail = remainder.split_off(next_split_index);
                            chunks.push(remainder);
                            remainder = tail;
                        }
                        None => {
                            chunks.push(remainder);
                            break;
                        }
                    }
                }

                Some(chunks)
            }
        }
    }

    fn truncate_warn(text: &mut String, label: &str, max_len: usize) {
        if let Some((byte_index, _)) = text.char_indices().nth(max_len) {
            let preview_num_chars = 20;
            let preview: String = text.chars().take(preview_num_chars).collect();

            eprintln!(
                "WARNING: {} '{}...' (len {}) exceeded max len {}. Truncating.",
                label,
                preview,
                text.chars().count(),
                max_len
            );

            text.truncate(byte_index);
        }
    }

    fn get_questions_contexts_pairs(
        questions: &[String],
        contexts: &[Vec<String>],
        titles: &Option<Vec<Vec<String>>>,
    ) -> Result<FlattenedQuestionsContexts> {
        let mut pair_input_texts = Vec::new();
        let mut pair_question_indices = Vec::new();
        let mut pair_context_indices = Vec::new();
        let mut pair_context_start_offset_bytes = Vec::new();

        for (question_i, question) in questions.iter().enumerate() {
            let normalize_question = true;

            let normalized_question = if normalize_question {
                Self::normalize_string(question)
            } else {
                question.to_string()
            };

            let question_contexts = contexts.get(question_i).context(format!(
                "Couldn't get contexts for question index {question_i}"
            ))?;

            for (context_i, context) in question_contexts.iter().enumerate() {
                let context = match titles {
                    Some(titles) => {
                        let context_title = titles
                            .get(question_i)
                            .context(format!(
                                "Couldn't get outer titles vec at index {question_i}"
                            ))?
                            .get(context_i)
                            .context(format!(
                                "Couldn't get inner title value at index {context_i}"
                            ))?;

                        format!("{context_title} {context}")
                    }
                    None => context.to_owned(),
                };

                let input = Self::format_input(&normalized_question, context.as_str());
                let context_start =
                    normalized_question.len() + 1 + config::SEPARATOR_TOKEN.len() + 1;

                pair_input_texts.push(input);
                pair_question_indices.push(question_i);
                pair_context_indices.push(context_i);
                pair_context_start_offset_bytes.push(context_start);
            }
        }

        Ok((
            pair_input_texts,
            pair_question_indices,
            pair_context_indices,
            pair_context_start_offset_bytes,
        ))
    }

    fn batch_from_encodings(
        encodings: &[Encoding],
        pad_id: u32,
        device: &candle_core::Device,
    ) -> Result<(Tensor, Tensor, Vec<usize>)> {
        let batch_size = encodings.len();

        if batch_size == 0 {
            let empty = Tensor::zeros(0, DTYPE, device)?.reshape((0, 0))?;

            return Ok((empty.clone(), empty, Vec::new()));
        }

        let sequence_lens: Vec<usize> = encodings.iter().map(|e| e.len()).collect();

        let max_len = *sequence_lens
            .iter()
            .max()
            .context("batch_from_encodings: couldn't get max_len")?;

        let mut all_ids = Vec::with_capacity(batch_size * max_len);
        let mut all_masks = Vec::with_capacity(batch_size * max_len);

        for encoding in encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let encoding_len = ids.len();
            let padding_amount = max_len - encoding_len;

            all_ids.extend_from_slice(ids);
            all_ids.extend(repeat_n(pad_id, padding_amount));

            all_masks.extend_from_slice(mask);
            all_masks.extend(repeat_n(0, padding_amount));
        }

        let input_ids = Tensor::from_vec(all_ids, (batch_size, max_len), device)?;
        let attention_mask = Tensor::from_vec(all_masks, (batch_size, max_len), device)?;

        Ok((input_ids, attention_mask, sequence_lens))
    }

    fn calculate_keep_probs(compression_logits: &Tensor) -> Result<Tensor> {
        let keep_prob_index = 1;

        let compression_dims = compression_logits.dims().len();
        let softmax_axis = compression_dims - 1;
        let compression_probs = candle_nn::ops::softmax(compression_logits, softmax_axis)?;

        compression_probs.i((.., .., keep_prob_index))
    }

    fn normalize_string(text: &str) -> String {
        let lower_no_punctuation: String = text
            .to_lowercase()
            .chars()
            .filter(|c| !c.is_ascii_punctuation())
            .collect();

        lower_no_punctuation
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
    }

    fn get_separator_index(encoding: &Encoding) -> Option<usize> {
        encoding
            .get_tokens()
            .iter()
            .position(|t| t == config::SEPARATOR_TOKEN)
    }

    fn get_keep_probabilities(output: &ProvenceOutput) -> Result<Vec<f32>> {
        let compression_logits = output.compression_logits.squeeze(0)?;
        let compression_probs = candle_nn::ops::softmax(&compression_logits, 1)?;

        let keep_probs = compression_probs.i((.., 1))?;
        let keep_probs_vec = keep_probs.to_vec1::<f32>()?;

        Ok(keep_probs_vec)
    }

    fn apply_keep_mask_after_skip(
        tokens: &[u32],
        skip: usize,
        keep_mask: &[bool],
    ) -> (Vec<u32>, Vec<u32>) {
        let mut kept_token_ids = Vec::new();
        let mut removed_token_ids = Vec::new();

        for (i, &token_id) in tokens.iter().enumerate().skip(skip + 1) {
            if keep_mask.get(i) == Some(&true) {
                kept_token_ids.push(token_id);
            } else {
                removed_token_ids.push(token_id);
            }
        }

        (kept_token_ids, removed_token_ids)
    }

    fn calculate_compression_rate(context: &str, pruned_context: &str) -> f32 {
        let context_len = context.chars().count() as f32;

        if context_len > 0.0 {
            (1.0 - pruned_context.chars().count() as f32 / context_len) * 100.0
        } else {
            0.0
        }
    }
}
