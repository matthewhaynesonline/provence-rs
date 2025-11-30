use std::{cmp::Ordering, iter::repeat_n};

use candle_core::{Context, Error, IndexOp, Result, Tensor, bail};
use either::Either;
use tokenizers::{Encoding, Tokenizer};

use super::{
    DTYPE, ProvenceModel, ProvenceOutput,
    sentence_rounding::{
        SentenceRoundingMode, sentence_rounding, split_sentences_and_track_from_encoding,
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
}

impl ProvenceModel {
    /// Process question / context with sentence-level rounding
    #[allow(clippy::too_many_arguments)]
    pub fn process(
        &self,
        tokenizer: &Tokenizer,
        question: Either<MultipleQuestions, &str>,
        context: Either<MultipleContexts, &str>,
        title: Option<Either<MultipleTitles, &str>>,
        threshold: Option<f32>,
        always_select_first: Option<bool>,
        batch_size: Option<usize>,
        reorder: Option<bool>,
        top_k: Option<usize>,
        enable_warnings: Option<bool>,
        rounding_mode: Option<SentenceRoundingMode>,
    ) -> Result<ProcessedResults> {
        let (questions, contexts, _titles) =
            Self::prepare_process_params(question, context, title)?;

        let threshold = threshold.unwrap_or(0.1);
        let always_select_first = always_select_first.unwrap_or(true);
        let batch_size = batch_size.unwrap_or(32);
        let reorder = reorder.unwrap_or(false);
        let top_k = top_k.unwrap_or(5);
        let _enable_warnings = enable_warnings.unwrap_or(true);
        let rounding_mode = rounding_mode.unwrap_or(SentenceRoundingMode::DecisionAverage);

        self.process_batched(
            tokenizer,
            &questions,
            &contexts,
            threshold,
            always_select_first,
            rounding_mode,
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
        threshold: f32,
        always_select_first: bool,
        rounding_mode: SentenceRoundingMode,
        batch_size: usize,
        reorder: bool,
        top_k: usize,
    ) -> Result<ProcessedResults> {
        let start = std::time::Instant::now();

        if questions.is_empty() {
            return Ok(ProcessedResults {
                pruned_context: Vec::new(),
                reranking_score: Vec::new(),
                compression_rate: Vec::new(),
            });
        }

        // Flatten question and context into input pairs for batching
        let (
            pair_input_texts,
            pair_question_indices,
            pair_context_indices,
            pair_context_start_offsets,
        ) = Self::get_questions_contexts_pairs(questions, contexts)?;

        if pair_input_texts.is_empty() {
            return Ok(ProcessedResults {
                pruned_context: vec![Vec::new(); questions.len()],
                reranking_score: vec![Vec::new(); questions.len()],
                compression_rate: vec![Vec::new(); questions.len()],
            });
        }

        println!(
            "get_questions_contexts_pairs time elapsed: {:?}",
            start.elapsed()
        );

        let mut pruned_context_by_question = vec![Vec::new(); questions.len()];
        let mut reranking_by_question = vec![Vec::new(); questions.len()];
        let mut compression_by_question = vec![Vec::new(); questions.len()];

        // Encode all pairs
        let encodings = tokenizer
            .encode_batch(pair_input_texts, true)
            .map_err(|e| Error::msg(format!("encode_batch failed: {}", e)))?;

        println!("encode_batch time elapsed: {:?}", start.elapsed());

        let total_pair_count = encodings.len();
        let pad_id = tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);
        let mut pair_cursor = 0;

        while pair_cursor < total_pair_count {
            let chunk_end = std::cmp::min(pair_cursor + batch_size, total_pair_count);

            let encoding_chunk = encodings
                .get(pair_cursor..chunk_end)
                .context("Failed to get encoding chunk")?;

            let (input_ids, attention_mask, sequence_lens) =
                Self::batch_from_encodings(encoding_chunk, pad_id, &self.device)?;

            println!(
                "while batch_from_encodings pair_cursor {} elapsed: {:?}",
                pair_cursor,
                start.elapsed()
            );

            let output = self.forward(&input_ids, Some(attention_mask))?;

            println!(
                "while output pair_cursor {} elapsed: {:?}",
                pair_cursor,
                start.elapsed()
            );

            let keep_probs_all = Self::calculate_keep_probs(&output.compression_logits)?;
            let ranking_scores = &output.ranking_scores;

            println!(
                "while calculate_keep_probs pair_cursor {} elapsed: {:?}",
                pair_cursor,
                start.elapsed()
            );

            for chunk_local_index in 0..encoding_chunk.len() {
                let pair_global_index = pair_cursor + chunk_local_index;

                let sample_sequence_len = *sequence_lens
                    .get(chunk_local_index)
                    .context("Failed to get sequence_len for sample")?;

                println!(
                    "for while sample_sequence_len chunk_local_index {} pair_cursor {} elapsed: {:?}",
                    chunk_local_index,
                    pair_cursor,
                    start.elapsed()
                );

                let sample_keep_probs: Vec<f32> = keep_probs_all
                    .i(chunk_local_index)?
                    .to_vec1()?
                    .into_iter()
                    .take(sample_sequence_len)
                    .collect();

                println!(
                    "for while sample_keep_probs chunk_local_index {} pair_cursor {} elapsed: {:?}",
                    chunk_local_index,
                    pair_cursor,
                    start.elapsed()
                );

                let sample_reranking_score = ranking_scores.i(chunk_local_index)?.to_vec0()?;

                let sample_encoding = encoding_chunk
                    .get(chunk_local_index)
                    .context("Failed to get encoding for sample")?;

                let encoding_tokens = sample_encoding.get_ids();
                let separator_token_index = Self::get_separator_index(sample_encoding)
                    .context("separator token missing")?;

                println!(
                    "for while get_separator_index chunk_local_index {} pair_cursor {} elapsed: {:?}",
                    chunk_local_index,
                    pair_cursor,
                    start.elapsed()
                );

                let context_start_offset = pair_context_start_offsets
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
                    sample_encoding,
                    *context_start_offset,
                )?;

                let keep_mask = sentence_rounding(
                    &sample_keep_probs,
                    &sentences_token_coords,
                    threshold,
                    always_select_first,
                    &rounding_mode,
                )?;

                let (kept_token_ids, _removed_token_ids) =
                    Self::group_context_tokens(encoding_tokens, separator_token_index, &keep_mask);

                let pruned_context = tokenizer
                    .decode(&kept_token_ids, true)
                    .map_err(|e| Error::msg(format!("Decoding failed: {}", e)))?;

                let compression_rate =
                    Self::calculate_compression_rate(&source_context_text, &pruned_context);

                let question_index = *pair_question_indices
                    .get(pair_global_index)
                    .context("Missing question mapping when appending results")?;

                pruned_context_by_question
                    .get_mut(question_index)
                    .context("Missing pruned_context bucket")?
                    .push(pruned_context);

                reranking_by_question
                    .get_mut(question_index)
                    .context("Missing reranking bucket")?
                    .push(sample_reranking_score);

                compression_by_question
                    .get_mut(question_index)
                    .context("Missing compression bucket")?
                    .push(compression_rate);
            }

            pair_cursor = chunk_end;
        }

        println!("post while time elapsed: {:?}", start.elapsed());

        // optional reorder
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

        println!("post reorder elapsed: {:?}", start.elapsed());

        Ok(ProcessedResults {
            pruned_context: pruned_context_by_question,
            reranking_score: reranking_by_question,
            compression_rate: compression_by_question,
        })
    }

    pub fn encode_input(&self, tokenizer: &Tokenizer, input_text: &str) -> Result<EncodedInput> {
        let encoding = tokenizer
            .encode(input_text, true)
            .map_err(|e| Error::msg(format!("Tokenization failed: {}", e)))?;

        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;

        let attention_mask =
            Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;

        Ok((encoding, input_ids, attention_mask))
    }

    pub fn format_input(question: &str, context: &str) -> String {
        format!("{} {} {}", question, config::SEPARATOR_TOKEN, context)
    }

    pub fn prepare_process_params(
        question: Either<MultipleQuestions, &str>,
        context: Either<MultipleContexts, &str>,
        title: Option<Either<MultipleTitles, &str>>,
    ) -> Result<PreparedProcessParams> {
        // Convert input format into questions of Vec[str] and contexts/titles of Vec[Vec[str]]
        let questions = match question {
            Either::Left(q) => q,
            Either::Right(q) => vec![q.to_owned()],
        };

        let contexts = match context {
            Either::Left(c) => c,
            Either::Right(c) => vec![vec![c.to_owned()]],
        };

        let titles = title.and_then(|t| match t {
            Either::Left(t) => Some(t),
            Either::Right(s) => {
                if s == config::TITLE_PARAM_SPECIAL_VALUE {
                    None
                } else {
                    Some(vec![vec![s.to_owned()]])
                }
            }
        });

        if let Some(ref titles) = titles {
            if titles.len() != questions.len() {
                bail!("'titles' must be a list of strings of the same length as 'questions'")
            }

            for (titles_item, contexts_item) in titles.iter().zip(contexts.iter()) {
                if titles_item.len() != contexts_item.len() {
                    bail!(
                        "Each list in 'titles' must have the same length as the corresponding list in 'context'"
                    )
                }
            }
        }

        if questions.len() != contexts.len() {
            bail!("'questions' and 'contexts' must have same lengths")
        }

        Ok((questions, contexts, titles))
    }

    fn get_questions_contexts_pairs(
        questions: &[String],
        contexts: &[Vec<String>],
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
                let input = Self::format_input(&normalized_question, context);
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

    // TODO: use tokenizers?
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

    pub fn get_separator_index(encoding: &Encoding) -> Option<usize> {
        encoding
            .get_tokens()
            .iter()
            .position(|t| t == config::SEPARATOR_TOKEN)
    }

    pub fn get_keep_probabilities(output: &ProvenceOutput) -> Result<Vec<f32>> {
        let compression_logits = output.compression_logits.squeeze(0)?;
        let compression_probs = candle_nn::ops::softmax(&compression_logits, 1)?;

        let keep_probs = compression_probs.i((.., 1))?;
        let keep_probs_vec = keep_probs.to_vec1::<f32>()?;

        Ok(keep_probs_vec)
    }

    pub fn group_context_tokens(
        tokens: &[u32],
        separator_index: usize,
        keep_mask: &[bool],
    ) -> (Vec<u32>, Vec<u32>) {
        let mut kept_token_ids = Vec::new();
        let mut removed_token_ids = Vec::new();

        for (i, &token_id) in tokens.iter().enumerate().skip(separator_index + 1) {
            if keep_mask.get(i).copied().unwrap_or(false) {
                kept_token_ids.push(token_id);
            } else {
                removed_token_ids.push(token_id);
            }
        }

        (kept_token_ids, removed_token_ids)
    }

    pub fn calculate_compression_rate(context: &str, pruned_context: &str) -> f32 {
        let context_len = context.chars().count() as f32;

        if context_len > 0.0 {
            (1.0 - pruned_context.chars().count() as f32 / context_len) * 100.0
        } else {
            0.0
        }
    }
}
