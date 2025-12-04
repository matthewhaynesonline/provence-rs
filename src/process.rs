use candle_core::{Context, Error, IndexOp, Result, Tensor, bail};
use either::Either;
use tokenizers::{Encoding, Tokenizer};

use crate::{ProvenceModel, ProvenceOutput, sentence_rounding::split_and_round_sentences};

pub type MultipleQuestions = Vec<String>;
pub type MultipleContexts = Vec<Vec<String>>;
pub type MultipleTitles = Vec<Vec<String>>;
pub type PreparedProcessParams = (MultipleQuestions, MultipleContexts, Option<MultipleTitles>);
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

        let mut pruned_context = Vec::with_capacity(questions.len());
        let mut reranking_score = Vec::with_capacity(questions.len());
        let mut compression_rate = Vec::with_capacity(questions.len());

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
        let _batch_size = batch_size.unwrap_or(32);
        let _enable_warnings = enable_warnings.unwrap_or(true);

        for (question_i, question) in questions.iter().enumerate() {
            let question_contexts = contexts.get(question_i).context(format!(
                "Couldn't get contexts for index {question_i} value {question}",
            ))?;

            // TODO: check python implementation
            let normalize_question = true;

            let normalized_question = if normalize_question {
                Self::normalize_string(question)
            } else {
                question.to_string()
            };

            let context_start_offset =
                normalized_question.len() + 1 + config::SEPARATOR_TOKEN.len() + 1;

            let mut context_buffer = Vec::with_capacity(question_contexts.len());
            let mut reranking_buffer = Vec::with_capacity(question_contexts.len());
            let mut compression_buffer = Vec::with_capacity(question_contexts.len());

            for (context_i, context) in question_contexts.iter().enumerate() {
                let context = match titles {
                    Some(ref titles) => {
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

                let result = self.process_question_context(
                    tokenizer,
                    &normalized_question,
                    context.as_str(),
                    context_start_offset,
                    threshold,
                    always_select_first,
                )?;

                context_buffer.push(result.pruned_context);
                reranking_buffer.push(result.reranking_score);
                compression_buffer.push(result.compression_rate);
            }

            if reorder {
                let mut combined: Vec<_> = reranking_buffer
                    .into_iter()
                    .zip(context_buffer) // yields (score, context)
                    .zip(compression_buffer) // yields ((score, context), compress)
                    .map(|((rerank_score, context), compression)| {
                        (rerank_score, context, compression)
                    }) // flatten the tuple structure
                    .collect();

                combined.sort_by(|(rerank_a, _, _), (rerank_b, _, _)| {
                    rerank_b
                        .partial_cmp(rerank_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                if combined.len() > top_k {
                    combined.truncate(top_k);
                }

                let combined_len = combined.len();
                let mut new_rerank = Vec::with_capacity(combined_len);
                let mut new_context = Vec::with_capacity(combined_len);
                let mut new_compress = Vec::with_capacity(combined_len);

                for (rerank_score, context, compression) in combined {
                    new_rerank.push(rerank_score);
                    new_context.push(context);
                    new_compress.push(compression);
                }

                reranking_buffer = new_rerank;
                context_buffer = new_context;
                compression_buffer = new_compress;
            }

            pruned_context.push(context_buffer);
            reranking_score.push(reranking_buffer);
            compression_rate.push(compression_buffer);
        }

        Ok(ProcessedResults {
            pruned_context,
            reranking_score,
            compression_rate,
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
