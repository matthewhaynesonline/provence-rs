use std::{cmp::Ordering, iter::repeat_n};

use candle_core::{Context, Error, IndexOp, Result, Tensor, bail};
use either::Either;
use tokenizers::{Encoding, Tokenizer};

use crate::sentence_rounding::Coordinate;

use super::{
    DTYPE, ProvenceModel, ProvenceOutput,
    sentence_rounding::{
        SentenceRoundingMode, sentence_rounding_tensor, split_sentences_and_track_from_encoding,
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
    pub const SEPARATOR_TOKEN: &str = "[SEP]";
    pub const TITLE_PARAM_SPECIAL_VALUE: &str = "first_sentence";
}

// Structure to hold CPU metadata
struct SampleMetadata {
    sentences_token_coords: Vec<Coordinate>,
    source_context_text: String,
    question_index: usize,
    token_ids_vec: Vec<u32>,
    separator_token_index: usize,
}

impl ProvenceModel {
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
        let _batch_size = batch_size.unwrap_or(1);
        let reorder = reorder.unwrap_or(false);
        let top_k = top_k.unwrap_or(5);
        let _enable_warnings = enable_warnings.unwrap_or(true);
        let rounding_mode = rounding_mode.unwrap_or(SentenceRoundingMode::DecisionAverage);

        self.process_one_at_a_time(
            tokenizer,
            &questions,
            &contexts,
            threshold,
            always_select_first,
            rounding_mode,
            reorder,
            top_k,
        )
    }

    /// Process contexts one at a time to force Metal command buffer execution between each
    #[allow(clippy::too_many_arguments)]
    fn process_one_at_a_time(
        &self,
        tokenizer: &Tokenizer,
        questions: &[String],
        contexts: &[Vec<String>],
        threshold: f32,
        always_select_first: bool,
        rounding_mode: SentenceRoundingMode,
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

        let mut pruned_context_by_question = vec![Vec::new(); questions.len()];
        let mut reranking_by_question = vec![Vec::new(); questions.len()];
        let mut compression_by_question = vec![Vec::new(); questions.len()];

        // Process each question's contexts one at a time
        for (question_idx, question) in questions.iter().enumerate() {
            let question_contexts = contexts
                .get(question_idx)
                .context(format!("Missing contexts for question {}", question_idx))?;

            for (context_idx, context) in question_contexts.iter().enumerate() {
                println!("\n=== Processing Q{} C{} ===", question_idx, context_idx);

                // Process single context
                let single_result = self.process_single_context(
                    tokenizer,
                    question,
                    context,
                    threshold,
                    always_select_first,
                    &rounding_mode,
                )?;

                pruned_context_by_question
                    .get_mut(question_idx)
                    .context("Missing pruned_context bucket")?
                    .push(single_result.pruned_context);

                reranking_by_question
                    .get_mut(question_idx)
                    .context("Missing reranking bucket")?
                    .push(single_result.reranking_score);

                compression_by_question
                    .get_mut(question_idx)
                    .context("Missing compression bucket")?
                    .push(single_result.compression_rate);

                println!("  elapsed: {:?}", start.elapsed());
            }
        }

        println!("\nAll contexts complete: {:?}", start.elapsed());

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

        println!("post_reorder elapsed: {:?}", start.elapsed());
        println!("TOTAL elapsed: {:?}", start.elapsed());

        Ok(ProcessedResults {
            pruned_context: pruned_context_by_question,
            reranking_score: reranking_by_question,
            compression_rate: compression_by_question,
        })
    }

    /// Process a single question-context pair
    #[allow(clippy::indexing_slicing)]
    fn process_single_context(
        &self,
        tokenizer: &Tokenizer,
        question: &str,
        context: &str,
        threshold: f32,
        always_select_first: bool,
        rounding_mode: &SentenceRoundingMode,
    ) -> Result<ProcessedResult> {
        let normalized_question = Self::normalize_string(question);
        let input_text = Self::format_input(&normalized_question, context);

        // Tokenize
        let encoding = tokenizer
            .encode(input_text, true)
            .map_err(|e| Error::msg(format!("Tokenization failed: {}", e)))?;

        let context_start_offset =
            normalized_question.len() + 1 + config::SEPARATOR_TOKEN.len() + 1;

        // Extract sentence coordinates
        let (_sentences, sentences_token_coords) =
            split_sentences_and_track_from_encoding(context, &encoding, context_start_offset)?;

        let token_ids_vec: Vec<u32> = encoding.get_ids().to_vec();
        let separator_token_id = tokenizer
            .token_to_id(config::SEPARATOR_TOKEN)
            .context("Separator token not in tokenizer vocabulary")?;

        let separator_token_index = token_ids_vec
            .iter()
            .position(|&id| id == separator_token_id)
            .context("separator token missing")?;

        // Create tensors
        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;

        // Forward pass
        let output = self.forward(&input_ids, Some(attention_mask))?;

        // Calculate keep probabilities
        let keep_probs_all = Self::calculate_keep_probs(&output.compression_logits)?;
        let sample_keep_probs = keep_probs_all.i(0)?.narrow(0, 0, encoding.len())?;

        // Sentence rounding
        let keep_mask_tensor = sentence_rounding_tensor(
            &sample_keep_probs,
            &sentences_token_coords,
            threshold,
            always_select_first,
            rounding_mode,
        )?;

        // Transfer from GPU (forces execution)
        let reranking_score = output.ranking_scores.to_vec1::<f32>()?[0];
        let keep_mask: Vec<bool> = keep_mask_tensor
            .to_vec1::<u8>()?
            .into_iter()
            .map(|x| x != 0)
            .collect();

        // Filter tokens
        let (kept_token_ids, _removed_token_ids) =
            Self::group_context_tokens(&token_ids_vec, separator_token_index, &keep_mask);

        // Decode
        let pruned_context = tokenizer
            .decode(&kept_token_ids, true)
            .map_err(|e| Error::msg(format!("Decoding failed: {}", e)))?;

        let compression_rate = Self::calculate_compression_rate(context, &pruned_context);

        Ok(ProcessedResult {
            question: question.to_string(),
            context: context.to_string(),
            pruned_context,
            reranking_score,
            compression_rate,
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
