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
    /// Process query / context with sentence-level rounding
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
        rounding_mode: Option<SentenceRoundingMode>,
    ) -> Result<ProcessedResults> {
        let (queries, contexts, _titles) = Self::prepare_process_params(question, context, title)?;

        let threshold = threshold.unwrap_or(0.1);
        let always_select_first = always_select_first.unwrap_or(true);
        let _batch_size = batch_size.unwrap_or(32);
        let reorder = reorder.unwrap_or(false);
        let top_k = top_k.unwrap_or(5);
        let _enable_warnings = enable_warnings.unwrap_or(true);
        let rounding_mode = rounding_mode.unwrap_or(SentenceRoundingMode::DecisionAverage);

        let mut pruned_context = Vec::with_capacity(queries.len());
        let mut reranking_score = Vec::with_capacity(queries.len());
        let mut compression_rate = Vec::with_capacity(queries.len());

        for (question_i, question) in queries.iter().enumerate() {
            let question_contexts = contexts.get(question_i).context(format!(
                "Couldn't get contexts for index {question_i} value {question}",
            ))?;

            let results = self.process_question_batched(
                tokenizer,
                question,
                question_contexts,
                threshold,
                always_select_first,
                rounding_mode.clone(),
            )?;

            let mut context_buffer = Vec::with_capacity(question_contexts.len());
            let mut reranking_buffer = Vec::with_capacity(question_contexts.len());
            let mut compression_buffer = Vec::with_capacity(question_contexts.len());

            for result in results.into_iter() {
                context_buffer.push(result.pruned_context);
                reranking_buffer.push(result.reranking_score);
                compression_buffer.push(result.compression_rate);
            }

            if reorder {
                let mut reranked_ids: Vec<usize> = (0..reranking_buffer.len()).collect();

                reranked_ids.sort_by(|&a, &b| {
                    let reranking_buffer_a = match reranking_buffer.get(a) {
                        Some(x) => x,
                        None => return Ordering::Equal,
                    };

                    let reranking_buffer_b = match reranking_buffer.get(b) {
                        Some(x) => x,
                        None => return Ordering::Equal,
                    };

                    reranking_buffer_b
                        .partial_cmp(reranking_buffer_a)
                        .unwrap_or(Ordering::Equal)
                });

                let reranked_ids: Vec<usize> = reranked_ids.into_iter().take(top_k).collect();

                let mut new_context: Vec<String> = Vec::with_capacity(top_k);
                let mut new_reranking = Vec::with_capacity(top_k);
                let mut new_compression = Vec::with_capacity(top_k);

                for reranked_id in reranked_ids {
                    if let Some(ctx) = context_buffer.get(reranked_id) {
                        new_context.push(ctx.clone());
                    }

                    if let Some(score) = reranking_buffer.get(reranked_id) {
                        new_reranking.push(*score);
                    }

                    if let Some(comp) = compression_buffer.get(reranked_id) {
                        new_compression.push(*comp);
                    }
                }

                context_buffer = new_context;
                reranking_buffer = new_reranking;
                compression_buffer = new_compression;
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

    pub fn process_question_batched(
        &self,
        tokenizer: &Tokenizer,
        question: &str,
        contexts: &[String],
        threshold: f32,
        always_select_first: bool,
        rounding_mode: SentenceRoundingMode,
    ) -> Result<Vec<ProcessedResult>> {
        if contexts.is_empty() {
            return Ok(Vec::new());
        }

        // 1 Build input texts and context_start_byte per context

        // TODO: configurable / check py implementation?
        let normalize_question = true;

        let normalized_question = if normalize_question {
            Self::normalize_string(question)
        } else {
            question.to_string()
        };

        let mut input_texts = Vec::with_capacity(contexts.len());
        let mut context_start_bytes = Vec::with_capacity(contexts.len());

        for context in contexts {
            let input = Self::format_input(&normalized_question, context);
            let context_start = normalized_question.len() + 1 + config::SEPARATOR_TOKEN.len() + 1;

            input_texts.push(input);
            context_start_bytes.push(context_start);
        }

        // 2 Batch encode
        let encodings = tokenizer
            .encode_batch(input_texts, true)
            .map_err(|e| Error::msg(format!("encode_batch failed: {}", e)))?;

        // 3 Build padded batch tensors for input / attention mask
        let pad_id = tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);

        let (input_ids, attention_mask, sequence_lens) =
            Self::batch_from_encodings(&encodings, pad_id, &self.device)?;

        // 4 forward pass for the whole batch
        let output = self.forward(&input_ids, Some(attention_mask))?;
        let keep_probs_all = Self::calculate_keep_probs(&output.compression_logits)?;
        let ranking_scores = &output.ranking_scores;

        // 5 Per sample post processing
        let mut results = Vec::with_capacity(contexts.len());

        for (context_index, context) in contexts.iter().enumerate() {
            let sequence_len = *sequence_lens
                .get(context_index)
                .context(format!("Couldn't get {context_index} from sequence_lens"))?;

            let keep_probs = {
                let row: Vec<f32> = keep_probs_all.i(context_index)?.to_vec1()?;
                row.into_iter().take(sequence_len).collect::<Vec<_>>()
            };

            let reranking_score = ranking_scores.i(context_index)?.to_vec0()?;

            let encoding = encodings
                .get(context_index)
                .context(format!("Couldn't get {context_index} from encodings"))?;

            let tokens = encoding.get_ids();

            let separator_index = Self::get_separator_index(encoding)
                .ok_or_else(|| Error::msg("separator token missing"))?;

            let context_start_byte = context_start_bytes.get(context_index).context(format!(
                "Couldn't get {context_index} from context_start_bytes"
            ))?;

            let (_sentence_texts, sentences_token_coords) =
                split_sentences_and_track_from_encoding(context, encoding, *context_start_byte)?;

            let keep_mask = sentence_rounding(
                &keep_probs,
                &sentences_token_coords,
                threshold,
                always_select_first,
                &rounding_mode,
            )?;

            let (kept_token_ids, _removed_token_ids) =
                Self::group_context_tokens(tokens, separator_index, &keep_mask);

            let pruned_context = tokenizer
                .decode(&kept_token_ids, true)
                .map_err(|e| Error::msg(format!("Decoding failed: {}", e)))?;

            let compression_rate = Self::calculate_compression_rate(context, &pruned_context);

            results.push(ProcessedResult {
                question: question.to_owned(),
                context: context.to_owned(),
                pruned_context,
                reranking_score,
                compression_rate,
            });
        }

        Ok(results)
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
        let queries = match question {
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
            if titles.len() != queries.len() {
                bail!("'titles' must be a list of strings of the same length as 'queries'")
            }

            for (titles_item, contexts_item) in titles.iter().zip(contexts.iter()) {
                if titles_item.len() != contexts_item.len() {
                    bail!(
                        "Each list in 'titles' must have the same length as the corresponding list in 'context'"
                    )
                }
            }
        }

        if queries.len() != contexts.len() {
            bail!("'queries' and 'contexts' must have same lengths")
        }

        Ok((queries, contexts, titles))
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
