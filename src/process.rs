use std::fmt;

use candle_core::{Context, Error, IndexOp, Result, Tensor};
use tokenizers::{Encoding, Tokenizer};

use super::{
    ProvenceModel, ProvenceOutput, config,
    sentence_rounding::{sentence_rounding, split_sentences_and_track_from_encoding},
};

pub type InputEncodingResult = (Encoding, Tensor, Tensor);

#[derive(Debug, Clone)]
pub struct ProcessedResult {
    pub pruned_context: String,
    pub reranking_score: f32,
    pub compression_rate: f32,
    pub token_details: Option<Vec<TokenDetail>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenStatus {
    QuestionOrSpecial,
    Kept,
    Dropped,
}

impl fmt::Display for TokenStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            TokenStatus::QuestionOrSpecial => "KEEP (Q/SPECIAL)",
            TokenStatus::Kept => "KEEP",
            TokenStatus::Dropped => "DROP",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone)]
pub struct TokenDetail {
    pub index: usize,
    pub token: String,
    pub probability: f32,
    pub status: TokenStatus,
}

impl ProvenceModel {
    /// Process a single query-context pair with sentence-level rounding
    pub fn process_single(
        &self,
        tokenizer: &Tokenizer,
        question: &str,
        context: &str,
        threshold: f32,
        always_select_first: bool,
        include_token_details: bool,
    ) -> Result<ProcessedResult> {
        // TODO: check python implementation
        let normalize_question = true;

        let input_text;
        let context_start_byte;

        if normalize_question {
            let normalized_question = Self::normalize_string(question);

            input_text = Self::format_input(&normalized_question, context);
            context_start_byte = normalized_question.len() + 1 + config::SEPARATOR_TOKEN.len() + 1;
        } else {
            input_text = Self::format_input(question, context);
            context_start_byte = question.len() + 1 + config::SEPARATOR_TOKEN.len() + 1;
        };

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

        let (_sentence_texts, sentences_token_coords) =
            split_sentences_and_track_from_encoding(context, &encoding, context_start_byte)?;

        let keep_probs = Self::get_keep_probabilities(&output)?;
        let keep_mask = sentence_rounding(
            &keep_probs,
            &sentences_token_coords,
            threshold,
            always_select_first,
        )?;

        let (kept_token_ids, _removed_token_ids) =
            Self::group_context_tokens(tokens, separator_index, &keep_mask);

        let pruned_context = tokenizer
            .decode(&kept_token_ids, true)
            .map_err(|e| Error::msg(format!("Decoding failed: {}", e)))?;

        let compression_rate = Self::calculate_compression_rate(context, &pruned_context);

        let token_details = if include_token_details {
            let token_details = Self::build_token_details(
                encoding.get_tokens(),
                separator_index,
                &keep_mask,
                &keep_probs,
            );

            Some(token_details)
        } else {
            None
        };

        Ok(ProcessedResult {
            pruned_context,
            reranking_score,
            compression_rate,
            token_details,
        })
    }

    pub fn encode_input(
        &self,
        tokenizer: &Tokenizer,
        input_text: &str,
    ) -> Result<InputEncodingResult> {
        let encoding = tokenizer
            .encode(input_text, true)
            .map_err(|e| Error::msg(format!("Tokenization failed: {}", e)))?;

        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;

        let attention_mask =
            Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;

        Ok((encoding, input_ids, attention_mask))
    }

    fn normalize_string(text: &str) -> String {
        let lower_no_punctuation: String = text
            .to_lowercase()
            .chars()
            .filter(|c| !c.is_ascii_punctuation())
            .collect();

        let normalized_white_space = lower_no_punctuation
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ");

        normalized_white_space
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
        if !context.is_empty() {
            (1.0 - pruned_context.len() as f32 / context.len() as f32) * 100.0
        } else {
            0.0
        }
    }

    pub fn build_token_details(
        token_strings: &[String],
        separator_index: usize,
        keep_mask: &[bool],
        keep_probs: &[f32],
    ) -> Vec<TokenDetail> {
        let mut token_details = Vec::with_capacity(token_strings.len());

        for (i, token_string) in token_strings.iter().enumerate() {
            let prob = keep_probs[i];

            let status = if i <= separator_index {
                TokenStatus::QuestionOrSpecial
            } else if keep_mask.get(i).copied().unwrap_or(false) {
                TokenStatus::Kept
            } else {
                TokenStatus::Dropped
            };

            token_details.push(TokenDetail {
                index: i,
                token: token_string.clone(),
                probability: prob,
                status,
            });
        }

        token_details
    }
}
