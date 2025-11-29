use std::vec;

use candle_core::{Context, Error, IndexOp, Result, Tensor, bail};
use either::Either;
use tokenizers::{Encoding, Tokenizer};

use super::{
    ProvenceModel, ProvenceOutput,
    sentence_rounding::{
        SentenceRoundingMode, sentence_rounding, split_sentences_and_track_from_encoding,
    },
};

pub type MultipleQuestions = Vec<String>;
pub type MultipleContext = Vec<Vec<String>>;
pub type MultipleTitle = Vec<Vec<String>>;
pub type PreparedProcessParams = (MultipleQuestions, MultipleContext, Option<MultipleTitle>);
pub type EncodedInput = (Encoding, Tensor, Tensor);

#[derive(Debug, Clone)]
pub struct ProcessedResults {
    pub pruned_context: MultipleContext,
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
    // pub token_details: Option<Vec<TokenDetail>>,
}

// #[derive(Debug, Clone, PartialEq)]
// pub enum TokenStatus {
//     QuestionOrSpecial,
//     Kept,
//     Dropped,
// }

// impl fmt::Display for TokenStatus {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         let s = match self {
//             TokenStatus::QuestionOrSpecial => "KEEP (Q/SPECIAL)",
//             TokenStatus::Kept => "KEEP",
//             TokenStatus::Dropped => "DROP",
//         };
//         write!(f, "{}", s)
//     }
// }

// #[derive(Debug, Clone)]
// pub struct TokenDetail {
//     pub index: usize,
//     pub token: String,
//     pub probability: f32,
//     pub status: TokenStatus,
// }

pub mod config {
    // TODO: don't hardcode
    pub const SEPARATOR_TOKEN: &str = "[SEP]";
    pub const TITLE_PARAM_SPECIAL_VALUE: &str = "first_sentence";
}

impl ProvenceModel {
    pub fn format_input(question: &str, context: &str) -> String {
        format!("{} {} {}", question, config::SEPARATOR_TOKEN, context)
    }

    /// Process query / context with sentence-level rounding
    #[allow(clippy::too_many_arguments)]
    pub fn process(
        &self,
        tokenizer: &Tokenizer,
        // TODO: make slices?
        question: Either<MultipleQuestions, &str>,
        context: Either<MultipleContext, &str>,
        title: Option<Either<MultipleTitle, &str>>,
        threshold: Option<f32>,
        always_select_first: Option<bool>,
        batch_size: Option<usize>,
        reorder: Option<bool>,
        top_k: Option<usize>,
        enable_warnings: Option<bool>,
        rounding_mode: Option<SentenceRoundingMode>,
    ) -> Result<ProcessedResults> {
        let (queries, contexts, titles) = Self::prepare_process_params(question, context, title)?;

        let threshold = threshold.unwrap_or(0.1);
        let always_select_first = always_select_first.unwrap_or(true);
        let batch_size = batch_size.unwrap_or(32);
        let reorder = reorder.unwrap_or(false);
        let top_k = top_k.unwrap_or(5);
        let enable_warnings = enable_warnings.unwrap_or(true);
        let rounding_mode = rounding_mode.unwrap_or(SentenceRoundingMode::DecisionAverage);

        let mut pruned_context = Vec::with_capacity(queries.len());
        let mut reranking_score = Vec::with_capacity(queries.len());
        let mut compression_rate = Vec::with_capacity(queries.len());

        for (question_i, question) in queries.iter().enumerate() {
            let question_contexts = contexts.get(question_i).context(format!(
                "Couldn't get contexts for index {question_i} value {question}",
            ))?;

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

                // TODO: tokenizer questions / sep / context separately, cache, then combine for forward?
                let result = self.process_question_context(
                    tokenizer,
                    question,
                    context.as_str(),
                    threshold,
                    always_select_first,
                    rounding_mode.clone(),
                )?;

                context_buffer.push(result.pruned_context);
                reranking_buffer.push(result.reranking_score);
                compression_buffer.push(result.compression_rate);
            }

            if reorder {
                let mut idxs: Vec<usize> = (0..reranking_buffer.len()).collect();

                idxs.sort_by(|&a, &b| {
                    reranking_buffer[b]
                        .partial_cmp(&reranking_buffer[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let idxs: Vec<usize> = idxs.into_iter().take(top_k).collect();

                context_buffer = idxs.iter().map(|&j| context_buffer[j].clone()).collect();
                reranking_buffer = idxs.iter().map(|&j| reranking_buffer[j]).collect();
                compression_buffer = idxs.iter().map(|&j| compression_buffer[j]).collect();
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

    pub fn prepare_process_params(
        question: Either<MultipleQuestions, &str>,
        context: Either<MultipleContext, &str>,
        title: Option<Either<MultipleTitle, &str>>,
    ) -> Result<PreparedProcessParams> {
        // Convert input format into questions of Vec[str] and contexts/titles of Vec[Vec[str]]
        let queries = match question {
            Either::Left(q) => q,
            Either::Right(q) => vec![q.to_owned()],
        };

        let contexts = match context {
            Either::Left(c) => c
                .into_iter()
                .map(|inner| inner.into_iter().collect())
                .collect(),
            Either::Right(c) => vec![vec![c.to_owned()]],
        };

        let titles = match title {
            Some(Either::Left(t)) => Some(
                t.iter()
                    .map(|inner| inner.iter().map(|s| s.to_string()).collect())
                    .collect(),
            ),
            Some(Either::Right(config::TITLE_PARAM_SPECIAL_VALUE)) => None,
            Some(Either::Right(s)) => Some(vec![vec![s.to_owned()]]),
            None => None,
        };

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

    pub fn process_question_context(
        &self,
        tokenizer: &Tokenizer,
        question: &str,
        context: &str,
        threshold: f32,
        always_select_first: bool,
        rounding_mode: SentenceRoundingMode,
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
            rounding_mode,
        )?;

        let (kept_token_ids, _removed_token_ids) =
            Self::group_context_tokens(tokens, separator_index, &keep_mask);

        let pruned_context = tokenizer
            .decode(&kept_token_ids, true)
            .map_err(|e| Error::msg(format!("Decoding failed: {}", e)))?;

        let compression_rate = Self::calculate_compression_rate(context, &pruned_context);

        // let token_details = if include_token_details {
        //     let token_details = Self::build_token_details(
        //         encoding.get_tokens(),
        //         separator_index,
        //         &keep_mask,
        //         &keep_probs,
        //     );

        //     Some(token_details)
        // } else {
        //     None
        // };

        Ok(ProcessedResult {
            question: question.to_owned(),
            context: context.to_owned(),
            pruned_context,
            reranking_score,
            compression_rate,
            // token_details,
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
        if !context.is_empty() {
            (1.0 - pruned_context.len() as f32 / context.len() as f32) * 100.0
        } else {
            0.0
        }
    }

    // pub fn build_token_details(
    //     token_strings: &[String],
    //     separator_index: usize,
    //     keep_mask: &[bool],
    //     keep_probs: &[f32],
    // ) -> Vec<TokenDetail> {
    //     let mut token_details = Vec::with_capacity(token_strings.len());

    //     for (i, token_string) in token_strings.iter().enumerate() {
    //         let prob = keep_probs[i];

    //         let status = if i <= separator_index {
    //             TokenStatus::QuestionOrSpecial
    //         } else if keep_mask.get(i).copied().unwrap_or(false) {
    //             TokenStatus::Kept
    //         } else {
    //             TokenStatus::Dropped
    //         };

    //         token_details.push(TokenDetail {
    //             index: i,
    //             token: token_string.clone(),
    //             probability: prob,
    //             status,
    //         });
    //     }

    //     token_details
    // }
}
