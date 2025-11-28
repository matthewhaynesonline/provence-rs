use candle_core::{Result, bail};
use tokenizers::Encoding;

pub type SplitAndTrimResult = Result<(Vec<String>, Vec<(usize, usize)>)>;
pub type TrimRangeResult = Result<Option<(usize, usize, String)>>;

pub mod config {
    pub const SENTENCE_ENDING: &[char] = &['.', '!', '?'];
}

#[derive(Debug, Clone)]
pub enum SentenceRoundingMode {
    /// Apply the `threshold` to every single token in a sentence, converting to a binary
    /// `True` (1) or `False` (0), then averages these binary values.
    /// This is the method from the original Python implementation.
    DecisionAverage,
    /// Take the probabilities for every token in a sentence, calculates their average,
    /// and then compares that average to the threshold.
    /// This is new for the Rust implementation.
    ProbabilityAverage,
}

impl std::fmt::Display for SentenceRoundingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::ProbabilityAverage => write!(f, "probability_average"),
            Self::DecisionAverage => write!(f, "decision_average"),
        }
    }
}

impl std::str::FromStr for SentenceRoundingMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "probability_average" => Ok(Self::ProbabilityAverage),
            "decision_average" => Ok(Self::DecisionAverage),
            _ => Err(format!(
                "Invalid rounding mode: '{}'. Valid options are: probability_average, decision_average",
                s
            )),
        }
    }
}

/// Represents a boundary in a sequence
#[derive(Debug, Copy, Clone)]
pub struct Coordinate {
    pub start: usize,
    pub end: usize,
}

/// Apply sentence-level rounding to token predictions.
///
/// A sentence is kept if the mean probability of its tokens exceeds `threshold`
/// If always_select_first is true, the first sentence is kept (only when
/// another sentence would be kept).
pub fn sentence_rounding(
    token_predictions: &[f32],
    sentences_token_coords: &[Coordinate],
    threshold: f32,
    always_select_first: bool,
    rounding_mode: SentenceRoundingMode,
) -> Result<Vec<bool>> {
    if sentences_token_coords.is_empty() {
        bail!("sentences_token_coords is empty");
    }

    let n_tokens = token_predictions.len();
    let mut sentence_means: Vec<f32> = Vec::with_capacity(sentences_token_coords.len());

    for coord in sentences_token_coords {
        if coord.start >= coord.end {
            bail!(
                "invalid Coordinate: start ({}) >= end ({})",
                coord.start,
                coord.end
            );
        }

        if coord.end > n_tokens {
            bail!(
                "invalid Coordinate: end ({}) > token_predictions.len() ({})",
                coord.end,
                n_tokens
            );
        }

        let token_coords = &token_predictions[coord.start..coord.end];

        let (tokens_sum, tokens_count) = match rounding_mode {
            SentenceRoundingMode::ProbabilityAverage => {
                token_coords
                    .iter()
                    .copied()
                    .fold((0.0, 0), |(sum, count), token_keep_prob| {
                        if token_keep_prob.is_nan() {
                            (sum, count)
                        } else {
                            (sum + token_keep_prob, count + 1)
                        }
                    })
            }
            SentenceRoundingMode::DecisionAverage => {
                token_coords
                    .iter()
                    .copied()
                    .fold((0.0, 0), |(sum, count), token_keep_prob| {
                        if token_keep_prob.is_nan() {
                            (sum, count)
                        } else {
                            let token_decision = if token_keep_prob > threshold {
                                1.0
                            } else {
                                0.0
                            };

                            (sum + token_decision, count + 1)
                        }
                    })
            }
        };

        let mean = if tokens_count == 0 {
            0.0
        } else {
            tokens_sum / (tokens_count as f32)
        };

        sentence_means.push(mean);
    }

    if always_select_first
        && sentence_means.len() > 1
        && sentence_means
            .iter()
            .enumerate()
            .any(|(index, &mean)| index != 0 && mean > threshold)
    {
        sentence_means[0] = 1.0;
    }

    let mut keep_mask = vec![false; n_tokens];

    for (coord, &mean) in sentences_token_coords.iter().zip(sentence_means.iter()) {
        if mean > threshold {
            for keep_token in &mut keep_mask[coord.start..coord.end] {
                *keep_token = true;
            }
        }
    }

    Ok(keep_mask)
}

/// Split context into sentences (simple splitter) but return token index coords
/// relative to the full encoding token indices by using encoding offsets.
pub fn split_sentences_and_track_from_encoding(
    context: &str,
    encoding: &Encoding,
    context_start_byte: usize,
) -> Result<(Vec<String>, Vec<Coordinate>)> {
    let offsets = encoding.get_offsets();

    let (sentences, sentence_ranges_rel_to_context) = split_and_trim_sentences(context)?;

    let sentences_token_coords =
        map_ranges_to_token_coords(offsets, &sentence_ranges_rel_to_context, context_start_byte);

    Ok((sentences, sentences_token_coords))
}

/// Split `context` into trimmed sentences and return
/// trimmed sentences and byte ranges relative to context
fn split_and_trim_sentences(context: &str) -> SplitAndTrimResult {
    let mut sentences: Vec<String> = Vec::new();
    let mut ranges: Vec<(usize, usize)> = Vec::new();

    #[cfg(feature = "punkt")]
    {
        // TODO: cache?
        // TODO: custom train?
        // TODO: use tokenizers?
        let data = punkt::TrainingData::english();

        for (start, end) in
            punkt::SentenceByteOffsetTokenizer::<punkt::params::Standard>::new(context, &data)
        {
            if let Some((trim_start, trim_end, sentence)) = trim_range(context, start, end)?
                && !sentence.is_empty()
            {
                sentences.push(sentence);
                ranges.push((trim_start, trim_end));
            }
        }
    }

    #[cfg(not(feature = "punkt"))]
    {
        let mut current_start = 0;

        for (i, ch) in context.char_indices() {
            let is_sentence_ending = config::SENTENCE_ENDING.contains(&ch);

            if is_sentence_ending {
                let next_i = i + ch.len_utf8();

                // if next char is whitespace or we are at EOF, treat as sentence boundary
                if next_i >= context.len()
                    || context[next_i..]
                        .chars()
                        .next()
                        .map(|next_ch| next_ch.is_whitespace())
                        .unwrap_or(false)
                {
                    if let Some((trim_start, trim_end, sentence)) =
                        trim_range(context, current_start, next_i)?
                        && !sentence.is_empty()
                    {
                        sentences.push(sentence);
                        ranges.push((trim_start, trim_end));
                    }

                    current_start = next_i;
                }
            }
        }

        // Handle any leftovers
        // Last part of the text that doesn't end with a punctuation mark
        if current_start < context.len()
            && let Some((trim_start, trim_end, sentence)) =
                trim_range(context, current_start, context.len())?
            && !sentence.is_empty()
        {
            sentences.push(sentence);
            ranges.push((trim_start, trim_end));
        }
    }

    Ok((sentences, ranges))
}

/// Trim a byte range `[start, end)` within `context` (both byte indices relative to `context`) and
/// return trim_start_rel, trim_end_rel, trimmed_string or None if trimmed string is empty.
fn trim_range(context: &str, start: usize, end: usize) -> TrimRangeResult {
    if start > end || end > context.len() {
        bail!(
            "trim_range: invalid range [{}, {}) for context len {}",
            start,
            end,
            context.len()
        );
    }

    if !context.is_char_boundary(start) || !context.is_char_boundary(end) {
        bail!(
            "trim_range: start or end not on char boundary: start={} end={}",
            start,
            end
        );
    }

    let context_slice = &context[start..end];

    let (leading_index, _) = match context_slice
        .char_indices()
        .find(|(_, ch)| !ch.is_whitespace())
    {
        Some(leading_index_ch) => leading_index_ch,
        None => return Ok(None),
    };

    let (trailing_index, trailing_ch) = match context_slice
        .char_indices()
        .rev()
        .find(|(_, ch)| !ch.is_whitespace())
    {
        Some(trail_index_ch) => trail_index_ch,
        None => return Ok(None),
    };

    let trailing_end = trailing_index + trailing_ch.len_utf8();

    let trim_start = start + leading_index;
    let trim_end = start + trailing_end;

    Ok(Some((
        trim_start,
        trim_end,
        context[trim_start..trim_end].to_string(),
    )))
}

/// Map trimmed byte ranges (relative to `context`) into token coordinate ranges using `offsets`.
/// offsets: slice of (token_byte_start, token_byte_end) for the full input.
fn map_ranges_to_token_coords(
    offsets: &[(usize, usize)],
    sentence_ranges_rel_to_context: &[(usize, usize)],
    context_start_byte: usize,
) -> Vec<Coordinate> {
    let mut sentences_token_coords: Vec<Coordinate> =
        Vec::with_capacity(sentence_ranges_rel_to_context.len());

    let mut token_index_sentence_start = 0;

    let n_tokens = offsets.len();

    for &(sentence_start, sentence_end) in sentence_ranges_rel_to_context {
        let sentence_start_abs = context_start_byte + sentence_start;
        let sentence_end_abs = context_start_byte + sentence_end;

        // advance token_index_sentence_start to the first token that could be in the sentence range
        // (skip tokens entirely before sentence)
        while token_index_sentence_start < n_tokens {
            let (token_start, token_end) = offsets[token_index_sentence_start];
            // skip tokens with no offsets (common for special tokens)
            if token_start == 0 && token_end == 0 {
                token_index_sentence_start += 1;
                continue;
            }

            // if token ends before or at sentence start, keep advancing
            if token_end <= sentence_start_abs {
                token_index_sentence_start += 1;
                continue;
            }

            // otherwise this token may be in range; stop advancing
            break;
        }

        // collect in_range tokens starting from token_index
        let mut first_token_index: Option<usize> = None;
        let mut last_token_index: Option<usize> = None;
        let mut token_index_in_sentence = token_index_sentence_start;

        while token_index_in_sentence < n_tokens {
            let (token_start, token_end) = offsets[token_index_in_sentence];
            // skip tokens with no offsets (common for special tokens)
            if token_start == 0 && token_end == 0 {
                token_index_in_sentence += 1;
                continue;
            }

            // if this token starts at or after sentence end, break
            if token_start >= sentence_end_abs {
                break;
            }

            if overlaps(
                (sentence_start_abs, sentence_end_abs),
                (token_start, token_end),
            ) {
                if first_token_index.is_none() {
                    first_token_index = Some(token_index_in_sentence);
                }

                last_token_index = Some(token_index_in_sentence);
            }

            token_index_in_sentence += 1;
        }

        // advance token_index for next sentence (no backtracking)
        token_index_sentence_start = token_index_in_sentence;

        if let (Some(first), Some(last)) = (first_token_index, last_token_index) {
            sentences_token_coords.push(Coordinate {
                start: first,
                end: last + 1, // end exclusive
            });
        } else {
            // no overlapping tokens found — skip; keeping behavior similar to original
        }
    }

    sentences_token_coords
}

// A and B overlap if
// B doesn't end before A starts, AND
// B doesn't start after A ends.
fn overlaps(a: (usize, usize), b: (usize, usize)) -> bool {
    b.1 > a.0 && b.0 < a.1
}
