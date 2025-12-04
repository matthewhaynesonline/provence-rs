use candle_core::{Context, DType, Result, Tensor, bail};
use tokenizers::Offsets;

pub type SplitAndTrimResult = Result<(Vec<String>, Vec<Offsets>)>;
pub type TrimRangeResult = Result<Option<(String, Offsets)>>;

pub mod config {
    pub const DROP_VALUE: f32 = 0.0;
    pub const KEEP_VALUE: f32 = 1.0;
    pub const SENTENCE_ENDING: &[char] = &['.', '!', '?'];
}

pub fn split_and_round_sentences(
    context: &str,
    encoding_offsets: &[Offsets],
    context_start_offset: usize,
    token_predictions: &[f32],
    threshold: f32,
    always_select_first: bool,
) -> Result<Vec<bool>> {
    let (_sentence_texts, sentences_to_tokens_ranges) =
        split_sentences_and_track_from_encoding(context, encoding_offsets, context_start_offset)?;

    sentence_rounding(
        token_predictions,
        &sentences_to_tokens_ranges,
        threshold,
        always_select_first,
    )
}

/// Apply sentence-level rounding to token predictions.
///
/// A sentence is kept if the mean decision of its tokens exceeds `threshold`
/// If always_select_first is true, the first sentence is kept (only when
/// another sentence would be kept).
pub fn sentence_rounding(
    token_predictions: &[f32],
    sentences_to_tokens_ranges: &[Offsets],
    threshold: f32,
    always_select_first: bool,
) -> Result<Vec<bool>> {
    if sentences_to_tokens_ranges.is_empty() {
        bail!("sentences_to_tokens_ranges is empty");
    }

    let n_tokens = token_predictions.len();
    let mut sentence_means: Vec<f32> = Vec::with_capacity(sentences_to_tokens_ranges.len());

    for coord in sentences_to_tokens_ranges {
        if coord.0 >= coord.1 {
            bail!("invalid Offsets: start ({}) >= end ({})", coord.0, coord.1);
        }

        if coord.1 > n_tokens {
            bail!(
                "invalid Offsets: end ({}) > token_predictions.len() ({})",
                coord.1,
                n_tokens
            );
        }

        let token_coords = token_predictions.get(coord.0..coord.1).context(format!(
            "Couldn't get token_coords {}..{}",
            coord.0, coord.1
        ))?;

        let (tokens_sum, tokens_count) =
            token_coords
                .iter()
                .fold((config::DROP_VALUE, 0), |(sum, count), token_keep_prob| {
                    if token_keep_prob.is_nan() {
                        (sum, count)
                    } else {
                        let token_decision = if token_keep_prob > &threshold {
                            config::KEEP_VALUE
                        } else {
                            config::DROP_VALUE
                        };

                        (sum + token_decision, count + 1)
                    }
                });

        let mean = if tokens_count == 0 {
            config::DROP_VALUE
        } else {
            tokens_sum / (tokens_count as f32)
        };

        sentence_means.push(mean);
    }

    if always_select_first
        && sentence_means.iter().skip(1).any(|&m| m > threshold)
        && let Some(first) = sentence_means.get_mut(0)
    {
        *first = config::KEEP_VALUE;
    }

    let mut keep_mask = vec![false; n_tokens];

    for (coord, &mean) in sentences_to_tokens_ranges.iter().zip(sentence_means.iter()) {
        if mean <= threshold {
            continue;
        }

        if let Some(mask_slice) = keep_mask.get_mut(coord.0..coord.1) {
            mask_slice.fill(true);
        }
    }

    Ok(keep_mask)
}

pub fn split_and_round_sentences_tensor(
    context: &str,
    encoding_offsets: &[Offsets],
    context_start_offset: usize,
    token_predictions: &Tensor,
    threshold: f32,
    always_select_first: bool,
) -> Result<Vec<bool>> {
    let (_sentence_texts, sentences_to_tokens_ranges) =
        split_sentences_and_track_from_encoding(context, encoding_offsets, context_start_offset)?;

    let mask_tensor = sentence_rounding_tensor(
        token_predictions,
        &sentences_to_tokens_ranges,
        threshold,
        always_select_first,
    )?;

    let mask_bool = mask_tensor
        .to_vec1()?
        .into_iter()
        .map(|v: u8| v != 0)
        .collect();

    Ok(mask_bool)
}

pub fn sentence_rounding_tensor(
    token_predictions: &Tensor,
    sentences_token_coords: &[Offsets],
    threshold: f32,
    always_select_first: bool,
) -> Result<Tensor> {
    // 1. Input Validation
    if sentences_token_coords.is_empty() {
        bail!("sentences_token_coords is empty");
    }

    let device = token_predictions.device();

    let &n_tokens = token_predictions
        .dims()
        .first()
        .context("Couldn't get n_tokens")?;

    let num_sentences = sentences_token_coords.len();

    for coord in sentences_token_coords {
        if coord.0 >= coord.1 || coord.1 > n_tokens {
            bail!(
                "invalid Coordinate: start={}, end={}, n_tokens={}",
                coord.0,
                coord.1,
                n_tokens
            );
        }
    }

    // 2. Compute Sentence Scores (The "Grouping" Phase)
    let mut sentence_mean_tensors = Vec::with_capacity(num_sentences);

    // 3. Determine "Keep" Status (The Decision Phase)
    for coord in sentences_token_coords {
        let sentence_tokens = token_predictions.narrow(0, coord.0, coord.1 - coord.0)?;

        let mean_tensor = sentence_tokens
            .ge(threshold)?
            .to_dtype(DType::F32)?
            .mean_all()?;

        sentence_mean_tensors.push(mean_tensor);
    }

    // Stack means and compare to threshold
    let means_stacked = Tensor::stack(&sentence_mean_tensors, 0)?; // [num_sentences]
    let mut keeps_tensor = means_stacked.gt(threshold)?; // [num_sentences] of u8 0/1

    // The "Always Select First" Logic
    if always_select_first && num_sentences > 1 {
        // Check if ANY sentence after first exceeds threshold
        let rest_keeps = keeps_tensor.narrow(0, 1, num_sentences - 1)?;
        let any_rest = rest_keeps.max_all()?;

        let first_keep = keeps_tensor.narrow(0, 0, 1)?; // [1]

        // OR operation: max(first_keep, any_rest)
        let new_first = first_keep.broadcast_maximum(&any_rest.reshape((1,))?)?;

        // Concatenate back
        keeps_tensor = Tensor::cat(&[new_first, rest_keeps], 0)?;
    }

    // Step 4: Build coordinates
    let starts = sentences_token_coords.iter().map(|c| c.0 as u32).collect();
    let ends = sentences_token_coords.iter().map(|c| c.1 as u32).collect();

    // Transfer coordinates to GPU
    let starts_tensor = Tensor::from_vec(starts, num_sentences, device)?;
    let ends_tensor = Tensor::from_vec(ends, num_sentences, device)?;

    // Step 5: Build token indices
    let token_indices = Tensor::arange(0u32, n_tokens as u32, device)?; // [n_tokens]

    // Step 6: Reshape for broadcasting
    let tokens_2d = token_indices.unsqueeze(1)?; // [n_tokens, 1]
    let starts_2d = starts_tensor.unsqueeze(0)?; // [1, num_sentences]
    let ends_2d = ends_tensor.unsqueeze(0)?; // [1, num_sentences]
    let keeps_2d = keeps_tensor.unsqueeze(0)?; // [1, num_sentences]

    // Step 7: Broadcast comparisons - creates [n_tokens, num_sentences] matrices
    // Each element [i,j] = 1 if token i is in sentence j's range and sentence j is kept
    let ge_start = tokens_2d.broadcast_ge(&starts_2d)?; // [n_tokens, num_sentences]
    let lt_end = tokens_2d.broadcast_lt(&ends_2d)?; // [n_tokens, num_sentences]

    // AND: in_range[i,j] = (token i >= start_j) AND (token i < end_j)
    let in_range = ge_start
        .to_dtype(DType::U8)?
        .mul(&lt_end.to_dtype(DType::U8)?)?; // [n_tokens, num_sentences]

    // AND with keep decisions: mask[i,j] = in_range[i,j] AND keep[j]
    let keeps_broadcasted = keeps_2d.broadcast_as((n_tokens, num_sentences))?;
    let masks = in_range.mul(&keeps_broadcasted.to_dtype(DType::U8)?)?; // [n_tokens, num_sentences]

    // Step 8: OR across all sentences - if ANY sentence keeps token i, keep it
    let final_mask = masks.max(1)?; // [n_tokens] - max across sentence dimension

    Ok(final_mask)
}

/// Split context into sentences but return token index coords
/// relative to the full encoding token indices by using encoding offsets.
pub fn split_sentences_and_track_from_encoding(
    context: &str,
    encoding_offsets: &[Offsets],
    context_start_offset: usize,
) -> Result<(Vec<String>, Vec<Offsets>)> {
    let (sentences, sentence_ranges_rel_to_context) = split_and_trim_sentences(context)?;
    let sentences_to_tokens_ranges = map_sentence_ranges_to_token_ranges(
        &sentence_ranges_rel_to_context,
        context_start_offset,
        encoding_offsets,
    )?;

    Ok((sentences, sentences_to_tokens_ranges))
}

/// Split `context` into trimmed sentences and return
/// trimmed sentences and byte ranges relative to context
fn split_and_trim_sentences(context: &str) -> SplitAndTrimResult {
    let mut sentences = Vec::new();
    let mut ranges = Vec::new();

    #[cfg(feature = "punkt")]
    {
        // TODO: cache?
        // TODO: custom train?
        // TODO: use tokenizers?
        let punkt_data = punkt::TrainingData::english();

        for (start, end) in
            punkt::SentenceByteOffsetTokenizer::<punkt::params::Standard>::new(context, &punkt_data)
        {
            if let Some((sentence, (trim_start, trim_end))) = trim_range(context, start, end)?
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

/// Trim a byte range `(start, end)` within `context` (both byte indices relative to `context`) and
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
        context[trim_start..trim_end].to_string(),
        (trim_start, trim_end),
    )))
}

/// Maps byte ranges of sentences (relative to `context`) to their corresponding
/// token index ranges.
///
/// Returns a list of (start_token_index, end_token_index), where the end index is exclusive.
///
/// ### Example
/// Text:          Hello dog.                     Hello cat.
/// Sentences:   [[0,                   9],      [10,                      20]]
/// Tokens:     0:[0,5], 1:[6,9], 2:[9,10],    3:[10,15], 4:[16,19], 5:[19,20]
///                "Hello"  "dog"    "."          "Hello"    "cat"      "."
/// Output:      [[0,                   3],      [3,                        6]]
///
/// Sentence 1 (0-9) overlaps tokens 0, 1, 2. Range is 0..3
/// Sentence 2 (10-20) overlaps tokens 3, 4, 5. Range is 3..6
fn map_sentence_ranges_to_token_ranges(
    sentence_ranges_rel_to_context: &[Offsets],
    context_start_offset: usize,
    token_offsets: &[Offsets],
) -> Result<Vec<Offsets>> {
    let mut sentences_to_tokens_ranges = Vec::with_capacity(sentence_ranges_rel_to_context.len());
    let mut token_index_sentence_start = 0;
    let n_tokens = token_offsets.len();

    for &(sentence_start, sentence_end) in sentence_ranges_rel_to_context {
        let sentence_start_abs = context_start_offset + sentence_start;
        let sentence_end_abs = context_start_offset + sentence_end;

        // Fast forward past tokens that are entirely before the current sentence
        while token_index_sentence_start < n_tokens {
            let &(token_start, token_end) =
                token_offsets
                    .get(token_index_sentence_start)
                    .context(format!(
                        "Couldn't get token_offsets {token_index_sentence_start}"
                    ))?;

            // skip tokens with no token_offsets (common for special tokens)
            if token_start == 0 && token_end == 0 {
                token_index_sentence_start += 1;
                continue;
            }

            if token_end <= sentence_start_abs {
                token_index_sentence_start += 1;
                continue;
            }

            break;
        }

        let mut first_token_index = None;
        let mut last_token_index = None;
        let mut token_index_in_sentence = token_index_sentence_start;

        // Scans the tokens that overlap with the sentence and stop as soon as it goes past the sentence
        while token_index_in_sentence < n_tokens {
            let &(token_start, token_end) = token_offsets.get(token_index_in_sentence).context(
                format!("Couldn't get token_offsets {token_index_in_sentence}"),
            )?;

            // skip tokens with no token_offsets (common for special tokens)
            if token_start == 0 && token_end == 0 {
                token_index_in_sentence += 1;
                continue;
            }

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
            sentences_to_tokens_ranges.push((
                first,
                last + 1, // end exclusive
            ));
        }
    }

    Ok(sentences_to_tokens_ranges)
}

// A and B overlap if
// B doesn't end before A starts, AND
// B doesn't start after A ends.
fn overlaps(a: Offsets, b: Offsets) -> bool {
    let (a_start, a_end) = a;
    let (b_start, b_end) = b;

    b_end > a_start && b_start < a_end
}
