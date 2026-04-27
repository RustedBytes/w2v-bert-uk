use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use half::f16;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct CtcCandidate {
    pub token_ids: Vec<u32>,
    pub ctc_log_prob: f32,
}

#[derive(Clone, Debug)]
pub struct CtcAlignmentConfig {
    pub blank_id: u32,
    pub index_duration: f64,
    pub score_min_mean_over_l: usize,
    pub blank_transition_cost_zero: bool,
    pub preamble_transition_cost_zero: bool,
}

impl Default for CtcAlignmentConfig {
    fn default() -> Self {
        Self {
            blank_id: 0,
            index_duration: 0.025,
            score_min_mean_over_l: 30,
            blank_transition_cost_zero: false,
            preamble_transition_cost_zero: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CtcAlignmentSegment {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub score: f64,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct CtcAlignmentResult {
    pub timings: Vec<f64>,
    pub frame_log_probs: Vec<f32>,
    pub states: Vec<Option<u32>>,
    pub utterance_begin_indices: Vec<usize>,
    pub segments: Vec<CtcAlignmentSegment>,
}

#[derive(Clone, Copy, Debug)]
struct BeamScore {
    blank: f32,
    non_blank: f32,
}

#[derive(Debug)]
struct BeamUpdate {
    prefix: Vec<u32>,
    blank: f32,
    non_blank: f32,
}

pub(crate) trait CtcLogit: Copy + Send + Sync {
    fn to_f32(self) -> f32;
}

impl CtcLogit for f16 {
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }
}

impl CtcLogit for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

pub(crate) fn threaded_ctc_beam_search_decode_n_best<T: CtcLogit>(
    shape: &[i64],
    logits: &[T],
    blank_id: u32,
    beam_width: usize,
    n_best: usize,
) -> Result<Vec<CtcCandidate>> {
    let (frames, vocab_size) = match shape {
        [batch, frames, vocab_size] => {
            if *batch != 1 {
                bail!("only batch size 1 is supported, got output shape {shape:?}");
            }
            (*frames, *vocab_size)
        }
        [frames, vocab_size] => (*frames, *vocab_size),
        _ => bail!(
            "expected CTC logits with shape [1, frames, vocab] or [frames, vocab], got {shape:?}"
        ),
    };

    let frames = usize::try_from(frames).context("negative frame dimension in output")?;
    let vocab_size = usize::try_from(vocab_size).context("negative vocab dimension in output")?;
    if vocab_size == 0 {
        bail!("model output has empty vocab dimension");
    }
    if logits.len() != frames * vocab_size {
        bail!(
            "output shape {shape:?} implies {} logits, got {}",
            frames * vocab_size,
            logits.len()
        );
    }

    let blank_index = usize::try_from(blank_id).context("blank ID does not fit usize")?;
    if blank_index >= vocab_size {
        bail!("blank ID {blank_id} is outside vocab size {vocab_size}");
    }

    let mut beams = vec![(
        Vec::new(),
        BeamScore {
            blank: 0.0,
            non_blank: f32::NEG_INFINITY,
        },
    )];

    for frame in logits.chunks_exact(vocab_size) {
        let log_probs = log_softmax_frame(frame);
        let updates = beams
            .par_iter()
            .flat_map_iter(|(prefix, score)| beam_updates(prefix, *score, &log_probs, blank_id))
            .collect::<Vec<_>>();

        let mut next = HashMap::<Vec<u32>, BeamScore>::with_capacity(updates.len());
        for update in updates {
            let entry = next.entry(update.prefix).or_insert(BeamScore {
                blank: f32::NEG_INFINITY,
                non_blank: f32::NEG_INFINITY,
            });
            entry.blank = log_add(entry.blank, update.blank);
            entry.non_blank = log_add(entry.non_blank, update.non_blank);
        }

        beams = next.into_iter().collect();
        beams.sort_unstable_by(|(_, a), (_, b)| beam_total(*b).total_cmp(&beam_total(*a)));
        beams.truncate(beam_width);
    }

    let mut candidates = beams
        .into_iter()
        .map(|(token_ids, score)| CtcCandidate {
            token_ids,
            ctc_log_prob: beam_total(score),
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable_by(|a, b| b.ctc_log_prob.total_cmp(&a.ctc_log_prob));
    candidates.truncate(n_best.max(1));

    if candidates.is_empty() {
        bail!("beam search produced no hypotheses");
    }

    Ok(candidates)
}

pub(crate) fn ctc_log_probs<T: CtcLogit>(
    shape: &[i64],
    logits: &[T],
) -> Result<(usize, usize, Vec<f32>)> {
    let (frames, vocab_size) = ctc_shape(shape)?;
    if logits.len() != frames * vocab_size {
        bail!(
            "output shape {shape:?} implies {} logits, got {}",
            frames * vocab_size,
            logits.len()
        );
    }

    let log_probs = logits
        .chunks_exact(vocab_size)
        .flat_map(log_softmax_frame)
        .collect::<Vec<_>>();
    Ok((frames, vocab_size, log_probs))
}

pub fn align_token_sequences(
    log_probs: &[f32],
    frames: usize,
    vocab_size: usize,
    utterances: &[Vec<u32>],
    texts: &[String],
    config: &CtcAlignmentConfig,
) -> Result<CtcAlignmentResult> {
    if frames == 0 {
        bail!("cannot align empty CTC output");
    }
    if vocab_size == 0 {
        bail!("cannot align CTC output with empty vocab");
    }
    if log_probs.len() != frames * vocab_size {
        bail!(
            "CTC log-prob shape implies {} values, got {}",
            frames * vocab_size,
            log_probs.len()
        );
    }
    if utterances.len() != texts.len() {
        bail!(
            "utterance/text count mismatch: {} token lists for {} text entries",
            utterances.len(),
            texts.len()
        );
    }
    let blank = usize::try_from(config.blank_id).context("blank ID does not fit usize")?;
    if blank >= vocab_size {
        bail!(
            "blank ID {} is outside vocab size {vocab_size}",
            config.blank_id
        );
    }

    let (ground_truth, utterance_begin_indices) = prepare_token_list(utterances, config.blank_id);
    if ground_truth.len() > frames {
        bail!(
            "audio is shorter than text: {} CTC frames for {} alignment states",
            frames,
            ground_truth.len()
        );
    }

    for &token in ground_truth.iter().filter(|&&token| token >= 0) {
        let token = usize::try_from(token).context("token ID does not fit usize")?;
        if token >= vocab_size {
            bail!("token ID {token} is outside vocab size {vocab_size}");
        }
    }

    let (table, mut t, mut c) = fill_alignment_table(
        log_probs,
        frames,
        vocab_size,
        &ground_truth,
        blank,
        config.blank_transition_cost_zero,
        config.preamble_transition_cost_zero,
    );

    let mut timings = vec![0.0; ground_truth.len()];
    let mut frame_log_probs = vec![0.0; frames];
    let mut states = vec![None; frames];

    while t != 0 || c != 0 {
        if c == 0 {
            frame_log_probs[t] = 0.0;
            t = t.saturating_sub(1);
            continue;
        }
        if t == 0 {
            bail!("CTC alignment backtracking reached the first frame before consuming all text");
        }

        let token = ground_truth[c] as usize;
        let switch_prob =
            table[(t - 1) * ground_truth.len() + (c - 1)] + log_probs[t * vocab_size + token];
        let stay_prob = stay_transition_log_prob(
            &table,
            log_probs,
            frames,
            vocab_size,
            ground_truth.len(),
            &ground_truth,
            t,
            c,
            blank,
            config.blank_transition_cost_zero,
            config.preamble_transition_cost_zero,
        );

        if switch_prob >= stay_prob {
            timings[c] = t as f64 * config.index_duration;
            frame_log_probs[t] = log_probs[t * vocab_size + token];
            states[t] = Some(token as u32);
            c -= 1;
            t -= 1;
        } else {
            let stay_prob = stay_frame_log_prob(log_probs, vocab_size, &ground_truth, t, c, blank);
            frame_log_probs[t] = stay_prob;
            states[t] = None;
            t -= 1;
        }
    }

    let segments = determine_utterance_segments(
        config,
        &utterance_begin_indices,
        &frame_log_probs,
        &timings,
        texts,
    );

    Ok(CtcAlignmentResult {
        timings,
        frame_log_probs,
        states,
        utterance_begin_indices,
        segments,
    })
}

fn ctc_shape(shape: &[i64]) -> Result<(usize, usize)> {
    let (frames, vocab_size) = match shape {
        [batch, frames, vocab_size] => {
            if *batch != 1 {
                bail!("only batch size 1 is supported, got output shape {shape:?}");
            }
            (*frames, *vocab_size)
        }
        [frames, vocab_size] => (*frames, *vocab_size),
        _ => bail!(
            "expected CTC logits with shape [1, frames, vocab] or [frames, vocab], got {shape:?}"
        ),
    };

    let frames = usize::try_from(frames).context("negative frame dimension in output")?;
    let vocab_size = usize::try_from(vocab_size).context("negative vocab dimension in output")?;
    if vocab_size == 0 {
        bail!("model output has empty vocab dimension");
    }

    Ok((frames, vocab_size))
}

fn prepare_token_list(utterances: &[Vec<u32>], blank_id: u32) -> (Vec<i64>, Vec<usize>) {
    let mut ground_truth = vec![-1];
    let mut utterance_begin_indices = Vec::with_capacity(utterances.len() + 1);

    for utterance in utterances {
        if ground_truth.last().copied() != Some(blank_id as i64) {
            ground_truth.push(blank_id as i64);
        }
        utterance_begin_indices.push(ground_truth.len() - 1);
        ground_truth.extend(utterance.iter().map(|&token| token as i64));
    }

    if ground_truth.last().copied() != Some(blank_id as i64) {
        ground_truth.push(blank_id as i64);
    }
    utterance_begin_indices.push(ground_truth.len() - 1);

    (ground_truth, utterance_begin_indices)
}

fn fill_alignment_table(
    log_probs: &[f32],
    frames: usize,
    vocab_size: usize,
    ground_truth: &[i64],
    blank: usize,
    blank_transition_cost_zero: bool,
    preamble_transition_cost_zero: bool,
) -> (Vec<f32>, usize, usize) {
    let states = ground_truth.len();
    let mut table = vec![f32::NEG_INFINITY; frames * states];
    table[0] = 0.0;

    for t in 1..frames {
        table[t * states] = if preamble_transition_cost_zero {
            0.0
        } else {
            table[(t - 1) * states] + log_probs[t * vocab_size + blank]
        };
    }

    for t in 1..frames {
        for c in 1..states {
            let token = ground_truth[c] as usize;
            let switch_prob = table[(t - 1) * states + (c - 1)] + log_probs[t * vocab_size + token];
            let stay_prob = stay_transition_log_prob(
                &table,
                log_probs,
                frames,
                vocab_size,
                states,
                ground_truth,
                t,
                c,
                blank,
                blank_transition_cost_zero,
                preamble_transition_cost_zero,
            );
            table[t * states + c] = switch_prob.max(stay_prob);
        }
    }

    let last_state = states - 1;
    let best_t = (0..frames)
        .max_by(|&a, &b| table[a * states + last_state].total_cmp(&table[b * states + last_state]))
        .unwrap_or(frames - 1);

    (table, best_t, last_state)
}

#[allow(clippy::too_many_arguments)]
fn stay_transition_log_prob(
    table: &[f32],
    log_probs: &[f32],
    _frames: usize,
    vocab_size: usize,
    states: usize,
    ground_truth: &[i64],
    t: usize,
    c: usize,
    blank: usize,
    blank_transition_cost_zero: bool,
    preamble_transition_cost_zero: bool,
) -> f32 {
    if t == 0 {
        return f32::NEG_INFINITY;
    }
    if preamble_transition_cost_zero && c == 0 {
        return 0.0;
    }
    if blank_transition_cost_zero && ground_truth[c] == blank as i64 {
        return table[(t - 1) * states + c];
    }

    table[(t - 1) * states + c]
        + stay_frame_log_prob(log_probs, vocab_size, ground_truth, t, c, blank)
}

fn stay_frame_log_prob(
    log_probs: &[f32],
    vocab_size: usize,
    ground_truth: &[i64],
    t: usize,
    c: usize,
    blank: usize,
) -> f32 {
    let token = ground_truth[c];
    let blank_prob = log_probs[t * vocab_size + blank];
    if token < 0 {
        return blank_prob;
    }
    blank_prob.max(log_probs[t * vocab_size + token as usize])
}

fn determine_utterance_segments(
    config: &CtcAlignmentConfig,
    utterance_begin_indices: &[usize],
    frame_log_probs: &[f32],
    timings: &[f64],
    texts: &[String],
) -> Vec<CtcAlignmentSegment> {
    let mut segments = Vec::with_capacity(texts.len());
    for i in 0..texts.len() {
        let start = compute_time(timings, utterance_begin_indices[i], AlignType::Begin);
        let end = compute_time(timings, utterance_begin_indices[i + 1], AlignType::End);
        let start_t = (start / config.index_duration).round().max(0.0) as usize;
        let end_t = (end / config.index_duration)
            .round()
            .max(0.0)
            .min(frame_log_probs.len() as f64) as usize;
        let score = min_mean_score(
            frame_log_probs,
            start_t,
            end_t,
            config.score_min_mean_over_l,
        );
        segments.push(CtcAlignmentSegment {
            start_seconds: start,
            end_seconds: end,
            score,
            text: texts[i].clone(),
        });
    }
    segments
}

#[derive(Clone, Copy)]
enum AlignType {
    Begin,
    End,
}

fn compute_time(timings: &[f64], index: usize, align_type: AlignType) -> f64 {
    let previous = timings[index.saturating_sub(1)];
    let current = timings[index];
    let middle = (current + previous) / 2.0;
    match align_type {
        AlignType::Begin => timings.get(index + 1).copied().unwrap_or(current) - 0.5,
        AlignType::End => previous + 0.5,
    }
    .max(match align_type {
        AlignType::Begin => middle,
        AlignType::End => f64::NEG_INFINITY,
    })
    .min(match align_type {
        AlignType::Begin => f64::INFINITY,
        AlignType::End => middle,
    })
}

fn min_mean_score(frame_log_probs: &[f32], start: usize, end: usize, window: usize) -> f64 {
    if end <= start {
        return -10_000_000_000.0;
    }
    let window = window.max(1);
    if end - start <= window {
        return mean(&frame_log_probs[start..end]);
    }

    let mut min_avg = 0.0;
    for t in start..(end - window) {
        let avg = mean(&frame_log_probs[t..t + window]);
        if avg < min_avg {
            min_avg = avg;
        }
    }
    min_avg
}

fn mean(values: &[f32]) -> f64 {
    values.iter().map(|&value| value as f64).sum::<f64>() / values.len() as f64
}

fn beam_updates(
    prefix: &[u32],
    score: BeamScore,
    log_probs: &[f32],
    blank_id: u32,
) -> Vec<BeamUpdate> {
    let total = beam_total(score);
    let mut updates = Vec::with_capacity(log_probs.len() + 1);
    updates.push(BeamUpdate {
        prefix: prefix.to_vec(),
        blank: total + log_probs[blank_id as usize],
        non_blank: f32::NEG_INFINITY,
    });

    let last_token = prefix.last().copied();
    for (token, &log_prob) in log_probs.iter().enumerate() {
        let token = token as u32;
        if token == blank_id {
            continue;
        }

        if Some(token) == last_token {
            updates.push(BeamUpdate {
                prefix: prefix.to_vec(),
                blank: f32::NEG_INFINITY,
                non_blank: score.non_blank + log_prob,
            });

            let mut extended = prefix.to_vec();
            extended.push(token);
            updates.push(BeamUpdate {
                prefix: extended,
                blank: f32::NEG_INFINITY,
                non_blank: score.blank + log_prob,
            });
        } else {
            let mut extended = prefix.to_vec();
            extended.push(token);
            updates.push(BeamUpdate {
                prefix: extended,
                blank: f32::NEG_INFINITY,
                non_blank: total + log_prob,
            });
        }
    }

    updates
}

fn log_softmax_frame<T: CtcLogit>(frame: &[T]) -> Vec<f32> {
    let max = frame
        .iter()
        .map(|&value| value.to_f32())
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp = frame
        .iter()
        .map(|&value| (value.to_f32() - max).exp())
        .sum::<f32>();
    let log_sum_exp = max + sum_exp.ln();

    frame
        .iter()
        .map(|&value| value.to_f32() - log_sum_exp)
        .collect()
}

fn beam_total(score: BeamScore) -> f32 {
    log_add(score.blank, score.non_blank)
}

fn log_add(a: f32, b: f32) -> f32 {
    if a.is_infinite() && a.is_sign_negative() {
        return b;
    }
    if b.is_infinite() && b.is_sign_negative() {
        return a;
    }

    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepares_token_list_like_reference_implementation() {
        let (ground_truth, begins) = prepare_token_list(&[vec![2, 1, 7], vec![3, 5, 4, 6]], 0);

        assert_eq!(ground_truth, vec![-1, 0, 2, 1, 7, 0, 3, 5, 4, 6, 0]);
        assert_eq!(begins, vec![1, 5, 10]);
    }

    #[test]
    fn determines_segments_like_reference_test_vector() {
        let config = CtcAlignmentConfig {
            index_duration: 1.0,
            score_min_mean_over_l: 2,
            ..CtcAlignmentConfig::default()
        };
        let timings = (0..10).map(|value| value as f64 + 0.5).collect::<Vec<_>>();
        let frame_log_probs = vec![-0.5; 10];
        let texts = vec!["catzz#".to_string(), "dogs!!".to_string()];

        let segments =
            determine_utterance_segments(&config, &[1, 4, 9], &frame_log_probs, &timings, &texts);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start_seconds, 2.0);
        assert_eq!(segments[0].end_seconds, 4.0);
        assert_eq!(segments[0].score, -0.5);
        assert_eq!(segments[1].start_seconds, 5.0);
        assert_eq!(segments[1].end_seconds, 9.0);
        assert_eq!(segments[1].score, -0.5);
    }

    #[test]
    fn aligns_token_sequences_to_high_probability_frames() {
        let frames = 7;
        let vocab_size = 3;
        let mut log_probs = vec![-10.0; frames * vocab_size];
        for (frame, token) in [(0, 0), (1, 0), (2, 1), (3, 0), (4, 2), (5, 0), (6, 0)] {
            log_probs[frame * vocab_size + token] = 0.0;
        }

        let result = align_token_sequences(
            &log_probs,
            frames,
            vocab_size,
            &[vec![1], vec![2]],
            &["a".to_string(), "b".to_string()],
            &CtcAlignmentConfig {
                index_duration: 1.0,
                score_min_mean_over_l: 1,
                ..CtcAlignmentConfig::default()
            },
        )
        .expect("alignment should succeed");

        assert_eq!(result.utterance_begin_indices, vec![1, 3, 5]);
        assert_eq!(result.states[2], Some(1));
        assert_eq!(result.states[4], Some(2));
        assert_eq!(result.segments.len(), 2);
        assert!(result.segments[0].start_seconds <= result.segments[0].end_seconds);
        assert!(result.segments[1].start_seconds <= result.segments[1].end_seconds);
    }
}
