use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use half::f16;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct CtcCandidate {
    pub token_ids: Vec<u32>,
    pub ctc_log_prob: f32,
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

pub fn threaded_ctc_beam_search_decode_n_best(
    shape: &[i64],
    logits: &[f16],
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

fn log_softmax_frame(frame: &[f16]) -> Vec<f32> {
    let max = frame
        .iter()
        .map(|value| value.to_f32())
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp = frame
        .iter()
        .map(|value| (value.to_f32() - max).exp())
        .sum::<f32>();
    let log_sum_exp = max + sum_exp.ln();

    frame
        .iter()
        .map(|value| value.to_f32() - log_sum_exp)
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
