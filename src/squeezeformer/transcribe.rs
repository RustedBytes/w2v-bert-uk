use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use asr_features::W2vBertFrontendConfig;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use splintr::SentencePieceTokenizer;

use crate::audio::{
    AudioDecodeConfig, AudioFeatures, audio_bytes_to_w2v_bert_features_with_config,
    audio_file_to_w2v_bert_features_with_config,
};
use crate::ctc::{CtcCandidate, threaded_ctc_beam_search_decode_n_best};
use crate::tokenizer::load_sentencepiece_tokenizer;
use crate::{
    CandidateProcessingConfig, CtcDecoderConfig, TextDecoderConfig, W2vBertEncoderConfig,
    normalize_spaces,
};

use super::SqueezeformerCtc;

#[derive(Clone, Debug, Default)]
pub struct SqueezeformerTranscriptionConfig {
    pub audio: AudioDecodeConfig,
    pub encoder: W2vBertEncoderConfig,
    pub ctc: CtcDecoderConfig,
    pub text: TextDecoderConfig,
}

pub struct SqueezeformerTranscriptionResult {
    pub transcript: String,
    pub candidates: Vec<SqueezeformerScoredCandidate>,
    pub timings: SqueezeformerTimingReport,
}

#[derive(Clone, Debug)]
pub struct SqueezeformerScoredCandidate {
    pub text: String,
    pub ctc_log_prob: f32,
    pub word_count: usize,
    pub total_score: f32,
}

pub struct SqueezeformerTimingReport {
    pub audio_duration_seconds: f64,
    pub feature_rows: usize,
    pub feature_cols: usize,
    pub feature_count: usize,
    pub audio_decode_elapsed: Duration,
    pub feature_elapsed: Duration,
    pub model_input_elapsed: Duration,
    pub model_inference_elapsed: Duration,
    pub ctc_elapsed: Duration,
    pub tokenizer_load_elapsed: Duration,
    pub text_decode_elapsed: Duration,
    pub best_candidate: Option<SqueezeformerScoredCandidate>,
}

impl SqueezeformerTimingReport {
    pub fn measured_elapsed(&self) -> Duration {
        self.audio_decode_elapsed
            + self.feature_elapsed
            + self.model_input_elapsed
            + self.model_inference_elapsed
            + self.ctc_elapsed
            + self.tokenizer_load_elapsed
            + self.text_decode_elapsed
    }

    pub fn real_time_factor(&self) -> f64 {
        self.measured_elapsed().as_secs_f64() / self.audio_duration_seconds
    }
}

pub struct SqueezeformerTranscriber<B: Backend> {
    audio: AudioDecodeConfig,
    frontend: W2vBertFrontendConfig,
    model: SqueezeformerCtc<B>,
    tokenizer: SentencePieceTokenizer,
    device: B::Device,
    ctc: CtcDecoderConfig,
    text: TextDecoderConfig,
    tokenizer_load_elapsed: Option<Duration>,
}

impl<B: Backend> SqueezeformerTranscriber<B> {
    pub fn new(
        model: SqueezeformerCtc<B>,
        device: B::Device,
        config: SqueezeformerTranscriptionConfig,
    ) -> Result<Self> {
        let tokenizer_start = Instant::now();
        let tokenizer = load_sentencepiece_tokenizer(&config.text.tokenizer_path)?;

        Ok(Self {
            audio: config.audio,
            frontend: config.encoder.to_frontend_config(),
            model,
            tokenizer,
            device,
            ctc: config.ctc,
            text: config.text,
            tokenizer_load_elapsed: Some(tokenizer_start.elapsed()),
        })
    }

    pub fn from_parts(
        model: SqueezeformerCtc<B>,
        tokenizer: SentencePieceTokenizer,
        device: B::Device,
        config: SqueezeformerTranscriptionConfig,
    ) -> Self {
        Self {
            audio: config.audio,
            frontend: config.encoder.to_frontend_config(),
            model,
            tokenizer,
            device,
            ctc: config.ctc,
            text: config.text,
            tokenizer_load_elapsed: Some(Duration::ZERO),
        }
    }

    pub fn transcribe_audio_file(
        &mut self,
        audio_path: impl AsRef<std::path::Path>,
    ) -> Result<SqueezeformerTranscriptionResult> {
        let audio =
            audio_file_to_w2v_bert_features_with_config(audio_path, &self.audio, &self.frontend)?;
        self.transcribe_features(audio)
    }

    pub fn transcribe_audio_bytes(
        &mut self,
        audio_bytes: impl Into<Vec<u8>>,
        format_hint: Option<&str>,
    ) -> Result<SqueezeformerTranscriptionResult> {
        let audio = audio_bytes_to_w2v_bert_features_with_config(
            audio_bytes,
            format_hint,
            &self.audio,
            &self.frontend,
        )?;
        self.transcribe_features(audio)
    }

    pub fn transcribe_features(
        &mut self,
        audio: AudioFeatures,
    ) -> Result<SqueezeformerTranscriptionResult> {
        let audio_duration_seconds = audio.duration_seconds();
        let feature_rows = audio.features.rows;
        let feature_cols = audio.features.cols;
        let feature_count = audio.features.values.len();
        let audio_decode_elapsed = audio.decode_elapsed;
        let feature_elapsed = audio.feature_elapsed;

        let model_output = self.run_model(audio)?;
        let (candidates, text_decode_elapsed) = self.decode_candidates(model_output.candidates)?;
        let transcript = candidates
            .first()
            .map(|candidate| candidate.text.clone())
            .ok_or_else(|| anyhow!("decoder produced no candidates"))?;

        Ok(SqueezeformerTranscriptionResult {
            transcript,
            timings: SqueezeformerTimingReport {
                audio_duration_seconds,
                feature_rows,
                feature_cols,
                feature_count,
                audio_decode_elapsed,
                feature_elapsed,
                model_input_elapsed: model_output.input_elapsed,
                model_inference_elapsed: model_output.inference_elapsed,
                ctc_elapsed: model_output.ctc_elapsed,
                tokenizer_load_elapsed: self
                    .tokenizer_load_elapsed
                    .take()
                    .unwrap_or(Duration::ZERO),
                text_decode_elapsed,
                best_candidate: candidates.first().cloned(),
            },
            candidates,
        })
    }

    fn run_model(&self, audio: AudioFeatures) -> Result<SqueezeformerModelOutput> {
        let input_start = Instant::now();
        let rows = audio.features.rows;
        let cols = audio.features.cols;
        let values = audio.features.values;
        if values.len() != rows * cols {
            bail!(
                "feature matrix shape {rows}x{cols} implies {} values, got {}",
                rows * cols,
                values.len()
            );
        }

        let input =
            Tensor::<B, 3>::from_data(TensorData::new(values, [1, rows, cols]), &self.device);
        let input_elapsed = input_start.elapsed();

        let inference_start = Instant::now();
        let logits = self.model.forward(input);
        let inference_elapsed = inference_start.elapsed();

        let [batch_size, frames, vocab_size] = logits.dims();
        let logits = logits
            .into_data()
            .to_vec::<f32>()
            .context("failed to read Squeezeformer logits from Burn tensor")?;
        let shape = [
            i64::try_from(batch_size).context("batch size does not fit i64")?,
            i64::try_from(frames).context("frame count does not fit i64")?,
            i64::try_from(vocab_size).context("vocab size does not fit i64")?,
        ];

        let ctc_start = Instant::now();
        let candidates = threaded_ctc_beam_search_decode_n_best(
            &shape,
            &logits,
            self.ctc.blank_id,
            self.ctc.beam_width,
            self.ctc.n_best,
        )?;

        Ok(SqueezeformerModelOutput {
            candidates,
            input_elapsed,
            inference_elapsed,
            ctc_elapsed: ctc_start.elapsed(),
        })
    }

    fn decode_candidates(
        &self,
        candidates: Vec<CtcCandidate>,
    ) -> Result<(Vec<SqueezeformerScoredCandidate>, Duration)> {
        let text_decode_start = Instant::now();
        let processing = CandidateProcessingConfig::from(&self.text);
        let mut scored = candidates
            .into_iter()
            .filter_map(|candidate| {
                let text = self.tokenizer.decode_lossy(&candidate.token_ids);
                score_candidate_text(text, candidate.ctc_log_prob, &processing)
            })
            .collect::<Vec<_>>();

        scored.sort_by(|a, b| b.total_score.total_cmp(&a.total_score));
        let text_decode_elapsed = text_decode_start.elapsed();

        if scored.is_empty() {
            bail!("text decoder produced no candidates");
        }

        Ok((scored, text_decode_elapsed))
    }
}

struct SqueezeformerModelOutput {
    candidates: Vec<CtcCandidate>,
    input_elapsed: Duration,
    inference_elapsed: Duration,
    ctc_elapsed: Duration,
}

fn score_candidate_text(
    text: String,
    ctc_log_prob: f32,
    processing: &CandidateProcessingConfig,
) -> Option<SqueezeformerScoredCandidate> {
    let text = if processing.normalize_spaces {
        normalize_spaces(&text)
    } else {
        text
    };

    if processing.drop_empty_candidates && text.is_empty() {
        return None;
    }

    Some(SqueezeformerScoredCandidate {
        word_count: text.split_whitespace().count(),
        text,
        ctc_log_prob,
        total_score: ctc_log_prob,
    })
}

pub fn logits_to_ctc_candidates<B: Backend>(
    logits: Tensor<B, 3>,
    ctc: &CtcDecoderConfig,
) -> Result<Vec<CtcCandidate>> {
    let [batch_size, frames, vocab_size] = logits.dims();
    let values = logits
        .into_data()
        .to_vec::<f32>()
        .context("failed to read Squeezeformer logits from Burn tensor")?;
    let shape = [
        i64::try_from(batch_size).context("batch size does not fit i64")?,
        i64::try_from(frames).context("frame count does not fit i64")?,
        i64::try_from(vocab_size).context("vocab size does not fit i64")?,
    ];

    threaded_ctc_beam_search_decode_n_best(
        &shape,
        &values,
        ctc.blank_id,
        ctc.beam_width,
        ctc.n_best,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn logits_to_ctc_candidates_decodes_best_path() {
        let device = Default::default();
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    10.0, 0.0, 0.0, //
                    0.0, 10.0, 0.0, //
                    10.0, 0.0, 0.0, //
                    0.0, 0.0, 10.0,
                ],
                [1, 4, 3],
            ),
            &device,
        );
        let ctc = CtcDecoderConfig {
            blank_id: 0,
            beam_width: 4,
            n_best: 1,
        };

        let candidates = logits_to_ctc_candidates(logits, &ctc).unwrap();

        assert_eq!(candidates[0].token_ids, vec![1, 2]);
    }
}
