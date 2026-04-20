use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use asr_features::{W2vBertFrontendConfig, w2v_bert_frontend_config};
use kenlm::{Config as KenlmConfig, Model as KenlmModel};
use splintr::SentencePieceTokenizer;

use crate::audio::{
    AudioDecodeConfig, AudioFeatures, audio_bytes_to_w2v_bert_features_with_config,
    audio_file_to_w2v_bert_features_with_config,
};
use crate::model::{CtcModel, ModelConfig, ModelOutput};
use crate::tokenizer::load_sentencepiece_tokenizer;

pub mod audio;
#[cfg(any(feature = "c", feature = "cpp", feature = "csharp"))]
#[path = "csharp.rs"]
mod c_abi;
pub mod ctc;
#[cfg(feature = "java")]
mod java;
pub mod model;
#[cfg(feature = "nodejs")]
mod nodejs;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "ruby")]
mod ruby;
#[cfg(feature = "swift")]
mod swift;
pub mod tokenizer;

#[derive(Clone, Debug, Default)]
pub struct TranscriptionConfig {
    pub runtime: RuntimeConfig,
    pub audio: AudioDecodeConfig,
    pub encoder: EncoderConfig,
    pub model: AcousticModelConfig,
    pub decoder: DecoderConfig,
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeConfig {
    pub ort_dylib_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Default)]
pub struct EncoderConfig {
    pub w2v_bert: W2vBertEncoderConfig,
}

#[derive(Clone, Debug, Default)]
pub struct W2vBertEncoderConfig {
    pub model_source: Option<String>,
    pub sample_rate: Option<u32>,
    pub feature_size: Option<usize>,
    pub stride: Option<usize>,
    pub feature_dim: Option<usize>,
    pub padding_value: Option<f32>,
}

impl W2vBertEncoderConfig {
    pub fn to_frontend_config(&self) -> W2vBertFrontendConfig {
        w2v_bert_frontend_config(
            self.model_source.clone(),
            self.sample_rate,
            self.feature_size,
            self.stride,
            self.feature_dim,
            self.padding_value,
        )
    }
}

#[derive(Clone, Debug)]
pub struct AcousticModelConfig {
    pub path: PathBuf,
    pub session: ModelConfig,
}

impl Default for AcousticModelConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("model_optimized.onnx"),
            session: ModelConfig::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecoderConfig {
    pub ctc: CtcDecoderConfig,
    pub text: TextDecoderConfig,
    pub language_model: Option<LmConfig>,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            ctc: CtcDecoderConfig::default(),
            text: TextDecoderConfig::default(),
            language_model: Some(LmConfig::default()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CtcDecoderConfig {
    pub blank_id: u32,
    pub beam_width: usize,
    pub n_best: usize,
}

impl Default for CtcDecoderConfig {
    fn default() -> Self {
        Self {
            blank_id: 0,
            beam_width: 32,
            n_best: 32,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TextDecoderConfig {
    pub tokenizer_path: PathBuf,
    pub normalize_spaces: bool,
    pub drop_empty_candidates: bool,
}

impl Default for TextDecoderConfig {
    fn default() -> Self {
        Self {
            tokenizer_path: PathBuf::from("tokenizer.model"),
            normalize_spaces: true,
            drop_empty_candidates: true,
        }
    }
}

impl TranscriptionConfig {
    pub fn with_legacy_paths(
        model_path: PathBuf,
        tokenizer_path: PathBuf,
        beam_width: usize,
        lm_config: Option<LmConfig>,
        ort_dylib_path: Option<PathBuf>,
    ) -> Self {
        Self {
            runtime: RuntimeConfig { ort_dylib_path },
            model: AcousticModelConfig {
                path: model_path,
                ..AcousticModelConfig::default()
            },
            decoder: DecoderConfig {
                ctc: CtcDecoderConfig {
                    beam_width,
                    n_best: beam_width,
                    ..CtcDecoderConfig::default()
                },
                text: TextDecoderConfig {
                    tokenizer_path,
                    ..TextDecoderConfig::default()
                },
                language_model: lm_config,
            },
            ..TranscriptionConfig::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct CandidateProcessingConfig {
    pub normalize_spaces: bool,
    pub drop_empty_candidates: bool,
}

impl Default for CandidateProcessingConfig {
    fn default() -> Self {
        Self {
            normalize_spaces: true,
            drop_empty_candidates: true,
        }
    }
}

impl From<&TextDecoderConfig> for CandidateProcessingConfig {
    fn from(value: &TextDecoderConfig) -> Self {
        Self {
            normalize_spaces: value.normalize_spaces,
            drop_empty_candidates: value.drop_empty_candidates,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LmConfig {
    pub path: PathBuf,
    pub weight: f32,
    pub word_bonus: f32,
    pub log_language_model: bool,
    pub bos: bool,
    pub eos: bool,
    pub candidate_processing: CandidateProcessingConfig,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("lm.binary"),
            weight: 0.45,
            word_bonus: 0.2,
            log_language_model: true,
            bos: true,
            eos: true,
            candidate_processing: CandidateProcessingConfig::default(),
        }
    }
}

pub struct TranscriptionResult {
    pub transcript: String,
    pub timings: TimingReport,
    pub candidates: Vec<ScoredCandidate>,
}

pub struct Transcriber {
    audio: AudioDecodeConfig,
    frontend: W2vBertFrontendConfig,
    model: CtcModel,
    tokenizer: SentencePieceTokenizer,
    ctc: CtcDecoderConfig,
    text: TextDecoderConfig,
    language_model: Option<LmDecoder>,
    next_session_elapsed: Option<Duration>,
}

impl Transcriber {
    pub fn new(config: TranscriptionConfig) -> Result<Self> {
        init_ort(config.runtime.ort_dylib_path.as_deref())?;
        let frontend = config.encoder.w2v_bert.to_frontend_config();
        let model = CtcModel::load(&config.model.path, &config.model.session)?;
        let next_session_elapsed = Some(model.session_elapsed());
        let tokenizer = load_sentencepiece_tokenizer(&config.decoder.text.tokenizer_path)?;
        let language_model = config
            .decoder
            .language_model
            .map(LmDecoder::new)
            .transpose()?;

        Ok(Self {
            audio: config.audio,
            frontend,
            model,
            tokenizer,
            ctc: config.decoder.ctc,
            text: config.decoder.text,
            language_model,
            next_session_elapsed,
        })
    }

    pub fn transcribe_audio_file(
        &mut self,
        audio_path: impl AsRef<Path>,
    ) -> Result<TranscriptionResult> {
        let audio =
            audio_file_to_w2v_bert_features_with_config(audio_path, &self.audio, &self.frontend)?;
        self.transcribe_features(audio)
    }

    pub fn transcribe_audio_bytes(
        &mut self,
        audio_bytes: impl Into<Vec<u8>>,
        format_hint: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let audio = audio_bytes_to_w2v_bert_features_with_config(
            audio_bytes,
            format_hint,
            &self.audio,
            &self.frontend,
        )?;
        self.transcribe_features(audio)
    }

    pub fn transcribe_features(&mut self, audio: AudioFeatures) -> Result<TranscriptionResult> {
        let audio_duration_seconds = audio.duration_seconds();
        let feature_rows = audio.features.rows;
        let feature_cols = audio.features.cols;
        let feature_count = audio.features.values.len();
        let reported_session_elapsed = self.next_session_elapsed.take().unwrap_or(Duration::ZERO);
        let model = self.model.run_with_reported_session_elapsed(
            audio.features,
            self.ctc.blank_id,
            self.ctc.beam_width,
            self.ctc.n_best,
            reported_session_elapsed,
        )?;

        decode_model_output(
            model,
            &self.tokenizer,
            &self.text,
            &self.language_model,
            audio_duration_seconds,
            feature_rows,
            feature_cols,
            feature_count,
            audio.decode_elapsed,
            audio.feature_elapsed,
            Duration::ZERO,
        )
    }
}

struct DecodedCandidate {
    text: String,
    ctc_log_prob: f32,
}

struct LmDecoder {
    model: KenlmModel,
    config: LmConfig,
}

impl LmDecoder {
    fn new(config: LmConfig) -> Result<Self> {
        if config.log_language_model {
            eprintln!(
                "KenLM: {} weight={:.3} word_bonus={:.3}",
                config.path.display(),
                config.weight,
                config.word_bonus
            );
        }

        let model = KenlmModel::with_config(
            &config.path,
            KenlmConfig {
                show_progress: false,
                ..KenlmConfig::default()
            },
        )
        .with_context(|| format!("failed to load KenLM model {}", config.path.display()))?;

        Ok(Self { model, config })
    }
}

#[derive(Clone)]
pub struct ScoredCandidate {
    pub text: String,
    pub ctc_log_prob: f32,
    pub lm_log_prob: f32,
    pub word_count: usize,
    pub total_score: f32,
}

pub struct TimingReport {
    pub audio_duration_seconds: f64,
    pub feature_rows: usize,
    pub feature_cols: usize,
    pub feature_count: usize,
    pub audio_decode_elapsed: Duration,
    pub feature_elapsed: Duration,
    pub model: ModelOutput,
    pub tokenizer_load_elapsed: Duration,
    pub text_decode_elapsed: Duration,
    pub lm_elapsed: Duration,
    pub best_candidate: Option<ScoredCandidate>,
}

impl TimingReport {
    pub fn measured_elapsed(&self) -> Duration {
        self.audio_decode_elapsed
            + self.feature_elapsed
            + self.model.session_elapsed
            + self.model.input_elapsed
            + self.model.inference_elapsed
            + self.model.ctc_elapsed
            + self.tokenizer_load_elapsed
            + self.text_decode_elapsed
            + self.lm_elapsed
    }

    pub fn real_time_factor(&self) -> f64 {
        self.measured_elapsed().as_secs_f64() / self.audio_duration_seconds
    }
}

#[cfg(not(feature = "ort-dynamic"))]
pub fn init_ort(ort_dylib_path: Option<&Path>) -> Result<bool> {
    if let Some(path) = ort_dylib_path {
        return Err(anyhow!(
            "{} was provided, but this binary was built with the bundled ONNX Runtime. Rebuild with --features ort-dynamic to load an external ONNX Runtime library.",
            path.display()
        ));
    }

    Ok(ort::init().commit())
}

#[cfg(feature = "ort-dynamic")]
pub fn init_ort(ort_dylib_path: Option<&Path>) -> Result<bool> {
    let committed = if let Some(path) = ort_dylib_path {
        ort::init_from(path)
            .map_err(|error| {
                anyhow!(
                    "failed to initialize ONNX Runtime from {}: {error}",
                    path.display()
                )
            })?
            .commit()
    } else {
        ort::init().commit()
    };

    Ok(committed)
}

#[cfg(feature = "cuda")]
pub fn preload_cuda_dylibs(
    cuda_lib_dir: Option<&Path>,
    cudnn_lib_dir: Option<&Path>,
) -> Result<()> {
    ort::ep::cuda::preload_dylibs(cuda_lib_dir, cudnn_lib_dir)
        .map_err(|error| anyhow!("failed to preload CUDA/cuDNN libraries: {error}"))
}

#[cfg(not(feature = "cuda"))]
pub fn preload_cuda_dylibs(
    _cuda_lib_dir: Option<&Path>,
    _cudnn_lib_dir: Option<&Path>,
) -> Result<()> {
    Err(anyhow!(
        "CUDA library preloading requires building with the cuda feature"
    ))
}

pub fn transcribe_audio_file(
    audio_path: impl AsRef<Path>,
    config: &TranscriptionConfig,
) -> Result<TranscriptionResult> {
    let mut transcriber = Transcriber::new(config.clone())?;
    transcriber.transcribe_audio_file(audio_path)
}

pub fn transcribe_audio_bytes(
    audio_bytes: impl Into<Vec<u8>>,
    format_hint: Option<&str>,
    config: &TranscriptionConfig,
) -> Result<TranscriptionResult> {
    let mut transcriber = Transcriber::new(config.clone())?;
    transcriber.transcribe_audio_bytes(audio_bytes, format_hint)
}

pub fn transcribe_features(
    audio: AudioFeatures,
    config: &TranscriptionConfig,
) -> Result<TranscriptionResult> {
    let mut transcriber = Transcriber::new(config.clone())?;
    transcriber.transcribe_features(audio)
}

fn decode_model_output(
    model: ModelOutput,
    tokenizer: &SentencePieceTokenizer,
    text_config: &TextDecoderConfig,
    language_model: &Option<LmDecoder>,
    audio_duration_seconds: f64,
    feature_rows: usize,
    feature_cols: usize,
    feature_count: usize,
    audio_decode_elapsed: Duration,
    feature_elapsed: Duration,
    tokenizer_load_elapsed: Duration,
) -> Result<TranscriptionResult> {
    let text_decode_start = Instant::now();
    let decoded_candidates = model
        .candidates
        .iter()
        .map(|candidate| DecodedCandidate {
            text: tokenizer.decode_lossy(&candidate.token_ids),
            ctc_log_prob: candidate.ctc_log_prob,
        })
        .collect::<Vec<_>>();
    let text_decode_elapsed = text_decode_start.elapsed();

    let lm_start = Instant::now();
    let ranked = if let Some(language_model) = language_model {
        rerank_with_kenlm(decoded_candidates, language_model)?
    } else {
        decoded_candidates
            .into_iter()
            .filter_map(|candidate| score_without_lm(candidate, text_config.into()))
            .collect()
    };
    let lm_elapsed = lm_start.elapsed();
    let transcript = ranked
        .first()
        .map(|candidate| candidate.text.clone())
        .ok_or_else(|| anyhow!("decoder produced no candidates"))?;

    Ok(TranscriptionResult {
        transcript,
        timings: TimingReport {
            audio_duration_seconds,
            feature_rows,
            feature_cols,
            feature_count,
            audio_decode_elapsed,
            feature_elapsed,
            model,
            tokenizer_load_elapsed,
            text_decode_elapsed,
            lm_elapsed,
            best_candidate: ranked.first().cloned(),
        },
        candidates: ranked,
    })
}

fn rerank_with_kenlm(
    candidates: Vec<DecodedCandidate>,
    language_model: &LmDecoder,
) -> Result<Vec<ScoredCandidate>> {
    let config = &language_model.config;

    let mut deduped = HashMap::<String, f32>::new();
    for candidate in candidates {
        let Some(text) = process_candidate_text(&candidate.text, &config.candidate_processing)
        else {
            continue;
        };

        deduped
            .entry(text)
            .and_modify(|best| *best = best.max(candidate.ctc_log_prob))
            .or_insert(candidate.ctc_log_prob);
    }

    let mut ranked = Vec::with_capacity(deduped.len());
    for (text, ctc_log_prob) in deduped {
        let lm_log_prob = language_model
            .model
            .score(&text, config.bos, config.eos)
            .with_context(|| format!("failed to score candidate with KenLM: {text}"))?
            * std::f32::consts::LN_10;
        let word_count = text.split_whitespace().count();
        let total_score =
            ctc_log_prob + config.weight * lm_log_prob + config.word_bonus * word_count as f32;

        ranked.push(ScoredCandidate {
            text,
            ctc_log_prob,
            lm_log_prob,
            word_count,
            total_score,
        });
    }

    ranked.sort_by(|a, b| b.total_score.total_cmp(&a.total_score));
    Ok(ranked)
}

fn score_without_lm(
    candidate: DecodedCandidate,
    processing: CandidateProcessingConfig,
) -> Option<ScoredCandidate> {
    let text = process_candidate_text(&candidate.text, &processing)?;
    Some(ScoredCandidate {
        word_count: text.split_whitespace().count(),
        text,
        ctc_log_prob: candidate.ctc_log_prob,
        lm_log_prob: 0.0,
        total_score: candidate.ctc_log_prob,
    })
}

fn process_candidate_text(text: &str, config: &CandidateProcessingConfig) -> Option<String> {
    let text = if config.normalize_spaces {
        normalize_spaces(text)
    } else {
        text.to_string()
    };

    if config.drop_empty_candidates && text.is_empty() {
        None
    } else {
        Some(text)
    }
}

pub fn normalize_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds >= 1.0 {
        format!("{seconds:.3}s")
    } else {
        format!("{:.3}ms", seconds * 1_000.0)
    }
}
