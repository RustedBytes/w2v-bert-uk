use std::path::PathBuf;
use std::time::Duration;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::{
    AcousticModelConfig, CandidateProcessingConfig, CtcDecoderConfig, DecoderConfig, EncoderConfig,
    LmConfig, RuntimeConfig, ScoredCandidate, TextDecoderConfig, TimingReport,
    Transcriber as RustTranscriber, TranscriptionConfig, TranscriptionResult, W2vBertEncoderConfig,
    audio::AudioDecodeConfig,
    init_ort,
    model::{ModelConfig, ModelOptimizationLevel},
    preload_cuda_dylibs as preload_cuda_dylibs_impl, transcribe_audio_bytes, transcribe_audio_file,
};

#[napi(object)]
pub struct TranscriptionOptions {
    pub model: Option<String>,
    pub tokenizer: Option<String>,
    pub beam_width: Option<u32>,
    pub lm: Option<String>,
    pub lm_weight: Option<f64>,
    pub word_bonus: Option<f64>,
    pub log_language_model: Option<bool>,
    pub ort_dylib_path: Option<String>,
    pub ort_optimization: Option<String>,
    pub log_accelerator: Option<bool>,
    pub fallback_sample_rate: Option<u32>,
    pub skip_decode_errors: Option<bool>,
    pub w2v_model_source: Option<String>,
    pub w2v_sample_rate: Option<u32>,
    pub w2v_feature_size: Option<u32>,
    pub w2v_stride: Option<u32>,
    pub w2v_feature_dim: Option<u32>,
    pub w2v_padding_value: Option<f64>,
    pub blank_id: Option<u32>,
    pub n_best: Option<u32>,
    pub normalize_spaces: Option<bool>,
    pub drop_empty_candidates: Option<bool>,
    pub lm_bos: Option<bool>,
    pub lm_eos: Option<bool>,
}

#[napi(object)]
pub struct TranscriptionReport {
    pub transcript: String,
    pub timings: TimingReportObject,
    pub candidates: Vec<ScoredCandidateObject>,
}

#[napi(object)]
pub struct TimingReportObject {
    pub audio_duration_seconds: f64,
    pub feature_rows: u32,
    pub feature_cols: u32,
    pub feature_count: u32,
    pub audio_decode_seconds: f64,
    pub feature_seconds: f64,
    pub model_session_seconds: f64,
    pub model_input_seconds: f64,
    pub model_inference_seconds: f64,
    pub model_ctc_seconds: f64,
    pub tokenizer_load_seconds: f64,
    pub text_decode_seconds: f64,
    pub lm_seconds: f64,
    pub measured_seconds: f64,
    pub best_candidate: Option<ScoredCandidateObject>,
}

#[napi(object)]
pub struct ScoredCandidateObject {
    pub text: String,
    pub ctc_log_prob: f64,
    pub lm_log_prob: f64,
    pub word_count: u32,
    pub total_score: f64,
}

#[napi(js_name = "initializeOrt")]
pub fn initialize_ort(ort_dylib_path: Option<String>) -> Result<bool> {
    init_ort(ort_dylib_path.map(PathBuf::from).as_deref()).map_err(to_napi_error)
}

#[napi(js_name = "preloadCudaDylibs")]
pub fn preload_cuda_dylibs(
    cuda_lib_dir: Option<String>,
    cudnn_lib_dir: Option<String>,
) -> Result<()> {
    preload_cuda_dylibs_impl(
        cuda_lib_dir.map(PathBuf::from).as_deref(),
        cudnn_lib_dir.map(PathBuf::from).as_deref(),
    )
    .map_err(to_napi_error)
}

#[napi(js_name = "transcribeFile")]
pub fn transcribe_file(
    audio_file: String,
    options: Option<TranscriptionOptions>,
) -> Result<String> {
    let config = build_config(options)?;
    transcribe_audio_file(PathBuf::from(audio_file), &config)
        .map(|result| result.transcript)
        .map_err(to_napi_error)
}

#[napi(js_name = "transcribeBytes")]
pub fn transcribe_bytes(
    audio_bytes: Buffer,
    format_hint: Option<String>,
    options: Option<TranscriptionOptions>,
) -> Result<String> {
    let config = build_config(options)?;
    transcribe_audio_bytes(audio_bytes.to_vec(), format_hint.as_deref(), &config)
        .map(|result| result.transcript)
        .map_err(to_napi_error)
}

#[napi(js_name = "transcribeFileWithReport")]
pub fn transcribe_file_with_report(
    audio_file: String,
    options: Option<TranscriptionOptions>,
) -> Result<TranscriptionReport> {
    let config = build_config(options)?;
    let result =
        transcribe_audio_file(PathBuf::from(audio_file), &config).map_err(to_napi_error)?;
    Ok(transcription_result_to_node(result))
}

#[napi(js_name = "transcribeBytesWithReport")]
pub fn transcribe_bytes_with_report(
    audio_bytes: Buffer,
    format_hint: Option<String>,
    options: Option<TranscriptionOptions>,
) -> Result<TranscriptionReport> {
    let config = build_config(options)?;
    let result = transcribe_audio_bytes(audio_bytes.to_vec(), format_hint.as_deref(), &config)
        .map_err(to_napi_error)?;
    Ok(transcription_result_to_node(result))
}

#[napi]
pub struct Transcriber {
    inner: RustTranscriber,
}

#[napi]
impl Transcriber {
    #[napi(constructor)]
    pub fn new(options: Option<TranscriptionOptions>) -> Result<Self> {
        let config = build_config(options)?;
        let inner = RustTranscriber::new(config).map_err(to_napi_error)?;
        Ok(Self { inner })
    }

    #[napi(js_name = "transcribeFile")]
    pub fn transcribe_file(&mut self, audio_file: String) -> Result<String> {
        self.inner
            .transcribe_audio_file(PathBuf::from(audio_file))
            .map(|result| result.transcript)
            .map_err(to_napi_error)
    }

    #[napi(js_name = "transcribeFileWithReport")]
    pub fn transcribe_file_with_report(
        &mut self,
        audio_file: String,
    ) -> Result<TranscriptionReport> {
        let result = self
            .inner
            .transcribe_audio_file(PathBuf::from(audio_file))
            .map_err(to_napi_error)?;
        Ok(transcription_result_to_node(result))
    }

    #[napi(js_name = "transcribeBytes")]
    pub fn transcribe_bytes(
        &mut self,
        audio_bytes: Buffer,
        format_hint: Option<String>,
    ) -> Result<String> {
        self.inner
            .transcribe_audio_bytes(audio_bytes.to_vec(), format_hint.as_deref())
            .map(|result| result.transcript)
            .map_err(to_napi_error)
    }

    #[napi(js_name = "transcribeBytesWithReport")]
    pub fn transcribe_bytes_with_report(
        &mut self,
        audio_bytes: Buffer,
        format_hint: Option<String>,
    ) -> Result<TranscriptionReport> {
        let result = self
            .inner
            .transcribe_audio_bytes(audio_bytes.to_vec(), format_hint.as_deref())
            .map_err(to_napi_error)?;
        Ok(transcription_result_to_node(result))
    }
}

fn build_config(options: Option<TranscriptionOptions>) -> Result<TranscriptionConfig> {
    let options = options.unwrap_or_default();
    let beam_width = options.beam_width.unwrap_or(32) as usize;
    let normalize_spaces = options.normalize_spaces.unwrap_or(true);
    let drop_empty_candidates = options.drop_empty_candidates.unwrap_or(true);
    let candidate_processing = CandidateProcessingConfig {
        normalize_spaces,
        drop_empty_candidates,
    };

    Ok(TranscriptionConfig {
        runtime: RuntimeConfig {
            ort_dylib_path: options.ort_dylib_path.map(PathBuf::from),
        },
        audio: AudioDecodeConfig {
            fallback_sample_rate: options.fallback_sample_rate.unwrap_or(16_000),
            skip_decode_errors: options.skip_decode_errors.unwrap_or(true),
            ffmpeg_fallback: true,
        },
        encoder: EncoderConfig {
            w2v_bert: W2vBertEncoderConfig {
                model_source: options.w2v_model_source,
                sample_rate: options.w2v_sample_rate,
                feature_size: options.w2v_feature_size.map(|value| value as usize),
                stride: options.w2v_stride.map(|value| value as usize),
                feature_dim: options.w2v_feature_dim.map(|value| value as usize),
                padding_value: options.w2v_padding_value.map(|value| value as f32),
            },
        },
        model: AcousticModelConfig {
            path: options
                .model
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("model_optimized.onnx")),
            session: ModelConfig {
                optimization_level: parse_optimization_level(
                    options.ort_optimization.as_deref().unwrap_or("disable"),
                )?,
                log_accelerator: options.log_accelerator.unwrap_or(true),
            },
        },
        decoder: DecoderConfig {
            ctc: CtcDecoderConfig {
                blank_id: options.blank_id.unwrap_or(0),
                beam_width,
                n_best: options
                    .n_best
                    .map(|value| value as usize)
                    .unwrap_or(beam_width),
            },
            text: TextDecoderConfig {
                tokenizer_path: options
                    .tokenizer
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("tokenizer.model")),
                normalize_spaces,
                drop_empty_candidates,
            },
            language_model: options.lm.map(|path| LmConfig {
                path: PathBuf::from(path),
                weight: options.lm_weight.unwrap_or(0.45) as f32,
                word_bonus: options.word_bonus.unwrap_or(0.2) as f32,
                log_language_model: options.log_language_model.unwrap_or(true),
                bos: options.lm_bos.unwrap_or(true),
                eos: options.lm_eos.unwrap_or(true),
                candidate_processing,
            }),
        },
    })
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            model: None,
            tokenizer: None,
            beam_width: None,
            lm: None,
            lm_weight: None,
            word_bonus: None,
            log_language_model: None,
            ort_dylib_path: None,
            ort_optimization: None,
            log_accelerator: None,
            fallback_sample_rate: None,
            skip_decode_errors: None,
            w2v_model_source: None,
            w2v_sample_rate: None,
            w2v_feature_size: None,
            w2v_stride: None,
            w2v_feature_dim: None,
            w2v_padding_value: None,
            blank_id: None,
            n_best: None,
            normalize_spaces: None,
            drop_empty_candidates: None,
            lm_bos: None,
            lm_eos: None,
        }
    }
}

fn transcription_result_to_node(result: TranscriptionResult) -> TranscriptionReport {
    TranscriptionReport {
        transcript: result.transcript,
        timings: timings_to_node(&result.timings),
        candidates: result.candidates.iter().map(candidate_to_node).collect(),
    }
}

fn timings_to_node(timings: &TimingReport) -> TimingReportObject {
    TimingReportObject {
        audio_duration_seconds: timings.audio_duration_seconds,
        feature_rows: timings.feature_rows as u32,
        feature_cols: timings.feature_cols as u32,
        feature_count: timings.feature_count as u32,
        audio_decode_seconds: duration_seconds(timings.audio_decode_elapsed),
        feature_seconds: duration_seconds(timings.feature_elapsed),
        model_session_seconds: duration_seconds(timings.model.session_elapsed),
        model_input_seconds: duration_seconds(timings.model.input_elapsed),
        model_inference_seconds: duration_seconds(timings.model.inference_elapsed),
        model_ctc_seconds: duration_seconds(timings.model.ctc_elapsed),
        tokenizer_load_seconds: duration_seconds(timings.tokenizer_load_elapsed),
        text_decode_seconds: duration_seconds(timings.text_decode_elapsed),
        lm_seconds: duration_seconds(timings.lm_elapsed),
        measured_seconds: duration_seconds(timings.measured_elapsed()),
        best_candidate: timings.best_candidate.as_ref().map(candidate_to_node),
    }
}

fn candidate_to_node(candidate: &ScoredCandidate) -> ScoredCandidateObject {
    ScoredCandidateObject {
        text: candidate.text.clone(),
        ctc_log_prob: candidate.ctc_log_prob as f64,
        lm_log_prob: candidate.lm_log_prob as f64,
        word_count: candidate.word_count as u32,
        total_score: candidate.total_score as f64,
    }
}

fn duration_seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn to_napi_error(error: anyhow::Error) -> napi::Error {
    napi::Error::from_reason(error.to_string())
}

fn parse_optimization_level(value: &str) -> Result<ModelOptimizationLevel> {
    match value {
        "disable" => Ok(ModelOptimizationLevel::Disable),
        "level1" => Ok(ModelOptimizationLevel::Level1),
        "level2" => Ok(ModelOptimizationLevel::Level2),
        "level3" => Ok(ModelOptimizationLevel::Level3),
        "all" => Ok(ModelOptimizationLevel::All),
        other => Err(napi::Error::from_reason(format!(
            "invalid ort_optimization {other:?}; expected disable, level1, level2, level3, or all"
        ))),
    }
}
