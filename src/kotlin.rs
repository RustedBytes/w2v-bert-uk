use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::{
    AcousticModelConfig, CandidateProcessingConfig, CtcDecoderConfig, DecoderConfig, EncoderConfig,
    LmConfig, RuntimeConfig, TextDecoderConfig, Transcriber as RustTranscriber,
    TranscriptionConfig, W2vBertEncoderConfig,
    audio::AudioDecodeConfig,
    init_ort,
    model::{ModelConfig, ModelOptimizationLevel},
    preload_cuda_dylibs as preload_cuda_dylibs_impl, transcribe_audio_file,
};

#[derive(Clone, Debug, uniffi::Record)]
pub struct KotlinOptions {
    pub model: String,
    pub tokenizer: String,
    pub lm: Option<String>,
    pub ort_dylib_path: Option<String>,
    pub ort_optimization: String,
    pub w2v_model_source: Option<String>,
    pub beam_width: i32,
    pub lm_weight: f32,
    pub word_bonus: f32,
    pub hot_words: Vec<String>,
    pub hot_word_bonus: f32,
    pub fallback_sample_rate: i32,
    pub w2v_sample_rate: Option<i32>,
    pub w2v_feature_size: Option<i32>,
    pub w2v_stride: Option<i32>,
    pub w2v_feature_dim: Option<i32>,
    pub w2v_padding_value: Option<f32>,
    pub blank_id: i32,
    pub n_best: Option<i32>,
    pub log_language_model: bool,
    pub log_accelerator: bool,
    pub skip_decode_errors: bool,
    pub normalize_spaces: bool,
    pub drop_empty_candidates: bool,
    pub lm_bos: bool,
    pub lm_eos: bool,
}

impl Default for KotlinOptions {
    fn default() -> Self {
        Self {
            model: "model_optimized.onnx".to_string(),
            tokenizer: "tokenizer.model".to_string(),
            lm: Some("lm.binary".to_string()),
            ort_dylib_path: None,
            ort_optimization: "disable".to_string(),
            w2v_model_source: None,
            beam_width: 32,
            lm_weight: 0.45,
            word_bonus: 0.2,
            hot_words: Vec::new(),
            hot_word_bonus: 0.0,
            fallback_sample_rate: 16_000,
            w2v_sample_rate: None,
            w2v_feature_size: None,
            w2v_stride: None,
            w2v_feature_dim: None,
            w2v_padding_value: None,
            blank_id: 0,
            n_best: Some(32),
            log_language_model: true,
            log_accelerator: true,
            skip_decode_errors: true,
            normalize_spaces: true,
            drop_empty_candidates: true,
            lm_bos: true,
            lm_eos: true,
        }
    }
}

#[derive(Debug, uniffi::Error)]
pub enum KotlinBindingError {
    Message { message: String },
}

impl std::fmt::Display for KotlinBindingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message { message } => formatter.write_str(message),
        }
    }
}

impl std::error::Error for KotlinBindingError {}

impl From<anyhow::Error> for KotlinBindingError {
    fn from(error: anyhow::Error) -> Self {
        Self::Message {
            message: error.to_string(),
        }
    }
}

type KotlinResult<T> = Result<T, KotlinBindingError>;

#[derive(uniffi::Object)]
pub struct KotlinTranscriber {
    inner: Mutex<RustTranscriber>,
}

#[uniffi::export]
impl KotlinTranscriber {
    #[uniffi::constructor]
    pub fn new(options: KotlinOptions) -> KotlinResult<Arc<Self>> {
        Ok(Arc::new(Self {
            inner: Mutex::new(RustTranscriber::new(config_from_options(options)?)?),
        }))
    }

    pub fn transcribe_file(&self, audio_file: String) -> KotlinResult<String> {
        let mut transcriber = self
            .inner
            .lock()
            .map_err(|_| KotlinBindingError::message("transcriber mutex is poisoned"))?;
        Ok(transcriber
            .transcribe_audio_file(PathBuf::from(audio_file))?
            .transcript)
    }

    pub fn transcribe_bytes(
        &self,
        audio_bytes: Vec<u8>,
        format_hint: Option<String>,
    ) -> KotlinResult<String> {
        let mut transcriber = self
            .inner
            .lock()
            .map_err(|_| KotlinBindingError::message("transcriber mutex is poisoned"))?;
        Ok(transcriber
            .transcribe_audio_bytes(audio_bytes, format_hint.as_deref())?
            .transcript)
    }
}

impl KotlinBindingError {
    fn message(message: impl Into<String>) -> Self {
        Self::Message {
            message: message.into(),
        }
    }
}

#[uniffi::export]
pub fn default_options() -> KotlinOptions {
    KotlinOptions::default()
}

#[uniffi::export]
pub fn initialize_ort(ort_dylib_path: Option<String>) -> KotlinResult<bool> {
    Ok(init_ort(
        ort_dylib_path.as_deref().map(PathBuf::from).as_deref(),
    )?)
}

#[uniffi::export]
pub fn preload_cuda_dylibs(
    cuda_lib_dir: Option<String>,
    cudnn_lib_dir: Option<String>,
) -> KotlinResult<()> {
    let cuda = cuda_lib_dir.map(PathBuf::from);
    let cudnn = cudnn_lib_dir.map(PathBuf::from);
    Ok(preload_cuda_dylibs_impl(cuda.as_deref(), cudnn.as_deref())?)
}

#[uniffi::export]
pub fn transcribe_file(audio_file: String, options: KotlinOptions) -> KotlinResult<String> {
    Ok(
        transcribe_audio_file(PathBuf::from(audio_file), &config_from_options(options)?)?
            .transcript,
    )
}

fn config_from_options(options: KotlinOptions) -> anyhow::Result<TranscriptionConfig> {
    let normalize_spaces = options.normalize_spaces;
    let drop_empty_candidates = options.drop_empty_candidates;
    let beam_width = non_zero_i32(options.beam_width, 32) as usize;

    Ok(TranscriptionConfig {
        runtime: RuntimeConfig {
            ort_dylib_path: options.ort_dylib_path.map(PathBuf::from),
        },
        audio: AudioDecodeConfig {
            fallback_sample_rate: non_zero_i32(options.fallback_sample_rate, 16_000),
            skip_decode_errors: options.skip_decode_errors,
            ffmpeg_fallback: true,
        },
        encoder: EncoderConfig {
            w2v_bert: W2vBertEncoderConfig {
                model_source: options.w2v_model_source,
                sample_rate: positive_opt_i32(options.w2v_sample_rate),
                feature_size: positive_opt_i32(options.w2v_feature_size)
                    .map(|value| value as usize),
                stride: positive_opt_i32(options.w2v_stride).map(|value| value as usize),
                feature_dim: positive_opt_i32(options.w2v_feature_dim).map(|value| value as usize),
                padding_value: options.w2v_padding_value,
            },
        },
        model: AcousticModelConfig {
            path: path_or_default(options.model, "model_optimized.onnx"),
            session: ModelConfig {
                optimization_level: parse_optimization_level(&options.ort_optimization)?,
                log_accelerator: options.log_accelerator,
            },
        },
        decoder: DecoderConfig {
            ctc: CtcDecoderConfig {
                blank_id: options.blank_id.max(0) as u32,
                beam_width,
                n_best: positive_opt_i32(options.n_best)
                    .map(|value| value as usize)
                    .unwrap_or(beam_width),
            },
            text: TextDecoderConfig {
                tokenizer_path: path_or_default(options.tokenizer, "tokenizer.model"),
                normalize_spaces,
                drop_empty_candidates,
            },
            language_model: options
                .lm
                .filter(|path| !path.is_empty())
                .map(|path| LmConfig {
                    path: PathBuf::from(path),
                    weight: if options.lm_weight == 0.0 {
                        0.45
                    } else {
                        options.lm_weight
                    },
                    word_bonus: if options.word_bonus == 0.0 {
                        0.2
                    } else {
                        options.word_bonus
                    },
                    hot_words: options.hot_words,
                    hot_word_bonus: options.hot_word_bonus,
                    log_language_model: options.log_language_model,
                    bos: options.lm_bos,
                    eos: options.lm_eos,
                    candidate_processing: CandidateProcessingConfig {
                        normalize_spaces,
                        drop_empty_candidates,
                    },
                }),
        },
    })
}

fn parse_optimization_level(value: &str) -> anyhow::Result<ModelOptimizationLevel> {
    match value {
        "" | "disable" => Ok(ModelOptimizationLevel::Disable),
        "level1" => Ok(ModelOptimizationLevel::Level1),
        "level2" => Ok(ModelOptimizationLevel::Level2),
        "level3" => Ok(ModelOptimizationLevel::Level3),
        "all" => Ok(ModelOptimizationLevel::All),
        other => anyhow::bail!(
            "invalid ortOptimization {other:?}; expected disable, level1, level2, level3, or all"
        ),
    }
}

fn path_or_default(value: String, default: &str) -> PathBuf {
    if value.is_empty() {
        PathBuf::from(default)
    } else {
        PathBuf::from(value)
    }
}

fn non_zero_i32(value: i32, default: u32) -> u32 {
    if value <= 0 { default } else { value as u32 }
}

fn positive_opt_i32(value: Option<i32>) -> Option<u32> {
    value.and_then(|value| (value > 0).then_some(value as u32))
}
