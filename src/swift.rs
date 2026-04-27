use std::path::PathBuf;

use crate::{
    AcousticModelConfig, CandidateProcessingConfig, CtcDecoderConfig, DecoderConfig, EncoderConfig,
    LmConfig, RuntimeConfig, TextDecoderConfig, Transcriber as RustTranscriber,
    TranscriptionConfig, W2vBertEncoderConfig,
    audio::AudioDecodeConfig,
    init_ort,
    model::{ModelConfig, ModelOptimizationLevel},
    preload_cuda_dylibs as preload_cuda_dylibs_impl, transcribe_audio_bytes, transcribe_audio_file,
};

#[swift_bridge::bridge]
mod ffi {
    #[swift_bridge(swift_repr = "struct")]
    struct SwiftTranscriptionOptions {
        model: String,
        tokenizer: String,
        lm: String,
        ort_dylib_path: String,
        ort_optimization: String,
        w2v_model_source: String,
        beam_width: u32,
        lm_weight: f32,
        word_bonus: f32,
        fallback_sample_rate: u32,
        w2v_sample_rate: u32,
        w2v_feature_size: u32,
        w2v_stride: u32,
        w2v_feature_dim: u32,
        w2v_padding_value: f32,
        blank_id: u32,
        n_best: u32,
        use_language_model: bool,
        log_language_model: bool,
        log_accelerator: bool,
        skip_decode_errors: bool,
        normalize_spaces: bool,
        drop_empty_candidates: bool,
        lm_bos: bool,
        lm_eos: bool,
    }

    extern "Rust" {
        type SwiftTranscriber;

        fn swift_options_default() -> SwiftTranscriptionOptions;
        fn swift_initialize_ort(ort_dylib_path: String) -> Result<bool, String>;
        fn swift_preload_cuda_dylibs(
            cuda_lib_dir: String,
            cudnn_lib_dir: String,
        ) -> Result<(), String>;
        fn swift_transcribe_file(
            audio_file: String,
            options: SwiftTranscriptionOptions,
        ) -> Result<String, String>;
        fn swift_transcribe_bytes(
            audio_bytes: Vec<u8>,
            format_hint: String,
            options: SwiftTranscriptionOptions,
        ) -> Result<String, String>;
        fn swift_transcriber_new(
            options: SwiftTranscriptionOptions,
        ) -> Result<SwiftTranscriber, String>;
        fn transcribe_file(&mut self, audio_file: String) -> Result<String, String>;
        fn transcribe_bytes(
            &mut self,
            audio_bytes: Vec<u8>,
            format_hint: String,
        ) -> Result<String, String>;
    }
}

use ffi::SwiftTranscriptionOptions;

pub struct SwiftTranscriber {
    inner: RustTranscriber,
}

fn swift_options_default() -> SwiftTranscriptionOptions {
    SwiftTranscriptionOptions {
        model: "model_optimized.onnx".to_owned(),
        tokenizer: "tokenizer.model".to_owned(),
        lm: "lm.binary".to_owned(),
        ort_dylib_path: String::new(),
        ort_optimization: "disable".to_owned(),
        w2v_model_source: String::new(),
        beam_width: 32,
        lm_weight: 0.45,
        word_bonus: 0.2,
        fallback_sample_rate: 16_000,
        w2v_sample_rate: 0,
        w2v_feature_size: 0,
        w2v_stride: 0,
        w2v_feature_dim: 0,
        w2v_padding_value: f32::NAN,
        blank_id: 0,
        n_best: 0,
        use_language_model: true,
        log_language_model: true,
        log_accelerator: true,
        skip_decode_errors: true,
        normalize_spaces: true,
        drop_empty_candidates: true,
        lm_bos: true,
        lm_eos: true,
    }
}

fn swift_initialize_ort(ort_dylib_path: String) -> Result<bool, String> {
    init_ort(path_opt_from_string(ort_dylib_path).as_deref()).map_err(error_to_string)
}

fn swift_preload_cuda_dylibs(cuda_lib_dir: String, cudnn_lib_dir: String) -> Result<(), String> {
    preload_cuda_dylibs_impl(
        path_opt_from_string(cuda_lib_dir).as_deref(),
        path_opt_from_string(cudnn_lib_dir).as_deref(),
    )
    .map_err(error_to_string)
}

fn swift_transcribe_file(
    audio_file: String,
    options: SwiftTranscriptionOptions,
) -> Result<String, String> {
    let config = config_from_options(options)?;
    transcribe_audio_file(audio_file, &config)
        .map(|result| result.transcript)
        .map_err(error_to_string)
}

fn swift_transcribe_bytes(
    audio_bytes: Vec<u8>,
    format_hint: String,
    options: SwiftTranscriptionOptions,
) -> Result<String, String> {
    let config = config_from_options(options)?;
    transcribe_audio_bytes(
        audio_bytes,
        str_opt_from_string(format_hint).as_deref(),
        &config,
    )
    .map(|result| result.transcript)
    .map_err(error_to_string)
}

fn swift_transcriber_new(options: SwiftTranscriptionOptions) -> Result<SwiftTranscriber, String> {
    let config = config_from_options(options)?;
    RustTranscriber::new(config)
        .map(|inner| SwiftTranscriber { inner })
        .map_err(error_to_string)
}

impl SwiftTranscriber {
    fn transcribe_file(&mut self, audio_file: String) -> Result<String, String> {
        self.inner
            .transcribe_audio_file(audio_file)
            .map(|result| result.transcript)
            .map_err(error_to_string)
    }

    fn transcribe_bytes(
        &mut self,
        audio_bytes: Vec<u8>,
        format_hint: String,
    ) -> Result<String, String> {
        self.inner
            .transcribe_audio_bytes(audio_bytes, str_opt_from_string(format_hint).as_deref())
            .map(|result| result.transcript)
            .map_err(error_to_string)
    }
}

fn config_from_options(options: SwiftTranscriptionOptions) -> Result<TranscriptionConfig, String> {
    let beam_width = non_zero_u32(options.beam_width, 32) as usize;
    let normalize_spaces = options.normalize_spaces;
    let drop_empty_candidates = options.drop_empty_candidates;

    Ok(TranscriptionConfig {
        runtime: RuntimeConfig {
            ort_dylib_path: path_opt_from_string(options.ort_dylib_path),
        },
        audio: AudioDecodeConfig {
            fallback_sample_rate: non_zero_u32(options.fallback_sample_rate, 16_000),
            skip_decode_errors: options.skip_decode_errors,
            ffmpeg_fallback: true,
        },
        encoder: EncoderConfig {
            w2v_bert: W2vBertEncoderConfig {
                model_source: str_opt_from_string(options.w2v_model_source),
                sample_rate: non_zero_opt_u32(options.w2v_sample_rate),
                feature_size: non_zero_opt_u32(options.w2v_feature_size)
                    .map(|value| value as usize),
                stride: non_zero_opt_u32(options.w2v_stride).map(|value| value as usize),
                feature_dim: non_zero_opt_u32(options.w2v_feature_dim).map(|value| value as usize),
                padding_value: if options.w2v_padding_value.is_nan() {
                    None
                } else {
                    Some(options.w2v_padding_value)
                },
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
                blank_id: options.blank_id,
                beam_width,
                n_best: non_zero_opt_u32(options.n_best)
                    .map(|value| value as usize)
                    .unwrap_or(beam_width),
            },
            text: TextDecoderConfig {
                tokenizer_path: path_or_default(options.tokenizer, "tokenizer.model"),
                normalize_spaces,
                drop_empty_candidates,
            },
            language_model: if options.use_language_model {
                Some(LmConfig {
                    path: path_or_default(options.lm, "lm.binary"),
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
                    log_language_model: options.log_language_model,
                    bos: options.lm_bos,
                    eos: options.lm_eos,
                    candidate_processing: CandidateProcessingConfig {
                        normalize_spaces,
                        drop_empty_candidates,
                    },
                })
            } else {
                None
            },
        },
    })
}

fn parse_optimization_level(value: &str) -> Result<ModelOptimizationLevel, String> {
    match value {
        "" | "disable" => Ok(ModelOptimizationLevel::Disable),
        "level1" => Ok(ModelOptimizationLevel::Level1),
        "level2" => Ok(ModelOptimizationLevel::Level2),
        "level3" => Ok(ModelOptimizationLevel::Level3),
        "all" => Ok(ModelOptimizationLevel::All),
        other => Err(format!(
            "invalid ort_optimization {other:?}; expected disable, level1, level2, level3, or all"
        )),
    }
}

fn path_or_default(value: String, default: &str) -> PathBuf {
    path_opt_from_string(value).unwrap_or_else(|| PathBuf::from(default))
}

fn path_opt_from_string(value: String) -> Option<PathBuf> {
    str_opt_from_string(value).map(PathBuf::from)
}

fn str_opt_from_string(value: String) -> Option<String> {
    if value.is_empty() { None } else { Some(value) }
}

fn non_zero_u32(value: u32, default: u32) -> u32 {
    if value == 0 { default } else { value }
}

fn non_zero_opt_u32(value: u32) -> Option<u32> {
    if value == 0 { None } else { Some(value) }
}

fn error_to_string(error: anyhow::Error) -> String {
    error.to_string()
}
