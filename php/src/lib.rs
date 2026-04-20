#![cfg_attr(windows, feature(abi_vectorcall))]

use std::path::PathBuf;
use std::time::Duration;

use ext_php_rs::binary_slice::BinarySlice;
use ext_php_rs::boxed::ZBox;
use ext_php_rs::prelude::*;
use ext_php_rs::types::{ZendHashTable, Zval};

use w2v_bert_uk::{
    AcousticModelConfig, CandidateProcessingConfig, CtcDecoderConfig, DecoderConfig, EncoderConfig,
    LmConfig, RuntimeConfig, ScoredCandidate, TextDecoderConfig, TimingReport,
    Transcriber as RustTranscriber, TranscriptionConfig, TranscriptionResult, W2vBertEncoderConfig,
    audio::AudioDecodeConfig,
    init_ort,
    model::{ModelConfig, ModelOptimizationLevel},
    preload_cuda_dylibs as preload_cuda_dylibs_impl, transcribe_audio_bytes, transcribe_audio_file,
};

#[php_class]
#[php(name = "W2vBertUk\\Transcriber")]
pub struct Transcriber {
    inner: RustTranscriber,
}

#[php_impl]
impl Transcriber {
    pub fn __construct(options: Option<&ZendHashTable>) -> PhpResult<Self> {
        let config = build_config(options)?;
        let inner = RustTranscriber::new(config)?;
        Ok(Self { inner })
    }

    pub fn transcribe_file(&mut self, audio_file: String) -> PhpResult<String> {
        Ok(self
            .inner
            .transcribe_audio_file(PathBuf::from(audio_file))?
            .transcript)
    }

    pub fn transcribe_file_with_report(
        &mut self,
        audio_file: String,
    ) -> PhpResult<ZBox<ZendHashTable>> {
        let result = self
            .inner
            .transcribe_audio_file(PathBuf::from(audio_file))?;
        transcription_result_to_php(result)
    }

    pub fn transcribe_bytes(
        &mut self,
        audio_bytes: BinarySlice<u8>,
        format_hint: Option<String>,
    ) -> PhpResult<String> {
        Ok(self
            .inner
            .transcribe_audio_bytes(
                audio_bytes.iter().copied().collect(),
                format_hint.as_deref(),
            )?
            .transcript)
    }

    pub fn transcribe_bytes_with_report(
        &mut self,
        audio_bytes: BinarySlice<u8>,
        format_hint: Option<String>,
    ) -> PhpResult<ZBox<ZendHashTable>> {
        let result = self.inner.transcribe_audio_bytes(
            audio_bytes.iter().copied().collect(),
            format_hint.as_deref(),
        )?;
        transcription_result_to_php(result)
    }
}

#[php_function]
#[php(name = "w2v_bert_uk_initialize_ort")]
pub fn initialize_ort(ort_dylib_path: Option<String>) -> PhpResult<bool> {
    Ok(init_ort(ort_dylib_path.map(PathBuf::from).as_deref())?)
}

#[php_function]
#[php(name = "w2v_bert_uk_preload_cuda_dylibs")]
pub fn preload_cuda_dylibs(
    cuda_lib_dir: Option<String>,
    cudnn_lib_dir: Option<String>,
) -> PhpResult<()> {
    Ok(preload_cuda_dylibs_impl(
        cuda_lib_dir.map(PathBuf::from).as_deref(),
        cudnn_lib_dir.map(PathBuf::from).as_deref(),
    )?)
}

#[php_function]
#[php(name = "w2v_bert_uk_transcribe_file")]
pub fn transcribe_file(audio_file: String, options: Option<&ZendHashTable>) -> PhpResult<String> {
    let config = build_config(options)?;
    Ok(transcribe_audio_file(PathBuf::from(audio_file), &config)?.transcript)
}

#[php_function]
#[php(name = "w2v_bert_uk_transcribe_file_with_report")]
pub fn transcribe_file_with_report(
    audio_file: String,
    options: Option<&ZendHashTable>,
) -> PhpResult<ZBox<ZendHashTable>> {
    let config = build_config(options)?;
    let result = transcribe_audio_file(PathBuf::from(audio_file), &config)?;
    transcription_result_to_php(result)
}

#[php_function]
#[php(name = "w2v_bert_uk_transcribe_bytes")]
pub fn transcribe_bytes(
    audio_bytes: BinarySlice<u8>,
    format_hint: Option<String>,
    options: Option<&ZendHashTable>,
) -> PhpResult<String> {
    let config = build_config(options)?;
    Ok(transcribe_audio_bytes(
        audio_bytes.iter().copied().collect(),
        format_hint.as_deref(),
        &config,
    )?
    .transcript)
}

#[php_function]
#[php(name = "w2v_bert_uk_transcribe_bytes_with_report")]
pub fn transcribe_bytes_with_report(
    audio_bytes: BinarySlice<u8>,
    format_hint: Option<String>,
    options: Option<&ZendHashTable>,
) -> PhpResult<ZBox<ZendHashTable>> {
    let config = build_config(options)?;
    let result = transcribe_audio_bytes(
        audio_bytes.iter().copied().collect(),
        format_hint.as_deref(),
        &config,
    )?;
    transcription_result_to_php(result)
}

fn build_config(options: Option<&ZendHashTable>) -> PhpResult<TranscriptionConfig> {
    let beam_width = option_usize(options, "beam_width")?.unwrap_or(32);
    let normalize_spaces = option_bool(options, "normalize_spaces")?.unwrap_or(true);
    let drop_empty_candidates = option_bool(options, "drop_empty_candidates")?.unwrap_or(true);
    let language_model = option_path(options, "lm")?
        .map(|path| {
            Ok::<_, ext_php_rs::exception::PhpException>(LmConfig {
                path,
                weight: option_f32(options, "lm_weight")?.unwrap_or(0.45),
                word_bonus: option_f32(options, "word_bonus")?.unwrap_or(0.2),
                log_language_model: option_bool(options, "log_language_model")?.unwrap_or(true),
                bos: option_bool(options, "lm_bos")?.unwrap_or(true),
                eos: option_bool(options, "lm_eos")?.unwrap_or(true),
                candidate_processing: CandidateProcessingConfig {
                    normalize_spaces,
                    drop_empty_candidates,
                },
            })
        })
        .transpose()?;
    let candidate_processing = CandidateProcessingConfig {
        normalize_spaces,
        drop_empty_candidates,
    };

    Ok(TranscriptionConfig {
        runtime: RuntimeConfig {
            ort_dylib_path: option_path(options, "ort_dylib_path")?,
        },
        audio: AudioDecodeConfig {
            fallback_sample_rate: option_u32(options, "fallback_sample_rate")?.unwrap_or(16_000),
            skip_decode_errors: option_bool(options, "skip_decode_errors")?.unwrap_or(true),
        },
        encoder: EncoderConfig {
            w2v_bert: W2vBertEncoderConfig {
                model_source: option_string(options, "w2v_model_source")?,
                sample_rate: option_u32(options, "w2v_sample_rate")?,
                feature_size: option_usize(options, "w2v_feature_size")?,
                stride: option_usize(options, "w2v_stride")?,
                feature_dim: option_usize(options, "w2v_feature_dim")?,
                padding_value: option_f32(options, "w2v_padding_value")?,
            },
        },
        model: AcousticModelConfig {
            path: option_path(options, "model")?
                .unwrap_or_else(|| PathBuf::from("model_optimized.onnx")),
            session: ModelConfig {
                optimization_level: parse_optimization_level(
                    option_string(options, "ort_optimization")?
                        .as_deref()
                        .unwrap_or("disable"),
                )?,
                log_accelerator: option_bool(options, "log_accelerator")?.unwrap_or(true),
            },
        },
        decoder: DecoderConfig {
            ctc: CtcDecoderConfig {
                blank_id: option_u32(options, "blank_id")?.unwrap_or(0),
                beam_width,
                n_best: option_usize(options, "n_best")?.unwrap_or(beam_width),
            },
            text: TextDecoderConfig {
                tokenizer_path: option_path(options, "tokenizer")?
                    .unwrap_or_else(|| PathBuf::from("tokenizer.model")),
                normalize_spaces,
                drop_empty_candidates,
            },
            language_model,
        },
    })
}

#[php_module]
pub fn get_module(module: ModuleBuilder) -> ModuleBuilder {
    module
        .class::<Transcriber>()
        .function(wrap_function!(initialize_ort))
        .function(wrap_function!(preload_cuda_dylibs))
        .function(wrap_function!(transcribe_file))
        .function(wrap_function!(transcribe_file_with_report))
        .function(wrap_function!(transcribe_bytes))
        .function(wrap_function!(transcribe_bytes_with_report))
}

fn transcription_result_to_php(result: TranscriptionResult) -> PhpResult<ZBox<ZendHashTable>> {
    let mut hash = ZendHashTable::new();
    hash.insert("transcript", result.transcript)?;
    hash.insert("timings", timings_to_php(&result.timings)?)?;
    hash.insert("candidates", candidates_to_php(&result.candidates)?)?;
    Ok(hash)
}

fn timings_to_php(timings: &TimingReport) -> PhpResult<ZBox<ZendHashTable>> {
    let mut hash = ZendHashTable::new();
    hash.insert("audio_duration_seconds", timings.audio_duration_seconds)?;
    hash.insert("feature_rows", timings.feature_rows as i64)?;
    hash.insert("feature_cols", timings.feature_cols as i64)?;
    hash.insert("feature_count", timings.feature_count as i64)?;
    hash.insert(
        "audio_decode_seconds",
        duration_seconds(timings.audio_decode_elapsed),
    )?;
    hash.insert("feature_seconds", duration_seconds(timings.feature_elapsed))?;
    hash.insert(
        "model_session_seconds",
        duration_seconds(timings.model.session_elapsed),
    )?;
    hash.insert(
        "model_input_seconds",
        duration_seconds(timings.model.input_elapsed),
    )?;
    hash.insert(
        "model_inference_seconds",
        duration_seconds(timings.model.inference_elapsed),
    )?;
    hash.insert(
        "model_ctc_seconds",
        duration_seconds(timings.model.ctc_elapsed),
    )?;
    hash.insert(
        "tokenizer_load_seconds",
        duration_seconds(timings.tokenizer_load_elapsed),
    )?;
    hash.insert(
        "text_decode_seconds",
        duration_seconds(timings.text_decode_elapsed),
    )?;
    hash.insert("lm_seconds", duration_seconds(timings.lm_elapsed))?;
    hash.insert(
        "measured_seconds",
        duration_seconds(timings.measured_elapsed()),
    )?;

    match &timings.best_candidate {
        Some(candidate) => hash.insert("best_candidate", candidate_to_php(candidate)?)?,
        None => hash.insert("best_candidate", Zval::new())?,
    }

    Ok(hash)
}

fn candidates_to_php(candidates: &[ScoredCandidate]) -> PhpResult<ZBox<ZendHashTable>> {
    let mut array = ZendHashTable::new();
    for candidate in candidates {
        array.push(candidate_to_php(candidate)?)?;
    }
    Ok(array)
}

fn candidate_to_php(candidate: &ScoredCandidate) -> PhpResult<ZBox<ZendHashTable>> {
    let mut hash = ZendHashTable::new();
    hash.insert("text", candidate.text.clone())?;
    hash.insert("ctc_log_prob", candidate.ctc_log_prob as f64)?;
    hash.insert("lm_log_prob", candidate.lm_log_prob as f64)?;
    hash.insert("word_count", candidate.word_count as i64)?;
    hash.insert("total_score", candidate.total_score as f64)?;
    Ok(hash)
}

fn option_value<'a>(options: Option<&'a ZendHashTable>, key: &str) -> Option<&'a Zval> {
    options
        .and_then(|hash| hash.get(key))
        .filter(|value| !value.is_null())
}

fn option_string(options: Option<&ZendHashTable>, key: &str) -> PhpResult<Option<String>> {
    option_value(options, key)
        .map(|value| {
            value
                .coerce_to_string()
                .ok_or_else(|| format!("option {key:?} must be a string").into())
        })
        .transpose()
}

fn option_path(options: Option<&ZendHashTable>, key: &str) -> PhpResult<Option<PathBuf>> {
    Ok(option_string(options, key)?.map(PathBuf::from))
}

fn option_bool(options: Option<&ZendHashTable>, key: &str) -> PhpResult<Option<bool>> {
    Ok(option_value(options, key).map(Zval::coerce_to_bool))
}

fn option_u32(options: Option<&ZendHashTable>, key: &str) -> PhpResult<Option<u32>> {
    option_long(options, key)?
        .map(|value| {
            u32::try_from(value).map_err(|_| format!("option {key:?} must fit in u32").into())
        })
        .transpose()
}

fn option_usize(options: Option<&ZendHashTable>, key: &str) -> PhpResult<Option<usize>> {
    option_long(options, key)?
        .map(|value| {
            usize::try_from(value).map_err(|_| format!("option {key:?} must fit in usize").into())
        })
        .transpose()
}

fn option_long(options: Option<&ZendHashTable>, key: &str) -> PhpResult<Option<i64>> {
    option_value(options, key)
        .map(|value| {
            value
                .coerce_to_long()
                .ok_or_else(|| format!("option {key:?} must be an integer").into())
        })
        .transpose()
}

fn option_f32(options: Option<&ZendHashTable>, key: &str) -> PhpResult<Option<f32>> {
    option_value(options, key)
        .map(|value| {
            value
                .coerce_to_double()
                .map(|value| value as f32)
                .ok_or_else(|| format!("option {key:?} must be a number").into())
        })
        .transpose()
}

fn duration_seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn parse_optimization_level(value: &str) -> PhpResult<ModelOptimizationLevel> {
    match value {
        "disable" => Ok(ModelOptimizationLevel::Disable),
        "level1" => Ok(ModelOptimizationLevel::Level1),
        "level2" => Ok(ModelOptimizationLevel::Level2),
        "level3" => Ok(ModelOptimizationLevel::Level3),
        "all" => Ok(ModelOptimizationLevel::All),
        other => Err(format!(
            "invalid ort_optimization {other:?}; expected disable, level1, level2, level3, or all"
        )
        .into()),
    }
}
