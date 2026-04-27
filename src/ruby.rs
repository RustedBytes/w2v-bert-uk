use std::cell::RefCell;
use std::path::PathBuf;
use std::time::Duration;

use magnus::{
    Error, RArray, RHash, RString, Ruby, TryConvert, Value, function, method, prelude::*,
};

use crate::{
    AcousticModelConfig, CandidateProcessingConfig, CtcDecoderConfig, DecoderConfig, EncoderConfig,
    LmConfig, RuntimeConfig, ScoredCandidate, TextDecoderConfig, TimingReport,
    Transcriber as RustTranscriber, TranscriptionConfig, TranscriptionResult, W2vBertEncoderConfig,
    audio::AudioDecodeConfig,
    init_ort,
    model::{ModelConfig, ModelOptimizationLevel},
    preload_cuda_dylibs as preload_cuda_dylibs_impl, transcribe_audio_bytes, transcribe_audio_file,
};

#[magnus::wrap(class = "W2vBertUk::Transcriber")]
struct Transcriber(RefCell<RustTranscriber>);

impl Transcriber {
    fn new(args: &[Value]) -> Result<Self, Error> {
        expect_arg_count(args, 0, 1, "Transcriber.new")?;
        let config = build_config(optional_hash(args, 0)?)?;
        let inner = RustTranscriber::new(config).map_err(to_ruby_error)?;
        Ok(Self(RefCell::new(inner)))
    }

    fn transcribe_file(&self, args: &[Value]) -> Result<String, Error> {
        expect_arg_count(args, 1, 1, "transcribe_file")?;
        let audio_file = PathBuf::try_convert(args[0])?;
        self.0
            .borrow_mut()
            .transcribe_audio_file(audio_file)
            .map(|result| result.transcript)
            .map_err(to_ruby_error)
    }

    fn transcribe_file_with_report(&self, args: &[Value]) -> Result<RHash, Error> {
        expect_arg_count(args, 1, 1, "transcribe_file_with_report")?;
        let audio_file = PathBuf::try_convert(args[0])?;
        let result = self
            .0
            .borrow_mut()
            .transcribe_audio_file(audio_file)
            .map_err(to_ruby_error)?;
        transcription_result_to_ruby(result)
    }

    fn transcribe_bytes(&self, args: &[Value]) -> Result<String, Error> {
        expect_arg_count(args, 1, 2, "transcribe_bytes")?;
        let audio_bytes = bytes_from_value(args[0])?;
        let format_hint: Option<String> = optional_arg(args, 1)?;
        self.0
            .borrow_mut()
            .transcribe_audio_bytes(audio_bytes, format_hint.as_deref())
            .map(|result| result.transcript)
            .map_err(to_ruby_error)
    }

    fn transcribe_bytes_with_report(&self, args: &[Value]) -> Result<RHash, Error> {
        expect_arg_count(args, 1, 2, "transcribe_bytes_with_report")?;
        let audio_bytes = bytes_from_value(args[0])?;
        let format_hint: Option<String> = optional_arg(args, 1)?;
        let result = self
            .0
            .borrow_mut()
            .transcribe_audio_bytes(audio_bytes, format_hint.as_deref())
            .map_err(to_ruby_error)?;
        transcription_result_to_ruby(result)
    }
}

fn initialize_ort(args: &[Value]) -> Result<bool, Error> {
    expect_arg_count(args, 0, 1, "initialize_ort")?;
    let ort_dylib_path: Option<PathBuf> = optional_arg(args, 0)?;
    init_ort(ort_dylib_path.as_deref()).map_err(to_ruby_error)
}

fn preload_cuda_dylibs(args: &[Value]) -> Result<(), Error> {
    expect_arg_count(args, 0, 2, "preload_cuda_dylibs")?;
    let cuda_lib_dir: Option<PathBuf> = optional_arg(args, 0)?;
    let cudnn_lib_dir: Option<PathBuf> = optional_arg(args, 1)?;
    preload_cuda_dylibs_impl(cuda_lib_dir.as_deref(), cudnn_lib_dir.as_deref())
        .map_err(to_ruby_error)
}

fn transcribe_file(args: &[Value]) -> Result<String, Error> {
    expect_arg_count(args, 1, 2, "transcribe_file")?;
    let audio_file = PathBuf::try_convert(args[0])?;
    let config = build_config(optional_hash(args, 1)?)?;
    transcribe_audio_file(audio_file, &config)
        .map(|result| result.transcript)
        .map_err(to_ruby_error)
}

fn transcribe_file_with_report(args: &[Value]) -> Result<RHash, Error> {
    expect_arg_count(args, 1, 2, "transcribe_file_with_report")?;
    let audio_file = PathBuf::try_convert(args[0])?;
    let config = build_config(optional_hash(args, 1)?)?;
    let result = transcribe_audio_file(audio_file, &config).map_err(to_ruby_error)?;
    transcription_result_to_ruby(result)
}

fn transcribe_bytes(args: &[Value]) -> Result<String, Error> {
    expect_arg_count(args, 1, 3, "transcribe_bytes")?;
    let audio_bytes = bytes_from_value(args[0])?;
    let format_hint = optional_string(args, 1)?;
    let config = build_config(optional_hash(args, 2)?)?;
    transcribe_audio_bytes(audio_bytes, format_hint.as_deref(), &config)
        .map(|result| result.transcript)
        .map_err(to_ruby_error)
}

fn transcribe_bytes_with_report(args: &[Value]) -> Result<RHash, Error> {
    expect_arg_count(args, 1, 3, "transcribe_bytes_with_report")?;
    let audio_bytes = bytes_from_value(args[0])?;
    let format_hint = optional_string(args, 1)?;
    let config = build_config(optional_hash(args, 2)?)?;
    let result = transcribe_audio_bytes(audio_bytes, format_hint.as_deref(), &config)
        .map_err(to_ruby_error)?;
    transcription_result_to_ruby(result)
}

fn build_config(options: Option<RHash>) -> Result<TranscriptionConfig, Error> {
    let beam_width = hash_get(options, "beam_width")?.unwrap_or(32usize);
    let normalize_spaces = hash_get(options, "normalize_spaces")?.unwrap_or(true);
    let drop_empty_candidates = hash_get(options, "drop_empty_candidates")?.unwrap_or(true);
    let candidate_processing = CandidateProcessingConfig {
        normalize_spaces,
        drop_empty_candidates,
    };
    let language_model = hash_get::<PathBuf>(options, "lm")?
        .map(|path| {
            Ok::<_, Error>(LmConfig {
                path,
                weight: hash_get(options, "lm_weight")?.unwrap_or(0.45),
                word_bonus: hash_get(options, "word_bonus")?.unwrap_or(0.2),
                log_language_model: hash_get(options, "log_language_model")?.unwrap_or(true),
                bos: hash_get(options, "lm_bos")?.unwrap_or(true),
                eos: hash_get(options, "lm_eos")?.unwrap_or(true),
                candidate_processing,
            })
        })
        .transpose()?;

    Ok(TranscriptionConfig {
        runtime: RuntimeConfig {
            ort_dylib_path: hash_get(options, "ort_dylib_path")?,
        },
        audio: AudioDecodeConfig {
            fallback_sample_rate: hash_get(options, "fallback_sample_rate")?.unwrap_or(16_000),
            skip_decode_errors: hash_get(options, "skip_decode_errors")?.unwrap_or(true),
            ffmpeg_fallback: true,
        },
        encoder: EncoderConfig {
            w2v_bert: W2vBertEncoderConfig {
                model_source: hash_get(options, "w2v_model_source")?,
                sample_rate: hash_get(options, "w2v_sample_rate")?,
                feature_size: hash_get(options, "w2v_feature_size")?,
                stride: hash_get(options, "w2v_stride")?,
                feature_dim: hash_get(options, "w2v_feature_dim")?,
                padding_value: hash_get(options, "w2v_padding_value")?,
            },
        },
        model: AcousticModelConfig {
            path: hash_get(options, "model")?
                .unwrap_or_else(|| PathBuf::from("model_optimized.onnx")),
            session: ModelConfig {
                optimization_level: parse_optimization_level(
                    hash_get::<String>(options, "ort_optimization")?
                        .as_deref()
                        .unwrap_or("disable"),
                )?,
                log_accelerator: hash_get(options, "log_accelerator")?.unwrap_or(true),
            },
        },
        decoder: DecoderConfig {
            ctc: CtcDecoderConfig {
                blank_id: hash_get(options, "blank_id")?.unwrap_or(0),
                beam_width,
                n_best: hash_get(options, "n_best")?.unwrap_or(beam_width),
            },
            text: TextDecoderConfig {
                tokenizer_path: hash_get(options, "tokenizer")?
                    .unwrap_or_else(|| PathBuf::from("tokenizer.model")),
                normalize_spaces,
                drop_empty_candidates,
            },
            language_model,
        },
    })
}

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let module = ruby.define_module("W2vBertUk")?;
    module.define_module_function("initialize_ort", function!(initialize_ort, -1))?;
    module.define_module_function("preload_cuda_dylibs", function!(preload_cuda_dylibs, -1))?;
    module.define_module_function("transcribe_file", function!(transcribe_file, -1))?;
    module.define_module_function("transcribe_bytes", function!(transcribe_bytes, -1))?;
    module.define_module_function(
        "transcribe_file_with_report",
        function!(transcribe_file_with_report, -1),
    )?;
    module.define_module_function(
        "transcribe_bytes_with_report",
        function!(transcribe_bytes_with_report, -1),
    )?;

    let class = module.define_class("Transcriber", ruby.class_object())?;
    class.define_singleton_method("new", function!(Transcriber::new, -1))?;
    class.define_method("transcribe_file", method!(Transcriber::transcribe_file, -1))?;
    class.define_method(
        "transcribe_file_with_report",
        method!(Transcriber::transcribe_file_with_report, -1),
    )?;
    class.define_method(
        "transcribe_bytes",
        method!(Transcriber::transcribe_bytes, -1),
    )?;
    class.define_method(
        "transcribe_bytes_with_report",
        method!(Transcriber::transcribe_bytes_with_report, -1),
    )?;
    Ok(())
}

fn transcription_result_to_ruby(result: TranscriptionResult) -> Result<RHash, Error> {
    let ruby = Ruby::get().map_err(|error| Error::new(runtime_error(), error.to_string()))?;
    let hash = ruby.hash_new();
    hash.aset("transcript", result.transcript)?;
    hash.aset("timings", timings_to_ruby(&ruby, &result.timings)?)?;
    hash.aset("candidates", candidates_to_ruby(&ruby, &result.candidates)?)?;
    Ok(hash)
}

fn timings_to_ruby(ruby: &Ruby, timings: &TimingReport) -> Result<RHash, Error> {
    let hash = ruby.hash_new();
    hash.aset("audio_duration_seconds", timings.audio_duration_seconds)?;
    hash.aset("feature_rows", timings.feature_rows)?;
    hash.aset("feature_cols", timings.feature_cols)?;
    hash.aset("feature_count", timings.feature_count)?;
    hash.aset(
        "audio_decode_seconds",
        duration_seconds(timings.audio_decode_elapsed),
    )?;
    hash.aset("feature_seconds", duration_seconds(timings.feature_elapsed))?;
    hash.aset(
        "model_session_seconds",
        duration_seconds(timings.model.session_elapsed),
    )?;
    hash.aset(
        "model_input_seconds",
        duration_seconds(timings.model.input_elapsed),
    )?;
    hash.aset(
        "model_inference_seconds",
        duration_seconds(timings.model.inference_elapsed),
    )?;
    hash.aset(
        "model_ctc_seconds",
        duration_seconds(timings.model.ctc_elapsed),
    )?;
    hash.aset(
        "tokenizer_load_seconds",
        duration_seconds(timings.tokenizer_load_elapsed),
    )?;
    hash.aset(
        "text_decode_seconds",
        duration_seconds(timings.text_decode_elapsed),
    )?;
    hash.aset("lm_seconds", duration_seconds(timings.lm_elapsed))?;
    hash.aset(
        "measured_seconds",
        duration_seconds(timings.measured_elapsed()),
    )?;
    hash.aset(
        "best_candidate",
        timings
            .best_candidate
            .as_ref()
            .map(|candidate| candidate_to_ruby(ruby, candidate))
            .transpose()?,
    )?;
    Ok(hash)
}

fn candidates_to_ruby(ruby: &Ruby, candidates: &[ScoredCandidate]) -> Result<RArray, Error> {
    let array = ruby.ary_new();
    for candidate in candidates {
        array.push(candidate_to_ruby(ruby, candidate)?)?;
    }
    Ok(array)
}

fn candidate_to_ruby(ruby: &Ruby, candidate: &ScoredCandidate) -> Result<RHash, Error> {
    let hash = ruby.hash_new();
    hash.aset("text", candidate.text.clone())?;
    hash.aset("ctc_log_prob", candidate.ctc_log_prob)?;
    hash.aset("lm_log_prob", candidate.lm_log_prob)?;
    hash.aset("word_count", candidate.word_count)?;
    hash.aset("total_score", candidate.total_score)?;
    Ok(hash)
}

fn hash_get<T>(options: Option<RHash>, key: &str) -> Result<Option<T>, Error>
where
    T: TryConvert,
{
    let Some(options) = options else {
        return Ok(None);
    };
    let ruby = Ruby::get_with(options);
    options
        .get(ruby.to_symbol(key))
        .or_else(|| options.get(key))
        .map(T::try_convert)
        .transpose()
}

fn optional_hash(args: &[Value], index: usize) -> Result<Option<RHash>, Error> {
    optional_value(args, index).map(|value| value.map(RHash::try_convert).transpose())?
}

fn optional_arg<T>(args: &[Value], index: usize) -> Result<Option<T>, Error>
where
    T: TryConvert,
{
    optional_value(args, index).map(|value| value.map(T::try_convert).transpose())?
}

fn optional_string(args: &[Value], index: usize) -> Result<Option<String>, Error> {
    optional_arg(args, index)
}

fn optional_value(args: &[Value], index: usize) -> Result<Option<Value>, Error> {
    Ok(args.get(index).copied().filter(|value| !value.is_nil()))
}

fn bytes_from_value(value: Value) -> Result<Vec<u8>, Error> {
    let string = RString::try_convert(value)?;
    Ok(unsafe { string.as_slice().to_vec() })
}

fn expect_arg_count(args: &[Value], min: usize, max: usize, name: &str) -> Result<(), Error> {
    if (min..=max).contains(&args.len()) {
        Ok(())
    } else {
        Err(Error::new(
            arg_error(),
            format!("{name} expects {min}..{max} arguments, got {}", args.len()),
        ))
    }
}

fn duration_seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn parse_optimization_level(value: &str) -> Result<ModelOptimizationLevel, Error> {
    match value {
        "disable" => Ok(ModelOptimizationLevel::Disable),
        "level1" => Ok(ModelOptimizationLevel::Level1),
        "level2" => Ok(ModelOptimizationLevel::Level2),
        "level3" => Ok(ModelOptimizationLevel::Level3),
        "all" => Ok(ModelOptimizationLevel::All),
        other => Err(Error::new(
            arg_error(),
            format!(
                "invalid ort_optimization {other:?}; expected disable, level1, level2, level3, or all"
            ),
        )),
    }
}

fn to_ruby_error(error: anyhow::Error) -> Error {
    Error::new(runtime_error(), error.to_string())
}

fn arg_error() -> magnus::exception::ExceptionClass {
    Ruby::get()
        .expect("Ruby VM is not available")
        .exception_arg_error()
}

fn runtime_error() -> magnus::exception::ExceptionClass {
    Ruby::get()
        .expect("Ruby VM is not available")
        .exception_runtime_error()
}
