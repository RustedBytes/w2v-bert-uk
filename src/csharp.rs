use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_uchar};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::ptr;
use std::slice;

use crate::{
    AcousticModelConfig, CandidateProcessingConfig, CtcDecoderConfig, DecoderConfig, EncoderConfig,
    LmConfig, RuntimeConfig, TextDecoderConfig, Transcriber as RustTranscriber,
    TranscriptionConfig, W2vBertEncoderConfig,
    audio::AudioDecodeConfig,
    init_ort,
    model::{ModelConfig, ModelOptimizationLevel},
    preload_cuda_dylibs as preload_cuda_dylibs_impl, transcribe_audio_bytes, transcribe_audio_file,
};

pub const W2V_BERT_UK_OK: i32 = 0;
pub const W2V_BERT_UK_ERROR: i32 = -1;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

#[repr(C)]
pub struct W2vBertUkOptions {
    pub model: *const c_char,
    pub tokenizer: *const c_char,
    pub lm: *const c_char,
    pub ort_dylib_path: *const c_char,
    pub ort_optimization: *const c_char,
    pub w2v_model_source: *const c_char,
    pub hot_words: *const *const c_char,
    pub beam_width: u32,
    pub lm_weight: f32,
    pub word_bonus: f32,
    pub hot_words_len: usize,
    pub hot_word_bonus: f32,
    pub fallback_sample_rate: u32,
    pub w2v_sample_rate: u32,
    pub w2v_feature_size: u32,
    pub w2v_stride: u32,
    pub w2v_feature_dim: u32,
    pub w2v_padding_value: f32,
    pub blank_id: u32,
    pub n_best: u32,
    pub log_language_model: i32,
    pub log_accelerator: i32,
    pub skip_decode_errors: i32,
    pub normalize_spaces: i32,
    pub drop_empty_candidates: i32,
    pub lm_bos: i32,
    pub lm_eos: i32,
}

#[repr(C)]
pub struct W2vBertUkTranscriber {
    _private: [u8; 0],
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn w2v_bert_uk_options_default() -> W2vBertUkOptions {
    W2vBertUkOptions {
        model: ptr::null(),
        tokenizer: ptr::null(),
        lm: ptr::null(),
        ort_dylib_path: ptr::null(),
        ort_optimization: ptr::null(),
        w2v_model_source: ptr::null(),
        hot_words: ptr::null(),
        beam_width: 32,
        lm_weight: 0.45,
        word_bonus: 0.2,
        hot_words_len: 0,
        hot_word_bonus: 0.0,
        fallback_sample_rate: 16_000,
        w2v_sample_rate: 0,
        w2v_feature_size: 0,
        w2v_stride: 0,
        w2v_feature_dim: 0,
        w2v_padding_value: f32::NAN,
        blank_id: 0,
        n_best: 0,
        log_language_model: -1,
        log_accelerator: -1,
        skip_decode_errors: -1,
        normalize_spaces: -1,
        drop_empty_candidates: -1,
        lm_bos: -1,
        lm_eos: -1,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn w2v_bert_uk_initialize_ort(
    ort_dylib_path: *const c_char,
    initialized: *mut bool,
) -> i32 {
    ffi_status(|| {
        let path = path_opt_from_ptr(ort_dylib_path)?;
        let committed = init_ort(path.as_deref())?;
        write_out(initialized, committed)?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn w2v_bert_uk_preload_cuda_dylibs(
    cuda_lib_dir: *const c_char,
    cudnn_lib_dir: *const c_char,
) -> i32 {
    ffi_status(|| {
        let cuda = path_opt_from_ptr(cuda_lib_dir)?;
        let cudnn = path_opt_from_ptr(cudnn_lib_dir)?;
        preload_cuda_dylibs_impl(cuda.as_deref(), cudnn.as_deref())?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn w2v_bert_uk_transcribe_file(
    audio_file: *const c_char,
    options: *const W2vBertUkOptions,
    transcript: *mut *mut c_char,
) -> i32 {
    ffi_status(|| {
        let audio_file = path_from_ptr(audio_file)?;
        let config = config_from_options(options)?;
        let result = transcribe_audio_file(audio_file, &config)?;
        write_c_string(transcript, result.transcript)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn w2v_bert_uk_transcribe_bytes(
    audio_bytes: *const c_uchar,
    audio_bytes_len: usize,
    format_hint: *const c_char,
    options: *const W2vBertUkOptions,
    transcript: *mut *mut c_char,
) -> i32 {
    ffi_status(|| {
        if audio_bytes.is_null() && audio_bytes_len != 0 {
            anyhow::bail!("audio_bytes is null but audio_bytes_len is {audio_bytes_len}");
        }

        let bytes = if audio_bytes_len == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(audio_bytes, audio_bytes_len) }.to_vec()
        };
        let format_hint = str_opt_from_ptr(format_hint)?;
        let config = config_from_options(options)?;
        let result = transcribe_audio_bytes(bytes, format_hint.as_deref(), &config)?;
        write_c_string(transcript, result.transcript)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn w2v_bert_uk_transcriber_new(
    options: *const W2vBertUkOptions,
    transcriber: *mut *mut W2vBertUkTranscriber,
) -> i32 {
    ffi_status(|| {
        let config = config_from_options(options)?;
        let inner = RustTranscriber::new(config)?;
        let handle = Box::into_raw(Box::new(inner)).cast::<W2vBertUkTranscriber>();
        write_out(transcriber, handle)?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn w2v_bert_uk_transcriber_free(transcriber: *mut W2vBertUkTranscriber) {
    if !transcriber.is_null() {
        drop(unsafe { Box::from_raw(transcriber.cast::<RustTranscriber>()) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn w2v_bert_uk_transcriber_transcribe_file(
    transcriber: *mut W2vBertUkTranscriber,
    audio_file: *const c_char,
    transcript: *mut *mut c_char,
) -> i32 {
    ffi_status(|| {
        let transcriber = unsafe { transcriber.cast::<RustTranscriber>().as_mut() }
            .ok_or_else(|| anyhow::anyhow!("transcriber is null"))?;
        let audio_file = path_from_ptr(audio_file)?;
        let result = transcriber.transcribe_audio_file(audio_file)?;
        write_c_string(transcript, result.transcript)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn w2v_bert_uk_transcriber_transcribe_bytes(
    transcriber: *mut W2vBertUkTranscriber,
    audio_bytes: *const c_uchar,
    audio_bytes_len: usize,
    format_hint: *const c_char,
    transcript: *mut *mut c_char,
) -> i32 {
    ffi_status(|| {
        let transcriber = unsafe { transcriber.cast::<RustTranscriber>().as_mut() }
            .ok_or_else(|| anyhow::anyhow!("transcriber is null"))?;
        if audio_bytes.is_null() && audio_bytes_len != 0 {
            anyhow::bail!("audio_bytes is null but audio_bytes_len is {audio_bytes_len}");
        }

        let bytes = if audio_bytes_len == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(audio_bytes, audio_bytes_len) }.to_vec()
        };
        let format_hint = str_opt_from_ptr(format_hint)?;
        let result = transcriber.transcribe_audio_bytes(bytes, format_hint.as_deref())?;
        write_c_string(transcript, result.transcript)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn w2v_bert_uk_last_error_message() -> *mut c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(message) => message.as_c_str().to_owned().into_raw(),
        None => ptr::null_mut(),
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn w2v_bert_uk_string_free(value: *mut c_char) {
    if !value.is_null() {
        drop(unsafe { CString::from_raw(value) });
    }
}

fn ffi_status(operation: impl FnOnce() -> anyhow::Result<()>) -> i32 {
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => W2V_BERT_UK_OK,
        Ok(Err(error)) => {
            set_last_error(error.to_string());
            W2V_BERT_UK_ERROR
        }
        Err(_) => {
            set_last_error("native panic while executing w2v-bert-uk FFI call");
            W2V_BERT_UK_ERROR
        }
    }
}

fn config_from_options(options: *const W2vBertUkOptions) -> anyhow::Result<TranscriptionConfig> {
    let defaults = unsafe { w2v_bert_uk_options_default() };
    let options = if options.is_null() {
        &defaults
    } else {
        unsafe { &*options }
    };

    let normalize_spaces = bool_or_default(options.normalize_spaces, true);
    let drop_empty_candidates = bool_or_default(options.drop_empty_candidates, true);
    let candidate_processing = CandidateProcessingConfig {
        normalize_spaces,
        drop_empty_candidates,
    };
    let beam_width = non_zero_u32(options.beam_width, 32) as usize;
    let hot_words = strings_from_ptrs(options.hot_words, options.hot_words_len)?;

    Ok(TranscriptionConfig {
        runtime: RuntimeConfig {
            ort_dylib_path: path_opt_from_ptr(options.ort_dylib_path)?,
        },
        audio: AudioDecodeConfig {
            fallback_sample_rate: non_zero_u32(options.fallback_sample_rate, 16_000),
            skip_decode_errors: bool_or_default(options.skip_decode_errors, true),
            ffmpeg_fallback: true,
        },
        encoder: EncoderConfig {
            w2v_bert: W2vBertEncoderConfig {
                model_source: str_opt_from_ptr(options.w2v_model_source)?,
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
            path: path_opt_from_ptr(options.model)?
                .unwrap_or_else(|| PathBuf::from("model_optimized.onnx")),
            session: ModelConfig {
                optimization_level: parse_optimization_level(
                    str_opt_from_ptr(options.ort_optimization)?
                        .as_deref()
                        .unwrap_or("disable"),
                )?,
                log_accelerator: bool_or_default(options.log_accelerator, true),
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
                tokenizer_path: path_opt_from_ptr(options.tokenizer)?
                    .unwrap_or_else(|| PathBuf::from("tokenizer.model")),
                normalize_spaces,
                drop_empty_candidates,
            },
            language_model: path_opt_from_ptr(options.lm)?.map(|path| LmConfig {
                path,
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
                hot_words,
                hot_word_bonus: options.hot_word_bonus,
                log_language_model: bool_or_default(options.log_language_model, true),
                bos: bool_or_default(options.lm_bos, true),
                eos: bool_or_default(options.lm_eos, true),
                candidate_processing,
            }),
        },
    })
}

fn parse_optimization_level(value: &str) -> anyhow::Result<ModelOptimizationLevel> {
    match value {
        "disable" => Ok(ModelOptimizationLevel::Disable),
        "level1" => Ok(ModelOptimizationLevel::Level1),
        "level2" => Ok(ModelOptimizationLevel::Level2),
        "level3" => Ok(ModelOptimizationLevel::Level3),
        "all" => Ok(ModelOptimizationLevel::All),
        other => anyhow::bail!(
            "invalid ort_optimization {other:?}; expected disable, level1, level2, level3, or all"
        ),
    }
}

fn path_from_ptr(value: *const c_char) -> anyhow::Result<PathBuf> {
    str_from_ptr(value).map(PathBuf::from)
}

fn path_opt_from_ptr(value: *const c_char) -> anyhow::Result<Option<PathBuf>> {
    str_opt_from_ptr(value).map(|value| value.map(PathBuf::from))
}

fn str_from_ptr(value: *const c_char) -> anyhow::Result<String> {
    if value.is_null() {
        anyhow::bail!("required string pointer is null");
    }

    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map(str::to_owned)
        .map_err(Into::into)
}

fn str_opt_from_ptr(value: *const c_char) -> anyhow::Result<Option<String>> {
    if value.is_null() {
        Ok(None)
    } else {
        str_from_ptr(value).map(Some)
    }
}

fn strings_from_ptrs(ptr: *const *const c_char, len: usize) -> anyhow::Result<Vec<String>> {
    if ptr.is_null() || len == 0 {
        return Ok(Vec::new());
    }

    unsafe { slice::from_raw_parts(ptr, len) }
        .iter()
        .copied()
        .filter(|value| !value.is_null())
        .map(str_from_ptr)
        .collect()
}

fn write_c_string(out: *mut *mut c_char, value: String) -> anyhow::Result<()> {
    write_out(out, CString::new(value)?.into_raw())
}

fn write_out<T>(out: *mut T, value: T) -> anyhow::Result<()> {
    if out.is_null() {
        anyhow::bail!("output pointer is null");
    }

    unsafe {
        *out = value;
    }
    Ok(())
}

fn bool_or_default(value: i32, default: bool) -> bool {
    match value {
        -1 => default,
        0 => false,
        _ => true,
    }
}

fn non_zero_u32(value: u32, default: u32) -> u32 {
    if value == 0 { default } else { value }
}

fn non_zero_opt_u32(value: u32) -> Option<u32> {
    if value == 0 { None } else { Some(value) }
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn set_last_error(message: impl Into<String>) {
    let message = message.into().replace('\0', "\\0");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(message).ok();
    });
}
