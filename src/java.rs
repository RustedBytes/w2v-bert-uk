use std::path::PathBuf;

use jni::{
    Env, EnvUnowned,
    errors::ThrowRuntimeExAndDefault,
    objects::{JByteArray, JClass, JObjectArray, JString},
    refs::Reference,
    sys::{jboolean, jfloat, jint, jlong},
};

use crate::{
    AcousticModelConfig, CandidateProcessingConfig, CtcDecoderConfig, DecoderConfig, EncoderConfig,
    LmConfig, RuntimeConfig, TextDecoderConfig, Transcriber as RustTranscriber,
    TranscriptionConfig, W2vBertEncoderConfig,
    audio::AudioDecodeConfig,
    init_ort,
    model::{ModelConfig, ModelOptimizationLevel},
    preload_cuda_dylibs as preload_cuda_dylibs_impl, transcribe_audio_file,
};

type JavaResult<T> = Result<T, JavaBindingError>;

#[derive(Debug)]
struct JavaBindingError(String);

impl std::fmt::Display for JavaBindingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for JavaBindingError {}

impl From<anyhow::Error> for JavaBindingError {
    fn from(error: anyhow::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<jni::errors::Error> for JavaBindingError {
    fn from(error: jni::errors::Error) -> Self {
        Self(error.to_string())
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_rustedbytes_w2vbertuk_W2vBertUk_nativeInitializeOrt<
    'local,
>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    ort_dylib_path: JString<'local>,
) -> jboolean {
    env.with_env(|env| -> JavaResult<jboolean> {
        let path = path_opt_from_jstring(env, &ort_dylib_path)?;
        let initialized = init_ort(path.as_deref())?;
        Ok(bool_to_jboolean(initialized))
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_rustedbytes_w2vbertuk_W2vBertUk_nativePreloadCudaDylibs<
    'local,
>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    cuda_lib_dir: JString<'local>,
    cudnn_lib_dir: JString<'local>,
) {
    env.with_env(|env| -> JavaResult<()> {
        let cuda = path_opt_from_jstring(env, &cuda_lib_dir)?;
        let cudnn = path_opt_from_jstring(env, &cudnn_lib_dir)?;
        preload_cuda_dylibs_impl(cuda.as_deref(), cudnn.as_deref())?;
        Ok(())
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_rustedbytes_w2vbertuk_W2vBertUk_nativeTranscribeFile<
    'local,
>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    audio_file: JString<'local>,
    model: JString<'local>,
    tokenizer: JString<'local>,
    lm: JString<'local>,
    ort_dylib_path: JString<'local>,
    ort_optimization: JString<'local>,
    w2v_model_source: JString<'local>,
    beam_width: jint,
    lm_weight: jfloat,
    word_bonus: jfloat,
    hot_words: JObjectArray<'local, JString<'local>>,
    hot_word_bonus: jfloat,
    fallback_sample_rate: jint,
    w2v_sample_rate: jint,
    w2v_feature_size: jint,
    w2v_stride: jint,
    w2v_feature_dim: jint,
    w2v_padding_value: jfloat,
    blank_id: jint,
    n_best: jint,
    log_language_model: jboolean,
    log_accelerator: jboolean,
    skip_decode_errors: jboolean,
    normalize_spaces: jboolean,
    drop_empty_candidates: jboolean,
    lm_bos: jboolean,
    lm_eos: jboolean,
) -> JString<'local> {
    env.with_env(|env| -> JavaResult<JString<'local>> {
        let audio_file = path_from_jstring(env, &audio_file)?;
        let config = config_from_jni(
            env,
            &model,
            &tokenizer,
            &lm,
            &ort_dylib_path,
            &ort_optimization,
            &w2v_model_source,
            beam_width,
            lm_weight,
            word_bonus,
            &hot_words,
            hot_word_bonus,
            fallback_sample_rate,
            w2v_sample_rate,
            w2v_feature_size,
            w2v_stride,
            w2v_feature_dim,
            w2v_padding_value,
            blank_id,
            n_best,
            log_language_model,
            log_accelerator,
            skip_decode_errors,
            normalize_spaces,
            drop_empty_candidates,
            lm_bos,
            lm_eos,
        )?;
        let result = transcribe_audio_file(audio_file, &config)?;
        Ok(JString::from_str(env, result.transcript)?)
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_rustedbytes_w2vbertuk_W2vBertUk_nativeCreateTranscriber<
    'local,
>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    model: JString<'local>,
    tokenizer: JString<'local>,
    lm: JString<'local>,
    ort_dylib_path: JString<'local>,
    ort_optimization: JString<'local>,
    w2v_model_source: JString<'local>,
    beam_width: jint,
    lm_weight: jfloat,
    word_bonus: jfloat,
    hot_words: JObjectArray<'local, JString<'local>>,
    hot_word_bonus: jfloat,
    fallback_sample_rate: jint,
    w2v_sample_rate: jint,
    w2v_feature_size: jint,
    w2v_stride: jint,
    w2v_feature_dim: jint,
    w2v_padding_value: jfloat,
    blank_id: jint,
    n_best: jint,
    log_language_model: jboolean,
    log_accelerator: jboolean,
    skip_decode_errors: jboolean,
    normalize_spaces: jboolean,
    drop_empty_candidates: jboolean,
    lm_bos: jboolean,
    lm_eos: jboolean,
) -> jlong {
    env.with_env(|env| -> JavaResult<jlong> {
        let config = config_from_jni(
            env,
            &model,
            &tokenizer,
            &lm,
            &ort_dylib_path,
            &ort_optimization,
            &w2v_model_source,
            beam_width,
            lm_weight,
            word_bonus,
            &hot_words,
            hot_word_bonus,
            fallback_sample_rate,
            w2v_sample_rate,
            w2v_feature_size,
            w2v_stride,
            w2v_feature_dim,
            w2v_padding_value,
            blank_id,
            n_best,
            log_language_model,
            log_accelerator,
            skip_decode_errors,
            normalize_spaces,
            drop_empty_candidates,
            lm_bos,
            lm_eos,
        )?;
        let transcriber = RustTranscriber::new(config)?;
        Ok(Box::into_raw(Box::new(transcriber)) as jlong)
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_rustedbytes_w2vbertuk_W2vBertUk_nativeFreeTranscriber<
    'local,
>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) {
    env.with_env(|_env| -> JavaResult<()> {
        if handle != 0 {
            drop(unsafe { Box::from_raw(handle as *mut RustTranscriber) });
        }
        Ok(())
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_rustedbytes_w2vbertuk_W2vBertUk_nativeTranscriberTranscribeFile<
    'local,
>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    audio_file: JString<'local>,
) -> JString<'local> {
    env.with_env(|env| -> JavaResult<JString<'local>> {
        let transcriber = transcriber_from_handle(handle)?;
        let audio_file = path_from_jstring(env, &audio_file)?;
        let result = transcriber.transcribe_audio_file(audio_file)?;
        Ok(JString::from_str(env, result.transcript)?)
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_rustedbytes_w2vbertuk_W2vBertUk_nativeTranscriberTranscribeBytes<
    'local,
>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    audio_bytes: JByteArray<'local>,
    format_hint: JString<'local>,
) -> JString<'local> {
    env.with_env(|env| -> JavaResult<JString<'local>> {
        let transcriber = transcriber_from_handle(handle)?;
        let bytes = env.convert_byte_array(&audio_bytes)?;
        let format_hint = str_opt_from_jstring(env, &format_hint)?;
        let result = transcriber.transcribe_audio_bytes(bytes, format_hint.as_deref())?;
        Ok(JString::from_str(env, result.transcript)?)
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

fn config_from_jni(
    env: &mut Env<'_>,
    model: &JString<'_>,
    tokenizer: &JString<'_>,
    lm: &JString<'_>,
    ort_dylib_path: &JString<'_>,
    ort_optimization: &JString<'_>,
    w2v_model_source: &JString<'_>,
    beam_width: jint,
    lm_weight: jfloat,
    word_bonus: jfloat,
    hot_words: &JObjectArray<'_, JString<'_>>,
    hot_word_bonus: jfloat,
    fallback_sample_rate: jint,
    w2v_sample_rate: jint,
    w2v_feature_size: jint,
    w2v_stride: jint,
    w2v_feature_dim: jint,
    w2v_padding_value: jfloat,
    blank_id: jint,
    n_best: jint,
    log_language_model: jboolean,
    log_accelerator: jboolean,
    skip_decode_errors: jboolean,
    normalize_spaces: jboolean,
    drop_empty_candidates: jboolean,
    lm_bos: jboolean,
    lm_eos: jboolean,
) -> anyhow::Result<TranscriptionConfig> {
    let normalize_spaces = jboolean_to_bool(normalize_spaces);
    let drop_empty_candidates = jboolean_to_bool(drop_empty_candidates);
    let beam_width = non_zero_jint(beam_width, 32) as usize;
    let hot_words = strings_from_jarray(env, hot_words)?;

    Ok(TranscriptionConfig {
        runtime: RuntimeConfig {
            ort_dylib_path: path_opt_from_jstring(env, ort_dylib_path)?,
        },
        audio: AudioDecodeConfig {
            fallback_sample_rate: non_zero_jint(fallback_sample_rate, 16_000),
            skip_decode_errors: jboolean_to_bool(skip_decode_errors),
            ffmpeg_fallback: true,
        },
        encoder: EncoderConfig {
            w2v_bert: W2vBertEncoderConfig {
                model_source: str_opt_from_jstring(env, w2v_model_source)?,
                sample_rate: non_zero_opt_jint(w2v_sample_rate),
                feature_size: non_zero_opt_jint(w2v_feature_size).map(|value| value as usize),
                stride: non_zero_opt_jint(w2v_stride).map(|value| value as usize),
                feature_dim: non_zero_opt_jint(w2v_feature_dim).map(|value| value as usize),
                padding_value: if w2v_padding_value.is_nan() {
                    None
                } else {
                    Some(w2v_padding_value)
                },
            },
        },
        model: AcousticModelConfig {
            path: path_or_default(path_opt_from_jstring(env, model)?, "model_optimized.onnx"),
            session: ModelConfig {
                optimization_level: parse_optimization_level(
                    str_opt_from_jstring(env, ort_optimization)?
                        .as_deref()
                        .unwrap_or("disable"),
                )?,
                log_accelerator: jboolean_to_bool(log_accelerator),
            },
        },
        decoder: DecoderConfig {
            ctc: CtcDecoderConfig {
                blank_id: blank_id.max(0) as u32,
                beam_width,
                n_best: non_zero_opt_jint(n_best)
                    .map(|value| value as usize)
                    .unwrap_or(beam_width),
            },
            text: TextDecoderConfig {
                tokenizer_path: path_or_default(
                    path_opt_from_jstring(env, tokenizer)?,
                    "tokenizer.model",
                ),
                normalize_spaces,
                drop_empty_candidates,
            },
            language_model: path_opt_from_jstring(env, lm)?.map(|path| LmConfig {
                path,
                weight: if lm_weight == 0.0 { 0.45 } else { lm_weight },
                word_bonus: if word_bonus == 0.0 { 0.2 } else { word_bonus },
                hot_words,
                hot_word_bonus,
                log_language_model: jboolean_to_bool(log_language_model),
                bos: jboolean_to_bool(lm_bos),
                eos: jboolean_to_bool(lm_eos),
                candidate_processing: CandidateProcessingConfig {
                    normalize_spaces,
                    drop_empty_candidates,
                },
            }),
        },
    })
}

fn strings_from_jarray(
    env: &mut Env<'_>,
    values: &JObjectArray<'_, JString<'_>>,
) -> anyhow::Result<Vec<String>> {
    if values.is_null() {
        return Ok(Vec::new());
    }

    let len = values.len(env)?;
    let mut strings = Vec::with_capacity(len);
    for index in 0..len {
        let value = values.get_element(env, index)?;
        if !value.is_null() {
            strings.push(str_from_jstring(env, &value)?);
        }
    }
    Ok(strings)
}

fn transcriber_from_handle(handle: jlong) -> anyhow::Result<&'static mut RustTranscriber> {
    if handle == 0 {
        anyhow::bail!("transcriber handle is 0");
    }

    unsafe { (handle as *mut RustTranscriber).as_mut() }
        .ok_or_else(|| anyhow::anyhow!("transcriber handle is invalid"))
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

fn path_from_jstring(env: &Env<'_>, value: &JString<'_>) -> anyhow::Result<PathBuf> {
    Ok(PathBuf::from(str_from_jstring(env, value)?))
}

fn path_opt_from_jstring(env: &Env<'_>, value: &JString<'_>) -> anyhow::Result<Option<PathBuf>> {
    Ok(str_opt_from_jstring(env, value)?.map(PathBuf::from))
}

fn str_from_jstring(env: &Env<'_>, value: &JString<'_>) -> anyhow::Result<String> {
    if value.is_null() {
        anyhow::bail!("required Java string is null");
    }

    Ok(value.try_to_string(env)?)
}

fn str_opt_from_jstring(env: &Env<'_>, value: &JString<'_>) -> anyhow::Result<Option<String>> {
    if value.is_null() {
        Ok(None)
    } else {
        let value = value.try_to_string(env)?;
        Ok((!value.is_empty()).then_some(value))
    }
}

fn path_or_default(value: Option<PathBuf>, default: &str) -> PathBuf {
    value.unwrap_or_else(|| PathBuf::from(default))
}

fn non_zero_jint(value: jint, default: u32) -> u32 {
    if value <= 0 { default } else { value as u32 }
}

fn non_zero_opt_jint(value: jint) -> Option<u32> {
    if value <= 0 { None } else { Some(value as u32) }
}

fn jboolean_to_bool(value: jboolean) -> bool {
    value
}

fn bool_to_jboolean(value: bool) -> jboolean {
    value
}
