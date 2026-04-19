use std::path::PathBuf;
use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList, PyListMethods};

use crate::{
    AcousticModelConfig, CandidateProcessingConfig, CtcDecoderConfig, DecoderConfig, EncoderConfig,
    LmConfig, RuntimeConfig, ScoredCandidate, TextDecoderConfig, TimingReport,
    Transcriber as RustTranscriber, TranscriptionConfig, TranscriptionResult, W2vBertEncoderConfig,
    audio::AudioDecodeConfig,
    init_ort, preload_cuda_dylibs as preload_cuda_dylibs_impl,
    model::{ModelConfig, ModelOptimizationLevel},
    transcribe_audio_bytes, transcribe_audio_file,
};

#[pyfunction]
#[pyo3(signature = (ort_dylib_path = None))]
fn initialize_ort(ort_dylib_path: Option<PathBuf>) -> PyResult<bool> {
    init_ort(ort_dylib_path.as_deref()).map_err(to_py_error)
}

#[pyfunction]
#[pyo3(signature = (cuda_lib_dir = None, cudnn_lib_dir = None))]
fn preload_cuda_dylibs(
    cuda_lib_dir: Option<PathBuf>,
    cudnn_lib_dir: Option<PathBuf>,
) -> PyResult<()> {
    preload_cuda_dylibs_impl(cuda_lib_dir.as_deref(), cudnn_lib_dir.as_deref()).map_err(to_py_error)
}

#[pyfunction]
#[pyo3(signature = (
    audio_file,
    model = PathBuf::from("model_optimized.onnx"),
    tokenizer = PathBuf::from("tokenizer.model"),
    beam_width = 32,
    lm = None,
    lm_weight = 0.45,
    word_bonus = 0.2,
    log_language_model = true,
    ort_dylib_path = None,
    ort_optimization = "disable",
    log_accelerator = true,
    fallback_sample_rate = 16000,
    skip_decode_errors = true,
    w2v_model_source = None,
    w2v_sample_rate = None,
    w2v_feature_size = None,
    w2v_stride = None,
    w2v_feature_dim = None,
    w2v_padding_value = None,
    blank_id = 0,
    n_best = None,
    normalize_spaces = true,
    drop_empty_candidates = true,
    lm_bos = true,
    lm_eos = true,
))]
fn transcribe_file(
    audio_file: PathBuf,
    model: PathBuf,
    tokenizer: PathBuf,
    beam_width: usize,
    lm: Option<PathBuf>,
    lm_weight: f32,
    word_bonus: f32,
    log_language_model: bool,
    ort_dylib_path: Option<PathBuf>,
    ort_optimization: &str,
    log_accelerator: bool,
    fallback_sample_rate: u32,
    skip_decode_errors: bool,
    w2v_model_source: Option<String>,
    w2v_sample_rate: Option<u32>,
    w2v_feature_size: Option<usize>,
    w2v_stride: Option<usize>,
    w2v_feature_dim: Option<usize>,
    w2v_padding_value: Option<f32>,
    blank_id: u32,
    n_best: Option<usize>,
    normalize_spaces: bool,
    drop_empty_candidates: bool,
    lm_bos: bool,
    lm_eos: bool,
) -> PyResult<String> {
    let config = build_config(
        model,
        tokenizer,
        beam_width,
        lm,
        lm_weight,
        word_bonus,
        log_language_model,
        ort_dylib_path,
        ort_optimization,
        log_accelerator,
        fallback_sample_rate,
        skip_decode_errors,
        w2v_model_source,
        w2v_sample_rate,
        w2v_feature_size,
        w2v_stride,
        w2v_feature_dim,
        w2v_padding_value,
        blank_id,
        n_best,
        normalize_spaces,
        drop_empty_candidates,
        lm_bos,
        lm_eos,
    )?;

    transcribe_audio_file(audio_file, &config)
        .map(|result| result.transcript)
        .map_err(to_py_error)
}

#[pyfunction]
#[pyo3(signature = (
    audio_bytes,
    format_hint = None,
    model = PathBuf::from("model_optimized.onnx"),
    tokenizer = PathBuf::from("tokenizer.model"),
    beam_width = 32,
    lm = None,
    lm_weight = 0.45,
    word_bonus = 0.2,
    log_language_model = true,
    ort_dylib_path = None,
    ort_optimization = "disable",
    log_accelerator = true,
    fallback_sample_rate = 16000,
    skip_decode_errors = true,
    w2v_model_source = None,
    w2v_sample_rate = None,
    w2v_feature_size = None,
    w2v_stride = None,
    w2v_feature_dim = None,
    w2v_padding_value = None,
    blank_id = 0,
    n_best = None,
    normalize_spaces = true,
    drop_empty_candidates = true,
    lm_bos = true,
    lm_eos = true,
))]
fn transcribe_bytes(
    audio_bytes: Vec<u8>,
    format_hint: Option<&str>,
    model: PathBuf,
    tokenizer: PathBuf,
    beam_width: usize,
    lm: Option<PathBuf>,
    lm_weight: f32,
    word_bonus: f32,
    log_language_model: bool,
    ort_dylib_path: Option<PathBuf>,
    ort_optimization: &str,
    log_accelerator: bool,
    fallback_sample_rate: u32,
    skip_decode_errors: bool,
    w2v_model_source: Option<String>,
    w2v_sample_rate: Option<u32>,
    w2v_feature_size: Option<usize>,
    w2v_stride: Option<usize>,
    w2v_feature_dim: Option<usize>,
    w2v_padding_value: Option<f32>,
    blank_id: u32,
    n_best: Option<usize>,
    normalize_spaces: bool,
    drop_empty_candidates: bool,
    lm_bos: bool,
    lm_eos: bool,
) -> PyResult<String> {
    let config = build_config(
        model,
        tokenizer,
        beam_width,
        lm,
        lm_weight,
        word_bonus,
        log_language_model,
        ort_dylib_path,
        ort_optimization,
        log_accelerator,
        fallback_sample_rate,
        skip_decode_errors,
        w2v_model_source,
        w2v_sample_rate,
        w2v_feature_size,
        w2v_stride,
        w2v_feature_dim,
        w2v_padding_value,
        blank_id,
        n_best,
        normalize_spaces,
        drop_empty_candidates,
        lm_bos,
        lm_eos,
    )?;

    transcribe_audio_bytes(audio_bytes, format_hint, &config)
        .map(|result| result.transcript)
        .map_err(to_py_error)
}

#[pyfunction]
#[pyo3(signature = (
    audio_file,
    model = PathBuf::from("model_optimized.onnx"),
    tokenizer = PathBuf::from("tokenizer.model"),
    beam_width = 32,
    lm = None,
    lm_weight = 0.45,
    word_bonus = 0.2,
    log_language_model = true,
    ort_dylib_path = None,
    ort_optimization = "disable",
    log_accelerator = true,
    fallback_sample_rate = 16000,
    skip_decode_errors = true,
    w2v_model_source = None,
    w2v_sample_rate = None,
    w2v_feature_size = None,
    w2v_stride = None,
    w2v_feature_dim = None,
    w2v_padding_value = None,
    blank_id = 0,
    n_best = None,
    normalize_spaces = true,
    drop_empty_candidates = true,
    lm_bos = true,
    lm_eos = true,
))]
fn transcribe_file_with_report<'py>(
    py: Python<'py>,
    audio_file: PathBuf,
    model: PathBuf,
    tokenizer: PathBuf,
    beam_width: usize,
    lm: Option<PathBuf>,
    lm_weight: f32,
    word_bonus: f32,
    log_language_model: bool,
    ort_dylib_path: Option<PathBuf>,
    ort_optimization: &str,
    log_accelerator: bool,
    fallback_sample_rate: u32,
    skip_decode_errors: bool,
    w2v_model_source: Option<String>,
    w2v_sample_rate: Option<u32>,
    w2v_feature_size: Option<usize>,
    w2v_stride: Option<usize>,
    w2v_feature_dim: Option<usize>,
    w2v_padding_value: Option<f32>,
    blank_id: u32,
    n_best: Option<usize>,
    normalize_spaces: bool,
    drop_empty_candidates: bool,
    lm_bos: bool,
    lm_eos: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let config = build_config(
        model,
        tokenizer,
        beam_width,
        lm,
        lm_weight,
        word_bonus,
        log_language_model,
        ort_dylib_path,
        ort_optimization,
        log_accelerator,
        fallback_sample_rate,
        skip_decode_errors,
        w2v_model_source,
        w2v_sample_rate,
        w2v_feature_size,
        w2v_stride,
        w2v_feature_dim,
        w2v_padding_value,
        blank_id,
        n_best,
        normalize_spaces,
        drop_empty_candidates,
        lm_bos,
        lm_eos,
    )?;

    let result = transcribe_audio_file(audio_file, &config).map_err(to_py_error)?;
    transcription_result_to_py(py, result)
}

#[pyfunction]
#[pyo3(signature = (
    audio_bytes,
    format_hint = None,
    model = PathBuf::from("model_optimized.onnx"),
    tokenizer = PathBuf::from("tokenizer.model"),
    beam_width = 32,
    lm = None,
    lm_weight = 0.45,
    word_bonus = 0.2,
    log_language_model = true,
    ort_dylib_path = None,
    ort_optimization = "disable",
    log_accelerator = true,
    fallback_sample_rate = 16000,
    skip_decode_errors = true,
    w2v_model_source = None,
    w2v_sample_rate = None,
    w2v_feature_size = None,
    w2v_stride = None,
    w2v_feature_dim = None,
    w2v_padding_value = None,
    blank_id = 0,
    n_best = None,
    normalize_spaces = true,
    drop_empty_candidates = true,
    lm_bos = true,
    lm_eos = true,
))]
fn transcribe_bytes_with_report<'py>(
    py: Python<'py>,
    audio_bytes: Vec<u8>,
    format_hint: Option<&str>,
    model: PathBuf,
    tokenizer: PathBuf,
    beam_width: usize,
    lm: Option<PathBuf>,
    lm_weight: f32,
    word_bonus: f32,
    log_language_model: bool,
    ort_dylib_path: Option<PathBuf>,
    ort_optimization: &str,
    log_accelerator: bool,
    fallback_sample_rate: u32,
    skip_decode_errors: bool,
    w2v_model_source: Option<String>,
    w2v_sample_rate: Option<u32>,
    w2v_feature_size: Option<usize>,
    w2v_stride: Option<usize>,
    w2v_feature_dim: Option<usize>,
    w2v_padding_value: Option<f32>,
    blank_id: u32,
    n_best: Option<usize>,
    normalize_spaces: bool,
    drop_empty_candidates: bool,
    lm_bos: bool,
    lm_eos: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let config = build_config(
        model,
        tokenizer,
        beam_width,
        lm,
        lm_weight,
        word_bonus,
        log_language_model,
        ort_dylib_path,
        ort_optimization,
        log_accelerator,
        fallback_sample_rate,
        skip_decode_errors,
        w2v_model_source,
        w2v_sample_rate,
        w2v_feature_size,
        w2v_stride,
        w2v_feature_dim,
        w2v_padding_value,
        blank_id,
        n_best,
        normalize_spaces,
        drop_empty_candidates,
        lm_bos,
        lm_eos,
    )?;

    let result = transcribe_audio_bytes(audio_bytes, format_hint, &config).map_err(to_py_error)?;
    transcription_result_to_py(py, result)
}

#[pyclass(unsendable)]
struct Transcriber {
    inner: RustTranscriber,
}

#[pymethods]
impl Transcriber {
    #[new]
    #[pyo3(signature = (
        model = PathBuf::from("model_optimized.onnx"),
        tokenizer = PathBuf::from("tokenizer.model"),
        beam_width = 32,
        lm = None,
        lm_weight = 0.45,
        word_bonus = 0.2,
        log_language_model = true,
        ort_dylib_path = None,
        ort_optimization = "disable",
        log_accelerator = true,
        fallback_sample_rate = 16000,
        skip_decode_errors = true,
        w2v_model_source = None,
        w2v_sample_rate = None,
        w2v_feature_size = None,
        w2v_stride = None,
        w2v_feature_dim = None,
        w2v_padding_value = None,
        blank_id = 0,
        n_best = None,
        normalize_spaces = true,
        drop_empty_candidates = true,
        lm_bos = true,
        lm_eos = true,
    ))]
    fn new(
        model: PathBuf,
        tokenizer: PathBuf,
        beam_width: usize,
        lm: Option<PathBuf>,
        lm_weight: f32,
        word_bonus: f32,
        log_language_model: bool,
        ort_dylib_path: Option<PathBuf>,
        ort_optimization: &str,
        log_accelerator: bool,
        fallback_sample_rate: u32,
        skip_decode_errors: bool,
        w2v_model_source: Option<String>,
        w2v_sample_rate: Option<u32>,
        w2v_feature_size: Option<usize>,
        w2v_stride: Option<usize>,
        w2v_feature_dim: Option<usize>,
        w2v_padding_value: Option<f32>,
        blank_id: u32,
        n_best: Option<usize>,
        normalize_spaces: bool,
        drop_empty_candidates: bool,
        lm_bos: bool,
        lm_eos: bool,
    ) -> PyResult<Self> {
        let config = build_config(
            model,
            tokenizer,
            beam_width,
            lm,
            lm_weight,
            word_bonus,
            log_language_model,
            ort_dylib_path,
            ort_optimization,
            log_accelerator,
            fallback_sample_rate,
            skip_decode_errors,
            w2v_model_source,
            w2v_sample_rate,
            w2v_feature_size,
            w2v_stride,
            w2v_feature_dim,
            w2v_padding_value,
            blank_id,
            n_best,
            normalize_spaces,
            drop_empty_candidates,
            lm_bos,
            lm_eos,
        )?;
        let inner = RustTranscriber::new(config).map_err(to_py_error)?;
        Ok(Self { inner })
    }

    fn transcribe_file(&mut self, audio_file: PathBuf) -> PyResult<String> {
        self.inner
            .transcribe_audio_file(audio_file)
            .map(|result| result.transcript)
            .map_err(to_py_error)
    }

    fn transcribe_file_with_report<'py>(
        &mut self,
        py: Python<'py>,
        audio_file: PathBuf,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self
            .inner
            .transcribe_audio_file(audio_file)
            .map_err(to_py_error)?;
        transcription_result_to_py(py, result)
    }

    #[pyo3(signature = (audio_bytes, format_hint = None))]
    fn transcribe_bytes(
        &mut self,
        audio_bytes: Vec<u8>,
        format_hint: Option<&str>,
    ) -> PyResult<String> {
        self.inner
            .transcribe_audio_bytes(audio_bytes, format_hint)
            .map(|result| result.transcript)
            .map_err(to_py_error)
    }

    #[pyo3(signature = (audio_bytes, format_hint = None))]
    fn transcribe_bytes_with_report<'py>(
        &mut self,
        py: Python<'py>,
        audio_bytes: Vec<u8>,
        format_hint: Option<&str>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self
            .inner
            .transcribe_audio_bytes(audio_bytes, format_hint)
            .map_err(to_py_error)?;
        transcription_result_to_py(py, result)
    }
}

#[allow(clippy::too_many_arguments)]
fn build_config(
    model: PathBuf,
    tokenizer: PathBuf,
    beam_width: usize,
    lm: Option<PathBuf>,
    lm_weight: f32,
    word_bonus: f32,
    log_language_model: bool,
    ort_dylib_path: Option<PathBuf>,
    ort_optimization: &str,
    log_accelerator: bool,
    fallback_sample_rate: u32,
    skip_decode_errors: bool,
    w2v_model_source: Option<String>,
    w2v_sample_rate: Option<u32>,
    w2v_feature_size: Option<usize>,
    w2v_stride: Option<usize>,
    w2v_feature_dim: Option<usize>,
    w2v_padding_value: Option<f32>,
    blank_id: u32,
    n_best: Option<usize>,
    normalize_spaces: bool,
    drop_empty_candidates: bool,
    lm_bos: bool,
    lm_eos: bool,
) -> PyResult<TranscriptionConfig> {
    let candidate_processing = CandidateProcessingConfig {
        normalize_spaces,
        drop_empty_candidates,
    };
    Ok(TranscriptionConfig {
        runtime: RuntimeConfig { ort_dylib_path },
        audio: AudioDecodeConfig {
            fallback_sample_rate,
            skip_decode_errors,
        },
        encoder: EncoderConfig {
            w2v_bert: W2vBertEncoderConfig {
                model_source: w2v_model_source,
                sample_rate: w2v_sample_rate,
                feature_size: w2v_feature_size,
                stride: w2v_stride,
                feature_dim: w2v_feature_dim,
                padding_value: w2v_padding_value,
            },
        },
        model: AcousticModelConfig {
            path: model,
            session: ModelConfig {
                optimization_level: parse_optimization_level(ort_optimization)?,
                log_accelerator,
            },
        },
        decoder: DecoderConfig {
            ctc: CtcDecoderConfig {
                blank_id,
                beam_width,
                n_best: n_best.unwrap_or(beam_width),
            },
            text: TextDecoderConfig {
                tokenizer_path: tokenizer,
                normalize_spaces,
                drop_empty_candidates,
            },
            language_model: lm.map(|path| LmConfig {
                path,
                weight: lm_weight,
                word_bonus,
                log_language_model,
                bos: lm_bos,
                eos: lm_eos,
                candidate_processing,
            }),
        },
    })
}

#[pymodule]
fn w2v_bert_uk(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(initialize_ort, module)?)?;
    module.add_function(wrap_pyfunction!(preload_cuda_dylibs, module)?)?;
    module.add_function(wrap_pyfunction!(transcribe_file, module)?)?;
    module.add_function(wrap_pyfunction!(transcribe_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(transcribe_file_with_report, module)?)?;
    module.add_function(wrap_pyfunction!(transcribe_bytes_with_report, module)?)?;
    module.add_class::<Transcriber>()?;
    Ok(())
}

fn transcription_result_to_py<'py>(
    py: Python<'py>,
    result: TranscriptionResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("transcript", result.transcript)?;
    dict.set_item("timings", timings_to_py(py, &result.timings)?)?;
    dict.set_item("candidates", candidates_to_py(py, &result.candidates)?)?;
    Ok(dict)
}

fn timings_to_py<'py>(py: Python<'py>, timings: &TimingReport) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("audio_duration_seconds", timings.audio_duration_seconds)?;
    dict.set_item("feature_rows", timings.feature_rows)?;
    dict.set_item("feature_cols", timings.feature_cols)?;
    dict.set_item("feature_count", timings.feature_count)?;
    dict.set_item(
        "audio_decode_seconds",
        duration_seconds(timings.audio_decode_elapsed),
    )?;
    dict.set_item("feature_seconds", duration_seconds(timings.feature_elapsed))?;
    dict.set_item(
        "model_session_seconds",
        duration_seconds(timings.model.session_elapsed),
    )?;
    dict.set_item(
        "model_input_seconds",
        duration_seconds(timings.model.input_elapsed),
    )?;
    dict.set_item(
        "model_inference_seconds",
        duration_seconds(timings.model.inference_elapsed),
    )?;
    dict.set_item(
        "model_ctc_seconds",
        duration_seconds(timings.model.ctc_elapsed),
    )?;
    dict.set_item(
        "tokenizer_load_seconds",
        duration_seconds(timings.tokenizer_load_elapsed),
    )?;
    dict.set_item(
        "text_decode_seconds",
        duration_seconds(timings.text_decode_elapsed),
    )?;
    dict.set_item("lm_seconds", duration_seconds(timings.lm_elapsed))?;
    dict.set_item(
        "measured_seconds",
        duration_seconds(timings.measured_elapsed()),
    )?;
    match &timings.best_candidate {
        Some(candidate) => dict.set_item("best_candidate", candidate_to_py(py, candidate)?)?,
        None => dict.set_item("best_candidate", py.None())?,
    }
    Ok(dict)
}

fn candidates_to_py<'py>(
    py: Python<'py>,
    candidates: &[ScoredCandidate],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for candidate in candidates {
        list.append(candidate_to_py(py, candidate)?)?;
    }
    Ok(list)
}

fn candidate_to_py<'py>(
    py: Python<'py>,
    candidate: &ScoredCandidate,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("text", &candidate.text)?;
    dict.set_item("ctc_log_prob", candidate.ctc_log_prob)?;
    dict.set_item("lm_log_prob", candidate.lm_log_prob)?;
    dict.set_item("word_count", candidate.word_count)?;
    dict.set_item("total_score", candidate.total_score)?;
    Ok(dict)
}

fn duration_seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn to_py_error(error: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn parse_optimization_level(value: &str) -> PyResult<ModelOptimizationLevel> {
    match value {
        "disable" => Ok(ModelOptimizationLevel::Disable),
        "level1" => Ok(ModelOptimizationLevel::Level1),
        "level2" => Ok(ModelOptimizationLevel::Level2),
        "level3" => Ok(ModelOptimizationLevel::Level3),
        "all" => Ok(ModelOptimizationLevel::All),
        other => Err(PyRuntimeError::new_err(format!(
            "invalid ort_optimization {other:?}; expected disable, level1, level2, level3, or all"
        ))),
    }
}
