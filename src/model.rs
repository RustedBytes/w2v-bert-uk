use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use asr_features::FeatureMatrix;
use half::f16;
use log::info;
#[cfg(any(feature = "coreml", feature = "cuda"))]
use ort::ep;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Tensor, TensorElementType, ValueType};

use crate::ctc::{CtcCandidate, ctc_log_probs, threaded_ctc_beam_search_decode_n_best};

#[derive(Clone, Copy, Debug)]
pub enum ModelOptimizationLevel {
    Disable,
    Level1,
    Level2,
    Level3,
    All,
}

impl From<ModelOptimizationLevel> for GraphOptimizationLevel {
    fn from(value: ModelOptimizationLevel) -> Self {
        match value {
            ModelOptimizationLevel::Disable => GraphOptimizationLevel::Disable,
            ModelOptimizationLevel::Level1 => GraphOptimizationLevel::Level1,
            ModelOptimizationLevel::Level2 => GraphOptimizationLevel::Level2,
            ModelOptimizationLevel::Level3 => GraphOptimizationLevel::Level3,
            ModelOptimizationLevel::All => GraphOptimizationLevel::All,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub optimization_level: ModelOptimizationLevel,
    pub log_accelerator: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            optimization_level: ModelOptimizationLevel::Disable,
            log_accelerator: true,
        }
    }
}

pub struct ModelOutput {
    pub candidates: Vec<CtcCandidate>,
    pub session_elapsed: Duration,
    pub input_elapsed: Duration,
    pub inference_elapsed: Duration,
    pub ctc_elapsed: Duration,
}

pub struct CtcLogProbs {
    pub frames: usize,
    pub vocab_size: usize,
    pub values: Vec<f32>,
    pub session_elapsed: Duration,
    pub input_elapsed: Duration,
    pub inference_elapsed: Duration,
    pub log_softmax_elapsed: Duration,
}

pub struct CtcModel {
    session: Session,
    input_name: String,
    input_precision: ModelFloatPrecision,
    output_precision: ModelFloatPrecision,
    session_elapsed: Duration,
}

#[derive(Clone, Copy, Debug)]
enum ModelFloatPrecision {
    F16,
    F32,
}

impl CtcModel {
    pub fn load(model_path: &Path, config: &ModelConfig) -> Result<Self> {
        let session_start = Instant::now();
        let builder = Session::builder()
            .map_err(|error| anyhow!("failed to create ONNX Runtime session builder: {error}"))?;
        let builder = builder
            // The shipped/default model is already graph-optimized. Re-running ORT's
            // optimizer dominates cold session creation and does not improve this
            // model in practice.
            .with_optimization_level(config.optimization_level.into())
            .map_err(|error| {
                anyhow!("failed to configure ONNX Runtime graph optimization: {error}")
            })?;

        let mut builder = builder
            .with_execution_providers(configured_execution_providers())
            .map_err(|error| {
                anyhow!("failed to configure ONNX Runtime execution providers: {error}")
            })?;
        if config.log_accelerator {
            info!("onnx accelerator: {}", configured_accelerator_label());
        }

        let session = builder.commit_from_file(model_path).map_err(|error| {
            anyhow!(
                "failed to load ONNX model {}: {error}",
                model_path.display()
            )
        })?;
        let session_elapsed = session_start.elapsed();

        let input = session
            .inputs()
            .first()
            .ok_or_else(|| anyhow!("ONNX model has no inputs"))?;
        let input_name = input.name().to_string();
        let input_precision = model_float_precision(input.dtype(), "input")?;
        let output_precision = session
            .outputs()
            .first()
            .ok_or_else(|| anyhow!("ONNX model has no outputs"))
            .and_then(|output| model_float_precision(output.dtype(), "output"))?;

        Ok(Self {
            session,
            input_name,
            input_precision,
            output_precision,
            session_elapsed,
        })
    }

    pub fn session_elapsed(&self) -> Duration {
        self.session_elapsed
    }

    pub fn run(
        &mut self,
        features: FeatureMatrix,
        blank_id: u32,
        beam_width: usize,
        n_best: usize,
    ) -> Result<ModelOutput> {
        self.run_with_reported_session_elapsed(
            features,
            blank_id,
            beam_width,
            n_best,
            Duration::ZERO,
        )
    }

    pub fn run_log_probs(&mut self, features: FeatureMatrix) -> Result<CtcLogProbs> {
        let session_elapsed = self.session_elapsed();
        self.run_log_probs_with_reported_session_elapsed(features, session_elapsed)
    }

    pub(crate) fn run_with_reported_session_elapsed(
        &mut self,
        features: FeatureMatrix,
        blank_id: u32,
        beam_width: usize,
        n_best: usize,
        session_elapsed: Duration,
    ) -> Result<ModelOutput> {
        let raw_output = self.run_raw_output(features)?;

        let ctc_start = Instant::now();
        let candidates = match &raw_output {
            RawModelOutput::F16 { shape, logits, .. } => {
                threaded_ctc_beam_search_decode_n_best(shape, logits, blank_id, beam_width, n_best)?
            }
            RawModelOutput::F32 { shape, logits, .. } => {
                threaded_ctc_beam_search_decode_n_best(shape, logits, blank_id, beam_width, n_best)?
            }
        };

        Ok(ModelOutput {
            candidates,
            session_elapsed,
            input_elapsed: raw_output.input_elapsed(),
            inference_elapsed: raw_output.inference_elapsed(),
            ctc_elapsed: ctc_start.elapsed(),
        })
    }

    pub(crate) fn run_log_probs_with_reported_session_elapsed(
        &mut self,
        features: FeatureMatrix,
        session_elapsed: Duration,
    ) -> Result<CtcLogProbs> {
        let raw_output = self.run_raw_output(features)?;
        let log_softmax_start = Instant::now();
        let (frames, vocab_size, values) = match &raw_output {
            RawModelOutput::F16 { shape, logits, .. } => ctc_log_probs(shape, logits)?,
            RawModelOutput::F32 { shape, logits, .. } => ctc_log_probs(shape, logits)?,
        };

        Ok(CtcLogProbs {
            frames,
            vocab_size,
            values,
            session_elapsed,
            input_elapsed: raw_output.input_elapsed(),
            inference_elapsed: raw_output.inference_elapsed(),
            log_softmax_elapsed: log_softmax_start.elapsed(),
        })
    }

    fn run_raw_output(&mut self, features: FeatureMatrix) -> Result<RawModelOutput> {
        let input_start = Instant::now();
        let rows = features.rows;
        let cols = features.cols;
        let values = features.values;
        let input_elapsed;
        let inference_start;
        let outputs = match self.input_precision {
            ModelFloatPrecision::F16 => {
                let values = values.into_iter().map(f16::from_f32).collect::<Vec<_>>();
                let input = Tensor::from_array(([1usize, rows, cols], values))
                    .map_err(|error| anyhow!("failed to create f16 ONNX input tensor: {error}"))?;
                input_elapsed = input_start.elapsed();

                inference_start = Instant::now();
                self.session
                    .run(ort::inputs! {
                        self.input_name.as_str() => input,
                    })
                    .map_err(|error| anyhow!("failed to run ONNX inference: {error}"))?
            }
            ModelFloatPrecision::F32 => {
                let input = Tensor::from_array(([1usize, rows, cols], values))
                    .map_err(|error| anyhow!("failed to create f32 ONNX input tensor: {error}"))?;
                input_elapsed = input_start.elapsed();

                inference_start = Instant::now();
                self.session
                    .run(ort::inputs! {
                        self.input_name.as_str() => input,
                    })
                    .map_err(|error| anyhow!("failed to run ONNX inference: {error}"))?
            }
        };
        let inference_elapsed = inference_start.elapsed();

        let first_output = outputs
            .values()
            .next()
            .ok_or_else(|| anyhow!("ONNX model returned no outputs"))?;

        match self.output_precision {
            ModelFloatPrecision::F16 => {
                let (shape, logits) =
                    first_output.try_extract_tensor::<f16>().map_err(|error| {
                        anyhow!("failed to read ONNX output tensor as f16 logits: {error}")
                    })?;
                Ok(RawModelOutput::F16 {
                    shape: shape.to_vec(),
                    logits: logits.to_vec(),
                    input_elapsed,
                    inference_elapsed,
                })
            }
            ModelFloatPrecision::F32 => {
                let (shape, logits) =
                    first_output.try_extract_tensor::<f32>().map_err(|error| {
                        anyhow!("failed to read ONNX output tensor as f32 logits: {error}")
                    })?;
                Ok(RawModelOutput::F32 {
                    shape: shape.to_vec(),
                    logits: logits.to_vec(),
                    input_elapsed,
                    inference_elapsed,
                })
            }
        }
    }
}

enum RawModelOutput {
    F16 {
        shape: Vec<i64>,
        logits: Vec<f16>,
        input_elapsed: Duration,
        inference_elapsed: Duration,
    },
    F32 {
        shape: Vec<i64>,
        logits: Vec<f32>,
        input_elapsed: Duration,
        inference_elapsed: Duration,
    },
}

impl RawModelOutput {
    fn input_elapsed(&self) -> Duration {
        match self {
            Self::F16 { input_elapsed, .. } | Self::F32 { input_elapsed, .. } => *input_elapsed,
        }
    }

    fn inference_elapsed(&self) -> Duration {
        match self {
            Self::F16 {
                inference_elapsed, ..
            }
            | Self::F32 {
                inference_elapsed, ..
            } => *inference_elapsed,
        }
    }
}

fn model_float_precision(dtype: &ValueType, label: &str) -> Result<ModelFloatPrecision> {
    let ValueType::Tensor { ty, .. } = dtype else {
        return Err(anyhow!(
            "ONNX model {label} must be a tensor, got {dtype:?}"
        ));
    };

    match ty {
        TensorElementType::Float16 => Ok(ModelFloatPrecision::F16),
        TensorElementType::Float32 => Ok(ModelFloatPrecision::F32),
        ty => Err(anyhow!(
            "ONNX model {label} tensor must use f16 or f32, got {ty}"
        )),
    }
}

pub fn run_ctc_model(
    features: FeatureMatrix,
    model_path: &Path,
    beam_width: usize,
) -> Result<ModelOutput> {
    run_ctc_model_with_config(
        features,
        model_path,
        &ModelConfig::default(),
        0,
        beam_width,
        beam_width,
    )
}

pub fn run_ctc_model_with_config(
    features: FeatureMatrix,
    model_path: &Path,
    config: &ModelConfig,
    blank_id: u32,
    beam_width: usize,
    n_best: usize,
) -> Result<ModelOutput> {
    let mut model = CtcModel::load(model_path, config)?;
    let session_elapsed = model.session_elapsed();
    model.run_with_reported_session_elapsed(features, blank_id, beam_width, n_best, session_elapsed)
}

#[cfg(any(feature = "coreml", feature = "cuda"))]
fn configured_execution_providers() -> Vec<ep::ExecutionProviderDispatch> {
    let mut providers = Vec::new();

    #[cfg(feature = "cuda")]
    providers.push(
        ep::CUDA::default()
            .with_device_id(0)
            .with_tf32(true)
            .with_conv_max_workspace(true)
            .with_conv1d_pad_to_nc1d(true)
            .with_fuse_conv_bias(true)
            .build()
            .error_on_failure(),
    );

    #[cfg(feature = "coreml")]
    providers.push(
        ep::CoreML::default()
            .with_static_input_shapes(true)
            .with_compute_units(ep::coreml::ComputeUnits::All)
            .with_model_format(ep::coreml::ModelFormat::MLProgram)
            .with_specialization_strategy(ep::coreml::SpecializationStrategy::FastPrediction)
            .with_model_cache_dir("target/coreml-cache")
            .build(),
    );

    providers
}

#[cfg(not(any(feature = "coreml", feature = "cuda")))]
fn configured_execution_providers() -> [ort::ep::ExecutionProviderDispatch; 0] {
    []
}

#[cfg(all(feature = "cuda", feature = "coreml"))]
fn configured_accelerator_label() -> &'static str {
    "CUDA, CoreML (CPU fallback enabled)"
}

#[cfg(all(feature = "cuda", not(feature = "coreml")))]
fn configured_accelerator_label() -> &'static str {
    "CUDA (CPU fallback enabled)"
}

#[cfg(all(feature = "coreml", not(feature = "cuda")))]
fn configured_accelerator_label() -> &'static str {
    "CoreML (CPU fallback enabled)"
}

#[cfg(not(any(feature = "coreml", feature = "cuda")))]
fn configured_accelerator_label() -> &'static str {
    "CPU"
}
