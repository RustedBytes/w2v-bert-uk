use std::fs;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use burn::module::AutodiffModule;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{FloatDType, Int, IntDType, Tensor, TensorData, set_default_dtypes};
use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn_nn::loss::{CTCLossConfig, Reduction};
use burn_optim::{AdamWConfig, GradientsParams, Optimizer};
use kenlm::{Config as KenlmConfig, Model as KenlmModel};
use serde_json::{Value, json};
use splintr::SentencePieceTokenizer;

use crate::ctc::{CtcCandidate, threaded_ctc_beam_search_decode_n_best};
use crate::normalize_spaces;
use crate::paraformer::{
    EnhancedParaformerV2, EnhancedParaformerV2Config, ParaformerAlignmentMode, ParaformerV2,
    ParaformerV2Config,
};
use crate::squeezeformer::{SqueezeformerCtc, SqueezeformerCtcConfig, SqueezeformerEncoderConfig};
use crate::tokenizer::load_sentencepiece_tokenizer;
use crate::wav2vec::{Wav2VecBertConfig, Wav2VecBertCtc, Wav2VecBertCtcConfig};
use crate::zipformer::{ZipformerConfig, ZipformerCtc, ZipformerCtcConfig};

#[derive(Clone, Debug)]
pub enum TrainArchitecture {
    Squeezeformer,
    Zipformer,
    Paraformer,
    Wav2VecBert,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainBackendKind {
    Cpu,
    Cuda,
    Wgpu,
}

impl std::str::FromStr for TrainBackendKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "cpu" | "ndarray" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda),
            "wgpu" | "vulkan" | "metal" => Ok(Self::Wgpu),
            other => bail!("unknown training backend '{other}'"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainPrecision {
    F32,
    F16,
    Bf16,
}

impl std::str::FromStr for TrainPrecision {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "f32" | "fp32" | "float32" => Ok(Self::F32),
            "f16" | "fp16" | "float16" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            other => bail!("unknown training precision '{other}'"),
        }
    }
}

impl std::str::FromStr for TrainArchitecture {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "squeezeformer" => Ok(Self::Squeezeformer),
            "zipformer" => Ok(Self::Zipformer),
            "paraformer" => Ok(Self::Paraformer),
            "w2v-bert" | "w2v_bert" | "wav2vec" | "wav2vec-bert" => Ok(Self::Wav2VecBert),
            other => bail!("unknown architecture '{other}'"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BurnTrainConfig {
    pub architecture: TrainArchitecture,
    pub train_manifest: PathBuf,
    pub val_manifest: Option<PathBuf>,
    pub output_dir: PathBuf,
    pub variant: Option<String>,
    pub input_dim: usize,
    pub vocab_size: usize,
    pub blank_id: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub batch_size: usize,
    pub adaptive_batch: Option<AdaptiveBatchConfig>,
    pub sort_by_length_desc: bool,
    pub sort_buffer_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub log_every: usize,
    pub validate_every_steps: Option<usize>,
    pub max_train_samples: Option<usize>,
    pub max_val_samples: Option<usize>,
    pub tokenizer_path: Option<PathBuf>,
    pub val_beam_width: usize,
    pub val_n_best: usize,
    pub val_lm_path: Option<PathBuf>,
    pub val_lm_weight: f32,
    pub val_lm_word_bonus: f32,
    pub val_lm_bos: bool,
    pub val_lm_eos: bool,
    pub dry_run: bool,
    pub paraformer_alignment_mode: ParaformerAlignmentMode,
    pub paraformer_enhanced: bool,
    pub w2v_hf_model_dir: Option<PathBuf>,
    pub w2v_hf_load_weights: bool,
    pub w2v_activation_checkpointing: bool,
    pub resume_from: Option<PathBuf>,
    pub backend: TrainBackendKind,
    pub device_index: usize,
    pub precision: TrainPrecision,
}

impl Default for BurnTrainConfig {
    fn default() -> Self {
        Self {
            architecture: TrainArchitecture::Squeezeformer,
            train_manifest: PathBuf::from("train.jsonl"),
            val_manifest: None,
            output_dir: PathBuf::from("runs/burn"),
            variant: Some("sm".to_string()),
            input_dim: 80,
            vocab_size: 256,
            blank_id: 0,
            d_model: 256,
            num_layers: 16,
            num_heads: 4,
            batch_size: 8,
            adaptive_batch: None,
            sort_by_length_desc: false,
            sort_buffer_size: 4096,
            epochs: 10,
            learning_rate: 1.0e-3,
            weight_decay: 1.0e-2,
            log_every: 10,
            validate_every_steps: None,
            max_train_samples: None,
            max_val_samples: None,
            tokenizer_path: None,
            val_beam_width: 1,
            val_n_best: 1,
            val_lm_path: None,
            val_lm_weight: 0.45,
            val_lm_word_bonus: 0.0,
            val_lm_bos: true,
            val_lm_eos: true,
            dry_run: false,
            paraformer_alignment_mode: ParaformerAlignmentMode::Viterbi,
            paraformer_enhanced: false,
            w2v_hf_model_dir: None,
            w2v_hf_load_weights: false,
            w2v_activation_checkpointing: false,
            resume_from: None,
            backend: TrainBackendKind::Cpu,
            device_index: 0,
            precision: TrainPrecision::F32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveBatchUnit {
    Samples,
    Frames,
    PaddedFrames,
    FeatureValues,
}

impl std::str::FromStr for AdaptiveBatchUnit {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "samples" => Ok(Self::Samples),
            "frames" => Ok(Self::Frames),
            "padded-frames" | "padded_frames" => Ok(Self::PaddedFrames),
            "feature-values" | "feature_values" | "values" => Ok(Self::FeatureValues),
            other => bail!("unknown adaptive batch unit '{other}'"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveBatchConfig {
    pub unit: AdaptiveBatchUnit,
    pub budget: usize,
    pub max_samples: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct FeatureRecord {
    pub id: String,
    pub rows: usize,
    pub cols: usize,
    pub features: Vec<f32>,
    pub tokens: Vec<i64>,
    pub text: Option<String>,
}

#[derive(Clone, Debug)]
pub struct TrainBatch {
    pub features: Vec<f32>,
    pub batch_size: usize,
    pub max_frames: usize,
    pub feature_dim: usize,
    pub feature_lengths: Vec<usize>,
    pub targets: Vec<i64>,
    pub max_target_len: usize,
    pub target_lengths: Vec<usize>,
    pub reference_texts: Vec<Option<String>>,
}

#[derive(Clone, Debug)]
pub struct TrainSummary {
    pub epochs: usize,
    pub steps: usize,
    pub last_train_loss: Option<f32>,
    pub last_val_loss: Option<f32>,
    pub last_val_cer: Option<f32>,
    pub last_val_wer: Option<f32>,
}

#[derive(Clone, Debug, Default)]
struct ValidationSummary {
    loss: f32,
    token_error_rate: Option<f32>,
    cer: Option<f32>,
    wer: Option<f32>,
    decoded_samples: usize,
}

#[derive(Clone, Debug, Default)]
struct EditStats {
    edits: usize,
    reference_len: usize,
}

impl EditStats {
    fn add(&mut self, edits: usize, reference_len: usize) {
        self.edits += edits;
        self.reference_len += reference_len;
    }

    fn rate(&self) -> Option<f32> {
        (self.reference_len > 0).then_some(self.edits as f32 / self.reference_len as f32)
    }
}

struct ValidationDecoder {
    tokenizer: Option<SentencePieceTokenizer>,
    language_model: Option<ValidationLanguageModel>,
    beam_width: usize,
    n_best: usize,
}

struct ValidationLanguageModel {
    model: KenlmModel,
    weight: f32,
    word_bonus: f32,
    bos: bool,
    eos: bool,
}

#[derive(Clone, Debug)]
struct ResumeCheckpoint {
    dir: PathBuf,
    epoch: usize,
    epoch_complete: bool,
    global_step: usize,
    last_train_loss: Option<f32>,
    last_val_loss: Option<f32>,
    last_val_cer: Option<f32>,
    last_val_wer: Option<f32>,
}

impl ValidationDecoder {
    fn from_config(config: &BurnTrainConfig) -> Result<Self> {
        let tokenizer = config
            .tokenizer_path
            .as_deref()
            .map(load_sentencepiece_tokenizer)
            .transpose()?;
        let language_model = config
            .val_lm_path
            .as_ref()
            .map(|path| {
                if tokenizer.is_none() {
                    bail!(
                        "--val-lm-path requires --tokenizer so CTC candidates can be scored as text"
                    );
                }
                let model = KenlmModel::with_config(
                    path,
                    KenlmConfig {
                        show_progress: false,
                        ..KenlmConfig::default()
                    },
                )
                .with_context(|| {
                    format!("failed to load validation KenLM model {}", path.display())
                })?;
                Ok(ValidationLanguageModel {
                    model,
                    weight: config.val_lm_weight,
                    word_bonus: config.val_lm_word_bonus,
                    bos: config.val_lm_bos,
                    eos: config.val_lm_eos,
                })
            })
            .transpose()?;
        Ok(Self {
            tokenizer,
            language_model,
            beam_width: config.val_beam_width.max(1),
            n_best: config.val_n_best.max(1),
        })
    }

    fn decode_tokens(&self, tokens: &[u32]) -> String {
        if let Some(tokenizer) = &self.tokenizer {
            tokenizer.decode_lossy(tokens)
        } else {
            tokens
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        }
    }

    fn reference_text(&self, text: Option<&String>, tokens: &[i64]) -> String {
        text.cloned().unwrap_or_else(|| {
            let token_ids = tokens
                .iter()
                .filter_map(|token| u32::try_from(*token).ok())
                .collect::<Vec<_>>();
            self.decode_tokens(&token_ids)
        })
    }

    fn decode_best(
        &self,
        frame_logits: &[f32],
        frames: usize,
        vocab_size: usize,
        blank_id: usize,
    ) -> Result<Vec<u32>> {
        if self.beam_width <= 1 && self.language_model.is_none() {
            return greedy_decode_frames(frame_logits, frames, vocab_size, blank_id);
        }
        let candidates = threaded_ctc_beam_search_decode_n_best(
            &[frames as i64, vocab_size as i64],
            frame_logits,
            blank_id as u32,
            self.beam_width,
            self.n_best,
        )?;
        if let Some(language_model) = &self.language_model {
            self.rerank_with_lm(candidates, language_model)
        } else {
            candidates
                .into_iter()
                .next()
                .map(|candidate| candidate.token_ids)
                .ok_or_else(|| anyhow!("beam search produced no candidates"))
        }
    }

    fn rerank_with_lm(
        &self,
        candidates: Vec<CtcCandidate>,
        language_model: &ValidationLanguageModel,
    ) -> Result<Vec<u32>> {
        let mut best: Option<(Vec<u32>, f32)> = None;
        for candidate in candidates {
            let text = normalize_spaces(&self.decode_tokens(&candidate.token_ids));
            if text.is_empty() {
                continue;
            }
            let lm_log_prob = language_model
                .model
                .score(&text, language_model.bos, language_model.eos)
                .with_context(|| {
                    format!("failed to score validation candidate with KenLM: {text}")
                })?
                * std::f32::consts::LN_10;
            let word_count = text.split_whitespace().count();
            let total = candidate.ctc_log_prob
                + language_model.weight * lm_log_prob
                + language_model.word_bonus * word_count as f32;
            if best
                .as_ref()
                .is_none_or(|(_, best_score)| total > *best_score)
            {
                best = Some((candidate.token_ids, total));
            }
        }
        best.map(|(tokens, _)| tokens)
            .ok_or_else(|| anyhow!("language-model decoder produced no candidates"))
    }
}

pub fn run_burn_training(config: BurnTrainConfig) -> Result<TrainSummary> {
    fs::create_dir_all(&config.output_dir).with_context(|| {
        format!(
            "failed to create output directory {}",
            config.output_dir.display()
        )
    })?;
    validate_config(&config)?;
    write_run_config(&config)?;

    match config.backend {
        TrainBackendKind::Cpu => run_burn_training_cpu(config),
        TrainBackendKind::Cuda => run_burn_training_cuda(config),
        TrainBackendKind::Wgpu => run_burn_training_wgpu(config),
    }
}

fn run_burn_training_cpu(config: BurnTrainConfig) -> Result<TrainSummary> {
    if config.device_index != 0 {
        bail!("cpu backend only supports --device-index 0");
    }
    type InnerBackend = burn_ndarray::NdArray<f32>;
    let device = Default::default();
    run_burn_training_inner::<InnerBackend>(&config, &device)
}

#[cfg(feature = "burn-cuda-backend")]
fn run_burn_training_cuda(config: BurnTrainConfig) -> Result<TrainSummary> {
    let device = burn_cuda::CudaDevice {
        index: config.device_index,
    };
    match config.precision {
        TrainPrecision::F32 => run_burn_training_inner::<burn_cuda::Cuda<f32>>(&config, &device),
        TrainPrecision::F16 => {
            run_burn_training_inner::<burn_cuda::Cuda<burn::tensor::f16>>(&config, &device)
        }
        TrainPrecision::Bf16 => {
            run_burn_training_inner::<burn_cuda::Cuda<burn::tensor::bf16>>(&config, &device)
        }
    }
}

#[cfg(not(feature = "burn-cuda-backend"))]
fn run_burn_training_cuda(_config: BurnTrainConfig) -> Result<TrainSummary> {
    bail!("CUDA training requires building with --features burn-cuda-backend")
}

#[cfg(feature = "burn-wgpu-backend")]
fn run_burn_training_wgpu(config: BurnTrainConfig) -> Result<TrainSummary> {
    let device = burn_wgpu::WgpuDevice::DiscreteGpu(config.device_index);
    match config.precision {
        TrainPrecision::F32 => run_burn_training_inner::<burn_wgpu::Wgpu<f32>>(&config, &device),
        TrainPrecision::F16 => {
            run_burn_training_inner::<burn_wgpu::Wgpu<burn::tensor::f16>>(&config, &device)
        }
        TrainPrecision::Bf16 => bail!("BF16 training is not supported by Burn WGPU backend"),
    }
}

#[cfg(not(feature = "burn-wgpu-backend"))]
fn run_burn_training_wgpu(_config: BurnTrainConfig) -> Result<TrainSummary> {
    bail!("WGPU training requires building with --features burn-wgpu-backend")
}

fn run_burn_training_inner<InnerBackend>(
    config: &BurnTrainConfig,
    device: &InnerBackend::Device,
) -> Result<TrainSummary>
where
    InnerBackend: Backend,
{
    type TrainBackend<InnerBackend> = burn_autodiff::Autodiff<InnerBackend>;
    configure_training_precision::<TrainBackend<InnerBackend>>(config, device)?;

    match config.architecture {
        TrainArchitecture::Squeezeformer => {
            let encoder = config
                .variant
                .as_deref()
                .and_then(SqueezeformerEncoderConfig::variant)
                .unwrap_or_else(|| {
                    SqueezeformerEncoderConfig::new(
                        config.input_dim,
                        config.d_model,
                        config.num_layers,
                        config.num_heads,
                    )
                });
            let model = SqueezeformerCtcConfig {
                encoder,
                vocab_size: config.vocab_size,
            }
            .init::<TrainBackend<InnerBackend>>(device);
            train_ctc_model(model, config, device)
        }
        TrainArchitecture::Zipformer => {
            let mut encoder = config
                .variant
                .as_deref()
                .and_then(ZipformerConfig::variant)
                .unwrap_or_else(|| ZipformerConfig::new(config.input_dim));
            encoder.input_dim = config.input_dim;
            let model = ZipformerCtcConfig {
                encoder,
                vocab_size: config.vocab_size,
            }
            .init::<TrainBackend<InnerBackend>>(device);
            train_ctc_model(model, config, device)
        }
        TrainArchitecture::Paraformer => {
            if config.paraformer_enhanced {
                let model_config = config
                    .variant
                    .as_deref()
                    .and_then(|variant| {
                        EnhancedParaformerV2Config::variant(
                            variant,
                            config.input_dim,
                            config.vocab_size,
                            config.blank_id,
                        )
                    })
                    .unwrap_or_else(|| {
                        let mut value =
                            EnhancedParaformerV2Config::new(config.input_dim, config.vocab_size)
                                .with_blank_id(config.blank_id);
                        value.base.encoder_dim = config.d_model;
                        value.base.decoder_dim = config.d_model;
                        value.base.encoder_layers = config.num_layers;
                        value.base.attention_heads = config.num_heads;
                        value
                    });
                let model = model_config.init::<TrainBackend<InnerBackend>>(device);
                train_enhanced_paraformer_model(model, config, device)
            } else {
                let model_config = config
                    .variant
                    .as_deref()
                    .and_then(|variant| {
                        ParaformerV2Config::variant(
                            variant,
                            config.input_dim,
                            config.vocab_size,
                            config.blank_id,
                        )
                    })
                    .unwrap_or_else(|| {
                        let mut value =
                            ParaformerV2Config::new(config.input_dim, config.vocab_size)
                                .with_blank_id(config.blank_id);
                        value.encoder_dim = config.d_model;
                        value.decoder_dim = config.d_model;
                        value.encoder_layers = config.num_layers;
                        value.attention_heads = config.num_heads;
                        value
                    });
                let model = model_config.init::<TrainBackend<InnerBackend>>(device);
                train_paraformer_model(model, config, device)
            }
        }
        TrainArchitecture::Wav2VecBert => {
            if config.w2v_activation_checkpointing {
                type CheckpointBackend<InnerBackend> =
                    burn_autodiff::Autodiff<InnerBackend, BalancedCheckpointing>;
                configure_training_precision::<CheckpointBackend<InnerBackend>>(config, device)?;
                train_wav2vec_model::<CheckpointBackend<InnerBackend>>(config, device)
            } else {
                train_wav2vec_model::<TrainBackend<InnerBackend>>(config, device)
            }
        }
    }
}

fn configure_training_precision<B>(config: &BurnTrainConfig, device: &B::Device) -> Result<()>
where
    B: Backend,
{
    let dtype = match config.precision {
        TrainPrecision::F32 => FloatDType::F32,
        TrainPrecision::F16 => FloatDType::F16,
        TrainPrecision::Bf16 => FloatDType::BF16,
    };
    set_default_dtypes::<B>(device, dtype, IntDType::I64).with_context(|| {
        format!(
            "backend {:?} does not support {:?} precision on device index {}",
            config.backend, config.precision, config.device_index
        )
    })
}

fn train_wav2vec_model<B>(config: &BurnTrainConfig, device: &B::Device) -> Result<TrainSummary>
where
    B: AutodiffBackend,
{
    let mut model_config = if let Some(path) = &config.w2v_hf_model_dir {
        Wav2VecBertCtcConfig::from_huggingface_dir(path, Some(config.vocab_size))?
    } else {
        let encoder =
            Wav2VecBertConfig::new(config.input_dim, config.d_model).with_layers(config.num_layers);
        Wav2VecBertCtcConfig {
            encoder,
            vocab_size: config.vocab_size,
        }
    };
    model_config.encoder.activation_checkpointing = config.w2v_activation_checkpointing;
    let mut train_config = config.clone();
    train_config.input_dim = model_config.encoder.feature_dim;
    train_config.d_model = model_config.encoder.hidden_size;
    train_config.num_layers = model_config.encoder.num_hidden_layers;
    train_config.num_heads = model_config.encoder.num_attention_heads;
    let mut model = model_config.init::<B>(device);
    if config.w2v_hf_load_weights {
        let path = config
            .w2v_hf_model_dir
            .as_ref()
            .context("--w2v-hf-load-weights requires --w2v-hf-model-dir")?;
        let report = model.load_huggingface_weights(path, device)?;
        log::info!(
            "loaded {} Hugging Face W2V-BERT tensors from {} file(s), skipped_missing={}, skipped_shape={}",
            report.loaded_count(),
            report.source_files.len(),
            report.skipped_missing.len(),
            report.skipped_shape.len()
        );
    }
    train_ctc_model(model, &train_config, device)
}

trait TrainableCtc<B: AutodiffBackend>: AutodiffModule<B> {
    fn ctc_logits(&self, features: Tensor<B, 3>, lengths: Vec<usize>)
    -> (Tensor<B, 3>, Vec<usize>);
}

impl<B: AutodiffBackend> TrainableCtc<B> for SqueezeformerCtc<B> {
    fn ctc_logits(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        self.forward_with_lengths(features, Some(lengths))
    }
}

impl<B: AutodiffBackend> TrainableCtc<B> for ZipformerCtc<B> {
    fn ctc_logits(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        self.forward_with_lengths(features, lengths)
    }
}

impl<B: AutodiffBackend> TrainableCtc<B> for Wav2VecBertCtc<B> {
    fn ctc_logits(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        self.forward_with_lengths(features, lengths)
    }
}

impl<B: AutodiffBackend> TrainableCtc<B> for ParaformerV2<B> {
    fn ctc_logits(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let output = self.forward(features, lengths);
        (output.ctc_log_probs, output.encoder_lengths)
    }
}

fn train_ctc_model<B, M>(
    mut model: M,
    config: &BurnTrainConfig,
    device: &B::Device,
) -> Result<TrainSummary>
where
    B: AutodiffBackend,
    M: TrainableCtc<B>,
{
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    let resume = load_resume_checkpoint(config)?;
    model = load_model_checkpoint(model, &resume, device)?;
    optimizer = load_optimizer_checkpoint::<B, M, _>(optimizer, &resume, device)?;
    let mut global_step = resume
        .as_ref()
        .map_or(0, |checkpoint| checkpoint.global_step);
    let mut last_train_loss = resume
        .as_ref()
        .and_then(|checkpoint| checkpoint.last_train_loss);
    let mut last_val_loss = resume
        .as_ref()
        .and_then(|checkpoint| checkpoint.last_val_loss);
    let mut last_val_cer = resume
        .as_ref()
        .and_then(|checkpoint| checkpoint.last_val_cer);
    let mut last_val_wer = resume
        .as_ref()
        .and_then(|checkpoint| checkpoint.last_val_wer);
    let start_epoch = resume_start_epoch(&resume);
    let decoder = ValidationDecoder::from_config(config)?;
    let started = Instant::now();

    for epoch in start_epoch..=config.epochs {
        let mut train_batches = StreamingBatchLoader::new(
            config.train_manifest.clone(),
            config.batch_size,
            config.adaptive_batch,
            config.sort_by_length_desc,
            config.sort_buffer_size,
            config.input_dim,
            config.max_train_samples,
        )?;
        while let Some(batch) = train_batches.next_batch()? {
            let loss = ctc_loss_for_batch::<B, M>(&model, &batch, config.blank_id, device);
            let loss_value = scalar_value(loss.clone())?;
            last_train_loss = Some(loss_value);

            if !config.dry_run {
                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optimizer.step(config.learning_rate, model, grads);
            }

            global_step += 1;
            if global_step == 1 || global_step % config.log_every == 0 {
                println!(
                    "epoch={epoch} step={global_step} train_ctc_loss={loss_value:.6} elapsed_sec={:.1}",
                    started.elapsed().as_secs_f64()
                );
            }

            if let (Some(_), Some(every)) = (&config.val_manifest, config.validate_every_steps) {
                if every > 0 && global_step % every == 0 {
                    let val = evaluate_ctc_model::<B, M>(&model, config, device, &decoder)?;
                    println!(
                        "{}",
                        format_validation_summary(epoch, Some(global_step), "val_ctc_loss", &val)
                    );
                    last_val_loss = Some(val.loss);
                    last_val_cer = val.cer;
                    last_val_wer = val.wer;
                    save_training_checkpoint::<B, M, _>(
                        &model,
                        &optimizer,
                        config,
                        epoch,
                        false,
                        global_step,
                        last_train_loss,
                        last_val_loss,
                        last_val_cer,
                        last_val_wer,
                    )?;
                }
            }
        }

        if config.val_manifest.is_some() {
            let val = evaluate_ctc_model::<B, M>(&model, config, device, &decoder)?;
            println!(
                "{}",
                format_validation_summary(epoch, None, "val_ctc_loss", &val)
            );
            last_val_loss = Some(val.loss);
            last_val_cer = val.cer;
            last_val_wer = val.wer;
        }
        save_training_checkpoint::<B, M, _>(
            &model,
            &optimizer,
            config,
            epoch,
            true,
            global_step,
            last_train_loss,
            last_val_loss,
            last_val_cer,
            last_val_wer,
        )?;
    }

    Ok(TrainSummary {
        epochs: config.epochs,
        steps: global_step,
        last_train_loss,
        last_val_loss,
        last_val_cer,
        last_val_wer,
    })
}

fn evaluate_ctc_model<B, M>(
    model: &M,
    config: &BurnTrainConfig,
    device: &B::Device,
    decoder: &ValidationDecoder,
) -> Result<ValidationSummary>
where
    B: AutodiffBackend,
    M: TrainableCtc<B>,
{
    let mut total = 0.0f64;
    let mut count = 0usize;
    let mut metrics = ValidationMetrics::default();
    let val_manifest = config
        .val_manifest
        .as_ref()
        .ok_or_else(|| anyhow!("validation requested without val_manifest"))?;
    let mut batches = StreamingBatchLoader::new(
        val_manifest.clone(),
        config.batch_size,
        config.adaptive_batch,
        config.sort_by_length_desc,
        config.sort_buffer_size,
        config.input_dim,
        config.max_val_samples,
    )?;
    while let Some(batch) = batches.next_batch()? {
        let (logits_or_log_probs, output_lengths) =
            ctc_logits_for_batch::<B, M>(model, &batch, device);
        let loss = ctc_loss_from_logits(
            logits_or_log_probs.clone(),
            output_lengths.clone(),
            &batch,
            config.blank_id,
            device,
        );
        total += f64::from(scalar_value(loss)?);
        let predictions = decode_validation_batch(
            logits_or_log_probs,
            &output_lengths,
            config.blank_id,
            decoder,
        )?;
        metrics.update(&batch, &predictions, decoder);
        count += 1;
    }
    if count == 0 {
        bail!("validation manifest is empty");
    }
    Ok(metrics.summary((total / count as f64) as f32))
}

fn train_paraformer_model<B>(
    mut model: ParaformerV2<B>,
    config: &BurnTrainConfig,
    device: &B::Device,
) -> Result<TrainSummary>
where
    B: AutodiffBackend,
{
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    let resume = load_resume_checkpoint(config)?;
    model = load_model_checkpoint(model, &resume, device)?;
    optimizer = load_optimizer_checkpoint::<B, ParaformerV2<B>, _>(optimizer, &resume, device)?;
    let mut global_step = resume
        .as_ref()
        .map_or(0, |checkpoint| checkpoint.global_step);
    let mut last_train_loss = resume
        .as_ref()
        .and_then(|checkpoint| checkpoint.last_train_loss);
    let mut last_val_loss = resume
        .as_ref()
        .and_then(|checkpoint| checkpoint.last_val_loss);
    let mut last_val_cer = resume
        .as_ref()
        .and_then(|checkpoint| checkpoint.last_val_cer);
    let mut last_val_wer = resume
        .as_ref()
        .and_then(|checkpoint| checkpoint.last_val_wer);
    let start_epoch = resume_start_epoch(&resume);
    let decoder = ValidationDecoder::from_config(config)?;
    let started = Instant::now();

    for epoch in start_epoch..=config.epochs {
        let mut train_batches = StreamingBatchLoader::new(
            config.train_manifest.clone(),
            config.batch_size,
            config.adaptive_batch,
            config.sort_by_length_desc,
            config.sort_buffer_size,
            config.input_dim,
            config.max_train_samples,
        )?;
        while let Some(batch) = train_batches.next_batch()? {
            let loss_output = paraformer_loss_for_batch(&model, &batch, config, device);
            let loss_value = scalar_value(loss_output.loss.clone())?;
            let ctc_value = scalar_value(loss_output.ctc_loss.clone())?;
            let ce_value = scalar_value(loss_output.ce_loss.clone())?;
            last_train_loss = Some(loss_value);

            if !config.dry_run {
                let grads = GradientsParams::from_grads(loss_output.loss.backward(), &model);
                model = optimizer.step(config.learning_rate, model, grads);
            }

            global_step += 1;
            if global_step == 1 || global_step % config.log_every == 0 {
                println!(
                    "epoch={epoch} step={global_step} train_loss={loss_value:.6} train_ctc_loss={ctc_value:.6} train_ce_loss={ce_value:.6} elapsed_sec={:.1}",
                    started.elapsed().as_secs_f64()
                );
            }

            if let (Some(_), Some(every)) = (&config.val_manifest, config.validate_every_steps) {
                if every > 0 && global_step % every == 0 {
                    let val = evaluate_paraformer_model(&model, config, device, &decoder)?;
                    println!(
                        "{}",
                        format_validation_summary(epoch, Some(global_step), "val_loss", &val)
                    );
                    last_val_loss = Some(val.loss);
                    last_val_cer = val.cer;
                    last_val_wer = val.wer;
                    save_training_checkpoint::<B, ParaformerV2<B>, _>(
                        &model,
                        &optimizer,
                        config,
                        epoch,
                        false,
                        global_step,
                        last_train_loss,
                        last_val_loss,
                        last_val_cer,
                        last_val_wer,
                    )?;
                }
            }
        }

        if config.val_manifest.is_some() {
            let val = evaluate_paraformer_model(&model, config, device, &decoder)?;
            println!(
                "{}",
                format_validation_summary(epoch, None, "val_loss", &val)
            );
            last_val_loss = Some(val.loss);
            last_val_cer = val.cer;
            last_val_wer = val.wer;
        }
        save_training_checkpoint::<B, ParaformerV2<B>, _>(
            &model,
            &optimizer,
            config,
            epoch,
            true,
            global_step,
            last_train_loss,
            last_val_loss,
            last_val_cer,
            last_val_wer,
        )?;
    }

    Ok(TrainSummary {
        epochs: config.epochs,
        steps: global_step,
        last_train_loss,
        last_val_loss,
        last_val_cer,
        last_val_wer,
    })
}

fn evaluate_paraformer_model<B>(
    model: &ParaformerV2<B>,
    config: &BurnTrainConfig,
    device: &B::Device,
    decoder: &ValidationDecoder,
) -> Result<ValidationSummary>
where
    B: AutodiffBackend,
{
    let mut total = 0.0f64;
    let mut count = 0usize;
    let mut metrics = ValidationMetrics::default();
    let val_manifest = config
        .val_manifest
        .as_ref()
        .ok_or_else(|| anyhow!("validation requested without val_manifest"))?;
    let mut batches = StreamingBatchLoader::new(
        val_manifest.clone(),
        config.batch_size,
        config.adaptive_batch,
        config.sort_by_length_desc,
        config.sort_buffer_size,
        config.input_dim,
        config.max_val_samples,
    )?;
    while let Some(batch) = batches.next_batch()? {
        let output = paraformer_loss_for_batch(model, &batch, config, device);
        total += f64::from(scalar_value(output.loss)?);
        let predictions = decode_validation_batch(
            output.output.ctc_log_probs,
            &output.output.encoder_lengths,
            config.blank_id,
            decoder,
        )?;
        metrics.update(&batch, &predictions, decoder);
        count += 1;
    }
    if count == 0 {
        bail!("validation manifest is empty");
    }
    Ok(metrics.summary((total / count as f64) as f32))
}

fn train_enhanced_paraformer_model<B>(
    mut model: EnhancedParaformerV2<B>,
    config: &BurnTrainConfig,
    device: &B::Device,
) -> Result<TrainSummary>
where
    B: AutodiffBackend,
{
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    let resume = load_resume_checkpoint(config)?;
    model = load_model_checkpoint(model, &resume, device)?;
    optimizer =
        load_optimizer_checkpoint::<B, EnhancedParaformerV2<B>, _>(optimizer, &resume, device)?;
    let mut global_step = resume
        .as_ref()
        .map_or(0, |checkpoint| checkpoint.global_step);
    let mut last_train_loss = resume
        .as_ref()
        .and_then(|checkpoint| checkpoint.last_train_loss);
    let mut last_val_loss = resume
        .as_ref()
        .and_then(|checkpoint| checkpoint.last_val_loss);
    let mut last_val_cer = resume
        .as_ref()
        .and_then(|checkpoint| checkpoint.last_val_cer);
    let mut last_val_wer = resume
        .as_ref()
        .and_then(|checkpoint| checkpoint.last_val_wer);
    let start_epoch = resume_start_epoch(&resume);
    let decoder = ValidationDecoder::from_config(config)?;
    let started = Instant::now();

    for epoch in start_epoch..=config.epochs {
        let mut train_batches = StreamingBatchLoader::new(
            config.train_manifest.clone(),
            config.batch_size,
            config.adaptive_batch,
            config.sort_by_length_desc,
            config.sort_buffer_size,
            config.input_dim,
            config.max_train_samples,
        )?;
        while let Some(batch) = train_batches.next_batch()? {
            let loss_output = enhanced_paraformer_loss_for_batch(&model, &batch, config, device);
            let loss_value = scalar_value(loss_output.loss.clone())?;
            let ctc_value = scalar_value(loss_output.ctc_loss.clone())?;
            let shallow_value = scalar_value(loss_output.shallow_ctc_loss.clone())?;
            let ce_value = scalar_value(loss_output.ce_loss.clone())?;
            let boundary_value = scalar_value(loss_output.boundary_loss.clone())?;
            last_train_loss = Some(loss_value);

            if !config.dry_run {
                let grads = GradientsParams::from_grads(loss_output.loss.backward(), &model);
                model = optimizer.step(config.learning_rate, model, grads);
            }

            global_step += 1;
            if global_step == 1 || global_step % config.log_every == 0 {
                println!(
                    "epoch={epoch} step={global_step} train_loss={loss_value:.6} train_ctc_loss={ctc_value:.6} train_shallow_ctc_loss={shallow_value:.6} train_ce_loss={ce_value:.6} train_boundary_loss={boundary_value:.6} elapsed_sec={:.1}",
                    started.elapsed().as_secs_f64()
                );
            }

            if let (Some(_), Some(every)) = (&config.val_manifest, config.validate_every_steps) {
                if every > 0 && global_step % every == 0 {
                    let val = evaluate_enhanced_paraformer_model(&model, config, device, &decoder)?;
                    println!(
                        "{}",
                        format_validation_summary(epoch, Some(global_step), "val_loss", &val)
                    );
                    last_val_loss = Some(val.loss);
                    last_val_cer = val.cer;
                    last_val_wer = val.wer;
                    save_training_checkpoint::<B, EnhancedParaformerV2<B>, _>(
                        &model,
                        &optimizer,
                        config,
                        epoch,
                        false,
                        global_step,
                        last_train_loss,
                        last_val_loss,
                        last_val_cer,
                        last_val_wer,
                    )?;
                }
            }
        }

        if config.val_manifest.is_some() {
            let val = evaluate_enhanced_paraformer_model(&model, config, device, &decoder)?;
            println!(
                "{}",
                format_validation_summary(epoch, None, "val_loss", &val)
            );
            last_val_loss = Some(val.loss);
            last_val_cer = val.cer;
            last_val_wer = val.wer;
        }
        save_training_checkpoint::<B, EnhancedParaformerV2<B>, _>(
            &model,
            &optimizer,
            config,
            epoch,
            true,
            global_step,
            last_train_loss,
            last_val_loss,
            last_val_cer,
            last_val_wer,
        )?;
    }

    Ok(TrainSummary {
        epochs: config.epochs,
        steps: global_step,
        last_train_loss,
        last_val_loss,
        last_val_cer,
        last_val_wer,
    })
}

fn evaluate_enhanced_paraformer_model<B>(
    model: &EnhancedParaformerV2<B>,
    config: &BurnTrainConfig,
    device: &B::Device,
    decoder: &ValidationDecoder,
) -> Result<ValidationSummary>
where
    B: AutodiffBackend,
{
    let mut total = 0.0f64;
    let mut count = 0usize;
    let mut metrics = ValidationMetrics::default();
    let val_manifest = config
        .val_manifest
        .as_ref()
        .ok_or_else(|| anyhow!("validation requested without val_manifest"))?;
    let mut batches = StreamingBatchLoader::new(
        val_manifest.clone(),
        config.batch_size,
        config.adaptive_batch,
        config.sort_by_length_desc,
        config.sort_buffer_size,
        config.input_dim,
        config.max_val_samples,
    )?;
    while let Some(batch) = batches.next_batch()? {
        let output = enhanced_paraformer_loss_for_batch(model, &batch, config, device);
        total += f64::from(scalar_value(output.loss)?);
        let predictions = decode_validation_batch(
            output.output.ctc_log_probs,
            &output.output.encoder_lengths,
            config.blank_id,
            decoder,
        )?;
        metrics.update(&batch, &predictions, decoder);
        count += 1;
    }
    if count == 0 {
        bail!("validation manifest is empty");
    }
    Ok(metrics.summary((total / count as f64) as f32))
}

fn paraformer_loss_for_batch<B>(
    model: &ParaformerV2<B>,
    batch: &TrainBatch,
    config: &BurnTrainConfig,
    device: &B::Device,
) -> crate::paraformer::ParaformerLossOutput<B>
where
    B: AutodiffBackend,
{
    let features = Tensor::<B, 3>::from_data(
        TensorData::new(
            batch.features.clone(),
            [batch.batch_size, batch.max_frames, batch.feature_dim],
        ),
        device,
    );
    let targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            batch.targets.clone(),
            [batch.batch_size, batch.max_target_len],
        ),
        device,
    );
    model.loss(
        features,
        batch.feature_lengths.clone(),
        targets,
        &batch.targets,
        batch.target_lengths.clone(),
        config.blank_id,
        config.paraformer_alignment_mode,
    )
}

fn enhanced_paraformer_loss_for_batch<B>(
    model: &EnhancedParaformerV2<B>,
    batch: &TrainBatch,
    config: &BurnTrainConfig,
    device: &B::Device,
) -> crate::paraformer::EnhancedParaformerLossOutput<B>
where
    B: AutodiffBackend,
{
    let features = Tensor::<B, 3>::from_data(
        TensorData::new(
            batch.features.clone(),
            [batch.batch_size, batch.max_frames, batch.feature_dim],
        ),
        device,
    );
    let targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            batch.targets.clone(),
            [batch.batch_size, batch.max_target_len],
        ),
        device,
    );
    model.loss(
        features,
        batch.feature_lengths.clone(),
        targets,
        &batch.targets,
        batch.target_lengths.clone(),
        config.blank_id,
        config.paraformer_alignment_mode,
    )
}

fn ctc_loss_for_batch<B, M>(
    model: &M,
    batch: &TrainBatch,
    blank_id: usize,
    device: &B::Device,
) -> Tensor<B, 1>
where
    B: AutodiffBackend,
    M: TrainableCtc<B>,
{
    let (logits_or_log_probs, output_lengths) = ctc_logits_for_batch(model, batch, device);
    ctc_loss_from_logits(logits_or_log_probs, output_lengths, batch, blank_id, device)
}

fn ctc_logits_for_batch<B, M>(
    model: &M,
    batch: &TrainBatch,
    device: &B::Device,
) -> (Tensor<B, 3>, Vec<usize>)
where
    B: AutodiffBackend,
    M: TrainableCtc<B>,
{
    let features = batch_features_tensor(batch, device);
    model.ctc_logits(features, batch.feature_lengths.clone())
}

fn batch_features_tensor<B: Backend>(batch: &TrainBatch, device: &B::Device) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(
        TensorData::new(
            batch.features.clone(),
            [batch.batch_size, batch.max_frames, batch.feature_dim],
        ),
        device,
    )
}

fn ctc_loss_from_logits<B: AutodiffBackend>(
    logits_or_log_probs: Tensor<B, 3>,
    output_lengths: Vec<usize>,
    batch: &TrainBatch,
    blank_id: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let log_probs = log_softmax(logits_or_log_probs, 2).swap_dims(0, 1);
    let targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            batch.targets.clone(),
            [batch.batch_size, batch.max_target_len],
        ),
        device,
    );
    let input_lengths = Tensor::<B, 1, Int>::from_data(
        TensorData::new(to_i64(output_lengths), [batch.batch_size]),
        device,
    );
    let target_lengths = Tensor::<B, 1, Int>::from_data(
        TensorData::new(to_i64(batch.target_lengths.clone()), [batch.batch_size]),
        device,
    );
    CTCLossConfig::new()
        .with_blank(blank_id)
        .with_zero_infinity(true)
        .init()
        .forward_with_reduction(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            Reduction::Mean,
        )
}

#[derive(Default)]
struct ValidationMetrics {
    token_stats: EditStats,
    char_stats: EditStats,
    word_stats: EditStats,
    decoded_samples: usize,
}

impl ValidationMetrics {
    fn update(
        &mut self,
        batch: &TrainBatch,
        predictions: &[Vec<u32>],
        decoder: &ValidationDecoder,
    ) {
        for sample_index in 0..batch.batch_size {
            let reference_tokens = target_tokens_for_batch_item(batch, sample_index);
            let predicted_tokens = predictions
                .get(sample_index)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>();
            self.token_stats.add(
                edit_distance(&predicted_tokens, &reference_tokens),
                reference_tokens.len(),
            );

            let predicted_u32 = predicted_tokens
                .iter()
                .filter_map(|token| u32::try_from(*token).ok())
                .collect::<Vec<_>>();
            let predicted_text = decoder.decode_tokens(&predicted_u32);
            let reference_text = decoder.reference_text(
                batch.reference_texts[sample_index].as_ref(),
                &reference_tokens,
            );
            let predicted_chars = predicted_text.chars().collect::<Vec<_>>();
            let reference_chars = reference_text.chars().collect::<Vec<_>>();
            self.char_stats.add(
                edit_distance(&predicted_chars, &reference_chars),
                reference_chars.len(),
            );
            let predicted_words = predicted_text.split_whitespace().collect::<Vec<_>>();
            let reference_words = reference_text.split_whitespace().collect::<Vec<_>>();
            self.word_stats.add(
                edit_distance(&predicted_words, &reference_words),
                reference_words.len(),
            );
            self.decoded_samples += 1;
        }
    }

    fn summary(self, loss: f32) -> ValidationSummary {
        ValidationSummary {
            loss,
            token_error_rate: self.token_stats.rate(),
            cer: self.char_stats.rate(),
            wer: self.word_stats.rate(),
            decoded_samples: self.decoded_samples,
        }
    }
}

fn decode_validation_batch<B: Backend>(
    logits_or_log_probs: Tensor<B, 3>,
    output_lengths: &[usize],
    blank_id: usize,
    decoder: &ValidationDecoder,
) -> Result<Vec<Vec<u32>>> {
    let [batch_size, frames, vocab_size] = logits_or_log_probs.dims();
    if vocab_size == 0 {
        bail!("cannot decode logits with empty vocab dimension");
    }
    if blank_id >= vocab_size {
        bail!("blank_id {blank_id} is outside vocab size {vocab_size}");
    }
    let values = logits_or_log_probs
        .into_data()
        .to_vec::<f32>()
        .context("failed to read validation logits")?;
    let mut decoded = Vec::with_capacity(batch_size);
    for batch_index in 0..batch_size {
        let length = output_lengths
            .get(batch_index)
            .copied()
            .unwrap_or(frames)
            .min(frames);
        let start = batch_index * frames * vocab_size;
        let end = start + length * vocab_size;
        let tokens = decoder.decode_best(&values[start..end], length, vocab_size, blank_id)?;
        decoded.push(tokens);
    }
    Ok(decoded)
}

fn greedy_decode_frames(
    frame_logits: &[f32],
    frames: usize,
    vocab_size: usize,
    blank_id: usize,
) -> Result<Vec<u32>> {
    if frame_logits.len() != frames * vocab_size {
        bail!(
            "frame logits shape {frames}x{vocab_size} implies {} values, got {}",
            frames * vocab_size,
            frame_logits.len()
        );
    }
    let mut previous = None;
    let mut tokens = Vec::new();
    for frame_values in frame_logits.chunks_exact(vocab_size) {
        let token = frame_values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .unwrap_or(blank_id);
        if token != blank_id && previous != Some(token) {
            tokens.push(u32::try_from(token).context("token id does not fit u32")?);
        }
        previous = Some(token);
    }
    Ok(tokens)
}

fn target_tokens_for_batch_item(batch: &TrainBatch, sample_index: usize) -> Vec<i64> {
    let length = batch.target_lengths[sample_index];
    let start = sample_index * batch.max_target_len;
    batch.targets[start..start + length].to_vec()
}

fn edit_distance<T: Eq>(hypothesis: &[T], reference: &[T]) -> usize {
    if reference.is_empty() {
        return hypothesis.len();
    }
    if hypothesis.is_empty() {
        return reference.len();
    }
    let mut previous = (0..=reference.len()).collect::<Vec<_>>();
    let mut current = vec![0; reference.len() + 1];
    for (hyp_index, hyp_item) in hypothesis.iter().enumerate() {
        current[0] = hyp_index + 1;
        for (ref_index, ref_item) in reference.iter().enumerate() {
            let substitution = previous[ref_index] + usize::from(hyp_item != ref_item);
            let insertion = current[ref_index] + 1;
            let deletion = previous[ref_index + 1] + 1;
            current[ref_index + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[reference.len()]
}

fn format_validation_summary(
    epoch: usize,
    step: Option<usize>,
    loss_name: &str,
    summary: &ValidationSummary,
) -> String {
    let mut parts = vec![format!("epoch={epoch}")];
    if let Some(step) = step {
        parts.push(format!("step={step}"));
    }
    parts.push(format!("{loss_name}={:.6}", summary.loss));
    if let Some(cer) = summary.cer {
        parts.push(format!("val_cer={cer:.6}"));
    }
    if let Some(wer) = summary.wer {
        parts.push(format!("val_wer={wer:.6}"));
    }
    if let Some(token_error_rate) = summary.token_error_rate {
        parts.push(format!("val_token_error_rate={token_error_rate:.6}"));
    }
    parts.push(format!("decoded_samples={}", summary.decoded_samples));
    parts.join(" ")
}

fn make_batch(records: &[FeatureRecord], expected_dim: usize) -> Result<TrainBatch> {
    if records.is_empty() {
        bail!("cannot build an empty batch");
    }
    let batch_size = records.len();
    let max_frames = records.iter().map(|record| record.rows).max().unwrap_or(0);
    let max_target_len = records
        .iter()
        .map(|record| record.tokens.len())
        .max()
        .unwrap_or(0);
    if max_frames == 0 || max_target_len == 0 {
        bail!("batch contains empty features or targets");
    }

    let mut features = vec![0.0; batch_size * max_frames * expected_dim];
    let mut targets = vec![0i64; batch_size * max_target_len];
    let mut feature_lengths = Vec::with_capacity(batch_size);
    let mut target_lengths = Vec::with_capacity(batch_size);
    let mut reference_texts = Vec::with_capacity(batch_size);

    for (batch_index, record) in records.iter().enumerate() {
        if record.cols != expected_dim {
            bail!(
                "record '{}' has feature dim {}, expected {}",
                record.id,
                record.cols,
                expected_dim
            );
        }
        feature_lengths.push(record.rows);
        target_lengths.push(record.tokens.len());
        reference_texts.push(record.text.clone());
        for row in 0..record.rows {
            let src = row * expected_dim;
            let dst = (batch_index * max_frames + row) * expected_dim;
            features[dst..dst + expected_dim]
                .copy_from_slice(&record.features[src..src + expected_dim]);
        }
        let dst = batch_index * max_target_len;
        targets[dst..dst + record.tokens.len()].copy_from_slice(&record.tokens);
    }

    Ok(TrainBatch {
        features,
        batch_size,
        max_frames,
        feature_dim: expected_dim,
        feature_lengths,
        targets,
        max_target_len,
        target_lengths,
        reference_texts,
    })
}

pub fn load_manifest(path: &Path, limit: Option<usize>) -> Result<Vec<FeatureRecord>> {
    let files = manifest_files(path)?;
    let mut records = Vec::new();
    for file in files {
        let remaining = limit.map(|limit| limit.saturating_sub(records.len()));
        if remaining == Some(0) {
            break;
        }
        records.extend(load_manifest_file(&file, remaining)?);
    }
    if records.is_empty() {
        bail!("manifest {} contains no records", path.display());
    }
    Ok(records)
}

pub struct StreamingBatchLoader {
    files: Vec<PathBuf>,
    file_index: usize,
    current: Option<CurrentManifestFile>,
    batch_size: usize,
    adaptive_batch: Option<AdaptiveBatchConfig>,
    sort_by_length_desc: bool,
    sort_buffer_size: usize,
    expected_dim: usize,
    yielded: usize,
    limit: Option<usize>,
    pending: Option<FeatureRecord>,
    sort_buffer: Vec<FeatureRecordMetadata>,
}

struct CurrentManifestFile {
    path: PathBuf,
    base_dir: PathBuf,
    reader: BufReader<fs::File>,
    line_number: usize,
}

#[derive(Clone, Debug)]
struct FeatureRecordMetadata {
    manifest_path: PathBuf,
    base_dir: PathBuf,
    byte_offset: u64,
    line_number: usize,
    id: String,
    rows: usize,
}

impl StreamingBatchLoader {
    pub fn new(
        manifest: PathBuf,
        batch_size: usize,
        adaptive_batch: Option<AdaptiveBatchConfig>,
        sort_by_length_desc: bool,
        sort_buffer_size: usize,
        expected_dim: usize,
        limit: Option<usize>,
    ) -> Result<Self> {
        if batch_size == 0 {
            bail!("batch_size must be > 0");
        }
        if adaptive_batch.is_some_and(|config| config.budget == 0) {
            bail!("adaptive batch budget must be > 0");
        }
        if sort_by_length_desc && sort_buffer_size == 0 {
            bail!("sort_buffer_size must be > 0 when length sorting is enabled");
        }
        Ok(Self {
            files: manifest_files(&manifest)?,
            file_index: 0,
            current: None,
            batch_size,
            adaptive_batch,
            sort_by_length_desc,
            sort_buffer_size,
            expected_dim,
            yielded: 0,
            limit,
            pending: None,
            sort_buffer: Vec::new(),
        })
    }

    pub fn next_batch(&mut self) -> Result<Option<TrainBatch>> {
        let mut records = Vec::with_capacity(self.batch_size);
        while self.can_add_more_samples(records.len()) {
            if self.limit.is_some_and(|limit| self.yielded >= limit) {
                break;
            }
            match self.next_record()? {
                Some(record) => {
                    if !records.is_empty() && !self.fits_adaptive_budget(&records, &record) {
                        self.pending = Some(record);
                        break;
                    }
                    self.yielded += 1;
                    records.push(record);
                }
                None => break,
            }
        }

        if records.is_empty() {
            Ok(None)
        } else {
            make_batch(&records, self.expected_dim).map(Some)
        }
    }

    fn can_add_more_samples(&self, current_len: usize) -> bool {
        if let Some(config) = self.adaptive_batch {
            current_len < config.max_samples.unwrap_or(self.batch_size)
        } else {
            current_len < self.batch_size
        }
    }

    fn fits_adaptive_budget(&self, records: &[FeatureRecord], candidate: &FeatureRecord) -> bool {
        let Some(config) = self.adaptive_batch else {
            return true;
        };
        adaptive_cost(
            records.iter().chain(std::iter::once(candidate)),
            config.unit,
            self.expected_dim,
        ) <= config.budget
    }

    fn next_record(&mut self) -> Result<Option<FeatureRecord>> {
        if self.pending.is_some() {
            return Ok(self.pending.take());
        }
        if self.sort_by_length_desc {
            return self.next_sorted_record();
        }
        self.next_raw_record()
    }

    fn next_sorted_record(&mut self) -> Result<Option<FeatureRecord>> {
        if self.sort_buffer.is_empty() {
            let remaining = self
                .limit
                .map(|limit| limit.saturating_sub(self.yielded))
                .unwrap_or(self.sort_buffer_size);
            let target = self.sort_buffer_size.min(remaining);
            for _ in 0..target {
                let Some(metadata) = self.next_raw_metadata()? else {
                    break;
                };
                self.sort_buffer.push(metadata);
            }
            self.sort_buffer.sort_by(|left, right| {
                right
                    .rows
                    .cmp(&left.rows)
                    .then_with(|| left.id.cmp(&right.id))
            });
        }
        if self.sort_buffer.is_empty() {
            Ok(None)
        } else {
            self.sort_buffer.remove(0).load_record().map(Some)
        }
    }

    fn next_raw_record(&mut self) -> Result<Option<FeatureRecord>> {
        self.next_raw_line()?
            .map(|raw| raw.parse_record())
            .transpose()
    }

    fn next_raw_metadata(&mut self) -> Result<Option<FeatureRecordMetadata>> {
        self.next_raw_line()?
            .map(|raw| raw.parse_metadata())
            .transpose()
    }

    fn next_raw_line(&mut self) -> Result<Option<RawManifestLine>> {
        loop {
            if self.current.is_none() {
                if self.file_index >= self.files.len() {
                    return Ok(None);
                }
                let path = self.files[self.file_index].clone();
                self.file_index += 1;
                self.current = Some(CurrentManifestFile::open(path)?);
            }

            let current = self.current.as_mut().expect("manifest file must be open");
            let mut line = String::new();
            let byte_offset = current
                .reader
                .stream_position()
                .with_context(|| format!("failed to seek {}", current.path.display()))?;
            let bytes = current
                .reader
                .read_line(&mut line)
                .with_context(|| format!("failed reading {}", current.path.display()))?;
            if bytes == 0 {
                self.current = None;
                continue;
            }
            current.line_number += 1;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            return Ok(Some(RawManifestLine {
                manifest_path: current.path.clone(),
                base_dir: current.base_dir.clone(),
                byte_offset,
                line_number: current.line_number,
                line: line.to_string(),
            }));
        }
    }
}

struct RawManifestLine {
    manifest_path: PathBuf,
    base_dir: PathBuf,
    byte_offset: u64,
    line_number: usize,
    line: String,
}

impl RawManifestLine {
    fn parse_record(&self) -> Result<FeatureRecord> {
        if self.line.starts_with('{') {
            parse_json_record(&self.line, &self.base_dir, self.line_number)
        } else {
            parse_tsv_record(&self.line, &self.base_dir, self.line_number)
        }
    }

    fn parse_metadata(&self) -> Result<FeatureRecordMetadata> {
        let (id, rows) = if self.line.starts_with('{') {
            parse_json_record_metadata(&self.line, self.line_number)?
        } else {
            parse_tsv_record_metadata(&self.line, self.line_number)?
        };
        Ok(FeatureRecordMetadata {
            manifest_path: self.manifest_path.clone(),
            base_dir: self.base_dir.clone(),
            byte_offset: self.byte_offset,
            line_number: self.line_number,
            id,
            rows,
        })
    }
}

impl FeatureRecordMetadata {
    fn load_record(&self) -> Result<FeatureRecord> {
        let mut reader =
            BufReader::new(fs::File::open(&self.manifest_path).with_context(|| {
                format!("failed to open manifest {}", self.manifest_path.display())
            })?);
        reader
            .seek(SeekFrom::Start(self.byte_offset))
            .with_context(|| format!("failed to seek {}", self.manifest_path.display()))?;
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .with_context(|| format!("failed reading {}", self.manifest_path.display()))?;
        let line = line.trim();
        if line.starts_with('{') {
            parse_json_record(line, &self.base_dir, self.line_number)
        } else {
            parse_tsv_record(line, &self.base_dir, self.line_number)
        }
    }
}

impl CurrentManifestFile {
    fn open(path: PathBuf) -> Result<Self> {
        let file = fs::File::open(&path)
            .with_context(|| format!("failed to open manifest {}", path.display()))?;
        let base_dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        Ok(Self {
            path,
            base_dir,
            reader: BufReader::new(file),
            line_number: 0,
        })
    }
}

fn manifest_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_dir() {
        let mut files = fs::read_dir(path)
            .with_context(|| format!("failed to read manifest directory {}", path.display()))?
            .map(|entry| {
                entry
                    .map(|entry| entry.path())
                    .with_context(|| format!("failed to read entry in {}", path.display()))
            })
            .collect::<Result<Vec<_>>>()?;
        files.retain(|path| is_manifest_file(path));
        files.sort();
        if files.is_empty() {
            bail!(
                "manifest directory {} contains no .jsonl/.json files",
                path.display()
            );
        }
        return Ok(files);
    }
    Ok(vec![path.to_path_buf()])
}

fn is_manifest_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| matches!(extension, "jsonl" | "json"))
}

fn adaptive_cost<'a>(
    records: impl Iterator<Item = &'a FeatureRecord>,
    unit: AdaptiveBatchUnit,
    expected_dim: usize,
) -> usize {
    let mut samples = 0usize;
    let mut frames = 0usize;
    let mut max_frames = 0usize;
    for record in records {
        samples += 1;
        frames += record.rows;
        max_frames = max_frames.max(record.rows);
    }
    match unit {
        AdaptiveBatchUnit::Samples => samples,
        AdaptiveBatchUnit::Frames => frames,
        AdaptiveBatchUnit::PaddedFrames => samples * max_frames,
        AdaptiveBatchUnit::FeatureValues => samples * max_frames * expected_dim,
    }
}

fn load_manifest_file(path: &Path, limit: Option<usize>) -> Result<Vec<FeatureRecord>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read manifest {}", path.display()))?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut records = Vec::new();

    for (line_index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let record = if line.starts_with('{') {
            parse_json_record(line, base_dir, line_index + 1)?
        } else {
            parse_tsv_record(line, base_dir, line_index + 1)?
        };
        records.push(record);
        if limit.is_some_and(|limit| records.len() >= limit) {
            break;
        }
    }

    if records.is_empty() {
        bail!("manifest {} contains no records", path.display());
    }
    Ok(records)
}

fn parse_json_record(line: &str, base_dir: &Path, line_number: usize) -> Result<FeatureRecord> {
    let value: Value = serde_json::from_str(line)
        .with_context(|| format!("invalid JSON manifest line {line_number}"))?;
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("line-{line_number}"));
    let tokens = parse_tokens_value(
        value
            .get("tokens")
            .or_else(|| value.get("target"))
            .or_else(|| value.get("targets"))
            .ok_or_else(|| anyhow!("record '{id}' is missing tokens"))?,
    )?;
    let text = value
        .get("text")
        .or_else(|| value.get("transcript"))
        .or_else(|| value.get("sentence"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);

    if let Some(features) = value.get("features") {
        let rows = value
            .get("rows")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let cols = value
            .get("cols")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let (features, rows, cols) = parse_inline_features(features, rows, cols, &id)?;
        return Ok(FeatureRecord {
            id,
            rows,
            cols,
            features,
            tokens,
            text,
        });
    }

    let feature_path = value
        .get("features_path")
        .or_else(|| value.get("feature_path"))
        .or_else(|| value.get("path"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("record '{id}' is missing features or features_path"))?;
    let rows = value
        .get("rows")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("record '{id}' with feature file must include rows"))?
        as usize;
    let cols = value
        .get("cols")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("record '{id}' with feature file must include cols"))?
        as usize;
    let features = read_feature_file(&resolve_path(base_dir, feature_path), rows, cols)?;
    Ok(FeatureRecord {
        id,
        rows,
        cols,
        features,
        tokens,
        text,
    })
}

fn parse_tsv_record(line: &str, base_dir: &Path, line_number: usize) -> Result<FeatureRecord> {
    let parts = line.split('\t').collect::<Vec<_>>();
    if parts.len() < 4 {
        bail!(
            "TSV manifest line {line_number} must have: features_path<TAB>rows<TAB>cols<TAB>tokens"
        );
    }
    let rows = parts[1]
        .parse::<usize>()
        .with_context(|| format!("invalid rows on line {line_number}"))?;
    let cols = parts[2]
        .parse::<usize>()
        .with_context(|| format!("invalid cols on line {line_number}"))?;
    let tokens = parse_token_string(parts[3])?;
    let features = read_feature_file(&resolve_path(base_dir, parts[0]), rows, cols)?;
    Ok(FeatureRecord {
        id: format!("line-{line_number}"),
        rows,
        cols,
        features,
        tokens,
        text: parts.get(4).map(|value| (*value).to_string()),
    })
}

fn parse_json_record_metadata(line: &str, line_number: usize) -> Result<(String, usize)> {
    let value: Value = serde_json::from_str(line)
        .with_context(|| format!("invalid JSON manifest line {line_number}"))?;
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("line-{line_number}"));

    if let Some(features) = value.get("features") {
        let rows = value
            .get("rows")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let cols = value
            .get("cols")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let (rows, _cols) = inline_feature_shape(features, rows, cols, &id)?;
        return Ok((id, rows));
    }

    let rows = value
        .get("rows")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("record '{id}' with feature file must include rows"))?
        as usize;
    Ok((id, rows))
}

fn parse_tsv_record_metadata(line: &str, line_number: usize) -> Result<(String, usize)> {
    let parts = line.split('\t').collect::<Vec<_>>();
    if parts.len() < 4 {
        bail!(
            "TSV manifest line {line_number} must have: features_path<TAB>rows<TAB>cols<TAB>tokens"
        );
    }
    let rows = parts[1]
        .parse::<usize>()
        .with_context(|| format!("invalid rows on line {line_number}"))?;
    Ok((format!("line-{line_number}"), rows))
}

fn parse_inline_features(
    value: &Value,
    rows: Option<usize>,
    cols: Option<usize>,
    id: &str,
) -> Result<(Vec<f32>, usize, usize)> {
    if let Some(outer) = value.as_array().filter(|items| {
        items
            .first()
            .is_some_and(|first| matches!(first, Value::Array(_)))
    }) {
        let rows = outer.len();
        let cols = outer
            .first()
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        let mut features = Vec::with_capacity(rows * cols);
        for row in outer {
            let row = row
                .as_array()
                .ok_or_else(|| anyhow!("record '{id}' has a non-array feature row"))?;
            if row.len() != cols {
                bail!("record '{id}' has ragged inline features");
            }
            for value in row {
                features.push(
                    value
                        .as_f64()
                        .ok_or_else(|| anyhow!("record '{id}' has a non-number feature"))?
                        as f32,
                );
            }
        }
        return Ok((features, rows, cols));
    }

    let rows = rows.ok_or_else(|| anyhow!("record '{id}' inline flat features require rows"))?;
    let cols = cols.ok_or_else(|| anyhow!("record '{id}' inline flat features require cols"))?;
    let features = value
        .as_array()
        .ok_or_else(|| anyhow!("record '{id}' features must be an array"))?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or_else(|| anyhow!("record '{id}' has a non-number feature"))
        })
        .collect::<Result<Vec<_>>>()?;
    if features.len() != rows * cols {
        bail!(
            "record '{id}' feature shape {rows}x{cols} implies {} values, got {}",
            rows * cols,
            features.len()
        );
    }
    Ok((features, rows, cols))
}

fn inline_feature_shape(
    value: &Value,
    rows: Option<usize>,
    cols: Option<usize>,
    id: &str,
) -> Result<(usize, usize)> {
    if let Some(outer) = value.as_array().filter(|items| {
        items
            .first()
            .is_some_and(|first| matches!(first, Value::Array(_)))
    }) {
        let rows = outer.len();
        let cols = outer
            .first()
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        for row in outer {
            let row = row
                .as_array()
                .ok_or_else(|| anyhow!("record '{id}' has a non-array feature row"))?;
            if row.len() != cols {
                bail!("record '{id}' has ragged inline features");
            }
        }
        return Ok((rows, cols));
    }

    let rows = rows.ok_or_else(|| anyhow!("record '{id}' inline flat features require rows"))?;
    let cols = cols.ok_or_else(|| anyhow!("record '{id}' inline flat features require cols"))?;
    let len = value
        .as_array()
        .ok_or_else(|| anyhow!("record '{id}' features must be an array"))?
        .len();
    if len != rows * cols {
        bail!(
            "record '{id}' feature shape {rows}x{cols} implies {} values, got {}",
            rows * cols,
            len
        );
    }
    Ok((rows, cols))
}

fn parse_tokens_value(value: &Value) -> Result<Vec<i64>> {
    if let Some(text) = value.as_str() {
        return parse_token_string(text);
    }
    value
        .as_array()
        .ok_or_else(|| anyhow!("tokens must be an array or string"))?
        .iter()
        .map(|value| {
            value
                .as_i64()
                .ok_or_else(|| anyhow!("token array contains a non-integer"))
        })
        .collect()
}

fn parse_token_string(value: &str) -> Result<Vec<i64>> {
    let tokens = value
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<i64>()
                .with_context(|| format!("invalid token id '{part}'"))
        })
        .collect::<Result<Vec<_>>>()?;
    if tokens.is_empty() {
        bail!("target token sequence is empty");
    }
    Ok(tokens)
}

fn read_feature_file(path: &Path, rows: usize, cols: usize) -> Result<Vec<f32>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read feature file {}", path.display()))?;
    let values = text
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<f32>()
                .with_context(|| format!("invalid feature value '{part}' in {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    if values.len() != rows * cols {
        bail!(
            "feature file {} shape {rows}x{cols} implies {} values, got {}",
            path.display(),
            rows * cols,
            values.len()
        );
    }
    Ok(values)
}

fn resolve_path(base_dir: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

fn scalar_value<B: Backend>(tensor: Tensor<B, 1>) -> Result<f32> {
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .context("failed to read scalar loss")?;
    values
        .first()
        .copied()
        .ok_or_else(|| anyhow!("loss tensor was empty"))
}

fn to_i64(values: Vec<usize>) -> Vec<i64> {
    values.into_iter().map(|value| value as i64).collect()
}

fn validate_config(config: &BurnTrainConfig) -> Result<()> {
    if config.batch_size == 0 {
        bail!("batch_size must be > 0");
    }
    if let Some(adaptive_batch) = config.adaptive_batch {
        if adaptive_batch.budget == 0 {
            bail!("adaptive_batch.budget must be > 0");
        }
        if adaptive_batch.max_samples == Some(0) {
            bail!("adaptive_batch.max_samples must be > 0 when set");
        }
    }
    if config.sort_by_length_desc && config.sort_buffer_size == 0 {
        bail!("sort_buffer_size must be > 0 when length sorting is enabled");
    }
    if config.epochs == 0 {
        bail!("epochs must be > 0");
    }
    if config.input_dim == 0 || config.vocab_size == 0 {
        bail!("input_dim and vocab_size must be > 0");
    }
    if config.blank_id >= config.vocab_size {
        bail!("blank_id must be smaller than vocab_size");
    }
    if config.val_beam_width == 0 {
        bail!("val_beam_width must be > 0");
    }
    if config.val_n_best == 0 {
        bail!("val_n_best must be > 0");
    }
    if config.val_lm_path.is_some() && config.tokenizer_path.is_none() {
        bail!("val_lm_path requires tokenizer_path");
    }
    Ok(())
}

fn write_run_config(config: &BurnTrainConfig) -> Result<()> {
    let path = config.output_dir.join("training_config.json");
    fs::write(
        &path,
        serde_json::to_string_pretty(&run_config_json(config))?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn architecture_name(architecture: &TrainArchitecture) -> &'static str {
    match architecture {
        TrainArchitecture::Squeezeformer => "squeezeformer",
        TrainArchitecture::Zipformer => "zipformer",
        TrainArchitecture::Paraformer => "paraformer",
        TrainArchitecture::Wav2VecBert => "w2v_bert",
    }
}

fn adaptive_batch_json(config: &BurnTrainConfig) -> Option<Value> {
    config.adaptive_batch.map(|adaptive| {
        json!({
            "unit": match adaptive.unit {
                AdaptiveBatchUnit::Samples => "samples",
                AdaptiveBatchUnit::Frames => "frames",
                AdaptiveBatchUnit::PaddedFrames => "padded_frames",
                AdaptiveBatchUnit::FeatureValues => "feature_values",
            },
            "budget": adaptive.budget,
            "max_samples": adaptive.max_samples,
        })
    })
}

fn run_config_json(config: &BurnTrainConfig) -> Value {
    json!({
        "architecture": architecture_name(&config.architecture),
        "train_manifest": config.train_manifest,
        "val_manifest": config.val_manifest,
        "variant": config.variant,
        "input_dim": config.input_dim,
        "vocab_size": config.vocab_size,
        "blank_id": config.blank_id,
        "d_model": config.d_model,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "batch_size": config.batch_size,
        "adaptive_batch": adaptive_batch_json(config),
        "sort_by_length_desc": config.sort_by_length_desc,
        "sort_buffer_size": config.sort_buffer_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "tokenizer_path": config.tokenizer_path,
        "val_beam_width": config.val_beam_width,
        "val_n_best": config.val_n_best,
        "val_lm_path": config.val_lm_path,
        "val_lm_weight": config.val_lm_weight,
        "val_lm_word_bonus": config.val_lm_word_bonus,
        "val_lm_bos": config.val_lm_bos,
        "val_lm_eos": config.val_lm_eos,
        "paraformer_alignment_mode": match config.paraformer_alignment_mode {
            ParaformerAlignmentMode::Viterbi => "viterbi",
            ParaformerAlignmentMode::Uniform => "uniform",
            ParaformerAlignmentMode::Greedy => "greedy",
        },
        "paraformer_enhanced": config.paraformer_enhanced,
        "w2v_hf_model_dir": config.w2v_hf_model_dir,
        "w2v_hf_load_weights": config.w2v_hf_load_weights,
        "w2v_activation_checkpointing": config.w2v_activation_checkpointing,
        "backend": match config.backend {
            TrainBackendKind::Cpu => "cpu",
            TrainBackendKind::Cuda => "cuda",
            TrainBackendKind::Wgpu => "wgpu",
        },
        "device_index": config.device_index,
        "precision": match config.precision {
            TrainPrecision::F32 => "f32",
            TrainPrecision::F16 => "f16",
            TrainPrecision::Bf16 => "bf16",
        },
    })
}

fn checkpoint_dir(config: &BurnTrainConfig) -> PathBuf {
    config.output_dir.join("checkpoint_latest")
}

fn checkpoint_metadata_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("checkpoint.json")
    } else {
        path.to_path_buf()
    }
}

fn checkpoint_base_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(stem)
}

fn checkpoint_file_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.bin"))
}

fn save_training_checkpoint<B, M, O>(
    model: &M,
    optimizer: &O,
    config: &BurnTrainConfig,
    epoch: usize,
    epoch_complete: bool,
    global_step: usize,
    train_loss: Option<f32>,
    val_loss: Option<f32>,
    val_cer: Option<f32>,
    val_wer: Option<f32>,
) -> Result<()>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    let dir = checkpoint_dir(config);
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create checkpoint directory {}", dir.display()))?;
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(checkpoint_base_path(&dir, "model"), &recorder)
        .with_context(|| {
            format!(
                "failed to write {}",
                checkpoint_file_path(&dir, "model").display()
            )
        })?;
    recorder
        .record(
            optimizer.to_record(),
            checkpoint_base_path(&dir, "optimizer"),
        )
        .with_context(|| {
            format!(
                "failed to write {}",
                checkpoint_file_path(&dir, "optimizer").display()
            )
        })?;

    let metadata = json!({
        "epoch": epoch,
        "epoch_complete": epoch_complete,
        "global_step": global_step,
        "train_ctc_loss": train_loss,
        "val_ctc_loss": val_loss,
        "val_cer": val_cer,
        "val_wer": val_wer,
        "checkpoint_dir": dir,
        "model_path": checkpoint_file_path(&dir, "model"),
        "optimizer_path": checkpoint_file_path(&dir, "optimizer"),
        "training_config": run_config_json(config),
    });
    let path = checkpoint_metadata_path(&dir);
    fs::write(&path, serde_json::to_string_pretty(&metadata)?)
        .with_context(|| format!("failed to write {}", path.display()))?;

    let legacy_path = config.output_dir.join("checkpoint_latest.json");
    fs::write(&legacy_path, serde_json::to_string_pretty(&metadata)?)
        .with_context(|| format!("failed to write {}", legacy_path.display()))?;
    Ok(())
}

fn load_resume_checkpoint(config: &BurnTrainConfig) -> Result<Option<ResumeCheckpoint>> {
    let Some(path) = &config.resume_from else {
        return Ok(None);
    };
    let metadata_path = checkpoint_metadata_path(path);
    let metadata_text = fs::read_to_string(&metadata_path).with_context(|| {
        format!(
            "failed to read checkpoint metadata {}",
            metadata_path.display()
        )
    })?;
    let metadata: Value = serde_json::from_str(&metadata_text).with_context(|| {
        format!(
            "failed to parse checkpoint metadata {}",
            metadata_path.display()
        )
    })?;
    let saved_config = metadata
        .get("training_config")
        .context("checkpoint metadata does not contain training_config")?;
    validate_resume_config(config, saved_config)?;
    let dir = if path.is_dir() {
        path.clone()
    } else {
        metadata_path
            .parent()
            .map(Path::to_path_buf)
            .context("checkpoint metadata path has no parent directory")?
    };
    Ok(Some(ResumeCheckpoint {
        dir,
        epoch: metadata
            .get("epoch")
            .and_then(Value::as_u64)
            .context("checkpoint metadata does not contain numeric epoch")? as usize,
        epoch_complete: metadata
            .get("epoch_complete")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        global_step: metadata
            .get("global_step")
            .and_then(Value::as_u64)
            .context("checkpoint metadata does not contain numeric global_step")?
            as usize,
        last_train_loss: optional_f32(&metadata, "train_ctc_loss"),
        last_val_loss: optional_f32(&metadata, "val_ctc_loss"),
        last_val_cer: optional_f32(&metadata, "val_cer"),
        last_val_wer: optional_f32(&metadata, "val_wer"),
    }))
}

fn resume_start_epoch(resume: &Option<ResumeCheckpoint>) -> usize {
    resume.as_ref().map_or(1, |checkpoint| {
        if checkpoint.epoch_complete {
            checkpoint.epoch + 1
        } else {
            checkpoint.epoch
        }
    })
}

fn optional_f32(metadata: &Value, key: &str) -> Option<f32> {
    metadata
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
}

fn validate_resume_config(config: &BurnTrainConfig, saved_config: &Value) -> Result<()> {
    let current = run_config_json(config);
    for key in [
        "architecture",
        "variant",
        "input_dim",
        "vocab_size",
        "blank_id",
        "d_model",
        "num_layers",
        "num_heads",
        "paraformer_alignment_mode",
        "paraformer_enhanced",
        "w2v_hf_model_dir",
        "w2v_activation_checkpointing",
        "precision",
    ] {
        if current.get(key) != saved_config.get(key) {
            bail!(
                "resume checkpoint config mismatch for '{key}': current={:?} checkpoint={:?}",
                current.get(key),
                saved_config.get(key)
            );
        }
    }
    Ok(())
}

fn load_model_checkpoint<B, M>(
    model: M,
    resume: &Option<ResumeCheckpoint>,
    device: &B::Device,
) -> Result<M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let Some(checkpoint) = resume else {
        return Ok(model);
    };
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .load_file(
            checkpoint_base_path(&checkpoint.dir, "model"),
            &recorder,
            device,
        )
        .with_context(|| {
            format!(
                "failed to load {}",
                checkpoint_file_path(&checkpoint.dir, "model").display()
            )
        })
}

fn load_optimizer_checkpoint<B, M, O>(
    optimizer: O,
    resume: &Option<ResumeCheckpoint>,
    device: &B::Device,
) -> Result<O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    let Some(checkpoint) = resume else {
        return Ok(optimizer);
    };
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(checkpoint_base_path(&checkpoint.dir, "optimizer"), device)
        .with_context(|| {
            format!(
                "failed to load {}",
                checkpoint_file_path(&checkpoint.dir, "optimizer").display()
            )
        })?;
    Ok(optimizer.load_record(record))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn manifest_loads_inline_jsonl_records() {
        let dir =
            std::env::temp_dir().join(format!("w2v_bert_uk_train_test_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let manifest = dir.join("train.jsonl");
        fs::write(
            &manifest,
            r#"{"id":"a","features":[[0.1,0.2],[0.3,0.4]],"tokens":[1,2]}"#,
        )
        .unwrap();

        let records = load_manifest(&manifest, None).unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].rows, 2);
        assert_eq!(records[0].cols, 2);
        assert_eq!(records[0].tokens, vec![1, 2]);
        assert_eq!(records[0].text, None);
    }

    #[test]
    fn manifest_loads_optional_reference_text() {
        let dir = std::env::temp_dir().join(format!(
            "w2v_bert_uk_train_text_test_{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let manifest = dir.join("train_text.jsonl");
        fs::write(
            &manifest,
            r#"{"id":"a","features":[[0.1,0.2]],"tokens":[1],"text":"hello world"}"#,
        )
        .unwrap();

        let records = load_manifest(&manifest, None).unwrap();

        assert_eq!(records[0].text, Some("hello world".to_string()));
    }

    #[test]
    fn batch_pads_features_and_targets() {
        let records = vec![
            FeatureRecord {
                id: "a".to_string(),
                rows: 2,
                cols: 2,
                features: vec![1.0, 2.0, 3.0, 4.0],
                tokens: vec![1, 2],
                text: Some("one two".to_string()),
            },
            FeatureRecord {
                id: "b".to_string(),
                rows: 1,
                cols: 2,
                features: vec![5.0, 6.0],
                tokens: vec![3],
                text: None,
            },
        ];

        let batch = make_batch(&records, 2).unwrap();

        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.max_frames, 2);
        assert_eq!(batch.feature_lengths, vec![2, 1]);
        assert_eq!(batch.target_lengths, vec![2, 1]);
        assert_eq!(batch.reference_texts[0], Some("one two".to_string()));
    }

    #[test]
    fn greedy_decoder_collapses_repeats_and_blanks() {
        let device = Default::default();
        let logits = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(
                vec![
                    5.0, 1.0, 0.0, // blank
                    0.0, 6.0, 1.0, // 1
                    0.0, 7.0, 1.0, // repeat 1
                    8.0, 0.0, 0.0, // blank
                    0.0, 1.0, 9.0, // 2
                ],
                [1, 5, 3],
            ),
            &device,
        );

        let decoder = ValidationDecoder {
            tokenizer: None,
            language_model: None,
            beam_width: 1,
            n_best: 1,
        };
        let decoded = decode_validation_batch(logits, &[5], 0, &decoder).unwrap();

        assert_eq!(decoded, vec![vec![1, 2]]);
    }

    #[test]
    fn beam_decoder_uses_ctc_prefix_search_when_enabled() {
        let device = Default::default();
        let logits = Tensor::<NdArray<f32>, 3>::from_data(
            TensorData::new(
                vec![
                    4.0, 3.0, 0.0, // blank/1
                    3.5, 3.4, 0.0, // blank/1
                    0.0, 0.5, 4.0, // 2
                ],
                [1, 3, 3],
            ),
            &device,
        );
        let decoder = ValidationDecoder {
            tokenizer: None,
            language_model: None,
            beam_width: 4,
            n_best: 2,
        };

        let decoded = decode_validation_batch(logits, &[3], 0, &decoder).unwrap();

        assert!(!decoded[0].is_empty());
    }

    #[test]
    fn validation_metrics_compute_token_cer_and_wer() {
        let records = vec![FeatureRecord {
            id: "a".to_string(),
            rows: 1,
            cols: 2,
            features: vec![0.0, 0.0],
            tokens: vec![1, 2, 3],
            text: None,
        }];
        let batch = make_batch(&records, 2).unwrap();
        let decoder = ValidationDecoder {
            tokenizer: None,
            language_model: None,
            beam_width: 1,
            n_best: 1,
        };
        let mut metrics = ValidationMetrics::default();

        metrics.update(&batch, &[vec![1, 3]], &decoder);
        let summary = metrics.summary(0.5);

        assert_eq!(summary.loss, 0.5);
        assert_eq!(summary.decoded_samples, 1);
        assert_eq!(summary.token_error_rate, Some(1.0 / 3.0));
        assert_eq!(summary.cer, Some(2.0 / 5.0));
        assert_eq!(summary.wer, Some(1.0 / 3.0));
    }

    #[test]
    fn resume_config_validation_rejects_model_shape_mismatch() {
        let mut checkpoint_config = BurnTrainConfig::default();
        checkpoint_config.vocab_size = 32;
        let saved = run_config_json(&checkpoint_config);
        assert!(validate_resume_config(&checkpoint_config, &saved).is_ok());

        let mut current = checkpoint_config;
        current.input_dim += 1;
        let err = validate_resume_config(&current, &saved).unwrap_err();
        assert!(err.to_string().contains("input_dim"));
    }

    #[test]
    fn manifest_directory_loads_jsonl_shards_in_order() {
        let dir = std::env::temp_dir().join(format!(
            "w2v_bert_uk_manifest_dir_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("b.jsonl"),
            r#"{"id":"b","features":[[0.2,0.3]],"tokens":[2]}"#,
        )
        .unwrap();
        fs::write(
            dir.join("a.jsonl"),
            r#"{"id":"a","features":[[0.1,0.2]],"tokens":[1]}"#,
        )
        .unwrap();

        let records = load_manifest(&dir, Some(1)).unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "a");
    }

    #[test]
    fn streaming_loader_respects_adaptive_padded_frame_budget() {
        let dir = std::env::temp_dir().join(format!(
            "w2v_bert_uk_adaptive_batch_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("train.jsonl"),
            [
                r#"{"id":"a","features":[[0.1,0.2],[0.3,0.4]],"tokens":[1]}"#,
                r#"{"id":"b","features":[[0.1,0.2],[0.3,0.4],[0.5,0.6]],"tokens":[1]}"#,
                r#"{"id":"c","features":[[0.1,0.2]],"tokens":[1]}"#,
            ]
            .join("\n"),
        )
        .unwrap();

        let mut loader = StreamingBatchLoader::new(
            dir.join("train.jsonl"),
            8,
            Some(AdaptiveBatchConfig {
                unit: AdaptiveBatchUnit::PaddedFrames,
                budget: 6,
                max_samples: None,
            }),
            false,
            4096,
            2,
            None,
        )
        .unwrap();

        let first = loader.next_batch().unwrap().unwrap();
        let second = loader.next_batch().unwrap().unwrap();

        assert_eq!(first.batch_size, 2);
        assert_eq!(first.feature_lengths, vec![2, 3]);
        assert_eq!(second.batch_size, 1);
        assert_eq!(second.feature_lengths, vec![1]);
    }

    #[test]
    fn streaming_loader_sorts_by_length_desc_within_buffer() {
        let dir = std::env::temp_dir().join(format!(
            "w2v_bert_uk_sort_batch_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("train.jsonl"),
            [
                r#"{"id":"short","features":[[0.1,0.2]],"tokens":[1]}"#,
                r#"{"id":"long","features":[[0.1,0.2],[0.3,0.4],[0.5,0.6]],"tokens":[1]}"#,
                r#"{"id":"mid","features":[[0.1,0.2],[0.3,0.4]],"tokens":[1]}"#,
            ]
            .join("\n"),
        )
        .unwrap();

        let mut loader =
            StreamingBatchLoader::new(dir.join("train.jsonl"), 3, None, true, 3, 2, None).unwrap();

        let batch = loader.next_batch().unwrap().unwrap();

        assert_eq!(batch.feature_lengths, vec![3, 2, 1]);
    }
}
