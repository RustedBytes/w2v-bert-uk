use std::collections::{HashMap, VecDeque};
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::container::TensorContainer;
use burn::tensor::{FloatDType, Int, IntDType, Tensor, TensorData, set_default_dtypes};
use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn_nn::loss::{CTCLossConfig, Reduction};
use burn_optim::{
    AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer,
    adaptor::OptimizerAdaptor, grad_clipping::GradientClippingConfig,
};
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    execute, queue,
    style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use kenlm::{Config as KenlmConfig, Model as KenlmModel};
use polars::prelude::{AnyValue, DataFrame, ParquetReader, SerReader, Series};
use rand::Rng;
use rayon::prelude::*;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;
use serde_json::{Value, json};
use splintr::SentencePieceTokenizer;

use crate::audio::{
    AudioDecodeConfig, FeatureExtractorConfig, WaveformAugmentConfig,
    audio_bytes_to_features_with_augmentation, audio_bytes_to_features_with_config,
    audio_file_to_features_with_augmentation, audio_file_to_features_with_config,
};
use crate::ctc::{CtcCandidate, threaded_ctc_beam_search_decode_n_best};
use crate::paraformer::{
    EnhancedParaformerV2, EnhancedParaformerV2Config, ParaformerAlignmentMode, ParaformerV2,
    ParaformerV2Config,
};
use crate::squeezeformer::{SqueezeformerCtc, SqueezeformerCtcConfig, SqueezeformerEncoderConfig};
use crate::tokenizer::{load_sentencepiece_tokenizer, load_sentencepiece_transcript_tokenizer};
use crate::wav2vec::{Wav2VecBertConfig, Wav2VecBertCtc, Wav2VecBertCtcConfig};
use crate::zipformer::{ZipformerConfig, ZipformerCtc, ZipformerCtcConfig};
use crate::{W2vBertEncoderConfig, normalize_spaces};

#[derive(Clone, Copy, Debug)]
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
    pub dataset_index_dir: Option<PathBuf>,
    pub spec_augment: SpecAugmentConfig,
    pub waveform_augment: WaveformAugmentConfig,
    pub epochs: usize,
    pub learning_rate: f64,
    pub lr_warmup_steps: usize,
    pub lr_hold_steps: usize,
    pub lr_decay_steps: usize,
    pub lr_warmup_epochs: usize,
    pub lr_hold_epochs: usize,
    pub lr_decay_exponent: f64,
    pub lr_min: f64,
    pub weight_decay: f64,
    pub gradient_accumulation_steps: usize,
    pub gradient_clip_norm: Option<f32>,
    pub gradient_clip_value: Option<f32>,
    pub ema_decay: Option<f64>,
    pub ema_start_step: usize,
    pub log_every: usize,
    pub validate_every_steps: Option<usize>,
    pub max_train_samples: Option<usize>,
    pub max_val_samples: Option<usize>,
    pub max_audio_duration_ms: Option<usize>,
    pub tokenizer_path: Option<PathBuf>,
    pub val_beam_width: usize,
    pub val_n_best: usize,
    pub val_lm_path: Option<PathBuf>,
    pub val_lm_weight: f32,
    pub val_lm_word_bonus: f32,
    pub val_lm_bos: bool,
    pub val_lm_eos: bool,
    pub val_log_samples: usize,
    pub dry_run: bool,
    pub paraformer_alignment_mode: ParaformerAlignmentMode,
    pub paraformer_enhanced: bool,
    pub w2v_hf_model_dir: Option<PathBuf>,
    pub w2v_hf_load_weights: bool,
    pub w2v_activation_checkpointing: bool,
    pub w2v_num_adapter_layers: usize,
    pub w2v_adapter_stride: usize,
    pub w2v_adapter_kernel_size: usize,
    pub init_from: Option<PathBuf>,
    pub resume_from: Option<PathBuf>,
    pub backend: TrainBackendKind,
    pub device_index: usize,
    pub device_indices: Vec<usize>,
    pub precision: TrainPrecision,
    pub tui: bool,
    pub hf_upload_checkpoints: bool,
    pub hf_upload_repo_id: Option<String>,
    pub hf_upload_revision: Option<String>,
    pub hf_upload_private: bool,
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
            dataset_index_dir: None,
            spec_augment: SpecAugmentConfig::default(),
            waveform_augment: WaveformAugmentConfig::default(),
            epochs: 500,
            learning_rate: 3.0e-4,
            lr_warmup_steps: 0,
            lr_hold_steps: 0,
            lr_decay_steps: 0,
            lr_warmup_epochs: 20,
            lr_hold_epochs: 160,
            lr_decay_exponent: 1.0,
            lr_min: 0.0,
            weight_decay: 5.0e-4,
            gradient_accumulation_steps: 1,
            gradient_clip_norm: None,
            gradient_clip_value: None,
            ema_decay: None,
            ema_start_step: 0,
            log_every: 10,
            validate_every_steps: None,
            max_train_samples: None,
            max_val_samples: None,
            max_audio_duration_ms: None,
            tokenizer_path: None,
            val_beam_width: 1,
            val_n_best: 1,
            val_lm_path: None,
            val_lm_weight: 0.45,
            val_lm_word_bonus: 0.0,
            val_lm_bos: true,
            val_lm_eos: true,
            val_log_samples: 0,
            dry_run: false,
            paraformer_alignment_mode: ParaformerAlignmentMode::Viterbi,
            paraformer_enhanced: false,
            w2v_hf_model_dir: None,
            w2v_hf_load_weights: false,
            w2v_activation_checkpointing: false,
            w2v_num_adapter_layers: 0,
            w2v_adapter_stride: 2,
            w2v_adapter_kernel_size: 3,
            init_from: None,
            resume_from: None,
            backend: TrainBackendKind::Cpu,
            device_index: 0,
            device_indices: vec![0],
            precision: TrainPrecision::F32,
            tui: false,
            hf_upload_checkpoints: false,
            hf_upload_repo_id: None,
            hf_upload_revision: None,
            hf_upload_private: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveBatchUnit {
    Samples,
    Frames,
    PaddedFrames,
    FeatureValues,
    DurationMs,
    PaddedDurationMs,
}

impl std::str::FromStr for AdaptiveBatchUnit {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "samples" => Ok(Self::Samples),
            "frames" => Ok(Self::Frames),
            "padded-frames" | "padded_frames" => Ok(Self::PaddedFrames),
            "feature-values" | "feature_values" | "values" => Ok(Self::FeatureValues),
            "duration-ms" | "duration_ms" => Ok(Self::DurationMs),
            "padded-duration-ms" | "padded_duration_ms" => Ok(Self::PaddedDurationMs),
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

#[derive(Clone, Copy, Debug, Default)]
pub struct SpecAugmentConfig {
    pub time_masks: usize,
    pub time_mask_max_frames: usize,
    pub frequency_masks: usize,
    pub frequency_mask_max_bins: usize,
}

impl SpecAugmentConfig {
    fn is_enabled(&self) -> bool {
        (self.time_masks > 0 && self.time_mask_max_frames > 0)
            || (self.frequency_masks > 0 && self.frequency_mask_max_bins > 0)
    }
}

#[derive(Clone, Debug)]
pub struct FeatureRecord {
    pub id: String,
    pub rows: usize,
    pub cols: usize,
    pub features: Vec<f32>,
    pub duration_ms: usize,
    pub tokens: Vec<i64>,
    pub text: Option<String>,
}

#[derive(Clone, Debug)]
pub struct TrainBatch {
    pub ids: Vec<String>,
    pub features: Vec<f32>,
    pub batch_size: usize,
    pub max_frames: usize,
    pub feature_dim: usize,
    pub feature_lengths: Vec<usize>,
    pub durations_ms: Vec<usize>,
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

#[derive(Clone, Debug)]
pub struct BurnInferenceConfig {
    pub checkpoint: PathBuf,
    pub manifest: PathBuf,
    pub output: Option<PathBuf>,
    pub use_ema: bool,
    pub batch_size: Option<usize>,
    pub max_samples: Option<usize>,
    pub tokenizer_path: Option<PathBuf>,
    pub beam_width: Option<usize>,
    pub n_best: Option<usize>,
    pub lm_path: Option<PathBuf>,
    pub lm_weight: Option<f32>,
    pub lm_word_bonus: Option<f32>,
    pub lm_bos: bool,
    pub lm_eos: bool,
}

impl Default for BurnInferenceConfig {
    fn default() -> Self {
        Self {
            checkpoint: PathBuf::from("runs/burn/checkpoint_latest"),
            manifest: PathBuf::from("val.jsonl"),
            output: None,
            use_ema: false,
            batch_size: None,
            max_samples: None,
            tokenizer_path: None,
            beam_width: None,
            n_best: None,
            lm_path: None,
            lm_weight: None,
            lm_word_bonus: None,
            lm_bos: true,
            lm_eos: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BurnInferenceSummary {
    pub architecture: TrainArchitecture,
    pub decoded_samples: usize,
    pub output: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct BurnExportConfig {
    pub checkpoint: PathBuf,
    pub output_dir: PathBuf,
    pub use_ema: bool,
    pub hf_repo_id: Option<String>,
    pub hf_revision: Option<String>,
    pub hf_private: bool,
}

#[derive(Clone, Debug)]
pub struct BurnExportSummary {
    pub architecture: TrainArchitecture,
    pub model_path: PathBuf,
    pub metadata_path: PathBuf,
    pub readme_path: PathBuf,
    pub training_config_path: PathBuf,
    pub hf_upload: Option<HfUploadSummary>,
}

#[derive(Clone, Debug)]
pub struct HfUploadSummary {
    pub repo_id: String,
    pub revision: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct ValidationSummary {
    loss: f32,
    token_error_rate: Option<f32>,
    cer: Option<f32>,
    wer: Option<f32>,
    decoded_samples: usize,
    sample_predictions: Vec<ValidationSamplePrediction>,
}

#[derive(Clone, Debug)]
struct ValidationSamplePrediction {
    id: String,
    prediction_text: String,
    reference_text: String,
    prediction_tokens: Vec<u32>,
    reference_tokens: Vec<i64>,
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

pub fn run_burn_inference(config: BurnInferenceConfig) -> Result<BurnInferenceSummary> {
    type Backend = burn_ndarray::NdArray<f32>;
    let device = Default::default();
    let mut train_config = load_checkpoint_train_config(&config.checkpoint)?;
    train_config.train_manifest = config.manifest.clone();
    train_config.batch_size = config.batch_size.unwrap_or(train_config.batch_size);
    train_config.max_train_samples = config.max_samples;
    if let Some(tokenizer_path) = config.tokenizer_path.clone() {
        train_config.tokenizer_path = Some(tokenizer_path);
    }
    if let Some(beam_width) = config.beam_width {
        train_config.val_beam_width = beam_width;
    }
    train_config.val_n_best = config.n_best.unwrap_or(train_config.val_beam_width);
    if let Some(lm_path) = config.lm_path.clone() {
        train_config.val_lm_path = Some(lm_path);
    }
    if let Some(lm_weight) = config.lm_weight {
        train_config.val_lm_weight = lm_weight;
    }
    if let Some(lm_word_bonus) = config.lm_word_bonus {
        train_config.val_lm_word_bonus = lm_word_bonus;
    }
    train_config.val_lm_bos = config.lm_bos;
    train_config.val_lm_eos = config.lm_eos;
    train_config.backend = TrainBackendKind::Cpu;
    train_config.device_index = 0;
    train_config.device_indices = vec![0];
    train_config.precision = TrainPrecision::F32;
    validate_config(&train_config)?;

    let decoder = ValidationDecoder::from_config(&train_config)?;
    let checkpoint_dir = checkpoint_dir_from_path(&config.checkpoint)?;
    let mut writer = match &config.output {
        Some(path) => Some(
            fs::File::create(path)
                .with_context(|| format!("failed to create inference output {}", path.display()))?,
        ),
        None => None,
    };

    let decoded_samples = match train_config.architecture {
        TrainArchitecture::Squeezeformer => {
            let model = load_inference_model::<Backend, _>(
                build_squeezeformer_model::<Backend>(&train_config, &device),
                &checkpoint_dir,
                config.use_ema,
                &device,
            )?;
            infer_ctc_batches(
                &model,
                &train_config,
                &device,
                &decoder,
                writer.as_mut(),
                |model, features, lengths| model.forward_with_lengths(features, Some(lengths)),
            )?
        }
        TrainArchitecture::Zipformer => {
            let model = load_inference_model::<Backend, _>(
                build_zipformer_model::<Backend>(&train_config, &device),
                &checkpoint_dir,
                config.use_ema,
                &device,
            )?;
            infer_ctc_batches(
                &model,
                &train_config,
                &device,
                &decoder,
                writer.as_mut(),
                |model, features, lengths| model.forward_with_lengths(features, lengths),
            )?
        }
        TrainArchitecture::Paraformer if train_config.paraformer_enhanced => {
            let model = load_inference_model::<Backend, _>(
                build_enhanced_paraformer_model::<Backend>(&train_config, &device),
                &checkpoint_dir,
                config.use_ema,
                &device,
            )?;
            infer_ctc_batches(
                &model,
                &train_config,
                &device,
                &decoder,
                writer.as_mut(),
                |model, features, lengths| model.ctc_log_probs(features, lengths),
            )?
        }
        TrainArchitecture::Paraformer => {
            let model = load_inference_model::<Backend, _>(
                build_paraformer_model::<Backend>(&train_config, &device),
                &checkpoint_dir,
                config.use_ema,
                &device,
            )?;
            infer_ctc_batches(
                &model,
                &train_config,
                &device,
                &decoder,
                writer.as_mut(),
                |model, features, lengths| {
                    let output = model.forward(features, lengths);
                    (output.ctc_log_probs, output.encoder_lengths)
                },
            )?
        }
        TrainArchitecture::Wav2VecBert => {
            let model = load_inference_model::<Backend, _>(
                build_wav2vec_model::<Backend>(&train_config, &device)?,
                &checkpoint_dir,
                config.use_ema,
                &device,
            )?;
            infer_ctc_batches(
                &model,
                &train_config,
                &device,
                &decoder,
                writer.as_mut(),
                |model, features, lengths| model.forward_with_lengths(features, lengths),
            )?
        }
    };

    Ok(BurnInferenceSummary {
        architecture: train_config.architecture,
        decoded_samples,
        output: config.output,
    })
}

pub fn run_burn_export(config: BurnExportConfig) -> Result<BurnExportSummary> {
    let train_config = load_checkpoint_train_config(&config.checkpoint)?;
    let checkpoint_dir = checkpoint_dir_from_path(&config.checkpoint)?;
    fs::create_dir_all(&config.output_dir).with_context(|| {
        format!(
            "failed to create export directory {}",
            config.output_dir.display()
        )
    })?;
    let source_stem = if config.use_ema { "ema_model" } else { "model" };
    let source_model = checkpoint_file_path(&checkpoint_dir, source_stem);
    if !source_model.exists() {
        bail!(
            "checkpoint model file does not exist: {}",
            source_model.display()
        );
    }
    let model_path = config.output_dir.join(format!("{source_stem}.bin"));
    fs::copy(&source_model, &model_path).with_context(|| {
        format!(
            "failed to copy {} to {}",
            source_model.display(),
            model_path.display()
        )
    })?;
    let training_config_path = config.output_dir.join("training_config.json");
    fs::write(
        &training_config_path,
        serde_json::to_string_pretty(&run_config_json(&train_config))?,
    )
    .with_context(|| format!("failed to write {}", training_config_path.display()))?;
    let readme_path = config.output_dir.join("README.md");
    fs::write(
        &readme_path,
        export_readme(&train_config, &model_path, config.use_ema),
    )
    .with_context(|| format!("failed to write {}", readme_path.display()))?;
    let metadata_path = config.output_dir.join("burn_export.json");
    let checkpoint_metadata_path = config.output_dir.join("checkpoint.json");
    let checkpoint_metadata = json!({
        "epoch": Value::Null,
        "epoch_complete": true,
        "global_step": Value::Null,
        "train_ctc_loss": Value::Null,
        "val_ctc_loss": Value::Null,
        "val_cer": Value::Null,
        "val_wer": Value::Null,
        "checkpoint_dir": config.output_dir,
        "model_path": if config.use_ema { Value::Null } else { json!(&model_path) },
        "optimizer_path": Value::Null,
        "ema_model_path": if config.use_ema { json!(&model_path) } else { Value::Null },
        "training_config": run_config_json(&train_config),
    });
    fs::write(
        &checkpoint_metadata_path,
        serde_json::to_string_pretty(&checkpoint_metadata)?,
    )
    .with_context(|| format!("failed to write {}", checkpoint_metadata_path.display()))?;
    fs::write(
        &metadata_path,
        serde_json::to_string_pretty(&json!({
            "format": "burn-bin-full-precision",
            "architecture": architecture_name(&train_config.architecture),
            "model_path": &model_path,
            "readme_path": &readme_path,
            "training_config_path": &training_config_path,
            "checkpoint_metadata_path": &checkpoint_metadata_path,
            "source_checkpoint": &checkpoint_dir,
            "source": if config.use_ema { "ema_model" } else { "model" },
            "package_files": [
                &model_path,
                &readme_path,
                &training_config_path,
                &metadata_path,
                &checkpoint_metadata_path,
            ],
            "training_config": run_config_json(&train_config),
        }))?,
    )
    .with_context(|| format!("failed to write {}", metadata_path.display()))?;
    let hf_upload = if let Some(repo_id) = config.hf_repo_id.clone() {
        upload_export_to_huggingface(
            &repo_id,
            config.hf_revision.as_deref(),
            config.hf_private,
            &config.output_dir,
        )?;
        Some(HfUploadSummary {
            repo_id,
            revision: config.hf_revision.clone(),
        })
    } else {
        None
    };

    Ok(BurnExportSummary {
        architecture: train_config.architecture,
        model_path,
        metadata_path,
        readme_path,
        training_config_path,
        hf_upload,
    })
}

fn export_readme(config: &BurnTrainConfig, model_path: &Path, use_ema: bool) -> String {
    format!(
        "---\nlibrary_name: burn\ntags:\n- automatic-speech-recognition\n- burn\n- ukrainian\n---\n\n# {} Burn ASR Export\n\nThis package contains a Burn full-precision checkpoint exported from `w2v-bert-uk`.\n\n- Architecture: `{}`\n- Model file: `{}`\n- Source weights: `{}`\n- Input feature dimension: `{}`\n- Vocabulary size: `{}`\n- Blank token id: `{}`\n\nUse `burn-infer --checkpoint <checkpoint-or-export-dir> --manifest <features.jsonl>` with a compatible feature manifest.\n",
        architecture_name(&config.architecture),
        architecture_name(&config.architecture),
        model_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("model.bin"),
        if use_ema { "ema_model" } else { "model" },
        config.input_dim,
        config.vocab_size,
        config.blank_id,
    )
}

fn upload_export_to_huggingface(
    repo_id: &str,
    revision: Option<&str>,
    private: bool,
    output_dir: &Path,
) -> Result<()> {
    upload_path_to_huggingface(repo_id, revision, private, output_dir, ".")
}

fn upload_path_to_huggingface(
    repo_id: &str,
    revision: Option<&str>,
    private: bool,
    source_path: &Path,
    repo_path: &str,
) -> Result<()> {
    let mut create = Command::new("huggingface-cli");
    create.args(["repo", "create", repo_id, "--type", "model", "--exist-ok"]);
    if private {
        create.arg("--private");
    }
    let status = create
        .status()
        .context("failed to run huggingface-cli; install huggingface_hub CLI and login first")?;
    if !status.success() {
        bail!("huggingface-cli repo create failed with status {status}");
    }

    let mut upload = Command::new("huggingface-cli");
    upload.args(["upload", repo_id]);
    upload.arg(source_path);
    upload.arg(repo_path);
    upload.args(["--repo-type", "model"]);
    if let Some(revision) = revision {
        upload.args(["--revision", revision]);
    }
    let status = upload
        .status()
        .context("failed to run huggingface-cli; install huggingface_hub CLI and login first")?;
    if !status.success() {
        bail!("huggingface-cli upload failed with status {status}");
    }
    Ok(())
}

fn maybe_upload_training_checkpoint(config: &BurnTrainConfig, checkpoint_dir: &Path) -> Result<()> {
    if !config.hf_upload_checkpoints {
        return Ok(());
    }
    let repo_id = config
        .hf_upload_repo_id
        .as_deref()
        .context("hf_upload_checkpoints requires hf_upload_repo_id")?;
    upload_path_to_huggingface(
        repo_id,
        config.hf_upload_revision.as_deref(),
        config.hf_upload_private,
        checkpoint_dir,
        "checkpoint_latest",
    )?;
    let legacy_metadata = config.output_dir.join("checkpoint_latest.json");
    if legacy_metadata.exists() {
        upload_path_to_huggingface(
            repo_id,
            config.hf_upload_revision.as_deref(),
            config.hf_upload_private,
            &legacy_metadata,
            "checkpoint_latest.json",
        )?;
    }
    Ok(())
}

fn run_burn_training_cpu(config: BurnTrainConfig) -> Result<TrainSummary> {
    if config.device_indices != [0] {
        bail!("cpu backend only supports device index 0");
    }
    type InnerBackend = burn_ndarray::NdArray<f32>;
    let device = Default::default();
    let devices = vec![device];
    run_burn_training_inner::<InnerBackend>(&config, &devices)
}

#[cfg(feature = "burn-cuda-backend")]
fn run_burn_training_cuda(config: BurnTrainConfig) -> Result<TrainSummary> {
    let devices = config
        .device_indices
        .iter()
        .map(|index| burn_cuda::CudaDevice { index: *index })
        .collect::<Vec<_>>();
    match config.precision {
        TrainPrecision::F32 => run_burn_training_inner::<burn_cuda::Cuda<f32>>(&config, &devices),
        TrainPrecision::F16 => {
            run_burn_training_inner::<burn_cuda::Cuda<burn::tensor::f16>>(&config, &devices)
        }
        TrainPrecision::Bf16 => {
            run_burn_training_inner::<burn_cuda::Cuda<burn::tensor::bf16>>(&config, &devices)
        }
    }
}

#[cfg(not(feature = "burn-cuda-backend"))]
fn run_burn_training_cuda(_config: BurnTrainConfig) -> Result<TrainSummary> {
    bail!("CUDA training requires building with --features burn-cuda-backend")
}

#[cfg(feature = "burn-wgpu-backend")]
fn run_burn_training_wgpu(config: BurnTrainConfig) -> Result<TrainSummary> {
    let devices = config
        .device_indices
        .iter()
        .map(|index| burn_wgpu::WgpuDevice::DiscreteGpu(*index))
        .collect::<Vec<_>>();
    match config.precision {
        TrainPrecision::F32 => run_burn_training_inner::<burn_wgpu::Wgpu<f32>>(&config, &devices),
        TrainPrecision::F16 => {
            run_burn_training_inner::<burn_wgpu::Wgpu<burn::tensor::f16>>(&config, &devices)
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
    devices: &[InnerBackend::Device],
) -> Result<TrainSummary>
where
    InnerBackend: Backend,
{
    type TrainBackend<InnerBackend> = burn_autodiff::Autodiff<InnerBackend>;
    let device = devices
        .first()
        .ok_or_else(|| anyhow!("at least one training device is required"))?;
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
            train_ctc_model(model, config, devices)
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
            train_ctc_model(model, config, devices)
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
                train_enhanced_paraformer_model(model, config, devices)
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
                train_paraformer_model(model, config, devices)
            }
        }
        TrainArchitecture::Wav2VecBert => {
            if config.w2v_activation_checkpointing {
                type CheckpointBackend<InnerBackend> =
                    burn_autodiff::Autodiff<InnerBackend, BalancedCheckpointing>;
                configure_training_precision::<CheckpointBackend<InnerBackend>>(config, device)?;
                train_wav2vec_model::<CheckpointBackend<InnerBackend>>(config, devices)
            } else {
                train_wav2vec_model::<TrainBackend<InnerBackend>>(config, devices)
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

fn build_squeezeformer_model<B: Backend>(
    config: &BurnTrainConfig,
    device: &B::Device,
) -> SqueezeformerCtc<B> {
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
    SqueezeformerCtcConfig {
        encoder,
        vocab_size: config.vocab_size,
    }
    .init::<B>(device)
}

fn build_zipformer_model<B: Backend>(
    config: &BurnTrainConfig,
    device: &B::Device,
) -> ZipformerCtc<B> {
    let mut encoder = config
        .variant
        .as_deref()
        .and_then(ZipformerConfig::variant)
        .unwrap_or_else(|| ZipformerConfig::new(config.input_dim));
    encoder.input_dim = config.input_dim;
    ZipformerCtcConfig {
        encoder,
        vocab_size: config.vocab_size,
    }
    .init::<B>(device)
}

fn build_paraformer_model<B: Backend>(
    config: &BurnTrainConfig,
    device: &B::Device,
) -> ParaformerV2<B> {
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
            let mut value = ParaformerV2Config::new(config.input_dim, config.vocab_size)
                .with_blank_id(config.blank_id);
            value.encoder_dim = config.d_model;
            value.decoder_dim = config.d_model;
            value.encoder_layers = config.num_layers;
            value.attention_heads = config.num_heads;
            value
        });
    model_config.init::<B>(device)
}

fn build_enhanced_paraformer_model<B: Backend>(
    config: &BurnTrainConfig,
    device: &B::Device,
) -> EnhancedParaformerV2<B> {
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
            let mut value = EnhancedParaformerV2Config::new(config.input_dim, config.vocab_size)
                .with_blank_id(config.blank_id);
            value.base.encoder_dim = config.d_model;
            value.base.decoder_dim = config.d_model;
            value.base.encoder_layers = config.num_layers;
            value.base.attention_heads = config.num_heads;
            value
        });
    model_config.init::<B>(device)
}

fn build_wav2vec_model<B: Backend>(
    config: &BurnTrainConfig,
    device: &B::Device,
) -> Result<Wav2VecBertCtc<B>> {
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
    if config.w2v_num_adapter_layers > 0 {
        model_config.encoder = model_config
            .encoder
            .with_adapter(config.w2v_adapter_stride, config.w2v_num_adapter_layers)
            .with_adapter_kernel_size(config.w2v_adapter_kernel_size);
    }
    Ok(model_config.init::<B>(device))
}

fn train_wav2vec_model<B>(config: &BurnTrainConfig, devices: &[B::Device]) -> Result<TrainSummary>
where
    B: AutodiffBackend,
{
    let device = devices
        .first()
        .ok_or_else(|| anyhow!("at least one training device is required"))?;
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
    if config.w2v_num_adapter_layers > 0 {
        model_config.encoder = model_config
            .encoder
            .with_adapter(config.w2v_adapter_stride, config.w2v_num_adapter_layers)
            .with_adapter_kernel_size(config.w2v_adapter_kernel_size);
    }
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
    train_ctc_model(model, &train_config, devices)
}

trait TrainableCtc<B: AutodiffBackend>: AutodiffModule<B> {
    fn ctc_logits(&self, features: Tensor<B, 3>, lengths: Vec<usize>)
    -> (Tensor<B, 3>, Vec<usize>);
}

trait ValidCtc<B: Backend>: Module<B> {
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

impl<B: Backend> ValidCtc<B> for SqueezeformerCtc<B> {
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

impl<B: Backend> ValidCtc<B> for ZipformerCtc<B> {
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

impl<B: Backend> ValidCtc<B> for Wav2VecBertCtc<B> {
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

impl<B: Backend> ValidCtc<B> for ParaformerV2<B> {
    fn ctc_logits(
        &self,
        features: Tensor<B, 3>,
        lengths: Vec<usize>,
    ) -> (Tensor<B, 3>, Vec<usize>) {
        let output = self.forward(features, lengths);
        (output.ctc_log_probs, output.encoder_lengths)
    }
}

fn adamw_optimizer<B, M>(config: &BurnTrainConfig) -> OptimizerAdaptor<AdamW, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    if let Some(clipping) = gradient_clipping_config(config) {
        optimizer = optimizer.with_grad_clipping(clipping.init());
    }
    optimizer
}

fn gradient_clipping_config(config: &BurnTrainConfig) -> Option<GradientClippingConfig> {
    config
        .gradient_clip_norm
        .map(GradientClippingConfig::Norm)
        .or_else(|| {
            config
                .gradient_clip_value
                .map(GradientClippingConfig::Value)
        })
}

fn scheduled_learning_rate(config: &BurnTrainConfig, optimizer_step: usize, epoch: usize) -> f64 {
    let base = config.learning_rate;
    let min_lr = config.lr_min;
    let step = optimizer_step.max(1);

    let has_step_schedule =
        config.lr_warmup_steps > 0 || config.lr_hold_steps > 0 || config.lr_decay_steps > 0;
    if has_step_schedule {
        if config.lr_warmup_steps > 0 && step <= config.lr_warmup_steps {
            return base * step as f64 / config.lr_warmup_steps as f64;
        }

        let after_warmup = step.saturating_sub(config.lr_warmup_steps);
        if config.lr_hold_steps > 0 && after_warmup <= config.lr_hold_steps {
            return base;
        }

        if config.lr_decay_steps == 0 {
            return base;
        }

        let decay_step = after_warmup.saturating_sub(config.lr_hold_steps);
        if decay_step >= config.lr_decay_steps {
            return min_lr;
        }

        let progress = decay_step as f64 / config.lr_decay_steps as f64;
        return min_lr + (base - min_lr) * (1.0 - progress);
    }

    let epoch = epoch.max(1);
    if config.lr_warmup_epochs > 0 && epoch <= config.lr_warmup_epochs {
        return base * epoch as f64 / config.lr_warmup_epochs as f64;
    }

    let after_warmup = epoch.saturating_sub(config.lr_warmup_epochs);
    if config.lr_hold_epochs > 0 && after_warmup <= config.lr_hold_epochs {
        return base;
    }

    if config.lr_decay_exponent == 0.0 {
        return base;
    }

    let warmup = config.lr_warmup_epochs.max(1) as f64;
    let decay_epoch = epoch.saturating_sub(config.lr_hold_epochs).max(1) as f64;
    let lr = base * (warmup / decay_epoch).powf(config.lr_decay_exponent);
    lr.max(min_lr)
}

fn scale_gradients<B, M>(module: &M, grads: GradientsParams, scale: f64) -> GradientsParams
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let mut visitor = GradientsScaler::<B> {
        grads_in: grads,
        grads_out: GradientsParams::new(),
        scale,
        phantom: std::marker::PhantomData,
    };
    module.visit(&mut visitor);
    visitor.grads_out
}

struct GradientsScaler<B: AutodiffBackend> {
    grads_in: GradientsParams,
    grads_out: GradientsParams,
    scale: f64,
    phantom: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradientsScaler<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(grad) = self.grads_in.remove::<B::InnerBackend, D>(param.id) {
            self.grads_out
                .register::<B::InnerBackend, D>(param.id, grad.mul_scalar(self.scale));
        }
    }
}

fn initialize_ema_model<B, M>(model: &M, config: &BurnTrainConfig) -> Option<M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    config.ema_decay.map(|_| model.clone().no_grad())
}

fn update_ema_after_step<B, M>(
    ema_model: &mut Option<M>,
    model: &M,
    config: &BurnTrainConfig,
    global_step: usize,
) where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let Some(decay) = config.ema_decay else {
        return;
    };
    if global_step < config.ema_start_step {
        return;
    }
    if let Some(ema) = ema_model.take() {
        *ema_model = Some(update_ema_model::<B, M>(ema, model, decay));
    }
}

fn update_ema_model<B, M>(ema: M, model: &M, decay: f64) -> M
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let mut collector = EmaSourceCollector::<B> {
        tensors: TensorContainer::new(),
        phantom: std::marker::PhantomData,
    };
    model.visit(&mut collector);
    let mut mapper = EmaMapper::<B> {
        tensors: collector.tensors,
        decay,
        phantom: std::marker::PhantomData,
    };
    ema.map(&mut mapper).no_grad()
}

struct EmaSourceCollector<B: AutodiffBackend> {
    tensors: TensorContainer<ParamId>,
    phantom: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for EmaSourceCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.tensors
            .register::<B>(param.id, param.val().detach().into_primitive());
    }
}

struct EmaMapper<B: AutodiffBackend> {
    tensors: TensorContainer<ParamId>,
    decay: f64,
    phantom: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleMapper<B> for EmaMapper<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, ema_tensor, mapper) = param.consume();
        let Some(source) = self.tensors.remove::<B>(&id) else {
            return Param::from_mapped_value(id, ema_tensor, mapper);
        };
        let source = Tensor::<B, D>::from_primitive(source).detach();
        let updated =
            ema_tensor.detach().mul_scalar(self.decay) + source.mul_scalar(1.0 - self.decay);
        Param::from_mapped_value(id, updated.set_require_grad(false), mapper)
    }
}

fn maybe_step_accumulated<B, M, O>(
    model: M,
    optimizer: &mut O,
    accumulator: &mut GradientsAccumulator<M>,
    accumulated_batches: &mut usize,
    optimizer_step: &mut usize,
    epoch: usize,
    config: &BurnTrainConfig,
    force: bool,
) -> (M, Option<f64>)
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    if *accumulated_batches == 0
        || (!force && *accumulated_batches < config.gradient_accumulation_steps)
    {
        return (model, None);
    }

    let grads = accumulator.grads();
    let grads = scale_gradients::<B, M>(&model, grads, 1.0 / *accumulated_batches as f64);
    *accumulated_batches = 0;
    *optimizer_step += 1;
    let lr = scheduled_learning_rate(config, *optimizer_step, epoch);
    (optimizer.step(lr, model, grads), Some(lr))
}

fn collect_parallel_gradients<B, M, R, F>(
    model: &M,
    batches: Vec<TrainBatch>,
    devices: &[B::Device],
    main_device: &B::Device,
    loss_fn: &F,
) -> Result<Vec<(GradientsParams, R)>>
where
    B: AutodiffBackend,
    B::Device: Clone,
    M: AutodiffModule<B>,
    R: Send,
    F: Fn(&M, &TrainBatch, &B::Device) -> Result<(Tensor<B, 1>, R)> + Sync,
{
    std::thread::scope(|scope| {
        let handles = batches
            .into_iter()
            .zip(devices.iter().cloned())
            .map(|(batch, device)| {
                let local_model = model.clone().fork(&device);
                let main_device = main_device.clone();
                scope.spawn(move || {
                    let (loss, metrics) = loss_fn(&local_model, &batch, &device)?;
                    let grads = GradientsParams::from_grads(loss.backward(), &local_model)
                        .to_device::<B, M>(&main_device, &local_model);
                    Ok::<_, anyhow::Error>((grads, metrics))
                })
            })
            .collect::<Vec<_>>();

        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            results.push(
                handle
                    .join()
                    .map_err(|_| anyhow!("multi-GPU worker thread panicked"))??,
            );
        }
        Ok(results)
    })
}

fn collect_device_batches<B>(
    first: TrainBatch,
    train_batches: &mut StreamingBatchLoader,
    devices: &[B::Device],
) -> Result<Vec<TrainBatch>>
where
    B: Backend,
{
    let mut batches = vec![first];
    while batches.len() < devices.len() {
        let Some(batch) = train_batches.next_batch()? else {
            break;
        };
        batches.push(batch);
    }
    Ok(batches)
}

fn average_metric_triplets(results: &[(GradientsParams, [f32; 3])]) -> [f32; 3] {
    let mut values = [0.0; 3];
    for (_, metrics) in results {
        for index in 0..3 {
            values[index] += metrics[index];
        }
    }
    for value in &mut values {
        *value /= results.len() as f32;
    }
    values
}

fn average_metric_quintets(results: &[(GradientsParams, [f32; 5])]) -> [f32; 5] {
    let mut values = [0.0; 5];
    for (_, metrics) in results {
        for index in 0..5 {
            values[index] += metrics[index];
        }
    }
    for value in &mut values {
        *value /= results.len() as f32;
    }
    values
}

#[derive(Clone, Copy, Debug, Default)]
struct BatchDiagnostics {
    batches: usize,
    samples: usize,
    total_duration_ms: usize,
    max_duration_ms: usize,
    padded_duration_ms: usize,
    total_frames: usize,
    max_frames: usize,
    padded_frames: usize,
    feature_values: usize,
    target_tokens: usize,
    max_target_len: usize,
}

impl BatchDiagnostics {
    fn from_batches(batches: &[TrainBatch]) -> Self {
        let mut diagnostics = Self {
            batches: batches.len(),
            ..Self::default()
        };
        for batch in batches {
            diagnostics.samples += batch.batch_size;
            diagnostics.total_duration_ms += batch.durations_ms.iter().sum::<usize>();
            diagnostics.max_duration_ms = diagnostics
                .max_duration_ms
                .max(batch.durations_ms.iter().copied().max().unwrap_or(0));
            diagnostics.padded_duration_ms +=
                batch.batch_size * batch.durations_ms.iter().copied().max().unwrap_or(0);
            diagnostics.total_frames += batch.feature_lengths.iter().sum::<usize>();
            diagnostics.max_frames = diagnostics.max_frames.max(batch.max_frames);
            diagnostics.padded_frames += batch.batch_size * batch.max_frames;
            diagnostics.feature_values += batch.batch_size * batch.max_frames * batch.feature_dim;
            diagnostics.target_tokens += batch.target_lengths.iter().sum::<usize>();
            diagnostics.max_target_len = diagnostics.max_target_len.max(batch.max_target_len);
        }
        diagnostics
    }

    fn as_json(self) -> Value {
        json!({
            "batches": self.batches,
            "samples": self.samples,
            "total_duration_ms": self.total_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "padded_duration_ms": self.padded_duration_ms,
            "duration_min": self.total_duration_ms as f64 / 60_000.0,
            "padded_duration_min": self.padded_duration_ms as f64 / 60_000.0,
            "total_frames": self.total_frames,
            "max_frames": self.max_frames,
            "padded_frames": self.padded_frames,
            "feature_values": self.feature_values,
            "target_tokens": self.target_tokens,
            "max_target_len": self.max_target_len,
            "padding_ratio": if self.total_frames > 0 {
                Some(self.padded_frames as f64 / self.total_frames as f64)
            } else {
                None
            },
        })
    }

    fn throughput_json(self, elapsed: std::time::Duration) -> Value {
        let seconds = elapsed.as_secs_f64().max(1.0e-9);
        json!({
            "step_elapsed_sec": elapsed.as_secs_f64(),
            "samples_per_sec": self.samples as f64 / seconds,
            "frames_per_sec": self.total_frames as f64 / seconds,
            "feature_values_per_sec": self.feature_values as f64 / seconds,
        })
    }
}

fn should_log_progress(global_step: usize, log_every: usize) -> bool {
    global_step == 0 || (log_every > 0 && global_step % log_every == 0)
}

fn log_batch_start(
    logger: &mut RunLogger<'_>,
    epoch: usize,
    global_step: usize,
    devices: usize,
    diagnostics: BatchDiagnostics,
    dry_run: bool,
) -> Result<()> {
    if should_log_progress(global_step, logger.config.log_every) {
        let padding_ratio = if diagnostics.total_frames > 0 {
            diagnostics.padded_frames as f64 / diagnostics.total_frames as f64
        } else {
            0.0
        };
        log::info!(
            "batch_start epoch={} next_step={} devices={} micro_batches={} samples={} max_frames={} padded_frames={} feature_values={} padding_ratio={:.3} dry_run={}",
            epoch,
            global_step + 1,
            devices,
            diagnostics.batches,
            diagnostics.samples,
            diagnostics.max_frames,
            diagnostics.padded_frames,
            diagnostics.feature_values,
            padding_ratio,
            dry_run
        );
    }
    logger.log(
        "batch_start",
        json!({
            "epoch": epoch,
            "next_step": global_step + 1,
            "devices": devices,
            "dry_run": dry_run,
            "batch": diagnostics.as_json(),
        }),
    )
}

fn train_ctc_model<B, M>(
    mut model: M,
    config: &BurnTrainConfig,
    devices: &[B::Device],
) -> Result<TrainSummary>
where
    B: AutodiffBackend,
    M: TrainableCtc<B>,
    M::InnerModule: ValidCtc<B::InnerBackend>,
{
    let device = devices
        .first()
        .ok_or_else(|| anyhow!("at least one training device is required"))?;
    let mut optimizer = adamw_optimizer::<B, M>(config);
    let resume = load_resume_checkpoint(config)?;
    model = load_model_checkpoint(model, &resume, device)?;
    model = load_initial_weights(model, config, &resume, device)?;
    optimizer = load_optimizer_checkpoint::<B, M, _>(optimizer, &resume, device)?;
    let mut ema_model = initialize_ema_model::<B, M>(&model, config);
    ema_model = load_ema_checkpoint::<B, M>(ema_model, &resume, device)?;
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
    let mut logger = RunLogger::new(config, started)?;
    logger.log(
        "run_start",
        json!({
            "architecture": architecture_name(&config.architecture),
            "start_epoch": start_epoch,
            "global_step": global_step,
            "devices": devices.len(),
            "resume_from": config.resume_from,
            "dry_run": config.dry_run,
        }),
    )?;
    let mut accumulator = GradientsAccumulator::<M>::new();
    let mut accumulated_batches = 0usize;

    for epoch in start_epoch..=config.epochs {
        let mut train_batches = StreamingBatchLoader::new(
            config.train_manifest.clone(),
            config.batch_size,
            config.adaptive_batch,
            config.sort_by_length_desc,
            config.sort_buffer_size,
            config.input_dim,
            config.max_train_samples,
            config.max_audio_duration_ms,
            config.tokenizer_path.clone(),
            dataset_index_path(config, "train"),
            config.waveform_augment,
            feature_extractor_for_architecture(config),
        )?;
        while let Some(batch) = train_batches.next_batch()? {
            let batches = if !config.dry_run && devices.len() > 1 {
                collect_device_batches::<B>(batch, &mut train_batches, devices)?
            } else {
                vec![batch]
            };
            let step_started = Instant::now();
            let batch_diagnostics = BatchDiagnostics::from_batches(&batches);
            log_batch_start(
                &mut logger,
                epoch,
                global_step,
                batches.len(),
                batch_diagnostics,
                config.dry_run,
            )?;

            if !config.dry_run {
                let loss_value;
                if devices.len() > 1 {
                    let active_devices = &devices[..batches.len()];
                    let results = collect_parallel_gradients::<B, M, f32, _>(
                        &model,
                        batches,
                        active_devices,
                        device,
                        &|local_model, batch, local_device| {
                            let loss = ctc_loss_for_batch::<B, M>(
                                local_model,
                                batch,
                                config,
                                local_device,
                            );
                            let loss_value = scalar_value(loss.clone())?;
                            Ok((loss, loss_value))
                        },
                    )?;
                    loss_value =
                        results.iter().map(|(_, value)| *value).sum::<f32>() / results.len() as f32;
                    for (grads, _) in results {
                        accumulator.accumulate::<B>(&model, grads);
                        accumulated_batches += 1;
                    }
                } else {
                    let loss = ctc_loss_for_batch::<B, M>(&model, &batches[0], config, device);
                    loss_value = scalar_value(loss.clone())?;
                    let grads = GradientsParams::from_grads(loss.backward(), &model);
                    accumulator.accumulate::<B>(&model, grads);
                    accumulated_batches += 1;
                }
                last_train_loss = Some(loss_value);
                let pending_micro_batches = accumulated_batches;
                let (next_model, lr) = maybe_step_accumulated::<B, M, _>(
                    model,
                    &mut optimizer,
                    &mut accumulator,
                    &mut accumulated_batches,
                    &mut global_step,
                    epoch,
                    config,
                    false,
                );
                model = next_model;
                if let Some(lr) = lr {
                    update_ema_after_step::<B, M>(&mut ema_model, &model, config, global_step);
                    if global_step == 1 || global_step % config.log_every == 0 {
                        println!(
                            "epoch={epoch} step={global_step} lr={lr:.8} train_ctc_loss={loss_value:.6} elapsed_sec={:.1}",
                            started.elapsed().as_secs_f64()
                        );
                    }
                    logger.log(
                        "train_step",
                        json!({
                            "epoch": epoch,
                            "global_step": global_step,
                            "learning_rate": lr,
                            "losses": {"ctc": loss_value},
                            "batch": batch_diagnostics.as_json(),
                            "throughput": batch_diagnostics.throughput_json(step_started.elapsed()),
                            "micro_batches": pending_micro_batches,
                            "dry_run": false,
                        }),
                    )?;
                }
            } else {
                let loss = ctc_loss_for_batch::<B, M>(&model, &batches[0], config, device);
                let loss_value = scalar_value(loss.clone())?;
                last_train_loss = Some(loss_value);
                global_step += 1;
                if global_step == 1 || global_step % config.log_every == 0 {
                    println!(
                        "epoch={epoch} step={global_step} train_ctc_loss={loss_value:.6} elapsed_sec={:.1}",
                        started.elapsed().as_secs_f64()
                    );
                }
                logger.log(
                    "train_step",
                    json!({
                        "epoch": epoch,
                        "global_step": global_step,
                        "learning_rate": Value::Null,
                        "losses": {"ctc": loss_value},
                        "batch": batch_diagnostics.as_json(),
                        "throughput": batch_diagnostics.throughput_json(step_started.elapsed()),
                        "dry_run": true,
                    }),
                )?;
            }

            if global_step > 0
                && (config.dry_run || accumulated_batches == 0)
                && let (Some(_), Some(every)) = (&config.val_manifest, config.validate_every_steps)
            {
                if every > 0 && global_step % every == 0 {
                    logger.log(
                        "validation_start",
                        validation_start_event(epoch, Some(global_step), "ctc"),
                    )?;
                    let val = evaluate_ctc_model::<B, M>(&model, config, device, &decoder)?;
                    println!(
                        "{}",
                        format_validation_summary(epoch, Some(global_step), "val_ctc_loss", &val)
                    );
                    last_val_loss = Some(val.loss);
                    last_val_cer = val.cer;
                    last_val_wer = val.wer;
                    logger.log(
                        "validation",
                        validation_event(epoch, Some(global_step), "ctc", &val),
                    )?;
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
                        ema_model.as_ref(),
                    )?;
                }
            }
        }

        if !config.dry_run {
            let pending_micro_batches = accumulated_batches;
            let (next_model, lr) = maybe_step_accumulated::<B, M, _>(
                model,
                &mut optimizer,
                &mut accumulator,
                &mut accumulated_batches,
                &mut global_step,
                epoch,
                config,
                true,
            );
            model = next_model;
            if let Some(lr) = lr {
                update_ema_after_step::<B, M>(&mut ema_model, &model, config, global_step);
                println!(
                    "epoch={epoch} step={global_step} lr={lr:.8} flushed_accumulated_gradients=true elapsed_sec={:.1}",
                    started.elapsed().as_secs_f64()
                );
                logger.log(
                    "optimizer_flush",
                    json!({
                        "epoch": epoch,
                        "global_step": global_step,
                        "learning_rate": lr,
                        "micro_batches": pending_micro_batches,
                    }),
                )?;
            }
        }

        if config.val_manifest.is_some() {
            logger.log(
                "validation_start",
                validation_start_event(epoch, None, "ctc"),
            )?;
            let val = evaluate_ctc_model::<B, M>(&model, config, device, &decoder)?;
            println!(
                "{}",
                format_validation_summary(epoch, None, "val_ctc_loss", &val)
            );
            last_val_loss = Some(val.loss);
            last_val_cer = val.cer;
            last_val_wer = val.wer;
            logger.log("validation", validation_event(epoch, None, "ctc", &val))?;
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
            ema_model.as_ref(),
        )?;
    }

    let summary = TrainSummary {
        epochs: config.epochs,
        steps: global_step,
        last_train_loss,
        last_val_loss,
        last_val_cer,
        last_val_wer,
    };
    logger.log(
        "run_complete",
        json!({
            "epochs": summary.epochs,
            "steps": summary.steps,
            "last_train_loss": summary.last_train_loss,
            "last_val_loss": summary.last_val_loss,
            "last_val_cer": summary.last_val_cer,
            "last_val_wer": summary.last_val_wer,
        }),
    )?;
    Ok(summary)
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
    M::InnerModule: ValidCtc<B::InnerBackend>,
{
    let mut total = 0.0f64;
    let mut count = 0usize;
    let mut metrics = ValidationMetrics::default();
    let mut sample_predictions = Vec::new();
    let eval_model = model.valid();
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
        config.max_audio_duration_ms,
        config.tokenizer_path.clone(),
        dataset_index_path(config, "val"),
        WaveformAugmentConfig::default(),
        feature_extractor_for_architecture(config),
    )?;
    while let Some(batch) = batches.next_batch()? {
        let (logits_or_log_probs, output_lengths) = valid_ctc_logits_for_batch::<
            B::InnerBackend,
            M::InnerModule,
        >(&eval_model, &batch, device);
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
        collect_validation_sample_predictions(
            &mut sample_predictions,
            config.val_log_samples,
            &batch,
            &predictions,
            decoder,
        );
        count += 1;
    }
    if count == 0 {
        bail!("validation manifest is empty");
    }
    Ok(metrics.summary((total / count as f64) as f32, sample_predictions))
}

fn train_paraformer_model<B>(
    mut model: ParaformerV2<B>,
    config: &BurnTrainConfig,
    devices: &[B::Device],
) -> Result<TrainSummary>
where
    B: AutodiffBackend,
{
    let device = devices
        .first()
        .ok_or_else(|| anyhow!("at least one training device is required"))?;
    let mut optimizer = adamw_optimizer::<B, ParaformerV2<B>>(config);
    let resume = load_resume_checkpoint(config)?;
    model = load_model_checkpoint(model, &resume, device)?;
    model = load_initial_weights(model, config, &resume, device)?;
    optimizer = load_optimizer_checkpoint::<B, ParaformerV2<B>, _>(optimizer, &resume, device)?;
    let mut ema_model = initialize_ema_model::<B, ParaformerV2<B>>(&model, config);
    ema_model = load_ema_checkpoint::<B, ParaformerV2<B>>(ema_model, &resume, device)?;
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
    let mut logger = RunLogger::new(config, started)?;
    logger.log(
        "run_start",
        json!({
            "architecture": architecture_name(&config.architecture),
            "start_epoch": start_epoch,
            "global_step": global_step,
            "devices": devices.len(),
            "resume_from": config.resume_from,
            "dry_run": config.dry_run,
            "paraformer_enhanced": false,
        }),
    )?;
    let mut accumulator = GradientsAccumulator::<ParaformerV2<B>>::new();
    let mut accumulated_batches = 0usize;

    for epoch in start_epoch..=config.epochs {
        let mut train_batches = StreamingBatchLoader::new(
            config.train_manifest.clone(),
            config.batch_size,
            config.adaptive_batch,
            config.sort_by_length_desc,
            config.sort_buffer_size,
            config.input_dim,
            config.max_train_samples,
            config.max_audio_duration_ms,
            config.tokenizer_path.clone(),
            dataset_index_path(config, "train"),
            config.waveform_augment,
            feature_extractor_for_architecture(config),
        )?;
        while let Some(batch) = train_batches.next_batch()? {
            let batches = if !config.dry_run && devices.len() > 1 {
                collect_device_batches::<B>(batch, &mut train_batches, devices)?
            } else {
                vec![batch]
            };
            let step_started = Instant::now();
            let batch_diagnostics = BatchDiagnostics::from_batches(&batches);
            log_batch_start(
                &mut logger,
                epoch,
                global_step,
                batches.len(),
                batch_diagnostics,
                config.dry_run,
            )?;

            if !config.dry_run {
                let metric_values;
                if devices.len() > 1 {
                    let active_devices = &devices[..batches.len()];
                    let results = collect_parallel_gradients::<B, ParaformerV2<B>, [f32; 3], _>(
                        &model,
                        batches,
                        active_devices,
                        device,
                        &|local_model, batch, local_device| {
                            let loss_output = paraformer_loss_for_batch(
                                local_model,
                                batch,
                                config,
                                local_device,
                                true,
                            );
                            let metrics = [
                                scalar_value(loss_output.loss.clone())?,
                                scalar_value(loss_output.ctc_loss.clone())?,
                                scalar_value(loss_output.ce_loss.clone())?,
                            ];
                            Ok((loss_output.loss, metrics))
                        },
                    )?;
                    metric_values = average_metric_triplets(&results);
                    for (grads, _) in results {
                        accumulator.accumulate::<B>(&model, grads);
                        accumulated_batches += 1;
                    }
                } else {
                    let loss_output =
                        paraformer_loss_for_batch(&model, &batches[0], config, device, true);
                    metric_values = [
                        scalar_value(loss_output.loss.clone())?,
                        scalar_value(loss_output.ctc_loss.clone())?,
                        scalar_value(loss_output.ce_loss.clone())?,
                    ];
                    let grads = GradientsParams::from_grads(loss_output.loss.backward(), &model);
                    accumulator.accumulate::<B>(&model, grads);
                    accumulated_batches += 1;
                }
                last_train_loss = Some(metric_values[0]);
                let pending_micro_batches = accumulated_batches;
                let (next_model, lr) = maybe_step_accumulated::<B, ParaformerV2<B>, _>(
                    model,
                    &mut optimizer,
                    &mut accumulator,
                    &mut accumulated_batches,
                    &mut global_step,
                    epoch,
                    config,
                    false,
                );
                model = next_model;
                if let Some(lr) = lr {
                    update_ema_after_step::<B, ParaformerV2<B>>(
                        &mut ema_model,
                        &model,
                        config,
                        global_step,
                    );
                    if global_step == 1 || global_step % config.log_every == 0 {
                        println!(
                            "epoch={epoch} step={global_step} lr={lr:.8} train_loss={:.6} train_ctc_loss={:.6} train_ce_loss={:.6} elapsed_sec={:.1}",
                            metric_values[0],
                            metric_values[1],
                            metric_values[2],
                            started.elapsed().as_secs_f64()
                        );
                    }
                    logger.log(
                        "train_step",
                        json!({
                            "epoch": epoch,
                            "global_step": global_step,
                            "learning_rate": lr,
                            "losses": {
                                "total": metric_values[0],
                                "ctc": metric_values[1],
                                "ce": metric_values[2],
                            },
                            "batch": batch_diagnostics.as_json(),
                            "throughput": batch_diagnostics.throughput_json(step_started.elapsed()),
                            "micro_batches": pending_micro_batches,
                            "dry_run": false,
                        }),
                    )?;
                }
            } else {
                let loss_output =
                    paraformer_loss_for_batch(&model, &batches[0], config, device, true);
                let metric_values = [
                    scalar_value(loss_output.loss.clone())?,
                    scalar_value(loss_output.ctc_loss.clone())?,
                    scalar_value(loss_output.ce_loss.clone())?,
                ];
                last_train_loss = Some(metric_values[0]);
                global_step += 1;
                if global_step == 1 || global_step % config.log_every == 0 {
                    println!(
                        "epoch={epoch} step={global_step} train_loss={:.6} train_ctc_loss={:.6} train_ce_loss={:.6} elapsed_sec={:.1}",
                        metric_values[0],
                        metric_values[1],
                        metric_values[2],
                        started.elapsed().as_secs_f64()
                    );
                }
                logger.log(
                    "train_step",
                    json!({
                        "epoch": epoch,
                        "global_step": global_step,
                        "learning_rate": Value::Null,
                        "losses": {
                            "total": metric_values[0],
                            "ctc": metric_values[1],
                            "ce": metric_values[2],
                        },
                        "batch": batch_diagnostics.as_json(),
                        "throughput": batch_diagnostics.throughput_json(step_started.elapsed()),
                        "dry_run": true,
                    }),
                )?;
            }

            if global_step > 0
                && (config.dry_run || accumulated_batches == 0)
                && let (Some(_), Some(every)) = (&config.val_manifest, config.validate_every_steps)
            {
                if every > 0 && global_step % every == 0 {
                    logger.log(
                        "validation_start",
                        validation_start_event(epoch, Some(global_step), "loss"),
                    )?;
                    let val = evaluate_paraformer_model(&model, config, device, &decoder)?;
                    println!(
                        "{}",
                        format_validation_summary(epoch, Some(global_step), "val_loss", &val)
                    );
                    last_val_loss = Some(val.loss);
                    last_val_cer = val.cer;
                    last_val_wer = val.wer;
                    logger.log(
                        "validation",
                        validation_event(epoch, Some(global_step), "loss", &val),
                    )?;
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
                        ema_model.as_ref(),
                    )?;
                }
            }
        }

        if !config.dry_run {
            let pending_micro_batches = accumulated_batches;
            let (next_model, lr) = maybe_step_accumulated::<B, ParaformerV2<B>, _>(
                model,
                &mut optimizer,
                &mut accumulator,
                &mut accumulated_batches,
                &mut global_step,
                epoch,
                config,
                true,
            );
            model = next_model;
            if let Some(lr) = lr {
                update_ema_after_step::<B, ParaformerV2<B>>(
                    &mut ema_model,
                    &model,
                    config,
                    global_step,
                );
                println!(
                    "epoch={epoch} step={global_step} lr={lr:.8} flushed_accumulated_gradients=true elapsed_sec={:.1}",
                    started.elapsed().as_secs_f64()
                );
                logger.log(
                    "optimizer_flush",
                    json!({
                        "epoch": epoch,
                        "global_step": global_step,
                        "learning_rate": lr,
                        "micro_batches": pending_micro_batches,
                    }),
                )?;
            }
        }

        if config.val_manifest.is_some() {
            logger.log(
                "validation_start",
                validation_start_event(epoch, None, "loss"),
            )?;
            let val = evaluate_paraformer_model(&model, config, device, &decoder)?;
            println!(
                "{}",
                format_validation_summary(epoch, None, "val_loss", &val)
            );
            last_val_loss = Some(val.loss);
            last_val_cer = val.cer;
            last_val_wer = val.wer;
            logger.log("validation", validation_event(epoch, None, "loss", &val))?;
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
            ema_model.as_ref(),
        )?;
    }

    let summary = TrainSummary {
        epochs: config.epochs,
        steps: global_step,
        last_train_loss,
        last_val_loss,
        last_val_cer,
        last_val_wer,
    };
    logger.log(
        "run_complete",
        json!({
            "epochs": summary.epochs,
            "steps": summary.steps,
            "last_train_loss": summary.last_train_loss,
            "last_val_loss": summary.last_val_loss,
            "last_val_cer": summary.last_val_cer,
            "last_val_wer": summary.last_val_wer,
        }),
    )?;
    Ok(summary)
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
    let mut sample_predictions = Vec::new();
    let eval_model = model.valid();
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
        config.max_audio_duration_ms,
        config.tokenizer_path.clone(),
        dataset_index_path(config, "val"),
        WaveformAugmentConfig::default(),
        feature_extractor_for_architecture(config),
    )?;
    while let Some(batch) = batches.next_batch()? {
        let features = batch_features_tensor::<B::InnerBackend>(&batch, device);
        let output = eval_model.forward(features, batch.feature_lengths.clone());
        let loss = ctc_loss_from_log_probs(
            output.ctc_log_probs.clone(),
            output.encoder_lengths.clone(),
            &batch,
            config.blank_id,
            device,
        );
        total += f64::from(scalar_value(loss)?);
        let predictions = decode_validation_batch(
            output.ctc_log_probs,
            &output.encoder_lengths,
            config.blank_id,
            decoder,
        )?;
        metrics.update(&batch, &predictions, decoder);
        collect_validation_sample_predictions(
            &mut sample_predictions,
            config.val_log_samples,
            &batch,
            &predictions,
            decoder,
        );
        count += 1;
    }
    if count == 0 {
        bail!("validation manifest is empty");
    }
    Ok(metrics.summary((total / count as f64) as f32, sample_predictions))
}

fn train_enhanced_paraformer_model<B>(
    mut model: EnhancedParaformerV2<B>,
    config: &BurnTrainConfig,
    devices: &[B::Device],
) -> Result<TrainSummary>
where
    B: AutodiffBackend,
{
    let device = devices
        .first()
        .ok_or_else(|| anyhow!("at least one training device is required"))?;
    let mut optimizer = adamw_optimizer::<B, EnhancedParaformerV2<B>>(config);
    let resume = load_resume_checkpoint(config)?;
    model = load_model_checkpoint(model, &resume, device)?;
    model = load_initial_weights(model, config, &resume, device)?;
    optimizer =
        load_optimizer_checkpoint::<B, EnhancedParaformerV2<B>, _>(optimizer, &resume, device)?;
    let mut ema_model = initialize_ema_model::<B, EnhancedParaformerV2<B>>(&model, config);
    ema_model = load_ema_checkpoint::<B, EnhancedParaformerV2<B>>(ema_model, &resume, device)?;
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
    let mut logger = RunLogger::new(config, started)?;
    logger.log(
        "run_start",
        json!({
            "architecture": architecture_name(&config.architecture),
            "start_epoch": start_epoch,
            "global_step": global_step,
            "devices": devices.len(),
            "resume_from": config.resume_from,
            "dry_run": config.dry_run,
            "paraformer_enhanced": true,
        }),
    )?;
    let mut accumulator = GradientsAccumulator::<EnhancedParaformerV2<B>>::new();
    let mut accumulated_batches = 0usize;

    for epoch in start_epoch..=config.epochs {
        let mut train_batches = StreamingBatchLoader::new(
            config.train_manifest.clone(),
            config.batch_size,
            config.adaptive_batch,
            config.sort_by_length_desc,
            config.sort_buffer_size,
            config.input_dim,
            config.max_train_samples,
            config.max_audio_duration_ms,
            config.tokenizer_path.clone(),
            dataset_index_path(config, "train"),
            config.waveform_augment,
            feature_extractor_for_architecture(config),
        )?;
        while let Some(batch) = train_batches.next_batch()? {
            let batches = if !config.dry_run && devices.len() > 1 {
                collect_device_batches::<B>(batch, &mut train_batches, devices)?
            } else {
                vec![batch]
            };
            let step_started = Instant::now();
            let batch_diagnostics = BatchDiagnostics::from_batches(&batches);
            log_batch_start(
                &mut logger,
                epoch,
                global_step,
                batches.len(),
                batch_diagnostics,
                config.dry_run,
            )?;

            if !config.dry_run {
                let metric_values;
                if devices.len() > 1 {
                    let active_devices = &devices[..batches.len()];
                    let results =
                        collect_parallel_gradients::<B, EnhancedParaformerV2<B>, [f32; 5], _>(
                            &model,
                            batches,
                            active_devices,
                            device,
                            &|local_model, batch, local_device| {
                                let loss_output = enhanced_paraformer_loss_for_batch(
                                    local_model,
                                    batch,
                                    config,
                                    local_device,
                                    true,
                                );
                                let metrics = [
                                    scalar_value(loss_output.loss.clone())?,
                                    scalar_value(loss_output.ctc_loss.clone())?,
                                    scalar_value(loss_output.shallow_ctc_loss.clone())?,
                                    scalar_value(loss_output.ce_loss.clone())?,
                                    scalar_value(loss_output.boundary_loss.clone())?,
                                ];
                                Ok((loss_output.loss, metrics))
                            },
                        )?;
                    metric_values = average_metric_quintets(&results);
                    for (grads, _) in results {
                        accumulator.accumulate::<B>(&model, grads);
                        accumulated_batches += 1;
                    }
                } else {
                    let loss_output = enhanced_paraformer_loss_for_batch(
                        &model,
                        &batches[0],
                        config,
                        device,
                        true,
                    );
                    metric_values = [
                        scalar_value(loss_output.loss.clone())?,
                        scalar_value(loss_output.ctc_loss.clone())?,
                        scalar_value(loss_output.shallow_ctc_loss.clone())?,
                        scalar_value(loss_output.ce_loss.clone())?,
                        scalar_value(loss_output.boundary_loss.clone())?,
                    ];
                    let grads = GradientsParams::from_grads(loss_output.loss.backward(), &model);
                    accumulator.accumulate::<B>(&model, grads);
                    accumulated_batches += 1;
                }
                last_train_loss = Some(metric_values[0]);
                let pending_micro_batches = accumulated_batches;
                let (next_model, lr) = maybe_step_accumulated::<B, EnhancedParaformerV2<B>, _>(
                    model,
                    &mut optimizer,
                    &mut accumulator,
                    &mut accumulated_batches,
                    &mut global_step,
                    epoch,
                    config,
                    false,
                );
                model = next_model;
                if let Some(lr) = lr {
                    update_ema_after_step::<B, EnhancedParaformerV2<B>>(
                        &mut ema_model,
                        &model,
                        config,
                        global_step,
                    );
                    if global_step == 1 || global_step % config.log_every == 0 {
                        println!(
                            "epoch={epoch} step={global_step} lr={lr:.8} train_loss={:.6} train_ctc_loss={:.6} train_shallow_ctc_loss={:.6} train_ce_loss={:.6} train_boundary_loss={:.6} elapsed_sec={:.1}",
                            metric_values[0],
                            metric_values[1],
                            metric_values[2],
                            metric_values[3],
                            metric_values[4],
                            started.elapsed().as_secs_f64()
                        );
                    }
                    logger.log(
                        "train_step",
                        json!({
                            "epoch": epoch,
                            "global_step": global_step,
                            "learning_rate": lr,
                            "losses": {
                                "total": metric_values[0],
                                "ctc": metric_values[1],
                                "shallow_ctc": metric_values[2],
                                "ce": metric_values[3],
                                "boundary": metric_values[4],
                            },
                            "batch": batch_diagnostics.as_json(),
                            "throughput": batch_diagnostics.throughput_json(step_started.elapsed()),
                            "micro_batches": pending_micro_batches,
                            "dry_run": false,
                        }),
                    )?;
                }
            } else {
                let loss_output =
                    enhanced_paraformer_loss_for_batch(&model, &batches[0], config, device, true);
                let metric_values = [
                    scalar_value(loss_output.loss.clone())?,
                    scalar_value(loss_output.ctc_loss.clone())?,
                    scalar_value(loss_output.shallow_ctc_loss.clone())?,
                    scalar_value(loss_output.ce_loss.clone())?,
                    scalar_value(loss_output.boundary_loss.clone())?,
                ];
                last_train_loss = Some(metric_values[0]);
                global_step += 1;
                if global_step == 1 || global_step % config.log_every == 0 {
                    println!(
                        "epoch={epoch} step={global_step} train_loss={:.6} train_ctc_loss={:.6} train_shallow_ctc_loss={:.6} train_ce_loss={:.6} train_boundary_loss={:.6} elapsed_sec={:.1}",
                        metric_values[0],
                        metric_values[1],
                        metric_values[2],
                        metric_values[3],
                        metric_values[4],
                        started.elapsed().as_secs_f64()
                    );
                }
                logger.log(
                    "train_step",
                    json!({
                        "epoch": epoch,
                        "global_step": global_step,
                        "learning_rate": Value::Null,
                        "losses": {
                            "total": metric_values[0],
                            "ctc": metric_values[1],
                            "shallow_ctc": metric_values[2],
                            "ce": metric_values[3],
                            "boundary": metric_values[4],
                        },
                        "batch": batch_diagnostics.as_json(),
                        "throughput": batch_diagnostics.throughput_json(step_started.elapsed()),
                        "dry_run": true,
                    }),
                )?;
            }

            if global_step > 0
                && (config.dry_run || accumulated_batches == 0)
                && let (Some(_), Some(every)) = (&config.val_manifest, config.validate_every_steps)
            {
                if every > 0 && global_step % every == 0 {
                    logger.log(
                        "validation_start",
                        validation_start_event(epoch, Some(global_step), "loss"),
                    )?;
                    let val = evaluate_enhanced_paraformer_model(&model, config, device, &decoder)?;
                    println!(
                        "{}",
                        format_validation_summary(epoch, Some(global_step), "val_loss", &val)
                    );
                    last_val_loss = Some(val.loss);
                    last_val_cer = val.cer;
                    last_val_wer = val.wer;
                    logger.log(
                        "validation",
                        validation_event(epoch, Some(global_step), "loss", &val),
                    )?;
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
                        ema_model.as_ref(),
                    )?;
                }
            }
        }

        if !config.dry_run {
            let pending_micro_batches = accumulated_batches;
            let (next_model, lr) = maybe_step_accumulated::<B, EnhancedParaformerV2<B>, _>(
                model,
                &mut optimizer,
                &mut accumulator,
                &mut accumulated_batches,
                &mut global_step,
                epoch,
                config,
                true,
            );
            model = next_model;
            if let Some(lr) = lr {
                update_ema_after_step::<B, EnhancedParaformerV2<B>>(
                    &mut ema_model,
                    &model,
                    config,
                    global_step,
                );
                println!(
                    "epoch={epoch} step={global_step} lr={lr:.8} flushed_accumulated_gradients=true elapsed_sec={:.1}",
                    started.elapsed().as_secs_f64()
                );
                logger.log(
                    "optimizer_flush",
                    json!({
                        "epoch": epoch,
                        "global_step": global_step,
                        "learning_rate": lr,
                        "micro_batches": pending_micro_batches,
                    }),
                )?;
            }
        }

        if config.val_manifest.is_some() {
            logger.log(
                "validation_start",
                validation_start_event(epoch, None, "loss"),
            )?;
            let val = evaluate_enhanced_paraformer_model(&model, config, device, &decoder)?;
            println!(
                "{}",
                format_validation_summary(epoch, None, "val_loss", &val)
            );
            last_val_loss = Some(val.loss);
            last_val_cer = val.cer;
            last_val_wer = val.wer;
            logger.log("validation", validation_event(epoch, None, "loss", &val))?;
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
            ema_model.as_ref(),
        )?;
    }

    let summary = TrainSummary {
        epochs: config.epochs,
        steps: global_step,
        last_train_loss,
        last_val_loss,
        last_val_cer,
        last_val_wer,
    };
    logger.log(
        "run_complete",
        json!({
            "epochs": summary.epochs,
            "steps": summary.steps,
            "last_train_loss": summary.last_train_loss,
            "last_val_loss": summary.last_val_loss,
            "last_val_cer": summary.last_val_cer,
            "last_val_wer": summary.last_val_wer,
        }),
    )?;
    Ok(summary)
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
    let mut sample_predictions = Vec::new();
    let eval_model = model.valid();
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
        config.max_audio_duration_ms,
        config.tokenizer_path.clone(),
        dataset_index_path(config, "val"),
        WaveformAugmentConfig::default(),
        feature_extractor_for_architecture(config),
    )?;
    while let Some(batch) = batches.next_batch()? {
        let features = batch_features_tensor::<B::InnerBackend>(&batch, device);
        let (ctc_log_probs, encoder_lengths) =
            eval_model.ctc_log_probs(features, batch.feature_lengths.clone());
        let loss = ctc_loss_from_log_probs(
            ctc_log_probs.clone(),
            encoder_lengths.clone(),
            &batch,
            config.blank_id,
            device,
        );
        total += f64::from(scalar_value(loss)?);
        let predictions =
            decode_validation_batch(ctc_log_probs, &encoder_lengths, config.blank_id, decoder)?;
        metrics.update(&batch, &predictions, decoder);
        collect_validation_sample_predictions(
            &mut sample_predictions,
            config.val_log_samples,
            &batch,
            &predictions,
            decoder,
        );
        count += 1;
    }
    if count == 0 {
        bail!("validation manifest is empty");
    }
    Ok(metrics.summary((total / count as f64) as f32, sample_predictions))
}

fn paraformer_loss_for_batch<B>(
    model: &ParaformerV2<B>,
    batch: &TrainBatch,
    config: &BurnTrainConfig,
    device: &B::Device,
    augment: bool,
) -> crate::paraformer::ParaformerLossOutput<B>
where
    B: AutodiffBackend,
{
    let features = Tensor::<B, 3>::from_data(
        TensorData::new(
            batch_features_values(batch, augment.then_some(config.spec_augment)),
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
    augment: bool,
) -> crate::paraformer::EnhancedParaformerLossOutput<B>
where
    B: AutodiffBackend,
{
    let features = Tensor::<B, 3>::from_data(
        TensorData::new(
            batch_features_values(batch, augment.then_some(config.spec_augment)),
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
    config: &BurnTrainConfig,
    device: &B::Device,
) -> Tensor<B, 1>
where
    B: AutodiffBackend,
    M: TrainableCtc<B>,
{
    let features = batch_features_tensor_with_augment(batch, device, Some(config.spec_augment));
    let (logits_or_log_probs, output_lengths) =
        model.ctc_logits(features, batch.feature_lengths.clone());
    ctc_loss_from_logits(
        logits_or_log_probs,
        output_lengths,
        batch,
        config.blank_id,
        device,
    )
}

fn valid_ctc_logits_for_batch<B, M>(
    model: &M,
    batch: &TrainBatch,
    device: &B::Device,
) -> (Tensor<B, 3>, Vec<usize>)
where
    B: Backend,
    M: ValidCtc<B>,
{
    let features = batch_features_tensor(batch, device);
    model.ctc_logits(features, batch.feature_lengths.clone())
}

fn batch_features_tensor<B: Backend>(batch: &TrainBatch, device: &B::Device) -> Tensor<B, 3> {
    batch_features_tensor_with_augment(batch, device, None)
}

fn batch_features_tensor_with_augment<B: Backend>(
    batch: &TrainBatch,
    device: &B::Device,
    spec_augment: Option<SpecAugmentConfig>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(
        TensorData::new(
            batch_features_values(batch, spec_augment),
            [batch.batch_size, batch.max_frames, batch.feature_dim],
        ),
        device,
    )
}

fn batch_features_values(batch: &TrainBatch, spec_augment: Option<SpecAugmentConfig>) -> Vec<f32> {
    let mut features = batch.features.clone();
    if let Some(config) = spec_augment.filter(SpecAugmentConfig::is_enabled) {
        apply_spec_augment(&mut features, batch, config);
    }
    features
}

fn apply_spec_augment(features: &mut [f32], batch: &TrainBatch, config: SpecAugmentConfig) {
    let mut rng = rand::rng();
    for sample_index in 0..batch.batch_size {
        let length = batch.feature_lengths[sample_index].min(batch.max_frames);
        for _ in 0..config.time_masks {
            if length == 0 || config.time_mask_max_frames == 0 {
                continue;
            }
            let width = rng.random_range(1..=config.time_mask_max_frames.min(length));
            let start = rng.random_range(0..=length - width);
            for frame in start..start + width {
                let offset = (sample_index * batch.max_frames + frame) * batch.feature_dim;
                features[offset..offset + batch.feature_dim].fill(0.0);
            }
        }
        for _ in 0..config.frequency_masks {
            if batch.feature_dim == 0 || config.frequency_mask_max_bins == 0 {
                continue;
            }
            let width = rng.random_range(1..=config.frequency_mask_max_bins.min(batch.feature_dim));
            let start = rng.random_range(0..=batch.feature_dim - width);
            for frame in 0..length {
                let offset = (sample_index * batch.max_frames + frame) * batch.feature_dim + start;
                features[offset..offset + width].fill(0.0);
            }
        }
    }
}

fn ctc_loss_from_logits<B: Backend>(
    logits_or_log_probs: Tensor<B, 3>,
    output_lengths: Vec<usize>,
    batch: &TrainBatch,
    blank_id: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let log_probs = log_softmax(logits_or_log_probs, 2).swap_dims(0, 1);
    ctc_loss_from_time_major_log_probs(log_probs, output_lengths, batch, blank_id, device)
}

fn ctc_loss_from_log_probs<B: Backend>(
    log_probs: Tensor<B, 3>,
    output_lengths: Vec<usize>,
    batch: &TrainBatch,
    blank_id: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    ctc_loss_from_time_major_log_probs(
        log_probs.swap_dims(0, 1),
        output_lengths,
        batch,
        blank_id,
        device,
    )
}

fn ctc_loss_from_time_major_log_probs<B: Backend>(
    log_probs: Tensor<B, 3>,
    output_lengths: Vec<usize>,
    batch: &TrainBatch,
    blank_id: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
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

    fn summary(
        self,
        loss: f32,
        sample_predictions: Vec<ValidationSamplePrediction>,
    ) -> ValidationSummary {
        ValidationSummary {
            loss,
            token_error_rate: self.token_stats.rate(),
            cer: self.char_stats.rate(),
            wer: self.word_stats.rate(),
            decoded_samples: self.decoded_samples,
            sample_predictions,
        }
    }
}

fn collect_validation_sample_predictions(
    samples: &mut Vec<ValidationSamplePrediction>,
    limit: usize,
    batch: &TrainBatch,
    predictions: &[Vec<u32>],
    decoder: &ValidationDecoder,
) {
    if limit == 0 || samples.len() >= limit {
        return;
    }
    for sample_index in 0..batch.batch_size {
        if samples.len() >= limit {
            break;
        }
        let prediction_tokens = predictions.get(sample_index).cloned().unwrap_or_default();
        let reference_tokens = target_tokens_for_batch_item(batch, sample_index);
        let prediction_text = normalize_spaces(&decoder.decode_tokens(&prediction_tokens));
        let reference_text = normalize_spaces(&decoder.reference_text(
            batch.reference_texts[sample_index].as_ref(),
            &reference_tokens,
        ));
        samples.push(ValidationSamplePrediction {
            id: batch.ids[sample_index].clone(),
            prediction_text,
            reference_text,
            prediction_tokens,
            reference_tokens,
        });
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
        .cast(FloatDType::F32)
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

fn validation_event(
    epoch: usize,
    step: Option<usize>,
    loss_name: &str,
    summary: &ValidationSummary,
) -> Value {
    json!({
        "epoch": epoch,
        "global_step": step,
        "loss_name": loss_name,
        "loss": summary.loss,
        "cer": summary.cer,
        "wer": summary.wer,
        "token_error_rate": summary.token_error_rate,
        "decoded_samples": summary.decoded_samples,
        "samples": summary.sample_predictions.iter().map(|sample| {
            json!({
                "id": sample.id,
                "prediction_text": sample.prediction_text,
                "reference_text": sample.reference_text,
                "prediction_tokens": sample.prediction_tokens,
                "reference_tokens": sample.reference_tokens,
            })
        }).collect::<Vec<_>>(),
    })
}

fn validation_start_event(epoch: usize, step: Option<usize>, loss_name: &str) -> Value {
    json!({
        "epoch": epoch,
        "global_step": step,
        "loss_name": loss_name,
    })
}

fn load_inference_model<B, M>(
    model: M,
    checkpoint_dir: &Path,
    use_ema: bool,
    device: &B::Device,
) -> Result<M>
where
    B: Backend,
    M: Module<B>,
{
    let stem = if use_ema { "ema_model" } else { "model" };
    let path = checkpoint_file_path(checkpoint_dir, stem);
    if !path.exists() {
        bail!("checkpoint model file does not exist: {}", path.display());
    }
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .load_file(
            checkpoint_base_path(checkpoint_dir, stem),
            &recorder,
            device,
        )
        .with_context(|| format!("failed to load {}", path.display()))
}

fn infer_ctc_batches<B, M, F>(
    model: &M,
    config: &BurnTrainConfig,
    device: &B::Device,
    decoder: &ValidationDecoder,
    mut writer: Option<&mut fs::File>,
    forward: F,
) -> Result<usize>
where
    B: Backend,
    F: Fn(&M, Tensor<B, 3>, Vec<usize>) -> (Tensor<B, 3>, Vec<usize>),
{
    let mut batches = StreamingBatchLoader::new(
        config.train_manifest.clone(),
        config.batch_size,
        config.adaptive_batch,
        config.sort_by_length_desc,
        config.sort_buffer_size,
        config.input_dim,
        config.max_train_samples,
        config.max_audio_duration_ms,
        config.tokenizer_path.clone(),
        None,
        WaveformAugmentConfig::default(),
        feature_extractor_for_architecture(config),
    )?;
    let mut decoded_samples = 0usize;
    while let Some(batch) = batches.next_batch()? {
        let features = batch_features_tensor::<B>(&batch, device);
        let (logits_or_log_probs, output_lengths) =
            forward(model, features, batch.feature_lengths.clone());
        let predictions = decode_validation_batch(
            logits_or_log_probs,
            &output_lengths,
            config.blank_id,
            decoder,
        )?;
        for sample_index in 0..batch.batch_size {
            let tokens = predictions.get(sample_index).cloned().unwrap_or_default();
            let text = decoder.decode_tokens(&tokens);
            let item = json!({
                "id": batch.ids[sample_index],
                "text": normalize_spaces(&text),
                "tokens": tokens,
                "reference_text": batch.reference_texts[sample_index],
            });
            if let Some(writer) = writer.as_deref_mut() {
                writeln!(writer, "{}", serde_json::to_string(&item)?)?;
            } else {
                println!("{}", serde_json::to_string(&item)?);
            }
            decoded_samples += 1;
        }
    }
    Ok(decoded_samples)
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
    let mut ids = Vec::with_capacity(batch_size);
    let mut feature_lengths = Vec::with_capacity(batch_size);
    let mut durations_ms = Vec::with_capacity(batch_size);
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
        ids.push(record.id.clone());
        feature_lengths.push(record.rows);
        durations_ms.push(record.duration_ms);
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
        ids,
        features,
        batch_size,
        max_frames,
        feature_dim: expected_dim,
        feature_lengths,
        durations_ms,
        targets,
        max_target_len,
        target_lengths,
        reference_texts,
    })
}

fn duration_ms_from_seconds(seconds: f64) -> Option<usize> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }
    Some((seconds * 1000.0).ceil().max(1.0) as usize)
}

fn duration_ms_from_audio(sample_count: usize, sample_rate: u32) -> usize {
    duration_ms_from_seconds(sample_count as f64 / sample_rate.max(1) as f64).unwrap_or(1)
}

fn estimated_duration_ms_from_rows(rows: usize, frontend: &FeatureExtractorConfig) -> usize {
    match frontend {
        FeatureExtractorConfig::Audio(config) => {
            let samples = rows.saturating_mul(config.hop_length);
            duration_ms_from_audio(samples, config.sample_rate)
        }
        FeatureExtractorConfig::W2vBert(config) => {
            let samples = rows
                .saturating_mul(160)
                .saturating_mul(config.stride.max(1));
            duration_ms_from_audio(samples, config.sample_rate)
        }
    }
}

fn json_duration_ms(value: &Value) -> Option<usize> {
    ["duration_ms", "audio_duration_ms"]
        .into_iter()
        .find_map(|name| value.get(name).and_then(Value::as_u64))
        .map(|value| value.max(1) as usize)
        .or_else(|| {
            [
                "duration",
                "duration_sec",
                "duration_secs",
                "duration_seconds",
                "audio_duration",
                "audio_duration_sec",
                "audio_duration_seconds",
            ]
            .into_iter()
            .find_map(|name| value.get(name).and_then(Value::as_f64))
            .and_then(duration_ms_from_seconds)
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

pub fn extract_feature_records(
    path: &Path,
    limit: Option<usize>,
    tokenizer_path: Option<&Path>,
    audio_frontend: &FeatureExtractorConfig,
    max_audio_duration_ms: Option<usize>,
) -> Result<Vec<FeatureRecord>> {
    extract_feature_records_with_progress(
        path,
        limit,
        tokenizer_path,
        audio_frontend,
        max_audio_duration_ms,
        |_| Ok(()),
    )
}

#[derive(Clone, Debug)]
pub struct FeatureExtractionProgress {
    pub input: PathBuf,
    pub records: usize,
    pub skipped_duration: usize,
    pub last_id: Option<String>,
    pub last_rows: Option<usize>,
    pub last_duration_ms: Option<usize>,
}

enum ExtractedFeatureRecord {
    Keep(FeatureRecord),
    SkipDuration {
        id: String,
        rows: usize,
        duration_ms: usize,
    },
}

pub fn extract_feature_records_with_progress(
    path: &Path,
    limit: Option<usize>,
    tokenizer_path: Option<&Path>,
    audio_frontend: &FeatureExtractorConfig,
    max_audio_duration_ms: Option<usize>,
    mut on_progress: impl FnMut(&FeatureExtractionProgress) -> Result<()>,
) -> Result<Vec<FeatureRecord>> {
    let files = manifest_files(path)?;
    let tokenizer = tokenizer_path
        .map(load_sentencepiece_transcript_tokenizer)
        .transpose()?;
    let mut records = Vec::new();
    let mut skipped_duration = 0usize;
    let audio_decode = AudioDecodeConfig::default();
    for file in files {
        if limit.is_some_and(|limit| records.len() >= limit) {
            break;
        }
        let base_dir = file.parent().unwrap_or_else(|| Path::new("."));
        let raw_lines = if is_audio_dataset_file(&file) {
            vec![RawManifestLine::from_audio_path(file.clone())]
        } else if is_parquet_file(&file) {
            parquet_raw_lines(&file)?
        } else {
            let text = fs::read_to_string(&file)
                .with_context(|| format!("failed to read manifest {}", file.display()))?;
            text.lines()
                .enumerate()
                .filter_map(|(line_index, line)| {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with('#') {
                        None
                    } else {
                        Some(RawManifestLine {
                            manifest_path: file.clone(),
                            base_dir: base_dir.to_path_buf(),
                            byte_offset: 0,
                            line_number: line_index + 1,
                            line: trimmed.to_string(),
                        })
                    }
                })
                .collect()
        };
        let extracted = raw_lines
            .par_iter()
            .map(|raw| {
                let record = raw.parse_record(
                    tokenizer.as_ref(),
                    &audio_decode,
                    audio_frontend,
                    WaveformAugmentConfig::default(),
                )?;
                if max_audio_duration_ms.is_some_and(|max| record.duration_ms > max) {
                    Ok(ExtractedFeatureRecord::SkipDuration {
                        id: record.id,
                        rows: record.rows,
                        duration_ms: record.duration_ms,
                    })
                } else {
                    Ok(ExtractedFeatureRecord::Keep(record))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        for extracted in extracted {
            if limit.is_some_and(|limit| records.len() >= limit) {
                break;
            }
            match extracted {
                ExtractedFeatureRecord::SkipDuration {
                    id,
                    rows,
                    duration_ms,
                } => {
                    skipped_duration += 1;
                    on_progress(&FeatureExtractionProgress {
                        input: file.clone(),
                        records: records.len(),
                        skipped_duration,
                        last_id: Some(id),
                        last_rows: Some(rows),
                        last_duration_ms: Some(duration_ms),
                    })?;
                }
                ExtractedFeatureRecord::Keep(record) => {
                    let progress = FeatureExtractionProgress {
                        input: file.clone(),
                        records: records.len() + 1,
                        skipped_duration,
                        last_id: Some(record.id.clone()),
                        last_rows: Some(record.rows),
                        last_duration_ms: Some(record.duration_ms),
                    };
                    records.push(record);
                    on_progress(&progress)?;
                }
            }
        }
    }
    if records.is_empty() {
        bail!(
            "manifest {} contains no records after filtering",
            path.display()
        );
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
    max_audio_duration_ms: Option<usize>,
    tokenizer: Option<SentencePieceTokenizer>,
    audio_decode: AudioDecodeConfig,
    audio_frontend: FeatureExtractorConfig,
    waveform_augment: WaveformAugmentConfig,
    pending: Option<FeatureRecord>,
    raw_pending: VecDeque<RawManifestLine>,
    sort_buffer: Vec<FeatureRecordMetadata>,
    index_records: Vec<FeatureRecordMetadata>,
    index_cursor: usize,
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
        max_audio_duration_ms: Option<usize>,
        tokenizer_path: Option<PathBuf>,
        index_path: Option<PathBuf>,
        waveform_augment: WaveformAugmentConfig,
        audio_frontend: FeatureExtractorConfig,
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
        let tokenizer = tokenizer_path
            .as_deref()
            .map(load_sentencepiece_transcript_tokenizer)
            .transpose()?;
        let files = manifest_files(&manifest)?;
        let index_records = if sort_by_length_desc {
            index_path
                .as_deref()
                .map(|path| load_or_build_dataset_index(&files, path))
                .transpose()?
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        Ok(Self {
            files,
            file_index: 0,
            current: None,
            batch_size,
            adaptive_batch,
            sort_by_length_desc,
            sort_buffer_size,
            expected_dim,
            yielded: 0,
            limit,
            max_audio_duration_ms,
            tokenizer,
            audio_decode: AudioDecodeConfig::default(),
            audio_frontend,
            waveform_augment,
            pending: None,
            raw_pending: VecDeque::new(),
            sort_buffer: Vec::new(),
            index_records,
            index_cursor: 0,
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
                    if self
                        .max_audio_duration_ms
                        .is_some_and(|max_duration_ms| record.duration_ms > max_duration_ms)
                    {
                        continue;
                    }
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
        if !self.index_records.is_empty() {
            return self.next_indexed_sorted_record();
        }
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
            let metadata = self.sort_buffer.remove(0);
            metadata
                .load_record(
                    self.tokenizer.as_ref(),
                    &self.audio_decode,
                    &self.audio_frontend,
                    self.waveform_augment,
                )
                .map(Some)
        }
    }

    fn next_indexed_sorted_record(&mut self) -> Result<Option<FeatureRecord>> {
        if self.index_cursor >= self.index_records.len() {
            return Ok(None);
        }
        let metadata = self.index_records[self.index_cursor].clone();
        self.index_cursor += 1;
        metadata
            .load_record(
                self.tokenizer.as_ref(),
                &self.audio_decode,
                &self.audio_frontend,
                self.waveform_augment,
            )
            .map(Some)
    }

    fn next_raw_record(&mut self) -> Result<Option<FeatureRecord>> {
        self.next_raw_line()?
            .map(|raw| {
                raw.parse_record(
                    self.tokenizer.as_ref(),
                    &self.audio_decode,
                    &self.audio_frontend,
                    self.waveform_augment,
                )
            })
            .transpose()
    }

    fn next_raw_metadata(&mut self) -> Result<Option<FeatureRecordMetadata>> {
        self.next_raw_line()?
            .map(|raw| raw.parse_metadata())
            .transpose()
    }

    fn next_raw_line(&mut self) -> Result<Option<RawManifestLine>> {
        loop {
            if let Some(raw) = self.raw_pending.pop_front() {
                return Ok(Some(raw));
            }
            if self.current.is_none() {
                if self.file_index >= self.files.len() {
                    return Ok(None);
                }
                let path = self.files[self.file_index].clone();
                self.file_index += 1;
                if is_audio_dataset_file(&path) {
                    return Ok(Some(RawManifestLine::from_audio_path(path)));
                }
                if is_parquet_file(&path) {
                    self.raw_pending = parquet_raw_lines(&path)?.into();
                    continue;
                }
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
    fn from_audio_path(path: PathBuf) -> Self {
        let base_dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        Self {
            manifest_path: path,
            base_dir,
            byte_offset: 0,
            line_number: 1,
            line: String::new(),
        }
    }

    fn parse_record(
        &self,
        tokenizer: Option<&SentencePieceTokenizer>,
        audio_decode: &AudioDecodeConfig,
        audio_frontend: &FeatureExtractorConfig,
        waveform_augment: WaveformAugmentConfig,
    ) -> Result<FeatureRecord> {
        if is_parquet_file(&self.manifest_path) {
            return parse_parquet_record(
                &self.manifest_path,
                self.line_number.saturating_sub(1),
                &self.base_dir,
                tokenizer,
                audio_decode,
                audio_frontend,
                waveform_augment,
            );
        }
        if is_audio_dataset_file(&self.manifest_path) {
            return parse_raw_audio_record(
                &self.manifest_path,
                &self.base_dir,
                tokenizer,
                audio_decode,
                audio_frontend,
                waveform_augment,
            );
        }
        if self.line.starts_with('{') {
            parse_json_record(
                &self.line,
                &self.base_dir,
                self.line_number,
                tokenizer,
                audio_decode,
                audio_frontend,
                waveform_augment,
            )
        } else {
            parse_tsv_record(&self.line, &self.base_dir, self.line_number)
        }
    }

    fn parse_metadata(&self) -> Result<FeatureRecordMetadata> {
        if is_parquet_file(&self.manifest_path) {
            let (id, rows) = parse_parquet_raw_line_metadata(&self.line)?;
            return Ok(FeatureRecordMetadata {
                manifest_path: self.manifest_path.clone(),
                base_dir: self.base_dir.clone(),
                byte_offset: self.byte_offset,
                line_number: self.line_number,
                id,
                rows,
            });
        }
        if is_audio_dataset_file(&self.manifest_path) {
            return Ok(raw_audio_metadata(&self.manifest_path, &self.base_dir));
        }
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
    fn load_record(
        &self,
        tokenizer: Option<&SentencePieceTokenizer>,
        audio_decode: &AudioDecodeConfig,
        audio_frontend: &FeatureExtractorConfig,
        waveform_augment: WaveformAugmentConfig,
    ) -> Result<FeatureRecord> {
        if is_parquet_file(&self.manifest_path) {
            return parse_parquet_record(
                &self.manifest_path,
                self.line_number.saturating_sub(1),
                &self.base_dir,
                tokenizer,
                audio_decode,
                audio_frontend,
                waveform_augment,
            );
        }
        if is_audio_dataset_file(&self.manifest_path) {
            return parse_raw_audio_record(
                &self.manifest_path,
                &self.base_dir,
                tokenizer,
                audio_decode,
                audio_frontend,
                waveform_augment,
            );
        }
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
            parse_json_record(
                line,
                &self.base_dir,
                self.line_number,
                tokenizer,
                audio_decode,
                audio_frontend,
                waveform_augment,
            )
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
            collect_raw_audio_files(path, &mut files)?;
            files.sort();
            if files.is_empty() {
                bail!(
                    "manifest directory {} contains no .jsonl/.json files or raw audio files",
                    path.display()
                );
            }
        }
        return Ok(files);
    }
    if !path.exists() {
        bail!("manifest path does not exist: {}", path.display());
    }
    Ok(vec![path.to_path_buf()])
}

fn is_manifest_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .is_some_and(|extension| matches!(extension.as_str(), "jsonl" | "json" | "parquet"))
}

fn is_parquet_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("parquet"))
}

fn is_audio_dataset_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .is_some_and(|extension| {
                matches!(
                    extension.as_str(),
                    "wav" | "flac" | "mp3" | "ogg" | "opus" | "m4a" | "aac" | "webm"
                )
            })
}

fn collect_raw_audio_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)
        .with_context(|| format!("failed to read audio directory {}", dir.display()))?
    {
        let path = entry
            .with_context(|| format!("failed to read entry in {}", dir.display()))?
            .path();
        if path.is_dir() {
            collect_raw_audio_files(&path, files)?;
        } else if is_audio_dataset_file(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn dataset_index_path(config: &BurnTrainConfig, split: &str) -> Option<PathBuf> {
    config
        .dataset_index_dir
        .as_ref()
        .map(|dir| dir.join(format!("{split}.index.json")))
}

fn load_or_build_dataset_index(
    files: &[PathBuf],
    index_path: &Path,
) -> Result<Vec<FeatureRecordMetadata>> {
    if let Some(records) = load_dataset_index_if_fresh(files, index_path)? {
        return Ok(records);
    }
    let records = build_dataset_index(files)?;
    write_dataset_index(files, index_path, &records)?;
    Ok(records)
}

fn load_dataset_index_if_fresh(
    files: &[PathBuf],
    index_path: &Path,
) -> Result<Option<Vec<FeatureRecordMetadata>>> {
    if !index_path.exists() {
        return Ok(None);
    }
    let value: Value = serde_json::from_str(
        &fs::read_to_string(index_path)
            .with_context(|| format!("failed to read dataset index {}", index_path.display()))?,
    )
    .with_context(|| format!("failed to parse dataset index {}", index_path.display()))?;
    let Some(files_value) = value.get("files").and_then(Value::as_array) else {
        return Ok(None);
    };
    if files_value.len() != files.len() {
        return Ok(None);
    }
    for (actual, cached) in files.iter().zip(files_value) {
        if cached.get("path").and_then(Value::as_str) != Some(&actual.to_string_lossy()) {
            return Ok(None);
        }
        let signature = file_signature(actual)?;
        if cached.get("len").and_then(Value::as_u64) != Some(signature.0)
            || cached.get("modified_ms").and_then(Value::as_u64) != Some(signature.1)
        {
            return Ok(None);
        }
    }
    let mut records = match value.get("records").and_then(Value::as_array) {
        Some(records) => records
            .iter()
            .map(dataset_index_record_from_json)
            .collect::<Result<Vec<_>>>()?,
        None => Vec::new(),
    };
    sort_index_records(&mut records);
    Ok(Some(records))
}

fn build_dataset_index(files: &[PathBuf]) -> Result<Vec<FeatureRecordMetadata>> {
    let mut records = Vec::new();
    for path in files {
        if is_parquet_file(path) {
            records.extend(
                parquet_raw_lines(path)?
                    .into_iter()
                    .map(|raw| {
                        let (id, rows) = parse_parquet_raw_line_metadata(&raw.line)?;
                        Ok(FeatureRecordMetadata {
                            manifest_path: raw.manifest_path,
                            base_dir: raw.base_dir,
                            byte_offset: raw.byte_offset,
                            line_number: raw.line_number,
                            id,
                            rows,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
            continue;
        }
        if is_audio_dataset_file(path) {
            records.push(raw_audio_metadata(
                path,
                path.parent().unwrap_or_else(|| Path::new(".")),
            ));
            continue;
        }
        let mut current = CurrentManifestFile::open(path.clone())?;
        loop {
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
                break;
            }
            current.line_number += 1;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let raw = RawManifestLine {
                manifest_path: current.path.clone(),
                base_dir: current.base_dir.clone(),
                byte_offset,
                line_number: current.line_number,
                line: line.to_string(),
            };
            records.push(raw.parse_metadata()?);
        }
    }
    sort_index_records(&mut records);
    Ok(records)
}

fn write_dataset_index(
    files: &[PathBuf],
    index_path: &Path,
    records: &[FeatureRecordMetadata],
) -> Result<()> {
    if let Some(parent) = index_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create dataset index dir {}", parent.display()))?;
    }
    let files = files
        .iter()
        .map(|path| {
            let (len, modified_ms) = file_signature(path)?;
            Ok(json!({
                "path": path,
                "len": len,
                "modified_ms": modified_ms,
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let records = records
        .iter()
        .map(|record| {
            json!({
                "manifest_path": record.manifest_path,
                "base_dir": record.base_dir,
                "byte_offset": record.byte_offset,
                "line_number": record.line_number,
                "id": record.id,
                "rows": record.rows,
            })
        })
        .collect::<Vec<_>>();
    fs::write(
        index_path,
        serde_json::to_string_pretty(&json!({
            "version": 1,
            "files": files,
            "records": records,
        }))?,
    )
    .with_context(|| format!("failed to write dataset index {}", index_path.display()))
}

fn dataset_index_record_from_json(value: &Value) -> Result<FeatureRecordMetadata> {
    Ok(FeatureRecordMetadata {
        manifest_path: json_path(value, "manifest_path")
            .context("dataset index record missing manifest_path")?,
        base_dir: json_path(value, "base_dir").context("dataset index record missing base_dir")?,
        byte_offset: value
            .get("byte_offset")
            .and_then(Value::as_u64)
            .context("dataset index record missing byte_offset")?,
        line_number: json_usize(value, "line_number")
            .context("dataset index record missing line_number")?,
        id: value
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        rows: json_usize(value, "rows").context("dataset index record missing rows")?,
    })
}

fn sort_index_records(records: &mut [FeatureRecordMetadata]) {
    records.sort_by(|left, right| {
        right
            .rows
            .cmp(&left.rows)
            .then_with(|| left.id.cmp(&right.id))
    });
}

fn parquet_raw_lines(path: &Path) -> Result<Vec<RawManifestLine>> {
    let df = read_parquet_dataframe(path)?;
    let base_dir = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let mut lines = Vec::with_capacity(df.height());
    for row in 0..df.height() {
        let id = parquet_row_id(&df, row, path, &base_dir)?;
        let rows = parquet_row_count_hint(&df, row).unwrap_or(usize::MAX);
        lines.push(RawManifestLine {
            manifest_path: path.to_path_buf(),
            base_dir: base_dir.clone(),
            byte_offset: row as u64,
            line_number: row + 1,
            line: format!("{id}\t{rows}"),
        });
    }
    Ok(lines)
}

fn parse_parquet_raw_line_metadata(line: &str) -> Result<(String, usize)> {
    let mut parts = line.splitn(2, '\t');
    let id = parts.next().unwrap_or_default().to_string();
    let rows = parts
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    Ok((id, rows))
}

fn parse_parquet_record(
    path: &Path,
    row: usize,
    base_dir: &Path,
    tokenizer: Option<&SentencePieceTokenizer>,
    audio_decode: &AudioDecodeConfig,
    audio_frontend: &FeatureExtractorConfig,
    waveform_augment: WaveformAugmentConfig,
) -> Result<FeatureRecord> {
    let df = read_parquet_dataframe(path)?;
    if row >= df.height() {
        bail!(
            "parquet row {} is out of bounds for {} rows in {}",
            row,
            df.height(),
            path.display()
        );
    }
    let id = parquet_row_id(&df, row, path, base_dir)?;
    let text = parquet_optional_string(
        &df,
        row,
        &[
            "text",
            "transcript",
            "transcription",
            "sentence",
            "normalized_text",
        ],
    )?;
    let tokens = match parquet_optional_tokens(&df, row)? {
        Some(tokens) => tokens,
        None => {
            parse_record_tokens(&json!({}), text.as_deref(), tokenizer, &id).with_context(|| {
                format!("parquet row '{id}' needs tokens or transcript text plus tokenizer_path")
            })?
        }
    };

    if let Some((features, rows, cols)) = parquet_optional_features(&df, row, &id)? {
        let duration_ms = parquet_optional_duration_ms(&df, row)?
            .unwrap_or_else(|| estimated_duration_ms_from_rows(rows, audio_frontend));
        return Ok(FeatureRecord {
            id,
            rows,
            cols,
            features,
            duration_ms,
            tokens,
            text,
        });
    }

    if let Some((bytes, hint)) = parquet_optional_audio_bytes(&df, row)? {
        let audio = if waveform_augment.is_enabled() {
            audio_bytes_to_features_with_augmentation(
                bytes,
                hint.as_deref(),
                audio_decode,
                audio_frontend,
                waveform_augment,
            )?
        } else {
            audio_bytes_to_features_with_config(
                bytes,
                hint.as_deref(),
                audio_decode,
                audio_frontend,
            )?
        };
        return Ok(FeatureRecord {
            id,
            rows: audio.features.rows,
            cols: audio.features.cols,
            features: audio.features.values,
            duration_ms: duration_ms_from_audio(audio.sample_count, audio.sample_rate),
            tokens,
            text,
        });
    }

    if let Some(audio_path) =
        parquet_optional_string(&df, row, &["audio_path", "path", "file", "file_path"])?
    {
        let audio_path = resolve_path(base_dir, &audio_path);
        let audio = if waveform_augment.is_enabled() {
            audio_file_to_features_with_augmentation(
                audio_path,
                audio_decode,
                audio_frontend,
                waveform_augment,
            )?
        } else {
            audio_file_to_features_with_config(audio_path, audio_decode, audio_frontend)?
        };
        return Ok(FeatureRecord {
            id,
            rows: audio.features.rows,
            cols: audio.features.cols,
            features: audio.features.values,
            duration_ms: duration_ms_from_audio(audio.sample_count, audio.sample_rate),
            tokens,
            text,
        });
    }

    bail!("parquet row '{id}' is missing features or audio bytes/path")
}

fn read_parquet_dataframe(path: &Path) -> Result<DataFrame> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open parquet file {}", path.display()))?;
    ParquetReader::new(file)
        .finish()
        .with_context(|| format!("failed to read parquet file {}", path.display()))
}

fn parquet_row_id(df: &DataFrame, row: usize, path: &Path, base_dir: &Path) -> Result<String> {
    if let Some(id) = parquet_optional_string(
        df,
        row,
        &["id", "utt_id", "utterance_id", "key", "sample_id"],
    )? {
        return Ok(id);
    }
    if let Some(id) =
        parquet_optional_string(df, row, &["audio_path", "path", "file", "file_path"])?
    {
        return Ok(Path::new(&id)
            .with_extension("")
            .to_string_lossy()
            .replace('\\', "/"));
    }
    let file_id = path
        .strip_prefix(base_dir)
        .unwrap_or(path)
        .with_extension("")
        .to_string_lossy()
        .replace('\\', "/");
    Ok(format!("{file_id}-{row}"))
}

fn parquet_row_count_hint(df: &DataFrame, row: usize) -> Option<usize> {
    parquet_optional_usize(df, row, &["rows", "num_frames", "frames", "feature_rows"])
        .ok()
        .flatten()
        .or_else(|| {
            parquet_optional_features(df, row, "metadata")
                .ok()
                .flatten()
                .map(|(_, rows, _)| rows)
        })
}

fn parquet_optional_string(df: &DataFrame, row: usize, names: &[&str]) -> Result<Option<String>> {
    for name in names {
        if let Ok(column) = df.column(name) {
            match column.get(row)? {
                AnyValue::String(value) => return Ok(Some(value.to_string())),
                AnyValue::StringOwned(value) => return Ok(Some(value.to_string())),
                AnyValue::Null => {}
                other => return Ok(Some(other.to_string())),
            }
        }
    }
    Ok(None)
}

fn parquet_optional_usize(df: &DataFrame, row: usize, names: &[&str]) -> Result<Option<usize>> {
    for name in names {
        if let Ok(column) = df.column(name) {
            let value = column.get(row)?;
            if matches!(value, AnyValue::Null) {
                continue;
            }
            return anyvalue_to_i64(value)
                .and_then(|value| usize::try_from(value).ok())
                .map(Some)
                .ok_or_else(|| {
                    anyhow!("parquet column '{name}' must contain non-negative integers")
                });
        }
    }
    Ok(None)
}

fn parquet_optional_duration_ms(df: &DataFrame, row: usize) -> Result<Option<usize>> {
    if let Some(value) = parquet_optional_usize(df, row, &["duration_ms", "audio_duration_ms"])? {
        return Ok(Some(value.max(1)));
    }
    for name in [
        "duration",
        "duration_sec",
        "duration_secs",
        "duration_seconds",
        "audio_duration",
        "audio_duration_sec",
        "audio_duration_seconds",
    ] {
        if let Ok(column) = df.column(name) {
            let value = column.get(row)?;
            if matches!(value, AnyValue::Null) {
                continue;
            }
            let seconds = anyvalue_to_f64(value)
                .ok_or_else(|| anyhow!("parquet column '{name}' must contain numeric seconds"))?;
            return Ok(duration_ms_from_seconds(seconds));
        }
    }
    Ok(None)
}

fn parquet_optional_tokens(df: &DataFrame, row: usize) -> Result<Option<Vec<i64>>> {
    for name in ["tokens", "target", "targets", "labels", "label_ids"] {
        if let Ok(column) = df.column(name) {
            let value = column.get(row)?;
            if matches!(value, AnyValue::Null) {
                continue;
            }
            return Ok(Some(anyvalue_to_i64_vec(value).with_context(|| {
                format!("failed to parse parquet token column '{name}'")
            })?));
        }
    }
    Ok(None)
}

fn parquet_optional_features(
    df: &DataFrame,
    row: usize,
    id: &str,
) -> Result<Option<(Vec<f32>, usize, usize)>> {
    for name in [
        "features",
        "input_features",
        "feature",
        "fbank",
        "filterbank",
    ] {
        if let Ok(column) = df.column(name) {
            let value = column.get(row)?;
            if matches!(value, AnyValue::Null) {
                continue;
            }
            let (features, inferred) = anyvalue_to_f32_matrix(value)
                .with_context(|| format!("failed to parse parquet feature column '{name}'"))?;
            let (rows, cols) = if let Some(shape) = inferred {
                shape
            } else {
                let rows = parquet_optional_usize(
                    df,
                    row,
                    &["rows", "num_frames", "frames", "feature_rows"],
                )?
                .ok_or_else(|| anyhow!("parquet row '{id}' flat features require rows"))?;
                let cols =
                    parquet_optional_usize(df, row, &["cols", "feature_dim", "num_features"])?
                        .ok_or_else(|| anyhow!("parquet row '{id}' flat features require cols"))?;
                (rows, cols)
            };
            if features.len() != rows * cols {
                bail!(
                    "parquet row '{id}' feature shape {rows}x{cols} implies {} values, got {}",
                    rows * cols,
                    features.len()
                );
            }
            return Ok(Some((features, rows, cols)));
        }
    }
    Ok(None)
}

fn parquet_optional_audio_bytes(
    df: &DataFrame,
    row: usize,
) -> Result<Option<(Vec<u8>, Option<String>)>> {
    if let Ok(column) = df.column("audio") {
        if let Ok(audio) = column.struct_() {
            let bytes = audio
                .field_by_name("bytes")
                .ok()
                .and_then(|series| anyvalue_to_bytes(series.get(row).ok()?).ok().flatten());
            let hint = audio
                .field_by_name("path")
                .ok()
                .and_then(|series| anyvalue_to_string(series.get(row).ok()?))
                .and_then(|path| audio_format_hint_from_path(&path));
            if let Some(bytes) = bytes {
                let hint = hint.or_else(|| audio_format_hint_from_bytes(&bytes));
                return Ok(Some((bytes, hint)));
            }
        }
        if let Some(bytes) = anyvalue_to_bytes(column.get(row)?)? {
            let hint = audio_format_hint_from_bytes(&bytes);
            return Ok(Some((bytes, hint)));
        }
    }
    for name in ["audio_bytes", "bytes", "wav", "audio_data"] {
        if let Ok(column) = df.column(name) {
            if let Some(bytes) = anyvalue_to_bytes(column.get(row)?)? {
                let hint = if name == "wav" {
                    Some("audio.wav".to_string())
                } else {
                    audio_format_hint_from_bytes(&bytes)
                };
                return Ok(Some((bytes, hint)));
            }
        }
    }
    Ok(None)
}

fn anyvalue_to_string(value: AnyValue<'_>) -> Option<String> {
    match value {
        AnyValue::String(value) => Some(value.to_string()),
        AnyValue::StringOwned(value) => Some(value.to_string()),
        AnyValue::Null => None,
        other => Some(other.to_string()),
    }
}

fn anyvalue_to_bytes(value: AnyValue<'_>) -> Result<Option<Vec<u8>>> {
    match value {
        AnyValue::Binary(bytes) => Ok(Some(bytes.to_vec())),
        AnyValue::BinaryOwned(bytes) => Ok(Some(bytes)),
        AnyValue::Null => Ok(None),
        other => bail!("expected binary parquet audio bytes, got {other:?}"),
    }
}

fn audio_format_hint_from_path(path: &str) -> Option<String> {
    Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| format!("audio.{extension}"))
}

fn audio_format_hint_from_bytes(bytes: &[u8]) -> Option<String> {
    if bytes.starts_with(b"OggS") && bytes.windows(b"OpusHead".len()).any(|w| w == b"OpusHead") {
        return Some("audio.opus".to_string());
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        return Some("audio.wav".to_string());
    }
    if bytes.starts_with(b"fLaC") {
        return Some("audio.flac".to_string());
    }
    None
}

fn anyvalue_to_i64_vec(value: AnyValue<'_>) -> Result<Vec<i64>> {
    match value {
        AnyValue::List(series) | AnyValue::Array(series, _) => series
            .iter()
            .map(anyvalue_to_i64_result)
            .collect::<Result<Vec<_>>>(),
        AnyValue::String(value) => parse_token_string(value),
        AnyValue::StringOwned(value) => parse_token_string(value.as_str()),
        other => anyvalue_to_i64_result(other).map(|value| vec![value]),
    }
}

fn anyvalue_to_i64_result(value: AnyValue<'_>) -> Result<i64> {
    anyvalue_to_i64(value).ok_or_else(|| anyhow!("expected integer value"))
}

fn anyvalue_to_i64(value: AnyValue<'_>) -> Option<i64> {
    match value {
        AnyValue::Int8(value) => Some(i64::from(value)),
        AnyValue::Int16(value) => Some(i64::from(value)),
        AnyValue::Int32(value) => Some(i64::from(value)),
        AnyValue::Int64(value) => Some(value),
        AnyValue::UInt8(value) => Some(i64::from(value)),
        AnyValue::UInt16(value) => Some(i64::from(value)),
        AnyValue::UInt32(value) => Some(i64::from(value)),
        AnyValue::UInt64(value) => i64::try_from(value).ok(),
        _ => None,
    }
}

fn anyvalue_to_f32_matrix(value: AnyValue<'_>) -> Result<(Vec<f32>, Option<(usize, usize)>)> {
    match value {
        AnyValue::List(series) | AnyValue::Array(series, _) => series_to_f32_matrix(&series),
        other => anyvalue_to_f32(other)
            .map(|value| (vec![value], None))
            .ok_or_else(|| anyhow!("expected numeric feature list")),
    }
}

fn series_to_f32_matrix(series: &Series) -> Result<(Vec<f32>, Option<(usize, usize)>)> {
    if series.is_empty() {
        return Ok((Vec::new(), Some((0, 0))));
    }
    let first = series.get(0)?;
    if matches!(first, AnyValue::List(_) | AnyValue::Array(_, _)) {
        let rows = series.len();
        let mut cols = None;
        let mut values = Vec::new();
        for item in series.iter() {
            let row = anyvalue_to_f32_vec(item)?;
            if let Some(expected) = cols {
                if row.len() != expected {
                    bail!("ragged nested parquet features");
                }
            } else {
                cols = Some(row.len());
            }
            values.extend(row);
        }
        Ok((values, Some((rows, cols.unwrap_or(0)))))
    } else {
        Ok((anyvalue_to_f32_vec(AnyValue::List(series.clone()))?, None))
    }
}

fn anyvalue_to_f32_vec(value: AnyValue<'_>) -> Result<Vec<f32>> {
    match value {
        AnyValue::List(series) | AnyValue::Array(series, _) => series
            .iter()
            .map(anyvalue_to_f32_result)
            .collect::<Result<Vec<_>>>(),
        other => anyvalue_to_f32_result(other).map(|value| vec![value]),
    }
}

fn anyvalue_to_f32_result(value: AnyValue<'_>) -> Result<f32> {
    anyvalue_to_f32(value).ok_or_else(|| anyhow!("expected numeric value"))
}

fn anyvalue_to_f64(value: AnyValue<'_>) -> Option<f64> {
    match value {
        AnyValue::Float32(value) => Some(f64::from(value)),
        AnyValue::Float64(value) => Some(value),
        AnyValue::Int8(value) => Some(f64::from(value)),
        AnyValue::Int16(value) => Some(f64::from(value)),
        AnyValue::Int32(value) => Some(f64::from(value)),
        AnyValue::Int64(value) => Some(value as f64),
        AnyValue::UInt8(value) => Some(f64::from(value)),
        AnyValue::UInt16(value) => Some(f64::from(value)),
        AnyValue::UInt32(value) => Some(f64::from(value)),
        AnyValue::UInt64(value) => Some(value as f64),
        _ => None,
    }
}

fn anyvalue_to_f32(value: AnyValue<'_>) -> Option<f32> {
    match value {
        AnyValue::Float32(value) => Some(value),
        AnyValue::Float64(value) => Some(value as f32),
        AnyValue::Int8(value) => Some(f32::from(value)),
        AnyValue::Int16(value) => Some(f32::from(value)),
        AnyValue::Int32(value) => Some(value as f32),
        AnyValue::Int64(value) => Some(value as f32),
        AnyValue::UInt8(value) => Some(f32::from(value)),
        AnyValue::UInt16(value) => Some(f32::from(value)),
        AnyValue::UInt32(value) => Some(value as f32),
        AnyValue::UInt64(value) => Some(value as f32),
        _ => None,
    }
}

fn raw_audio_metadata(path: &Path, base_dir: &Path) -> FeatureRecordMetadata {
    FeatureRecordMetadata {
        manifest_path: path.to_path_buf(),
        base_dir: base_dir.to_path_buf(),
        byte_offset: 0,
        line_number: 1,
        id: raw_audio_id(path, base_dir),
        rows: read_raw_audio_rows_sidecar(path).unwrap_or(usize::MAX),
    }
}

fn parse_raw_audio_record(
    path: &Path,
    base_dir: &Path,
    tokenizer: Option<&SentencePieceTokenizer>,
    audio_decode: &AudioDecodeConfig,
    audio_frontend: &FeatureExtractorConfig,
    waveform_augment: WaveformAugmentConfig,
) -> Result<FeatureRecord> {
    let id = raw_audio_id(path, base_dir);
    let text = read_raw_audio_text_sidecar(path)?;
    let tokens = match read_raw_audio_tokens_sidecar(path)? {
        Some(tokens) => tokens,
        None => parse_record_tokens(&json!({}), text.as_deref(), tokenizer, &id).with_context(
            || {
                format!(
                    "raw audio file {} requires a .tokens sidecar or transcript sidecar plus tokenizer_path",
                    path.display()
                )
            },
        )?,
    };
    let audio = if waveform_augment.is_enabled() {
        audio_file_to_features_with_augmentation(
            path,
            audio_decode,
            audio_frontend,
            waveform_augment,
        )?
    } else {
        audio_file_to_features_with_config(path, audio_decode, audio_frontend)?
    };
    Ok(FeatureRecord {
        id,
        rows: audio.features.rows,
        cols: audio.features.cols,
        features: audio.features.values,
        duration_ms: duration_ms_from_audio(audio.sample_count, audio.sample_rate),
        tokens,
        text,
    })
}

fn raw_audio_id(path: &Path, base_dir: &Path) -> String {
    path.strip_prefix(base_dir)
        .unwrap_or(path)
        .with_extension("")
        .to_string_lossy()
        .replace('\\', "/")
}

fn read_raw_audio_text_sidecar(path: &Path) -> Result<Option<String>> {
    for extension in ["txt", "lab", "transcript"] {
        let sidecar = path.with_extension(extension);
        if sidecar.exists() {
            let text = fs::read_to_string(&sidecar)
                .with_context(|| {
                    format!("failed to read transcript sidecar {}", sidecar.display())
                })?
                .trim()
                .to_string();
            if text.is_empty() {
                bail!("transcript sidecar {} is empty", sidecar.display());
            }
            return Ok(Some(text));
        }
    }
    Ok(None)
}

fn read_raw_audio_tokens_sidecar(path: &Path) -> Result<Option<Vec<i64>>> {
    for extension in ["tokens", "tok"] {
        let sidecar = path.with_extension(extension);
        if sidecar.exists() {
            return parse_token_string(
                &fs::read_to_string(&sidecar).with_context(|| {
                    format!("failed to read token sidecar {}", sidecar.display())
                })?,
            )
            .map(Some);
        }
    }
    Ok(None)
}

fn read_raw_audio_rows_sidecar(path: &Path) -> Option<usize> {
    ["rows", "frames"].into_iter().find_map(|extension| {
        let sidecar = path.with_extension(extension);
        fs::read_to_string(sidecar)
            .ok()
            .and_then(|text| text.trim().parse::<usize>().ok())
    })
}

fn file_signature(path: &Path) -> Result<(u64, u64)> {
    let metadata =
        fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    let modified_ms = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0);
    Ok((metadata.len(), modified_ms))
}

fn adaptive_cost<'a>(
    records: impl Iterator<Item = &'a FeatureRecord>,
    unit: AdaptiveBatchUnit,
    expected_dim: usize,
) -> usize {
    let mut samples = 0usize;
    let mut frames = 0usize;
    let mut max_frames = 0usize;
    let mut duration_ms = 0usize;
    let mut max_duration_ms = 0usize;
    for record in records {
        samples += 1;
        frames += record.rows;
        max_frames = max_frames.max(record.rows);
        duration_ms += record.duration_ms.max(1);
        max_duration_ms = max_duration_ms.max(record.duration_ms.max(1));
    }
    match unit {
        AdaptiveBatchUnit::Samples => samples,
        AdaptiveBatchUnit::Frames => frames,
        AdaptiveBatchUnit::PaddedFrames => samples * max_frames,
        AdaptiveBatchUnit::FeatureValues => samples * max_frames * expected_dim,
        AdaptiveBatchUnit::DurationMs => duration_ms,
        AdaptiveBatchUnit::PaddedDurationMs => samples * max_duration_ms,
    }
}

fn load_manifest_file(path: &Path, limit: Option<usize>) -> Result<Vec<FeatureRecord>> {
    if is_parquet_file(path) {
        let raws = parquet_raw_lines(path)?;
        let mut records = Vec::new();
        for raw in raws.into_iter().take(limit.unwrap_or(usize::MAX)) {
            records.push(raw.parse_record(
                None,
                &AudioDecodeConfig::default(),
                &default_training_feature_extractor(),
                WaveformAugmentConfig::default(),
            )?);
        }
        return Ok(records);
    }

    if is_audio_dataset_file(path) {
        return Ok(vec![parse_raw_audio_record(
            path,
            path.parent().unwrap_or_else(|| Path::new(".")),
            None,
            &AudioDecodeConfig::default(),
            &default_training_feature_extractor(),
            WaveformAugmentConfig::default(),
        )?]);
    }

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
            parse_json_record(
                line,
                base_dir,
                line_index + 1,
                None,
                &AudioDecodeConfig::default(),
                &default_training_feature_extractor(),
                WaveformAugmentConfig::default(),
            )?
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

fn parse_json_record(
    line: &str,
    base_dir: &Path,
    line_number: usize,
    tokenizer: Option<&SentencePieceTokenizer>,
    audio_decode: &AudioDecodeConfig,
    audio_frontend: &FeatureExtractorConfig,
    waveform_augment: WaveformAugmentConfig,
) -> Result<FeatureRecord> {
    let value: Value = serde_json::from_str(line)
        .with_context(|| format!("invalid JSON manifest line {line_number}"))?;
    if let Some(parquet_path) = value.get("parquet_path").and_then(Value::as_str) {
        let row = value
            .get("parquet_row")
            .or_else(|| value.get("row"))
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                anyhow!(
                    "JSON manifest line {line_number} parquet reference must include parquet_row"
                )
            })? as usize;
        let parquet_path = resolve_path(base_dir, parquet_path);
        let parquet_base_dir = parquet_path.parent().unwrap_or_else(|| Path::new("."));
        return parse_parquet_record(
            &parquet_path,
            row,
            parquet_base_dir,
            tokenizer,
            audio_decode,
            audio_frontend,
            waveform_augment,
        );
    }
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("line-{line_number}"));
    let text = value
        .get("text")
        .or_else(|| value.get("transcript"))
        .or_else(|| value.get("transcription"))
        .or_else(|| value.get("sentence"))
        .or_else(|| value.get("normalized_text"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let tokens = parse_record_tokens(&value, text.as_deref(), tokenizer, &id)?;
    let manifest_duration_ms = json_duration_ms(&value);

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
            duration_ms: manifest_duration_ms
                .unwrap_or_else(|| estimated_duration_ms_from_rows(rows, audio_frontend)),
            tokens,
            text,
        });
    }

    let feature_path = value
        .get("features_path")
        .or_else(|| value.get("feature_path"))
        .or_else(|| value.get("path"))
        .and_then(Value::as_str);
    if let Some(feature_path) = feature_path {
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
        return Ok(FeatureRecord {
            id,
            rows,
            cols,
            features,
            duration_ms: manifest_duration_ms
                .unwrap_or_else(|| estimated_duration_ms_from_rows(rows, audio_frontend)),
            tokens,
            text,
        });
    }

    let audio_path = value
        .get("audio_path")
        .or_else(|| value.get("audio"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            anyhow!("record '{id}' is missing features, features_path, or audio_path")
        })?;
    let audio_path = resolve_path(base_dir, audio_path);
    let audio = if waveform_augment.is_enabled() {
        audio_file_to_features_with_augmentation(
            audio_path,
            audio_decode,
            audio_frontend,
            waveform_augment,
        )?
    } else {
        audio_file_to_features_with_config(audio_path, audio_decode, audio_frontend)?
    };
    Ok(FeatureRecord {
        id,
        rows: audio.features.rows,
        cols: audio.features.cols,
        features: audio.features.values,
        duration_ms: duration_ms_from_audio(audio.sample_count, audio.sample_rate),
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
    let duration_ms = parts
        .get(5)
        .and_then(|value| value.parse::<f64>().ok())
        .and_then(duration_ms_from_seconds)
        .unwrap_or_else(|| rows.saturating_mul(10).max(1));
    Ok(FeatureRecord {
        id: format!("line-{line_number}"),
        rows,
        cols,
        features,
        duration_ms,
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

    if value.get("parquet_path").and_then(Value::as_str).is_some() {
        let rows = value
            .get("rows")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(usize::MAX);
        return Ok((id, rows));
    }

    if value
        .get("audio_path")
        .or_else(|| value.get("audio"))
        .and_then(Value::as_str)
        .is_some()
    {
        let rows = value
            .get("rows")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(usize::MAX);
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

fn parse_record_tokens(
    value: &Value,
    text: Option<&str>,
    tokenizer: Option<&SentencePieceTokenizer>,
    id: &str,
) -> Result<Vec<i64>> {
    if let Some(tokens) = value
        .get("tokens")
        .or_else(|| value.get("target"))
        .or_else(|| value.get("targets"))
    {
        return parse_tokens_value(tokens);
    }
    let text = text.ok_or_else(|| anyhow!("record '{id}' is missing tokens or transcript text"))?;
    let tokenizer = tokenizer.ok_or_else(|| {
        anyhow!("record '{id}' needs tokenizer_path to derive tokens from transcript text")
    })?;
    let tokens = tokenizer
        .encode(text)
        .into_iter()
        .map(i64::from)
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        bail!("record '{id}' transcript encoded to an empty token sequence");
    }
    Ok(tokens)
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
        .cast(FloatDType::F32)
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
    if config.dataset_index_dir.is_some() && !config.sort_by_length_desc {
        bail!("dataset_index_dir requires sort_by_length_desc so cached row metadata is used");
    }
    if config.spec_augment.time_masks > 0 && config.spec_augment.time_mask_max_frames == 0 {
        bail!("spec_augment.time_mask_max_frames must be > 0 when time masks are enabled");
    }
    if config.spec_augment.frequency_masks > 0 && config.spec_augment.frequency_mask_max_bins == 0 {
        bail!("spec_augment.frequency_mask_max_bins must be > 0 when frequency masks are enabled");
    }
    if config.waveform_augment.noise_std < 0.0 {
        bail!("waveform_augment.noise_std must be >= 0");
    }
    if let (Some(min_gain), Some(max_gain)) = (
        config.waveform_augment.gain_min_db,
        config.waveform_augment.gain_max_db,
    ) {
        if min_gain > max_gain {
            bail!("waveform_augment.gain_min_db must be <= gain_max_db");
        }
    }
    if config.epochs == 0 {
        bail!("epochs must be > 0");
    }
    if config.learning_rate < 0.0 {
        bail!("learning_rate must be >= 0");
    }
    if config.lr_min < 0.0 {
        bail!("lr_min must be >= 0");
    }
    if config.lr_decay_exponent < 0.0 {
        bail!("lr_decay_exponent must be >= 0");
    }
    if config.gradient_accumulation_steps == 0 {
        bail!("gradient_accumulation_steps must be > 0");
    }
    if config.log_every == 0 {
        bail!("log_every must be > 0");
    }
    if config.max_audio_duration_ms == Some(0) {
        bail!("max_audio_duration_ms must be > 0 when set");
    }
    if config.gradient_clip_norm.is_some() && config.gradient_clip_value.is_some() {
        bail!("set at most one of gradient_clip_norm or gradient_clip_value");
    }
    if config.device_indices.is_empty() {
        bail!("device_indices must contain at least one device");
    }
    if config.backend == TrainBackendKind::Cpu && config.device_indices != [0] {
        bail!("cpu backend only supports device index 0");
    }
    if matches!(config.gradient_clip_norm, Some(value) if value <= 0.0) {
        bail!("gradient_clip_norm must be > 0 when set");
    }
    if matches!(config.gradient_clip_value, Some(value) if value <= 0.0) {
        bail!("gradient_clip_value must be > 0 when set");
    }
    if matches!(config.ema_decay, Some(value) if value <= 0.0 || value >= 1.0) {
        bail!("ema_decay must be > 0 and < 1 when set");
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
    if config.hf_upload_checkpoints && config.hf_upload_repo_id.is_none() {
        bail!("hf_upload_checkpoints requires hf_upload_repo_id");
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

fn structured_log_path(config: &BurnTrainConfig) -> PathBuf {
    config.output_dir.join("events.jsonl")
}

fn append_structured_event(config: &BurnTrainConfig, event: &str, payload: Value) -> Result<()> {
    fs::create_dir_all(&config.output_dir).with_context(|| {
        format!(
            "failed to create output directory {}",
            config.output_dir.display()
        )
    })?;
    let path = structured_log_path(config);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open structured log {}", path.display()))?;
    writeln!(
        file,
        "{}",
        serde_json::to_string(&structured_event(event, payload))?
    )
    .with_context(|| format!("failed to write structured log {}", path.display()))
}

fn structured_event(event: &str, payload: Value) -> Value {
    json!({
        "event": event,
        "timestamp_ms": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or(0),
        "data": payload,
    })
}

struct RunLogger<'a> {
    config: &'a BurnTrainConfig,
    file: fs::File,
    started: Instant,
    tui: Option<TrainingTui>,
}

impl<'a> RunLogger<'a> {
    fn new(config: &'a BurnTrainConfig, started: Instant) -> Result<Self> {
        fs::create_dir_all(&config.output_dir).with_context(|| {
            format!(
                "failed to create output directory {}",
                config.output_dir.display()
            )
        })?;
        let path = structured_log_path(config);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("failed to open structured log {}", path.display()))?;
        let tui = if config.tui {
            Some(TrainingTui::new(config, started)?)
        } else {
            None
        };
        Ok(Self {
            config,
            file,
            started,
            tui,
        })
    }

    fn log(&mut self, event: &str, payload: Value) -> Result<()> {
        let mut payload = payload;
        if let Some(object) = payload.as_object_mut() {
            object.insert(
                "elapsed_sec".to_string(),
                json!(self.started.elapsed().as_secs_f64()),
            );
        }
        writeln!(
            self.file,
            "{}",
            serde_json::to_string(&structured_event(event, payload.clone()))?
        )
        .with_context(|| {
            format!(
                "failed to write structured log {}",
                structured_log_path(self.config).display()
            )
        })?;
        self.file.flush().with_context(|| {
            format!(
                "failed to flush structured log {}",
                structured_log_path(self.config).display()
            )
        })?;
        if let Some(tui) = self.tui.as_mut() {
            tui.update(event, &payload)?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct TuiMetrics {
    epoch: Option<usize>,
    step: Option<usize>,
    phase: String,
    train_loss: Option<f64>,
    previous_train_loss: Option<f64>,
    train_ctc_loss: Option<f64>,
    train_ce_loss: Option<f64>,
    val_loss: Option<f64>,
    previous_val_loss: Option<f64>,
    val_cer: Option<f64>,
    val_wer: Option<f64>,
    learning_rate: Option<f64>,
    samples: Option<usize>,
    duration_min: Option<f64>,
    padded_duration_min: Option<f64>,
    max_frames: Option<usize>,
    padded_frames: Option<usize>,
    padding_ratio: Option<f64>,
    samples_per_sec: Option<f64>,
    frames_per_sec: Option<f64>,
    last_event: String,
}

struct TrainingTui {
    started: Instant,
    stdout: std::io::Stdout,
    architecture: String,
    cubecl_kernel_status: String,
    output_dir: PathBuf,
    metrics: TuiMetrics,
}

impl TrainingTui {
    fn new(config: &BurnTrainConfig, started: Instant) -> Result<Self> {
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, Hide, Clear(ClearType::All))
            .context("failed to initialize training TUI")?;
        let mut tui = Self {
            started,
            stdout,
            architecture: architecture_name(&config.architecture).to_string(),
            cubecl_kernel_status: cubecl_kernel_status(config).to_string(),
            output_dir: config.output_dir.clone(),
            metrics: TuiMetrics {
                phase: "starting".to_string(),
                ..TuiMetrics::default()
            },
        };
        tui.draw()?;
        Ok(tui)
    }

    fn update(&mut self, event: &str, payload: &Value) -> Result<()> {
        self.metrics.last_event = event.to_string();
        match event {
            "run_start" => self.metrics.phase = "training".to_string(),
            "batch_start" => {
                self.metrics.phase = "loading batch / extracting audio".to_string();
                self.metrics.epoch = json_usize(payload, "epoch");
                self.metrics.step = json_usize(payload, "next_step");
                if let Some(batch) = payload.get("batch") {
                    self.metrics.samples = json_usize(batch, "samples");
                    self.metrics.duration_min = json_f64(batch, "duration_min");
                    self.metrics.padded_duration_min = json_f64(batch, "padded_duration_min");
                    self.metrics.max_frames = json_usize(batch, "max_frames");
                    self.metrics.padded_frames = json_usize(batch, "padded_frames");
                    self.metrics.padding_ratio = json_f64(batch, "padding_ratio");
                }
            }
            "train_step" => {
                self.metrics.phase = "training".to_string();
                self.metrics.epoch = json_usize(payload, "epoch");
                self.metrics.step = json_usize(payload, "global_step");
                self.metrics.learning_rate = json_f64(payload, "learning_rate");
                if let Some(losses) = payload.get("losses") {
                    if let Some(train_loss) =
                        json_f64(losses, "total").or_else(|| json_f64(losses, "ctc"))
                    {
                        self.metrics.previous_train_loss = self.metrics.train_loss;
                        self.metrics.train_loss = Some(train_loss);
                    }
                    self.metrics.train_ctc_loss =
                        json_f64(losses, "ctc").or(self.metrics.train_ctc_loss);
                    self.metrics.train_ce_loss =
                        json_f64(losses, "ce").or(self.metrics.train_ce_loss);
                }
                if let Some(throughput) = payload.get("throughput") {
                    self.metrics.samples_per_sec = json_f64(throughput, "samples_per_sec");
                    self.metrics.frames_per_sec = json_f64(throughput, "frames_per_sec");
                }
            }
            "validation_start" => {
                self.metrics.phase = "validation".to_string();
                self.metrics.epoch = json_usize(payload, "epoch").or(self.metrics.epoch);
                self.metrics.step = json_usize(payload, "global_step").or(self.metrics.step);
            }
            "validation" => {
                self.metrics.phase = "validation".to_string();
                self.metrics.epoch = json_usize(payload, "epoch").or(self.metrics.epoch);
                self.metrics.step = json_usize(payload, "global_step").or(self.metrics.step);
                if let Some(val_loss) = json_f64(payload, "loss") {
                    self.metrics.previous_val_loss = self.metrics.val_loss;
                    self.metrics.val_loss = Some(val_loss);
                }
                self.metrics.val_cer = json_f64(payload, "cer");
                self.metrics.val_wer = json_f64(payload, "wer");
            }
            "checkpoint_saved" => self.metrics.phase = "checkpoint saved".to_string(),
            "run_complete" => self.metrics.phase = "complete".to_string(),
            _ => {}
        }
        self.draw()
    }

    fn draw(&mut self) -> Result<()> {
        queue!(
            self.stdout,
            MoveTo(0, 0),
            Clear(ClearType::All),
            SetForegroundColor(Color::Cyan),
            SetAttribute(Attribute::Bold),
            Print("w2v-bert-uk training monitor\n"),
            ResetColor,
            SetAttribute(Attribute::Reset),
            Print(format!("architecture: {}\n", self.architecture)),
            Print(format!("cubecl kernels: {}\n", self.cubecl_kernel_status)),
            Print(format!("output: {}\n", self.output_dir.display())),
            Print(format!(
                "elapsed: {:.1}s   phase: {}   event: {}\n\n",
                self.started.elapsed().as_secs_f64(),
                self.metrics.phase,
                self.metrics.last_event
            )),
            SetForegroundColor(Color::Yellow),
            Print("Progress\n"),
            ResetColor,
            Print(format!(
                "  epoch: {}   step: {}   lr: {}\n",
                fmt_opt_usize(self.metrics.epoch),
                fmt_opt_usize(self.metrics.step),
                fmt_opt_f64(self.metrics.learning_rate, 8)
            )),
            Print(format!(
                "  duration_min: {}   padded_duration_min: {}\n",
                fmt_opt_f64(self.metrics.duration_min, 2),
                fmt_opt_f64(self.metrics.padded_duration_min, 2)
            )),
            Print(format!(
                "  samples: {}   max_frames: {}   padded_frames: {}   padding_ratio: {}\n",
                fmt_opt_usize(self.metrics.samples),
                fmt_opt_usize(self.metrics.max_frames),
                fmt_opt_usize(self.metrics.padded_frames),
                fmt_opt_f64(self.metrics.padding_ratio, 3)
            )),
            Print(format!(
                "  samples/sec: {}   frames/sec: {}\n\n",
                fmt_opt_f64(self.metrics.samples_per_sec, 2),
                fmt_opt_f64(self.metrics.frames_per_sec, 0)
            )),
            SetForegroundColor(Color::Green),
            Print("Loss\n"),
            ResetColor,
            Print(format!(
                "  train: {}   prev: {}   ctc: {}   ce: {}\n",
                fmt_opt_f64(self.metrics.train_loss, 6),
                fmt_opt_f64(self.metrics.previous_train_loss, 6),
                fmt_opt_f64(self.metrics.train_ctc_loss, 6),
                fmt_opt_f64(self.metrics.train_ce_loss, 6)
            )),
            Print(format!(
                "  val: {}   prev: {}   cer: {}   wer: {}\n\n",
                fmt_opt_f64(self.metrics.val_loss, 6),
                fmt_opt_f64(self.metrics.previous_val_loss, 6),
                fmt_opt_f64(self.metrics.val_cer, 6),
                fmt_opt_f64(self.metrics.val_wer, 6)
            )),
            SetForegroundColor(Color::DarkGrey),
            Print("Press Ctrl-C to stop. Structured events are still written to events.jsonl.\n"),
            ResetColor
        )
        .context("failed to draw training TUI")?;
        self.stdout.flush().context("failed to flush training TUI")
    }
}

fn cubecl_kernel_status(config: &BurnTrainConfig) -> &'static str {
    if !cfg!(feature = "asr-cubecl-kernels") {
        return "inactive (feature off)";
    }

    match config.backend {
        TrainBackendKind::Cpu => "available, not routed (cpu backend)",
        TrainBackendKind::Cuda => "available, explicit kernels only (cuda)",
        TrainBackendKind::Wgpu => "available, explicit kernels only (wgpu)",
    }
}

impl Drop for TrainingTui {
    fn drop(&mut self) {
        let _ = execute!(self.stdout, Show, LeaveAlternateScreen);
    }
}

fn fmt_opt_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn fmt_opt_f64(value: Option<f64>, decimals: usize) -> String {
    value
        .map(|value| format!("{value:.decimals$}"))
        .unwrap_or_else(|| "-".to_string())
}

fn architecture_name(architecture: &TrainArchitecture) -> &'static str {
    match architecture {
        TrainArchitecture::Squeezeformer => "squeezeformer",
        TrainArchitecture::Zipformer => "zipformer",
        TrainArchitecture::Paraformer => "paraformer",
        TrainArchitecture::Wav2VecBert => "w2v_bert",
    }
}

pub fn feature_extractor_for_train_architecture(
    architecture: TrainArchitecture,
) -> FeatureExtractorConfig {
    match architecture {
        TrainArchitecture::Zipformer => {
            FeatureExtractorConfig::Audio(asr_features::zipformer_frontend_config())
        }
        TrainArchitecture::Squeezeformer => {
            FeatureExtractorConfig::Audio(asr_features::squeezeformer_frontend_config())
        }
        TrainArchitecture::Paraformer => {
            FeatureExtractorConfig::Audio(asr_features::paraformer_frontend_config())
        }
        TrainArchitecture::Wav2VecBert => {
            FeatureExtractorConfig::W2vBert(W2vBertEncoderConfig::default().to_frontend_config())
        }
    }
}

fn feature_extractor_for_architecture(config: &BurnTrainConfig) -> FeatureExtractorConfig {
    feature_extractor_for_train_architecture(config.architecture)
}

fn default_training_feature_extractor() -> FeatureExtractorConfig {
    FeatureExtractorConfig::W2vBert(W2vBertEncoderConfig::default().to_frontend_config())
}

fn adaptive_batch_json(config: &BurnTrainConfig) -> Option<Value> {
    config.adaptive_batch.map(|adaptive| {
        json!({
            "unit": match adaptive.unit {
                AdaptiveBatchUnit::Samples => "samples",
                AdaptiveBatchUnit::Frames => "frames",
                AdaptiveBatchUnit::PaddedFrames => "padded_frames",
                AdaptiveBatchUnit::FeatureValues => "feature_values",
                AdaptiveBatchUnit::DurationMs => "duration_ms",
                AdaptiveBatchUnit::PaddedDurationMs => "padded_duration_ms",
            },
            "budget": adaptive.budget,
            "max_samples": adaptive.max_samples,
        })
    })
}

fn spec_augment_json(config: SpecAugmentConfig) -> Value {
    json!({
        "time_masks": config.time_masks,
        "time_mask_max_frames": config.time_mask_max_frames,
        "frequency_masks": config.frequency_masks,
        "frequency_mask_max_bins": config.frequency_mask_max_bins,
    })
}

fn waveform_augment_json(config: WaveformAugmentConfig) -> Value {
    json!({
        "gain_min_db": config.gain_min_db,
        "gain_max_db": config.gain_max_db,
        "noise_std": config.noise_std,
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
        "dataset_index_dir": config.dataset_index_dir,
        "spec_augment": spec_augment_json(config.spec_augment),
        "waveform_augment": waveform_augment_json(config.waveform_augment),
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "lr_warmup_steps": config.lr_warmup_steps,
        "lr_hold_steps": config.lr_hold_steps,
        "lr_decay_steps": config.lr_decay_steps,
        "lr_warmup_epochs": config.lr_warmup_epochs,
        "lr_hold_epochs": config.lr_hold_epochs,
        "lr_decay_exponent": config.lr_decay_exponent,
        "lr_min": config.lr_min,
        "weight_decay": config.weight_decay,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clip_norm": config.gradient_clip_norm,
        "gradient_clip_value": config.gradient_clip_value,
        "ema_decay": config.ema_decay,
        "ema_start_step": config.ema_start_step,
        "max_train_samples": config.max_train_samples,
        "max_val_samples": config.max_val_samples,
        "max_audio_duration_ms": config.max_audio_duration_ms,
        "tokenizer_path": config.tokenizer_path,
        "val_beam_width": config.val_beam_width,
        "val_n_best": config.val_n_best,
        "val_lm_path": config.val_lm_path,
        "val_lm_weight": config.val_lm_weight,
        "val_lm_word_bonus": config.val_lm_word_bonus,
        "val_lm_bos": config.val_lm_bos,
        "val_lm_eos": config.val_lm_eos,
        "val_log_samples": config.val_log_samples,
        "paraformer_alignment_mode": match config.paraformer_alignment_mode {
            ParaformerAlignmentMode::Viterbi => "viterbi",
            ParaformerAlignmentMode::Uniform => "uniform",
            ParaformerAlignmentMode::Greedy => "greedy",
        },
        "paraformer_enhanced": config.paraformer_enhanced,
        "w2v_hf_model_dir": config.w2v_hf_model_dir,
        "w2v_hf_load_weights": config.w2v_hf_load_weights,
        "w2v_activation_checkpointing": config.w2v_activation_checkpointing,
        "w2v_num_adapter_layers": config.w2v_num_adapter_layers,
        "w2v_adapter_stride": config.w2v_adapter_stride,
        "w2v_adapter_kernel_size": config.w2v_adapter_kernel_size,
        "init_from": config.init_from,
        "backend": match config.backend {
            TrainBackendKind::Cpu => "cpu",
            TrainBackendKind::Cuda => "cuda",
            TrainBackendKind::Wgpu => "wgpu",
        },
        "cubecl_kernels": {
            "compiled": cfg!(feature = "asr-cubecl-kernels"),
            "active": false,
            "status": cubecl_kernel_status(config),
        },
        "device_index": config.device_index,
        "device_indices": config.device_indices,
        "precision": match config.precision {
            TrainPrecision::F32 => "f32",
            TrainPrecision::F16 => "f16",
            TrainPrecision::Bf16 => "bf16",
        },
        "tui": config.tui,
        "hf_upload_checkpoints": config.hf_upload_checkpoints,
        "hf_upload_repo_id": config.hf_upload_repo_id,
        "hf_upload_revision": config.hf_upload_revision,
        "hf_upload_private": config.hf_upload_private,
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

fn checkpoint_dir_from_path(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        Ok(path.to_path_buf())
    } else {
        checkpoint_metadata_path(path)
            .parent()
            .map(Path::to_path_buf)
            .context("checkpoint metadata path has no parent directory")
    }
}

fn load_checkpoint_metadata(path: &Path) -> Result<Value> {
    let metadata_path = checkpoint_metadata_path(path);
    let metadata_text = fs::read_to_string(&metadata_path).with_context(|| {
        format!(
            "failed to read checkpoint metadata {}",
            metadata_path.display()
        )
    })?;
    serde_json::from_str(&metadata_text).with_context(|| {
        format!(
            "failed to parse checkpoint metadata {}",
            metadata_path.display()
        )
    })
}

fn load_checkpoint_train_config(path: &Path) -> Result<BurnTrainConfig> {
    let metadata = load_checkpoint_metadata(path)?;
    let value = metadata
        .get("training_config")
        .context("checkpoint metadata does not contain training_config")?;
    train_config_from_json(value)
}

fn train_config_from_json(value: &Value) -> Result<BurnTrainConfig> {
    let mut config = BurnTrainConfig::default();
    if let Some(architecture) = value.get("architecture").and_then(Value::as_str) {
        config.architecture = architecture.parse()?;
    }
    config.train_manifest = json_path(value, "train_manifest").unwrap_or(config.train_manifest);
    config.val_manifest = json_path(value, "val_manifest");
    config.variant = value
        .get("variant")
        .and_then(Value::as_str)
        .map(str::to_string);
    config.input_dim = json_usize(value, "input_dim").unwrap_or(config.input_dim);
    config.vocab_size = json_usize(value, "vocab_size").unwrap_or(config.vocab_size);
    config.blank_id = json_usize(value, "blank_id").unwrap_or(config.blank_id);
    config.d_model = json_usize(value, "d_model").unwrap_or(config.d_model);
    config.num_layers = json_usize(value, "num_layers").unwrap_or(config.num_layers);
    config.num_heads = json_usize(value, "num_heads").unwrap_or(config.num_heads);
    config.batch_size = json_usize(value, "batch_size").unwrap_or(config.batch_size);
    config.adaptive_batch = json_adaptive_batch(value)?;
    config.sort_by_length_desc =
        json_bool(value, "sort_by_length_desc").unwrap_or(config.sort_by_length_desc);
    config.sort_buffer_size =
        json_usize(value, "sort_buffer_size").unwrap_or(config.sort_buffer_size);
    config.dataset_index_dir = json_path(value, "dataset_index_dir");
    config.spec_augment = json_spec_augment(value);
    config.waveform_augment = json_waveform_augment(value);
    config.epochs = json_usize(value, "epochs").unwrap_or(config.epochs);
    config.learning_rate = json_f64(value, "learning_rate").unwrap_or(config.learning_rate);
    config.lr_warmup_steps = json_usize(value, "lr_warmup_steps").unwrap_or(config.lr_warmup_steps);
    config.lr_hold_steps = json_usize(value, "lr_hold_steps").unwrap_or(config.lr_hold_steps);
    config.lr_decay_steps = json_usize(value, "lr_decay_steps").unwrap_or(config.lr_decay_steps);
    config.lr_warmup_epochs =
        json_usize(value, "lr_warmup_epochs").unwrap_or(config.lr_warmup_epochs);
    config.lr_hold_epochs = json_usize(value, "lr_hold_epochs").unwrap_or(config.lr_hold_epochs);
    config.lr_decay_exponent =
        json_f64(value, "lr_decay_exponent").unwrap_or(config.lr_decay_exponent);
    config.lr_min = json_f64(value, "lr_min").unwrap_or(config.lr_min);
    config.weight_decay = json_f64(value, "weight_decay").unwrap_or(config.weight_decay);
    config.gradient_accumulation_steps = json_usize(value, "gradient_accumulation_steps")
        .unwrap_or(config.gradient_accumulation_steps);
    config.gradient_clip_norm = json_f32(value, "gradient_clip_norm");
    config.gradient_clip_value = json_f32(value, "gradient_clip_value");
    config.ema_decay = json_f64(value, "ema_decay");
    config.ema_start_step = json_usize(value, "ema_start_step").unwrap_or(config.ema_start_step);
    config.max_train_samples = json_usize(value, "max_train_samples");
    config.max_val_samples = json_usize(value, "max_val_samples");
    config.max_audio_duration_ms = json_usize(value, "max_audio_duration_ms");
    config.tokenizer_path = json_path(value, "tokenizer_path");
    config.val_beam_width = json_usize(value, "val_beam_width").unwrap_or(config.val_beam_width);
    config.val_n_best = json_usize(value, "val_n_best").unwrap_or(config.val_n_best);
    config.val_lm_path = json_path(value, "val_lm_path");
    config.val_lm_weight = json_f32(value, "val_lm_weight").unwrap_or(config.val_lm_weight);
    config.val_lm_word_bonus =
        json_f32(value, "val_lm_word_bonus").unwrap_or(config.val_lm_word_bonus);
    config.val_lm_bos = json_bool(value, "val_lm_bos").unwrap_or(config.val_lm_bos);
    config.val_lm_eos = json_bool(value, "val_lm_eos").unwrap_or(config.val_lm_eos);
    config.val_log_samples = json_usize(value, "val_log_samples").unwrap_or(config.val_log_samples);
    config.paraformer_alignment_mode = match value
        .get("paraformer_alignment_mode")
        .and_then(Value::as_str)
        .unwrap_or("viterbi")
    {
        "uniform" => ParaformerAlignmentMode::Uniform,
        "greedy" => ParaformerAlignmentMode::Greedy,
        _ => ParaformerAlignmentMode::Viterbi,
    };
    config.paraformer_enhanced =
        json_bool(value, "paraformer_enhanced").unwrap_or(config.paraformer_enhanced);
    config.w2v_hf_model_dir = json_path(value, "w2v_hf_model_dir");
    config.w2v_hf_load_weights =
        json_bool(value, "w2v_hf_load_weights").unwrap_or(config.w2v_hf_load_weights);
    config.w2v_activation_checkpointing = json_bool(value, "w2v_activation_checkpointing")
        .unwrap_or(config.w2v_activation_checkpointing);
    config.w2v_num_adapter_layers =
        json_usize(value, "w2v_num_adapter_layers").unwrap_or(config.w2v_num_adapter_layers);
    config.w2v_adapter_stride =
        json_usize(value, "w2v_adapter_stride").unwrap_or(config.w2v_adapter_stride);
    config.w2v_adapter_kernel_size =
        json_usize(value, "w2v_adapter_kernel_size").unwrap_or(config.w2v_adapter_kernel_size);
    config.init_from = json_path(value, "init_from");
    config.backend = value
        .get("backend")
        .and_then(Value::as_str)
        .unwrap_or("cpu")
        .parse()
        .unwrap_or(TrainBackendKind::Cpu);
    config.device_index = json_usize(value, "device_index").unwrap_or(config.device_index);
    config.device_indices = value
        .get("device_indices")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_u64().map(|value| value as usize))
                .collect::<Vec<_>>()
        })
        .filter(|items| !items.is_empty())
        .unwrap_or_else(|| vec![config.device_index]);
    config.precision = value
        .get("precision")
        .and_then(Value::as_str)
        .unwrap_or("f32")
        .parse()
        .unwrap_or(TrainPrecision::F32);
    config.tui = json_bool(value, "tui").unwrap_or(config.tui);
    config.hf_upload_checkpoints =
        json_bool(value, "hf_upload_checkpoints").unwrap_or(config.hf_upload_checkpoints);
    config.hf_upload_repo_id = value
        .get("hf_upload_repo_id")
        .and_then(Value::as_str)
        .map(str::to_string);
    config.hf_upload_revision = value
        .get("hf_upload_revision")
        .and_then(Value::as_str)
        .map(str::to_string);
    config.hf_upload_private =
        json_bool(value, "hf_upload_private").unwrap_or(config.hf_upload_private);
    Ok(config)
}

fn json_path(value: &Value, key: &str) -> Option<PathBuf> {
    value.get(key).and_then(Value::as_str).map(PathBuf::from)
}

fn json_usize(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
}

fn json_f64(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

fn json_f32(value: &Value, key: &str) -> Option<f32> {
    json_f64(value, key).map(|value| value as f32)
}

fn json_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn json_adaptive_batch(value: &Value) -> Result<Option<AdaptiveBatchConfig>> {
    let Some(adaptive) = value.get("adaptive_batch") else {
        return Ok(None);
    };
    if adaptive.is_null() {
        return Ok(None);
    }
    let unit = adaptive
        .get("unit")
        .and_then(Value::as_str)
        .unwrap_or("samples")
        .parse()?;
    let budget = adaptive
        .get("budget")
        .and_then(Value::as_u64)
        .context("adaptive_batch.budget must be present in checkpoint config")?
        as usize;
    let max_samples = adaptive
        .get("max_samples")
        .and_then(Value::as_u64)
        .map(|value| value as usize);
    Ok(Some(AdaptiveBatchConfig {
        unit,
        budget,
        max_samples,
    }))
}

fn json_spec_augment(value: &Value) -> SpecAugmentConfig {
    let Some(augment) = value.get("spec_augment") else {
        return SpecAugmentConfig::default();
    };
    SpecAugmentConfig {
        time_masks: json_usize(augment, "time_masks").unwrap_or(0),
        time_mask_max_frames: json_usize(augment, "time_mask_max_frames").unwrap_or(0),
        frequency_masks: json_usize(augment, "frequency_masks").unwrap_or(0),
        frequency_mask_max_bins: json_usize(augment, "frequency_mask_max_bins").unwrap_or(0),
    }
}

fn json_waveform_augment(value: &Value) -> WaveformAugmentConfig {
    let Some(augment) = value.get("waveform_augment") else {
        return WaveformAugmentConfig::default();
    };
    WaveformAugmentConfig {
        gain_min_db: json_f32(augment, "gain_min_db"),
        gain_max_db: json_f32(augment, "gain_max_db"),
        noise_std: json_f32(augment, "noise_std").unwrap_or(0.0),
    }
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
    ema_model: Option<&M>,
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
    if let Some(ema_model) = ema_model {
        ema_model
            .clone()
            .save_file(checkpoint_base_path(&dir, "ema_model"), &recorder)
            .with_context(|| {
                format!(
                    "failed to write {}",
                    checkpoint_file_path(&dir, "ema_model").display()
                )
            })?;
    }

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
        "ema_model_path": ema_model.map(|_| checkpoint_file_path(&dir, "ema_model")),
        "ema_decay": config.ema_decay,
        "ema_start_step": config.ema_start_step,
        "training_config": run_config_json(config),
    });
    let path = checkpoint_metadata_path(&dir);
    fs::write(&path, serde_json::to_string_pretty(&metadata)?)
        .with_context(|| format!("failed to write {}", path.display()))?;

    let legacy_path = config.output_dir.join("checkpoint_latest.json");
    fs::write(&legacy_path, serde_json::to_string_pretty(&metadata)?)
        .with_context(|| format!("failed to write {}", legacy_path.display()))?;
    append_structured_event(
        config,
        "checkpoint_saved",
        json!({
            "epoch": epoch,
            "epoch_complete": epoch_complete,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_cer": val_cer,
            "val_wer": val_wer,
            "checkpoint_dir": dir,
            "model_path": checkpoint_file_path(&dir, "model"),
            "optimizer_path": checkpoint_file_path(&dir, "optimizer"),
            "ema_model_path": ema_model.map(|_| checkpoint_file_path(&dir, "ema_model")),
        }),
    )?;
    maybe_upload_training_checkpoint(config, &dir)?;
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
        "w2v_num_adapter_layers",
        "w2v_adapter_stride",
        "w2v_adapter_kernel_size",
        "precision",
        "max_audio_duration_ms",
        "gradient_accumulation_steps",
        "lr_warmup_steps",
        "lr_hold_steps",
        "lr_decay_steps",
        "lr_warmup_epochs",
        "lr_hold_epochs",
        "lr_decay_exponent",
        "lr_min",
        "ema_decay",
        "ema_start_step",
    ] {
        let current_value = current.get(key).cloned();
        let checkpoint_value = saved_config
            .get(key)
            .cloned()
            .or_else(|| resume_config_default_value(key));
        if current_value != checkpoint_value {
            bail!(
                "resume checkpoint config mismatch for '{key}': current={:?} checkpoint={:?}",
                current_value,
                checkpoint_value
            );
        }
    }
    Ok(())
}

fn resume_config_default_value(key: &str) -> Option<Value> {
    match key {
        "ema_decay" => Some(Value::Null),
        "ema_start_step" => Some(json!(0)),
        "max_audio_duration_ms" => Some(Value::Null),
        "lr_warmup_epochs" => Some(json!(0)),
        "lr_hold_epochs" => Some(json!(0)),
        "lr_decay_exponent" => Some(json!(0.0)),
        _ => None,
    }
}

fn load_initial_weights<B, M>(
    model: M,
    config: &BurnTrainConfig,
    resume: &Option<ResumeCheckpoint>,
    device: &B::Device,
) -> Result<M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let Some(path) = &config.init_from else {
        return Ok(model);
    };
    if resume.is_some() {
        bail!("--init-from cannot be combined with --resume-from");
    }
    if is_positiveloss_weights_json(path) {
        let weights_path = resolve_positiveloss_weights_json(path)?;
        return load_safetensors_initial_weights(model, &weights_path, device);
    }
    if is_safetensors_path(path) {
        return load_safetensors_initial_weights(model, path, device);
    }
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .load_file(resolve_model_checkpoint_base_path(path), &recorder, device)
        .with_context(|| format!("failed to initialize model from {}", path.display()))
}

fn is_safetensors_path(path: &Path) -> bool {
    path.extension().and_then(|value| value.to_str()) == Some("safetensors")
}

fn is_positiveloss_weights_json(path: &Path) -> bool {
    path.file_name().and_then(|value| value.to_str()) == Some("weights.json")
}

fn resolve_positiveloss_weights_json(path: &Path) -> Result<PathBuf> {
    let sibling = path.with_file_name("model.safetensors");
    if sibling.exists() {
        return Ok(sibling);
    }
    let text = fs::read_to_string(path).with_context(|| {
        format!(
            "failed to read PositiveLoss weights metadata {}",
            path.display()
        )
    })?;
    let metadata: Value = serde_json::from_str(&text).with_context(|| {
        format!(
            "failed to parse PositiveLoss weights metadata {}",
            path.display()
        )
    })?;
    let weights = json_path(&metadata, "weights").with_context(|| {
        format!(
            "PositiveLoss weights metadata {} does not contain a weights path and sibling model.safetensors is missing",
            path.display()
        )
    })?;
    Ok(weights)
}

fn resolve_model_checkpoint_base_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        checkpoint_base_path(path, "model")
    } else if path.file_name().and_then(|value| value.to_str()) == Some("checkpoint.json") {
        path.parent()
            .map(|dir| checkpoint_base_path(dir, "model"))
            .unwrap_or_else(|| path.with_file_name("model"))
    } else if path.extension().and_then(|value| value.to_str()) == Some("bin") {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

fn load_safetensors_initial_weights<B, M>(model: M, path: &Path, device: &B::Device) -> Result<M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    let mut weights = HashMap::new();
    let mut skipped_dtype = 0usize;
    for name in tensors.names() {
        let view = tensors
            .tensor(name)
            .with_context(|| format!("failed to read tensor '{name}'"))?;
        if view.dtype() != Dtype::F32 {
            skipped_dtype += 1;
            continue;
        }
        let shape = view.shape().to_vec();
        let values = f32_values_from_safetensor(view.data())?;
        let data = TensorData::new(values, shape);
        insert_warm_start_aliases(&mut weights, name, data);
    }

    let mut mapper = WarmStartMapper::<B> {
        weights,
        path: Vec::new(),
        device,
        loaded: 0,
        skipped_shape: 0,
        transposed: 0,
        skipped_shape_examples: Vec::new(),
        phantom: std::marker::PhantomData,
    };
    let model = model.map(&mut mapper);
    log::info!(
        "initialized {} tensor(s) from {}, transposed={}, skipped_shape={}, skipped_dtype={}",
        mapper.loaded,
        path.display(),
        mapper.transposed,
        mapper.skipped_shape,
        skipped_dtype,
    );
    if !mapper.skipped_shape_examples.is_empty() {
        log::warn!(
            "warm-start skipped shape examples: {}",
            mapper.skipped_shape_examples.join(", ")
        );
    }
    Ok(model)
}

fn f32_values_from_safetensor(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        bail!("f32 safetensor byte length is not divisible by 4");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn safetensor_name_aliases(name: &str) -> Vec<String> {
    let mut aliases = vec![name.to_string()];
    for prefix in ["module.", "model.", "model.model."] {
        if let Some(stripped) = name.strip_prefix(prefix) {
            aliases.push(stripped.to_string());
        }
    }
    aliases.sort();
    aliases.dedup();
    aliases
}

fn insert_warm_start_aliases(
    weights: &mut HashMap<String, TensorData>,
    name: &str,
    data: TensorData,
) {
    for alias in safetensor_name_aliases(name) {
        weights.entry(alias).or_insert_with(|| data.clone());
    }
    for (alias, alias_data) in paraformer_split_in_proj_aliases(name, &data) {
        weights.entry(alias).or_insert(alias_data);
    }
}

fn paraformer_split_in_proj_aliases(name: &str, data: &TensorData) -> Vec<(String, TensorData)> {
    let Some(prefix) = name
        .strip_suffix(".in_proj_weight")
        .or_else(|| name.strip_suffix(".in_proj_bias"))
    else {
        return Vec::new();
    };
    let Some(target_prefix) = paraformer_in_proj_target_prefix(prefix) else {
        return Vec::new();
    };
    let shape = data.shape.as_slice();
    let is_weight = name.ends_with(".in_proj_weight");
    let Ok(values) = data.clone().into_vec::<f32>() else {
        return Vec::new();
    };
    let projections = ["query", "key", "value"];
    if is_weight {
        if shape.len() != 2 || shape[0] % 3 != 0 {
            return Vec::new();
        }
        let rows = shape[0] / 3;
        let cols = shape[1];
        let mut aliases = Vec::with_capacity(3);
        for (index, projection) in projections.iter().enumerate() {
            let start = index * rows * cols;
            let end = start + rows * cols;
            aliases.push((
                format!("{target_prefix}.{projection}.weight"),
                TensorData::new(values[start..end].to_vec(), [rows, cols]),
            ));
        }
        aliases
    } else {
        if shape.len() != 1 || shape[0] % 3 != 0 {
            return Vec::new();
        }
        let len = shape[0] / 3;
        let mut aliases = Vec::with_capacity(3);
        for (index, projection) in projections.iter().enumerate() {
            let start = index * len;
            let end = start + len;
            aliases.push((
                format!("{target_prefix}.{projection}.bias"),
                TensorData::new(values[start..end].to_vec(), [len]),
            ));
        }
        aliases
    }
}

fn paraformer_in_proj_target_prefix(prefix: &str) -> Option<String> {
    if let Some(layer) = prefix
        .strip_prefix("encoder.layers.")
        .and_then(|suffix| suffix.strip_suffix(".self_attn"))
    {
        return Some(format!("encoder.layers.{layer}.self_attn"));
    }
    let parts = prefix.split('.').collect::<Vec<_>>();
    if parts.len() == 4 && parts[0] == "decoder" && parts[1] == "layers" {
        let layer = parts[2];
        return match parts[3] {
            "self_attn" => Some(format!("decoder.layers.{layer}.self_attn")),
            "multihead_attn" => Some(format!("decoder.layers.{layer}.cross_attn")),
            _ => None,
        };
    }
    None
}

struct WarmStartMapper<'a, B: AutodiffBackend> {
    weights: HashMap<String, TensorData>,
    path: Vec<String>,
    device: &'a B::Device,
    loaded: usize,
    skipped_shape: usize,
    transposed: usize,
    skipped_shape_examples: Vec<String>,
    phantom: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleMapper<B> for WarmStartMapper<'_, B> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, current, mapper) = param.consume();
        let path = self.path.join(".");
        let data = warm_start_name_candidates(&path)
            .into_iter()
            .find_map(|candidate| self.weights.get(&candidate).cloned());
        let Some(data) = data else {
            return Param::from_mapped_value(id, current, mapper);
        };
        let (data, transposed) = maybe_transpose_warm_start_data(&path, data, &current.dims());
        let data = maybe_reshape_warm_start_scalar(data, &current.dims());
        if transposed {
            self.transposed += 1;
        }
        if data.shape.as_slice() != current.dims() {
            self.skipped_shape += 1;
            if self.skipped_shape_examples.len() < 8 {
                self.skipped_shape_examples.push(format!(
                    "{path}: source={:?} target={:?}",
                    data.shape.as_slice(),
                    current.dims()
                ));
            }
            return Param::from_mapped_value(id, current, mapper);
        }
        let require_grad = current.is_require_grad();
        let tensor = Tensor::<B, D>::from_data(data, self.device).set_require_grad(require_grad);
        self.loaded += 1;
        Param::from_mapped_value(id, tensor, mapper)
    }
}

fn maybe_reshape_warm_start_scalar<const D: usize>(
    data: TensorData,
    target_shape: &[usize; D],
) -> TensorData {
    if data.shape.as_slice().is_empty() && target_shape.as_slice() == [1] {
        let Ok(values) = data.into_vec::<f32>() else {
            return TensorData::new(Vec::<f32>::new(), [0]);
        };
        return TensorData::new(values, [1]);
    }
    data
}

fn maybe_transpose_warm_start_data<const D: usize>(
    path: &str,
    data: TensorData,
    target_shape: &[usize; D],
) -> (TensorData, bool) {
    if !should_transpose_warm_start_2d(path) || data.shape.as_slice().len() != 2 {
        return (data, false);
    }
    let source_shape = data.shape.as_slice();
    let [rows, cols] = [source_shape[0], source_shape[1]];
    if target_shape.as_slice() != [cols, rows] && target_shape.as_slice() != [rows, cols] {
        return (data, false);
    }
    let Ok(values) = data.into_vec::<f32>() else {
        return (TensorData::new(Vec::<f32>::new(), [0]), false);
    };
    let mut transposed = vec![0.0f32; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = values[row * cols + col];
        }
    }
    (TensorData::new(transposed, [cols, rows]), true)
}

fn should_transpose_warm_start_2d(path: &str) -> bool {
    path.ends_with(".weight")
        && (path == "classifier.weight"
            || path.contains("input_projection.")
            || path.contains("feature_projection.projection.")
            || path.contains("encoder.subsampling.projection.")
            || path.contains(".attention.")
            || path.contains(".mha.")
            || path.contains(".ff1.linear_")
            || path.contains(".ff2.linear_")
            || path.contains(".linear_q.")
            || path.contains(".linear_k.")
            || path.contains(".linear_v.")
            || path.contains(".linear_out.")
            || path.contains(".intermediate_dense.")
            || path.contains(".output_dense.")
            || path.contains(".feed_forward.")
            || path.contains(".feed_forward1.")
            || path.contains(".feed_forward2.")
            || path.contains(".feed_forward3.")
            || path.contains(".attention_weights.")
            || path.contains(".non_linear_attention.")
            || path.contains(".self_attention1.")
            || path.contains(".self_attention2.")
            || path.contains(".conv1.input_proj.")
            || path.contains(".conv1.output_proj.")
            || path.contains(".conv2.input_proj.")
            || path.contains(".conv2.output_proj.")
            || path.contains(".conv_embed.convnext_in.")
            || path.contains(".conv_embed.convnext_out.")
            || path.contains(".conv_embed.output_projection.")
            || path.contains(".pwff.")
            || path.contains(".self_attn.")
            || path.contains(".cross_attn.")
            || path == "ctc_projection.weight"
            || path == "posterior_embed.weight"
            || path == "decoder_projection.weight"
            || path.contains("time_recovery.projection."))
}

fn warm_start_name_candidates(path: &str) -> Vec<String> {
    let mut candidates = vec![path.to_string()];
    if let Some(alias) = squeezeformer_positiveloss_alias(path) {
        candidates.push(alias);
    }
    if let Some(alias) = wav2vec_positiveloss_alias(path) {
        candidates.push(alias);
    }
    if let Some(alias) = paraformer_positiveloss_alias(path) {
        candidates.push(alias);
    }
    if let Some(alias) = zipformer_positiveloss_alias(path) {
        candidates.push(alias);
    }
    candidates.sort();
    candidates.dedup();
    candidates
}

fn zipformer_positiveloss_alias(path: &str) -> Option<String> {
    if let Some(suffix) = path.strip_prefix("encoder.conv_embed.convnext_depthwise.") {
        return Some(format!(
            "encoder.encoder.conv_embed.convnext.depthwise.{suffix}"
        ));
    }
    if let Some(suffix) = path.strip_prefix("encoder.conv_embed.convnext_in.") {
        return Some(format!(
            "encoder.encoder.conv_embed.convnext.pointwise_in.{suffix}"
        ));
    }
    if let Some(suffix) = path.strip_prefix("encoder.conv_embed.convnext_out.") {
        return Some(format!(
            "encoder.encoder.conv_embed.convnext.pointwise_out.{suffix}"
        ));
    }
    if let Some(suffix) = path.strip_prefix("encoder.conv_embed.") {
        return Some(format!("encoder.encoder.conv_embed.{suffix}"));
    }
    if let Some(suffix) = path.strip_prefix("encoder.output_downsample.") {
        return Some(format!("encoder.encoder.output_downsample.{suffix}"));
    }

    let parts = path.split('.').collect::<Vec<_>>();
    if parts.len() >= 5 && parts[0] == "encoder" && parts[1] == "stacks" {
        let stack = parts[2];
        let rest = &parts[3..];
        if rest.first() == Some(&"downsample") {
            return Some(format!("encoder.encoder.stacks.{stack}.{}", rest.join(".")));
        }
        if rest.first() == Some(&"output_bypass") {
            return Some(format!("encoder.encoder.stacks.{stack}.{}", rest.join(".")));
        }
        if rest.first() == Some(&"blocks") && rest.len() >= 3 {
            let block = rest[1];
            let tail = &rest[2..];
            let block_prefix = if stack == "0" {
                format!("encoder.encoder.stacks.{stack}.blocks.{block}")
            } else {
                format!("encoder.encoder.stacks.{stack}.stack.blocks.{block}")
            };
            return zipformer_block_alias(&block_prefix, tail);
        }
    }
    None
}

fn zipformer_block_alias(prefix: &str, rest: &[&str]) -> Option<String> {
    match rest {
        [
            "feed_forward1" | "feed_forward2" | "feed_forward3",
            "linear_in",
            tail @ ..,
        ] => Some(format!("{prefix}.{}.in_proj.{}", rest[0], tail.join("."))),
        [
            "feed_forward1" | "feed_forward2" | "feed_forward3",
            "linear_out",
            tail @ ..,
        ] => Some(format!("{prefix}.{}.out_proj.{}", rest[0], tail.join("."))),
        _ => Some(format!("{prefix}.{}", rest.join("."))),
    }
}

fn squeezeformer_positiveloss_alias(path: &str) -> Option<String> {
    if let Some(suffix) = path.strip_prefix("encoder.subsampling.depthwise.") {
        return Some(format!("encoder.subsampling.conv2_dw.{suffix}"));
    }
    if let Some(suffix) = path.strip_prefix("encoder.subsampling.pointwise.") {
        return Some(format!("encoder.subsampling.conv2_pw.{suffix}"));
    }
    if let Some(suffix) = path.strip_prefix("encoder.time_reduction.") {
        return Some(format!("encoder.time_reduce.7.{suffix}"));
    }
    if let Some(suffix) = path.strip_prefix("encoder.time_recovery.projection.") {
        return Some(format!("encoder.time_recover.15.proj.{suffix}"));
    }
    if let Some(suffix) = path.strip_prefix("encoder.input_norm.") {
        return Some(format!(
            "encoder.input_norm.{}",
            layer_norm_suffix_alias(suffix)
        ));
    }

    let parts = path.split('.').collect::<Vec<_>>();
    if parts.len() < 5 || parts[0] != "encoder" || parts[1] != "blocks" {
        return None;
    }
    let block = parts[2];
    let rest = &parts[3..];
    if rest.first() == Some(&"mhsa_ff") {
        return squeezeformer_mhsa_ff_alias(block, &rest[1..]);
    }
    if rest.first() == Some(&"conv_ff") {
        return squeezeformer_conv_ff_alias(block, &rest[1..]);
    }
    None
}

fn paraformer_positiveloss_alias(path: &str) -> Option<String> {
    if let Some(suffix) = path.strip_prefix("encoder.subsampling.conv1.") {
        return Some(format!("encoder.subsampling.conv.0.{suffix}"));
    }
    if let Some(suffix) = path.strip_prefix("encoder.subsampling.conv2.") {
        return Some(format!("encoder.subsampling.conv.2.{suffix}"));
    }
    if let Some(suffix) = path.strip_prefix("encoder.subsampling.projection.") {
        return Some(format!("encoder.subsampling.proj.{suffix}"));
    }

    let parts = path.split('.').collect::<Vec<_>>();
    if parts.len() >= 5 && parts[0] == "encoder" && parts[1] == "layers" {
        let layer = parts[2];
        let rest = &parts[3..];
        return paraformer_encoder_layer_alias(layer, rest);
    }
    if parts.len() >= 4 && parts[0] == "decoder" && parts[1] == "layers" {
        let layer = parts[2];
        let rest = &parts[3..];
        return paraformer_decoder_layer_alias(layer, rest);
    }
    None
}

fn paraformer_encoder_layer_alias(layer: &str, rest: &[&str]) -> Option<String> {
    match rest {
        ["ff1", "norm", suffix] => Some(format!(
            "encoder.layers.{layer}.ff1.net.0.{}",
            layer_norm_suffix_alias(suffix)
        )),
        ["ff1", "linear_in", tail @ ..] => Some(format!(
            "encoder.layers.{layer}.ff1.net.1.{}",
            tail.join(".")
        )),
        ["ff1", "linear_out", tail @ ..] => Some(format!(
            "encoder.layers.{layer}.ff1.net.4.{}",
            tail.join(".")
        )),
        ["self_attn_norm", suffix] => Some(format!(
            "encoder.layers.{layer}.self_attn_norm.{}",
            layer_norm_suffix_alias(suffix)
        )),
        ["self_attn", "output", tail @ ..] => Some(format!(
            "encoder.layers.{layer}.self_attn.out_proj.{}",
            tail.join(".")
        )),
        ["conv_norm", suffix] => Some(format!(
            "encoder.layers.{layer}.conv_norm.{}",
            layer_norm_suffix_alias(suffix)
        )),
        ["conv_in", tail @ ..] => Some(format!("encoder.layers.{layer}.conv.0.{}", tail.join("."))),
        ["depthwise", tail @ ..] => {
            Some(format!("encoder.layers.{layer}.conv.2.{}", tail.join(".")))
        }
        ["batch_norm", "gamma"] => Some(format!("encoder.layers.{layer}.conv.3.weight")),
        ["batch_norm", "beta"] => Some(format!("encoder.layers.{layer}.conv.3.bias")),
        ["batch_norm", tail @ ..] => {
            Some(format!("encoder.layers.{layer}.conv.3.{}", tail.join(".")))
        }
        ["conv_out", tail @ ..] => {
            Some(format!("encoder.layers.{layer}.conv.5.{}", tail.join(".")))
        }
        ["ff2", "norm", suffix] => Some(format!(
            "encoder.layers.{layer}.ff2.net.0.{}",
            layer_norm_suffix_alias(suffix)
        )),
        ["ff2", "linear_in", tail @ ..] => Some(format!(
            "encoder.layers.{layer}.ff2.net.1.{}",
            tail.join(".")
        )),
        ["ff2", "linear_out", tail @ ..] => Some(format!(
            "encoder.layers.{layer}.ff2.net.4.{}",
            tail.join(".")
        )),
        ["final_norm", suffix] => Some(format!(
            "encoder.layers.{layer}.final_norm.{}",
            layer_norm_suffix_alias(suffix)
        )),
        _ => None,
    }
}

fn paraformer_decoder_layer_alias(layer: &str, rest: &[&str]) -> Option<String> {
    match rest {
        ["self_attn", "output", tail @ ..] => Some(format!(
            "decoder.layers.{layer}.self_attn.out_proj.{}",
            tail.join(".")
        )),
        ["cross_attn", "output", tail @ ..] => Some(format!(
            "decoder.layers.{layer}.multihead_attn.out_proj.{}",
            tail.join(".")
        )),
        ["pwff", "linear_inner", tail @ ..] => {
            Some(format!("decoder.layers.{layer}.linear1.{}", tail.join(".")))
        }
        ["pwff", "linear_outer", tail @ ..] => {
            Some(format!("decoder.layers.{layer}.linear2.{}", tail.join(".")))
        }
        ["norm_1", suffix] => Some(format!(
            "decoder.layers.{layer}.norm1.{}",
            layer_norm_suffix_alias(suffix)
        )),
        ["norm_2", suffix] => Some(format!(
            "decoder.layers.{layer}.norm2.{}",
            layer_norm_suffix_alias(suffix)
        )),
        ["norm_3", suffix] => Some(format!(
            "decoder.layers.{layer}.norm3.{}",
            layer_norm_suffix_alias(suffix)
        )),
        _ => None,
    }
}

fn wav2vec_positiveloss_alias(path: &str) -> Option<String> {
    if let Some(suffix) = path.strip_prefix("classifier.") {
        return Some(format!("lm_head.{suffix}"));
    }
    if let Some(suffix) = path.strip_prefix("encoder.feature_projection.layer_norm.") {
        return Some(format!(
            "wav2vec2_bert.feature_projection.layer_norm.{}",
            layer_norm_suffix_alias(suffix)
        ));
    }
    if let Some(suffix) = path.strip_prefix("encoder.feature_projection.projection.") {
        return Some(format!(
            "wav2vec2_bert.feature_projection.projection.{suffix}"
        ));
    }
    if path == "encoder.masked_spec_embed" {
        return Some("wav2vec2_bert.masked_spec_embed".to_string());
    }
    if let Some(suffix) = path.strip_prefix("encoder.adapter.layers.") {
        return Some(wav2vec_direct_alias("wav2vec2_bert.adapter.layers", suffix));
    }

    let parts = path.split('.').collect::<Vec<_>>();
    if parts.len() < 6 || parts[0] != "encoder" || parts[1] != "encoder" || parts[2] != "layers" {
        return None;
    }
    let layer = parts[3];
    let rest = &parts[4..];
    match rest {
        ["self_attn", tail @ ..] => Some(wav2vec_direct_alias(
            &format!("wav2vec2_bert.encoder.layers.{layer}.self_attn"),
            &tail.join("."),
        )),
        ["ffn1", tail @ ..] => Some(wav2vec_direct_alias(
            &format!("wav2vec2_bert.encoder.layers.{layer}.ffn1"),
            &tail.join("."),
        )),
        ["ffn2", tail @ ..] => Some(wav2vec_direct_alias(
            &format!("wav2vec2_bert.encoder.layers.{layer}.ffn2"),
            &tail.join("."),
        )),
        ["conv_module", tail @ ..] => Some(wav2vec_direct_alias(
            &format!("wav2vec2_bert.encoder.layers.{layer}.conv_module"),
            &tail.join("."),
        )),
        [norm, suffix] if norm.ends_with("_layer_norm") => Some(format!(
            "wav2vec2_bert.encoder.layers.{layer}.{norm}.{}",
            layer_norm_suffix_alias(suffix)
        )),
        ["mha", projection, tail @ ..] => {
            let projection = match *projection {
                "query" => "linear_q",
                "key" => "linear_k",
                "value" => "linear_v",
                "output" => "linear_out",
                other => other,
            };
            Some(format!(
                "wav2vec2_bert.encoder.layers.{layer}.self_attn.{projection}.{}",
                tail.join(".")
            ))
        }
        ["pwff", "linear_inner", tail @ ..] => Some(format!(
            "wav2vec2_bert.encoder.layers.{layer}.ffn1.intermediate_dense.{}",
            tail.join(".")
        )),
        ["pwff", "linear_outer", tail @ ..] => Some(format!(
            "wav2vec2_bert.encoder.layers.{layer}.ffn1.output_dense.{}",
            tail.join(".")
        )),
        ["norm_2", suffix] => Some(format!(
            "wav2vec2_bert.encoder.layers.{layer}.self_attn_layer_norm.{}",
            layer_norm_suffix_alias(suffix)
        )),
        ["norm_1", suffix] => Some(format!(
            "wav2vec2_bert.encoder.layers.{layer}.final_layer_norm.{}",
            layer_norm_suffix_alias(suffix)
        )),
        _ => None,
    }
}

fn wav2vec_direct_alias(prefix: &str, suffix: &str) -> String {
    let suffix = suffix
        .split('.')
        .map(layer_norm_suffix_alias)
        .collect::<Vec<_>>()
        .join(".");
    if suffix.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

fn layer_norm_suffix_alias(suffix: &str) -> &str {
    match suffix {
        "gamma" => "weight",
        "beta" => "bias",
        other => other,
    }
}

fn squeezeformer_mhsa_ff_alias(block: &str, rest: &[&str]) -> Option<String> {
    match rest {
        ["attention", "input_transform", tail @ ..] => Some(format!(
            "encoder.blocks.{block}.layers.0.attn.input_transform.{}",
            tail.join(".")
        )),
        ["attention", "attention", projection, tail @ ..] => {
            let projection = match *projection {
                "query_proj" => "query",
                "key_proj" => "key",
                "value_proj" => "value",
                other => other,
            };
            Some(join_weight_path(
                &format!("encoder.blocks.{block}.layers.0.attn.attn.{projection}"),
                tail,
            ))
        }
        ["mid_norm", tail @ ..] => Some(join_weight_path(
            &format!("encoder.blocks.{block}.layers.0.mid_norm"),
            tail,
        )),
        ["feed_forward", "input_transform", tail @ ..] => Some(format!(
            "encoder.blocks.{block}.layers.0.ff.input_transform.{}",
            tail.join(".")
        )),
        ["feed_forward", linear, tail @ ..] => {
            let linear = match *linear {
                "linear_in" => "linear1",
                "linear_out" => "linear2",
                other => other,
            };
            Some(format!(
                "encoder.blocks.{block}.layers.0.ff.{linear}.{}",
                tail.join(".")
            ))
        }
        ["out_norm", tail @ ..] => Some(join_weight_path(
            &format!("encoder.blocks.{block}.layers.0.out_norm"),
            tail,
        )),
        _ => None,
    }
}

fn squeezeformer_conv_ff_alias(block: &str, rest: &[&str]) -> Option<String> {
    match rest {
        ["convolution", tail @ ..] => Some(join_weight_path(
            &format!("encoder.blocks.{block}.layers.2.conv"),
            tail,
        )),
        ["mid_norm", tail @ ..] => Some(join_weight_path(
            &format!("encoder.blocks.{block}.layers.2.mid_norm"),
            tail,
        )),
        ["feed_forward", "input_transform", tail @ ..] => Some(format!(
            "encoder.blocks.{block}.layers.2.ff.input_transform.{}",
            tail.join(".")
        )),
        ["feed_forward", linear, tail @ ..] => {
            let linear = match *linear {
                "linear_in" => "linear1",
                "linear_out" => "linear2",
                other => other,
            };
            Some(format!(
                "encoder.blocks.{block}.layers.2.ff.{linear}.{}",
                tail.join(".")
            ))
        }
        ["out_norm", tail @ ..] => Some(join_weight_path(
            &format!("encoder.blocks.{block}.layers.2.out_norm"),
            tail,
        )),
        _ => None,
    }
}

fn join_weight_path(prefix: &str, tail: &[&str]) -> String {
    if tail.is_empty() {
        return prefix.to_string();
    }
    let suffix = tail
        .iter()
        .map(|part| match *part {
            "gamma" => "weight",
            "beta" => "bias",
            other => other,
        })
        .collect::<Vec<_>>()
        .join(".");
    format!("{prefix}.{suffix}")
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

fn load_ema_checkpoint<B, M>(
    ema_model: Option<M>,
    resume: &Option<ResumeCheckpoint>,
    device: &B::Device,
) -> Result<Option<M>>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let Some(model) = ema_model else {
        return Ok(None);
    };
    let Some(checkpoint) = resume else {
        return Ok(Some(model));
    };
    let path = checkpoint_file_path(&checkpoint.dir, "ema_model");
    if !path.exists() {
        return Ok(Some(model));
    }
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .load_file(
            checkpoint_base_path(&checkpoint.dir, "ema_model"),
            &recorder,
            device,
        )
        .map(Some)
        .with_context(|| format!("failed to load {}", path.display()))
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
    fn manifest_derives_tokens_from_transcript_with_tokenizer() {
        let tokenizer = SentencePieceTokenizer::new(
            vec![
                "<unk>".to_string(),
                "▁".to_string(),
                "a".to_string(),
                "▁a".to_string(),
            ],
            Vec::new(),
            None,
            0,
        )
        .unwrap();
        let value = json!({"text": "a"});

        let tokens = parse_record_tokens(&value, Some("a"), Some(&tokenizer), "a").unwrap();

        assert_eq!(tokens, vec![3]);
    }

    #[test]
    fn manifest_extracts_features_from_audio_path() {
        let dir =
            std::env::temp_dir().join(format!("w2v_bert_uk_audio_manifest_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let audio_path = dir.join("sample.wav");
        let samples = vec![0_i16; 16_000];
        fs::write(&audio_path, mono_pcm_wav_bytes(16_000, &samples)).unwrap();
        let line = r#"{"id":"a","audio_path":"sample.wav","tokens":[1],"text":"a"}"#;

        let record = parse_json_record(
            line,
            &dir,
            1,
            None,
            &AudioDecodeConfig::default(),
            &default_training_feature_extractor(),
            WaveformAugmentConfig::default(),
        )
        .unwrap();

        assert_eq!(record.id, "a");
        assert_eq!(record.tokens, vec![1]);
        assert!(record.rows > 0);
        assert!(record.cols > 0);
        assert_eq!(record.features.len(), record.rows * record.cols);
    }

    #[test]
    fn architecture_selects_matching_feature_extractor() {
        let squeezeformer = BurnTrainConfig {
            architecture: TrainArchitecture::Squeezeformer,
            ..BurnTrainConfig::default()
        };
        let zipformer = BurnTrainConfig {
            architecture: TrainArchitecture::Zipformer,
            ..BurnTrainConfig::default()
        };
        let paraformer = BurnTrainConfig {
            architecture: TrainArchitecture::Paraformer,
            ..BurnTrainConfig::default()
        };
        let wav2vec = BurnTrainConfig {
            architecture: TrainArchitecture::Wav2VecBert,
            ..BurnTrainConfig::default()
        };

        assert_eq!(
            feature_extractor_for_architecture(&squeezeformer).feature_dim(),
            80
        );
        assert_eq!(
            feature_extractor_for_architecture(&zipformer).feature_dim(),
            80
        );
        assert_eq!(
            feature_extractor_for_architecture(&paraformer).feature_dim(),
            80
        );
        let FeatureExtractorConfig::Audio(paraformer_config) =
            feature_extractor_for_architecture(&paraformer)
        else {
            panic!("paraformer should use audio frontend features");
        };
        assert_eq!(
            paraformer_config.preemphasis,
            asr_features::paraformer_frontend_config().preemphasis
        );
        assert_eq!(
            feature_extractor_for_architecture(&wav2vec).feature_dim(),
            160
        );
    }

    #[test]
    fn raw_audio_directory_loads_with_token_sidecars() {
        let dir = std::env::temp_dir().join(format!(
            "w2v_bert_uk_raw_audio_dataset_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("nested")).unwrap();
        let audio_path = dir.join("nested").join("sample.wav");
        fs::write(
            &audio_path,
            mono_pcm_wav_bytes(16_000, &vec![0_i16; 16_000]),
        )
        .unwrap();
        fs::write(audio_path.with_extension("tokens"), "1 2 3").unwrap();
        fs::write(audio_path.with_extension("txt"), "hello").unwrap();

        let mut loader = StreamingBatchLoader::new(
            dir.clone(),
            1,
            None,
            false,
            4096,
            160,
            None,
            None,
            None,
            None,
            WaveformAugmentConfig::default(),
            default_training_feature_extractor(),
        )
        .unwrap();
        let batch = loader.next_batch().unwrap().unwrap();

        assert_eq!(batch.ids, vec!["sample"]);
        assert_eq!(batch.target_lengths, vec![3]);
        assert_eq!(batch.reference_texts, vec![Some("hello".to_string())]);
        assert!(batch.feature_lengths[0] > 0);
    }

    #[test]
    fn parquet_folder_discovers_local_testdata_metadata() {
        let path = Path::new("testdata");
        if !path.exists() {
            return;
        }

        let files = manifest_files(path).unwrap();
        assert!(files.iter().any(|path| is_parquet_file(path)));
        let parquet = files.iter().find(|path| is_parquet_file(path)).unwrap();
        let raw_rows = parquet_raw_lines(parquet).unwrap();
        assert!(!raw_rows.is_empty());
        let metadata = raw_rows[0].parse_metadata().unwrap();
        assert!(!metadata.id.is_empty());
    }

    #[test]
    fn batch_pads_features_and_targets() {
        let records = vec![
            FeatureRecord {
                id: "a".to_string(),
                rows: 2,
                cols: 2,
                features: vec![1.0, 2.0, 3.0, 4.0],
                duration_ms: 20,
                tokens: vec![1, 2],
                text: Some("one two".to_string()),
            },
            FeatureRecord {
                id: "b".to_string(),
                rows: 1,
                cols: 2,
                features: vec![5.0, 6.0],
                duration_ms: 10,
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
            duration_ms: 10,
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
        let summary = metrics.summary(0.5, Vec::new());

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
    fn resume_config_accepts_pre_ema_checkpoints_when_ema_disabled() {
        let config = BurnTrainConfig::default();
        let mut saved = run_config_json(&config);
        saved.as_object_mut().unwrap().remove("ema_decay");
        saved.as_object_mut().unwrap().remove("ema_start_step");

        assert!(validate_resume_config(&config, &saved).is_ok());
    }

    #[test]
    fn scheduler_warmup_hold_decay_sequence() {
        let config = BurnTrainConfig {
            learning_rate: 1.0,
            lr_warmup_steps: 2,
            lr_hold_steps: 1,
            lr_decay_steps: 2,
            lr_min: 0.1,
            ..BurnTrainConfig::default()
        };

        let values = (1..=6)
            .map(|step| scheduled_learning_rate(&config, step, 1))
            .collect::<Vec<_>>();

        assert_eq!(values, vec![0.5, 1.0, 1.0, 0.55, 0.1, 0.1]);
    }

    #[test]
    fn scheduler_epoch_warmup_hold_decay_sequence() {
        let config = BurnTrainConfig {
            learning_rate: 1.0,
            lr_warmup_steps: 0,
            lr_hold_steps: 0,
            lr_decay_steps: 0,
            lr_warmup_epochs: 2,
            lr_hold_epochs: 1,
            lr_decay_exponent: 1.0,
            lr_min: 0.1,
            ..BurnTrainConfig::default()
        };

        let values = (1..=6)
            .map(|epoch| scheduled_learning_rate(&config, 1, epoch))
            .collect::<Vec<_>>();

        assert_eq!(values, vec![0.5, 1.0, 1.0, 2.0 / 3.0, 0.5, 0.4]);
    }

    #[test]
    fn ema_decay_validation_rejects_invalid_values() {
        for ema_decay in [Some(0.0), Some(1.0), Some(1.5)] {
            let config = BurnTrainConfig {
                ema_decay,
                ..BurnTrainConfig::default()
            };
            let err = validate_config(&config).unwrap_err();
            assert!(err.to_string().contains("ema_decay"));
        }
    }

    #[test]
    fn spec_augment_masks_feature_values() {
        let records = vec![FeatureRecord {
            id: "utt".to_string(),
            rows: 4,
            cols: 3,
            features: vec![1.0; 12],
            duration_ms: 40,
            tokens: vec![1],
            text: None,
        }];
        let batch = make_batch(&records, 3).unwrap();
        let mut features = batch.features.clone();

        apply_spec_augment(
            &mut features,
            &batch,
            SpecAugmentConfig {
                time_masks: 1,
                time_mask_max_frames: 4,
                frequency_masks: 1,
                frequency_mask_max_bins: 3,
            },
        );

        assert!(features.iter().any(|value| *value == 0.0));
    }

    #[test]
    fn augmentation_validation_rejects_invalid_values() {
        let config = BurnTrainConfig {
            spec_augment: SpecAugmentConfig {
                time_masks: 1,
                time_mask_max_frames: 0,
                ..SpecAugmentConfig::default()
            },
            ..BurnTrainConfig::default()
        };
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("time_mask_max_frames"));

        let config = BurnTrainConfig {
            waveform_augment: WaveformAugmentConfig {
                noise_std: -0.1,
                ..WaveformAugmentConfig::default()
            },
            ..BurnTrainConfig::default()
        };
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("noise_std"));

        let config = BurnTrainConfig {
            waveform_augment: WaveformAugmentConfig {
                gain_min_db: Some(3.0),
                gain_max_db: Some(-3.0),
                noise_std: 0.0,
            },
            ..BurnTrainConfig::default()
        };
        let err = validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("gain_min_db"));
    }

    #[test]
    fn parquet_audio_hints_are_filename_like_for_asr_features() {
        assert_eq!(
            audio_format_hint_from_path("clips/sample.opus"),
            Some("audio.opus".to_string())
        );
        assert_eq!(
            audio_format_hint_from_bytes(b"OggS......OpusHead"),
            Some("audio.opus".to_string())
        );
        assert_eq!(
            audio_format_hint_from_bytes(b"RIFF....WAVEfmt "),
            Some("audio.wav".to_string())
        );
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
            None,
            None,
            None,
            WaveformAugmentConfig::default(),
            default_training_feature_extractor(),
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
    fn streaming_loader_respects_adaptive_padded_duration_budget() {
        let dir = std::env::temp_dir().join(format!(
            "w2v_bert_uk_adaptive_duration_batch_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("train.jsonl"),
            [
                r#"{"id":"a","features":[[0.1,0.2],[0.3,0.4]],"duration_ms":2000,"tokens":[1]}"#,
                r#"{"id":"b","features":[[0.1,0.2],[0.3,0.4],[0.5,0.6]],"duration_ms":3000,"tokens":[1]}"#,
                r#"{"id":"c","features":[[0.1,0.2]],"duration_ms":1000,"tokens":[1]}"#,
            ]
            .join("\n"),
        )
        .unwrap();

        let mut loader = StreamingBatchLoader::new(
            dir.join("train.jsonl"),
            8,
            Some(AdaptiveBatchConfig {
                unit: AdaptiveBatchUnit::PaddedDurationMs,
                budget: 6000,
                max_samples: None,
            }),
            false,
            4096,
            2,
            None,
            None,
            None,
            None,
            WaveformAugmentConfig::default(),
            default_training_feature_extractor(),
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
    fn streaming_loader_filters_long_audio_duration() {
        let dir = std::env::temp_dir().join(format!(
            "w2v_bert_uk_duration_filter_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("train.jsonl"),
            [
                r#"{"id":"keep","features":[[0.1,0.2]],"duration_ms":19999,"tokens":[1]}"#,
                r#"{"id":"drop","features":[[0.1,0.2]],"duration_ms":20001,"tokens":[1]}"#,
            ]
            .join("\n"),
        )
        .unwrap();

        let mut loader = StreamingBatchLoader::new(
            dir.join("train.jsonl"),
            2,
            None,
            false,
            4096,
            2,
            None,
            Some(20_000),
            None,
            None,
            WaveformAugmentConfig::default(),
            default_training_feature_extractor(),
        )
        .unwrap();

        let batch = loader.next_batch().unwrap().unwrap();
        assert_eq!(batch.ids, vec!["keep"]);
        assert!(loader.next_batch().unwrap().is_none());
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

        let mut loader = StreamingBatchLoader::new(
            dir.join("train.jsonl"),
            3,
            None,
            true,
            3,
            2,
            None,
            None,
            None,
            None,
            WaveformAugmentConfig::default(),
            default_training_feature_extractor(),
        )
        .unwrap();

        let batch = loader.next_batch().unwrap().unwrap();

        assert_eq!(batch.feature_lengths, vec![3, 2, 1]);
    }

    #[test]
    fn streaming_loader_writes_and_reads_dataset_index_cache() {
        let dir = std::env::temp_dir().join(format!(
            "w2v_bert_uk_dataset_index_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("train.jsonl"),
            [
                r#"{"id":"short","features":[[0.1,0.2]],"tokens":[1]}"#,
                r#"{"id":"long","features":[[0.1,0.2],[0.3,0.4],[0.5,0.6]],"tokens":[1]}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let index_path = dir.join("index").join("train.index.json");

        let mut first_loader = StreamingBatchLoader::new(
            dir.join("train.jsonl"),
            2,
            None,
            true,
            2,
            2,
            None,
            None,
            None,
            Some(index_path.clone()),
            WaveformAugmentConfig::default(),
            default_training_feature_extractor(),
        )
        .unwrap();
        let first = first_loader.next_batch().unwrap().unwrap();
        assert_eq!(first.ids, vec!["long", "short"]);
        assert!(index_path.exists());

        let mut second_loader = StreamingBatchLoader::new(
            dir.join("train.jsonl"),
            2,
            None,
            true,
            2,
            2,
            None,
            None,
            None,
            Some(index_path),
            WaveformAugmentConfig::default(),
            default_training_feature_extractor(),
        )
        .unwrap();
        let second = second_loader.next_batch().unwrap().unwrap();
        assert_eq!(second.ids, vec!["long", "short"]);
    }

    fn mono_pcm_wav_bytes(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let data_len = (samples.len() * 2) as u32;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);

        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        bytes
    }
}
