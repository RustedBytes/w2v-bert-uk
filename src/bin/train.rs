use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    execute, queue,
    style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use env_logger::Env;
use polars::prelude::{
    AnyValue, DataFrame, IntoColumn, IntoSeries, ListChunked, NamedFrom, ParquetReader,
    ParquetWriter, SerReader, Series,
};
use serde_json::{Value, json};
use rust_asr::audio::{FeatureExtractorConfig, WaveformAugmentConfig};
use rust_asr::paraformer::ParaformerAlignmentMode;
use rust_asr::tokenizer::sentencepiece_vocab_size;
use rust_asr::train::{
    AdaptiveBatchConfig, AdaptiveBatchUnit, BurnTrainConfig, FeatureExtractionProgress,
    SpecAugmentConfig, TrainArchitecture, TrainBackendKind, TrainPrecision,
    extract_feature_records, extract_feature_records_with_progress,
    feature_extractor_for_train_architecture, run_burn_training,
};

#[derive(Parser)]
#[command(
    author,
    version,
    about = "Train Burn ASR architectures with a Python-train.py-style CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<CommandArg>,

    #[command(flatten)]
    run: RunArgs,
}

#[derive(Subcommand)]
enum CommandArg {
    /// Run Burn model training. This is also the default when no subcommand is provided.
    Run(RunArgs),
    /// Train a SentencePiece tokenizer from manifest transcripts or text files.
    Tokenizer(TokenizerArgs),
    /// Extract audio features and token ids into a trainer-ready Parquet file.
    ExtractFeatures(ExtractFeaturesArgs),
}

#[derive(ClapArgs)]
struct RunArgs {
    /// Architecture to train.
    #[arg(long, value_enum, default_value_t = ArchitectureArg::Squeezeformer)]
    architecture: ArchitectureArg,

    /// Alias for --architecture zipformer.
    #[arg(long)]
    zipformer: bool,

    /// Alias for --architecture paraformer.
    #[arg(long)]
    paraformer: bool,

    /// Alias for --architecture w2v-bert.
    #[arg(long)]
    w2v_bert: bool,

    /// Training manifest, JSONL or TSV.
    #[arg(long)]
    train_manifest: Option<PathBuf>,

    /// Directory containing JSONL manifests. Uses train.jsonl for training and val.jsonl,
    /// validation.jsonl, or dev.jsonl for validation when present.
    #[arg(long)]
    manifest_dir: Option<PathBuf>,

    /// Optional validation manifest/set: file or folder with JSONL, TSV, Parquet, or audio records.
    #[arg(
        long,
        visible_aliases = ["validation-manifest", "valid-manifest", "validation-set", "val-set"]
    )]
    val_manifest: Option<PathBuf>,

    /// Output directory for run metadata and checkpoints.
    #[arg(long, default_value = "runs/burn")]
    output_dir: PathBuf,

    /// Model size preset: xs, s, sm, m, ml, l when supported.
    #[arg(long)]
    variant: Option<String>,

    /// Input feature dimension.
    #[arg(long, default_value_t = 80)]
    input_dim: usize,

    /// Vocabulary size including the blank symbol for CTC models.
    #[arg(long)]
    vocab_size: Option<usize>,

    /// CTC blank token id.
    #[arg(long, default_value_t = 0)]
    blank_id: usize,

    /// Model dimension for custom configs.
    #[arg(long, default_value_t = 256)]
    d_model: usize,

    /// Number of encoder layers for custom configs.
    #[arg(long, default_value_t = 16)]
    num_layers: usize,

    /// Number of attention heads for custom configs.
    #[arg(long, default_value_t = 4)]
    num_heads: usize,

    /// Batch size.
    #[arg(long, default_value_t = 8)]
    batch_size: usize,

    /// Adaptive batch unit: samples, frames, padded-frames, feature-values, duration-ms, or padded-duration-ms.
    #[arg(long, value_enum)]
    adaptive_batch_unit: Option<AdaptiveBatchUnitArg>,

    /// Adaptive batch budget measured in --adaptive-batch-unit.
    #[arg(long)]
    adaptive_batch_budget: Option<usize>,

    /// Optional hard cap on samples per adaptive batch. Defaults to --batch-size.
    #[arg(long)]
    adaptive_batch_max_samples: Option<usize>,

    /// Sort streamed records by descending frame length within a bounded buffer.
    #[arg(long)]
    sort_by_length_desc: bool,

    /// Number of records to hold for bounded length sorting.
    #[arg(long, default_value_t = 4096)]
    sort_buffer_size: usize,

    /// Directory for reusable manifest offset/length indexes.
    #[arg(long)]
    dataset_index_dir: Option<PathBuf>,

    /// Number of SpecAugment time masks per training batch.
    #[arg(long, default_value_t = 0)]
    spec_time_masks: usize,

    /// Maximum SpecAugment time-mask width in frames.
    #[arg(long, default_value_t = 0)]
    spec_time_mask_max_frames: usize,

    /// Number of SpecAugment frequency masks per training batch.
    #[arg(long, default_value_t = 0)]
    spec_freq_masks: usize,

    /// Maximum SpecAugment frequency-mask width in feature bins.
    #[arg(long, default_value_t = 0)]
    spec_freq_mask_max_bins: usize,

    /// Minimum random waveform gain in dB for audio_path records.
    #[arg(long)]
    waveform_gain_min_db: Option<f32>,

    /// Maximum random waveform gain in dB for audio_path records.
    #[arg(long)]
    waveform_gain_max_db: Option<f32>,

    /// Uniform waveform noise amplitude for audio_path records.
    #[arg(long, default_value_t = 0.0)]
    waveform_noise_std: f32,

    /// Number of epochs. Defaults to the PositiveLoss paper recipe.
    #[arg(long)]
    epochs: Option<usize>,

    /// AdamW learning rate. Defaults to the architecture recipe.
    #[arg(long)]
    learning_rate: Option<f64>,

    /// Linear warmup optimizer steps before reaching --learning-rate.
    #[arg(long)]
    lr_warmup_steps: Option<usize>,

    /// Optimizer steps to hold --learning-rate after warmup.
    #[arg(long)]
    lr_hold_steps: Option<usize>,

    /// Linear decay optimizer steps after warmup/hold.
    #[arg(long)]
    lr_decay_steps: Option<usize>,

    /// Linear warmup epochs before reaching --learning-rate.
    #[arg(long, alias = "warmup-epochs")]
    lr_warmup_epochs: Option<usize>,

    /// Epochs to hold --learning-rate after warmup.
    #[arg(long, alias = "hold-epochs")]
    lr_hold_epochs: Option<usize>,

    /// Inverse epoch-decay exponent after warmup/hold.
    #[arg(long, alias = "decay-exponent")]
    lr_decay_exponent: Option<f64>,

    /// Final learning rate after decay.
    #[arg(long, default_value_t = 0.0)]
    lr_min: f64,

    /// AdamW weight decay. Defaults to the PositiveLoss paper recipe.
    #[arg(long)]
    weight_decay: Option<f64>,

    /// Number of micro-batches to accumulate before each optimizer step.
    #[arg(long, default_value_t = 1)]
    gradient_accumulation_steps: usize,

    /// Clip gradients by L2 norm before optimizer updates.
    #[arg(long)]
    gradient_clip_norm: Option<f32>,

    /// Clip gradient values elementwise before optimizer updates.
    #[arg(long)]
    gradient_clip_value: Option<f32>,

    /// EMA decay for shadow model tracking, e.g. 0.9999. Disabled when unset.
    #[arg(long)]
    ema_decay: Option<f64>,

    /// First optimizer step that updates the EMA shadow model.
    #[arg(long, default_value_t = 0)]
    ema_start_step: usize,

    /// Log every N optimizer steps.
    #[arg(long, default_value_t = 10)]
    log_every: usize,

    /// Run validation every N steps; also validates at epoch end when --val-manifest is set.
    #[arg(long)]
    validate_every_steps: Option<usize>,

    /// Limit number of training samples, useful for smoke tests.
    #[arg(long)]
    max_train_samples: Option<usize>,

    /// Limit number of validation samples.
    #[arg(long)]
    max_val_samples: Option<usize>,

    /// Drop samples longer than this many seconds before batching.
    #[arg(long)]
    max_audio_duration_sec: Option<f64>,

    /// Optional SentencePiece tokenizer for validation CER/WER text decoding.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Validation CTC beam width. Use 1 for greedy decoding.
    #[arg(long, default_value_t = 1)]
    val_beam_width: usize,

    /// Number of validation CTC hypotheses to keep before optional LM reranking.
    #[arg(long)]
    val_n_best: Option<usize>,

    /// Optional KenLM model path for validation CTC N-best reranking.
    #[arg(long)]
    val_lm_path: Option<PathBuf>,

    /// KenLM shallow-fusion weight for validation decoding.
    #[arg(long, default_value_t = 0.45)]
    val_lm_weight: f32,

    /// Word insertion bonus for validation LM reranking.
    #[arg(long, default_value_t = 0.0)]
    val_lm_word_bonus: f32,

    /// Score validation LM candidates without beginning-of-sentence context.
    #[arg(long)]
    val_lm_no_bos: bool,

    /// Score validation LM candidates without end-of-sentence context.
    #[arg(long)]
    val_lm_no_eos: bool,

    /// Number of validation sample predictions to include in structured events.
    #[arg(long, default_value_t = 0)]
    val_log_samples: usize,

    /// Run forward/loss only and skip optimizer updates.
    #[arg(long)]
    dry_run: bool,

    /// Paraformer alignment strategy for decoder-query construction.
    #[arg(long, value_enum, default_value_t = ParaformerAlignmentArg::Viterbi)]
    paraformer_alignment_mode: ParaformerAlignmentArg,

    /// Use enhanced Paraformer-v2 with shallow CTC, boundary, and refinement heads.
    #[arg(long)]
    paraformer_enhanced: bool,

    /// Local Hugging Face W2V-BERT directory or config.json for Burn config loading.
    #[arg(long)]
    w2v_hf_model_dir: Option<PathBuf>,

    /// Import compatible W2V-BERT tensors from .safetensors in --w2v-hf-model-dir.
    #[arg(long)]
    w2v_hf_load_weights: bool,

    /// Use Burn balanced autodiff checkpointing for W2V-BERT training.
    #[arg(long)]
    w2v_activation_checkpointing: bool,

    /// Number of W2V-BERT adapter layers after the Conformer encoder.
    #[arg(long, default_value_t = 0)]
    w2v_num_adapter_layers: usize,

    /// W2V-BERT adapter convolution stride.
    #[arg(long, default_value_t = 2)]
    w2v_adapter_stride: usize,

    /// W2V-BERT adapter convolution kernel size.
    #[arg(long, default_value_t = 3)]
    w2v_adapter_kernel_size: usize,

    /// Resume from a checkpoint directory or checkpoint.json file.
    #[arg(long)]
    resume_from: Option<PathBuf>,

    /// Initialize model weights from a Burn checkpoint/export or PositiveLoss .safetensors file.
    #[arg(long)]
    init_from: Option<PathBuf>,

    /// Burn training backend: cpu, cuda, or wgpu.
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    backend: BackendArg,

    /// Device index for CUDA/WGPU backends.
    #[arg(long, default_value_t = 0)]
    device_index: usize,

    /// Comma-separated CUDA/WGPU device indices for replicated data-parallel training.
    #[arg(long, value_delimiter = ',')]
    device_indices: Vec<usize>,

    /// Training float precision: f32, f16, or bf16.
    #[arg(long, value_enum, default_value_t = PrecisionArg::F32)]
    precision: PrecisionArg,

    /// Shortcut for --precision f16.
    #[arg(long)]
    mixed_precision: bool,

    /// Show a live terminal UI with batch, throughput, loss, and validation metrics.
    #[arg(long)]
    tui: bool,

    /// Upload checkpoint_latest to Hugging Face after each checkpoint save.
    #[arg(long)]
    hf_upload_checkpoints: bool,

    /// Hugging Face model repository id for checkpoint uploads.
    #[arg(long)]
    hf_upload_repo_id: Option<String>,

    /// Checkpoint upload format. Rust training currently writes Burn .bin checkpoints.
    #[arg(long, value_enum, default_value_t = HfCheckpointFormatArg::BurnBin)]
    hf_upload_checkpoint_format: HfCheckpointFormatArg,

    /// Optional Hugging Face branch/revision for checkpoint uploads.
    #[arg(long)]
    hf_upload_revision: Option<String>,

    /// Create/use the Hugging Face repository as private.
    #[arg(long)]
    hf_upload_private: bool,
}

#[derive(ClapArgs)]
struct TokenizerArgs {
    /// Input manifest, text file, Parquet file, or directory. Can be passed multiple times.
    #[arg(long = "input", required = true)]
    inputs: Vec<PathBuf>,

    /// Output directory for the tokenizer files.
    #[arg(long, default_value = "tokenizer")]
    output_dir: PathBuf,

    /// SentencePiece model prefix. Creates <prefix>.model and <prefix>.vocab in --output-dir.
    #[arg(long, default_value = "tokenizer")]
    model_prefix: String,

    /// Vocabulary size for the SentencePiece model.
    #[arg(long, default_value_t = 5000)]
    vocab_size: usize,

    /// SentencePiece model type.
    #[arg(long, value_enum, default_value_t = SentencePieceModelTypeArg::Unigram)]
    model_type: SentencePieceModelTypeArg,

    /// Character coverage passed to SentencePiece.
    #[arg(long, default_value_t = 0.9995)]
    character_coverage: f64,

    /// Maximum sentence length accepted by SentencePiece.
    #[arg(long, default_value_t = 4192)]
    max_sentence_length: usize,

    /// Randomly sample this many sentences before training. 0 lets SentencePiece use all input.
    #[arg(long, default_value_t = 0)]
    input_sentence_size: usize,

    /// Shuffle sentences before SentencePiece sampling.
    #[arg(long)]
    shuffle_input_sentence: bool,

    /// Enable SentencePiece's extremely large corpus mode.
    #[arg(long)]
    train_extremely_large_corpus: bool,

    /// User-defined symbols, comma-separated or repeated.
    #[arg(long, value_delimiter = ',')]
    user_defined_symbols: Vec<String>,

    /// Control symbols, comma-separated or repeated.
    #[arg(long, value_delimiter = ',')]
    control_symbols: Vec<String>,

    /// Unknown token string.
    #[arg(long, default_value = "<unk>")]
    unk_piece: String,

    /// BOS token string.
    #[arg(long, default_value = "<s>")]
    bos_piece: String,

    /// EOS token string.
    #[arg(long, default_value = "</s>")]
    eos_piece: String,

    /// Padding token string.
    #[arg(long, default_value = "<pad>")]
    pad_piece: String,

    /// Unknown token id.
    #[arg(long, default_value_t = 0)]
    unk_id: i32,

    /// BOS token id. Use -1 to disable.
    #[arg(long, default_value_t = 1)]
    bos_id: i32,

    /// EOS token id. Use -1 to disable.
    #[arg(long, default_value_t = 2)]
    eos_id: i32,

    /// Padding token id. Use -1 to disable.
    #[arg(long, default_value_t = -1)]
    pad_id: i32,

    /// SentencePiece normalization rule name.
    #[arg(long, default_value = "nmt_nfkc")]
    normalization_rule_name: String,

    /// Enable SentencePiece byte fallback.
    #[arg(long)]
    byte_fallback: bool,

    /// Keep the generated plain-text corpus beside the tokenizer model.
    #[arg(long)]
    keep_corpus: bool,

    /// Explicit corpus output path. Implies --keep-corpus.
    #[arg(long)]
    corpus_output: Option<PathBuf>,

    /// SentencePiece trainer executable. Defaults to SENTENCEPIECE_TRAIN, spm_train, then sentencepiece-train.
    #[arg(long)]
    sentencepiece_command: Option<PathBuf>,
}

#[derive(ClapArgs)]
struct ExtractFeaturesArgs {
    /// Input manifest, Parquet file, audio file, or directory. Can be passed multiple times.
    #[arg(long = "input", required = true)]
    inputs: Vec<PathBuf>,

    /// Output Parquet file, or output directory when any input is a directory.
    #[arg(long)]
    output: PathBuf,

    /// Architecture/frontend used to extract features.
    #[arg(long, value_enum, default_value_t = ArchitectureArg::W2vBert)]
    architecture: ArchitectureArg,

    /// Alias for --architecture w2v-bert.
    #[arg(long)]
    w2v_bert: bool,

    /// SentencePiece tokenizer used when input rows have text but no token ids.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Drop samples longer than this many seconds before writing output.
    #[arg(long)]
    max_audio_duration_sec: Option<f64>,

    /// Limit number of output records.
    #[arg(long)]
    max_samples: Option<usize>,

    /// Number of Rayon worker threads used to decode audio and extract rows.
    #[arg(long)]
    jobs: Option<usize>,

    /// Show a live terminal UI while decoding audio and writing feature rows.
    #[arg(long)]
    tui: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SentencePieceModelTypeArg {
    Unigram,
    Bpe,
    Char,
    Word,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ArchitectureArg {
    Squeezeformer,
    Zipformer,
    Paraformer,
    #[value(alias = "wav2vec", alias = "wav2vec-bert", alias = "w2v_bert")]
    W2vBert,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendArg {
    Cpu,
    Cuda,
    Wgpu,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PrecisionArg {
    F32,
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum HfCheckpointFormatArg {
    #[value(alias = "burn_bin", alias = "bin")]
    BurnBin,
    Safetensors,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum AdaptiveBatchUnitArg {
    Samples,
    Frames,
    PaddedFrames,
    FeatureValues,
    DurationMs,
    PaddedDurationMs,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ParaformerAlignmentArg {
    Viterbi,
    Uniform,
    Greedy,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    match cli.command {
        Some(CommandArg::Run(args)) => run_training(args),
        Some(CommandArg::Tokenizer(args)) => run_tokenizer_training(args),
        Some(CommandArg::ExtractFeatures(args)) => run_feature_extraction(args),
        None => run_training(cli.run),
    }
}

fn run_training(args: RunArgs) -> Result<()> {
    if matches!(
        args.hf_upload_checkpoint_format,
        HfCheckpointFormatArg::Safetensors
    ) {
        bail!(
            "--hf-upload-checkpoint-format safetensors is not supported by Rust training yet; use burn-bin"
        );
    }
    let (train_manifest, val_manifest) = resolve_manifest_paths(&args)?;
    let adaptive_batch = resolve_adaptive_batch(&args)?;
    let precision = resolve_precision(&args);
    let device_indices = resolve_device_indices(&args);
    let architecture = resolve_architecture(&args);
    let defaults = training_defaults(architecture, args.variant.as_deref());
    let step_schedule_requested = args.lr_warmup_steps.is_some()
        || args.lr_hold_steps.is_some()
        || args.lr_decay_steps.is_some();
    let vocab_size = resolve_training_vocab_size(&args)?;
    let tokenizer_path = args.tokenizer.clone();
    let config = BurnTrainConfig {
        architecture,
        train_manifest,
        val_manifest,
        output_dir: args.output_dir,
        variant: args.variant,
        input_dim: args.input_dim,
        vocab_size,
        blank_id: args.blank_id,
        d_model: args.d_model,
        num_layers: args.num_layers,
        num_heads: args.num_heads,
        batch_size: args.batch_size,
        adaptive_batch,
        sort_by_length_desc: args.sort_by_length_desc,
        sort_buffer_size: args.sort_buffer_size,
        dataset_index_dir: args.dataset_index_dir,
        spec_augment: SpecAugmentConfig {
            time_masks: args.spec_time_masks,
            time_mask_max_frames: args.spec_time_mask_max_frames,
            frequency_masks: args.spec_freq_masks,
            frequency_mask_max_bins: args.spec_freq_mask_max_bins,
        },
        waveform_augment: WaveformAugmentConfig {
            gain_min_db: args.waveform_gain_min_db,
            gain_max_db: args.waveform_gain_max_db,
            noise_std: args.waveform_noise_std,
        },
        epochs: args.epochs.unwrap_or(defaults.epochs),
        learning_rate: args.learning_rate.unwrap_or(defaults.learning_rate),
        lr_warmup_steps: args.lr_warmup_steps.unwrap_or(0),
        lr_hold_steps: args.lr_hold_steps.unwrap_or(0),
        lr_decay_steps: args.lr_decay_steps.unwrap_or(0),
        lr_warmup_epochs: args.lr_warmup_epochs.unwrap_or(if step_schedule_requested {
            0
        } else {
            defaults.lr_warmup_epochs
        }),
        lr_hold_epochs: args.lr_hold_epochs.unwrap_or(if step_schedule_requested {
            0
        } else {
            defaults.lr_hold_epochs
        }),
        lr_decay_exponent: args
            .lr_decay_exponent
            .unwrap_or(if step_schedule_requested {
                0.0
            } else {
                defaults.lr_decay_exponent
            }),
        lr_min: args.lr_min,
        weight_decay: args.weight_decay.unwrap_or(defaults.weight_decay),
        gradient_accumulation_steps: args.gradient_accumulation_steps,
        gradient_clip_norm: args.gradient_clip_norm,
        gradient_clip_value: args.gradient_clip_value,
        ema_decay: args.ema_decay,
        ema_start_step: args.ema_start_step,
        log_every: args.log_every,
        validate_every_steps: args.validate_every_steps,
        max_train_samples: args.max_train_samples,
        max_val_samples: args.max_val_samples,
        max_audio_duration_ms: duration_sec_to_ms(args.max_audio_duration_sec)?,
        tokenizer_path,
        val_beam_width: args.val_beam_width,
        val_n_best: args.val_n_best.unwrap_or(args.val_beam_width),
        val_lm_path: args.val_lm_path,
        val_lm_weight: args.val_lm_weight,
        val_lm_word_bonus: args.val_lm_word_bonus,
        val_lm_bos: !args.val_lm_no_bos,
        val_lm_eos: !args.val_lm_no_eos,
        val_log_samples: args.val_log_samples,
        dry_run: args.dry_run,
        paraformer_alignment_mode: match args.paraformer_alignment_mode {
            ParaformerAlignmentArg::Viterbi => ParaformerAlignmentMode::Viterbi,
            ParaformerAlignmentArg::Uniform => ParaformerAlignmentMode::Uniform,
            ParaformerAlignmentArg::Greedy => ParaformerAlignmentMode::Greedy,
        },
        paraformer_enhanced: args.paraformer_enhanced,
        w2v_hf_model_dir: args.w2v_hf_model_dir,
        w2v_hf_load_weights: args.w2v_hf_load_weights,
        w2v_activation_checkpointing: args.w2v_activation_checkpointing,
        w2v_num_adapter_layers: args.w2v_num_adapter_layers,
        w2v_adapter_stride: args.w2v_adapter_stride,
        w2v_adapter_kernel_size: args.w2v_adapter_kernel_size,
        init_from: args.init_from,
        resume_from: args.resume_from,
        backend: match args.backend {
            BackendArg::Cpu => TrainBackendKind::Cpu,
            BackendArg::Cuda => TrainBackendKind::Cuda,
            BackendArg::Wgpu => TrainBackendKind::Wgpu,
        },
        device_index: args.device_index,
        device_indices,
        precision,
        tui: args.tui,
        hf_upload_checkpoints: args.hf_upload_checkpoints,
        hf_upload_repo_id: args.hf_upload_repo_id,
        hf_upload_revision: args.hf_upload_revision,
        hf_upload_private: args.hf_upload_private,
    };

    let summary = run_burn_training(config)?;
    println!(
        "training complete epochs={} steps={} last_train_loss={:?} last_val_loss={:?} last_val_cer={:?} last_val_wer={:?}",
        summary.epochs,
        summary.steps,
        summary.last_train_loss,
        summary.last_val_loss,
        summary.last_val_cer,
        summary.last_val_wer
    );
    Ok(())
}

fn resolve_device_indices(args: &RunArgs) -> Vec<usize> {
    if args.device_indices.is_empty() {
        vec![args.device_index]
    } else {
        args.device_indices.clone()
    }
}

fn resolve_training_vocab_size(args: &RunArgs) -> Result<usize> {
    if let Some(vocab_size) = args.vocab_size {
        return Ok(vocab_size);
    }
    let tokenizer = args.tokenizer.as_ref().ok_or_else(|| {
        anyhow!("--vocab-size is required unless --tokenizer is provided for auto-detection")
    })?;
    sentencepiece_vocab_size(tokenizer)
        .with_context(|| format!("failed to infer --vocab-size from {}", tokenizer.display()))
}

fn resolve_precision(args: &RunArgs) -> TrainPrecision {
    if args.mixed_precision && matches!(args.precision, PrecisionArg::F32) {
        return TrainPrecision::F16;
    }
    match args.precision {
        PrecisionArg::F32 => TrainPrecision::F32,
        PrecisionArg::F16 => TrainPrecision::F16,
        PrecisionArg::Bf16 => TrainPrecision::Bf16,
    }
}

#[derive(Clone, Copy, Debug)]
struct TrainingDefaults {
    epochs: usize,
    learning_rate: f64,
    weight_decay: f64,
    lr_warmup_epochs: usize,
    lr_hold_epochs: usize,
    lr_decay_exponent: f64,
}

fn training_defaults(architecture: TrainArchitecture, variant: Option<&str>) -> TrainingDefaults {
    TrainingDefaults {
        epochs: 500,
        learning_rate: architecture_default_learning_rate(architecture, variant),
        weight_decay: 5.0e-4,
        lr_warmup_epochs: 20,
        lr_hold_epochs: 160,
        lr_decay_exponent: 1.0,
    }
}

fn architecture_default_learning_rate(
    architecture: TrainArchitecture,
    variant: Option<&str>,
) -> f64 {
    const ADAMW_LR_CAP: f64 = 3.0e-4;
    match architecture {
        TrainArchitecture::Squeezeformer => variant_peak_learning_rate(variant).min(ADAMW_LR_CAP),
        TrainArchitecture::Zipformer | TrainArchitecture::Paraformer => 1.0e-3,
        TrainArchitecture::Wav2VecBert => 1.0e-4,
    }
}

fn variant_peak_learning_rate(variant: Option<&str>) -> f64 {
    match variant.unwrap_or("sm") {
        "xs" | "s" | "sm" => 2.0e-3,
        "m" => 1.5e-3,
        "ml" | "l" => 1.0e-3,
        _ => 1.0e-3,
    }
}

fn duration_sec_to_ms(value: Option<f64>) -> Result<Option<usize>> {
    value
        .map(|seconds| {
            if seconds <= 0.0 || !seconds.is_finite() {
                bail!("--max-audio-duration-sec must be a finite value > 0");
            }
            Ok((seconds * 1000.0).ceil() as usize)
        })
        .transpose()
}

fn resolve_architecture(args: &RunArgs) -> TrainArchitecture {
    if args.zipformer {
        return TrainArchitecture::Zipformer;
    }
    if args.paraformer {
        return TrainArchitecture::Paraformer;
    }
    if args.w2v_bert {
        return TrainArchitecture::Wav2VecBert;
    }
    match args.architecture {
        ArchitectureArg::Squeezeformer => TrainArchitecture::Squeezeformer,
        ArchitectureArg::Zipformer => TrainArchitecture::Zipformer,
        ArchitectureArg::Paraformer => TrainArchitecture::Paraformer,
        ArchitectureArg::W2vBert => TrainArchitecture::Wav2VecBert,
    }
}

fn resolve_extract_architecture(args: &ExtractFeaturesArgs) -> TrainArchitecture {
    if args.w2v_bert {
        TrainArchitecture::Wav2VecBert
    } else {
        match args.architecture {
            ArchitectureArg::Squeezeformer => TrainArchitecture::Squeezeformer,
            ArchitectureArg::Zipformer => TrainArchitecture::Zipformer,
            ArchitectureArg::Paraformer => TrainArchitecture::Paraformer,
            ArchitectureArg::W2vBert => TrainArchitecture::Wav2VecBert,
        }
    }
}

fn resolve_adaptive_batch(args: &RunArgs) -> Result<Option<AdaptiveBatchConfig>> {
    match (args.adaptive_batch_unit, args.adaptive_batch_budget) {
        (None, None) => Ok(None),
        (Some(unit), Some(budget)) => {
            if budget == 0 {
                anyhow::bail!("--adaptive-batch-budget must be > 0");
            }
            Ok(Some(AdaptiveBatchConfig {
                unit: match unit {
                    AdaptiveBatchUnitArg::Samples => AdaptiveBatchUnit::Samples,
                    AdaptiveBatchUnitArg::Frames => AdaptiveBatchUnit::Frames,
                    AdaptiveBatchUnitArg::PaddedFrames => AdaptiveBatchUnit::PaddedFrames,
                    AdaptiveBatchUnitArg::FeatureValues => AdaptiveBatchUnit::FeatureValues,
                    AdaptiveBatchUnitArg::DurationMs => AdaptiveBatchUnit::DurationMs,
                    AdaptiveBatchUnitArg::PaddedDurationMs => AdaptiveBatchUnit::PaddedDurationMs,
                },
                budget,
                max_samples: args.adaptive_batch_max_samples,
            }))
        }
        _ => {
            anyhow::bail!("--adaptive-batch-unit and --adaptive-batch-budget must be set together")
        }
    }
}

fn resolve_manifest_paths(args: &RunArgs) -> Result<(PathBuf, Option<PathBuf>)> {
    if let Some(manifest_dir) = &args.manifest_dir {
        let train_manifest = args
            .train_manifest
            .clone()
            .unwrap_or_else(|| manifest_dir.join("train.jsonl"));
        let val_manifest = args.val_manifest.clone().or_else(|| {
            ["val.jsonl", "validation.jsonl", "dev.jsonl"]
                .into_iter()
                .map(|name| manifest_dir.join(name))
                .find(|path| path.exists())
        });
        return resolve_auto_validation_split(train_manifest, val_manifest, &args.output_dir);
    }

    let train_manifest = args.train_manifest.clone().ok_or_else(|| {
        anyhow::anyhow!("--train-manifest is required unless --manifest-dir is provided")
    })?;
    resolve_auto_validation_split(train_manifest, args.val_manifest.clone(), &args.output_dir)
}

fn resolve_auto_validation_split(
    train_manifest: PathBuf,
    val_manifest: Option<PathBuf>,
    output_dir: &Path,
) -> Result<(PathBuf, Option<PathBuf>)> {
    if val_manifest.is_some() {
        return Ok((train_manifest, val_manifest));
    }
    let split_dir = output_dir.join("auto-validation-split");
    let entries = collect_split_manifest_entries(&train_manifest)?;
    let (train_entries, val_entries) = split_validation_entries(entries);
    if val_entries.is_empty() {
        log::warn!(
            "training input {} has fewer than two splittable records; validation split disabled",
            train_manifest.display()
        );
        return Ok((train_manifest, None));
    }
    fs::create_dir_all(&split_dir)
        .with_context(|| format!("failed to create {}", split_dir.display()))?;
    let split_train = split_dir.join("train.jsonl");
    let split_val = split_dir.join("validation.jsonl");
    fs::write(&split_train, format!("{}\n", train_entries.join("\n")))
        .with_context(|| format!("failed to write {}", split_train.display()))?;
    fs::write(&split_val, format!("{}\n", val_entries.join("\n")))
        .with_context(|| format!("failed to write {}", split_val.display()))?;
    log::info!(
        "created automatic validation split train_records={} val_records={} dir={}",
        train_entries.len(),
        val_entries.len(),
        split_dir.display()
    );
    Ok((split_train, Some(split_val)))
}

fn split_validation_entries(entries: Vec<String>) -> (Vec<String>, Vec<String>) {
    let len = entries.len();
    if len < 2 {
        return (entries, Vec::new());
    }
    let val_count = ((len + 9) / 10).max(1).min(len - 1);
    let train_cutoff = len - val_count;
    let mut train_entries = Vec::with_capacity(train_cutoff);
    let mut val_entries = Vec::with_capacity(val_count);
    for (index, entry) in entries.into_iter().enumerate() {
        if index >= train_cutoff {
            val_entries.push(entry);
        } else {
            train_entries.push(entry);
        }
    }
    (train_entries, val_entries)
}

fn collect_split_manifest_entries(path: &Path) -> Result<Vec<String>> {
    if path.is_dir() {
        let mut manifest_files = fs::read_dir(path)
            .with_context(|| format!("failed to read manifest directory {}", path.display()))?
            .map(|entry| {
                entry
                    .map(|entry| entry.path())
                    .with_context(|| format!("failed to read entry in {}", path.display()))
            })
            .collect::<Result<Vec<_>>>()?;
        manifest_files.retain(|path| is_split_manifest_file(path));
        manifest_files.sort();
        if manifest_files.is_empty() {
            let mut audio_files = Vec::new();
            collect_split_audio_files(path, &mut audio_files)?;
            audio_files.sort();
            return audio_files
                .iter()
                .map(|path| audio_split_entry(path))
                .collect();
        }

        let mut entries = Vec::new();
        for file in manifest_files {
            entries.extend(collect_split_manifest_entries(&file)?);
        }
        return Ok(entries);
    }

    if is_split_audio_file(path) {
        return Ok(vec![audio_split_entry(path)?]);
    }
    if is_split_parquet_file(path) {
        return parquet_split_entries(path);
    }
    line_split_entries(path)
}

fn is_split_manifest_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .is_some_and(|extension| {
                matches!(extension.as_str(), "jsonl" | "json" | "tsv" | "parquet")
            })
}

fn is_split_parquet_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("parquet"))
}

fn is_split_audio_file(path: &Path) -> bool {
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

fn collect_split_audio_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)
        .with_context(|| format!("failed to read audio directory {}", dir.display()))?
    {
        let path = entry
            .with_context(|| format!("failed to read entry in {}", dir.display()))?
            .path();
        if path.is_dir() {
            collect_split_audio_files(&path, files)?;
        } else if is_split_audio_file(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn parquet_split_entries(path: &Path) -> Result<Vec<String>> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open parquet file {}", path.display()))?;
    let df = ParquetReader::new(file)
        .finish()
        .with_context(|| format!("failed to read parquet file {}", path.display()))?;
    let path = absolute_path(path)?;
    (0..df.height())
        .map(|row| {
            Ok(serde_json::to_string(&json!({
                "parquet_path": path,
                "parquet_row": row,
            }))?)
        })
        .collect()
}

fn line_split_entries(path: &Path) -> Result<Vec<String>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read manifest {}", path.display()))?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    text.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                None
            } else {
                Some(normalize_split_manifest_line(trimmed, base_dir))
            }
        })
        .collect()
}

fn normalize_split_manifest_line(line: &str, base_dir: &Path) -> Result<String> {
    if line.starts_with('{') {
        let mut value =
            serde_json::from_str::<Value>(line).context("invalid JSON manifest line")?;
        absolutize_json_path_field(&mut value, base_dir, "features_path")?;
        absolutize_json_path_field(&mut value, base_dir, "feature_path")?;
        absolutize_json_path_field(&mut value, base_dir, "path")?;
        absolutize_json_path_field(&mut value, base_dir, "audio_path")?;
        absolutize_json_path_field(&mut value, base_dir, "audio")?;
        return serde_json::to_string(&value).context("failed to serialize split manifest line");
    }

    let mut parts = line.split('\t').map(str::to_string).collect::<Vec<_>>();
    if let Some(path) = parts.first_mut() {
        *path = absolute_child_path(base_dir, path)?
            .to_string_lossy()
            .into_owned();
    }
    Ok(parts.join("\t"))
}

fn absolutize_json_path_field(value: &mut Value, base_dir: &Path, key: &str) -> Result<()> {
    let Some(path) = value.get(key).and_then(Value::as_str).map(str::to_string) else {
        return Ok(());
    };
    let absolute = absolute_child_path(base_dir, &path)?;
    value[key] = Value::String(absolute.to_string_lossy().into_owned());
    Ok(())
}

fn audio_split_entry(path: &Path) -> Result<String> {
    let absolute = absolute_path(path)?;
    let mut value = json!({
        "id": raw_audio_split_id(path),
        "audio_path": absolute,
    });
    if let Some(text) = read_first_existing_sidecar(path, &["txt", "lab", "transcript"])? {
        value["text"] = Value::String(text.trim().to_string());
    }
    if let Some(tokens) = read_first_existing_sidecar(path, &["tokens", "tok"])? {
        value["tokens"] = Value::Array(
            tokens
                .split(|ch: char| ch.is_whitespace() || ch == ',')
                .filter(|token| !token.is_empty())
                .map(|token| {
                    token
                        .parse::<i64>()
                        .map(|value| Value::Number(value.into()))
                        .with_context(|| {
                            format!("invalid token id '{token}' for {}", path.display())
                        })
                })
                .collect::<Result<Vec<_>>>()?,
        );
    }
    serde_json::to_string(&value).context("failed to serialize audio split entry")
}

fn read_first_existing_sidecar(path: &Path, extensions: &[&str]) -> Result<Option<String>> {
    for extension in extensions {
        let sidecar = path.with_extension(extension);
        if sidecar.exists() {
            return fs::read_to_string(&sidecar)
                .with_context(|| format!("failed to read {}", sidecar.display()))
                .map(Some);
        }
    }
    Ok(None)
}

fn raw_audio_split_id(path: &Path) -> String {
    path.with_extension("").to_string_lossy().replace('\\', "/")
}

fn absolute_child_path(base_dir: &Path, path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        absolute_path(&base_dir.join(path))
    }
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .context("failed to read current directory")
            .map(|cwd| cwd.join(path))
    }
}

fn run_tokenizer_training(args: TokenizerArgs) -> Result<()> {
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create {}", args.output_dir.display()))?;

    let corpus_path = args.corpus_output.clone().unwrap_or_else(|| {
        args.output_dir
            .join(format!("{}.corpus.txt", args.model_prefix))
    });
    let sentence_count = write_tokenizer_corpus(&args.inputs, &corpus_path)?;
    if sentence_count == 0 {
        bail!("tokenizer inputs did not contain any transcript text");
    }

    let model_prefix = args.output_dir.join(&args.model_prefix);
    let mut spm_args = vec![
        format!("--input={}", corpus_path.display()),
        format!("--model_prefix={}", model_prefix.display()),
        format!("--vocab_size={}", args.vocab_size),
        format!("--model_type={}", sentencepiece_model_type(args.model_type)),
        format!("--character_coverage={}", args.character_coverage),
        format!("--max_sentence_length={}", args.max_sentence_length),
        format!("--unk_piece={}", args.unk_piece),
        format!("--bos_piece={}", args.bos_piece),
        format!("--eos_piece={}", args.eos_piece),
        format!("--pad_piece={}", args.pad_piece),
        format!("--unk_id={}", args.unk_id),
        format!("--bos_id={}", args.bos_id),
        format!("--eos_id={}", args.eos_id),
        format!("--pad_id={}", args.pad_id),
        format!("--normalization_rule_name={}", args.normalization_rule_name),
    ];
    if args.input_sentence_size > 0 {
        spm_args.push(format!(
            "--input_sentence_size={}",
            args.input_sentence_size
        ));
    }
    if args.shuffle_input_sentence {
        spm_args.push("--shuffle_input_sentence=true".to_string());
    }
    if args.train_extremely_large_corpus {
        spm_args.push("--train_extremely_large_corpus=true".to_string());
    }
    if args.byte_fallback {
        spm_args.push("--byte_fallback=true".to_string());
    }
    if !args.user_defined_symbols.is_empty() {
        spm_args.push(format!(
            "--user_defined_symbols={}",
            args.user_defined_symbols.join(",")
        ));
    }
    if !args.control_symbols.is_empty() {
        spm_args.push(format!(
            "--control_symbols={}",
            args.control_symbols.join(",")
        ));
    }

    run_sentencepiece(&args, &spm_args)?;

    let model_path = model_prefix.with_extension("model");
    let vocab_path = model_prefix.with_extension("vocab");
    rust_asr::tokenizer::load_sentencepiece_transcript_tokenizer(&model_path).with_context(
        || {
            format!(
                "SentencePiece model was created but could not be loaded by the Rust tokenizer: {}",
                model_path.display()
            )
        },
    )?;

    if !args.keep_corpus && args.corpus_output.is_none() {
        let _ = fs::remove_file(&corpus_path);
    }

    println!(
        "tokenizer complete sentences={} model={} vocab={}",
        sentence_count,
        model_path.display(),
        vocab_path.display()
    );
    Ok(())
}

fn run_sentencepiece(args: &TokenizerArgs, spm_args: &[String]) -> Result<()> {
    let mut candidates = Vec::new();
    if let Some(command) = &args.sentencepiece_command {
        candidates.push(command.clone());
    } else if let Ok(command) = std::env::var("SENTENCEPIECE_TRAIN") {
        candidates.push(PathBuf::from(command));
    } else {
        candidates.push(PathBuf::from("spm_train"));
        candidates.push(PathBuf::from("sentencepiece-train"));
    }

    let mut not_found = Vec::new();
    for command in candidates {
        match Command::new(&command).args(spm_args).status() {
            Ok(status) if status.success() => return Ok(()),
            Ok(status) => bail!(
                "SentencePiece trainer {} exited with status {}",
                command.display(),
                status
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                not_found.push(command.display().to_string());
            }
            Err(error) => {
                return Err(error).with_context(|| format!("failed to run {}", command.display()));
            }
        }
    }

    bail!(
        "could not find SentencePiece trainer executable. Tried: {}. Install sentencepiece so spm_train is on PATH, or pass --sentencepiece-command",
        not_found.join(", ")
    )
}

fn sentencepiece_model_type(model_type: SentencePieceModelTypeArg) -> &'static str {
    match model_type {
        SentencePieceModelTypeArg::Unigram => "unigram",
        SentencePieceModelTypeArg::Bpe => "bpe",
        SentencePieceModelTypeArg::Char => "char",
        SentencePieceModelTypeArg::Word => "word",
    }
}

fn run_feature_extraction(args: ExtractFeaturesArgs) -> Result<()> {
    if let Some(jobs) = args.jobs {
        if jobs == 0 {
            bail!("--jobs must be greater than 0");
        }
        return rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()
            .context("failed to build Rayon thread pool")?
            .install(|| run_feature_extraction_inner(args));
    }
    run_feature_extraction_inner(args)
}

fn run_feature_extraction_inner(args: ExtractFeaturesArgs) -> Result<()> {
    let max_audio_duration_ms = duration_sec_to_ms(args.max_audio_duration_sec)?;
    let architecture = resolve_extract_architecture(&args);
    let frontend = feature_extractor_for_train_architecture(architecture);
    if args.inputs.iter().any(|input| input.is_dir()) {
        return run_feature_extraction_to_directory(args, &frontend, max_audio_duration_ms);
    }

    let mut records = Vec::new();
    let mut tui = if args.tui {
        Some(ExtractionTui::new(&args.output)?)
    } else {
        None
    };
    for input in &args.inputs {
        let remaining = args
            .max_samples
            .map(|limit| limit.saturating_sub(records.len()));
        if remaining == Some(0) {
            break;
        }
        if let Some(tui) = tui.as_mut() {
            let extracted = extract_feature_records_with_progress(
                input,
                remaining,
                args.tokenizer.as_deref(),
                &frontend,
                max_audio_duration_ms,
                |progress| tui.update(progress),
            )?;
            records.extend(extracted);
        } else {
            records.extend(extract_feature_records(
                input,
                remaining,
                args.tokenizer.as_deref(),
                &frontend,
                max_audio_duration_ms,
            )?);
        }
    }
    if records.is_empty() {
        bail!("no records were extracted");
    }
    write_feature_records_parquet(&records, &args.output)?;
    if let Some(tui) = tui.as_mut() {
        tui.finish(records.len())?;
    }
    drop(tui);
    println!(
        "wrote {} feature records to {}",
        records.len(),
        args.output.display()
    );
    Ok(())
}

fn run_feature_extraction_to_directory(
    args: ExtractFeaturesArgs,
    frontend: &FeatureExtractorConfig,
    max_audio_duration_ms: Option<usize>,
) -> Result<()> {
    fs::create_dir_all(&args.output).with_context(|| {
        format!(
            "failed to create output directory {}",
            args.output.display()
        )
    })?;
    let mut total_records = 0usize;
    let mut written = 0usize;
    let mut tui = if args.tui {
        Some(ExtractionTui::new(&args.output)?)
    } else {
        None
    };

    for (input_index, input) in args.inputs.iter().enumerate() {
        let remaining = args
            .max_samples
            .map(|limit| limit.saturating_sub(total_records));
        if remaining == Some(0) {
            break;
        }
        let records = if let Some(tui) = tui.as_mut() {
            extract_feature_records_with_progress(
                input,
                remaining,
                args.tokenizer.as_deref(),
                frontend,
                max_audio_duration_ms,
                |progress| tui.update(progress),
            )?
        } else {
            extract_feature_records(
                input,
                remaining,
                args.tokenizer.as_deref(),
                frontend,
                max_audio_duration_ms,
            )?
        };
        if records.is_empty() {
            continue;
        }
        total_records += records.len();
        let output = directory_feature_output_path(&args.output, input, input_index);
        write_feature_records_parquet(&records, &output)?;
        written += 1;
        println!(
            "wrote {} feature records to {}",
            records.len(),
            output.display()
        );
    }

    if total_records == 0 {
        bail!("no records were extracted");
    }
    if let Some(tui) = tui.as_mut() {
        tui.finish(total_records)?;
    }
    drop(tui);
    println!(
        "wrote {} feature records across {} parquet files in {}",
        total_records,
        written,
        args.output.display()
    );
    Ok(())
}

fn directory_feature_output_path(output_dir: &Path, input: &Path, input_index: usize) -> PathBuf {
    let stem = input
        .file_stem()
        .or_else(|| input.file_name())
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("features");
    output_dir.join(format!("{input_index:04}-{stem}.parquet"))
}

struct ExtractionTui {
    started: std::time::Instant,
    stdout: std::io::Stdout,
    output: PathBuf,
    records: usize,
    skipped_duration: usize,
    input: Option<PathBuf>,
    last_id: Option<String>,
    last_rows: Option<usize>,
    last_duration_ms: Option<usize>,
    done: bool,
}

impl ExtractionTui {
    fn new(output: &Path) -> Result<Self> {
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, Hide, Clear(ClearType::All))
            .context("failed to initialize extraction TUI")?;
        let mut tui = Self {
            started: std::time::Instant::now(),
            stdout,
            output: output.to_path_buf(),
            records: 0,
            skipped_duration: 0,
            input: None,
            last_id: None,
            last_rows: None,
            last_duration_ms: None,
            done: false,
        };
        tui.draw()?;
        Ok(tui)
    }

    fn update(&mut self, progress: &FeatureExtractionProgress) -> Result<()> {
        self.input = Some(progress.input.clone());
        self.records = progress.records;
        self.skipped_duration = progress.skipped_duration;
        self.last_id = progress.last_id.clone();
        self.last_rows = progress.last_rows;
        self.last_duration_ms = progress.last_duration_ms;
        self.draw()
    }

    fn finish(&mut self, records: usize) -> Result<()> {
        self.records = records;
        self.done = true;
        self.draw()
    }

    fn draw(&mut self) -> Result<()> {
        queue!(
            self.stdout,
            MoveTo(0, 0),
            Clear(ClearType::All),
            SetForegroundColor(Color::Cyan),
            SetAttribute(Attribute::Bold),
            Print("rust-asr feature extraction monitor\n"),
            ResetColor,
            SetAttribute(Attribute::Reset),
            Print(format!("output: {}\n", self.output.display())),
            Print(format!(
                "elapsed: {:.1}s   status: {}\n\n",
                self.started.elapsed().as_secs_f64(),
                if self.done { "complete" } else { "extracting" }
            )),
            SetForegroundColor(Color::Yellow),
            Print("Progress\n"),
            ResetColor,
            Print(format!(
                "  input: {}\n",
                self.input
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "-".to_string())
            )),
            Print(format!(
                "  records: {}   skipped_duration: {}\n",
                self.records, self.skipped_duration
            )),
            Print(format!(
                "  last_id: {}   rows: {}   duration_ms: {}\n",
                self.last_id.as_deref().unwrap_or("-"),
                fmt_optional_usize(self.last_rows),
                fmt_optional_usize(self.last_duration_ms)
            )),
            SetForegroundColor(Color::DarkGrey),
            Print("\nPress Ctrl-C to stop.\n"),
            ResetColor
        )
        .context("failed to draw extraction TUI")?;
        self.stdout
            .flush()
            .context("failed to flush extraction TUI")
    }
}

impl Drop for ExtractionTui {
    fn drop(&mut self) {
        let _ = execute!(self.stdout, Show, LeaveAlternateScreen);
    }
}

fn fmt_optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn write_feature_records_parquet(
    records: &[rust_asr::train::FeatureRecord],
    output: &Path,
) -> Result<()> {
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let ids = records
        .iter()
        .map(|record| record.id.as_str())
        .collect::<Vec<_>>();
    let texts = records
        .iter()
        .map(|record| record.text.as_deref())
        .collect::<Vec<_>>();
    let rows = records
        .iter()
        .map(|record| record.rows as u32)
        .collect::<Vec<_>>();
    let cols = records
        .iter()
        .map(|record| record.cols as u32)
        .collect::<Vec<_>>();
    let duration_ms = records
        .iter()
        .map(|record| record.duration_ms as u64)
        .collect::<Vec<_>>();
    let feature_items = records
        .iter()
        .map(|record| Series::new("".into(), record.features.as_slice()))
        .collect::<Vec<_>>();
    let token_items = records
        .iter()
        .map(|record| Series::new("".into(), record.tokens.as_slice()))
        .collect::<Vec<_>>();
    let mut features = feature_items.iter().collect::<ListChunked>().into_series();
    features.rename("features".into());
    let mut tokens = token_items.iter().collect::<ListChunked>().into_series();
    tokens.rename("tokens".into());

    let mut df = DataFrame::new(
        records.len(),
        vec![
            Series::new("id".into(), ids).into_column(),
            Series::new("text".into(), texts).into_column(),
            Series::new("rows".into(), rows).into_column(),
            Series::new("cols".into(), cols).into_column(),
            Series::new("duration_ms".into(), duration_ms).into_column(),
            features.into_column(),
            tokens.into_column(),
        ],
    )?;
    let mut file = fs::File::create(output)
        .with_context(|| format!("failed to create {}", output.display()))?;
    ParquetWriter::new(&mut file)
        .finish(&mut df)
        .with_context(|| format!("failed to write parquet {}", output.display()))?;
    Ok(())
}

fn write_tokenizer_corpus(inputs: &[PathBuf], corpus_path: &Path) -> Result<usize> {
    if let Some(parent) = corpus_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let mut writer = fs::File::create(corpus_path)
        .with_context(|| format!("failed to create corpus {}", corpus_path.display()))?;
    let mut count = 0usize;
    for input in inputs {
        count += write_tokenizer_input(input, &mut writer)?;
    }
    Ok(count)
}

fn write_tokenizer_input(path: &Path, writer: &mut fs::File) -> Result<usize> {
    if path.is_dir() {
        let mut entries = fs::read_dir(path)
            .with_context(|| format!("failed to read directory {}", path.display()))?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()
            .with_context(|| format!("failed to list directory {}", path.display()))?;
        entries.sort();
        let mut count = 0usize;
        for entry in entries {
            if entry.is_dir() || is_tokenizer_input_file(&entry) {
                count += write_tokenizer_input(&entry, writer)?;
            }
        }
        return Ok(count);
    }

    if !path.exists() {
        bail!("tokenizer input does not exist: {}", path.display());
    }

    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .as_deref()
    {
        Some("parquet") => write_parquet_text(path, writer),
        Some("json") | Some("jsonl") => write_jsonl_text(path, writer),
        Some("tsv") => write_tsv_text(path, writer),
        Some("txt") | Some("text") | Some("transcript") | Some("lab") => {
            write_plain_text(path, writer)
        }
        Some("wav") | Some("flac") | Some("mp3") | Some("ogg") | Some("m4a") | Some("aac") => {
            write_audio_sidecar_text(path, writer)
        }
        _ => write_plain_text(path, writer),
    }
}

fn is_tokenizer_input_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref(),
        Some("json")
            | Some("jsonl")
            | Some("parquet")
            | Some("tsv")
            | Some("txt")
            | Some("text")
            | Some("transcript")
            | Some("lab")
            | Some("wav")
            | Some("flac")
            | Some("mp3")
            | Some("ogg")
            | Some("m4a")
            | Some("aac")
    )
}

fn write_jsonl_text(path: &Path, writer: &mut fs::File) -> Result<usize> {
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut count = 0usize;
    for (index, raw_line) in reader.lines().enumerate() {
        let line = raw_line.with_context(|| format!("failed reading {}", path.display()))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !line.starts_with('{') {
            if let Some(text) = line.split('\t').nth(4) {
                count += write_corpus_line(writer, text)?;
            }
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("invalid JSON line {} in {}", index + 1, path.display()))?;
        if let Some(text) = json_text_field(&value) {
            count += write_corpus_line(writer, text)?;
        }
    }
    Ok(count)
}

fn write_tsv_text(path: &Path, writer: &mut fs::File) -> Result<usize> {
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut count = 0usize;
    for raw_line in reader.lines() {
        let line = raw_line.with_context(|| format!("failed reading {}", path.display()))?;
        if let Some(text) = line.split('\t').nth(4) {
            count += write_corpus_line(writer, text)?;
        }
    }
    Ok(count)
}

fn write_plain_text(path: &Path, writer: &mut fs::File) -> Result<usize> {
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut count = 0usize;
    for raw_line in reader.lines() {
        let line = raw_line.with_context(|| format!("failed reading {}", path.display()))?;
        count += write_corpus_line(writer, &line)?;
    }
    Ok(count)
}

fn write_audio_sidecar_text(path: &Path, writer: &mut fs::File) -> Result<usize> {
    for extension in ["txt", "lab", "transcript"] {
        let sidecar = path.with_extension(extension);
        if sidecar.exists() {
            return write_plain_text(&sidecar, writer);
        }
    }
    Ok(0)
}

fn write_parquet_text(path: &Path, writer: &mut fs::File) -> Result<usize> {
    let df = read_parquet_dataframe(path)?;
    let mut count = 0usize;
    for row in 0..df.height() {
        if let Some(text) = parquet_optional_string(
            &df,
            row,
            &[
                "text",
                "transcript",
                "transcription",
                "sentence",
                "normalized_text",
            ],
        )? {
            count += write_corpus_line(writer, &text)?;
        }
    }
    Ok(count)
}

fn read_parquet_dataframe(path: &Path) -> Result<DataFrame> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open parquet file {}", path.display()))?;
    ParquetReader::new(file)
        .finish()
        .with_context(|| format!("failed to read parquet file {}", path.display()))
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

fn json_text_field(value: &Value) -> Option<&str> {
    value
        .get("text")
        .or_else(|| value.get("transcript"))
        .or_else(|| value.get("transcription"))
        .or_else(|| value.get("sentence"))
        .or_else(|| value.get("normalized_text"))
        .and_then(Value::as_str)
}

fn write_corpus_line(writer: &mut fs::File, text: &str) -> Result<usize> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(0);
    }
    writeln!(writer, "{text}").context("failed writing tokenizer corpus")?;
    Ok(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_validation_split_uses_about_ten_percent() {
        let entries = (0..10).map(|index| format!("record-{index}")).collect();
        let (train, val) = split_validation_entries(entries);

        assert_eq!(train.len(), 9);
        assert_eq!(val.len(), 1);
        assert_eq!(val[0], "record-9");
    }

    #[test]
    fn variant_peak_learning_rates_match_positiveloss_recipe() {
        assert_eq!(variant_peak_learning_rate(Some("xs")), 2.0e-3);
        assert_eq!(variant_peak_learning_rate(Some("s")), 2.0e-3);
        assert_eq!(variant_peak_learning_rate(Some("sm")), 2.0e-3);
        assert_eq!(variant_peak_learning_rate(Some("m")), 1.5e-3);
        assert_eq!(variant_peak_learning_rate(Some("ml")), 1.0e-3);
        assert_eq!(variant_peak_learning_rate(Some("l")), 1.0e-3);
    }

    #[test]
    fn architecture_default_learning_rates_follow_train_recipes() {
        for variant in ["xs", "s", "sm", "m", "ml", "l"] {
            let defaults = training_defaults(TrainArchitecture::Squeezeformer, Some(variant));
            assert_eq!(defaults.learning_rate, 3.0e-4);
        }
        assert_eq!(
            training_defaults(TrainArchitecture::Zipformer, Some("sm")).learning_rate,
            1.0e-3
        );
        assert_eq!(
            training_defaults(TrainArchitecture::Paraformer, Some("sm")).learning_rate,
            1.0e-3
        );
        assert_eq!(
            training_defaults(TrainArchitecture::Wav2VecBert, Some("sm")).learning_rate,
            1.0e-4
        );
    }

    #[test]
    fn split_manifest_json_paths_are_absolutized() {
        let base_dir = std::env::current_dir().unwrap().join("data");
        let line = r#"{"id":"a","audio_path":"audio/a.wav","tokens":[1]}"#;
        let normalized = normalize_split_manifest_line(line, &base_dir).unwrap();
        let value: Value = serde_json::from_str(&normalized).unwrap();

        assert_eq!(
            value.get("audio_path").and_then(Value::as_str),
            Some(base_dir.join("audio/a.wav").to_str().unwrap())
        );
    }
}
