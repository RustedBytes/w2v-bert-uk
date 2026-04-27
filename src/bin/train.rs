use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use env_logger::Env;
use w2v_bert_uk::paraformer::ParaformerAlignmentMode;
use w2v_bert_uk::train::{
    AdaptiveBatchConfig, AdaptiveBatchUnit, BurnTrainConfig, TrainArchitecture, TrainBackendKind,
    TrainPrecision, run_burn_training,
};

#[derive(Parser)]
#[command(
    author,
    version,
    about = "Train Burn ASR architectures with a Python-train.py-style CLI"
)]
struct Args {
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

    /// Optional validation manifest, JSONL or TSV.
    #[arg(long)]
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
    vocab_size: usize,

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

    /// Adaptive batch unit: samples, frames, padded-frames, or feature-values.
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

    /// Number of epochs.
    #[arg(long, default_value_t = 10)]
    epochs: usize,

    /// AdamW learning rate.
    #[arg(long, default_value_t = 1.0e-3)]
    learning_rate: f64,

    /// Linear warmup optimizer steps before reaching --learning-rate.
    #[arg(long, default_value_t = 0)]
    lr_warmup_steps: usize,

    /// Optimizer steps to hold --learning-rate after warmup.
    #[arg(long, default_value_t = 0)]
    lr_hold_steps: usize,

    /// Linear decay optimizer steps after warmup/hold.
    #[arg(long, default_value_t = 0)]
    lr_decay_steps: usize,

    /// Final learning rate after decay.
    #[arg(long, default_value_t = 0.0)]
    lr_min: f64,

    /// AdamW weight decay.
    #[arg(long, default_value_t = 1.0e-2)]
    weight_decay: f64,

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

    /// Resume from a checkpoint directory or checkpoint.json file.
    #[arg(long)]
    resume_from: Option<PathBuf>,

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
enum AdaptiveBatchUnitArg {
    Samples,
    Frames,
    PaddedFrames,
    FeatureValues,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ParaformerAlignmentArg {
    Viterbi,
    Uniform,
    Greedy,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    let (train_manifest, val_manifest) = resolve_manifest_paths(&args)?;
    let adaptive_batch = resolve_adaptive_batch(&args)?;
    let precision = resolve_precision(&args);
    let device_indices = resolve_device_indices(&args);
    let config = BurnTrainConfig {
        architecture: resolve_architecture(&args),
        train_manifest,
        val_manifest,
        output_dir: args.output_dir,
        variant: args.variant,
        input_dim: args.input_dim,
        vocab_size: args.vocab_size,
        blank_id: args.blank_id,
        d_model: args.d_model,
        num_layers: args.num_layers,
        num_heads: args.num_heads,
        batch_size: args.batch_size,
        adaptive_batch,
        sort_by_length_desc: args.sort_by_length_desc,
        sort_buffer_size: args.sort_buffer_size,
        epochs: args.epochs,
        learning_rate: args.learning_rate,
        lr_warmup_steps: args.lr_warmup_steps,
        lr_hold_steps: args.lr_hold_steps,
        lr_decay_steps: args.lr_decay_steps,
        lr_min: args.lr_min,
        weight_decay: args.weight_decay,
        gradient_accumulation_steps: args.gradient_accumulation_steps,
        gradient_clip_norm: args.gradient_clip_norm,
        gradient_clip_value: args.gradient_clip_value,
        ema_decay: args.ema_decay,
        ema_start_step: args.ema_start_step,
        log_every: args.log_every,
        validate_every_steps: args.validate_every_steps,
        max_train_samples: args.max_train_samples,
        max_val_samples: args.max_val_samples,
        tokenizer_path: args.tokenizer,
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
        resume_from: args.resume_from,
        backend: match args.backend {
            BackendArg::Cpu => TrainBackendKind::Cpu,
            BackendArg::Cuda => TrainBackendKind::Cuda,
            BackendArg::Wgpu => TrainBackendKind::Wgpu,
        },
        device_index: args.device_index,
        device_indices,
        precision,
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

fn resolve_device_indices(args: &Args) -> Vec<usize> {
    if args.device_indices.is_empty() {
        vec![args.device_index]
    } else {
        args.device_indices.clone()
    }
}

fn resolve_precision(args: &Args) -> TrainPrecision {
    if args.mixed_precision && matches!(args.precision, PrecisionArg::F32) {
        return TrainPrecision::F16;
    }
    match args.precision {
        PrecisionArg::F32 => TrainPrecision::F32,
        PrecisionArg::F16 => TrainPrecision::F16,
        PrecisionArg::Bf16 => TrainPrecision::Bf16,
    }
}

fn resolve_architecture(args: &Args) -> TrainArchitecture {
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

fn resolve_adaptive_batch(args: &Args) -> Result<Option<AdaptiveBatchConfig>> {
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

fn resolve_manifest_paths(args: &Args) -> Result<(PathBuf, Option<PathBuf>)> {
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
        return Ok((train_manifest, val_manifest));
    }

    let train_manifest = args.train_manifest.clone().ok_or_else(|| {
        anyhow::anyhow!("--train-manifest is required unless --manifest-dir is provided")
    })?;
    Ok((train_manifest, args.val_manifest.clone()))
}
