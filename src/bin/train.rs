use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use env_logger::Env;
use w2v_bert_uk::paraformer::ParaformerAlignmentMode;
use w2v_bert_uk::train::{
    AdaptiveBatchConfig, AdaptiveBatchUnit, BurnTrainConfig, TrainArchitecture, run_burn_training,
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

    /// AdamW weight decay.
    #[arg(long, default_value_t = 1.0e-2)]
    weight_decay: f64,

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

    /// Run forward/loss only and skip optimizer updates.
    #[arg(long)]
    dry_run: bool,

    /// Paraformer alignment strategy for decoder-query construction.
    #[arg(long, value_enum, default_value_t = ParaformerAlignmentArg::Viterbi)]
    paraformer_alignment_mode: ParaformerAlignmentArg,
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
        weight_decay: args.weight_decay,
        log_every: args.log_every,
        validate_every_steps: args.validate_every_steps,
        max_train_samples: args.max_train_samples,
        max_val_samples: args.max_val_samples,
        dry_run: args.dry_run,
        paraformer_alignment_mode: match args.paraformer_alignment_mode {
            ParaformerAlignmentArg::Viterbi => ParaformerAlignmentMode::Viterbi,
            ParaformerAlignmentArg::Uniform => ParaformerAlignmentMode::Uniform,
            ParaformerAlignmentArg::Greedy => ParaformerAlignmentMode::Greedy,
        },
    };

    let summary = run_burn_training(config)?;
    println!(
        "training complete epochs={} steps={} last_train_loss={:?} last_val_loss={:?}",
        summary.epochs, summary.steps, summary.last_train_loss, summary.last_val_loss
    );
    Ok(())
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
