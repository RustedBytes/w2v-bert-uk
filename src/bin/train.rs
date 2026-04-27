use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use env_logger::Env;
use polars::prelude::{AnyValue, DataFrame, ParquetReader, SerReader};
use serde_json::Value;
use w2v_bert_uk::audio::WaveformAugmentConfig;
use w2v_bert_uk::paraformer::ParaformerAlignmentMode;
use w2v_bert_uk::train::{
    AdaptiveBatchConfig, AdaptiveBatchUnit, BurnTrainConfig, SpecAugmentConfig, TrainArchitecture,
    TrainBackendKind, TrainPrecision, run_burn_training,
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
    let cli = Cli::parse();
    match cli.command {
        Some(CommandArg::Run(args)) => run_training(args),
        Some(CommandArg::Tokenizer(args)) => run_tokenizer_training(args),
        None => run_training(cli.run),
    }
}

fn run_training(args: RunArgs) -> Result<()> {
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
        vocab_size: args
            .vocab_size
            .ok_or_else(|| anyhow!("--vocab-size is required for model training"))?,
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

fn resolve_device_indices(args: &RunArgs) -> Vec<usize> {
    if args.device_indices.is_empty() {
        vec![args.device_index]
    } else {
        args.device_indices.clone()
    }
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
        return Ok((train_manifest, val_manifest));
    }

    let train_manifest = args.train_manifest.clone().ok_or_else(|| {
        anyhow::anyhow!("--train-manifest is required unless --manifest-dir is provided")
    })?;
    Ok((train_manifest, args.val_manifest.clone()))
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
    w2v_bert_uk::tokenizer::load_sentencepiece_transcript_tokenizer(&model_path).with_context(
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
            &["text", "transcript", "sentence", "normalized_text"],
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
