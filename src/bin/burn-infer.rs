use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use env_logger::Env;
use rust_asr::train::{BurnInferenceConfig, run_burn_inference};

#[derive(Parser)]
#[command(
    author,
    version,
    about = "Run Burn checkpoint inference for Squeezeformer, Zipformer, Paraformer, or W2V-BERT"
)]
struct Args {
    /// Checkpoint directory or checkpoint.json path.
    #[arg(long, default_value = "runs/burn/checkpoint_latest")]
    checkpoint: PathBuf,

    /// JSONL/TSV feature manifest or directory of JSONL manifests.
    #[arg(long)]
    manifest: PathBuf,

    /// Optional JSONL output path. Prints JSONL to stdout when unset.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Load ema_model.bin instead of model.bin.
    #[arg(long)]
    ema: bool,

    /// Override checkpoint batch size.
    #[arg(long)]
    batch_size: Option<usize>,

    /// Limit number of samples.
    #[arg(long)]
    max_samples: Option<usize>,

    /// Optional SentencePiece tokenizer for text decoding.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// CTC beam width. Defaults to the checkpoint training config.
    #[arg(long)]
    beam_width: Option<usize>,

    /// Number of CTC hypotheses before optional LM reranking.
    #[arg(long)]
    n_best: Option<usize>,

    /// Optional KenLM model for CTC reranking.
    #[arg(long)]
    lm_path: Option<PathBuf>,

    /// KenLM shallow-fusion weight.
    #[arg(long)]
    lm_weight: Option<f32>,

    /// Word insertion bonus for KenLM reranking.
    #[arg(long)]
    lm_word_bonus: Option<f32>,

    /// Score LM candidates without beginning-of-sentence context.
    #[arg(long)]
    lm_no_bos: bool,

    /// Score LM candidates without end-of-sentence context.
    #[arg(long)]
    lm_no_eos: bool,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    let summary = run_burn_inference(BurnInferenceConfig {
        checkpoint: args.checkpoint,
        manifest: args.manifest,
        output: args.output,
        use_ema: args.ema,
        batch_size: args.batch_size,
        max_samples: args.max_samples,
        tokenizer_path: args.tokenizer,
        beam_width: args.beam_width,
        n_best: args.n_best,
        lm_path: args.lm_path,
        lm_weight: args.lm_weight,
        lm_word_bonus: args.lm_word_bonus,
        lm_bos: !args.lm_no_bos,
        lm_eos: !args.lm_no_eos,
    })?;
    eprintln!(
        "burn inference complete architecture={:?} decoded_samples={} output={:?}",
        summary.architecture, summary.decoded_samples, summary.output
    );
    Ok(())
}
