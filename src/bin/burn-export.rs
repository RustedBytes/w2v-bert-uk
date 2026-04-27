use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use env_logger::Env;
use w2v_bert_uk::train::{BurnExportConfig, run_burn_export};

#[derive(Parser)]
#[command(
    author,
    version,
    about = "Package a Burn checkpoint for inference/export across supported ASR architectures"
)]
struct Args {
    /// Checkpoint directory or checkpoint.json path.
    #[arg(long, default_value = "runs/burn/checkpoint_latest")]
    checkpoint: PathBuf,

    /// Output directory for exported Burn model and metadata.
    #[arg(long)]
    output_dir: PathBuf,

    /// Export ema_model.bin instead of model.bin.
    #[arg(long)]
    ema: bool,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    let summary = run_burn_export(BurnExportConfig {
        checkpoint: args.checkpoint,
        output_dir: args.output_dir,
        use_ema: args.ema,
    })?;
    println!(
        "burn export complete architecture={:?} model={} metadata={}",
        summary.architecture,
        summary.model_path.display(),
        summary.metadata_path.display()
    );
    Ok(())
}
