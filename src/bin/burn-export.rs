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

    /// Optional Hugging Face Hub repo id to upload the export package to.
    #[arg(long)]
    hf_repo_id: Option<String>,

    /// Optional Hugging Face revision/branch for upload.
    #[arg(long)]
    hf_revision: Option<String>,

    /// Create the Hugging Face model repo as private when it does not exist.
    #[arg(long)]
    hf_private: bool,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    let summary = run_burn_export(BurnExportConfig {
        checkpoint: args.checkpoint,
        output_dir: args.output_dir,
        use_ema: args.ema,
        hf_repo_id: args.hf_repo_id,
        hf_revision: args.hf_revision,
        hf_private: args.hf_private,
    })?;
    println!(
        "burn export complete architecture={:?} model={} metadata={} readme={}",
        summary.architecture,
        summary.model_path.display(),
        summary.metadata_path.display(),
        summary.readme_path.display()
    );
    if let Some(upload) = summary.hf_upload {
        println!(
            "huggingface upload complete repo_id={} revision={:?}",
            upload.repo_id, upload.revision
        );
    }
    Ok(())
}
