use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use env_logger::Env;
use log::{info, warn};
use w2v_bert_uk::{
    AcousticModelConfig, CandidateProcessingConfig, CtcDecoderConfig, DecoderConfig, EncoderConfig,
    LmConfig, RuntimeConfig, TextDecoderConfig, TimingReport, TranscriptionConfig,
    TranscriptionResult, W2vBertEncoderConfig,
    audio::AudioDecodeConfig,
    format_duration,
    model::{ModelConfig, ModelOptimizationLevel},
    transcribe_audio_file,
};

#[derive(Parser)]
#[command(
    author,
    version,
    about = "Transcribe audio with a W2V-BERT CTC ONNX model"
)]
struct Args {
    /// Audio file to transcribe.
    audio_file: PathBuf,

    /// ONNX model path.
    #[arg(default_value = "model_optimized.onnx")]
    model: PathBuf,

    /// SentencePiece tokenizer model path.
    #[arg(default_value = "tokenizer.model")]
    tokenizer: PathBuf,

    /// CTC beam width.
    #[arg(default_value_t = 32, value_parser = parse_positive_usize)]
    beam_width: usize,

    /// KenLM binary model path.
    #[arg(default_value = "lm.binary")]
    lm: PathBuf,

    /// KenLM shallow-fusion weight.
    #[arg(default_value_t = 0.45)]
    lm_weight: f32,

    /// Word insertion bonus used during KenLM reranking.
    #[arg(default_value_t = 0.2)]
    word_bonus: f32,

    /// Hot word or phrase to boost during KenLM reranking. Can be repeated.
    #[arg(long = "hot-word")]
    hot_words: Vec<String>,

    /// Score bonus applied for each hot word or phrase match.
    #[arg(long, default_value_t = 0.0)]
    hot_word_bonus: f32,

    /// Optional ONNX Runtime dynamic library path.
    #[arg(long)]
    ort_dylib: Option<PathBuf>,

    /// ONNX Runtime graph optimization level.
    #[arg(long, value_enum, default_value_t = CliOptimizationLevel::Disable)]
    ort_optimization: CliOptimizationLevel,

    /// Disable accelerator logging.
    #[arg(long)]
    no_accelerator_log: bool,

    /// Disable KenLM logging.
    #[arg(long)]
    no_lm_log: bool,

    /// Fallback sample rate when the audio container does not report one.
    #[arg(long, default_value_t = 16_000)]
    fallback_sample_rate: u32,

    /// Fail on packet decode errors instead of skipping corrupt packets.
    #[arg(long)]
    strict_audio_decode: bool,

    /// W2V-BERT frontend model source label.
    #[arg(long)]
    w2v_model_source: Option<String>,

    /// W2V-BERT frontend target sample rate.
    #[arg(long)]
    w2v_sample_rate: Option<u32>,

    /// W2V-BERT frontend base feature size.
    #[arg(long)]
    w2v_feature_size: Option<usize>,

    /// W2V-BERT frontend stacking stride.
    #[arg(long)]
    w2v_stride: Option<usize>,

    /// W2V-BERT frontend final feature dimension.
    #[arg(long)]
    w2v_feature_dim: Option<usize>,

    /// W2V-BERT frontend padding value.
    #[arg(long)]
    w2v_padding_value: Option<f32>,

    /// CTC blank token ID.
    #[arg(long, default_value_t = 0)]
    blank_id: u32,

    /// Number of CTC hypotheses to keep before text/LM reranking.
    #[arg(long)]
    n_best: Option<usize>,

    /// Keep decoded whitespace as produced by the tokenizer.
    #[arg(long)]
    no_normalize_spaces: bool,

    /// Keep empty decoded candidates.
    #[arg(long)]
    keep_empty_candidates: bool,

    /// Score KenLM candidates without beginning-of-sentence context.
    #[arg(long)]
    lm_no_bos: bool,

    /// Score KenLM candidates without end-of-sentence context.
    #[arg(long)]
    lm_no_eos: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliOptimizationLevel {
    Disable,
    Level1,
    Level2,
    Level3,
    All,
}

impl From<CliOptimizationLevel> for ModelOptimizationLevel {
    fn from(value: CliOptimizationLevel) -> Self {
        match value {
            CliOptimizationLevel::Disable => ModelOptimizationLevel::Disable,
            CliOptimizationLevel::Level1 => ModelOptimizationLevel::Level1,
            CliOptimizationLevel::Level2 => ModelOptimizationLevel::Level2,
            CliOptimizationLevel::Level3 => ModelOptimizationLevel::Level3,
            CliOptimizationLevel::All => ModelOptimizationLevel::All,
        }
    }
}

fn parse_positive_usize(value: &str) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| "must be a positive integer".to_string())?;
    if parsed == 0 {
        Err("must be a positive integer".to_string())
    } else {
        Ok(parsed)
    }
}

fn main() -> Result<()> {
    init_logger();
    let total_start = Instant::now();
    let args = Args::parse();
    let candidate_processing = CandidateProcessingConfig {
        normalize_spaces: !args.no_normalize_spaces,
        drop_empty_candidates: !args.keep_empty_candidates,
    };
    let lm_config = if args.lm.exists() {
        Some(LmConfig {
            path: args.lm,
            weight: args.lm_weight,
            word_bonus: args.word_bonus,
            hot_words: args.hot_words,
            hot_word_bonus: args.hot_word_bonus,
            log_language_model: !args.no_lm_log,
            bos: !args.lm_no_bos,
            eos: !args.lm_no_eos,
            candidate_processing: candidate_processing.clone(),
        })
    } else {
        warn!("KenLM disabled: {} does not exist", args.lm.display());
        None
    };

    let result = transcribe_audio_file(
        &args.audio_file,
        &TranscriptionConfig {
            runtime: RuntimeConfig {
                ort_dylib_path: args.ort_dylib,
            },
            audio: AudioDecodeConfig {
                fallback_sample_rate: args.fallback_sample_rate,
                skip_decode_errors: !args.strict_audio_decode,
                ffmpeg_fallback: true,
            },
            encoder: EncoderConfig {
                w2v_bert: W2vBertEncoderConfig {
                    model_source: args.w2v_model_source,
                    sample_rate: args.w2v_sample_rate,
                    feature_size: args.w2v_feature_size,
                    stride: args.w2v_stride,
                    feature_dim: args.w2v_feature_dim,
                    padding_value: args.w2v_padding_value,
                },
            },
            model: AcousticModelConfig {
                path: args.model,
                session: ModelConfig {
                    optimization_level: args.ort_optimization.into(),
                    log_accelerator: !args.no_accelerator_log,
                },
            },
            decoder: DecoderConfig {
                ctc: CtcDecoderConfig {
                    blank_id: args.blank_id,
                    beam_width: args.beam_width,
                    n_best: args.n_best.unwrap_or(args.beam_width),
                },
                text: TextDecoderConfig {
                    tokenizer_path: args.tokenizer,
                    normalize_spaces: candidate_processing.normalize_spaces,
                    drop_empty_candidates: candidate_processing.drop_empty_candidates,
                },
                language_model: lm_config,
            },
        },
    )?;

    print_result(&result);
    println!("{}", result.transcript);
    info!("total: {}", format_duration(total_start.elapsed()));

    Ok(())
}

fn init_logger() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
}

fn print_result(result: &TranscriptionResult) {
    info!(
        "w2v-bert features: {} frames x {} dims ({} f32 values)",
        result.timings.feature_rows, result.timings.feature_cols, result.timings.feature_count
    );
    print_timings(&result.timings);
}

fn print_timings(report: &TimingReport) {
    info!("audio duration: {:.3}s", report.audio_duration_seconds);
    info!(
        "audio decode: {}",
        format_duration(report.audio_decode_elapsed)
    );
    info!(
        "feature extraction: {}",
        format_duration(report.feature_elapsed)
    );
    info!(
        "feature throughput: {:.0} values/s",
        report.feature_count as f64 / report.feature_elapsed.as_secs_f64()
    );
    info!(
        "model session: {}",
        format_duration(report.model.session_elapsed)
    );
    info!(
        "input tensor: {}",
        format_duration(report.model.input_elapsed)
    );
    info!(
        "onnx inference: {}",
        format_duration(report.model.inference_elapsed)
    );
    info!(
        "ctc beam search: {}",
        format_duration(report.model.ctc_elapsed)
    );
    info!(
        "tokenizer load: {}",
        format_duration(report.tokenizer_load_elapsed)
    );
    info!(
        "text decode: {}",
        format_duration(report.text_decode_elapsed)
    );
    info!("kenlm rerank: {}", format_duration(report.lm_elapsed));
    if let Some(candidate) = &report.best_candidate {
        info!(
            "best score: total={:.3} ctc={:.3} lm={:.3} words={}",
            candidate.total_score,
            candidate.ctc_log_prob,
            candidate.lm_log_prob,
            candidate.word_count
        );
    }
    info!(
        "measured pipeline: {}",
        format_duration(report.measured_elapsed())
    );
    info!("RTF/RFT: {:.3}x", report.real_time_factor());
}
