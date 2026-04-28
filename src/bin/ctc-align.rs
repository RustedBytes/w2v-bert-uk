use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use env_logger::Env;
use log::info;
use serde_json::json;
use rust_asr::{
    EncoderConfig, W2vBertEncoderConfig,
    audio::{AudioDecodeConfig, audio_file_to_w2v_bert_features_with_config},
    ctc::{CtcAlignmentConfig, CtcAlignmentSegment, align_token_sequences},
    format_duration, init_ort,
    model::{CtcModel, ModelConfig, ModelOptimizationLevel},
    tokenizer::load_sentencepiece_tokenizer,
};

#[derive(Parser)]
#[command(
    author,
    version,
    about = "Align transcript utterances to audio with CTC segmentation"
)]
struct Args {
    /// Audio file to align.
    audio_file: PathBuf,

    /// Text file with one utterance per line.
    transcript: PathBuf,

    /// ONNX model path.
    #[arg(default_value = "model_optimized.onnx")]
    model: PathBuf,

    /// SentencePiece tokenizer model path.
    #[arg(default_value = "tokenizer.model")]
    tokenizer: PathBuf,

    /// Optional ONNX Runtime dynamic library path.
    #[arg(long)]
    ort_dylib: Option<PathBuf>,

    /// ONNX Runtime graph optimization level.
    #[arg(long, value_enum, default_value_t = CliOptimizationLevel::Disable)]
    ort_optimization: CliOptimizationLevel,

    /// Disable accelerator logging.
    #[arg(long)]
    no_accelerator_log: bool,

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

    /// Seconds represented by one CTC output frame. Defaults to audio duration / CTC frames.
    #[arg(long)]
    index_duration: Option<f64>,

    /// Number of frames used by the minimum-mean confidence window.
    #[arg(long, default_value_t = 30)]
    score_min_mean_over_l: usize,

    /// Initial CTC segmentation window size in frames.
    #[arg(long, default_value_t = 8000)]
    min_window_size: usize,

    /// Maximum CTC segmentation window size in frames.
    #[arg(long, default_value_t = 100000)]
    max_window_size: usize,

    /// Treat blank-state self transitions as zero-cost.
    #[arg(long)]
    blank_transition_cost_zero: bool,

    /// Charge CTC blank cost for skipped audio before the transcript starts.
    #[arg(long)]
    no_preamble_transition_cost_zero: bool,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Tsv)]
    output_format: OutputFormat,

    /// Write alignment output to a file instead of stdout.
    #[arg(long)]
    output_file: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliOptimizationLevel {
    Disable,
    Level1,
    Level2,
    Level3,
    All,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Tsv,
    Jsonl,
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

fn main() -> Result<()> {
    init_logger();
    let total_start = Instant::now();
    let args = Args::parse();

    init_ort(args.ort_dylib.as_deref())?;

    let transcript = fs::read_to_string(&args.transcript)
        .with_context(|| format!("failed to read transcript {}", args.transcript.display()))?;
    let utterances = transcript
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if utterances.is_empty() {
        anyhow::bail!(
            "transcript {} contains no utterances",
            args.transcript.display()
        );
    }

    let encoder = EncoderConfig {
        w2v_bert: W2vBertEncoderConfig {
            model_source: args.w2v_model_source,
            sample_rate: args.w2v_sample_rate,
            feature_size: args.w2v_feature_size,
            stride: args.w2v_stride,
            feature_dim: args.w2v_feature_dim,
            padding_value: args.w2v_padding_value,
        },
    };
    let audio = audio_file_to_w2v_bert_features_with_config(
        &args.audio_file,
        &AudioDecodeConfig {
            fallback_sample_rate: args.fallback_sample_rate,
            skip_decode_errors: !args.strict_audio_decode,
            ffmpeg_fallback: true,
        },
        &encoder.w2v_bert.to_frontend_config(),
    )?;
    let audio_duration_seconds = audio.duration_seconds();

    let tokenizer = load_sentencepiece_tokenizer(&args.tokenizer)?;
    let tokenized = utterances
        .iter()
        .map(|utterance| transcript_tokens(&tokenizer, utterance, args.blank_id))
        .collect::<Vec<_>>();

    let mut model = CtcModel::load(
        &args.model,
        &ModelConfig {
            optimization_level: args.ort_optimization.into(),
            log_accelerator: !args.no_accelerator_log,
        },
    )?;
    let log_probs = model.run_log_probs(audio.features)?;
    let index_duration = args
        .index_duration
        .unwrap_or(audio_duration_seconds / log_probs.frames as f64);

    let alignment = align_token_sequences(
        &log_probs.values,
        log_probs.frames,
        log_probs.vocab_size,
        &tokenized,
        &utterances,
        &CtcAlignmentConfig {
            blank_id: args.blank_id,
            index_duration,
            score_min_mean_over_l: args.score_min_mean_over_l,
            min_window_size: args.min_window_size,
            max_window_size: args.max_window_size,
            blank_transition_cost_zero: args.blank_transition_cost_zero,
            preamble_transition_cost_zero: !args.no_preamble_transition_cost_zero,
        },
    )?;

    info!(
        "audio: {:.3}s, ctc: {} frames x {} vocab, frame: {:.6}s",
        audio_duration_seconds, log_probs.frames, log_probs.vocab_size, index_duration
    );
    info!("audio decode: {}", format_duration(audio.decode_elapsed));
    info!(
        "feature extraction: {}",
        format_duration(audio.feature_elapsed)
    );
    info!(
        "model session: {}",
        format_duration(log_probs.session_elapsed)
    );
    info!("input tensor: {}", format_duration(log_probs.input_elapsed));
    info!(
        "onnx inference: {}",
        format_duration(log_probs.inference_elapsed)
    );
    info!(
        "ctc log-softmax: {}",
        format_duration(log_probs.log_softmax_elapsed)
    );
    info!("total: {}", format_duration(total_start.elapsed()));

    write_segments(
        &alignment.segments,
        args.output_format,
        args.output_file.as_ref(),
    )?;

    Ok(())
}

fn init_logger() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
}

fn write_segments(
    segments: &[CtcAlignmentSegment],
    output_format: OutputFormat,
    output_file: Option<&PathBuf>,
) -> Result<()> {
    match output_file {
        Some(path) => {
            let file = fs::File::create(path)
                .with_context(|| format!("failed to create output file {}", path.display()))?;
            let mut writer = BufWriter::new(file);
            write_segments_to(&mut writer, segments, output_format)?;
            writer
                .flush()
                .with_context(|| format!("failed to flush output file {}", path.display()))
        }
        None => {
            let stdout = io::stdout();
            let mut writer = stdout.lock();
            write_segments_to(&mut writer, segments, output_format)
        }
    }
}

fn write_segments_to(
    writer: &mut impl Write,
    segments: &[CtcAlignmentSegment],
    output_format: OutputFormat,
) -> Result<()> {
    match output_format {
        OutputFormat::Tsv => {
            writeln!(writer, "start\tend\tscore\ttext")?;
            for segment in segments {
                writeln!(
                    writer,
                    "{:.3}\t{:.3}\t{:.6}\t{}",
                    segment.start_seconds,
                    segment.end_seconds,
                    segment.score,
                    segment.text.replace('\t', " ")
                )?;
            }
        }
        OutputFormat::Jsonl => {
            for segment in segments {
                let value = json!({
                    "start": segment.start_seconds,
                    "end": segment.end_seconds,
                    "score": segment.score,
                    "text": segment.text,
                });
                serde_json::to_writer(&mut *writer, &value)?;
                writeln!(writer)?;
            }
        }
    }

    Ok(())
}

fn transcript_tokens(
    tokenizer: &splintr::SentencePieceTokenizer,
    text: &str,
    blank_id: u32,
) -> Vec<u32> {
    let bos = tokenizer.bos_token_id();
    let eos = tokenizer.eos_token_id();
    tokenizer
        .encode(text)
        .into_iter()
        .filter(|&token| Some(token) != bos && token != eos && token != blank_id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments() -> Vec<CtcAlignmentSegment> {
        vec![CtcAlignmentSegment {
            start_seconds: 1.25,
            end_seconds: 2.5,
            score: -0.75,
            text: "hello\tсвіт".to_string(),
        }]
    }

    #[test]
    fn writes_tsv_segments() {
        let mut output = Vec::new();
        write_segments_to(&mut output, &segments(), OutputFormat::Tsv).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "start\tend\tscore\ttext\n1.250\t2.500\t-0.750000\thello світ\n"
        );
    }

    #[test]
    fn writes_jsonl_segments() {
        let mut output = Vec::new();
        write_segments_to(&mut output, &segments(), OutputFormat::Jsonl).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "{\"end\":2.5,\"score\":-0.75,\"start\":1.25,\"text\":\"hello\\tсвіт\"}\n"
        );
    }
}
