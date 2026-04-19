use std::path::PathBuf;

use anyhow::Result;
use w2v_uk_rs::{
    AcousticModelConfig, CtcDecoderConfig, DecoderConfig, LmConfig, TextDecoderConfig, Transcriber,
    TranscriptionConfig,
};

fn main() -> Result<()> {
    let audio_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("example_1.wav"));

    let mut transcriber = Transcriber::new(TranscriptionConfig {
        model: AcousticModelConfig {
            path: PathBuf::from("model_optimized.onnx"),
            ..AcousticModelConfig::default()
        },
        decoder: DecoderConfig {
            ctc: CtcDecoderConfig {
                beam_width: 32,
                n_best: 32,
                ..CtcDecoderConfig::default()
            },
            text: TextDecoderConfig {
                tokenizer_path: PathBuf::from("tokenizer.model"),
                ..TextDecoderConfig::default()
            },
            language_model: Some(LmConfig {
                path: PathBuf::from("lm.binary"),
                ..LmConfig::default()
            }),
        },
        ..TranscriptionConfig::default()
    })?;

    let result = transcriber.transcribe_audio_file(audio_path)?;
    println!("{}", result.transcript);

    Ok(())
}
