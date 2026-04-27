use std::path::Path;

use anyhow::{Context, Result};
use prost::Message;
use splintr::SentencePieceTokenizer;

pub fn load_sentencepiece_tokenizer(path: &Path) -> Result<SentencePieceTokenizer> {
    load_sentencepiece_tokenizer_with_bos(path, true)
}

pub fn load_sentencepiece_transcript_tokenizer(path: &Path) -> Result<SentencePieceTokenizer> {
    load_sentencepiece_tokenizer_with_bos(path, false)
}

pub fn sentencepiece_vocab_size(path: &Path) -> Result<usize> {
    let model = load_sentencepiece_model(path)?;
    Ok(model.pieces.len())
}

fn load_sentencepiece_tokenizer_with_bos(
    path: &Path,
    include_bos: bool,
) -> Result<SentencePieceTokenizer> {
    let model = load_sentencepiece_model(path)?;

    let mut tokens = Vec::with_capacity(model.pieces.len());
    let mut scores = Vec::with_capacity(model.pieces.len());

    for piece in model.pieces {
        tokens.push(piece.piece.unwrap_or_default());
        scores.push(piece.score.unwrap_or(0.0));
    }

    let eos_token_id = tokens.iter().position(|token| token == "</s>").unwrap_or(2) as u32;
    let bos_token_id = tokens
        .iter()
        .position(|token| token == "<s>")
        .filter(|_| include_bos)
        .map(|index| index as u32);

    SentencePieceTokenizer::new(tokens, scores, bos_token_id, eos_token_id)
        .context("failed to create Splintr SentencePiece tokenizer")
}

fn load_sentencepiece_model(path: &Path) -> Result<SentencePieceModel> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read tokenizer model {}", path.display()))?;
    SentencePieceModel::decode(bytes.as_slice())
        .with_context(|| format!("failed to parse SentencePiece model {}", path.display()))
}

#[derive(Clone, PartialEq, Message)]
struct SentencePieceModel {
    #[prost(message, repeated, tag = "1")]
    pieces: Vec<SentencePiece>,
}

#[derive(Clone, PartialEq, Message)]
struct SentencePiece {
    #[prost(string, optional, tag = "1")]
    piece: Option<String>,
    #[prost(float, optional, tag = "2")]
    score: Option<f32>,
}
