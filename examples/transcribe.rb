require "rust_asr"

audio_file = ARGV.fetch(0)
options = {
  model: ARGV.fetch(1, "model_optimized.onnx"),
  tokenizer: ARGV.fetch(2, "tokenizer.model"),
  lm: ARGV[3],
  beam_width: 32
}

puts RustAsr.transcribe_file(audio_file, options)
