#include <iostream>

#include "rust_asr.hpp"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0]
              << " <audio-file> <model.onnx> <tokenizer.model> [lm.binary]\n";
    return 2;
  }

  rust_asr::RustAsrOptions options = rust_asr::rust_asr_options_default();
  options.model = argv[2];
  options.tokenizer = argv[3];
  options.lm = argc > 4 ? argv[4] : nullptr;
  options.ort_optimization = "disable";
  const char *hot_words[] = {"Kyiv"};
  options.hot_words = hot_words;
  options.hot_words_len = 1;
  options.hot_word_bonus = 2.0f;

  char *transcript = nullptr;
  int status = rust_asr::rust_asr_transcribe_file(argv[1], &options, &transcript);
  if (status != rust_asr::RUST_ASR_OK) {
    char *message = rust_asr::rust_asr_last_error_message();
    std::cerr << "transcription failed: " << (message != nullptr ? message : "unknown error")
              << '\n';
    rust_asr::rust_asr_string_free(message);
    return 1;
  }

  std::cout << transcript << '\n';
  rust_asr::rust_asr_string_free(transcript);
  return 0;
}
