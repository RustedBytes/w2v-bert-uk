#include <stdio.h>

#include "rust_asr.h"

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <audio-file> <model.onnx> <tokenizer.model> [lm.binary]\n", argv[0]);
    return 2;
  }

  RustAsrOptions options = rust_asr_options_default();
  options.model = argv[2];
  options.tokenizer = argv[3];
  options.lm = argc > 4 ? argv[4] : NULL;
  options.ort_optimization = "disable";
  const char *hot_words[] = {"Kyiv"};
  options.hot_words = hot_words;
  options.hot_words_len = 1;
  options.hot_word_bonus = 2.0f;

  char *transcript = NULL;
  int status = rust_asr_transcribe_file(argv[1], &options, &transcript);
  if (status != RUST_ASR_OK) {
    char *message = rust_asr_last_error_message();
    fprintf(stderr, "transcription failed: %s\n", message != NULL ? message : "unknown error");
    rust_asr_string_free(message);
    return 1;
  }

  puts(transcript);
  rust_asr_string_free(transcript);
  return 0;
}
