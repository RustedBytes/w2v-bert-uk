#include <stdio.h>

#include "w2v_bert_uk.h"

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <audio-file> <model.onnx> <tokenizer.model> [lm.binary]\n", argv[0]);
    return 2;
  }

  W2vBertUkOptions options = w2v_bert_uk_options_default();
  options.model = argv[2];
  options.tokenizer = argv[3];
  options.lm = argc > 4 ? argv[4] : NULL;
  options.ort_optimization = "disable";
  const char *hot_words[] = {"Kyiv"};
  options.hot_words = hot_words;
  options.hot_words_len = 1;
  options.hot_word_bonus = 2.0f;

  char *transcript = NULL;
  int status = w2v_bert_uk_transcribe_file(argv[1], &options, &transcript);
  if (status != W2V_BERT_UK_OK) {
    char *message = w2v_bert_uk_last_error_message();
    fprintf(stderr, "transcription failed: %s\n", message != NULL ? message : "unknown error");
    w2v_bert_uk_string_free(message);
    return 1;
  }

  puts(transcript);
  w2v_bert_uk_string_free(transcript);
  return 0;
}
