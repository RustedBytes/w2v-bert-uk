#include <iostream>

#include "w2v_bert_uk.hpp"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0]
              << " <audio-file> <model.onnx> <tokenizer.model> [lm.binary]\n";
    return 2;
  }

  w2v_bert_uk::W2vBertUkOptions options = w2v_bert_uk::w2v_bert_uk_options_default();
  options.model = argv[2];
  options.tokenizer = argv[3];
  options.lm = argc > 4 ? argv[4] : nullptr;
  options.ort_optimization = "disable";
  const char *hot_words[] = {"Kyiv"};
  options.hot_words = hot_words;
  options.hot_words_len = 1;
  options.hot_word_bonus = 2.0f;

  char *transcript = nullptr;
  int status = w2v_bert_uk::w2v_bert_uk_transcribe_file(argv[1], &options, &transcript);
  if (status != w2v_bert_uk::W2V_BERT_UK_OK) {
    char *message = w2v_bert_uk::w2v_bert_uk_last_error_message();
    std::cerr << "transcription failed: " << (message != nullptr ? message : "unknown error")
              << '\n';
    w2v_bert_uk::w2v_bert_uk_string_free(message);
    return 1;
  }

  std::cout << transcript << '\n';
  w2v_bert_uk::w2v_bert_uk_string_free(transcript);
  return 0;
}
