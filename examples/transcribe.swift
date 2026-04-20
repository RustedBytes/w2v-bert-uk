import Foundation

let root = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()

func fromRoot(_ path: String) -> String {
    root.appendingPathComponent(path).path
}

do {
    var options = swift_options_default()
    options.model = RustString(fromRoot("model_optimized.onnx"))
    options.tokenizer = RustString(fromRoot("tokenizer.model"))
    options.beam_width = 32
    options.lm = RustString(fromRoot("news-titles.arpa"))
    options.lm_weight = 0.45
    options.word_bonus = 0.2
    options.use_language_model = true
    options.log_language_model = false
    options.ort_dylib_path = RustString("")
    options.ort_optimization = RustString("disable")
    options.log_accelerator = true
    options.fallback_sample_rate = 16_000
    options.skip_decode_errors = true
    options.w2v_model_source = RustString("")
    options.w2v_sample_rate = 0
    options.w2v_feature_size = 0
    options.w2v_stride = 0
    options.w2v_feature_dim = 0
    options.w2v_padding_value = Float.nan
    options.blank_id = 0
    options.n_best = 32
    options.normalize_spaces = true
    options.drop_empty_candidates = true
    options.lm_bos = true
    options.lm_eos = true

    let transcriber = try swift_transcriber_new(options)

    let first = try transcriber.transcribe_file(fromRoot("example_1.wav"))
    print(first.toString())

    let second = try transcriber.transcribe_file(fromRoot("example_2.wav"))
    print(second.toString())
} catch let error as RustString {
    fputs("\(error.toString())\n", stderr)
    exit(1)
} catch {
    fputs("\(error)\n", stderr)
    exit(1)
}
