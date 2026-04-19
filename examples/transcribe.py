from pathlib import Path

import w2v_uk_rs


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    transcriber = w2v_uk_rs.Transcriber(
        model=ROOT / "model_optimized.onnx",
        tokenizer=ROOT / "tokenizer.model",
        beam_width=32,
        lm=ROOT / "news-titles.arpa",
        lm_weight=0.45,
        word_bonus=0.2,
        log_language_model=False,
        ort_dylib_path=None,
        ort_optimization="disable",
        log_accelerator=True,
        fallback_sample_rate=16_000,
        skip_decode_errors=True,
        w2v_model_source=None,
        w2v_sample_rate=None,
        w2v_feature_size=None,
        w2v_stride=None,
        w2v_feature_dim=None,
        w2v_padding_value=None,
        blank_id=0,
        n_best=32,
        normalize_spaces=True,
        drop_empty_candidates=True,
        lm_bos=True,
        lm_eos=True,
    )

    text = transcriber.transcribe_file(ROOT / "example_1.wav")
    print(text)

    text = transcriber.transcribe_file(ROOT / "example_2.wav")
    print(text)


if __name__ == "__main__":
    main()
