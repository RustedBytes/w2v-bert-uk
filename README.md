# w2v-bert-uk

Rust CLI for transcribing audio with a W2V-BERT frontend, an ONNX CTC acoustic model, SentencePiece decoding, and optional KenLM reranking.

## Requirements

- Rust 1.94 or newer
- Local model artifacts:
  - `model_optimized.onnx`
  - `tokenizer.model`
  - optional KenLM model such as `lm.binary` or `news-titles.arpa` for reranking

The model files are ignored by git because they are large local artifacts.

## Usage

```bash
cargo run --release -- <audio-file> [model.onnx] [tokenizer.model] [beam-width] [lm.binary] [lm-weight] [word-bonus]
```

Defaults:

```text
model.onnx      model_optimized.onnx
tokenizer.model tokenizer.model
beam-width      32
n-best          beam-width
lm.binary       lm.binary
lm-weight       0.45
word-bonus      0.2
```

If the default LM path does not exist, the CLI disables KenLM and decodes without LM reranking.

Example:

```bash
cargo run --release -- example_1.wav
```

By default ONNX Runtime uses CPU execution. Hardware execution providers are opt-in Cargo features:

```bash
cargo run --release --features coreml -- example_1.wav
cargo run --release --features cuda -- example_1.wav
```

CoreML requires macOS. CUDA requires an ONNX Runtime CUDA build and compatible NVIDIA CUDA libraries.

By default, the matching ONNX Runtime binaries are downloaded at build time. To load an external ONNX Runtime dynamic library instead, build with `ort-dynamic` and set `ORT_DYLIB_PATH` or pass `--ort-dylib`:

```bash
ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib cargo run --release --features ort-dynamic -- example_1.wav
cargo run --release --features ort-dynamic -- example_1.wav --ort-dylib /path/to/libonnxruntime.dylib
```

Print help:

```bash
cargo run -- --help
```

The full pipeline is configurable from the CLI:

```bash
cargo run --release -- example_1.wav \
  --ort-optimization level1 \
  --fallback-sample-rate 16000 \
  --strict-audio-decode \
  --w2v-sample-rate 16000 \
  --w2v-feature-size 80 \
  --w2v-stride 2 \
  --blank-id 0 \
  --n-best 16 \
  --no-normalize-spaces \
  --no-accelerator-log \
  --no-lm-log \
  --lm-no-bos \
  --lm-no-eos
```

For Rust callers, `TranscriptionConfig` is split by processing stage:

```rust
use w2v_bert_uk::{
    AcousticModelConfig, CtcDecoderConfig, DecoderConfig, EncoderConfig,
    RuntimeConfig, TextDecoderConfig, TranscriptionConfig, W2vBertEncoderConfig,
    audio::AudioDecodeConfig,
    model::{ModelConfig, ModelOptimizationLevel},
};

let config = TranscriptionConfig {
    runtime: RuntimeConfig { ort_dylib_path: None },
    audio: AudioDecodeConfig {
        fallback_sample_rate: 16_000,
        skip_decode_errors: true,
    },
    encoder: EncoderConfig {
        w2v_bert: W2vBertEncoderConfig {
            sample_rate: Some(16_000),
            feature_size: Some(80),
            stride: Some(2),
            ..Default::default()
        },
    },
    model: AcousticModelConfig {
        path: "model_optimized.onnx".into(),
        session: ModelConfig {
            optimization_level: ModelOptimizationLevel::Disable,
            log_accelerator: true,
        },
    },
    decoder: DecoderConfig {
        ctc: CtcDecoderConfig {
            blank_id: 0,
            beam_width: 32,
            n_best: 32,
        },
        text: TextDecoderConfig {
            tokenizer_path: "tokenizer.model".into(),
            normalize_spaces: true,
            drop_empty_candidates: true,
        },
        language_model: None,
    },
};
```

## Python

The crate can also be built as a PyO3 extension module for Python 3.10+. Use `maturin` to install the extension into the active Python environment:

```bash
uvx maturin develop --release --features python
```

Build with accelerators by combining features:

```bash
uvx maturin develop --release --features "python coreml"
uvx maturin develop --release --features "python cuda"
```

CoreML is for macOS. CUDA requires a compatible NVIDIA CUDA runtime. If you use `uv run`, rebuild the environment after changing Rust/PyO3 signatures:

```bash
uv cache clean w2v-bert-uk
uv sync --reinstall-package w2v-bert-uk
```

Python API:

```python
import w2v_bert_uk

# Convenience one-shot call. This initializes the model for this call.
text = w2v_bert_uk.transcribe_file(
    "example_1.wav",
    model="model_optimized.onnx",
    tokenizer="tokenizer.model",
    beam_width=32,
    lm=None,
    lm_weight=0.45,
    word_bonus=0.2,
    log_language_model=False,
    ort_dylib_path=None,
    ort_optimization="disable",
    log_accelerator=True,
    fallback_sample_rate=16000,
    skip_decode_errors=True,
    w2v_model_source=None,
    w2v_sample_rate=16000,
    w2v_feature_size=80,
    w2v_stride=2,
    w2v_feature_dim=None,
    w2v_padding_value=None,
    blank_id=0,
    n_best=32,
    normalize_spaces=True,
    drop_empty_candidates=True,
    lm_bos=True,
    lm_eos=True,
)

# Reusable transcriber. The ONNX model session and tokenizer are initialized
# once and reused for each audio file.
transcriber = w2v_bert_uk.Transcriber(
    model="model_optimized.onnx",
    tokenizer="tokenizer.model",
    beam_width=32,
    lm="news-titles.arpa",
    lm_weight=0.45,
    word_bonus=0.2,
    log_language_model=False,
    ort_dylib_path=None,
    ort_optimization="disable",
    log_accelerator=True,
)

first = transcriber.transcribe_file("example_1.wav")
second = transcriber.transcribe_file("example_2.wav")
```

## Examples

Rust:

```bash
cargo run --example transcribe -- example_1.wav
```

Python:

```bash
uvx maturin develop --release --features python
uv run python examples/transcribe.py
```

## Wheels

The GitHub Actions workflow in `.github/workflows/python-bindings.yml` builds Python wheels on:

- `ubuntu-22.04` as `linux-x86_64`
- `macos-latest` as `macos-arm64`
- `macos-15-intel` as `macos-x86_64`
- `windows-latest` as `windows-x86_64`

Each job installs the wheel and runs an import smoke test before uploading the wheel artifact. Tag creation also uploads the wheels to the matching GitHub Release.

Build a wheel locally:

```bash
uvx maturin build --release --features python
uvx maturin build --release --features "python coreml"
uvx maturin build --release --features "python cuda"
```

## Output

The transcript is printed to stdout. Timings and decoder diagnostics are printed to stderr, so you can redirect the transcript cleanly:

```bash
cargo run --release -- example_1.wav > transcript.txt
```

Timing output includes audio duration, audio decode time, feature extraction time, ONNX session setup, inference, CTC beam search, KenLM reranking, total wall time, and real-time factor:

```text
audio duration: 8.515s
audio decode: 16.578ms
feature extraction: 426.010ms
onnx inference: 2.333s
ctc beam search: 5.336s
kenlm rerank: 21.690ms
RTF/RFT: 1.158x
```

## KenLM Reranking

If an LM path is configured, the decoder reranks CTC N-best candidates using shallow fusion:

```text
total = ctc_log_prob + lm_weight * lm_log_prob + word_bonus * word_count
```

Candidates are scored by KenLM with their decoded casing preserved, so the language model should use casing that matches the tokenizer output.

Tune `lm-weight` and `word-bonus` on validation audio before using the defaults for evaluation.

## Development

```bash
cargo fmt --check
cargo check
```

The CoreML feature uses static input shapes for CoreML subgraphs. Unsupported dynamic graph regions fall back through ONNX Runtime.
