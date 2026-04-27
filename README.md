# w2v-bert-uk

[![Crates.io Version](https://img.shields.io/crates/v/w2v-bert-uk)](https://crates.io/crates/w2v-bert-uk)

Rust CLI for transcribing audio with a W2V-BERT frontend, an ONNX CTC acoustic model, SentencePiece decoding, and optional KenLM reranking.

The ONNX acoustic model may use either fp16 or fp32 tensors. Input and output
precision are detected from the model metadata at load time.

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
ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib cargo run --release --no-default-features --features ort-dynamic -- example_1.wav
cargo run --release --no-default-features --features ort-dynamic -- example_1.wav --ort-dylib /path/to/libonnxruntime.dylib
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

## CTC Alignment

The package also includes a CTC-segmentation aligner based on the dynamic
programming method from arXiv:2007.09127v2. It aligns an existing transcript to
audio and prints tab-separated utterance segments:

```bash
cargo run --release --bin ctc-align -- audio.wav transcript.txt model_optimized.onnx tokenizer.model
cargo run --release --bin ctc-align -- audio.wav transcript.txt --output-format jsonl --output-file segments.jsonl
```

`transcript.txt` should contain one utterance per non-empty line. Output columns
for the default TSV format are:

```text
start	end	score	text
```

The `score` is the minimum mean frame log-probability window from the paper. By
default the tool infers the CTC frame duration from `audio duration / CTC
frames`; pass `--index-duration` if your model requires a fixed value. The
aligner uses the reference moving-window table fill to keep memory bounded by
`window_size * transcript_tokens`; tune `--min-window-size` and
`--max-window-size` for long recordings.

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
from pathlib import Path

import w2v_bert_uk

# fp16 and fp32 ONNX acoustic models are both supported. The extension detects
# the model tensor precision when it loads the ONNX session.

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

report = w2v_bert_uk.transcribe_file_with_report(
    "example_1.wav",
    model="model_optimized.onnx",
    tokenizer="tokenizer.model",
    lm=None,
)
print(report["transcript"])
print(report["candidates"][0]["total_score"])
print(report["timings"]["model_inference_seconds"])

audio_bytes = Path("example_1.wav").read_bytes()
text_from_bytes = w2v_bert_uk.transcribe_bytes(
    audio_bytes,
    format_hint="wav",
    model="model_optimized.onnx",
    tokenizer="tokenizer.model",
    lm=None,
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
first_report = transcriber.transcribe_file_with_report("example_1.wav")
first_from_bytes = transcriber.transcribe_bytes(audio_bytes, format_hint="wav")
```

## Kotlin

Kotlin/JVM bindings are generated with UniFFI:

```bash
CARGO_PROFILE_RELEASE_STRIP=false cargo build --release --no-default-features --features kotlin,ort-dynamic --lib
cargo install uniffi --version 0.31.1 --locked --features cli --root target/uniffi-tools
target/uniffi-tools/bin/uniffi-bindgen generate target/release/libw2v_bert_uk.so --language kotlin --out-dir kotlin/generated --no-format
```

The generated Kotlin uses JNA to load `w2v_bert_uk`, so make the native library and ONNX Runtime discoverable at runtime.

Kotlin API:

```kotlin
import io.github.rustedbytes.w2vbertuk.KotlinTranscriber
import io.github.rustedbytes.w2vbertuk.defaultOptions

val options = defaultOptions().copy(
    model = "model_optimized.onnx",
    tokenizer = "tokenizer.model",
    lm = "news-titles.arpa",
)

KotlinTranscriber(options).use { transcriber ->
    val text = transcriber.transcribeFile("example_1.wav")
    println(text)
}
```

## Go

The crate can be built as a Go cgo package by enabling the `go` feature. The
Go package uses the generated C ABI from `cbindgen`, so the native library,
`c/w2v_bert_uk.h`, and the `go/` package must be kept together:

```bash
cargo build --release --no-default-features --features go,ort-dynamic --lib
mkdir -p native
cp target/release/libw2v_bert_uk.so native/
go test ./go
go run ./examples/transcribe.go example_1.wav
```

Go API:

```go
package main

import (
	"fmt"
	"log"

	w2vbertuk "github.com/RustedBytes/w2v-bert-uk/go"
)

func main() {
	transcriber, err := w2vbertuk.NewTranscriber(w2vbertuk.Options{
		Model:     "model_optimized.onnx",
		Tokenizer: "tokenizer.model",
		LM:        "news-titles.arpa",
		BeamWidth: 32,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer transcriber.Close()

	text, err := transcriber.TranscribeFile("example_1.wav")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(text)
}
```

## Node.js

The crate can be built as a Node.js 16+ native extension through Node-API and napi-rs:

```bash
npm install
npm run build:nodejs -- --platform artifacts
```

The generated extension can be loaded from the output directory:

```js
const w2vBertUk = require("./artifacts");

const text = w2vBertUk.transcribeFile("example_1.wav", {
  model: "model_optimized.onnx",
  tokenizer: "tokenizer.model",
  lm: null,
  beamWidth: 32,
  ortOptimization: "disable",
  fallbackSampleRate: 16000,
  skipDecodeErrors: true,
});

const report = w2vBertUk.transcribeFileWithReport("example_1.wav", {
  model: "model_optimized.onnx",
  tokenizer: "tokenizer.model",
  lm: null,
});

const transcriber = new w2vBertUk.Transcriber({
  model: "model_optimized.onnx",
  tokenizer: "tokenizer.model",
  lm: "news-titles.arpa",
});

const reused = transcriber.transcribeFile("example_2.wav");
console.log(text, report.timings.modelInferenceSeconds, reused);
```

The Node.js build uses `ort-dynamic`, so set `ORT_DYLIB_PATH` before loading the extension when ONNX Runtime is not discoverable by the system loader.

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

Node.js:

```bash
npm run build:nodejs -- --platform artifacts
node examples/transcribe.js
```

Kotlin:

```bash
CARGO_PROFILE_RELEASE_STRIP=false cargo build --release --no-default-features --features kotlin,ort-dynamic --lib
target/uniffi-tools/bin/uniffi-bindgen generate target/release/libw2v_bert_uk.so --language kotlin --out-dir kotlin/generated --no-format
# Compile examples/Transcribe.kt together with kotlin/generated/**/*.kt and
# include JNA on the Kotlin/JVM classpath.
```

Swift:

```bash
cargo build --release --no-default-features --features swift,ort-dynamic --lib
# Compile examples/transcribe.swift together with swift/generated/*.swift and
# swift/generated/w2v-bert-uk/*.swift, and link it against the native library.
```

C#:

```bash
cargo build --release --no-default-features --features csharp,ort-dynamic --lib
# Compile examples/Transcribe.cs together with csharp/NativeMethods.g.cs,
# enabling unsafe code and making the native library discoverable at runtime.
```

C and C++:

```bash
cargo build --release --no-default-features --features c,cpp,ort-dynamic --lib
cc -Ic -c examples/transcribe.c -o c-smoke.o
c++ -Icpp -std=c++17 -c examples/transcribe.cpp -o cpp-smoke.o
# Link your application against the generated native library and make ONNX
# Runtime discoverable at runtime when using ort-dynamic.
```

## Wheels

The GitHub Actions workflow in `.github/workflows/python-bindings.yml` builds Python wheels on:

- `ubuntu-22.04` as `linux-x86_64`
- `macos-latest` as `macos-arm64`
- `windows-latest` as `windows-x86_64`

The Linux wheel is built with `ort-dynamic` because the current bundled ONNX Runtime Linux binary requires newer glibc symbols than common Python runners provide. Use `ORT_DYLIB_PATH` or `ort_dylib_path` to point it at a compatible ONNX Runtime shared library at runtime.

Each job installs the wheel and runs an import smoke test before uploading the wheel artifact. Tag creation also uploads the wheels to the matching GitHub Release.

Build a wheel locally:

```bash
uvx maturin build --release --features python
uvx maturin build --release --features "python coreml"
uvx maturin build --release --features "python cuda"
```

## Node.js Extensions

The GitHub Actions workflow in `.github/workflows/nodejs-bindings.yml` builds Node.js `.node` extensions on:

- `ubuntu-22.04` as `linux-x64-gnu`
- `macos-latest` as `macos-arm64`
- `windows-latest` as `windows-x64-msvc`

Each job loads the extension in Node.js 16 before uploading the platform artifact. Tag creation also uploads the extensions to the matching GitHub Release.

## Kotlin Extensions

The GitHub Actions workflow in `.github/workflows/kotlin-bindings.yml` builds the shared native library and UniFFI-generated Kotlin/JVM bindings on:

- `ubuntu-22.04` as `linux-x64-gnu`
- `macos-latest` as `macos-arm64`
- `windows-latest` as `windows-x64-msvc`

Each job builds with `ort-dynamic`, generates Kotlin from the native library, and uploads a zip with the native library, generated Kotlin sources, UniFFI config, and example. Tag creation also uploads the zip files to the matching GitHub Release.

## C and C++ Extensions

The GitHub Actions workflow in `.github/workflows/c-cpp-bindings.yml` builds the shared native library and generated `cbindgen` headers on:

- `ubuntu-22.04` as `linux-x64-gnu`
- `macos-latest` as `macos-arm64`
- `windows-latest` as `windows-x64-msvc`

Each job compiles C and C++ header smoke tests before uploading a zip with the native library, generated headers, and examples. Tag creation also uploads the zip files to the matching GitHub Release.

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
