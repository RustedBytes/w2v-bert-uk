# Training With The Rust Burn Trainer

This document explains how to run the Rust training loop in `src/bin/train.rs`.
The trainer supports Squeezeformer, Zipformer, Paraformer, enhanced Paraformer,
and W2V-BERT style CTC models through Burn.

## Quick Start

Run a tiny CPU smoke test first:

```bash
cargo run --bin train -- \
  --train-manifest testdata \
  --architecture squeezeformer \
  --input-dim 80 \
  --vocab-size <VOCAB_SIZE> \
  --batch-size 2 \
  --max-train-samples 4 \
  --epochs 1 \
  --dry-run
```

Run CUDA training on two GPUs:

```bash
RUST_LOG=info CUDA_VISIBLE_DEVICES=0,1 cargo run --release --features burn-cuda-backend --bin train -- \
  --train-manifest testdata \
  --architecture squeezeformer \
  --variant xs \
  --backend cuda \
  --device-indices 0,1 \
  --precision f32 \
  --input-dim 80 \
  --vocab-size <VOCAB_SIZE> \
  --batch-size 2 \
  --adaptive-batch-unit feature-values \
  --adaptive-batch-budget 40000 \
  --adaptive-batch-max-samples 2 \
  --sort-by-length-desc \
  --dataset-index-dir runs/index-cache \
  --log-every 1 \
  --epochs 10 \
  --learning-rate 0.001 \
  --gradient-clip-norm 5.0 \
  --output-dir runs/squeezeformer-2gpu
```

Replace `<VOCAB_SIZE>` with the tokenizer vocabulary size including the CTC blank
symbol. If your records contain transcript text but not token ids, pass
`--tokenizer tokenizer.model`.

## Train A Tokenizer

The `train` binary also has a SentencePiece tokenizer subcommand. It extracts
transcripts from JSONL, TSV, Parquet, plain text files, or audio transcript
sidecars, writes a temporary corpus, calls SentencePiece, and validates that the
resulting `.model` can be loaded by the Rust transcript tokenizer.

```bash
cargo run --bin train -- tokenizer \
  --input testdata \
  --output-dir tokenizer \
  --model-prefix ukrainian_asr \
  --vocab-size 5000 \
  --model-type unigram \
  --character-coverage 0.9995
```

This creates:

```text
tokenizer/ukrainian_asr.model
tokenizer/ukrainian_asr.vocab
```

SentencePiece must be installed so `spm_train` or `sentencepiece-train` is on
`PATH`. You can also point to an explicit binary:

```bash
cargo run --bin train -- tokenizer \
  --input manifests/train \
  --output-dir tokenizer \
  --model-prefix ukrainian_asr \
  --vocab-size 8000 \
  --sentencepiece-command /path/to/spm_train
```

Tokenizer inputs:

| Input type | Text source |
| --- | --- |
| JSONL/JSON | `text`, `transcript`, `transcription`, `sentence`, or `normalized_text` |
| TSV | fifth column, after `features_path`, `rows`, `cols`, `tokens` |
| Parquet | `text`, `transcript`, `transcription`, `sentence`, or `normalized_text` columns |
| Plain text | one training sentence per non-empty line |
| Audio files | `.txt`, `.lab`, or `.transcript` sidecar beside each audio file |

Useful tokenizer options:

| Argument | Default | Description |
| --- | --- | --- |
| `--input <PATH>` | required | File or folder. Repeat it for multiple roots. |
| `--output-dir <DIR>` | `tokenizer` | Directory for `.model` and `.vocab`. |
| `--model-prefix <NAME>` | `tokenizer` | Output filename prefix. |
| `--vocab-size <N>` | `5000` | SentencePiece vocabulary size. Training `--vocab-size` should usually be this value plus one CTC blank token. |
| `--model-type <TYPE>` | `unigram` | One of `unigram`, `bpe`, `char`, `word`. |
| `--input-sentence-size <N>` | `0` | Let SentencePiece sample at most `N` sentences. `0` uses all extracted lines. |
| `--shuffle-input-sentence` | false | Shuffle before SentencePiece sampling. |
| `--train-extremely-large-corpus` | false | Enable SentencePiece large-corpus mode. |
| `--user-defined-symbols <CSV>` | unset | Extra symbols to reserve. |
| `--byte-fallback` | false | Enable byte fallback pieces. |
| `--keep-corpus` | false | Keep `<model-prefix>.corpus.txt` after training. |

## Extract Feature Parquet

Use `extract-features` to preprocess Parquet/audio manifests into a new Parquet
file containing flattened `features`, `tokens`, `rows`, `cols`, `duration_ms`,
`id`, and `text` columns. The output is directly accepted by `--train-manifest`.

```bash
cargo run --release --bin train -- extract-features \
  --input /path/to/audio-or-parquet \
  --output data/features-w2v.parquet \
  --architecture w2v-bert \
  --tokenizer tokenizer.model \
  --max-audio-duration-sec 20 \
  --jobs 8 \
  --tui
```

Pass `--input` multiple times for multiple shards. Use `--max-samples` for a
bounded preprocessing run. Feature extraction uses Rayon; pass `--jobs` to set
the worker count. `--tui` opens a live terminal monitor for decoded records,
skipped long samples, the current input shard, and the last extracted sample.
When any `--input` is a directory, `--output` is treated as an output directory
and one numbered Parquet file is written per input.

## Build Modes

The trainer defaults to CPU. GPU backends are selected at build time:

```bash
# CPU
cargo run --release --bin train -- ...

# CUDA
cargo run --release --features burn-cuda-backend --bin train -- ...

# WGPU
cargo run --release --features burn-wgpu-backend --bin train -- ...
```

CUDA supports `--precision f32`, `--precision f16`, and `--precision bf16` when
the device/backend supports them. `--mixed-precision` is a shortcut for
`--precision f16`. WGPU does not support BF16 in this trainer.

### Optional ASR CubeCL Kernels

The crate has an experimental `asr-cubecl-kernels` feature with custom CubeCL
kernels for architecture-specific hot spots that are not already covered by
Burn's built-in matmul, convolution, softmax, and elementwise kernels:

| Kernel target | Architectures | Operation |
| --- | --- | --- |
| `ZipformerSwooshL` | Zipformer | fused `x * sigmoid(x - 4)` |
| `ZipformerSwooshR` | Zipformer | fused `x * sigmoid(x - 1)` |
| `RelativeShift` | Squeezeformer, Zipformer | relative-position attention shift |

Build with the feature when experimenting with direct CubeCL kernels:

```bash
cargo check --features asr-cubecl-kernels
cargo run --release --features burn-cuda-backend,asr-cubecl-kernels --bin train -- ...
```

The default trainer path still uses portable Burn tensor operations unless code
is explicitly routed through `w2v_bert_uk::cubecl_kernels`.

## Architecture And Feature Dimensions

Raw audio and Parquet audio bytes are feature-extracted according to
`--architecture`:

| Architecture | Flag | Extractor | `--input-dim` |
| --- | --- | --- | --- |
| Squeezeformer | `--architecture squeezeformer` | `asr_features::squeezeformer_frontend_config()` | `80` |
| Zipformer | `--architecture zipformer` or `--zipformer` | `asr_features::zipformer_frontend_config()` | `80` |
| Paraformer | `--architecture paraformer` or `--paraformer` | `asr_features::paraformer_frontend_config()` | `80` |
| W2V-BERT | `--architecture w2v-bert` or `--w2v-bert` | `W2vBertEncoderConfig::default()` | `160` |

Inline or precomputed feature manifests are not re-extracted. Their feature
dimension must match `--input-dim`.

## Dataset Inputs

Use exactly one of `--train-manifest` or `--manifest-dir`.

### JSONL Manifest

Each JSON object can contain inline features, feature files, audio paths, or
transcript text.

Inline features:

```json
{"id":"utt-1","features":[[0.1,0.2],[0.3,0.4]],"tokens":[1,2,3],"text":"hello"}
```

Flat features with shape:

```json
{"id":"utt-1","features":[0.1,0.2,0.3,0.4],"rows":2,"cols":2,"tokens":"1 2 3"}
```

Feature file:

```json
{"id":"utt-1","features_path":"utt-1.features.txt","rows":120,"cols":80,"tokens":[1,2,3]}
```

Audio file:

```json
{"id":"utt-1","audio_path":"audio/utt-1.wav","tokens":[1,2,3],"text":"hello"}
```

Accepted transcript fields are `text`, `transcript`, `transcription`, `sentence`, and `normalized_text`.
Accepted token fields are `tokens`, `target`, and `targets`.

### TSV Manifest

TSV rows use:

```text
features_path<TAB>rows<TAB>cols<TAB>tokens<TAB>optional_text
```

Example:

```text
utt-1.features.txt	120	80	1 2 3	hello
```

### Manifest Directory

`--manifest-dir data/manifests` resolves:

```text
data/manifests/train.jsonl
data/manifests/val.jsonl
```

If `val.jsonl` is absent, the trainer also checks `validation.jsonl` and
`dev.jsonl`. If no validation manifest is provided or found, the trainer writes
an automatic split under `output-dir/auto-validation-split`, using about 10% of
the training records for validation.

### Raw Audio Directory

If a directory has no `.jsonl`/`.json`/`.parquet` files, it is scanned
recursively for audio files:

```text
data/audio/train/utt-1.wav
data/audio/train/utt-1.tokens
data/audio/train/utt-1.txt
```

Supported audio extensions are `wav`, `flac`, `mp3`, `ogg`, `opus`, `m4a`,
`aac`, and `webm`.

Sidecars:

| Sidecar | Purpose |
| --- | --- |
| `.tokens` or `.tok` | Token ids, whitespace or comma separated |
| `.txt`, `.lab`, or `.transcript` | Transcript text |
| `.rows` or `.frames` | Optional frame count hint for sorting/indexing |

If only transcript text is present, pass `--tokenizer`.

### Parquet Folder

Folders can contain one or more `.parquet` files. Each row is treated as one
training example. The loader recognizes common column names:

| Meaning | Column names |
| --- | --- |
| id | `id`, `utt_id`, `utterance_id`, `key`, `sample_id` |
| transcript | `text`, `transcript`, `transcription`, `sentence`, `normalized_text` |
| tokens | `tokens`, `target`, `targets`, `labels`, `label_ids` |
| precomputed features | `features`, `input_features`, `feature`, `fbank`, `filterbank` |
| feature rows | `rows`, `num_frames`, `frames`, `feature_rows` |
| feature columns | `cols`, `feature_dim`, `num_features` |
| audio bytes | Hugging Face style `audio.bytes`, or `audio_bytes`, `bytes`, `wav`, `audio_data` |
| audio path | `audio_path`, `path`, `file`, `file_path` |

For Parquet audio bytes/path rows, features are extracted with the architecture
frontend. For Parquet feature rows, the features are used directly.

## Required Arguments

| Argument | Required | Description |
| --- | --- | --- |
| `--vocab-size <N>` | Yes, unless `--tokenizer` | Vocabulary size including the blank token. When omitted, it is inferred from `--tokenizer`. |
| `--train-manifest <PATH>` | Yes, unless `--manifest-dir` | File, folder, Parquet folder, or raw-audio folder. |
| `--val-manifest <PATH>` | No | Validation file or folder. Aliases: `--validation-manifest`, `--valid-manifest`, `--validation-set`, `--val-set`. |
| `--manifest-dir <DIR>` | Alternative | Directory containing `train.jsonl` and optional validation manifest. |
| `--input-dim <N>` | Usually | Feature dimension. Use `80` for Squeezeformer/Zipformer/Paraformer audio extraction, `160` for W2V-BERT. |

## Core Model Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--architecture <name>` | `squeezeformer` | One of `squeezeformer`, `zipformer`, `paraformer`, `w2v-bert`. |
| `--zipformer` | false | Alias for `--architecture zipformer`. |
| `--paraformer` | false | Alias for `--architecture paraformer`. |
| `--w2v-bert` | false | Alias for `--architecture w2v-bert`. |
| `--variant <name>` | unset | Size preset such as `xs`, `s`, `sm`, `m`, `ml`, `l` when supported. |
| `--blank-id <N>` | `0` | CTC blank token id. |
| `--d-model <N>` | `256` | Model dimension for custom/default configs. |
| `--num-layers <N>` | `16` | Number of encoder layers for custom/default configs. |
| `--num-heads <N>` | `4` | Number of attention heads for custom/default configs. |

## Batching And Streaming

The loader streams records instead of loading the whole dataset into memory.

| Argument | Default | Description |
| --- | --- | --- |
| `--batch-size <N>` | `8` | Maximum fixed batch size, or default max samples for adaptive batching. |
| `--adaptive-batch-unit <UNIT>` | unset | Enables adaptive batching. Units: `samples`, `frames`, `padded-frames`, `feature-values`, `duration-ms`, `padded-duration-ms`. |
| `--adaptive-batch-budget <N>` | unset | Budget measured in the selected unit. Must be set with `--adaptive-batch-unit`. |
| `--adaptive-batch-max-samples <N>` | `--batch-size` | Hard cap on samples per adaptive batch. |
| `--max-audio-duration-sec <F>` | unset | Drop training and validation samples longer than this duration before batching. |
| `--sort-by-length-desc` | false | Sort records by descending length within a bounded buffer. Useful for largest batches first. |
| `--sort-buffer-size <N>` | `4096` | Metadata records to hold while sorting. |
| `--dataset-index-dir <DIR>` | unset | Cache row offsets/lengths for sorted streaming. Requires `--sort-by-length-desc`. |

For large GPU runs, start with `feature-values`:

```bash
--adaptive-batch-unit feature-values \
--adaptive-batch-budget 40000 \
--adaptive-batch-max-samples 2
```

For audio-backed manifests, `padded-duration-ms` often tracks GPU memory more
directly because it budgets `batch_size * longest_sample_duration` after
padding:

```bash
--adaptive-batch-unit padded-duration-ms \
--adaptive-batch-budget 7000 \
--adaptive-batch-max-samples 2
```

For `f32`, start conservatively and increase the budget until GPU memory is
close to full without OOM. `bf16`/`f16` can usually use a larger budget.

## Augmentation

SpecAugment applies to feature tensors during training only. Waveform
augmentation applies before feature extraction for audio inputs.

| Argument | Default | Description |
| --- | --- | --- |
| `--spec-time-masks <N>` | `0` | Number of time masks. |
| `--spec-time-mask-max-frames <N>` | `0` | Maximum time-mask width. Must be > 0 if time masks are enabled. |
| `--spec-freq-masks <N>` | `0` | Number of frequency masks. |
| `--spec-freq-mask-max-bins <N>` | `0` | Maximum frequency-mask width. Must be > 0 if frequency masks are enabled. |
| `--waveform-gain-min-db <DB>` | unset | Minimum random gain for audio inputs. |
| `--waveform-gain-max-db <DB>` | unset | Maximum random gain for audio inputs. |
| `--waveform-noise-std <F>` | `0` | Uniform noise amplitude for audio inputs. |

Example:

```bash
--spec-time-masks 2 \
--spec-time-mask-max-frames 40 \
--spec-freq-masks 2 \
--spec-freq-mask-max-bins 12 \
--waveform-gain-min-db -3 \
--waveform-gain-max-db 3 \
--waveform-noise-std 0.003
```

## Optimizer And Schedule

The trainer uses AdamW.

| Argument | Default | Description |
| --- | --- | --- |
| `--epochs <N>` | `500` | Number of epochs. |
| `--learning-rate <F>` | architecture recipe | Peak AdamW learning rate. Defaults: Squeezeformer `min(variant_peak_lr, 0.0003)`, Zipformer/Paraformer `0.001`, W2V-BERT `0.0001`. |
| `--weight-decay <F>` | `0.0005` | AdamW weight decay. |
| `--lr-warmup-epochs <N>` / `--warmup-epochs <N>` | `20` | Linear warmup epochs. |
| `--lr-hold-epochs <N>` / `--hold-epochs <N>` | `160` | Hold peak LR after warmup. |
| `--lr-decay-exponent <F>` / `--decay-exponent <F>` | `1.0` | Inverse epoch-decay exponent after warmup/hold. |
| `--lr-warmup-steps <N>` | unset | Linear warmup steps. Step schedule overrides epoch schedule when any step schedule option is set. |
| `--lr-hold-steps <N>` | unset | Hold peak LR after step warmup. |
| `--lr-decay-steps <N>` | unset | Linear decay steps after step warmup/hold. |
| `--lr-min <F>` | `0` | Final LR after decay. |
| `--gradient-accumulation-steps <N>` | `1` | Number of micro-batches per optimizer step. |
| `--gradient-clip-norm <F>` | unset | Clip gradients by L2 norm. Mutually exclusive with value clipping. |
| `--gradient-clip-value <F>` | unset | Clip gradient values elementwise. Mutually exclusive with norm clipping. |
| `--ema-decay <F>` | unset | Enable EMA model tracking, for example `0.9999`. |
| `--ema-start-step <N>` | `0` | First optimizer step to update EMA. |

## Validation And Decoding

| Argument | Default | Description |
| --- | --- | --- |
| `--val-manifest <PATH>` | unset | Validation dataset. |
| `--validate-every-steps <N>` | unset | Run validation every N optimizer steps. Epoch-end validation also runs when validation data exists. |
| `--max-val-samples <N>` | unset | Limit validation samples. |
| `--tokenizer <PATH>` | unset | SentencePiece tokenizer for transcript-to-token conversion, CER/WER text decoding, and automatic `--vocab-size` inference. |
| `--val-beam-width <N>` | `1` | CTC beam width. `1` means greedy. |
| `--val-n-best <N>` | `--val-beam-width` | Number of hypotheses to keep before optional LM reranking. |
| `--val-lm-path <PATH>` | unset | KenLM model for validation decoding. Requires `--tokenizer`. |
| `--val-lm-weight <F>` | `0.45` | LM shallow-fusion weight. |
| `--val-lm-word-bonus <F>` | `0` | Word insertion bonus. |
| `--val-lm-no-bos` | false | Disable beginning-of-sentence LM context. |
| `--val-lm-no-eos` | false | Disable end-of-sentence LM context. |
| `--val-log-samples <N>` | `0` | Include sample predictions in structured validation events. |

## Paraformer Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--paraformer-alignment-mode <MODE>` | `viterbi` | Decoder-query alignment: `viterbi`, `uniform`, or `greedy`. |
| `--paraformer-enhanced` | false | Use enhanced Paraformer-v2 with shallow CTC, boundary, and refinement heads. |

## W2V-BERT Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--w2v-hf-model-dir <PATH>` | unset | Local Hugging Face W2V-BERT directory or `config.json`. |
| `--w2v-hf-load-weights` | false | Import compatible `.safetensors` weights from the HF directory. |
| `--w2v-activation-checkpointing` | false | Use Burn balanced autodiff checkpointing. |

For W2V-BERT audio extraction, use `--input-dim 160` unless you changed the
frontend config in code.

## Backend, Devices, And Precision

| Argument | Default | Description |
| --- | --- | --- |
| `--backend <BACKEND>` | `cpu` | `cpu`, `cuda`, or `wgpu`. |
| `--device-index <N>` | `0` | Single CUDA/WGPU device index. |
| `--device-indices <LIST>` | empty | Comma-separated devices for replicated multi-GPU training, for example `0,1`. |
| `--precision <P>` | `f32` | `f32`, `f16`, or `bf16`. |
| `--mixed-precision` | false | Shortcut for `--precision f16` when precision was not otherwise set. |

Multi-GPU uses replicated data-parallel training inside the Rust process. Use
`CUDA_VISIBLE_DEVICES` if you want to remap visible GPU ids:

```bash
CUDA_VISIBLE_DEVICES=2,3 cargo run --release --features burn-cuda-backend --bin train -- \
  --backend cuda \
  --device-indices 0,1 \
  ...
```

Inside the process, those visible GPUs are addressed as `0,1`.

## Checkpoints, Resume, And Logging

| Argument | Default | Description |
| --- | --- | --- |
| `--output-dir <DIR>` | `runs/burn` | Run directory. |
| `--init-from <PATH>` | unset | Warm-start model weights from a Burn checkpoint/export or PositiveLoss `.safetensors` file without resuming optimizer state. |
| `--resume-from <PATH>` | unset | Resume from checkpoint directory or `checkpoint.json`. |
| `--log-every <N>` | `10` | Log every N optimizer steps. |
| `--dry-run` | false | Forward/loss only; skip optimizer updates. Useful for smoke tests. |
| `--max-train-samples <N>` | unset | Limit training samples. Useful for smoke tests. |

When using `--init-from` with your own tokenizer, set `--tokenizer` to that
SentencePiece model. If its vocabulary size differs from the warm-start
checkpoint, compatible encoder weights are loaded and incompatible output-head
tensors are skipped.
| `--tui` | false | Open a live terminal UI showing batch extraction/loading, throughput, training loss, and validation metrics. |
| `--hf-upload-checkpoints` | false | Upload `checkpoint_latest/` and `checkpoint_latest.json` after each checkpoint save. |
| `--hf-upload-repo-id <ID>` | unset | Hugging Face model repository for checkpoint uploads. Required with `--hf-upload-checkpoints`. |
| `--hf-upload-revision <REV>` | unset | Optional branch/revision for checkpoint uploads. |
| `--hf-upload-private` | false | Create/use a private Hugging Face model repo. |
| `--hf-upload-checkpoint-format <FORMAT>` | `burn-bin` | Accepted values: `burn-bin`, `safetensors`. Rust training currently supports `burn-bin` only. |

The trainer writes:

```text
<output-dir>/training_config.json
<output-dir>/events.jsonl
<output-dir>/checkpoint_latest/
<output-dir>/checkpoint_latest.json
```

Resume validates model-shape and training-critical config before loading.

## Example Commands

### Squeezeformer From Parquet On Two CUDA GPUs

```bash
RUST_LOG=info CUDA_VISIBLE_DEVICES=0,1 cargo run --release --features burn-cuda-backend --bin train -- \
  --train-manifest testdata \
  --validation-manifest testdata/validation \
  --architecture squeezeformer \
  --variant xs \
  --backend cuda \
  --device-indices 0,1 \
  --precision f32 \
  --input-dim 80 \
  --vocab-size <VOCAB_SIZE> \
  --batch-size 2 \
  --adaptive-batch-unit feature-values \
  --adaptive-batch-budget 40000 \
  --adaptive-batch-max-samples 2 \
  --sort-by-length-desc \
  --dataset-index-dir runs/index-cache \
  --spec-time-masks 2 \
  --spec-time-mask-max-frames 40 \
  --spec-freq-masks 2 \
  --spec-freq-mask-max-bins 12 \
  --epochs 20 \
  --validate-every-steps 500 \
  --max-val-samples 2048 \
  --learning-rate 0.001 \
  --lr-warmup-steps 1000 \
  --lr-decay-steps 20000 \
  --gradient-clip-norm 5.0 \
  --log-every 1 \
  --ema-decay 0.9999 \
  --output-dir runs/squeezeformer-parquet
```

### Zipformer From JSONL

```bash
cargo run --release --features burn-cuda-backend --bin train -- \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --architecture zipformer \
  --backend cuda \
  --device-index 0 \
  --input-dim 80 \
  --vocab-size <VOCAB_SIZE> \
  --tokenizer tokenizer.model \
  --batch-size 16 \
  --sort-by-length-desc \
  --dataset-index-dir runs/index-cache \
  --epochs 10 \
  --output-dir runs/zipformer
```

### Enhanced Paraformer

```bash
cargo run --release --features burn-cuda-backend --bin train -- \
  --train-manifest data/train.parquet \
  --architecture paraformer \
  --paraformer-enhanced \
  --paraformer-alignment-mode viterbi \
  --backend cuda \
  --input-dim 80 \
  --vocab-size <VOCAB_SIZE> \
  --batch-size 8 \
  --epochs 10 \
  --output-dir runs/paraformer-enhanced
```

### W2V-BERT With Hugging Face Config

```bash
cargo run --release --features burn-cuda-backend --bin train -- \
  --train-manifest data/train.parquet \
  --architecture w2v-bert \
  --backend cuda \
  --precision bf16 \
  --input-dim 160 \
  --vocab-size <VOCAB_SIZE> \
  --w2v-hf-model-dir /path/to/w2v-bert \
  --w2v-hf-load-weights \
  --w2v-activation-checkpointing \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --epochs 5 \
  --output-dir runs/w2v-bert
```

### Resume

```bash
cargo run --release --features burn-cuda-backend --bin train -- \
  --resume-from runs/squeezeformer-parquet/checkpoint_latest \
  --train-manifest testdata \
  --architecture squeezeformer \
  --backend cuda \
  --device-indices 0,1 \
  --input-dim 80 \
  --vocab-size <VOCAB_SIZE> \
  --output-dir runs/squeezeformer-parquet
```

Keep model-shape and training-critical flags the same when resuming.

## Troubleshooting

### `record has feature dim X, expected Y`

Set `--input-dim` to match the data:

- Squeezeformer/Zipformer/Paraformer audio extraction: `80`
- W2V-BERT audio extraction: `160`
- Precomputed features: whatever `cols`/feature width your manifest contains

### `record needs tokenizer_path to derive tokens`

Your record has text but no token ids. Add:

```bash
--tokenizer tokenizer.model
```

or include `tokens`/`target`/`targets` in the dataset.

### CUDA backend not available

Build with:

```bash
--features burn-cuda-backend
```

and run with:

```bash
--backend cuda
```

### BF16/F16 unsupported

Switch precision:

```bash
--precision f32
```

or:

```bash
--mixed-precision
```

### GPU out of memory

Reduce one or more of:

```bash
--batch-size
--adaptive-batch-budget
--adaptive-batch-max-samples
```

For W2V-BERT, also consider:

```bash
--w2v-activation-checkpointing
--gradient-accumulation-steps 2
```
