# TODO

## Burn Architectures

- [x] Implement Squeezeformer architecture with Burn.
- [x] Add Squeezeformer CTC wrapper.
- [x] Add Squeezeformer transcription module.
- [x] Compare Squeezeformer against the Python reference in `/home/yehor/Work/PositiveLoss/squeezeformer-ukrainian`.
- [x] Implement Zipformer architecture with Burn.
- [x] Implement Paraformer architecture with Burn.
- [x] Implement Wav2Vec/W2V-BERT architecture with Burn.
- [x] Tighten Zipformer parity against the Python implementation, including custom normalization, balancing, and attention details.
- [x] Replace Zipformer balancer/whiten forward-compatible placeholders with Burn-compatible zero-forward gradient penalties.
- [x] Revisit Zipformer balancer/whiten against pinned Burn 0.21.0-pre.3 custom-backward APIs.
- [ ] Re-check Zipformer balancer/whiten when upgrading Burn, and replace zero-forward penalties if a public generic custom-backward hook is available.
- [x] Tighten Paraformer parity against the Python implementation, including predictor/alignment-specific training losses.
- [x] Add enhanced Paraformer-v2 shallow CTC, boundary, and refinement heads.
- [x] Tighten Wav2Vec/W2V-BERT parity against the Python implementation.
- [x] Add Hugging Face W2V-BERT weight import/config loading for Burn.
- [x] Add W2V-BERT activation checkpointing

## Training

- [x] Add Rust training CLI.
- [x] Support Squeezeformer training.
- [x] Support Zipformer training.
- [x] Support Paraformer training.
- [x] Support Wav2Vec/W2V-BERT training.
- [x] Add CTC training path for supported architectures.
- [x] Add dry-run mode for trainer smoke tests.
- [x] Add real checkpoint save/load for model weights.
- [x] Add optimizer state checkpointing.
- [x] Add resume support with config validation.
- [x] Add GPU backend/device selection.
- [x] Add mixed precision support + BF16.
- [ ] Add gradient accumulation.
- [ ] Add gradient clipping.
- [ ] Add learning-rate scheduler with warmup/hold/decay.
- [ ] Add EMA model tracking.
- [ ] Add distributed training support.
- [ ] Add multi-GPU training support.

## Data Loading

- [x] Add manifest path support.
- [x] Add manifest directory support.
- [x] Support JSONL manifests.
- [x] Support file-backed feature records.
- [x] Add streaming data loader for large datasets.
- [x] Add adaptive batching.
- [x] Add largest-batches-first sorting.
- [x] Make sorting metadata-only so large inline features are not buffered in memory.
- [ ] Add raw audio dataset loading.
- [ ] Add feature extraction from audio.
- [ ] Add tokenizer-driven transcript-to-token conversion.
- [ ] Add dataset cache/index support.
- [ ] Add SpecAugment and waveform augmentation.

## Validation And Inference

- [x] Add Squeezeformer greedy CTC transcription helper.
- [x] Add validation decoding for all architectures.
- [x] Add CER/WER metrics.
- [x] Add beam search decoding.
- [x] Add optional language-model decoding.
- [ ] Add sample prediction logging.
- [ ] Add inference/export entrypoints for all architectures.

## Experiment Ergonomics

- [x] Write training config metadata to output directory.
- [ ] Add structured run logging.
- [ ] Add Trackio or compatible experiment tracking.
- [ ] Add detailed diagnostics for losses, batch sizes, and throughput.
- [ ] Add model export packaging.
- [ ] Add Hugging Face upload support.

## Performance Optimization

- [ ] Run pprof profiling on training and inference to identify bottlenecks.
- [ ] Inline small helper functions in critical paths.
