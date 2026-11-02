# Reproduce benchmarks on your own hardware

Benchmark each model on suitable real recordings with references. Retain the exact model revision, corpus, input order, normalization, decoder and timing boundary when comparing settings.

## 1. Create a manifest

Example layout:

```text
data/
  english-test.jsonl
  audio/
    clip-001.wav
    clip-002.wav
```

Each JSONL record contains a unique audio path and its exact reference text:

```json
{"audio_filepath":"audio/clip-001.wav","text":"The actual spoken words go here."}
```

Relative paths resolve beside the manifest. Use separate language-appropriate manifests, such as English recordings for Parakeet and Hindi recordings for IndicConformer. Supply nonempty clips of at most 20 seconds. No duration field is required: timing uses audio duration measured from decoded samples. Do not duplicate clips to inflate throughput.

Possible evaluation sources include [LibriSpeech](https://www.openslr.org/12/) and [FLEURS](https://huggingface.co/datasets/google/fleurs); follow their own terms and test-split definitions. This repository includes a format example, not prerecorded benchmark results.

## 2. Run Parakeet

```bash
make prepare MODELS=parakeet PARAKEET_BATCH_SIZE=8
make benchmark MODEL=parakeet MANIFEST=data/english-test.jsonl \
  BATCH_SIZE=8 TRIALS=5 OUTPUT=results/parakeet-b8

make prepare MODELS=parakeet PARAKEET_BATCH_SIZE=32
make benchmark MODEL=parakeet MANIFEST=data/english-test.jsonl \
  BATCH_SIZE=32 TRIALS=5 OUTPUT=results/parakeet-b32
```

Export capacity is fixed. A smaller inference batch on a larger engine still uses that engine's capacity. The last partial batch counts only real input audio. Engine building is excluded from inference time.

## 3. Run IndicConformer

```bash
make prepare MODELS=indicconformer
make benchmark MODEL=indicconformer MANIFEST=data/hindi-test.jsonl \
  LANGUAGE=hi DECODER=ctc BATCH_SIZE=1 OUTPUT=results/indic-ctc
make benchmark MODEL=indicconformer MANIFEST=data/hindi-test.jsonl \
  LANGUAGE=hi DECODER=rnnt BATCH_SIZE=1 OUTPUT=results/indic-rnnt
```

The official wrapper has capacity 1. For batched native CTC, follow the [native CTC build steps](indicconformer.md#4-optional-native-tensorrt-ctc) and pass `INDIC_RUNTIME=trt` plus its language-specific `ARTIFACT` path. Include CPU frontend resources in your comparison.

## 4. Report results

```bash
make report
.venv/bin/python -m scripts.report --input results \
  --baseline results/parakeet-b8 --output results/PARAKEET_COMPARISON.md
```

Each offline run performs three first-batch warmups plus a full untimed corpus pass, followed by five measured trials by default. A run stores:

- `config.json`: model/backend, options, manifest hash, normalization and timing definition.
- `trials.csv` and `summary.json`: raw trials and medians.
- `predictions-trialN.jsonl`: every reference and prediction.
- `batches-trialN.json`: real audio duration, batch count and synchronized wall time.

RTFx is real audio seconds divided by waveform-to-text wall seconds. Files are decoded before timing; the frontend, inference, transfers and text decoding are included. Loading, warmups, engine building and scoring are excluded. WER/CER use NFC, casefolding, punctuation/symbol replacement and whitespace collapse; Indic combining marks are preserved.

Signed WER/CER deltas are reported only for a matching model and manifest hash. A changed manifest order changes its hash: use the same manifest for a precision/decoder comparison. Retain raw trial ranges, errors and empty-output counts. Treat a fastest tested configuration as an observed result, not a universal optimum or a serving latency guarantee.

## 5. Optional HTTP concurrency

With the server running:

```bash
.venv/bin/python -m scripts.benchmark_http --model parakeet \
  --manifest data/english-test.jsonl --concurrency 1 8 32 --trials 5 \
  --output results/parakeet-http

.venv/bin/python -m scripts.benchmark_http --model indicconformer --language hi --decoder ctc \
  --manifest data/hindi-test.jsonl --concurrency 1 8 32 --trials 5 \
  --output results/indic-http
```

These metrics include client upload, server decoding/resampling, queueing, inference and response. A worker serializes model calls. Concurrency creates a queue rather than automatic batching. Request p99 and whole-batch p99 measure different boundaries.
