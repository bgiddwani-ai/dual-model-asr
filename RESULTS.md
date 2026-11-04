# Measured benchmark results

Measured on **one NVIDIA H100 80GB**. These are observed results from the completed benchmark campaign, not estimates or a new run of the simplified container. The commands in the README recreate the same model revisions, audio selection, engine settings and timing procedure on your GPU. Hardware, driver, CPU allocation and engine tactic selection can change the result.

## Best tested configurations

All figures below are medians of **five trials**. RTFx = actual input audio seconds / synchronized waveform-to-text seconds; higher is faster.

| Configuration | Batch | CPU threads¹ | Median RTFx | Trial range | WER | CER | Empty outputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parakeet FP16, greedy, builder level 3 | 256 | — | **14,240.02** | 14,178.17–14,297.93 | 2.1585% | 0.6389% | 0 |
| Parakeet FP16, beam 6, builder level 5 | 256 | — | **12,345.09** | 12,291.42–12,376.85 | 2.0745% | 0.6122% | 0 |
| IndicConformer Hindi CTC, native TensorRT FP32/TF32 | 32 | 32 | **4,222.21** | 4,027.01–4,327.93 | 12.2476% | 4.7734% | 0 |

¹ TorchScript frontend threads for Indic; this setting does not apply to Parakeet's GPU frontend. The deployment reserves 32 CPUs.

The two README commands run the fastest tested setting for each model. To reproduce the Parakeet beam-6 row, run:

```bash
kubectl exec asr -- bash run.sh parakeet-beam
```

These are offline batched results. HTTP overhead, network latency and request queueing are outside the measurement. The two models use different corpora, so their RTFx and WER are not direct model-quality comparisons.

## Audio and measurement

| Model | Public test data | Recordings | Actual audio per trial | Frozen order |
|---|---|---:|---:|---|
| Parakeet | [LibriSpeech test-clean](https://www.openslr.org/12), clips ≤20 seconds | 2,528 | 17,252.785625 s | [Manifest](benchmarks/librispeech.jsonl) |
| IndicConformer | [FLEURS Hindi test](https://huggingface.co/datasets/google/fleurs), seed 42, clips 0.1–20 seconds | 256 | 2,782.5 s | [Manifest](benchmarks/fleurs-hi.jsonl) |

Both manifests use portable relative audio paths and ascending duration. The Hindi subset has 256 distinct audio recordings and 195 sentence IDs; some sentences have multiple speakers. Dataset downloads are checksum-verified. The preparation helper checks the seeded Hindi selection against the frozen manifest.

Each optimized configuration runs three first-batch warmups and one complete untimed corpus pass, followed by five trials. Audio files are decoded before timing. Timed work includes waveform padding, frontend, transfers, model execution and decoding through synchronized text output. Model downloads, export/build, disk audio decoding, warmups and WER/CER scoring are excluded. The final partial batch counts only real audio. WER/CER normalization: NFC, casefold, Unicode punctuation/symbols to spaces, collapse whitespace.

Parakeet uses FP16 encoder and decoder engines; profile audio limits are 0.1/8/20 seconds. Indic uses the official CPU TorchScript frontend, FP32 weights with TF32 allowed, the Hindi-selected CTC head, and the unchanged pinned TensorRT runner and CTC decoder. Its engine profile has min/opt/max batch 1/32/128, opt 1,000 frames and max 2,016 frames. This result is **CTC**, with no full-corpus RNNT throughput claim.

Key runtime versions: Python 3.12.16, CPU Torch 2.15.0, fast-gpu-asr 0.2.0, CUDA Toolkit 13.4.3, TensorRT 11.4.0.106, NumPy 2.5.4, ONNX Runtime 1.31.1, cuDNN 9.27.0.42 and PyCUDA 2026.1. The container includes the native Indic dependencies and builds engines on the selected device.

## Quality and tested limits

| Same-corpus reference | Reference WER / CER | Candidate | Signed WER / CER change |
|---|---:|---|---:|
| Parakeet FP16 beam 6, batch 32, builder level 3 | 2.0619% / 0.6114% | Fastest greedy | **+0.09659 / +0.02750 pp** |
| Same Parakeet reference | 2.0619% / 0.6114% | Best beam 6 | **+0.01260 / +0.00080 pp** |
| Indic official CUDA CTC, batch 1 | 12.1972% / 4.7598% | Native FP32/TF32, batch 32, 32 threads | **+0.05047 / +0.01354 pp** |

The official Indic reference measured 37.39 RTFx with three first-utterance warmups and **no full-corpus warm pass**. Its timing is not an identically warmed speedup baseline.

We tested **45 completed configurations / 225 trials**: 10 Parakeet and 35 Indic configurations. Parakeet batch 512 was slower than 256. Indic batch 16 / 16 threads reached 4,219.18 RTFx, essentially tied with the observed best at batch 32 / 32 threads. Seven completed Indic FP16 configurations produced 100% WER/CER and all-empty outputs; they are marked rejected and excluded from best selection. FP32/TF32 is the provided native Indic configuration. The observed best is limited to this tested workload and sweep.

## Recorded evidence

- [Every configuration: settings, medians, quality and trial range](benchmarks/results/summary.csv)
- [All 225 individual measured trials](benchmarks/results/trials.csv)

The CSVs contain numeric benchmark evidence and generic configuration names. Model weights, audio, serialized engines and credentials are downloaded or generated separately. Your new run saves its own per-trial predictions, batch timings, configuration and summary in `/artifacts/results`.

CI checks source, input handling, Kubernetes schemas, frozen corpus identities and recomputes published medians from the individual trials. The [clean Linux container build and runtime import checks](https://github.com/bgiddwani-ai/dual-model-asr/actions/runs/41760192220) passed. GPU inference was not rerun for this packaging update.
