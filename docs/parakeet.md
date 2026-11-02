# Parakeet-TDT-0.6B-v3

Follow these steps on a compatible Linux NVIDIA GPU host. For containers or a cluster, use [deployment](deployment.md) and select `MODELS=parakeet`.

## 1. Install

```bash
git clone --recurse-submodules https://github.com/bgiddwani-ai/dual-model-asr.git
cd dual-model-asr
make setup MODELS=parakeet
make doctor MODELS=parakeet
```

On Debian/Ubuntu, common host packages are `make`, `git`, `g++`, and `libsndfile1`. Install uv and a compatible NVIDIA driver first. Other Linux distributions can use their equivalent packages.

## 2. Download and prepare

```bash
make download MODELS=parakeet
make prepare MODELS=parakeet GPU=0 PARAKEET_BATCH_SIZE=8 PARAKEET_BEAM=6
```

The first command downloads the exact model revision in `configs/models.json`. The second invokes the pinned `fast-gpu-asr` exporter. The matching release wheel supplies its native plugins; source is imported from the unchanged submodule.

Paths are relative to the project by default:

| Artifact | Location |
|---|---|
| Snapshot | `artifacts/models/parakeet/` |
| Hardware-specific engines | `artifacts/engines/parakeet-b8-<cache-key>/` |
| Selected runtime paths | `artifacts/runtime.json` |

The cache key covers GPU model/architecture, TensorRT, CUDA/runtime compatibility, model revision, batch capacity, precision and beam. Build engines on each target GPU type. Set `ASR_ARTIFACTS_DIR` to move the artifact root.
Runtime selections use paths relative to that root. Advanced settings include `PARAKEET_PRECISION=bf16` on supported GPUs and `PARAKEET_OPTIMIZATION_LEVEL=5`; both create separate engine cache keys. Keep the same settings when preparing and serving.

## 3. Start and transcribe

```bash
make run MODELS=parakeet GPU=0 PARAKEET_BATCH_SIZE=8 PARAKEET_BEAM=6
```

In a second terminal:

```bash
python3 scripts/transcribe.py --model parakeet --audio data/english.wav
```

The default decoder is TDT. Language coverage follows the [official model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3). The API accepts up to the smaller of engine capacity and 32 clips per call; requests are serialized by the worker. HTTP concurrency queues requests and does not automatically create inference batches.

## 4. Tune and measure

```bash
# Build a larger fixed-capacity engine, then benchmark that capacity.
make prepare MODELS=parakeet PARAKEET_BATCH_SIZE=32
make benchmark MODEL=parakeet MANIFEST=data/english-test.jsonl BATCH_SIZE=32 OUTPUT=results/parakeet-beam6-b32

# A separately measured greedy candidate.
make prepare MODELS=parakeet PARAKEET_BATCH_SIZE=32 PARAKEET_BEAM=1
make benchmark MODEL=parakeet MANIFEST=data/english-test.jsonl BATCH_SIZE=32 OUTPUT=results/parakeet-greedy-b32
```

Keep the same evaluation recordings, references, input order and normalization. Report signed WER/CER changes before choosing a faster setting. Larger capacity, greedy search and reduced precision can affect transcripts; throughput is workload-dependent.
