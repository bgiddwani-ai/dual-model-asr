# Dual Model ASR

[![Validation](https://github.com/bgiddwani-ai/dual-model-asr/actions/workflows/ci.yaml/badge.svg)](https://github.com/bgiddwani-ai/dual-model-asr/actions/workflows/ci.yaml)

Run **Parakeet-TDT-0.6B-v3** and **IndicConformer-600M** from one repository, together or independently. Use a Linux GPU machine, Docker, or a Kubernetes cluster with NVIDIA GPUs.

| Model | Default runtime | Individual guide |
|---|---|---|
| [NVIDIA Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | TensorRT encoder and GPU TDT decoding | [Parakeet setup](docs/parakeet.md) |
| [AI4Bharat IndicConformer-600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | Official ONNX CUDA wrapper, CTC or RNNT | [IndicConformer setup](docs/indicconformer.md) |

## Quick start

Server requirements: **Linux x86-64**, Python 3.12, `make`, `git`, a C++ compiler, libsndfile, [uv](https://docs.astral.sh/uv/getting-started/installation/), and a supported NVIDIA GPU/driver. The pinned Parakeet stack uses CUDA 13; upstream specifies **driver 580 or newer**, with newer drivers potentially needed for PTX features. Choose GPU memory and batch sizes for your workload. The HTTP client works on any device with Python 3.

```bash
git clone --recurse-submodules https://github.com/bgiddwani-ai/dual-model-asr.git
cd dual-model-asr
make setup
```

For IndicConformer, first accept the [publisher's access conditions](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual), then authenticate:

```bash
make login
make run
```

`make run` downloads the pinned snapshots, builds Parakeet engines on your selected GPU, and starts both models on port 8000. The first launch includes download/export time. In another terminal:

```bash
curl --fail http://127.0.0.1:8000/healthz
python3 scripts/transcribe.py --model parakeet --audio data/english.wav
python3 scripts/transcribe.py --model indicconformer --audio data/hindi.wav --language hi
```

Supply your own WAV or FLAC files in `data/`. Input is downmixed and resampled to 16 kHz; each clip must be nonempty and at most 20 seconds. Stop the local server with Ctrl+C.

## Choose one model

| Task | Command |
|---|---|
| Set up Parakeet only | `make setup MODELS=parakeet` |
| Run Parakeet only | `make run MODELS=parakeet` |
| Set up IndicConformer only | `make setup MODELS=indicconformer` |
| Run IndicConformer only | `make run MODELS=indicconformer` |
| Choose a local GPU | `make run GPU=1` |
| Change Parakeet batch capacity | `make run PARAKEET_BATCH_SIZE=32` |
| Use greedy Parakeet decoding | `make run PARAKEET_BEAM=1` |
| Check server prerequisites | `make doctor MODELS=both` |

Defaults are both models, GPU ordinal 0, Parakeet batch capacity 8 and beam 6. Settings and quality checks are explained in the individual guides.

## Docker and Kubernetes

Use [the deployment guide](docs/deployment.md) for registry, storage, runtime and authentication settings.

```bash
# Docker on a GPU host
make image IMAGE=dual-model-asr:local
make docker-run IMAGE=dual-model-asr:local MODELS=both GPU=0

# Kubernetes: use an image reference reachable by your nodes
make image IMAGE=registry.example.com/team/dual-model-asr:v1
make push IMAGE=registry.example.com/team/dual-model-asr:v1
make deploy IMAGE=registry.example.com/team/dual-model-asr:v1 MODELS=both
make port-forward
```

Kubernetes requests one GPU for the enabled models. Engines are prepared **inside the serving container on its assigned GPU**. Engine cache keys include GPU and runtime compatibility settings; reusable model snapshots live on the PVC. The YAML has no fixed GPU product, node, storage class, cloud provider, or external endpoint.

## Benchmarks

Provide a JSONL manifest containing real audio paths and exact reference transcripts. Relative audio paths resolve beside the manifest. See [benchmarking](docs/benchmarking.md) for separate model commands, native Indic CTC, concurrency, and signed WER/CER comparisons.

```bash
make benchmark MODEL=parakeet MANIFEST=data/english-test.jsonl BATCH_SIZE=8
make benchmark MODEL=indicconformer MANIFEST=data/hindi-test.jsonl LANGUAGE=hi
make report
```

Results are generated on your own device under `results/`. This repository does not bundle machine-specific reports, model weights, engines, or credentials.

## Exact upstream sources

| Submodule | Pinned commit | Role |
|---|---|---|
| [SoundsGoodAI/fast-gpu-asr](https://github.com/SoundsGoodAI/fast-gpu-asr) | `10411b53603b1bb2b9bd81008ed155049e5f074a` | Parakeet export, TensorRT plugins and GPU TDT inference |
| [bgiddwani-ai/TensorRT-fastconformer](https://github.com/bgiddwani-ai/TensorRT-fastconformer) | `93586a8d949d364f26c600e2f5213a9cdccf69aa` | Unchanged TensorRT runner and CTC decoder definitions for optional native Indic CTC |

Upstream source files remain unchanged. An owned ONNX builder combines IndicConformer's official encoder and language-selected CTC head for the optional native path. Model snapshots are pinned in [configs/models.json](configs/models.json).

More: [API](docs/api.md), [troubleshooting](docs/troubleshooting.md), [third-party notices](THIRD_PARTY_NOTICES.md).

CI runs CPU contract tests, source checks, and Kubernetes schema checks. GPU export/inference and cluster deployment require a compatible target environment and real-audio verification.
