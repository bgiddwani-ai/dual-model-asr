# IndicConformer-600M multilingual

The default serving path loads the official ONNX models through CUDA Execution Provider and retains the publisher's frontend, language vocabulary and predictor token IDs. CTC and RNNT are separate decoding modes.

## 1. Install and authorize

```bash
git clone --recurse-submodules https://github.com/bgiddwani-ai/dual-model-asr.git
cd dual-model-asr
make setup MODELS=indicconformer
```

Accept [the model's Hugging Face access conditions](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) using your own account, then run:

```bash
make login
make doctor MODELS=indicconformer
make download MODELS=indicconformer
```

The revision is pinned in `configs/models.json`. Its official Python wrapper is downloaded with the snapshot; review `artifacts/models/indicconformer/model_onnx.py` before using it. The adapter instantiates that local class with explicit CUDA sessions. CPU PyTorch is intentional: the ONNX sessions perform GPU inference while the frontend runs on CPU.

## 2. Start and transcribe

```bash
make run MODELS=indicconformer GPU=0 INDIC_CPU_THREADS=4
```

In a second terminal:

```bash
# Hindi CTC
python3 scripts/transcribe.py --model indicconformer --audio data/hindi.wav --language hi --decoder ctc

# Hindi RNNT
python3 scripts/transcribe.py --model indicconformer --audio data/hindi.wav --language hi --decoder rnnt
```

The official wrapper accepts one clip per request. Choose a language code supported by the [publisher](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual), such as `hi`, `ta`, `te`, `bn`, `kn`, `ml`, or `mr`. Benchmark each language and decoder independently. This repository is not a multilingual accuracy certification.

## 3. Benchmark the official path

```bash
make prepare MODELS=indicconformer
make benchmark MODEL=indicconformer MANIFEST=data/hindi-test.jsonl LANGUAGE=hi DECODER=ctc OUTPUT=results/indic-ctc
make benchmark MODEL=indicconformer MANIFEST=data/hindi-test.jsonl LANGUAGE=hi DECODER=rnnt OUTPUT=results/indic-rnnt
```

Use correct references in the selected language. Audio loading, normalization and timing are documented in [benchmarking](benchmarking.md).

## 4. Optional: native TensorRT CTC

This offline path uses FP32 encoder weights, allows TF32 execution, and selects the same language rows as the official CTC mask. The original `TensorRT-fastconformer` runner and greedy decoder definitions execute unchanged. Build one head per language and verify WER/CER against the official path.

```bash
uv pip install --python .venv-indic/bin/python -r requirements-indic-benchmark.txt
.venv-indic/bin/python -m scripts.build_indic_ctc \
  --model artifacts/models/indicconformer \
  --output artifacts/engines/indic-hi-ctc \
  --language hi --max-batch 32

make benchmark MODEL=indicconformer INDIC_RUNTIME=trt \
  ARTIFACT=artifacts/engines/indic-hi-ctc \
  MANIFEST=data/hindi-test.jsonl LANGUAGE=hi BATCH_SIZE=32 \
  OUTPUT=results/indic-native-b32

.venv/bin/python -m scripts.report --input results \
  --baseline results/indic-ctc --output results/COMPARISON.md
```

Install the CUDA toolkit/development headers required to build PyCUDA if a compatible wheel is unavailable. Rebuild native plans for each target GPU/runtime. `INDIC_CPU_THREADS` tunes the timed CPU frontend. Increase CPU resources accordingly. The HTTP service continues using the official all-language wrapper; this optional native path is an offline CTC benchmark.
