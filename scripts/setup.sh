#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'The GPU server requires Linux x86-64. Use make setup-client for API/Kubernetes control tools.'
  exit 1
fi
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
if [[ ! -f upstream/fast-gpu-asr/pyproject.toml || ! -f upstream/TensorRT-fastconformer/infer_fastconformer_trt.py ]]; then
  git submodule update --init --recursive
fi
models="${ASR_ENABLED_MODELS:-parakeet,indicconformer}"
if [[ "$models" == both ]]; then models=parakeet,indicconformer; fi
case "$models" in parakeet|indicconformer|parakeet,indicconformer|indicconformer,parakeet) ;; *) echo 'Invalid model selection'; exit 1;; esac
bash scripts/setup_client.sh
if [[ ",$models," == *,parakeet,* ]]; then
  uv pip install --python .venv/bin/python 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv/bin/python -r requirements-parakeet.txt
  .venv/bin/python - <<'PY'
import importlib.util, shutil
from pathlib import Path
source = Path(importlib.util.find_spec('fast_gpu_asr').origin).parent / 'tensorrt_plugins'
target = Path('upstream/fast-gpu-asr/src/fast_gpu_asr/tensorrt_plugins')
for binary in source.glob('*.so'): shutil.copy2(binary, target / binary.name)
PY
fi
if [[ ",$models," == *,indicconformer,* ]]; then
  if [[ ! -x .venv-indic/bin/python ]]; then uv venv --python 3.12 .venv-indic; fi
  uv pip install --python .venv-indic/bin/python 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv-indic/bin/python -r requirements-indic.txt
fi
echo 'Setup complete. Use make login for gated model access, then make run.'
