#!/usr/bin/env python3
"""Check local server prerequisites without collecting host identifiers."""
import argparse
import platform
import shutil
import subprocess
from scripts.project import ROOT, selected_models

parser = argparse.ArgumentParser()
parser.add_argument("--models", default="both")
args = parser.parse_args()
names = selected_models(args.models)
if platform.system() != "Linux" or platform.machine() not in ["x86_64", "amd64"]:
    raise SystemExit("The GPU server requires Linux x86-64. The HTTP client works on other platforms.")
if not shutil.which("nvidia-smi"): raise SystemExit("Install a compatible NVIDIA driver first")
drivers = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True)
if any(int(line.split('.')[0]) < 580 for line in drivers.splitlines()):
    raise SystemExit("The CUDA 13 server stack requires driver 580 or newer; PTX features may require newer versions.")
for name in names:
    python = ROOT / (".venv-indic/bin/python" if name == "indicconformer" else ".venv/bin/python")
    code = "import fast_gpu_asr, tensorrt, cupy; cupy.cuda.runtime.getDeviceCount()" if name == "parakeet" else "import onnxruntime as o; o.preload_dlls(directory=''); assert 'CUDAExecutionProvider' in o.get_available_providers()"
    subprocess.run([str(python), "-c", code], check=True)
print("Server prerequisite checks passed. Model readiness is checked at /healthz after startup.")
