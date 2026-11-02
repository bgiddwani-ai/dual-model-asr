#!/usr/bin/env python3
"""Run the image with a selected GPU and a project-relative artifact mount."""
import argparse
import getpass
import os
import subprocess
from scripts.project import artifacts, selected_models

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--models", default="both")
parser.add_argument("--gpu", default="0")
args = parser.parse_args()
names = selected_models(args.models)
env = dict(os.environ)
if "indicconformer" in names and not env.get("HF_TOKEN"):
    try:
        from huggingface_hub import get_token
        env["HF_TOKEN"] = get_token() or ""
    except ImportError: pass
    if not env.get("HF_TOKEN"): env["HF_TOKEN"] = getpass.getpass("Hugging Face read token: ")
artifacts().mkdir(parents=True, exist_ok=True)
command = ["docker", "run", "--rm", "--gpus", f"device={args.gpu}", "--shm-size=4g", "-p", "8000:8000",
    "-v", f"{artifacts()}:/artifacts", "-e", f"ASR_ENABLED_MODELS={','.join(names)}"]
if "indicconformer" in names: command += ["-e", "HF_TOKEN"]
for key in ["PARAKEET_BATCH_SIZE", "PARAKEET_BEAM", "PARAKEET_PRECISION", "PARAKEET_OPTIMIZATION_LEVEL", "INDIC_CPU_THREADS", "ASR_REBUILD"]:
    if key in env: command += ["-e", key]
subprocess.run(command + [args.image], env=env, check=True)
