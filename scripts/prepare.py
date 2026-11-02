#!/usr/bin/env python3
"""Prepare selected artifacts on the GPU that will perform inference."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import importlib.metadata

from scripts.download_models import download
from scripts.project import ROOT, artifacts, atomic_json, model_config, selected_models


def engine_settings(batch, beam, precision, optimization_level=3):
    import cupy as cp
    import tensorrt as trt
    properties = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = properties["name"]
    if isinstance(gpu_name, bytes): gpu_name = gpu_name.decode()
    plugins = ROOT / "upstream/fast-gpu-asr/src/fast_gpu_asr/tensorrt_plugins"
    plugin_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(plugins.glob("*.so"))}
    return {"gpu_model": gpu_name, "compute_capability": [properties["major"], properties["minor"]],
        "multiprocessors": properties["multiProcessorCount"], "tensorrt": trt.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(), "cuda_driver_api": cp.cuda.runtime.driverGetVersion(),
        "native_plugins": plugin_hashes, "fast_gpu_asr_release": importlib.metadata.version("fast-gpu-asr"),
        "upstream_commit": "10411b53603b1bb2b9bd81008ed155049e5f074a",
        "model_revision": model_config()["parakeet"]["revision"],
        "batch_size": batch, "beam": beam, "precision": precision, "optimization_level": optimization_level, "max_audio_seconds": 20}


def prepare(names, batch=8, beam=6, precision="fp16", force=False, optimization_level=3):
    if batch < 1 or beam < 1: raise ValueError("Batch size and beam must be positive")
    download(names)
    selection = {}
    if "parakeet" in names:
        settings = engine_settings(batch, beam, precision, optimization_level)
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        destination = artifacts() / "engines" / f"parakeet-b{batch}-{key}"
        marker = destination / "ready.json"
        if force or not marker.exists() or any(not (destination / p).is_file() for p in ["parakeet.trt", "tdt_decoder.trt"]):
            destination.mkdir(parents=True, exist_ok=True)
            marker.unlink(missing_ok=True)
            checkpoint = artifacts() / "models/parakeet" / model_config()["parakeet"]["filename"]
            command = [sys.executable, "-m", "fast_gpu_asr.export.export_parakeet",
                "--model-path", str(checkpoint), "--output-dir", str(destination),
                "--batch-size", str(batch), "--beam", str(beam), "--decoder-type",
                "transducer_greedy_search" if beam == 1 else "transducer_modified_beam_search",
                "--encoder-precision", precision, "--decoder-precision", precision,
                "--min-audio-seconds", "0.1", "--opt-audio-seconds", "8", "--max-audio-seconds", "20",
                "--optimization-level", str(optimization_level)]
            subprocess.run(command, cwd=ROOT, check=True)
            atomic_json(marker, settings)
        selection["parakeet"] = str(destination.relative_to(artifacts()))
    if "indicconformer" in names:
        selection["indicconformer"] = "models/indicconformer"
    atomic_json(artifacts() / "runtime.json", selection)
    print("Selected artifacts are ready.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=os.environ.get("ASR_ENABLED_MODELS", "both"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("PARAKEET_BATCH_SIZE", "8")))
    parser.add_argument("--beam", type=int, default=int(os.environ.get("PARAKEET_BEAM", "6")))
    parser.add_argument("--precision", choices=["fp16", "bf16"], default=os.environ.get("PARAKEET_PRECISION", "fp16"))
    parser.add_argument("--optimization-level", type=int, choices=range(6), default=int(os.environ.get("PARAKEET_OPTIMIZATION_LEVEL", "3")))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(selected_models(args.models), args.batch_size, args.beam, args.precision,
        force=args.force, optimization_level=args.optimization_level)
