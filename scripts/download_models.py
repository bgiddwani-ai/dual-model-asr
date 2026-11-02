#!/usr/bin/env python3
"""Download selected, pinned snapshots; never store credentials in artifacts."""
import argparse
import json
import os

from huggingface_hub import hf_hub_download, snapshot_download
from scripts.project import artifacts, atomic_json, model_config, selected_models


def download(names):
    config = model_config()
    for name in names:
        item = config[name]
        destination = artifacts() / "models" / name
        marker = destination / "snapshot.json"
        if marker.exists() and json.loads(marker.read_text()).get("revision") == item["revision"]:
            required = [item["filename"]] if name == "parakeet" else ["config.json", "model_onnx.py", "assets/encoder.onnx", "assets/preprocessor.ts", "assets/ctc_decoder.onnx", "assets/rnnt_decoder.onnx"]
            if all((destination / filename).is_file() for filename in required):
                print(f"Snapshot ready: {name}", flush=True)
                continue
        destination.mkdir(parents=True, exist_ok=True)
        if name == "parakeet":
            hf_hub_download(item["repo_id"], item["filename"], revision=item["revision"], local_dir=destination)
        else:
            snapshot_download(item["repo_id"], revision=item["revision"], local_dir=destination)
        atomic_json(marker, item)
        print(f"Downloaded pinned snapshot: {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=os.environ.get("ASR_ENABLED_MODELS", "both"))
    args = parser.parse_args()
    download(selected_models(args.models))
