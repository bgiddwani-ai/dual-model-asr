"""Shared repository-relative configuration. No provider or host assumptions."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("parakeet", "indicconformer")


def selected_models(value):
    selected = list(MODELS) if value == "both" else list(dict.fromkeys(value.split(",")))
    if not selected or any(name not in MODELS for name in selected):
        raise ValueError("Models must be both, parakeet, indicconformer, or parakeet,indicconformer")
    return selected


def artifacts():
    return Path(os.environ.get("ASR_ARTIFACTS_DIR", ROOT / "artifacts")).expanduser().resolve()


def model_config():
    return json.loads((ROOT / "configs/models.json").read_text())


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
