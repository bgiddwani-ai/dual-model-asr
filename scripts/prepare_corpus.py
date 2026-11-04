"""Download public benchmark audio and keep the exact published record order."""
import argparse
import hashlib
import io
import json
import random
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlopen

import soundfile as sf

from scripts.project import ROOT, artifacts

LIBRI_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRI_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
FLEURS_URL = "https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/hi_in/test/0000.parquet"
FLEURS_SHA256 = "8a0be26c5584c1478233a6f47fb90275c713f72565327ea1da76c9f1cd5dd38b"


def fetch(url, target, algorithm, expected):
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".partial")
        print(f"Downloading {target.name}", flush=True)
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(target)
    with target.open("rb") as handle:
        digest = hashlib.file_digest(handle, algorithm).hexdigest()
    if digest != expected:
        raise ValueError(f"Checksum mismatch: {target.name}; remove it and retry")


def fleurs_records(parquet, directory):
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet).to_pylist()
    random.Random(42).shuffle(rows)
    records = []
    directory.mkdir(parents=True, exist_ok=True)
    for row in rows:
        signal, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        duration = len(signal) / rate
        text = row.get("transcription") or row.get("raw_transcription")
        if not text or not 0.1 <= duration <= 20:
            continue
        filename = f"{len(records):04d}.wav"
        sf.write(directory / filename, signal, rate, subtype="PCM_16")
        records.append({"audio_filepath": f"fleurs-hi/{filename}", "duration": duration,
                        "text": text, "language": "hi", "source_id": row.get("id")})
        if len(records) == 256:
            break
    return sorted(records, key=lambda row: row["duration"])


def prepare(model):
    root = artifacts() / "data"
    root.mkdir(parents=True, exist_ok=True)
    name = "librispeech.jsonl" if model == "parakeet" else "fleurs-hi.jsonl"
    frozen = ROOT / "benchmarks" / name
    expected = [json.loads(line) for line in frozen.read_text().splitlines()]
    manifest = root / name
    if model == "parakeet":
        if not all((root / row["audio_filepath"]).is_file() for row in expected):
            archive = root / "test-clean.tar.gz"
            fetch(LIBRI_URL, archive, "md5", LIBRI_MD5)
            with tarfile.open(archive) as tar:
                tar.extractall(root, filter="data")
    else:
        # Verify the exact seeded selection, including separate recordings that
        # share a FLEURS sentence ID. Sentence IDs alone are not audio IDs.
        if not manifest.exists() or manifest.read_bytes() != frozen.read_bytes() or not all((root / row["audio_filepath"]).is_file() for row in expected):
            parquet = root / "fleurs-hi-test.parquet"
            fetch(FLEURS_URL, parquet, "sha256", FLEURS_SHA256)
            actual = fleurs_records(parquet, root / "fleurs-hi")
            if actual != expected:
                raise ValueError("FLEURS selection differs from the published benchmark")
    total = 0.0
    for row in expected:
        info = sf.info(root / row["audio_filepath"])
        duration = info.frames / info.samplerate
        if abs(duration - row["duration"]) > 0.000051:
            raise ValueError(f"Audio duration differs for {row['audio_filepath']}")
        total += duration
    shutil.copyfile(frozen, manifest)
    print(f"Corpus ready: {len(expected)} recordings, {total:.6f} audio seconds", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["parakeet", "indicconformer"])
    print(prepare(parser.parse_args().model))
