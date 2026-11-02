#!/usr/bin/env python3
"""Measure warmed waveform-to-text wall time and save every prediction."""
import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
import unicodedata
from pathlib import Path

import jiwer
import numpy as np

from asr_stack.audio import decode_audio
from asr_stack.backends import IndicConformerBackend, ParakeetBackend


def normalize(text):
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join("".join(
        " " if unicodedata.category(c).startswith(("P", "S")) else c for c in text
    ).split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["parakeet", "indicconformer"], required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--language", default="hi")
    p.add_argument("--decoder", default=None)
    p.add_argument("--provider", default="CUDAExecutionProvider")
    p.add_argument("--indic-runtime", choices=["official", "cuda", "trt"], default="official")
    p.add_argument("--indic-model-path", default="models/indicconformer")
    p.add_argument("--indic-max-batch", type=int, default=128)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--warmup-passes", type=int, default=0,
                   help="Untimed complete corpus passes before measured trials")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.batch_size < 1 or args.trials < 1 or args.warmup < 1 or args.warmup_passes < 0:
        p.error("batch size, trials, and warmup must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest)
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not records or len({r["audio_filepath"] for r in records}) != len(records):
        raise ValueError("Manifest must contain unique, nonempty records")
    if any(not r.get("text", "").strip() for r in records):
        raise ValueError("Quality benchmark requires references for every utterance")
    # Relative paths resolve beside the manifest, so datasets can move intact.
    for record in records:
        audio_path = Path(record["audio_filepath"])
        if not audio_path.is_absolute(): audio_path = manifest.parent / audio_path
        record["audio_filepath"] = str(audio_path.resolve())
    if len({r["audio_filepath"] for r in records}) != len(records):
        raise ValueError("Manifest contains aliases for the same audio file")
    audios = [decode_audio(Path(r["audio_filepath"]).read_bytes()) for r in records]
    durations = [len(a) / 16000 for a in audios]
    if args.model == "parakeet":
        backend = ParakeetBackend(args.artifact)
    elif args.indic_runtime == "official":
        backend = IndicConformerBackend(args.artifact, args.provider)
    else:
        from asr_stack.indic_ctc import IndicBatchedCTCBackend
        backend = IndicBatchedCTCBackend(args.indic_model_path, args.artifact,
            runtime=args.indic_runtime, max_batch=args.indic_max_batch, language=args.language)
    if args.batch_size > backend.capacity:
        raise ValueError("Requested batch exceeds exported capacity")
    decoder = args.decoder or ("tdt" if args.model == "parakeet" else "ctc")
    config = {
        **vars(args), "backend": backend.metadata, "python": platform.python_version(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "utterances": len(records), "audio_seconds": sum(durations),
        "timing": "preloaded mono float32 waveform to text; synchronized backend return; excludes disk decode, model load, warmups and scoring",
        "normalization": "NFC; casefold; Unicode punctuation/symbols to spaces; collapse whitespace",
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    for _ in range(args.warmup):
        backend.transcribe(audios[:args.batch_size], args.language, decoder)
    for _ in range(args.warmup_passes):
        for start in range(0, len(audios), args.batch_size):
            backend.transcribe(audios[start:start + args.batch_size], args.language, decoder)
    rows = []
    for trial in range(1, args.trials + 1):
        latencies, predictions, batch_rows = [], [], []
        for start in range(0, len(audios), args.batch_size):
            batch = audios[start:start + args.batch_size]
            before = time.perf_counter()
            result = backend.transcribe(batch, args.language, decoder)
            elapsed = time.perf_counter() - before
            if len(result) != len(batch):
                raise RuntimeError("Prediction count does not match input count")
            latencies.append(elapsed * 1000)
            batch_rows.append({"start": start, "count": len(batch), "elapsed_s": elapsed,
                               "audio_s": sum(durations[start:start + len(batch)])})
            for index, value in enumerate(result, start):
                predictions.append({"audio_filepath": records[index]["audio_filepath"],
                                    "reference": records[index]["text"], **value})
        refs = [normalize(row["reference"]) for row in predictions]
        hyps = [normalize(row["text"]) for row in predictions]
        elapsed = sum(row["elapsed_s"] for row in batch_rows)
        row = {
            "model": args.model, "decoder": decoder, "batch_size": args.batch_size,
            "engine_capacity": backend.capacity, "trial": trial, "utterances": len(records),
            "audio_seconds": sum(durations), "inference_seconds": elapsed,
            "rtfx": sum(durations) / elapsed,
            "batch_p50_ms": float(np.percentile(latencies, 50)),
            "batch_p95_ms": float(np.percentile(latencies, 95)),
            "batch_p99_ms": float(np.percentile(latencies, 99)),
            "wer_pct": jiwer.wer(refs, hyps) * 100,
            "cer_pct": jiwer.cer(refs, hyps) * 100,
            "empty_outputs": sum(not x.strip() for x in hyps),
        }
        if not all(np.isfinite(x) for x in row.values() if isinstance(x, (float, int))):
            raise RuntimeError("Nonfinite benchmark result")
        rows.append(row)
        (out / f"predictions-trial{trial}.jsonl").write_text("".join(
            json.dumps(value, ensure_ascii=False) + "\n" for value in predictions))
        (out / f"batches-trial{trial}.json").write_text(json.dumps(batch_rows, indent=2))
        print(json.dumps(row), flush=True)
    with (out / "trials.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: statistics.median([row[key] for row in rows])
               for key in rows[0] if isinstance(rows[0][key], (int, float)) and key != "trial"}
    summary.update(model=args.model, decoder=decoder, trials=args.trials)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
