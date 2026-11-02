#!/usr/bin/env python3
"""Queue-inclusive HTTP latency, using one real utterance per request."""
import argparse
import asyncio
import base64
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import httpx
import jiwer
import numpy as np

from scripts.benchmark import normalize
from asr_stack.audio import decode_audio


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", choices=["parakeet", "indicconformer"], required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--language", default=None)
    p.add_argument("--decoder", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.trials < 1 or not args.concurrency or min(args.concurrency) < 1:
        p.error("trials and concurrency must be positive")
    manifest = Path(args.manifest)
    records = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    if not records or any(not r.get("text", "").strip() for r in records):
        raise ValueError("Referenced, nonempty manifests are required")
    for record in records:
        audio_path = Path(record["audio_filepath"])
        if not audio_path.is_absolute(): audio_path = manifest.parent / audio_path
        record["audio_filepath"] = str(audio_path.resolve())
    if len({r["audio_filepath"] for r in records}) != len(records):
        raise ValueError("Duplicate audio paths")
    durations = [len(decode_audio(Path(r["audio_filepath"]).read_bytes()))/16000 for r in records]
    payloads = [{"audio_base64": [base64.b64encode(Path(x["audio_filepath"]).read_bytes()).decode()],
                 "language": args.language, "decoder": args.decoder} for x in records]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    async with httpx.AsyncClient(timeout=180, limits=httpx.Limits(max_connections=max(args.concurrency))) as client:
        response = await client.get(args.url + "/healthz")
        response.raise_for_status()
        (output / "config.json").write_text(json.dumps({
            **vars(args), "health": response.json(),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "timing": "client HTTP; FLAC/WAV upload, server decode/resample, worker queue, inference, response; payloads preloaded",
        }, indent=2))
        for _ in range(3):
            response = await client.post(f"{args.url}/v1/transcribe/{args.model}", json=payloads[0])
            response.raise_for_status()
        for concurrency in args.concurrency:
            for trial in range(1, args.trials + 1):
                semaphore = asyncio.Semaphore(concurrency)

                async def request(index):
                    async with semaphore:
                        start = time.perf_counter()
                        response = await client.post(f"{args.url}/v1/transcribe/{args.model}", json=payloads[index])
                        elapsed = time.perf_counter() - start
                        response.raise_for_status()
                        result = response.json()["results"]
                        if len(result) != 1:
                            raise RuntimeError("Response count mismatch")
                        return {"index": index, "latency_ms": elapsed * 1000,
                                "reference": records[index]["text"], **result[0]}

                start = time.perf_counter()
                predictions = await asyncio.gather(*(request(i) for i in range(len(records))))
                wall = time.perf_counter() - start
                latency = [x["latency_ms"] for x in predictions]
                refs = [normalize(x["reference"]) for x in predictions]
                hyps = [normalize(x["text"]) for x in predictions]
                audio = sum(durations)
                row = {"model": args.model, "concurrency": concurrency, "trial": trial,
                       "requests": len(records), "audio_seconds": audio, "wall_seconds": wall,
                       "rtfx": audio / wall, "request_p50_ms": float(np.percentile(latency, 50)),
                       "request_p95_ms": float(np.percentile(latency, 95)),
                       "request_p99_ms": float(np.percentile(latency, 99)),
                       "wer_pct": jiwer.wer(refs, hyps) * 100, "cer_pct": jiwer.cer(refs, hyps) * 100,
                       "empty_outputs": sum(not x for x in hyps)}
                rows.append(row)
                (output / f"predictions-c{concurrency}-trial{trial}.jsonl").write_text("".join(
                    json.dumps(x, ensure_ascii=False) + "\n" for x in predictions))
                print(json.dumps(row), flush=True)
    with (output / "trials.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for concurrency in args.concurrency:
        group = [x for x in rows if x["concurrency"] == concurrency]
        summary = {k: statistics.median([x[k] for x in group])
                   for k in group[0] if isinstance(group[0][k], (int, float)) and k != "trial"}
        summary.update(model=args.model, trials=args.trials)
        summaries.append(summary)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
