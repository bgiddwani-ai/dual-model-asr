#!/usr/bin/env python3
"""Build a Markdown report only from the user's completed local measurements."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/REPORT.md"))
    parser.add_argument("--baseline", type=Path, help="Run directory used for signed WER/CER deltas")
    args = parser.parse_args()
    rows = []
    baseline = None
    if args.baseline:
        baseline = (json.loads((args.baseline / "summary.json").read_text()),
                    json.loads((args.baseline / "config.json").read_text()))
    for path in sorted(args.input.rglob("summary.json")):
        config = json.loads((path.parent / "config.json").read_text())
        summaries = json.loads(path.read_text())
        for row in summaries if isinstance(summaries, list) else [summaries]:
            delta = (None, None)
            if baseline and row["model"] == baseline[0]["model"] and config["manifest_sha256"] == baseline[1]["manifest_sha256"]:
                delta = (row["wer_pct"] - baseline[0]["wer_pct"], row["cer_pct"] - baseline[0]["cer_pct"])
            rows.append((path.parent.name, row, delta))
    if not rows: raise SystemExit("No completed measurements. Run make benchmark first.")
    text = ["# ASR benchmark report", "", "Only measurements generated from the supplied manifests are included.",
        "RTFx = actual input audio seconds / measured wall seconds. Offline timing includes frontend, inference and decoding; file loading, warmup, engine build and scoring are excluded. HTTP timing includes transfer, server decoding and queueing.",
        "Different models or corpora are not matched speedup comparisons. Batch latency is not request latency. Large batches have few latency samples per trial.", "",
        "| Run | Model | Batch or concurrency | Trials | Median RTFx | WER % | CER % | Empty | WER delta pp | CER delta pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row, delta in rows:
        d = ["n/a" if value is None else f"{value:+.5f}" for value in delta]
        text.append(f"| {name} | {row['model']} | {int(row.get('batch_size', row.get('concurrency', 1)))} | {row['trials']} | {row['rtfx']:.2f} | {row['wer_pct']:.4f} | {row['cer_pct']:.4f} | {int(row['empty_outputs'])} | {d[0]} | {d[1]} |")
    text += ["", "Signed deltas are shown only when model and manifest hash match the selected baseline. Inspect quality changes before changing serving settings. Empty-output candidates must not be selected solely for throughput.",
        "Every run retains its configuration, raw trials and predictions alongside its summary.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(text))
    print(args.output)


if __name__ == "__main__": main()
