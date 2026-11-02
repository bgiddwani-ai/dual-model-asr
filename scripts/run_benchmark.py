#!/usr/bin/env python3
"""Resolve selected artifacts and invoke the appropriate isolated runtime."""
import argparse
import json
import subprocess
from pathlib import Path

from scripts.project import ROOT, artifacts

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["parakeet", "indicconformer"], required=True)
parser.add_argument("--manifest", required=True)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--trials", type=int, default=5)
parser.add_argument("--language", default="hi")
parser.add_argument("--decoder")
parser.add_argument("--artifact")
parser.add_argument("--indic-runtime", choices=["official", "trt"], default="official")
parser.add_argument("--output", required=True)
args = parser.parse_args()
if args.model == "indicconformer" and args.indic_runtime == "trt" and not args.artifact:
    parser.error("Native CTC requires --artifact pointing to a language-specific engine")
selection = json.loads((artifacts() / "runtime.json").read_text())
artifact = args.artifact or str(artifacts() / selection[args.model])
python = ROOT / (".venv-indic/bin/python" if args.model == "indicconformer" else ".venv/bin/python")
command = [str(python), "-m", "scripts.benchmark", "--model", args.model, "--artifact", artifact,
    "--manifest", args.manifest, "--batch-size", str(args.batch_size), "--trials", str(args.trials),
    "--warmup", "3", "--warmup-passes", "1", "--language", args.language, "--output", args.output,
    "--indic-runtime", args.indic_runtime, "--indic-model-path", str(artifacts() / "models/indicconformer")]
if args.decoder: command += ["--decoder", args.decoder]
if args.model == "indicconformer" and args.indic_runtime == "trt":
    settings = json.loads((Path(artifact) / "settings.json").read_text())
    command += ["--indic-max-batch", str(settings["max_batch"])]
subprocess.run(command, cwd=ROOT, check=True)
