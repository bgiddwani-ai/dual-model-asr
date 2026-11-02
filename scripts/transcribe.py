#!/usr/bin/env python3
"""Portable HTTP client; Python standard library only, no GPU required."""
import argparse
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--model", choices=["parakeet", "indicconformer"], required=True)
parser.add_argument("--audio", type=Path, required=True)
parser.add_argument("--language")
parser.add_argument("--decoder", choices=["tdt", "ctc", "rnnt"])
args = parser.parse_args()
payload = {"audio_base64": [base64.b64encode(args.audio.read_bytes()).decode()]}
if args.language: payload["language"] = args.language
if args.decoder: payload["decoder"] = args.decoder
request = Request(f"{args.url.rstrip('/')}/v1/transcribe/{args.model}",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urlopen(request, timeout=240) as response:
    print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
