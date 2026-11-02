# HTTP API

The gateway listens on port 8000. Workers use private loopback ports 9001 and 9002.

| Route | Purpose |
|---|---|
| `GET /livez` | Gateway liveness |
| `GET /healthz` | Readiness and metadata for every enabled model |
| `POST /v1/transcribe/parakeet` | Parakeet TDT transcription |
| `POST /v1/transcribe/indicconformer` | Indic CTC or RNNT transcription |

```json
{
  "audio_base64": ["<base64-encoded WAV or FLAC file>"],
  "language": "hi",
  "decoder": "ctc"
}
```

For Parakeet, omit language and decoder or use `decoder="tdt"`. For IndicConformer, supply the language code and choose `ctc` or `rnnt`. Indic defaults to Hindi CTC.

Responses contain `results`, one text result per clip, and model runtime metadata. Parakeet also returns word timing information from its backend. The official Indic wrapper returns text without word timestamps through this API.

Clips must be nonempty and no longer than 20 seconds after decoding. The worker downmixes/resamples to 16 kHz. Parakeet accepts at most its engine capacity and at most 32 clips per call; the official Indic wrapper accepts one. Calls are serialized per worker. The public HTTP service has no built-in authentication: use your platform's ingress/authentication when exposing it beyond a trusted environment.

Example using only Python's standard library:

```bash
python3 scripts/transcribe.py --url http://127.0.0.1:8000 \
  --model indicconformer --audio data/hindi.wav --language hi --decoder rnnt
```
