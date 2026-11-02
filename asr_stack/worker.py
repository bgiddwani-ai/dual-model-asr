import asyncio
import base64
import binascii
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import soundfile as sf

from .audio import decode_audio
from .backends import IndicConformerBackend, ParakeetBackend


class Request(BaseModel):
    audio_base64: list[str] = Field(min_length=1, max_length=32)
    language: str | None = None
    decoder: str | None = None


@asynccontextmanager
async def lifespan(app):
    model = os.environ.get("ASR_MODEL", "parakeet")
    if model == "parakeet":
        backend = ParakeetBackend(os.environ["PARAKEET_BUNDLE"])
        # Readiness includes native decoder JIT, which needs the host compiler.
        await asyncio.to_thread(backend.transcribe, [np.zeros(16000, dtype=np.float32)], None, "tdt")
    elif model == "indicconformer":
        backend = IndicConformerBackend(
            os.environ["INDIC_MODEL_PATH"], os.environ.get("INDIC_PROVIDER", "CUDAExecutionProvider"),
        )
    else:
        raise ValueError(f"Unknown model {model}")
    app.state.backend = backend
    app.state.lock = asyncio.Lock()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def health():
    return {"ready": True, **app.state.backend.metadata}


@app.post("/transcribe")
async def transcribe(request: Request):
    backend = app.state.backend
    if len(request.audio_base64) > backend.capacity:
        raise HTTPException(422, f"This bundle accepts at most {backend.capacity} clips")
    if any(len(x) > 8_000_000 for x in request.audio_base64):
        raise HTTPException(413, "Encoded clip exceeds 8 MB")
    try:
        audios = [decode_audio(base64.b64decode(x, validate=True)) for x in request.audio_base64]
        decoder = request.decoder or ("tdt" if os.environ.get("ASR_MODEL", "parakeet") == "parakeet" else "ctc")
        async with app.state.lock:
            result = await asyncio.to_thread(backend.transcribe, audios, request.language, decoder)
        return {"results": result, "model": backend.metadata}
    except (ValueError, binascii.Error, sf.LibsndfileError) as exc:
        raise HTTPException(422, str(exc)) from exc
