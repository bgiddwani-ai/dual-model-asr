import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

WORKERS = {
    "parakeet": "http://127.0.0.1:9001",
    "indicconformer": "http://127.0.0.1:9002",
}
ENABLED = set(os.environ.get("ASR_ENABLED_MODELS", "parakeet,indicconformer").split(","))


@asynccontextmanager
async def lifespan(app):
    app.state.client = httpx.AsyncClient(timeout=180.0)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/livez")
async def live():
    return {"alive": True}


@app.get("/healthz")
async def health():
    models = {}
    for name in sorted(ENABLED):
        try:
            response = await app.state.client.get(f"{WORKERS[name]}/healthz", timeout=3)
            response.raise_for_status()
            models[name] = response.json()
        except (httpx.HTTPError, ValueError):
            raise HTTPException(503, f"{name} is not ready") from None
    return {"ready": True, "models": models}


@app.post("/v1/transcribe/{model}")
async def transcribe(model: str, request: Request):
    if model not in ENABLED:
        raise HTTPException(404, "Model is not enabled")
    # Starlette streams the body so oversized requests cannot allocate unbounded memory.
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 32_000_000:
            raise HTTPException(413, "Request exceeds 32 MB")
        chunks.append(chunk)
    try:
        response = await app.state.client.post(
            f"{WORKERS[model]}/transcribe", content=b"".join(chunks),
            headers={"Content-Type": "application/json"},
        )
    except httpx.HTTPError:
        raise HTTPException(503, "ASR worker unavailable") from None
    try:
        payload = response.json()
    except ValueError:
        raise HTTPException(502, "ASR worker returned an invalid response") from None
    return JSONResponse(status_code=response.status_code, content=payload)
