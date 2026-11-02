FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libsndfile1 libgomp1 g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/asr
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv bash scripts/setup.sh
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    ASR_ARTIFACTS_DIR=/artifacts \
    HF_HOME=/cache/huggingface \
    ASR_AUTO_PREPARE=1
EXPOSE 8000
CMD ["bash", "scripts/serve.sh"]
