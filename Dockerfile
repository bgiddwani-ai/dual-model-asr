FROM nvidia/cuda:13.4.3-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libsndfile1 libgomp1 g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/asr
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs bash scripts/setup.sh
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    ASR_ARTIFACTS_DIR=/artifacts \
    HF_HOME=/artifacts/cache/huggingface
ENTRYPOINT ["bash", "run.sh"]
CMD ["all"]
