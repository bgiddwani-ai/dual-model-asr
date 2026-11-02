#!/usr/bin/env bash
# Source this file from a script after changing to the repository root.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ASR_ENABLED_MODELS="${ASR_ENABLED_MODELS:-parakeet,indicconformer}"
export ASR_ARTIFACTS_DIR="${ASR_ARTIFACTS_DIR:-$PWD/artifacts}"
export PYTHONPATH="$PWD/upstream/fast-gpu-asr/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMEGACONF_MAX_YAML_EXPANDED_NODES=1000000
