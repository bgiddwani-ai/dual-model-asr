# Troubleshooting

| Symptom | Check or fix |
|---|---|
| GPU server on macOS/ARM or CPU-only hardware | Run the GPU service on supported Linux x86-64 NVIDIA hardware; use the HTTP client from your current device. |
| CUDA/driver incompatibility | Run `make doctor`. Use a driver compatible with the pinned CUDA 13/TensorRT stack and its PTX requirements. |
| Indic download returns 401/403 | Accept the publisher's model terms with your own account, then run `make login` with a read token for that account. |
| Model downloads blocked in a cluster | Check the pod's network access and `huggingface-access` Secret with key `token`. |
| GPU out of memory | Reduce `PARAKEET_BATCH_SIZE`, run one model, or choose a GPU with more memory. Match Kubernetes resource settings to the workload. |
| Plan cannot deserialize after moving GPUs | Use `ASR_REBUILD=1 make run`; build on the target GPU/runtime and retain its compatibility metadata. |
| Missing native plugin | Run `make setup MODELS=parakeet` and confirm submodules were cloned recursively. |
| First request/export is slow | Initial engine export and decoder JIT are startup work. Keep the C++ host compiler available. |
| Pod Pending | Inspect `kubectl describe pod -n asr` for GPU capacity, PVC/storage topology, runtime configuration or image pull errors. |
| Native Indic language mismatch | Build a separate CTC head for the requested language and pass the matching language code. |
| A benchmark is fast but outputs are empty | Reject that candidate. Inspect predictions and WER/CER before selecting settings. |

To move data, preserve the relative manifest/audio layout. To move artifacts, set `ASR_ARTIFACTS_DIR` and rebuild incompatible engines. For each deployment, confirm `/healthz` and transcribe real audio with every enabled model.
