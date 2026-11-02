# Docker and Kubernetes deployment

All paths below are project-relative or container paths. Choose your registry, GPU ordinal, namespace and storage settings. The GPU server requires a compatible Linux x86-64 NVIDIA stack; clients can run elsewhere. Use `make setup-client` for CPU-only deployment tools on a Linux/macOS control machine (or WSL); it does not install either GPU model environment.

## Docker

Install Docker, a compatible NVIDIA driver, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). GPU access must work in containers before starting this application.

```bash
make image IMAGE=dual-model-asr:local
make docker-run IMAGE=dual-model-asr:local MODELS=both GPU=0
```

The run helper mounts `./artifacts` at `/artifacts`, maps port 8000, and selects one host GPU. For IndicConformer it uses `HF_TOKEN`, a locally installed Hugging Face client's cached token, or a hidden token prompt. Credentials are passed through the process environment and are not written into source or build files. Accept the publisher's access conditions first.

Select `MODELS=parakeet` or `MODELS=indicconformer` to run one model. On first start the container downloads the selected snapshots and prepares any required Parakeet engines. `/healthz` becomes ready after the enabled workers start.

## Kubernetes prerequisites

The current kubectl context must target your intended cluster. It needs:

- GPU nodes with compatible drivers and [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) or GPU Operator exposing `nvidia.com/gpu`.
- Container runtime configured for NVIDIA GPU access.
- A working default StorageClass, or a storage class added to `k8s/base/pvc.yaml`.
- Sufficient GPU memory, CPU, RAM and persistent storage for your selected models and batch sizes.
- Network access for initial model downloads and an image registry reachable by the nodes.

The deployment requests one GPU, starts one pod with a `Recreate` strategy, and uses a ReadWriteOnce PVC. Standard Linux/AMD64 platform labels keep the image on compatible nodes; there is no fixed GPU product or host selection. The resource defaults are configurable deployment starting points. TensorRT engines are built in the **same serving container** before inference, so a preparation job cannot accidentally build for a different node's GPU.

Install the control tools on the machine running kubectl, for every deployment mode:

```bash
make setup-client
```

## 1. Build and push your image

Replace the example registry/image reference with your own. Authenticate to that registry with `docker login` before pushing.

```bash
make image IMAGE=registry.example.com/team/dual-model-asr:v1
make push IMAGE=registry.example.com/team/dual-model-asr:v1
```

Replace the example reference with your registry/repository and a versioned tag. `make image` targets Linux AMD64 explicitly; ARM development machines need a Docker builder with cross-platform support, or a remote AMD64 builder. No GPU is needed during image creation: export happens on the assigned serving GPU. A local development cluster may instead load the image into its nodes using that cluster's image-loading command.

## 2. Authenticate for IndicConformer

On the machine running deployment commands:

```bash
make login
```

During deployment, an existing Secret named `huggingface-access` is reused. If it is absent, the helper creates it from your local Hugging Face token through kubectl stdin. For managed secrets, create this Secret yourself with key `token` before deploying. Parakeet-only deployment does not require it.

## 3. Render and deploy

```bash
make k8s-render IMAGE=registry.example.com/team/dual-model-asr:v1 \
  MODELS=both NAMESPACE=asr PVC_SIZE=50Gi

# Inspect build/kubernetes.yaml, then deploy the same settings.
make deploy IMAGE=registry.example.com/team/dual-model-asr:v1 \
  MODELS=both NAMESPACE=asr PVC_SIZE=50Gi

make port-forward NAMESPACE=asr
```

Individual model deployments use the same commands with `MODELS=parakeet` or `MODELS=indicconformer`. The helper creates the namespace, applies storage/service/deployment, and waits up to 60 minutes for startup. Initial downloads and export run before the server starts.

Optional cluster settings:

```bash
# Only if your cluster requires a named NVIDIA RuntimeClass:
RUNTIME_CLASS=nvidia make deploy IMAGE=registry.example.com/team/dual-model-asr:v1

# Only if your image requires an existing registry pull Secret:
IMAGE_PULL_SECRET=registry-access make deploy IMAGE=registry.example.com/team/dual-model-asr:v1
```

For GitOps, commit your rendered YAML to your own deployment repository and manage model credentials separately. Do not commit generated Secrets. `k8s/kustomization.yaml` also provides a standard base; model-specific Kustomize overlays are under `k8s/overlays/`.

## 4. Verify

```bash
kubectl get pods,pvc -n asr
kubectl logs -n asr deployment/asr
curl --fail http://127.0.0.1:8000/healthz
python3 scripts/transcribe.py --model indicconformer --audio data/hindi.wav --language hi
```

Every enabled model must pass readiness and a real-audio transcription check. YAML validation alone does not establish working GPU inference. GPU ordinal 0 inside a pod refers to the assigned device; normal device-plugin allocation does not select a particular physical host index.

## Storage and tuning

Snapshots are reusable across GPUs. Parakeet engine cache directories are selected by compatibility metadata. Do not copy a TensorRT plan across unmatched GPU/runtime combinations. Set `ASR_REBUILD=1` for a forced local rebuild; in Kubernetes add this environment variable to the Deployment and recreate the pod.

Change resource requests/limits, startup timeout and other persistent settings in `k8s/base/deployment.yaml`, then render/deploy again. Batch size, beam, precision, builder level and CPU threads can also be supplied as the same Make variables used for local runs. If you edit `build/kubernetes.yaml` directly, apply that file with `kubectl apply -f build/kubernetes.yaml`; `make deploy` renders anew from the base.

Keep one active server per artifact root. Keep the default single replica for the ReadWriteOnce design; scaling requires an explicit artifact/storage and GPU-allocation strategy.
