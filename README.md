# Dual Model ASR

Reproduce **Parakeet-TDT-0.6B-v3** and **IndicConformer-600M** benchmarks with one container and one Kubernetes YAML.

Our measured results on **one NVIDIA H100 80GB**:

| Model | Median RTFx ↑ | WER ↓ | CER ↓ |
|---|---:|---:|---:|
| Parakeet — FP16, greedy, batch 256 | **14,240.02** | 2.1585% | 0.6389% |
| IndicConformer — Hindi CTC, FP32/TF32, batch 32 | **4,222.21** | 12.2476% | 4.7734% |

[Results, quality comparisons and all 225 measured trials →](RESULTS.md)

You need Docker, a container registry, `kubectl`, and a **Linux amd64 NVIDIA GPU node** with the NVIDIA device plugin and a driver compatible with CUDA 13.4. Use an **80GB GPU** for the published batch sizes; the YAML reserves **32 CPUs and 64GiB RAM**. It uses your cluster's default StorageClass; adjust resources, storage, GPU runtime class or image pull credentials for your cluster. Other hardware will produce different speeds.

## 1. Create the container

Replace the example image name with your registry. Build on any Docker host; inference runs on the GPU node.

```bash
git clone --recurse-submodules https://github.com/bgiddwani-ai/dual-model-asr.git
cd dual-model-asr
export IMAGE=registry.example.com/team/dual-model-asr:v1
docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"
```

## 2. Start the Kubernetes pod

For IndicConformer, accept the [model's access conditions](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual). Create a secret with your own Hugging Face read token; the `read` command hides input.

```bash
read -s HF_TOKEN
printf %s "$HF_TOKEN" | kubectl create secret generic hf-access --from-file=token=/dev/stdin
unset HF_TOKEN
```

In [k8s.yaml](k8s.yaml), set `image:` to the image you pushed, then:

```bash
kubectl apply -f k8s.yaml
kubectl wait --for=condition=Ready pod/asr --timeout=20m
```

## 3. Run each model

```bash
kubectl exec asr -- bash run.sh parakeet
kubectl exec asr -- bash run.sh indicconformer
```

Each command downloads the pinned model and benchmark audio, builds engines **on the assigned GPU**, and runs five measured trials after warmup. Downloads and engine builds take time on the first run. Run the commands sequentially. The terminal prints RTFx, WER and CER; predictions, timings and summaries are saved in the container's `/artifacts/results` directory.

```bash
kubectl cp asr:/artifacts/results ./results
kubectl delete pod asr
```

The PVC retains downloaded models, datasets, engines and results. Delete it with `kubectl delete pvc asr-data` when you no longer need them. On smaller GPUs, use `kubectl exec asr -- env BENCH_BATCH_SIZE=8 bash run.sh parakeet`; this changes the benchmark configuration.

The exact upstream repositories remain pinned and unchanged: [SoundsGoodAI/fast-gpu-asr](https://github.com/SoundsGoodAI/fast-gpu-asr/tree/10411b53603b1bb2b9bd81008ed155049e5f074a) for Parakeet, and [TensorRT-fastconformer](https://github.com/bgiddwani-ai/TensorRT-fastconformer/tree/93586a8d949d364f26c600e2f5213a9cdccf69aa) for the native Indic runner and CTC decoder. An owned builder combines IndicConformer's official encoder and Hindi CTC head. [Model revisions](configs/models.json) · [Third-party notices](THIRD_PARTY_NOTICES.md).
