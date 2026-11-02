SHELL := /bin/bash
.DEFAULT_GOAL := help
MODELS ?= both
MODEL ?= parakeet
GPU ?= 0
IMAGE ?= dual-model-asr:local
NAMESPACE ?= asr
PVC_SIZE ?= 50Gi
PARAKEET_BATCH_SIZE ?= 8
PARAKEET_BEAM ?= 6
PARAKEET_PRECISION ?= fp16
PARAKEET_OPTIMIZATION_LEVEL ?= 3
INDIC_CPU_THREADS ?= 4
BATCH_SIZE ?= 1
TRIALS ?= 5
OUTPUT ?= results/$(MODEL)-b$(BATCH_SIZE)
INDIC_RUNTIME ?= official
LANGUAGE ?= hi
ARTIFACT ?=
DECODER ?=
export ASR_ENABLED_MODELS := $(MODELS)
export CUDA_VISIBLE_DEVICES := $(GPU)
export PARAKEET_BATCH_SIZE PARAKEET_BEAM PARAKEET_PRECISION PARAKEET_OPTIMIZATION_LEVEL INDIC_CPU_THREADS

.PHONY: help setup login doctor download prepare run transcribe benchmark report image push docker-run k8s-render deploy status port-forward test check
help:
	@printf '%s\n' 'make setup [MODELS=both|parakeet|indicconformer]' 'make login                  Hugging Face authentication' 'make run [MODELS=...]        Download, prepare and serve' 'make transcribe MODEL=... AUDIO=audio.wav' 'make benchmark MODEL=... MANIFEST=data/test.jsonl' 'make report                 Summarize your measurements' 'make image IMAGE=registry/image:tag' 'make docker-run [MODELS=...] [GPU=0]' 'make deploy IMAGE=registry/image:tag [MODELS=...]' 'make port-forward           Access Kubernetes on localhost' 'make test                   CPU contract tests'
setup:
	bash scripts/setup.sh
login:
	.venv/bin/hf auth login
doctor:
	.venv/bin/python -m scripts.doctor --models '$(MODELS)'
download:
	.venv/bin/python -m scripts.download_models --models '$(MODELS)'
prepare:
	@bash -c 'source scripts/env.sh; exec .venv/bin/python -m scripts.prepare --models "$(MODELS)"'
run:
	bash scripts/serve.sh
transcribe:
	@test -n '$(AUDIO)' || { echo 'Set AUDIO to a WAV or FLAC file'; exit 1; }
	python3 scripts/transcribe.py --model '$(MODEL)' --audio '$(AUDIO)' $(if $(filter indicconformer,$(MODEL)),--language '$(LANGUAGE)',)
benchmark:
	@test -n '$(MANIFEST)' || { echo 'Set MANIFEST to a referenced JSONL manifest'; exit 1; }
	@bash -c 'source scripts/env.sh; exec .venv/bin/python -m scripts.run_benchmark --model "$(MODEL)" --manifest "$(MANIFEST)" --batch-size "$(BATCH_SIZE)" --trials "$(TRIALS)" --language "$(LANGUAGE)" --indic-runtime "$(INDIC_RUNTIME)" --output "$(OUTPUT)" $(if $(ARTIFACT),--artifact "$(ARTIFACT)",) $(if $(DECODER),--decoder "$(DECODER)",)'
report:
	python3 -m scripts.report --input results --output results/REPORT.md
image:
	docker build -f docker/Dockerfile -t '$(IMAGE)' .
push:
	docker push '$(IMAGE)'
docker-run:
	python3 -m scripts.docker_run --image '$(IMAGE)' --models '$(MODELS)' --gpu '$(GPU)'
k8s-render:
	@mkdir -p build
	.venv/bin/python -m scripts.kubernetes render --image '$(IMAGE)' --models '$(MODELS)' --namespace '$(NAMESPACE)' --storage '$(PVC_SIZE)' > build/kubernetes.yaml
deploy:
	.venv/bin/python -m scripts.kubernetes deploy --image '$(IMAGE)' --models '$(MODELS)' --namespace '$(NAMESPACE)' --storage '$(PVC_SIZE)'
status:
	kubectl rollout status deployment/asr -n '$(NAMESPACE)' --timeout=60m
port-forward:
	kubectl port-forward -n '$(NAMESPACE)' service/asr 8000:8000
test:
	PYTHONPATH='$(CURDIR)' .venv/bin/python -m unittest discover -s tests -v
check:
	python3 -m scripts.check_source
