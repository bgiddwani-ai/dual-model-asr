#!/usr/bin/env python3
"""Render portable resources or deploy using the current kubectl context."""
import argparse
import json
import os
import re
import subprocess
import sys

import yaml

from scripts.project import ROOT, selected_models


def render(image, models="both", namespace="asr", storage="50Gi", runtime_class=None, pull_secret=None):
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", namespace):
        raise ValueError("Namespace must be a DNS label")
    if not image.strip() or any(c.isspace() for c in image): raise ValueError("Invalid image reference")
    if not re.fullmatch(r"[1-9][0-9]*(?:Gi|Ti|Mi)", storage): raise ValueError("Use a positive storage quantity such as 50Gi")
    names = selected_models(models)
    resources = [yaml.safe_load((ROOT / "k8s/base" / name).read_text()) for name in
                 ["namespace.yaml", "pvc.yaml", "deployment.yaml", "service.yaml"]]
    for item in resources:
        if item["kind"] == "Namespace": item["metadata"]["name"] = namespace
        else: item["metadata"]["namespace"] = namespace
        if item["kind"] == "PersistentVolumeClaim": item["spec"]["resources"]["requests"]["storage"] = storage
        if item["kind"] == "Deployment":
            pod = item["spec"]["template"]["spec"]
            container = pod["containers"][0]
            container["image"] = image
            next(e for e in container["env"] if e["name"] == "ASR_ENABLED_MODELS")["value"] = ",".join(names)
            for key in ["PARAKEET_BATCH_SIZE", "PARAKEET_BEAM", "PARAKEET_PRECISION", "PARAKEET_OPTIMIZATION_LEVEL", "INDIC_CPU_THREADS", "ASR_REBUILD"]:
                if key in os.environ:
                    container["env"].append({"name": key, "value": os.environ[key]})
            if "indicconformer" not in names:
                container["env"] = [e for e in container["env"] if e["name"] != "HF_TOKEN"]
            if runtime_class: pod["runtimeClassName"] = runtime_class
            if pull_secret: pod["imagePullSecrets"] = [{"name": pull_secret}]
    return resources


def kubectl(*args, input=None, capture=False):
    return subprocess.run(["kubectl", *args], input=input, text=True, check=True,
        stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None)


def deploy(resources, names, namespace):
    kubectl("apply", "-f", "-", input=yaml.safe_dump(resources[0]))
    if "indicconformer" in names:
        existing = subprocess.run(["kubectl", "get", "secret", "huggingface-access", "-n", namespace],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if existing.returncode:
            from huggingface_hub import get_token
            token = get_token()
            if not token: raise RuntimeError("Run make login or create the huggingface-access Secret before deployment")
            secret = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "huggingface-access", "namespace": namespace},
                      "type": "Opaque", "stringData": {"token": token}}
            kubectl("apply", "-f", "-", input=json.dumps(secret))
    kubectl("apply", "-f", "-", input=yaml.safe_dump_all(resources[1:], sort_keys=False))
    kubectl("rollout", "status", "deployment/asr", "-n", namespace, "--timeout=60m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["render", "deploy"])
    parser.add_argument("--image", required=True)
    parser.add_argument("--models", default="both")
    parser.add_argument("--namespace", default="asr")
    parser.add_argument("--storage", default="50Gi")
    parser.add_argument("--runtime-class", default=os.environ.get("RUNTIME_CLASS"))
    parser.add_argument("--pull-secret", default=os.environ.get("IMAGE_PULL_SECRET"))
    args = parser.parse_args()
    resources = render(args.image, args.models, args.namespace, args.storage, args.runtime_class, args.pull_secret)
    if args.action == "render": yaml.safe_dump_all(resources, sys.stdout, sort_keys=False)
    else: deploy(resources, selected_models(args.models), args.namespace)


if __name__ == "__main__": main()
