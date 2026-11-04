"""One entrypoint for the three published benchmark configurations."""
import argparse
import hashlib
import json
import os
import subprocess

from scripts.project import ROOT, artifacts, atomic_json, model_config

PYTHON = str(ROOT / ".venv/bin/python")
INDIC_PYTHON = str(ROOT / ".venv-indic/bin/python")


def run(python, module, *args, env=None):
    subprocess.run([python, "-m", module, *map(str, args)], cwd=ROOT, check=True, env=env)


def benchmark(selection):
    native = selection == "indicconformer"
    model = "indicconformer" if native else "parakeet"
    python = INDIC_PYTHON if native else PYTHON
    batch = int(os.environ.get("BENCH_BATCH_SIZE", "32" if native else "256"))
    if batch < 1 or (native and batch > 128):
        raise ValueError("Batch must be positive; the Indic engine supports at most 128")
    run(python, "scripts.prepare_corpus", model)
    manifest = artifacts() / "data" / ("fleurs-hi.jsonl" if native else "librispeech.jsonl")
    if native:
        run(PYTHON, "scripts.download_models", "--models", model)
        # Query compatibility in a short-lived process; it releases its CUDA
        # context before export or measurement starts.
        settings = json.loads(subprocess.check_output([PYTHON, "-c",
            "import json; from scripts.prepare import engine_settings; "
            "print(json.dumps(engine_settings(128, 1, 'fp32', 3)))"], cwd=ROOT, text=True))
        settings.update(model_revision=model_config()[model]["revision"],
            upstream_commit="93586a8d949d364f26c600e2f5213a9cdccf69aa",
            language="hi", profile={"max_batch": 128, "opt_batch": 32, "opt_frames": 1000, "max_frames": 2016})
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        artifact = artifacts() / "engines" / f"indic-hi-fp32-{key}"
        marker = artifact / "ready.json"
        if not marker.exists() or not all((artifact / name).exists() for name in ["model.plan", "settings.json"]):
            marker.unlink(missing_ok=True)
            run(INDIC_PYTHON, "scripts.build_indic_ctc", "--model", artifacts() / "models/indicconformer",
                "--output", artifact, "--language", "hi", "--precision", "fp32", "--max-batch", 128)
            atomic_json(marker, settings)
    else:
        beam, level = (6, 5) if selection == "parakeet-beam" else (1, 3)
        run(PYTHON, "scripts.prepare", "--models", model, "--batch-size", batch,
            "--beam", beam, "--precision", "fp16", "--optimization-level", level)
        artifact = artifacts() / json.loads((artifacts() / "runtime.json").read_text())[model]
    output = artifacts() / "results" / selection
    command = ["--model", model, "--artifact", artifact, "--manifest", manifest,
        "--batch-size", batch, "--trials", 5, "--warmup", 3, "--warmup-passes", 1,
        "--decoder", "ctc" if native else "tdt", "--language", "hi", "--output", output]
    if native:
        command += ["--indic-runtime", "trt", "--indic-model-path", artifacts() / "models/indicconformer",
                    "--indic-max-batch", 128]
    env = dict(os.environ)
    env.setdefault("INDIC_CPU_THREADS", "32")
    run(python, "scripts.benchmark", *command, env=env)
    result = json.loads((output / "summary.json").read_text())
    print(f"\n{selection}: {result['rtfx']:.2f} RTFx | WER {result['wer_pct']:.4f}% | "
          f"CER {result['cer_pct']:.4f}% | empty {result['empty_outputs']}\n"
          f"Saved: {output}\n", flush=True)
    if result["empty_outputs"]:
        raise RuntimeError("Empty predictions detected; do not accept this throughput result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["all", "parakeet", "parakeet-beam", "indicconformer"], default="all", nargs="?")
    args = parser.parse_args()
    artifacts().mkdir(parents=True, exist_ok=True)
    # Serialize runs sharing the GPU and engine cache, including separate execs.
    import fcntl
    with (artifacts() / ".benchmark.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in (["parakeet", "parakeet-beam", "indicconformer"] if args.model == "all" else [args.model]):
            benchmark(name)
