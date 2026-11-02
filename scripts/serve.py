#!/usr/bin/env python3
"""Start selected workers and the shared HTTP gateway; clean up all children."""
import json
import os
import signal
import subprocess
import sys
import time

from scripts.project import ROOT, artifacts, model_config, selected_models


def main():
    names = selected_models(os.environ.get("ASR_ENABLED_MODELS", "both"))
    env = {**os.environ, "ASR_ENABLED_MODELS": ",".join(names)}
    if os.environ.get("ASR_AUTO_PREPARE", "1") == "1":
        command = [sys.executable, "-m", "scripts.prepare", "--models", ",".join(names)]
        if os.environ.get("ASR_REBUILD", "0") == "1": command.append("--force")
        subprocess.run(command, env=env, check=True)
    selection = json.loads((artifacts() / "runtime.json").read_text())
    if "parakeet" in names: env["PARAKEET_BUNDLE"] = str(artifacts() / selection["parakeet"])
    if "indicconformer" in names:
        env["INDIC_MODEL_PATH"] = str(artifacts() / selection["indicconformer"])
        env["INDIC_REVISION"] = model_config()["indicconformer"]["revision"]
    children = []
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for name, port in [("parakeet", 9001), ("indicconformer", 9002)]:
            if name not in names: continue
            python = ROOT / (".venv-indic/bin/python" if name == "indicconformer" else ".venv/bin/python")
            children.append(subprocess.Popen([str(python), "-m", "uvicorn", "asr_stack.worker:app",
                "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT, env={**env, "ASR_MODEL": name}))
        children.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "asr_stack.gateway:app",
            "--host", "0.0.0.0", "--port", os.environ.get("PORT", "8000")], cwd=ROOT, env=env))
        while not stopping:
            exited = [p for p in children if p.poll() is not None]
            if exited: raise SystemExit(exited[0].returncode or 1)
            time.sleep(0.25)
    finally:
        for child in children:
            if child.poll() is None: child.terminate()
        for child in children:
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__": main()
