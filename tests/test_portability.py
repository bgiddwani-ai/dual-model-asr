"""CPU checks for deployment portability and benchmark accounting."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from scripts.kubernetes import render
from scripts.project import selected_models


class Portability(unittest.TestCase):
    def test_model_selection(self):
        self.assertEqual(selected_models("both"), ["parakeet", "indicconformer"])
        self.assertEqual(selected_models("parakeet,parakeet"), ["parakeet"])
        for invalid in ["", "unknown", "parakeet,unknown"]:
            with self.assertRaises(ValueError): selected_models(invalid)

    def test_each_deployment_uses_one_assigned_gpu_and_generic_platform_labels(self):
        for models in ["both", "parakeet", "indicconformer"]:
            resources = render("registry.example.com/team/asr:v1", models)
            pod = next(r for r in resources if r["kind"] == "Deployment")["spec"]["template"]["spec"]
            self.assertEqual(pod["nodeSelector"], {"kubernetes.io/os": "linux", "kubernetes.io/arch": "amd64"})
            self.assertNotIn("initContainers", pod)
            container = pod["containers"][0]
            self.assertEqual(container["resources"]["limits"]["nvidia.com/gpu"], "1")
            env = {e["name"]: e for e in container["env"]}
            self.assertEqual(env["ASR_AUTO_PREPARE"]["value"], "1")
            self.assertEqual(env["ASR_ENABLED_MODELS"]["value"], ",".join(selected_models(models)))
            self.assertEqual("HF_TOKEN" in env, "indicconformer" in selected_models(models))

    def test_namespace_registry_storage_and_runtime_are_configurable(self):
        resources = render("registry.example.com/project/asr:v2", "indicconformer", "speech", "100Gi", "gpu", "pull-access")
        self.assertEqual(resources[0]["metadata"]["name"], "speech")
        for resource in resources[1:]: self.assertEqual(resource["metadata"]["namespace"], "speech")
        pvc = next(r for r in resources if r["kind"] == "PersistentVolumeClaim")
        self.assertEqual(pvc["spec"]["resources"]["requests"]["storage"], "100Gi")
        pod = next(r for r in resources if r["kind"] == "Deployment")["spec"]["template"]["spec"]
        self.assertEqual(pod["runtimeClassName"], "gpu")
        self.assertEqual(pod["imagePullSecrets"], [{"name": "pull-access"}])
        self.assertEqual(pod["containers"][0]["image"], "registry.example.com/project/asr:v2")

    def test_manifest_moves_with_audio_and_partial_batch_counts_real_duration(self):
        from scripts import benchmark
        class FakeBackend:
            capacity = 2
            metadata = {"backend": "test fixture"}
            def transcribe(self, audios, language, decoder):
                return [{"text": "hello", "words": []} for _ in audios]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audio").mkdir()
            records = []
            for i, samples in enumerate([3200, 4800, 6400]):
                sf.write(root / f"audio/{i}.wav", np.zeros(samples), 16000)
                records.append({"audio_filepath": f"audio/{i}.wav", "text": "hello", "duration": 999999})
            manifest = root / "test.jsonl"
            manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
            output = root / "output"
            arguments = ["benchmark", "--model", "parakeet", "--artifact", "unused", "--manifest", str(manifest),
                "--batch-size", "2", "--trials", "1", "--warmup", "1", "--output", str(output)]
            with patch("sys.argv", arguments), patch.object(benchmark, "ParakeetBackend", return_value=FakeBackend()), contextlib.redirect_stdout(io.StringIO()):
                benchmark.main()
            summary = json.loads((output / "summary.json").read_text())
            self.assertAlmostEqual(summary["audio_seconds"], 0.9)
            self.assertEqual(summary["utterances"], 3)
            batches = json.loads((output / "batches-trial1.json").read_text())
            self.assertEqual([b["count"] for b in batches], [2, 1])
            predictions = [json.loads(x) for x in (output / "predictions-trial1.jsonl").read_text().splitlines()]
            self.assertEqual({Path(p["audio_filepath"]).parent for p in predictions}, {(root / "audio").resolve()})

    def test_engine_cache_reuses_matching_target_and_rebuilds_for_changed_target(self):
        from scripts import prepare
        def fake_export(command, **kwargs):
            destination = Path(command[command.index("--output-dir") + 1])
            for name in ["parakeet.trt", "tdt_decoder.trt"]:
                (destination / name).write_bytes(b"test fixture")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(prepare, "artifacts", return_value=root), patch.object(prepare, "download"), \
                    patch.object(prepare, "engine_settings") as settings, \
                    patch.object(prepare.subprocess, "run", side_effect=fake_export) as exporter, \
                    contextlib.redirect_stdout(io.StringIO()):
                settings.return_value = {"gpu_model": "target-a", "tensorrt": "runtime-a"}
                prepare.prepare(["parakeet"])
                first = json.loads((root / "runtime.json").read_text())["parakeet"]
                prepare.prepare(["parakeet"])
                self.assertEqual(exporter.call_count, 1)
                settings.return_value = {"gpu_model": "target-b", "tensorrt": "runtime-a"}
                prepare.prepare(["parakeet"])
                second = json.loads((root / "runtime.json").read_text())["parakeet"]
                self.assertEqual(exporter.call_count, 2)
                self.assertNotEqual(first, second)
                self.assertFalse(Path(second).is_absolute())


if __name__ == "__main__": unittest.main()
