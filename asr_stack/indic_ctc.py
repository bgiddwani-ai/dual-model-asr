"""Batched IndicConformer CTC with the publisher frontend and pinned TRT runner."""
import ast
import hashlib
import json
import os
import time
import typing
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def upstream_component(filename, name, extra=None):
    path = ROOT / "upstream/TensorRT-fastconformer" / filename
    source = path.read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    # Execute the exact upstream definition without its unused NeMo CLI imports.
    namespace = {**vars(typing), "np": np, "torch": torch, "Path": Path, "time": time}
    namespace.update(extra or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name], hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()


class IndicBatchedCTCBackend:
    def __init__(self, model_path, artifact, runtime="cuda", max_batch=128, language="hi"):
        import onnxruntime as ort
        ort.preload_dlls(directory="")
        torch.set_num_threads(int(os.environ.get("INDIC_CPU_THREADS", "4")))
        source = Path(model_path) / "assets"
        artifact = Path(artifact)
        settings = json.loads((artifact / "settings.json").read_text())
        if settings["language"] != language:
            raise ValueError("The native CTC head must be built for the requested language")
        self.frontend = torch.jit.load(str(source / "preprocessor.ts"), map_location="cpu").eval()
        self.vocab = json.loads((source / "vocab.json").read_text())[language]
        self.language = language
        self.capacity = min(max_batch, int(settings["max_batch"]))
        self.runtime = runtime
        self.decode, decode_hash = upstream_component("infer_fastconformer_trt.py", "ctc_greedy_decode")
        if runtime == "cuda":
            search = os.environ.get("INDIC_CUDNN_SEARCH", "EXHAUSTIVE")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            self.encoder = ort.InferenceSession(str(source / "encoder.onnx"), sess_options=options,
                providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": search}), "CPUExecutionProvider"])
            self.head = ort.InferenceSession(str(artifact / "ctc_head.onnx"), sess_options=options,
                providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": search}), "CPUExecutionProvider"])
            if any("CUDAExecutionProvider" not in s.get_providers() for s in [self.encoder, self.head]):
                raise RuntimeError("CUDA provider initialization failed")
        else:
            import pycuda.autoprimaryctx
            import pycuda.driver as cuda
            import tensorrt as trt
            runner, runner_hash = upstream_component("benchmark_fastconformer_encoder_trt_file.py", "EncoderRunner",
                {"cuda": cuda, "trt": trt, "TRT_LOGGER": trt.Logger(trt.Logger.WARNING)})
            self.runner = runner(str(artifact / "model.plan"), "audio_signal", "length")
            self.stream = cuda.Stream()
            self.cuda = cuda
            self.pinned_inputs = {}
        self.metadata = {"model_id": "ai4bharat/indic-conformer-600m-multilingual",
            "backend": f"batched {runtime} CTC", "capacity": self.capacity, "language": language,
            "frontend": "official TorchScript CPU, batched", "torch_threads": torch.get_num_threads(),
            "upstream_decoder_definition_sha256": decode_hash,
            "upstream_runner_definition_sha256": runner_hash if runtime != "cuda" else None,
            "artifact": str(artifact)}
        self.metadata["precision"] = json.loads((artifact / "settings.json").read_text())["precision"]
        if runtime == "cuda":
            self.metadata["cudnn_conv_algo_search"] = self.encoder.get_provider_options()["CUDAExecutionProvider"]["cudnn_conv_algo_search"]

    def transcribe(self, audios, language="hi", decoder="ctc"):
        if decoder != "ctc" or language not in [None, self.language]:
            raise ValueError("This benchmark bundle is CTC for its configured language")
        if not 0 < len(audios) <= self.capacity:
            raise ValueError("Invalid batch size")
        lengths = torch.tensor([len(audio) for audio in audios], dtype=torch.int64)
        waveforms = torch.zeros(len(audios), int(lengths.max()), dtype=torch.float32)
        for index, audio in enumerate(audios):
            waveforms[index, :len(audio)] = torch.from_numpy(audio)
        with torch.inference_mode():
            features, feature_lengths = self.frontend(input_signal=waveforms, length=lengths)
        features, feature_lengths = features.numpy(), feature_lengths.numpy()
        if self.runtime == "cuda":
            encoded, encoded_lengths = self.encoder.run(["outputs", "encoded_lengths"],
                {"audio_signal": features, "length": feature_lengths})
            logits = self.head.run(["logprobs"], {"encoder_output": encoded})[0]
        else:
            entries = [(Path(str(index)), features[index], int(feature_lengths[index]), len(audio)/16000)
                       for index, audio in enumerate(audios)]
            self.runner.prepare(entries)
            # The unchanged upstream runner retains ordinary NumPy inputs;
            # pin their staging copies for CUDA's asynchronous H2D contract.
            for name in ("_audio_np", "_lengths_np"):
                value = getattr(self.runner, name)
                staged = self.pinned_inputs.get(name)
                if staged is None or staged.shape != value.shape:
                    staged = self.cuda.pagelocked_empty(value.shape, value.dtype)
                    self.pinned_inputs[name] = staged
                np.copyto(staged, value)
                setattr(self.runner, name, staged)
            _, end = self.runner.enqueue(self.stream)
            end.synchronize()
            shape = tuple(self.runner.context.get_tensor_shape("logprobs"))
            logits = self.runner.output_host["logprobs"][:int(np.prod(shape))].reshape(shape)
            encoded_lengths = self.runner.output_host["encoded_lengths"][:len(audios)]
        if not np.isfinite(logits).all():
            raise FloatingPointError("Nonfinite Indic CTC logits; reject this engine")
        return [{"text": self.decode(logits[index, :int(encoded_lengths[index])], self.vocab[:-1]), "words": []}
                for index in range(len(audios))]
