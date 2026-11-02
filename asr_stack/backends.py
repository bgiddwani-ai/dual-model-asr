import os
import importlib.util
import json
import types
from pathlib import Path


INDIC_LANGUAGES = {
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
}


class ParakeetBackend:
    def __init__(self, bundle: str):
        from fast_gpu_asr import ASR

        self.model = ASR(bundle, device_id=0)
        self.capacity = self.model.encoder.batch_size
        self.metadata = {
            "model_id": "nvidia/parakeet-tdt-0.6b-v3",
            "backend": "fast-gpu-asr TensorRT",
            "bundle": str(bundle), "capacity": self.capacity,
        }
        settings = Path(bundle) / "ready.json"
        if settings.exists():
            self.metadata["engine_settings"] = json.loads(settings.read_text())

    def transcribe(self, audios, language=None, decoder="tdt"):
        if decoder != "tdt":
            raise ValueError("Parakeet v3 uses the TDT decoder")
        texts, words = self.model(audios)
        return [{"text": text, "words": timestamps}
                for text, timestamps in zip(texts, words, strict=True)]


class IndicConformerBackend:
    """Official gated checkpoint; retain its frontend and language tokenizers.

    ONNX sessions must expose CUDA. CPU fallback nodes are permitted by ORT and
    are disclosed; a CPU-only model is rejected. TensorRT EP is an explicit
    alternative and is never silently reported as a measured TensorRT run.
    """

    def __init__(self, model_path: str, provider: str = "CUDAExecutionProvider"):
        import onnxruntime as ort
        import torch

        ort.preload_dlls(directory="")
        torch.set_num_threads(int(os.environ.get("INDIC_CPU_THREADS", "4")))
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"{provider} unavailable: {ort.get_available_providers()}")
        source = Path(model_path)
        if not source.is_dir():
            raise FileNotFoundError("Download the authorized, pinned model snapshot first")
        # The publisher's from_pretrained override ignores local paths and
        # revisions. Instantiate its reviewed local class with the downloaded
        # config, while injecting explicit ORT providers into this module only.
        spec = importlib.util.spec_from_file_location("indic_official_local", source / "model_onnx.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        providers = [provider]
        if provider == "TensorrtExecutionProvider":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        sessions = []
        def make_session(path, **ignored):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(path, sess_options=options, providers=providers)
            actual = session.get_providers()
            if provider not in actual:
                raise RuntimeError(f"GPU provider initialization failed: {actual}")
            sessions.append({"component": Path(path).stem, "providers": actual,
                             "inputs": [x.name for x in session.get_inputs()]})
            return session
        module.ort = types.SimpleNamespace(InferenceSession=make_session)
        configuration = json.loads((source / "config.json").read_text())
        self.model = module.IndicASRModel(module.IndicASRConfig(ts_folder=str(source.resolve()), **configuration))
        self.model.models["preprocessor"].eval()
        if not sessions:
            raise RuntimeError("No ONNX sessions found; inspect this model revision before serving")
        self.capacity = 1  # The official wrapper takes one waveform per call.
        snapshot = source / "snapshot.json"
        revision = json.loads(snapshot.read_text()).get("revision") if snapshot.exists() else "unrecorded"
        self.metadata = {
            "model_id": "ai4bharat/indic-conformer-600m-multilingual",
            "backend": provider, "sessions": sessions, "capacity": 1,
            "revision": os.environ.get("INDIC_REVISION", revision),
            "frontend": "official TorchScript CPU", "torch_threads": torch.get_num_threads(),
            "rnnt_token_mapping": "publisher language-local predictor IDs retained",
        }

    def transcribe(self, audios, language="hi", decoder="ctc"):
        import torch

        language = language or "hi"
        if language not in INDIC_LANGUAGES:
            raise ValueError(f"Unsupported IndicConformer language: {language}")
        if decoder not in {"ctc", "rnnt"}:
            raise ValueError("IndicConformer decoder must be ctc or rnnt")
        with torch.inference_mode():
            return [{"text": self.model(torch.from_numpy(audio).unsqueeze(0), language, decoder),
                     "words": []} for audio in audios]
