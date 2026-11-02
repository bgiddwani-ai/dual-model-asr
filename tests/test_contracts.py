"""CPU tests for input handling, model routing, and truthful quality scoring."""
import base64
import io
import os
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from asr_stack.audio import decode_audio
from scripts.benchmark import normalize


def wav(seconds=0.2, sr=16000, channels=1):
    output = io.BytesIO()
    sf.write(output, np.zeros((int(seconds * sr), channels)), sr, format="WAV")
    return output.getvalue()


class InputContracts(unittest.TestCase):
    def test_resampling_and_mono(self):
        result = decode_audio(wav(sr=48000, channels=2))
        self.assertEqual(result.shape, (3200,))
        self.assertEqual(result.dtype, np.float32)

    def test_empty_and_long_audio_rejected(self):
        for duration in (0, 20.1):
            with self.assertRaises(ValueError):
                decode_audio(wav(duration))

    def test_indic_marks_survive_normalization(self):
        self.assertEqual(normalize("  नमस्ते, दुनिया!  "), "नमस्ते दुनिया")
        self.assertEqual(normalize("Hello, WORLD!"), "hello world")

    def test_worker_batch_limit_and_default_tdt(self):
        from asr_stack import worker

        class Fake:
            capacity = 1
            metadata = {"backend": "fake"}

            def transcribe(self, audios, language, decoder):
                if decoder != "tdt":
                    raise ValueError("Wrong decoder")
                return [{"text": "hello", "words": []} for _ in audios]

        encoded = base64.b64encode(wav()).decode()
        with patch.dict(os.environ, {"ASR_MODEL": "parakeet", "PARAKEET_BUNDLE": "fake"}), \
                patch.object(worker, "ParakeetBackend", return_value=Fake()), \
                TestClient(worker.app) as client:
            self.assertEqual(client.post("/transcribe", json={"audio_base64": [encoded]}).status_code, 200)
            self.assertEqual(client.post("/transcribe", json={"audio_base64": [encoded, encoded]}).status_code, 422)
            self.assertEqual(client.post("/transcribe", json={"audio_base64": ["%%%"]}).status_code, 422)
            junk = base64.b64encode(b"not an audio file").decode()
            self.assertEqual(client.post("/transcribe", json={"audio_base64": [junk]}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
