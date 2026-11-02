import io
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def decode_audio(payload: bytes, max_seconds: float = 20.0) -> np.ndarray:
    with sf.SoundFile(io.BytesIO(payload)) as source:
        if source.frames <= 0 or source.frames / source.samplerate > max_seconds:
            raise ValueError(f"Audio must be nonempty and at most {max_seconds}s")
        audio = source.read(dtype="float32", always_2d=True).mean(axis=1)
        sr = source.samplerate
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // divisor, sr // divisor)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains nonfinite samples")
    return np.ascontiguousarray(np.clip(audio, -1, 1), dtype=np.float32)
