"""One-off sanity check for tribe_core, not a permanent test suite entry.
Verifies the TribeRuntime/Stimulus/BrainPrediction scaffold produces the
same result as the check_vram.py probe, and that on-disk caching works.
"""

import math
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tribe_core import Stimulus, TribeRuntime


def make_test_tone(path, seconds=3, freq=440, sample_rate=16000):
    n_samples = seconds * sample_rate
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(32767 * 0.5 * math.sin(2 * math.pi * freq * i / sample_rate))
            f.writeframesraw(struct.pack("<h", value))


def main():
    audio_path = Path(__file__).parent / "_test_tone.wav"
    make_test_tone(audio_path)

    runtime = TribeRuntime()
    stimulus = Stimulus(
        identifier="smoke_test_tone",
        modality="audio",
        source=str(audio_path),
        metadata={"purpose": "scaffold sanity check"},
    )

    prediction = runtime.predict(stimulus, use_cache=True)
    print("activity shape:", prediction.activity.shape)
    print("n_timesteps:", prediction.n_timesteps, "n_vertices:", prediction.n_vertices)
    print("timing rows:", len(prediction.timing))
    print("model_identity:", prediction.model_identity)

    cached = runtime.predict(stimulus, use_cache=True)
    assert (cached.activity == prediction.activity).all()
    print("cache round-trip OK")

    audio_path.unlink()


if __name__ == "__main__":
    main()
