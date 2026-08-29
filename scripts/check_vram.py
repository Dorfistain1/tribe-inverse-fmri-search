"""
Smoke test: load TRIBE v2 (audio branch only) and measure real VRAM usage.

Run this before building tribe_core to confirm the audio modality actually
fits on an 8GB card. Not part of the reusable runtime -- throwaway probe.

Usage:
    python scripts/check_vram.py
"""

import os
import time
import wave
import struct
import math
import pathlib
import contextlib

# The checkpoint's config.yaml was pickled on Linux and embeds PosixPath
# objects, which can't be instantiated on native Windows. Alias it to
# WindowsPath so yaml.UnsafeLoader can reconstruct them.
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

# Model/cache storage on the external HDD, not C:. Must be set before
# torch/transformers/tribev2 are imported.
MODEL_ROOT = r"G:\AI_Models"
os.environ.setdefault("HF_HOME", os.path.join(MODEL_ROOT, "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(MODEL_ROOT, "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(MODEL_ROOT, "torch"))
os.environ.setdefault("UV_CACHE_DIR", os.path.join(MODEL_ROOT, "uv_cache"))
TRIBE_CACHE = os.path.join(MODEL_ROOT, "tribev2_cache")

# whisperx (run via uvx, see below) shells out to a literal "ffmpeg"
# binary that isn't installed system-wide. imageio_ffmpeg ships one as a
# transitive dependency, but under a version-suffixed filename
# (ffmpeg-win-x86_64-*.exe), so a bare PATH lookup for "ffmpeg" still
# fails. check_vram.py copies it to G:\AI_Models\bin\ffmpeg.exe once;
# just add that dir to PATH here.
FFMPEG_BIN_DIR = os.path.join(MODEL_ROOT, "bin")
os.environ["PATH"] = FFMPEG_BIN_DIR + os.pathsep + os.environ["PATH"]

import torch
from huggingface_hub import snapshot_download
from tribev2 import TribeModel
from tribev2 import eventstransforms as _et


def _get_transcript_from_audio_cpu_safe(wav_filename, language):
    """
    Patched copy of ExtractWordsFromAudio._get_transcript_from_audio.
    The original hardcodes compute_type="float16", which ctranslate2
    rejects on CPU ("Requested float16 compute type, but the target
    device or backend do not support efficient float16 computation").
    Only change: pick compute_type based on device, same as whisperx's
    own CLI does when you let it auto-select.
    """
    import json
    import subprocess
    import tempfile

    language_codes = dict(english="en", french="fr", spanish="es", dutch="nl", chinese="zh")
    if language not in language_codes:
        raise ValueError(f"Language {language} not supported")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    with tempfile.TemporaryDirectory() as output_dir:
        cmd = [
            "uvx", "whisperx", str(wav_filename),
            "--model", "large-v3",
            "--language", language_codes[language],
            "--device", device,
            "--compute_type", compute_type,
            "--batch_size", "16",
            "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
            "--output_dir", output_dir,
            "--output_format", "json",
        ]
        cmd = [c for c in cmd if c]
        env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"whisperx failed:\n{result.stderr}")

        json_path = pathlib.Path(output_dir) / f"{wav_filename.stem}.json"
        transcript = json.loads(json_path.read_text())

    words = []
    for i, segment in enumerate(transcript["segments"]):
        sentence = segment["text"].replace('"', "")
        for word in segment["words"]:
            if "start" not in word:
                continue
            words.append({
                "text": word["word"].replace('"', ""),
                "start": word["start"],
                "duration": word["end"] - word["start"],
                "sequence_id": i,
                "sentence": sentence,
            })

    import pandas as pd
    return pd.DataFrame(words)


_et.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(_get_transcript_from_audio_cpu_safe)

# tribev2.demo_utils.TribeModel.from_pretrained wraps the repo id in
# pathlib.Path before deciding whether it's a local dir or a HF repo id.
# On Windows that turns "facebook/tribev2" into "facebook\tribev2", which
# HF Hub's repo-id validator then rejects. Work around it by resolving the
# snapshot to a real local directory ourselves and passing that instead.
def resolve_checkpoint_dir(repo_id, cache_dir):
    return snapshot_download(repo_id, cache_dir=cache_dir)


@contextlib.contextmanager
def force_cpu_transcription():
    """
    tribev2's word-timing step shells out to `whisperx` via `uvx`, which
    resolves its own isolated environment and (on this machine) pulled a
    CPU-only torch build. But eventstransforms.py picks --device based on
    *our* process's torch.cuda.is_available(), so it hands that subprocess
    a "cuda" flag it can't honor. Report no CUDA for the duration of that
    one call so it asks for CPU instead.
    """
    original = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        yield
    finally:
        torch.cuda.is_available = original


def make_test_tone(path, seconds=3, freq=440, sample_rate=16000):
    n_samples = seconds * sample_rate
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(32767 * 0.5 * math.sin(2 * math.pi * freq * i / sample_rate))
            f.writeframesraw(struct.pack("<h", value))


def main():
    if not torch.cuda.is_available():
        print("CUDA not available -- running on CPU, VRAM numbers meaningless.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))

    audio_path = os.path.join(os.path.dirname(__file__), "_test_tone.wav")
    make_test_tone(audio_path)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    t0 = time.time()
    checkpoint_dir = resolve_checkpoint_dir("facebook/tribev2", TRIBE_CACHE)
    model = TribeModel.from_pretrained(checkpoint_dir, cache_folder=TRIBE_CACHE)
    t1 = time.time()
    print(f"model load: {t1 - t0:.1f}s")
    if torch.cuda.is_available():
        print(f"VRAM after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    with force_cpu_transcription():
        df = model.get_events_dataframe(audio_path=audio_path)
    preds, segments = model.predict(events=df)
    t2 = time.time()
    print(f"predict: {t2 - t1:.1f}s")
    print("preds.shape:", preds.shape)

    if torch.cuda.is_available():
        print(f"VRAM allocated (current): {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"VRAM reserved  (current): {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"VRAM peak     (this run): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    os.remove(audio_path)


if __name__ == "__main__":
    main()
