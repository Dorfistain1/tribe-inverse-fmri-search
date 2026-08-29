"""
Workarounds for facebook/tribev2 bugs that only surface on native Windows.
The library was developed and tested on Linux; none of these are TRIBE
science, they're portability shims. Isolated here so runtime.py stays
readable and so it's obvious what to remove if upstream ever fixes them.

Fixes applied:
1. TribeModel.from_pretrained() casts the HF repo id through pathlib.Path,
   which normalizes "facebook/tribev2" to "facebook\\tribev2" on Windows,
   and HF Hub's repo-id validator then rejects it. Worked around in
   runtime.py by resolving the snapshot to a real local dir ourselves
   (resolve_checkpoint_dir) instead of passing the repo id through.
2. The checkpoint's config.yaml was pickled on Linux and embeds
   PosixPath objects, which raise NotImplementedError when Python tries
   to instantiate them on Windows. Aliased to WindowsPath here.
3. eventstransforms.ExtractWordsFromAudio shells out to `whisperx` via
   `uvx`, which resolves its own isolated environment -- on this machine
   that pulled a CPU-only torch build. But the library picks --device
   based on *our* process's torch.cuda.is_available(), handing that
   subprocess a "cuda" flag it can't honor, and hardcodes
   compute_type="float16", which ctranslate2 rejects on CPU. Patched to
   pick compute_type based on device, matching what whisperx's own CLI
   does when left to auto-select.
4. That same whisperx subprocess needs a literal `ffmpeg` executable on
   PATH, and Gradio's video component (tools/brain_viewer) separately
   needs `ffprobe` to validate a rendered video. Neither ships with a
   normal pip install on Windows. static-ffmpeg fetches both (one-time,
   cached in its own package dir) under their real names. See
   ensure_ffmpeg_on_path().
5. exca's dataloader-worker cache (used once real, non-trivial audio
   produces enough segments for multiple workers to race on the same
   cache key) checks whether the worker that claimed an item is still
   alive via `os.kill(pid, 0)`. That's a POSIX idiom -- Windows raises a
   plain OSError (WinError 87) instead of ProcessLookupError/
   PermissionError, which exca doesn't catch. Patched to use
   psutil.pid_exists(), which is what tribev2 already depends on
   transitively.
"""

from __future__ import annotations

import os
import pathlib

_applied = False


def apply(config) -> None:
    global _applied
    if _applied:
        return
    _patch_posix_path()
    _ensure_ffmpeg_on_path()
    _patch_pid_alive_check()
    _applied = True


def _patch_posix_path() -> None:
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath


def _ensure_ffmpeg_on_path() -> None:
    import static_ffmpeg

    # Fetches both binaries under their real names, cached after the
    # first call in static_ffmpeg's own package directory.
    ffmpeg_path, _ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ["PATH"]


def _patch_pid_alive_check() -> None:
    """See fix (5) above."""
    if os.name != "nt":
        return
    import psutil
    from exca.cachedict import inflight as _inflight

    _inflight._is_pid_alive = psutil.pid_exists


def resolve_checkpoint_dir(repo_id: str, cache_dir: str) -> str:
    """See fix (1) above -- pass a real local dir, not the bare repo id."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, cache_dir=cache_dir)


def patch_transcription_compute_type() -> None:
    """See fix (3) above."""
    import torch
    from tribev2 import eventstransforms as _et

    def _get_transcript_from_audio_cpu_safe(wav_filename, language):
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

    _et.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(
        _get_transcript_from_audio_cpu_safe
    )


class force_cpu_transcription:
    """See fix (3) above. Context manager: report no CUDA for the
    duration of a get_events_dataframe() call so eventstransforms.py
    asks the whisperx subprocess for CPU instead of a "cuda" flag its
    isolated environment can't honor."""

    def __enter__(self):
        import torch

        self._original = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        return self

    def __exit__(self, *exc_info):
        import torch

        torch.cuda.is_available = self._original
