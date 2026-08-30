"""
Plots waveform + spectrogram for a set of audio clips side by side, so
differences between candidates can be inspected visually instead of
(or alongside) listening -- a spectrogram in particular shows
frequency/texture content over time, which reveals musical
differences a raw waveform mostly doesn't (a waveform mostly just
shows the loudness envelope).

Reusable -- pass any {label: wav_path} mapping. Defaults to the
target/start/best clips from the sparse-readout Watcher result.

Run:
    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\visualize_audio_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def plot_comparison(files: dict[str, str], out_path: str) -> None:
    fig, axes = plt.subplots(2, len(files), figsize=(5 * len(files), 6))
    if len(files) == 1:
        axes = axes.reshape(2, 1)

    for i, (name, path) in enumerate(files.items()):
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        t = np.arange(len(audio)) / sr

        axes[0, i].plot(t, audio, linewidth=0.3, color="steelblue")
        axes[0, i].set_title(name)
        axes[0, i].set_ylim(-1, 1)
        axes[0, i].set_xlabel("time (s)")
        if i == 0:
            axes[0, i].set_ylabel("waveform")

        axes[1, i].specgram(audio, Fs=sr, cmap="magma")
        axes[1, i].set_xlabel("time (s)")
        if i == 0:
            axes[1, i].set_ylabel("frequency (Hz)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved to {out_path}", flush=True)


if __name__ == "__main__":
    base = Path("experiments/psyche_search/data/sparse_readout_reference")
    plot_comparison(
        {
            "target": str(base / "reference_target.wav"),
            "start": str(base / "reference_start.wav"),
            "best": str(base / "reference_best.wav"),
        },
        str(base / "comparison.png"),
    )
