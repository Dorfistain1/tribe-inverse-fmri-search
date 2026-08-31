"""
Cheap, local (no TRIBE) fitness: how close does a candidate's
spectrogram look to a fixed target clip's? Real audio, real
generation/mutation, fake fitness -- the "partial-fake" tier of the
testing pyramid (DESIGN.md), finally built. Point: sanity-check
whether the search mechanism (rediffusion mutation + decay) converges
toward something that actually sounds like a target, or toward noise,
using a target and metric a human can directly verify by ear/eye --
before spending real TRIBE time on a run where "does this even
converge to something coherent" is untested.

Log-magnitude STFT, not raw waveform correlation -- raw waveform
comparison is extremely sensitive to phase/timing misalignment (two
perceptually identical clips can have near-zero raw correlation if
shifted by a few samples); spectrogram comparison is standard practice
for audio similarity and far more robust to that.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy import signal

from tribe_core import Stimulus


def compute_log_spectrogram(audio_path: str, nperseg: int = 1024, noverlap: int = 512) -> np.ndarray:
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    _freqs, _times, spec = signal.spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    return np.log1p(spec)


def spectrogram_similarity_fitness(stimulus: Stimulus, target_spectrogram: np.ndarray) -> float:
    """Higher is better (closer to target), matching every other
    fitness_fn's convention -- negative mean-squared-error between
    log-magnitude spectrograms, cropped to the smaller shape of the two
    (should match exactly at the same duration/sample rate, cropping is
    just a safety margin against off-by-one frame-count differences)."""
    candidate_spec = compute_log_spectrogram(stimulus.source)
    min_freq = min(candidate_spec.shape[0], target_spectrogram.shape[0])
    min_time = min(candidate_spec.shape[1], target_spectrogram.shape[1])
    a = candidate_spec[:min_freq, :min_time]
    b = target_spectrogram[:min_freq, :min_time]
    return -float(np.mean((a - b) ** 2))


class LocalAudioRuntime:
    """Duck-typed against search.py's evaluate() -- skips TRIBE
    entirely. .predict() just returns the stimulus itself (it already
    carries .source, the wav path fitness_fn needs); no model to force-
    load, nothing to unload."""

    model = None

    def predict(self, stimulus: Stimulus) -> Stimulus:
        return stimulus

    def unload(self) -> None:
        pass
