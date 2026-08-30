"""
Objective, human-interpretable acoustic features for a decoded audio
candidate -- so "does this actually sound different" can be measured,
not just eyeballed from a waveform/spectrogram or trusted from a
fitness number that may not correspond to anything audible (see
FINDINGS.md's sparse-readout listening test, where 99% fitness barely
changed how the audio sounded).

No extra dependencies -- these are standard DSP quantities computable
directly from numpy/scipy, not full audio-analysis-library territory
(no librosa in this project's dependencies).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AcousticFeatures:
    spectral_centroid_hz: float  # "brightness" -- higher = more treble-heavy
    rms_energy: float  # overall loudness
    dynamic_range_db: float  # loud-vs-quiet spread -- higher = more dynamic, less flat/compressed
    zero_crossing_rate: float  # rough noisiness/percussiveness proxy


def compute_acoustic_features(audio: np.ndarray, sr: int) -> AcousticFeatures:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)

    # Spectral centroid: the "center of mass" of the frequency
    # spectrum, in Hz -- standard brightness measure.
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    centroid = float(np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-12))

    rms = float(np.sqrt(np.mean(audio**2)))

    # Dynamic range: spread between loud and quiet moments (short-
    # window RMS envelope), in dB -- distinguishes "consistently loud"
    # from "quiet verse / loud chorus" type variation.
    window = max(1, sr // 20)  # 50ms windows
    n_windows = len(audio) // window
    if n_windows > 1:
        windowed_rms = np.array([
            np.sqrt(np.mean(audio[i * window:(i + 1) * window] ** 2) + 1e-12)
            for i in range(n_windows)
        ])
        dynamic_range = float(20 * np.log10(windowed_rms.max() / (windowed_rms.min() + 1e-12)))
    else:
        dynamic_range = 0.0

    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

    return AcousticFeatures(
        spectral_centroid_hz=centroid,
        rms_energy=rms,
        dynamic_range_db=dynamic_range,
        zero_crossing_rate=zcr,
    )
