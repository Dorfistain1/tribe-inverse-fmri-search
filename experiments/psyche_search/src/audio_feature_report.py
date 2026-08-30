"""
Reports acoustic features (inverse_search/acoustic_features.py) for
every candidate in a manifest.csv + wav directory, and the correlation
between fitness and each feature -- answers "does higher fitness
actually correspond to an audible difference" objectively, instead of
guessing from listening or eyeballing a spectrogram.

Run:
    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\audio_feature_report.py <directory>

Defaults to experiments/psyche_search/data/evolution_run if no
directory given.
"""

import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import soundfile as sf

from inverse_search.acoustic_features import compute_acoustic_features


def correlation(a: list[float], b: list[float]) -> float:
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / n
    sd_a, sd_b = statistics.pstdev(a), statistics.pstdev(b)
    return cov / (sd_a * sd_b) if sd_a and sd_b else float("nan")


def main():
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/psyche_search/data/evolution_run")
    manifest_path = directory / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.open()))

    records = []
    for r in rows:
        wav_path = Path(r["source"])
        if not wav_path.exists():
            continue
        audio, sr = sf.read(wav_path)
        features = compute_acoustic_features(audio, sr)
        records.append((float(r["fitness"]), features))

    print(f"{len(records)} candidates with audio (of {len(rows)} manifest rows)")
    print(f"{'fitness':>10} | {'centroid_hz':>12} | {'rms':>8} | {'dyn_range_db':>12} | {'zcr':>8}")
    for fitness, f in sorted(records, key=lambda x: -x[0]):
        print(f"{fitness:>+10.4f} | {f.spectral_centroid_hz:>12.1f} | {f.rms_energy:>8.4f} | "
              f"{f.dynamic_range_db:>12.2f} | {f.zero_crossing_rate:>8.4f}")

    print()
    print("=== correlation with fitness (1.0 = feature perfectly tracks fitness) ===")
    fitnesses = [r[0] for r in records]
    for attr in ("spectral_centroid_hz", "rms_energy", "dynamic_range_db", "zero_crossing_rate"):
        values = [getattr(r[1], attr) for r in records]
        print(f"  {attr}: {correlation(fitnesses, values):+.3f}")


if __name__ == "__main__":
    main()
