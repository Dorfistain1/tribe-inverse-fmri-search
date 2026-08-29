"""
One-off sanity check: does the literature-encoded NeuralTarget +
network_delta_score even produce sensible, non-degenerate numbers on
real predictions? Doesn't need the audio generator (still a stub) --
scores whatever cached predictions already exist in
outputs/tribe_predictions/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.psyche_search.src.run_search import build_psychedelic_target
from inverse_search import network_delta_score
from tribe_core import BrainPrediction


def main():
    target = build_psychedelic_target()
    print("target:", target.name, target.network_deltas)
    print()

    cache_dir = Path(__file__).resolve().parent.parent / "outputs" / "tribe_predictions"
    npy_files = sorted(cache_dir.glob("*.npy"))
    if not npy_files:
        print(f"No cached predictions found in {cache_dir}")
        return

    for npy_path in npy_files:
        prediction = BrainPrediction.load(npy_path.with_suffix(""))
        try:
            score = network_delta_score(prediction, target)
            print(f"{npy_path.stem}: shape={prediction.activity.shape} score={score:+.4f}")
        except ValueError as e:
            print(f"{npy_path.stem}: shape={prediction.activity.shape} skipped ({e})")


if __name__ == "__main__":
    main()
