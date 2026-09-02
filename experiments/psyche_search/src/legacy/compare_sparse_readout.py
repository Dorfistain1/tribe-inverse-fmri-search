"""
Compares blind mutation vs. the Watcher (v2, gradient-directed) on the
sparse-readout fake target (inverse_search/sparse_readout.py) -- a
target designed to match the REAL task's structure (push a few bounded
readouts as far as possible, like network_delta_score actually does),
not "match one exact point in 164,000 dimensions" like every other
fake test used earlier today. See sparse_readout.py's docstring for
why that distinction matters.

Same target, same population/generations budget for both conditions.
Free (CPU-only).

Run:
    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\compare_sparse_readout.py
"""

import csv
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.fake_runtime import FakeTribeRuntime
from inverse_search.generators.audio import LATENT_CHANNELS, LATENT_FRAMES_PER_SECOND
from inverse_search.generators.fake_audio import FakeLatentGenerator
from inverse_search.generators.watcher_fake import GradientSurrogateFakeLatentGenerator
from inverse_search.sparse_readout import make_sparse_readout_target, sparse_readout_fitness

DURATION_S = 5.0
N_GENERATIONS = 200
POPULATION = 6
N_RELEVANT = 3
MUTATION_STRENGTH = 0.005
STEP_SCALE = 475
STEP_SCALE_DECAY = 0.7
MIN_STEP_SCALE = 1.0
STALL_PATIENCE = 5
OUT_DIR = Path("experiments/psyche_search/data/compare_sparse_readout")

N_FEATURES = LATENT_CHANNELS * round(DURATION_S * LATENT_FRAMES_PER_SECOND)


def best_so_far_by_generation(manifest_path: Path, n_generations: int) -> list[float]:
    rows = list(csv.DictReader(manifest_path.open()))
    curve = []
    running_best = float("-inf")
    for gen in range(n_generations):
        gen_rows = [float(r["fitness"]) for r in rows if int(r["generation"]) == gen]
        if gen_rows:
            running_best = max(running_best, max(gen_rows))
        curve.append(running_best)
    return curve


def run_one(name: str, generator, target) -> list[float]:
    out_dir = OUT_DIR / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator.output_dir = str(out_dir)

    runtime = FakeTribeRuntime()
    config = SearchConfig(population_size=POPULATION, n_generations=N_GENERATIONS)
    search = EvolutionarySearch(
        runtime=runtime, generator=generator, target=target,
        fitness_fn=sparse_readout_fitness, config=config,
    )
    start = time.monotonic()
    search.run()
    elapsed = time.monotonic() - start
    print(f"  {name}: {elapsed:.1f}s", flush=True)
    return best_so_far_by_generation(out_dir / "manifest.csv", N_GENERATIONS)


def main():
    target = make_sparse_readout_target(N_FEATURES, n_relevant=N_RELEVANT)

    print("=" * 70, flush=True)
    print(f"Sparse-readout fake target: {N_RELEVANT} relevant directions out of "
          f"{N_FEATURES} latent dims", flush=True)
    print(f"population={POPULATION} generations={N_GENERATIONS}", flush=True)
    print("0 = typical random latent, +1 = theoretical ceiling (fully saturated)", flush=True)
    print("=" * 70, flush=True)

    conditions = {
        "blind_fixed": FakeLatentGenerator(duration_s=DURATION_S, mutation_strength=MUTATION_STRENGTH),
        "watcher_decay": GradientSurrogateFakeLatentGenerator(
            duration_s=DURATION_S, mutation_strength=MUTATION_STRENGTH,
            step_scale=STEP_SCALE, step_scale_decay=STEP_SCALE_DECAY,
            min_step_scale=MIN_STEP_SCALE, stall_patience=STALL_PATIENCE,
        ),
    }

    curves = {name: run_one(name, gen, target) for name, gen in conditions.items()}

    print(flush=True)
    print(f"{'gen':>4} | {'blind_fixed':>12} | {'watcher_decay':>14}", flush=True)
    for gen in range(0, N_GENERATIONS, max(1, N_GENERATIONS // 20)):
        print(f"{gen:>4} | {curves['blind_fixed'][gen]:>12.4f} | {curves['watcher_decay'][gen]:>14.4f}",
              flush=True)

    print(flush=True)
    print("=== final best-so-far ===", flush=True)
    for name, curve in curves.items():
        print(f"  {name}: {curve[-1]:+.4f}", flush=True)


if __name__ == "__main__":
    main()
