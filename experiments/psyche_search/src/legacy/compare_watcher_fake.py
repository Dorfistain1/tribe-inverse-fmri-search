"""
Compares three mutation strategies on the fake tier -- same hidden
target, same population/generations budget, so the comparison is
fair:

1. blind_fixed      -- plain random mutation, mutation_strength fixed
                        at 0.005 (the best value found so far, see
                        FINDINGS.md's meta-search entry).
2. blind_adaptive   -- plain random mutation, mutation_strength starts
                        at 0.5 (the old default) and shrinks by 30%
                        whenever a generation fails to beat the
                        previous best, floored at 0.001.
3. watcher_fixed    -- SurrogateFakeLatentGenerator (see
                        generators/watcher_fake.py): PCA + ridge
                        regression surrogate picks the best of several
                        candidate mutations before spending a real
                        evaluation on it, mutation_strength fixed at
                        0.005.

All free (CPU-only, no GPU). Reports best-fitness-so-far by generation
for each, so you can see the shape of the climb, not just an endpoint.

Run:
    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\compare_watcher_fake.py
"""

import csv
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.fake_runtime import FakeTribeRuntime, latent_distance_fitness
from inverse_search.generators.fake_audio import FakeLatentGenerator
from inverse_search.generators.watcher_fake import SurrogateFakeLatentGenerator

DURATION_S = 5.0
TARGET_SEED = 999_999
N_GENERATIONS = 30
POPULATION = 6
OUT_DIR = Path("experiments/psyche_search/data/compare_watcher_fake")


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


def run_one(name: str, generator) -> list[float]:
    out_dir = OUT_DIR / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator.output_dir = str(out_dir)

    target_latent = FakeLatentGenerator(duration_s=DURATION_S)._random_latent(seed=TARGET_SEED)
    runtime = FakeTribeRuntime()
    config = SearchConfig(population_size=POPULATION, n_generations=N_GENERATIONS)
    search = EvolutionarySearch(
        runtime=runtime,
        generator=generator,
        target=target_latent,
        fitness_fn=latent_distance_fitness,
        config=config,
    )
    start = time.monotonic()
    search.run()
    elapsed = time.monotonic() - start
    print(f"  {name}: {elapsed:.1f}s")
    return best_so_far_by_generation(out_dir / "manifest.csv", N_GENERATIONS)


def main():
    print("=" * 70)
    print("Comparing mutation strategies on the fake tier")
    print(f"population={POPULATION} generations={N_GENERATIONS}, same hidden target")
    print("=" * 70)

    conditions = {
        "blind_fixed": FakeLatentGenerator(duration_s=DURATION_S, mutation_strength=0.005),
        "blind_adaptive": FakeLatentGenerator(
            duration_s=DURATION_S, mutation_strength=0.5, mutation_decay=0.7,
            min_mutation_strength=0.001,
        ),
        "watcher_fixed": SurrogateFakeLatentGenerator(
            duration_s=DURATION_S, mutation_strength=0.005
        ),
    }

    curves = {}
    for name, generator in conditions.items():
        curves[name] = run_one(name, generator)

    print()
    print(f"{'gen':>4} | {'blind_fixed':>12} | {'blind_adaptive':>15} | {'watcher_fixed':>14}")
    for gen in range(N_GENERATIONS):
        print(
            f"{gen:>4} | {curves['blind_fixed'][gen]:>12.4f} | "
            f"{curves['blind_adaptive'][gen]:>15.4f} | {curves['watcher_fixed'][gen]:>14.4f}"
        )

    print()
    print("=== final best-so-far ===")
    for name, curve in curves.items():
        print(f"  {name}: {curve[-1]:+.4f}")


if __name__ == "__main__":
    main()
