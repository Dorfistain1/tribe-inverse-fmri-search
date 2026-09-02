"""
Quick calibration for GradientSurrogateFakeLatentGenerator's
step_scale before running the full long comparison -- a unit-norm
gradient step spread over ~164,000 dimensions is tiny per-coordinate,
so step_scale likely needs to be a few times larger than 1 to matter
against noise at mutation_strength scale. Fake tier, free.

Run:
    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\tune_watcher_v2.py
"""

import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.fake_runtime import FakeTribeRuntime, latent_distance_fitness
from inverse_search.generators.fake_audio import FakeLatentGenerator
from inverse_search.generators.watcher_fake import GradientSurrogateFakeLatentGenerator

DURATION_S = 5.0
TARGET_SEED = 999_999
N_GENERATIONS = 30
POPULATION = 6
MUTATION_STRENGTH = 0.005
STEP_SCALES = [350, 400, 425, 450, 475, 500, 550]
OUT_DIR = Path("experiments/psyche_search/data/tune_watcher_v2")


def run_one(step_scale: float) -> float:
    out_dir = OUT_DIR / f"scale_{step_scale}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = GradientSurrogateFakeLatentGenerator(
        duration_s=DURATION_S,
        mutation_strength=MUTATION_STRENGTH,
        step_scale=step_scale,
        output_dir=str(out_dir),
    )
    target_latent = FakeLatentGenerator(duration_s=DURATION_S)._random_latent(seed=TARGET_SEED)
    runtime = FakeTribeRuntime()
    config = SearchConfig(population_size=POPULATION, n_generations=N_GENERATIONS)
    search = EvolutionarySearch(
        runtime=runtime, generator=generator, target=target_latent,
        fitness_fn=latent_distance_fitness, config=config,
    )
    search.run()

    rows = list(csv.DictReader((out_dir / "manifest.csv").open()))
    fitness = [float(r["fitness"]) for r in rows]
    gen0 = [float(r["fitness"]) for r in rows if r["generation"] == "0"]
    return max(fitness) - max(gen0)


BASELINE_DISTANCE = 572.07  # ~sqrt(2*164000) -- distance between two unrelated random latents


def main():
    print(f"{'step_scale':>10} | {'improvement':>12} | {'% of gap closed':>16}", flush=True)
    for scale in STEP_SCALES:
        improvement = run_one(scale)
        pct = improvement / BASELINE_DISTANCE * 100
        print(f"{scale:>10} | {improvement:>12.4f} | {pct:>15.2f}%", flush=True)
    print(flush=True)
    print("compare against blind_fixed's +0.0841 / 0.01% (30 gens, same target, FINDINGS.md)", flush=True)


if __name__ == "__main__":
    main()
