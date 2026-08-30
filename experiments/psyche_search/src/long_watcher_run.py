"""
Long fake-tier run with the Watcher v2 (gradient-directed surrogate
mutation, generators/watcher_fake.py's GradientSurrogateFakeLatentGenerator)
at its calibrated step_scale -- testing how much of the 0-to-baseline
gap (baseline ~572, two unrelated random latents; 0 = identical to
target) the search can actually close, and how many generations it
takes, not just whether it beats a tiny baseline improvement.

Free (CPU-only), so generation count here is cheap -- N_GENERATIONS is
set high specifically to see the real curve (plateau? keeps climbing?)
rather than stopping at an arbitrary small number.

Writes a status file every couple minutes (inverse_search/status_log.py)
so this can be watched directly instead of waiting for a summary --
open STATUS_PATH yourself anytime while this runs.

Run:
    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\long_watcher_run.py
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.fake_runtime import FakeTribeRuntime, latent_distance_fitness
from inverse_search.generators.fake_audio import FakeLatentGenerator
from inverse_search.generators.watcher_fake import GradientSurrogateFakeLatentGenerator
from inverse_search.status_log import StatusLogger

DURATION_S = 5.0
TARGET_SEED = 999_999
N_GENERATIONS = 1000
POPULATION = 6
MUTATION_STRENGTH = 0.005
STEP_SCALE = 475  # peak of tune_watcher_v2.py's calibration (350-550 sweep) -- update if that changes
BASELINE_DISTANCE = 572.07

OUT_DIR = Path("experiments/psyche_search/data/long_watcher_run")
STATUS_PATH = Path("experiments/psyche_search/data/long_watcher_run_status.json")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status = StatusLogger(STATUS_PATH, min_interval_s=120)

    generator = GradientSurrogateFakeLatentGenerator(
        duration_s=DURATION_S,
        mutation_strength=MUTATION_STRENGTH,
        step_scale=STEP_SCALE,
        output_dir=str(OUT_DIR),
    )
    target_latent = FakeLatentGenerator(duration_s=DURATION_S)._random_latent(seed=TARGET_SEED)
    runtime = FakeTribeRuntime()
    config = SearchConfig(population_size=POPULATION, n_generations=N_GENERATIONS)

    search = EvolutionarySearch(
        runtime=runtime, generator=generator, target=target_latent,
        fitness_fn=latent_distance_fitness, config=config,
    )

    print("=" * 70, flush=True)
    print("Long Watcher run: how much of the gap can gradient-guided mutation close?", flush=True)
    print(f"population={POPULATION} generations={N_GENERATIONS} step_scale={STEP_SCALE} "
          f"mutation_strength={MUTATION_STRENGTH}", flush=True)
    print(f"Status file: {STATUS_PATH}", flush=True)
    print("=" * 70, flush=True)

    start_time = time.monotonic()
    best_ever = float("-inf")
    milestones_hit = {}

    def progress(message: str) -> None:
        nonlocal best_ever
        print(message, flush=True)
        if "best fitness so far" in message:
            try:
                best_ever = float(message.rsplit("=", 1)[1].strip())
            except ValueError:
                pass
        pct = max(0.0, (BASELINE_DISTANCE - abs(best_ever)) / BASELINE_DISTANCE * 100) if best_ever != float("-inf") else 0.0
        for milestone in (10, 25, 50, 75, 90):
            if pct >= milestone and milestone not in milestones_hit:
                milestones_hit[milestone] = message
        status.update(
            last_message=message,
            best_fitness=best_ever,
            pct_gap_closed=round(pct, 2),
            elapsed_s=round(time.monotonic() - start_time, 1),
            milestones_hit=list(milestones_hit.keys()),
        )

    final_population = search.run(progress=progress)
    elapsed = time.monotonic() - start_time

    best_final = max(c.fitness for c in final_population)
    improvement = best_final - (-BASELINE_DISTANCE)
    pct_final = improvement / BASELINE_DISTANCE * 100

    status.update(
        force=True,
        finished=True,
        best_fitness=best_final,
        pct_gap_closed=round(pct_final, 2),
        elapsed_s=round(elapsed, 1),
        milestones_hit=list(milestones_hit.keys()),
    )

    print(flush=True)
    print("=== final result ===", flush=True)
    print(f"best fitness: {best_final:+.4f} (baseline: {-BASELINE_DISTANCE:+.2f})", flush=True)
    print(f"% of gap closed: {pct_final:.2f}%", flush=True)
    print(f"elapsed: {elapsed / 60:.1f} min", flush=True)
    print(f"milestones reached: {sorted(milestones_hit.keys())}", flush=True)


if __name__ == "__main__":
    main()
