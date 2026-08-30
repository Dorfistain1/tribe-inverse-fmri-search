"""
Long fake-tier run with the Watcher v2 (gradient-directed surrogate
mutation, generators/watcher_fake.py's GradientSurrogateFakeLatentGenerator)
-- testing how much of the 0-to-baseline gap (baseline ~572, two
unrelated random latents; 0 = identical to target) the search can
actually close, and how many generations it takes.

First attempt at this (fixed step_scale=475, no decay) found a real
problem: it jumped to 19.5% of the gap in 4 generations, then
completely plateaued for 347+ generations straight -- a fixed step
size that's big enough to find a good region fast is too big to refine
once there (every subsequent step overshoots by the same huge amount
regardless of direction quality). Caught via the status file's
batch_best staying far below running_best for hundreds of generations
in a row -- see FINDINGS.md.

This version adds step_scale_decay: shrink the step once a generation
fails to improve, so the search can switch from "leap toward a good
area" to "refine locally" without a fixed value forcing one or the
other for the whole run.

Free (CPU-only), so generation count here is cheap -- N_GENERATIONS is
set high specifically to see the real curve.

Writes a status file every couple minutes (inverse_search/status_log.py),
now tracking batch_best (this generation's best NEW candidate) alongside
running_best (all-time best) and a stall counter -- the gap between
batch_best and running_best staying wide for many generations is
exactly the plateau signal that was missed before. Open STATUS_PATH
yourself anytime while this runs.

Run:
    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\long_watcher_run.py
"""

import re
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
STEP_SCALE = 475  # peak of tune_watcher_v2.py's calibration (350-550 sweep)
STEP_SCALE_DECAY = 0.7  # shrink by 30% after enough consecutive non-improving generations
MIN_STEP_SCALE = 1.0
STALL_PATIENCE = 5  # only decay after this many CONSECUTIVE stalled generations
BASELINE_DISTANCE = 572.07

OUT_DIR = Path("experiments/psyche_search/data/long_watcher_run")
STATUS_PATH = Path("experiments/psyche_search/data/long_watcher_run_status.json")

EVALUATED_RE = re.compile(r"evaluated \d+/\d+: \S+ fitness=([+-][\d.]+)")
GENERATION_START_RE = re.compile(r"generation (\d+)/\d+: generating")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status = StatusLogger(STATUS_PATH, min_interval_s=120)

    generator = GradientSurrogateFakeLatentGenerator(
        duration_s=DURATION_S,
        mutation_strength=MUTATION_STRENGTH,
        step_scale=STEP_SCALE,
        step_scale_decay=STEP_SCALE_DECAY,
        min_step_scale=MIN_STEP_SCALE,
        stall_patience=STALL_PATIENCE,
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
          f"(decay={STEP_SCALE_DECAY}) mutation_strength={MUTATION_STRENGTH}", flush=True)
    print(f"Status file: {STATUS_PATH}", flush=True)
    print("=" * 70, flush=True)

    start_time = time.monotonic()
    running_best = float("-inf")
    current_batch_best = float("-inf")
    current_generation = 0
    stall_count = 0
    milestones_hit = {}

    def pct_of(fitness: float) -> float:
        if fitness == float("-inf"):
            return 0.0
        return max(0.0, (BASELINE_DISTANCE - abs(fitness)) / BASELINE_DISTANCE * 100)

    def progress(message: str) -> None:
        nonlocal running_best, current_batch_best, current_generation, stall_count
        print(message, flush=True)

        gen_match = GENERATION_START_RE.search(message)
        if gen_match:
            current_generation = int(gen_match.group(1))
            current_batch_best = float("-inf")

        eval_match = EVALUATED_RE.search(message)
        if eval_match:
            current_batch_best = max(current_batch_best, float(eval_match.group(1)))

        if "best fitness so far" in message:
            new_best = float(message.rsplit("=", 1)[1].strip())
            if new_best > running_best:
                running_best = new_best
                stall_count = 0
            else:
                stall_count += 1

        pct = pct_of(running_best)
        for milestone in (10, 25, 50, 75, 90):
            if pct >= milestone and milestone not in milestones_hit:
                milestones_hit[milestone] = current_generation

        status.update(
            last_message=message,
            generation=current_generation,
            running_best=running_best,
            batch_best=current_batch_best,
            stall_count=stall_count,
            current_step_scale=round(generator.step_scale, 4),
            pct_gap_closed=round(pct, 2),
            elapsed_s=round(time.monotonic() - start_time, 1),
            milestones_hit=milestones_hit,
        )

    final_population = search.run(progress=progress)
    elapsed = time.monotonic() - start_time

    best_final = max(c.fitness for c in final_population)
    pct_final = pct_of(best_final)

    status.update(
        force=True,
        finished=True,
        generation=current_generation,
        running_best=best_final,
        current_step_scale=round(generator.step_scale, 4),
        pct_gap_closed=round(pct_final, 2),
        elapsed_s=round(elapsed, 1),
        milestones_hit=milestones_hit,
    )

    print(flush=True)
    print("=== final result ===", flush=True)
    print(f"best fitness: {best_final:+.4f} (baseline: {-BASELINE_DISTANCE:+.2f})", flush=True)
    print(f"% of gap closed: {pct_final:.2f}%", flush=True)
    print(f"elapsed: {elapsed / 60:.1f} min", flush=True)
    print(f"final step_scale: {generator.step_scale:.4f} (started at {STEP_SCALE})", flush=True)
    print(f"milestones reached (generation): {milestones_hit}", flush=True)


if __name__ == "__main__":
    main()
