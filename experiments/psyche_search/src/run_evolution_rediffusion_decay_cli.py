"""
Same as run_evolution_rediffusion_cli.py, but with adaptive
redo_fraction decay -- direct fix for that run's result (FINDINGS.md):
climbed fast to +0.3708 by gen3, then completely flat for 6 more
generations at fixed redo_fraction=0.3. Same stall_patience-gated
decay pattern already proven on the fake tier (step_scale_decay),
just ported to this mutation mode for the first time here.

Separate checkpoint/output dir from the non-decay run -- same
n_generations (10) so the two are directly comparable, isolating
decay as the only changed variable.

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\run_evolution_rediffusion_decay_cli.py

Ctrl+C to pause, re-run the same command to resume.

Status file: experiments/psyche_search/data/evolution_rediffusion_decay_status.json
-- now also tracks current_redo_fraction and stall_count, so you can
watch the decay actually happen (or not) live.

Do not run this at the same time as another script using TRIBE or
Stable Audio Open.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.psyche_search.src.run_search import build_psychedelic_target
from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.generators.audio import AudioGenerator
from inverse_search.status_log import StatusLogger
from tribe_core import TribeRuntime

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "evolution_checkpoint_rediffusion_decay_p3.json"
OUTPUT_DIR = "experiments/psyche_search/data/evolution_run_rediffusion_decay_p3"
STATUS_PATH = Path("experiments/psyche_search/data/evolution_rediffusion_decay_p3_status.json")
PROMPT = "acoustic guitar song with a clear, memorable melody"
REDO_FRACTION = 0.3  # same starting point as the non-decay run, for a fair comparison
REDO_FRACTION_DECAY = 0.7  # same factor as step_scale_decay's proven value on the fake tier
MIN_REDO_FRACTION = 0.05
N_GENERATIONS = 10  # matches the non-decay run exactly
# 3, not the default 5: measured directly on the first decay run's
# manifest -- 6 straight stalled generations (gen4-9) only triggered
# ONE decay cycle at patience=5, barely denting redo_fraction before
# the run ended. Within a fixed 10-generation budget, patience=3
# would've triggered twice in the same stretch (0.3->0.21->0.147),
# more chances to actually refine. Not as aggressive as patience=1
# (the very first attempt, which decayed too fast and underperformed
# no decay at all, FINDINGS.md) -- a middle ground informed by two
# real data points now, not a guess.
STALL_PATIENCE = 3


def print_progress(message: str) -> None:
    print(message, flush=True)


def main():
    print("=" * 60)
    print("psyche_search: evolutionary search (rediffusion mutation + adaptive decay)")
    print("Ctrl+C to pause (saved after the current generation finishes")
    print("evaluating). Re-run this exact command to resume.")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Status file: {STATUS_PATH}")
    print("=" * 60)
    print()

    status = StatusLogger(STATUS_PATH, min_interval_s=120)
    runtime = TribeRuntime()
    target = build_psychedelic_target()
    generator = AudioGenerator(
        prompt=PROMPT,
        output_dir=OUTPUT_DIR,
        mutation_mode="rediffusion",
        redo_fraction=REDO_FRACTION,
        redo_fraction_decay=REDO_FRACTION_DECAY,
        min_redo_fraction=MIN_REDO_FRACTION,
        stall_patience=STALL_PATIENCE,
    )
    config = SearchConfig(n_generations=N_GENERATIONS)

    search = EvolutionarySearch(runtime=runtime, generator=generator, target=target, config=config)

    start_time = time.monotonic()
    best_ever = float("-inf")

    def progress(message: str) -> None:
        nonlocal best_ever
        print_progress(message)
        if "best fitness so far" in message:
            try:
                best_ever = float(message.rsplit("=", 1)[1].strip())
            except ValueError:
                pass
        status.update(
            last_message=message,
            best_fitness=best_ever,
            current_redo_fraction=round(generator.redo_fraction, 4),
            stall_count=generator._consecutive_stalls,
            elapsed_s=round(time.monotonic() - start_time, 1),
        )

    final_population = search.run(checkpoint_path=CHECKPOINT_PATH, progress=progress)
    elapsed = time.monotonic() - start_time

    status.update(
        force=True,
        finished=True,
        best_fitness=best_ever,
        current_redo_fraction=round(generator.redo_fraction, 4),
        elapsed_s=round(elapsed, 1),
    )

    print()
    print("=== final population, best fitness first ===")
    for c in final_population:
        print(
            f"  gen={c.generation} fitness={c.fitness:+.4f} "
            f"id={c.stimulus.identifier} file={c.stimulus.source}"
        )
    print()
    print(f"Elapsed: {elapsed / 60:.1f} min ({elapsed:.0f}s)")
    print(f"Final redo_fraction: {generator.redo_fraction:.4f} (started at {REDO_FRACTION})")


if __name__ == "__main__":
    main()
