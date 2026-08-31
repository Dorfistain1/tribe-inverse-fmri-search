"""
Real evolutionary search against the real psychedelic target, using
the partial-re-diffusion mutation (mutate_by_rediffusion(), verified
working -- see FINDINGS.md) instead of raw-latent-noise mutation.

This is a longer real run than the earlier ~80min ones: population 6,
10 generations (~30-45 evaluations more than before). Note this uses
the FIXED, verified mutation mechanism (a controllable redo_fraction
dial), not yet the smart PCA-surrogate "Watcher" -- that guided-
direction logic has only been built/tested on the fake tier so far
(generators/watcher_fake.py), never ported to real audio. This run
tests whether the *fixed* mechanism (proven to produce controllable,
graded audio change) helps real TRIBE-based search, not the smart one.

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\run_evolution_rediffusion_cli.py

Ctrl+C to pause (saved after the current generation finishes
evaluating). Re-run this exact command to resume from checkpoint.

Writes a status file every couple minutes
(experiments/psyche_search/data/evolution_rediffusion_status.json) --
open it yourself anytime to check progress without needing this
terminal open, same pattern as long_watcher_run.py.

Do not run this at the same time as another script using TRIBE or
Stable Audio Open -- they don't fit in VRAM together.
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

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "evolution_checkpoint_rediffusion.json"
OUTPUT_DIR = "experiments/psyche_search/data/evolution_run_rediffusion"
STATUS_PATH = Path("experiments/psyche_search/data/evolution_rediffusion_status.json")
PROMPT = "acoustic guitar song with a clear, memorable melody"
# 0.3: "moderate, still coherent" per the visual comparison in
# FINDINGS.md (0.1 barely changed, 0.3 kept structure while getting
# denser, 0.6+ changed substantially). A reasonable middle ground for
# a first real run -- not yet tuned against real TRIBE fitness the way
# mutation_strength/step_scale were tuned on the fake tier.
REDO_FRACTION = 0.3
N_GENERATIONS = 10  # "longer" per request -- was 5 in every earlier real run


def print_progress(message: str) -> None:
    print(message, flush=True)


def main():
    print("=" * 60)
    print("psyche_search: evolutionary search (partial-re-diffusion mutation)")
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
            elapsed_s=round(time.monotonic() - start_time, 1),
        )

    final_population = search.run(checkpoint_path=CHECKPOINT_PATH, progress=progress)
    elapsed = time.monotonic() - start_time

    status.update(force=True, finished=True, best_fitness=best_ever, elapsed_s=round(elapsed, 1))

    print()
    print("=== final population, best fitness first ===")
    for c in final_population:
        print(
            f"  gen={c.generation} fitness={c.fitness:+.4f} "
            f"id={c.stimulus.identifier} file={c.stimulus.source}"
        )
    print()
    print(f"Elapsed: {elapsed / 60:.1f} min ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
