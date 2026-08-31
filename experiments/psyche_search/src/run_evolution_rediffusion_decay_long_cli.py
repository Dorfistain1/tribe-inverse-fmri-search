"""
Same as run_evolution_rediffusion_decay_cli.py (stall_patience=3,
redo_fraction_decay=0.7), but with enough generations to actually give
decay real post-engagement runway -- every real run so far (0.3708,
0.5725, 0.427, FINDINGS.md) found its best fitness BEFORE decay ever
engaged, so none of them have actually tested whether decay helps.

27 generations: extrapolated from the 3 completed 10-generation runs'
real elapsed times (avg 5.89 min/gen, worst-case 6.23 min/gen) to fit
inside a 3-hour budget with margin (168min worst case vs 180min cap).

Fresh checkpoint/output, not a resume of an earlier run -- resuming
would silently reset redo_fraction back to 0.3 (checkpoint.py only
persists the candidate population, not generator state like
redo_fraction/stall counters), which would defeat the point of this
test by adding more undecayed generations before decay could
re-trigger.

Status file now also tracks generation, best_ever_generation, and
first_decay_generation directly -- so whether the best was found
before or after decay engaged is readable at a glance, without
re-deriving it from the manifest by hand every time (as the last
three runs all required).

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\run_evolution_rediffusion_decay_long_cli.py

Ctrl+C to pause, re-run the same command to resume (generator state
resets to REDO_FRACTION on resume too -- same caveat as above, only
matters if you pause mid-run).
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.psyche_search.src.run_search import build_psychedelic_target
from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.generators.audio import AudioGenerator
from inverse_search.status_log import StatusLogger
from tribe_core import TribeRuntime

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "evolution_checkpoint_rediffusion_decay_long.json"
OUTPUT_DIR = "experiments/psyche_search/data/evolution_run_rediffusion_decay_long"
STATUS_PATH = Path("experiments/psyche_search/data/evolution_rediffusion_decay_long_status.json")
PROMPT = "acoustic guitar song with a clear, memorable melody"
REDO_FRACTION = 0.3
REDO_FRACTION_DECAY = 0.7
MIN_REDO_FRACTION = 0.05
STALL_PATIENCE = 3
N_GENERATIONS = 27  # see docstring for the extrapolation

GENERATION_START_RE = re.compile(r"generation (\d+)/\d+: generating")


def print_progress(message: str) -> None:
    print(message, flush=True)


def main():
    print("=" * 60)
    print("psyche_search: rediffusion + decay, extended run (27 generations)")
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
    best_ever_generation = None
    first_decay_generation = None
    current_generation = 0
    last_redo_fraction = REDO_FRACTION

    def progress(message: str) -> None:
        nonlocal best_ever, best_ever_generation, first_decay_generation
        nonlocal current_generation, last_redo_fraction
        print_progress(message)

        gen_match = GENERATION_START_RE.search(message)
        if gen_match:
            current_generation = int(gen_match.group(1))

        if "best fitness so far" in message:
            try:
                new_best = float(message.rsplit("=", 1)[1].strip())
                if new_best > best_ever:
                    best_ever = new_best
                    best_ever_generation = current_generation
            except ValueError:
                pass

        if first_decay_generation is None and generator.redo_fraction < last_redo_fraction:
            first_decay_generation = current_generation
        last_redo_fraction = generator.redo_fraction

        status.update(
            last_message=message,
            generation=current_generation,
            best_fitness=best_ever,
            best_ever_generation=best_ever_generation,
            first_decay_generation=first_decay_generation,
            best_found_after_decay=(
                None if best_ever_generation is None or first_decay_generation is None
                else best_ever_generation >= first_decay_generation
            ),
            current_redo_fraction=round(generator.redo_fraction, 4),
            stall_count=generator._consecutive_stalls,
            elapsed_s=round(time.monotonic() - start_time, 1),
        )

    final_population = search.run(checkpoint_path=CHECKPOINT_PATH, progress=progress)
    elapsed = time.monotonic() - start_time

    status.update(
        force=True,
        finished=True,
        generation=current_generation,
        best_fitness=best_ever,
        best_ever_generation=best_ever_generation,
        first_decay_generation=first_decay_generation,
        best_found_after_decay=(
            None if best_ever_generation is None or first_decay_generation is None
            else best_ever_generation >= first_decay_generation
        ),
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
    print(f"Best fitness: {best_ever:+.4f}, found at generation {best_ever_generation}")
    print(f"Decay first engaged at generation: {first_decay_generation}")
    print(f"Final redo_fraction: {generator.redo_fraction:.4f} (started at {REDO_FRACTION})")


if __name__ == "__main__":
    main()
