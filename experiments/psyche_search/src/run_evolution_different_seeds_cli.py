"""
Checks whether "climbs for ~5 generations then plateaus" is a real
property of this search, or an artifact of always starting from the
same gen0 population. Found (FINDINGS.md, 2026-09-01): AudioGenerator's
seed counter always starts at 0 in a fresh process, so EVERY real run
in this project -- including both trustworthy post-determinism-fix runs
that peaked at generation 5 with suspiciously similar fitness (0.3852,
0.3519) -- generated the identical 6 gen0 candidates. Never actually
tested whether the pattern generalizes to a different starting draw.

Only change from run_evolution_deterministic_hour_cli.py (the original
trustworthy baseline, stall_patience=3): `initial_seed_offset=100`
instead of the implicit 0. Everything else held identical on purpose
(same prompt, same budget, same redo_fraction/decay/patience) --
isolates just this one variable. If this run also climbs for ~5
generations and plateaus around a similar fitness, that's real evidence
of a general property (maybe a genuine ceiling for this prompt/setup).
If it looks very different, the pattern was specific to the one shared
starting population every prior run happened to use.

Time budget: same shape as the last two real runs, ~62-65min expected
(9 generations at the now-measured ~6.9-7.2 min/gen pace).

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\run_evolution_different_seeds_cli.py
"""

import csv
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tribe_core.env import configure_deterministic_mode, configure_hf_cache

MODEL_ROOT = "G:/AI_Models"
configure_hf_cache(MODEL_ROOT)  # before any other project import -- see FINDINGS.md's HF cache gotcha
configure_deterministic_mode()  # same rule -- before any other project import

from experiments.psyche_search.src.run_search import build_psychedelic_target
from experiments.psyche_search.src.visualize_audio_comparison import plot_comparison
from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.generators.audio import AudioGenerator
from inverse_search.status_log import StatusLogger
from tribe_core import TribeRuntime

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "evolution_checkpoint_different_seeds.json"
OUTPUT_DIR = Path("experiments/psyche_search/data/evolution_run_different_seeds")
REFERENCE_DIR = OUTPUT_DIR / "reference"
STATUS_PATH = Path("experiments/psyche_search/data/evolution_different_seeds_status.json")

PROMPT = "psychedelic ambient soundscape with swirling, shifting layers and hypnotic texture"  # psy_ambient, unchanged

REDO_FRACTION = 0.3
REDO_FRACTION_DECAY = 0.7
MIN_REDO_FRACTION = 0.05
STALL_PATIENCE = 3  # matches the ORIGINAL trustworthy baseline, not the decay fair-test
N_GENERATIONS = 9
INITIAL_SEED_OFFSET = 100  # the one variable this run tests -- any value != 0 works, 100 is arbitrary

GENERATION_START_RE = re.compile(r"generation (\d+)/\d+: generating")


def print_progress(message: str) -> None:
    print(message, flush=True)


def main():
    print("=" * 60, flush=True)
    print("psyche_search: different starting seeds (initial_seed_offset=100)", flush=True)
    print(f"Prompt: {PROMPT!r}", flush=True)
    print("Ctrl+C to pause, re-run this exact command to resume.", flush=True)
    print(f"Checkpoint: {CHECKPOINT_PATH}", flush=True)
    print(f"Status file: {STATUS_PATH}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    status = StatusLogger(STATUS_PATH, min_interval_s=120)
    runtime = TribeRuntime()
    target = build_psychedelic_target()
    generator = AudioGenerator(
        prompt=PROMPT,
        output_dir=str(OUTPUT_DIR),
        mutation_mode="rediffusion",
        redo_fraction=REDO_FRACTION,
        redo_fraction_decay=REDO_FRACTION_DECAY,
        min_redo_fraction=MIN_REDO_FRACTION,
        stall_patience=STALL_PATIENCE,
        initial_seed_offset=INITIAL_SEED_OFFSET,
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

    print(flush=True)
    print("=== final population, best fitness first ===", flush=True)
    for c in final_population:
        print(
            f"  gen={c.generation} fitness={c.fitness:+.4f} "
            f"id={c.stimulus.identifier} file={c.stimulus.source}",
            flush=True,
        )
    print(flush=True)
    print(f"Elapsed: {elapsed / 60:.1f} min ({elapsed:.0f}s)", flush=True)
    print(f"Best fitness: {best_ever:+.4f}, found at generation {best_ever_generation}", flush=True)
    print(f"Decay first engaged at generation: {first_decay_generation}", flush=True)
    print(f"Final redo_fraction: {generator.redo_fraction:.4f} (started at {REDO_FRACTION})", flush=True)

    # --- start/best reference clips + comparison image ---
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = list(csv.DictReader((OUTPUT_DIR / "manifest.csv").open()))
    start_row = next(r for r in manifest_rows if r["generation"] == "0")
    best_candidate = final_population[0]

    start_dst = REFERENCE_DIR / f"start_g00_fit{float(start_row['fitness']):+.3f}.wav"
    best_dst = REFERENCE_DIR / (
        f"best_g{best_candidate.generation:02d}_fit{best_candidate.fitness:+.3f}.wav"
    )
    shutil.copy(start_row["source"], start_dst)
    shutil.copy(best_candidate.stimulus.source, best_dst)
    print(flush=True)
    print(f"Reference clips copied to {REFERENCE_DIR}:", flush=True)
    print(f"  {start_dst.name}", flush=True)
    print(f"  {best_dst.name}", flush=True)

    comparison_png = REFERENCE_DIR / "comparison.png"
    plot_comparison({"start": str(start_dst), "best": str(best_dst)}, str(comparison_png))
    print(f"Comparison image: {comparison_png}", flush=True)


if __name__ == "__main__":
    main()
