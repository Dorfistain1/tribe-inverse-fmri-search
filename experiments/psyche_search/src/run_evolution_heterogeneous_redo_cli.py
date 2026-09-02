"""
Three real, trustworthy runs (FINDINGS.md, 2026-09-01) all showed the
same shape -- fast early climb, then a plateau -- regardless of
starting population, and redo_fraction decay never clearly helped
escape it in either fair test. Enough evidence to stop tuning that
lever and try something structurally different instead of more decay
parameter sweeps.

This run: heterogeneous redo_fraction. Instead of one shared value
(fixed or globally decaying), each mutation call in a generation draws
its OWN redo_fraction uniformly from REDO_FRACTION_RANGE -- some
candidates get a small refinement-scale edit, some a bigger jump,
simultaneously, every generation. Tests whether within-generation
diversity in edit size does what between-generation decay couldn't.

No decay in this run -- redo_fraction_range bypasses self.redo_fraction
entirely per mutate() call, so redo_fraction_decay would be a silent
no-op if set (see AudioGenerator's docstring for that param). Seed
offset back to 0 (the original two runs' shared starting population),
not 100 -- isolates heterogeneity as the one new variable against two
existing same-seed baselines instead of adding a second one at once.

Time budget: same shape as prior real runs, ~62-65min expected under
normal conditions (9 generations at the measured ~6.9-7.2 min/gen
pace) -- will run slower if the PC is under load at the same time
(confirmed cause of the last run's 79.9min, not a code issue).

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\run_evolution_heterogeneous_redo_cli.py
"""

import csv
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

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "evolution_checkpoint_heterogeneous_redo.json"
OUTPUT_DIR = Path("experiments/psyche_search/data/evolution_run_heterogeneous_redo")
REFERENCE_DIR = OUTPUT_DIR / "reference"
STATUS_PATH = Path("experiments/psyche_search/data/evolution_heterogeneous_redo_status.json")

PROMPT = "psychedelic ambient soundscape with swirling, shifting layers and hypnotic texture"  # psy_ambient, unchanged

REDO_FRACTION_RANGE = (0.1, 0.5)  # the one variable this run tests
N_GENERATIONS = 9
INITIAL_SEED_OFFSET = 0  # matches the original two runs' shared starting population


def print_progress(message: str) -> None:
    print(message, flush=True)


def main():
    print("=" * 60, flush=True)
    print(f"psyche_search: heterogeneous redo_fraction, range={REDO_FRACTION_RANGE}", flush=True)
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
        redo_fraction_range=REDO_FRACTION_RANGE,
        initial_seed_offset=INITIAL_SEED_OFFSET,
    )
    config = SearchConfig(n_generations=N_GENERATIONS)
    search = EvolutionarySearch(runtime=runtime, generator=generator, target=target, config=config)

    start_time = time.monotonic()
    best_ever = float("-inf")
    best_ever_generation = None
    current_generation = 0

    def progress(message: str) -> None:
        nonlocal best_ever, best_ever_generation, current_generation
        print_progress(message)

        if "generating" in message and "generation" in message:
            try:
                current_generation = int(message.split("generation ")[1].split("/")[0])
            except (IndexError, ValueError):
                pass

        if "best fitness so far" in message:
            try:
                new_best = float(message.rsplit("=", 1)[1].strip())
                if new_best > best_ever:
                    best_ever = new_best
                    best_ever_generation = current_generation
            except ValueError:
                pass

        status.update(
            last_message=message,
            generation=current_generation,
            best_fitness=best_ever,
            best_ever_generation=best_ever_generation,
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
