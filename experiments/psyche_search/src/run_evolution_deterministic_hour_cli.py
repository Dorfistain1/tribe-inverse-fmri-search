"""
First real-TRIBE run since two same-day fixes:

1. `mutate_by_rediffusion()` no longer hardcodes `manual_seed(0)` --
   every mutation call was silently producing substantially different
   audio anyway (GPU float non-determinism, not the seed), so the old
   fixed seed bought nothing but false reproducibility. Now seeds
   randomly per call and logs the seed in `Stimulus.metadata`.
2. `configure_deterministic_mode()` (tribe_core/env.py) forces bit-
   reproducible CUDA ops -- verified directly (verify_deterministic_mode.py)
   this closes the gap completely on this model with no unsupported-op
   fallback needed. Combined with fix #1's explicit seed, `redo_fraction`
   is now a real, clean, reproducible-given-the-seed exploration dial for
   the first time all project -- every earlier real run's "decay
   helps/doesn't help" conclusion was confounded by unlogged jitter (see
   FINDINGS.md, 2026-09-01/02 entries).

Also switches off the placeholder "acoustic guitar song" prompt onto a
deliberately-chosen psychedelic-style one (see probe_psychedelic_prompts.py
-- pick whichever of its 3 candidates sounded best; PROMPT below defaults
to "psy_ambient", change it here before running if a different one won).
This is a disclosed style choice, not a hidden default -- see FINDINGS.md
2026-09-01 for why ("we pick target style via prompt; search
explores/shapes fitness within it").

Time budget: ~1hr requested. Historical real-run pace (pre-deterministic-
mode) was 5.89-6.23 min/generation (FINDINGS.md, the 27-gen run's
extrapolation). Deterministic mode's speed cost on THIS pipeline is
UNMEASURED -- verify_deterministic_mode.py only checked correctness, not
timing. N_GENERATIONS below is sized conservatively (9) against the
worst-case historical pace (9 * 6.23min = ~56min) to leave margin for an
unknown deterministic-mode slowdown; if it's much slower than expected,
Ctrl+C is safe (same resume caveat as every other script here -- generator
state like redo_fraction resets on resume, only matters if you pause).

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\run_evolution_deterministic_hour_cli.py
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

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "evolution_checkpoint_deterministic_hour.json"
OUTPUT_DIR = Path("experiments/psyche_search/data/evolution_run_deterministic_hour")
REFERENCE_DIR = OUTPUT_DIR / "reference"
STATUS_PATH = Path("experiments/psyche_search/data/evolution_deterministic_hour_status.json")

PROMPTS = {
    "psy_ambient": "psychedelic ambient soundscape with swirling, shifting layers and hypnotic texture",
    "psy_electronic": "trippy electronic music with warped, morphing textures and a hypnotic pulse",
    "psy_rock": "hypnotic psychedelic rock with driving rhythm and swirling guitar effects",
}
PROMPT = PROMPTS["psy_ambient"]  # swap key here if a different probe clip won

REDO_FRACTION = 0.3
REDO_FRACTION_DECAY = 0.7
MIN_REDO_FRACTION = 0.05
STALL_PATIENCE = 3
N_GENERATIONS = 9  # see docstring's timing math

GENERATION_START_RE = re.compile(r"generation (\d+)/\d+: generating")


def print_progress(message: str) -> None:
    print(message, flush=True)


def main():
    print("=" * 60, flush=True)
    print("psyche_search: deterministic rediffusion + decay, psychedelic prompt (~1hr budget)", flush=True)
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
