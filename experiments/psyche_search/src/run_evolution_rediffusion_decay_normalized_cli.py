"""
Same as run_evolution_rediffusion_decay_long_cli.py (27 gens,
stall_patience=3, redo_fraction_decay=0.7), but with the amplitude-
normalization fix (AudioGenerator.mutate_by_rediffusion now peak-
normalizes every candidate to 0.9) -- testing whether the fitness
climb (0.37 -> 0.86 across prior runs) holds up once clipping-driven
loudness can't inflate it. See FINDINGS.md for the clipping finding
this fixes.

At the end, copies "start" (the very first gen0 candidate) and "best"
(the final winner) into reference/start.wav and reference/best.wav --
clearly named, easy to listen to -- and renders a waveform+spectrogram
comparison PNG for both. No "target" clip: unlike the fake sparse-
readout target (a constructible latent), a real TRIBE target has no
audio form at all -- it's brain-connectivity math, not a point in
latent space, so there is nothing honest to decode as "target."

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\run_evolution_rediffusion_decay_normalized_cli.py

Ctrl+C to pause, re-run to resume (generator state resets on resume,
same caveat as the long/decay scripts -- only matters if you pause).
"""

import csv
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.psyche_search.src.run_search import build_psychedelic_target
from experiments.psyche_search.src.visualize_audio_comparison import plot_comparison
from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.generators.audio import AudioGenerator
from inverse_search.status_log import StatusLogger
from tribe_core import TribeRuntime

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "evolution_checkpoint_rediffusion_decay_norm.json"
OUTPUT_DIR = Path("experiments/psyche_search/data/evolution_run_rediffusion_decay_norm")
REFERENCE_DIR = OUTPUT_DIR / "reference"
STATUS_PATH = Path("experiments/psyche_search/data/evolution_rediffusion_decay_norm_status.json")
PROMPT = "acoustic guitar song with a clear, memorable melody"
REDO_FRACTION = 0.3
REDO_FRACTION_DECAY = 0.7
MIN_REDO_FRACTION = 0.05
STALL_PATIENCE = 3
N_GENERATIONS = 27

GENERATION_START_RE = re.compile(r"generation (\d+)/\d+: generating")


def print_progress(message: str) -> None:
    print(message, flush=True)


def main():
    print("=" * 60, flush=True)
    print("psyche_search: rediffusion + decay + amplitude normalization (27 gens)", flush=True)
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
