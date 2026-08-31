"""
Partial-fake pre-check: does rediffusion mutation + decay converge
toward sounding like a target, or toward noise? Real audio generation
(Stable Audio Open) and real mutation (mutate_by_rediffusion, with the
amplitude-normalization fix), but a cheap LOCAL fitness (spectrogram
similarity to a fixed target clip) instead of real TRIBE -- no TRIBE
call per candidate, so this can afford far more generations than a
real run in far less time, specifically to sanity-check the mechanism
before spending real TRIBE time on it.

Target clip is generated once via Stable Audio Open itself (same
prompt, a fixed seed distinct from the search's own candidates) --
not a real/copyrighted song, sidesteps any rights question, still a
real, concrete, judge-by-ear target.

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\run_spectrogram_similarity_cli.py

Ctrl+C to pause, re-run to resume (same caveat as the real-TRIBE
scripts: generator state like redo_fraction resets on resume).

At the end: reference/target.wav, reference/start.wav, reference/best.wav
plus a waveform+spectrogram comparison PNG.
"""

import csv
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tribe_core.env import configure_hf_cache

MODEL_ROOT = "G:/AI_Models"
configure_hf_cache(MODEL_ROOT)  # before any other project import -- see FINDINGS.md's HF cache gotcha

from experiments.psyche_search.src.visualize_audio_comparison import plot_comparison
from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.generators.audio import AudioGenerator
from inverse_search.spectrogram_similarity import LocalAudioRuntime, compute_log_spectrogram, spectrogram_similarity_fitness
from inverse_search.status_log import StatusLogger

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "spectrogram_similarity_checkpoint.json"
OUTPUT_DIR = Path("experiments/psyche_search/data/spectrogram_similarity_run")
REFERENCE_DIR = OUTPUT_DIR / "reference"
STATUS_PATH = Path("experiments/psyche_search/data/spectrogram_similarity_status.json")
PROMPT = "acoustic guitar song with a clear, memorable melody"
DURATION_S = 5.0
TARGET_SEED = 999  # distinct from the search's own candidate seeds (start at 0)
REDO_FRACTION = 0.3
REDO_FRACTION_DECAY = 0.7
MIN_REDO_FRACTION = 0.05
STALL_PATIENCE = 3
POPULATION_SIZE = 6
N_GENERATIONS = 15  # far more than a real-TRIBE run could afford, since there's no TRIBE call per candidate


def print_progress(message: str) -> None:
    print(message, flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating target clip...", flush=True)
    target_gen = AudioGenerator(model_root=MODEL_ROOT, prompt=PROMPT, duration_s=DURATION_S, output_dir=str(REFERENCE_DIR))
    target_gen.load()
    target_latent = target_gen._random_latent(seed=TARGET_SEED)
    target_stimulus = target_gen.decode_latent(target_latent, identifier="target")
    target_dst = REFERENCE_DIR / "target.wav"
    Path(target_stimulus.source).replace(target_dst)
    target_spectrogram = compute_log_spectrogram(str(target_dst))
    target_gen.unload()
    print(f"  target: {target_dst}", flush=True)

    status = StatusLogger(STATUS_PATH, min_interval_s=60)
    runtime = LocalAudioRuntime()
    generator = AudioGenerator(
        model_root=MODEL_ROOT,
        prompt=PROMPT,
        duration_s=DURATION_S,
        output_dir=str(OUTPUT_DIR),
        mutation_mode="rediffusion",
        redo_fraction=REDO_FRACTION,
        redo_fraction_decay=REDO_FRACTION_DECAY,
        min_redo_fraction=MIN_REDO_FRACTION,
        stall_patience=STALL_PATIENCE,
    )
    config = SearchConfig(population_size=POPULATION_SIZE, n_generations=N_GENERATIONS)
    search = EvolutionarySearch(
        runtime=runtime, generator=generator, target=target_spectrogram,
        fitness_fn=spectrogram_similarity_fitness, config=config,
    )

    print("=" * 60, flush=True)
    print("Spectrogram-similarity search (no TRIBE) -- sanity-checking the mechanism", flush=True)
    print(f"population={POPULATION_SIZE} generations={N_GENERATIONS}", flush=True)
    print(f"Status file: {STATUS_PATH}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    start_time = time.monotonic()

    def progress(message: str) -> None:
        print_progress(message)
        status.update(last_message=message, elapsed_s=round(time.monotonic() - start_time, 1))

    final_population = search.run(checkpoint_path=CHECKPOINT_PATH, progress=progress)
    elapsed = time.monotonic() - start_time

    status.update(force=True, finished=True, elapsed_s=round(elapsed, 1))

    print(flush=True)
    print("=== final population, best fitness first ===", flush=True)
    for c in final_population:
        print(f"  gen={c.generation} fitness={c.fitness:+.4f} file={c.stimulus.source}", flush=True)
    print(flush=True)
    print(f"Elapsed: {elapsed / 60:.1f} min ({elapsed:.0f}s)", flush=True)

    manifest_rows = list(csv.DictReader((OUTPUT_DIR / "manifest.csv").open()))
    start_row = next(r for r in manifest_rows if r["generation"] == "0")
    best_candidate = final_population[0]

    start_dst = REFERENCE_DIR / "start.wav"
    best_dst = REFERENCE_DIR / "best.wav"
    shutil.copy(start_row["source"], start_dst)
    shutil.copy(best_candidate.stimulus.source, best_dst)
    print(f"Reference clips: {REFERENCE_DIR}", flush=True)

    comparison_png = REFERENCE_DIR / "comparison.png"
    plot_comparison(
        {"target": str(target_dst), "start": str(start_dst), "best": str(best_dst)},
        str(comparison_png),
    )
    print(f"Comparison image: {comparison_png}", flush=True)
    print(flush=True)
    print("Listen to target/start/best -- does best actually sound closer to target than start?", flush=True)


if __name__ == "__main__":
    main()
