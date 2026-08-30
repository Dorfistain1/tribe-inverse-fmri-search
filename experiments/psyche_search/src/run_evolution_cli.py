"""
Standalone CLI for the evolutionary search, with live progress and
pause/resume.

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\run_evolution_cli.py

Not through an agent's background tool call -- the whole point of this
script is a terminal window you can actually watch and interrupt.

Press Ctrl+C to pause. Progress is saved after every fully-completed
generation (so closing the window instead of Ctrl+C loses at most the
generation in progress, never everything already done). Run the exact
same command again to resume from the last checkpoint.

Do not run this at the same time as another script using TRIBE or
Stable Audio Open -- they don't fit in VRAM together, and neither does
two copies of either.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.psyche_search.src.run_search import build_psychedelic_target
from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.generators.audio import AudioGenerator
from tribe_core import TribeRuntime

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "evolution_checkpoint.json"
OUTPUT_DIR = "experiments/psyche_search/data/evolution_run"
PROMPT = "ambient drone music, sustained atmospheric pads"


def print_progress(message: str) -> None:
    print(message, flush=True)


def main():
    print("=" * 60)
    print("psyche_search: evolutionary search")
    print("Ctrl+C to pause (saved after the current generation finishes")
    print("evaluating). Re-run this exact command to resume.")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 60)
    print()

    runtime = TribeRuntime()
    target = build_psychedelic_target()
    generator = AudioGenerator(prompt=PROMPT, output_dir=OUTPUT_DIR)
    config = SearchConfig()  # tuned defaults: population 6, 5 generations

    search = EvolutionarySearch(
        runtime=runtime, generator=generator, target=target, config=config
    )
    start_time = time.monotonic()
    final_population = search.run(checkpoint_path=CHECKPOINT_PATH, progress=print_progress)
    elapsed = time.monotonic() - start_time

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
