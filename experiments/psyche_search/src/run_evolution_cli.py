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

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "evolution_checkpoint_mut0.1.json"
OUTPUT_DIR = "experiments/psyche_search/data/evolution_run_mut0.1"
# 0.5 (AudioGenerator's own default) gave -0.115 real parent-child
# correlation (FINDINGS.md) and, on the fake tier, needed dropping
# ~100x before correlation stopped being ~0 (also FINDINGS.md) --
# testing a real run at a meaningfully smaller value to see if it
# actually helps on TRIBE's real landscape, not just the fake one.
MUTATION_STRENGTH = 0.1
# Structured, not ambient/drone -- a human needs something with melody/
# rhythm to actually tell candidates apart by ear (drone all sounds the
# same regardless of fitness, see FINDINGS.md). No vocals: dropped
# before ever running the vocal version -- this model isn't built for
# voice mimicry, and mumbled quasi-singing wasn't worth the risk versus
# an instrumental prompt that still has real melody to judge by ear.
# No drums/"upbeat" either, on reflection -- that's exactly the failure
# mode AudioGenerator's docstring already documents (precise/crisp
# transients, tested: EDM drums, came out worse than sustained/melodic
# content on this model). Using its one tested-good example instead:
# a clear, structured melody without percussive transients.
PROMPT = "acoustic guitar song with a clear, memorable melody"


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
    generator = AudioGenerator(
        prompt=PROMPT, output_dir=OUTPUT_DIR, mutation_strength=MUTATION_STRENGTH
    )
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
