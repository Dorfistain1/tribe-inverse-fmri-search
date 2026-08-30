"""
Standalone CLI for the random-search baseline, matched to
run_evolution_cli.py's real budget (SearchConfig defaults: population
6, 5 generations, 30 evaluations) so the two runs' best-fitness-so-far
curves can be compared directly -- see FINDINGS.md's
"first real-budget evolutionary run" entry for why this comparison
matters (a rising fitness curve alone doesn't prove selection/mutation
helped; only a matched-budget random run can show that).

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\run_baseline_cli.py

No pause/resume here (run_random_baseline() doesn't support
checkpointing, see search.py's docstring) -- if you need to stop it,
just close the window; there's nothing to resume, you'd rerun from
scratch. Do not run this at the same time as another script using
TRIBE or Stable Audio Open -- they don't fit in VRAM together.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.psyche_search.src.run_search import build_psychedelic_target
from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.generators.audio import AudioGenerator
from tribe_core import TribeRuntime

OUTPUT_DIR = "experiments/psyche_search/data/baseline_run"
# Kept matched to run_evolution_cli.py's prompt in case this script
# gets run again later -- see FINDINGS.md's "random-baseline comparison
# runs deprioritized" entry for why it isn't part of the regular
# workflow anymore.
PROMPT = "upbeat instrumental pop song with drums, bass, and a catchy melody"


def print_progress(message: str) -> None:
    print(message, flush=True)


def main():
    print("=" * 60)
    print("psyche_search: random-search baseline (matched budget)")
    print("No pause/resume -- close the window to stop.")
    print("=" * 60)
    print()

    runtime = TribeRuntime()
    target = build_psychedelic_target()
    generator = AudioGenerator(prompt=PROMPT, output_dir=OUTPUT_DIR)
    config = SearchConfig()  # same defaults as run_evolution_cli.py: population 6, 5 generations

    search = EvolutionarySearch(
        runtime=runtime, generator=generator, target=target, config=config
    )
    start_time = time.monotonic()
    all_candidates = search.run_random_baseline(progress=print_progress)
    elapsed = time.monotonic() - start_time

    print()
    print("=== all candidates, best fitness first ===")
    for c in all_candidates:
        print(
            f"  gen={c.generation} fitness={c.fitness:+.4f} "
            f"id={c.stimulus.identifier} file={c.stimulus.source}"
        )
    print()
    print(f"Elapsed: {elapsed / 60:.1f} min ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
