"""
Small-budget REAL mutation_strength sweep -- meant to be run and
watched directly, not summarized after the fact.

Real audio generation + real TRIBE for every candidate (unlike
sweep_mutation_strength.py, which is pure CPU fake-fitness), just a
much smaller population/generations per value (3x2 = 5 evals) than a
full search (SearchConfig default 6x5 = 30), so each value takes
minutes, not ~80.

Values tested: 0.5 (current default, for a same-budget baseline --
the existing -0.115 correlation measurement used a different, larger
budget), 0.02 and 0.005 (the two points where the fake-tier sweep,
see FINDINGS.md, first showed non-degenerate parent-child
correlation). 0.1 is skipped here -- the fake sweep already put it in
the same dead cluster as 0.5, not worth spending real GPU time to
recheck before the more informative points.

Run yourself, watch the numbers live:
    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\real_mutation_sweep_cli.py

Ctrl+C between values is safe -- each value's results are written to
its own folder before the next one starts, so stopping early loses
nothing already completed. No checkpoint/resume within a single
value's short run -- if interrupted mid-value, that one value's run is
simply incomplete; rerun the whole script (cheap, small budget) rather
than trying to resume it.
"""

import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.psyche_search.src.run_search import build_psychedelic_target
from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.generators.audio import AudioGenerator
from tribe_core import TribeRuntime

PROMPT = "acoustic guitar song with a clear, memorable melody"
MUTATION_STRENGTHS = [0.5, 0.02, 0.005]
SMALL_CONFIG = SearchConfig(population_size=3, n_generations=2)
BASE_DIR = Path("experiments/psyche_search/data/real_mutation_sweep")


def print_progress(message: str) -> None:
    print(message, flush=True)


def analyze_manifest(manifest_path: Path) -> dict:
    rows = list(csv.DictReader(manifest_path.open()))
    by_id = {r["identifier"]: float(r["fitness"]) for r in rows}
    pairs = [
        (by_id[r["parent_id"]], float(r["fitness"]))
        for r in rows if r["parent_id"] and r["parent_id"] in by_id
    ]
    corr = float("nan")
    if len(pairs) >= 2:
        parents = [p for p, c in pairs]
        children = [c for p, c in pairs]
        n = len(pairs)
        mean_p, mean_c = sum(parents) / n, sum(children) / n
        cov = sum((p - mean_p) * (c - mean_c) for p, c in pairs) / n
        sd_p, sd_c = statistics.pstdev(parents), statistics.pstdev(children)
        corr = cov / (sd_p * sd_c) if sd_p and sd_c else float("nan")
    gen0 = [float(r["fitness"]) for r in rows if r["generation"] == "0"]
    return {
        "gen0_best": max(gen0) if gen0 else float("nan"),
        "overall_best": max(by_id.values()),
        "correlation": corr,
        "n_pairs": len(pairs),
    }


def main():
    target = build_psychedelic_target()
    runtime = TribeRuntime()

    print("=" * 70)
    print("Small-budget REAL mutation_strength sweep")
    print(f"population={SMALL_CONFIG.population_size} generations={SMALL_CONFIG.n_generations} "
          f"({SMALL_CONFIG.population_size + (SMALL_CONFIG.population_size - 1)} evals/value)")
    print("Ctrl+C between values is safe -- completed values are already saved.")
    print("=" * 70)

    results = []
    for strength in MUTATION_STRENGTHS:
        output_dir = BASE_DIR / f"strength_{strength}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print()
        print(f"--- mutation_strength={strength} ---")

        generator = AudioGenerator(
            prompt=PROMPT, output_dir=str(output_dir), mutation_strength=strength
        )
        search = EvolutionarySearch(
            runtime=runtime, generator=generator, target=target, config=SMALL_CONFIG
        )
        start = time.monotonic()
        search.run(progress=print_progress)
        elapsed = time.monotonic() - start

        result = analyze_manifest(output_dir / "manifest.csv")
        result["strength"] = strength
        result["elapsed_s"] = elapsed
        results.append(result)

        print(
            f"  gen0_best={result['gen0_best']:+.4f} overall_best={result['overall_best']:+.4f} "
            f"correlation={result['correlation']:.3f} ({result['n_pairs']} pairs) "
            f"elapsed={elapsed:.0f}s"
        )

    print()
    print("=" * 70)
    print("=== summary (real TRIBE, small budget -- 5 evals/value, not 30) ===")
    print(f"{'mutation_strength':>18} | {'gen0_best':>10} | {'overall_best':>12} | "
          f"{'corr':>7} | {'elapsed':>8}")
    for r in results:
        print(
            f"{r['strength']:>18} | {r['gen0_best']:>10.4f} | {r['overall_best']:>12.4f} | "
            f"{r['correlation']:>7.3f} | {r['elapsed_s']:>7.0f}s"
        )
    print()
    print("5 evals/value is too few to trust the correlation number alone --")
    print("this is for you to look at real numbers and pick a direction, not")
    print("a statistically solid answer by itself. Tell me which value (or")
    print("none) looks worth a full-budget run (run_evolution_cli.py, ~80min).")


if __name__ == "__main__":
    main()
