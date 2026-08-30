"""
Compares EvolutionarySearch's real mutation/selection loop against a
random-search baseline with the same total evaluation budget.

Per restructure.md section 15: "fitness improved" during a real search
run doesn't by itself prove evolution helped -- it could just be lucky
sampling. This runs both with matched population_size x n_generations
budgets and reports best-fitness-per-generation for each, so the
comparison is honest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from experiments.psyche_search.src.run_search import build_psychedelic_target
from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.generators.audio import AudioGenerator
from tribe_core import TribeRuntime

POPULATION_SIZE = 4
N_GENERATIONS = 3
PROMPT = "ambient drone music, sustained atmospheric pads"


def summarize(name: str, candidates: list) -> float:
    print(f"\n=== {name} ===", flush=True)
    best_per_gen = {}
    for c in candidates:
        best_per_gen[c.generation] = max(best_per_gen.get(c.generation, float("-inf")), c.fitness)
    for gen in sorted(best_per_gen):
        print(f"  generation {gen}: best fitness so far = {best_per_gen[gen]:+.4f}", flush=True)
    overall_best = max(c.fitness for c in candidates)
    print(f"  overall best: {overall_best:+.4f}", flush=True)
    return overall_best


def main():
    runtime = TribeRuntime()
    target = build_psychedelic_target()
    config = SearchConfig(
        population_size=POPULATION_SIZE, n_generations=N_GENERATIONS, elite_fraction=0.4
    )

    print("Running evolutionary search...", flush=True)
    gen1 = AudioGenerator(
        prompt=PROMPT, output_dir="experiments/psyche_search/data/compare_evolutionary"
    )
    search1 = EvolutionarySearch(runtime=runtime, generator=gen1, target=target, config=config)
    evolutionary_best = summarize("EVOLUTIONARY", search1.run())

    print("\nRunning random-search baseline...", flush=True)
    gen2 = AudioGenerator(
        prompt=PROMPT, output_dir="experiments/psyche_search/data/compare_random"
    )
    search2 = EvolutionarySearch(runtime=runtime, generator=gen2, target=target, config=config)
    random_best = summarize("RANDOM BASELINE", search2.run_random_baseline())

    print("\n=== COMPARISON ===", flush=True)
    print(f"Evolutionary best: {evolutionary_best:+.4f}", flush=True)
    print(f"Random best:       {random_best:+.4f}", flush=True)
    print(
        "Evolution helped" if evolutionary_best > random_best else "No clear advantage over random",
        flush=True,
    )


if __name__ == "__main__":
    main()
