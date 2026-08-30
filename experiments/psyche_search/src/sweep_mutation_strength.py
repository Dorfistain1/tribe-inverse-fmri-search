"""
Sweeps mutation_strength on the full-fake search tier (see
inverse_search/DESIGN.md's "testing pyramid" section) to find a value
where the fitness landscape isn't degenerate, before spending real GPU
time re-tuning against TRIBE.

Motivation: both the real evolutionary run (parent-child correlation
-0.115, see FINDINGS.md) and the first fake-tier run (mean mutation
delta -34.6, stdev 0.5 -- every mutation moved almost exactly the same
distance regardless of direction) point at mutation_strength=0.5 being
too large relative to the latent's ~164,000 dimensions, not a property
of TRIBE's specific landscape. This sweep is fast (~1s/value, no audio
decode at all) precisely because it only needs latent-space distances,
not real audio or real TRIBE.

Caveat (see DESIGN.md): whatever value looks best here isn't
guaranteed optimal on TRIBE's real landscape -- confirm with one real
run before trusting it, don't skip straight to using this value blind.

Run:
    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\sweep_mutation_strength.py
"""

import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.fake_runtime import FakeTribeRuntime, latent_distance_fitness
from inverse_search.generators.fake_audio import FakeLatentGenerator

DURATION_S = 5.0
TARGET_SEED = 999_999
# Zoomed in around 0.005 (the first value that showed a real
# correlation in the original wide sweep [0.5..0.005], see
# FINDINGS.md) instead of spanning orders of magnitude -- and more
# generations (20, not SearchConfig's default 5) since fake runs cost
# nothing and 5 generations may just be too short to show whether a
# value keeps improving or plateaus.
MUTATION_STRENGTHS = [0.02, 0.015, 0.01, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
N_GENERATIONS = 20
SWEEP_DIR = Path("experiments/psyche_search/data/mutation_sweep")


def analyze_manifest(manifest_path: Path):
    import csv

    rows = list(csv.DictReader(manifest_path.open()))
    by_id = {r["identifier"]: float(r["fitness"]) for r in rows}
    gen0_fitness = [float(r["fitness"]) for r in rows if r["generation"] == "0"]
    final_gen = max(int(r["generation"]) for r in rows)
    final_gen_fitness = [float(r["fitness"]) for r in rows if r["generation"] == str(final_gen)]

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

    best_seen_by_gen0 = max(gen0_fitness)
    overall_best = max(by_id.values())
    return {
        "gen0_best": best_seen_by_gen0,
        "overall_best": overall_best,
        "improvement": overall_best - best_seen_by_gen0,
        "correlation": corr,
        "n_pairs": len(pairs),
    }


def run_one(mutation_strength: float) -> dict:
    output_dir = SWEEP_DIR / f"strength_{mutation_strength}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fake_gen = FakeLatentGenerator(
        duration_s=DURATION_S, mutation_strength=mutation_strength, output_dir=str(output_dir)
    )
    target_latent = FakeLatentGenerator(duration_s=DURATION_S)._random_latent(seed=TARGET_SEED)
    runtime = FakeTribeRuntime()
    config = SearchConfig(n_generations=N_GENERATIONS)

    search = EvolutionarySearch(
        runtime=runtime,
        generator=fake_gen,
        target=target_latent,
        fitness_fn=latent_distance_fitness,
        config=config,
    )
    search.run()
    return analyze_manifest(output_dir / "manifest.csv")


def main():
    print(f"{'mutation_strength':>18} | {'gen0_best':>10} | {'overall_best':>12} | "
          f"{'improvement':>11} | {'parent-child corr':>18}")
    print("-" * 80)
    for strength in MUTATION_STRENGTHS:
        result = run_one(strength)
        print(
            f"{strength:>18.3f} | {result['gen0_best']:>10.2f} | "
            f"{result['overall_best']:>12.2f} | {result['improvement']:>11.2f} | "
            f"{result['correlation']:>18.3f}"
        )
    print()
    print("improvement > 0 means the search beat its own initial random population.")
    print("correlation near 0 (or negative) means mutation isn't meaningfully local --")
    print("a child's fitness doesn't predict from its parent's, so refinement can't work.")


if __name__ == "__main__":
    main()
