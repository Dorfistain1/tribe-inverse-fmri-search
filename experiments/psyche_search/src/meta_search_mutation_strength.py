"""
Uses EvolutionarySearch itself to search for good mutation_strength
values, instead of a manual grid sweep -- reuses the exact same
generic engine (inverse_search.EvolutionarySearch) applied to a
different kind of "stimulus": a single scalar value instead of audio.

Each "candidate" is a mutation_strength value. Evaluating one runs
REPEATS_PER_CANDIDATE independent small fake-tier searches (see
sweep_mutation_strength.py) at that value and averages the improvement
(overall_best - gen0_best) -- multiple repeats, not one, specifically
because a single run is noisy (FINDINGS.md: 0.015 scored worse than
both its neighbors 0.02 and 0.01 in one run, almost certainly noise,
not a real non-monotonic effect). An outer search optimizing on a
single noisy run per candidate would just chase that noise.

Mutating a value happens in log-space (multiplicative: child = parent
* exp(gaussian noise)), not additive -- good values span three orders
of magnitude (0.001-0.5), so a fixed additive step would either do
nothing at the small end or overshoot wildly at the large end.

"Multiple bests" = elite_fraction > a single individual (SearchConfig
already supports this; POPULATION/ELITE_FRACTION below keep 3).

Same caveat as everywhere else in this experiment: this only searches
the FAKE landscape (raw latent distance), not TRIBE's real one -- a
precise answer here narrows where to look on the real search, it
isn't the final answer by itself.

Run:
    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\meta_search_mutation_strength.py
"""

from __future__ import annotations

import random
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import torch

from experiments.psyche_search.src.sweep_mutation_strength import analyze_manifest
from inverse_search import Candidate, EvolutionarySearch, SearchConfig
from inverse_search.fake_runtime import FakeTribeRuntime, latent_distance_fitness
from inverse_search.generators.base import StimulusGenerator
from inverse_search.generators.fake_audio import FakeLatentGenerator
from tribe_core import Stimulus

DURATION_S = 5.0
TARGET_SEED_BASE = 999_999
INNER_POPULATION = 6
INNER_GENERATIONS = 20
REPEATS_PER_CANDIDATE = 3

# Outer search over mutation_strength itself.
OUTER_POPULATION = 10
OUTER_GENERATIONS = 6
OUTER_ELITE_FRACTION = 0.3  # n_elite = round(10*0.3) = 3 -- "multiple bests"
LOG_MUTATION_SIGMA = 0.4  # child = parent * exp(N(0, sigma)) -- ~40% typical multiplicative step
VALUE_RANGE = (1e-4, 0.05)  # clip range; signal lives well below the 0.5 default (FINDINGS.md)


def _measure_improvement(mutation_strength: float, seed_offset: int) -> float:
    """One inner fake search at this mutation_strength -- returns
    overall_best - gen0_best, via the same manifest.csv analysis
    sweep_mutation_strength.py uses (search.run()'s returned population
    is only the final generation, not full history, so gen0's best
    can't be read off it directly once elites replace it). seed_offset
    decorrelates repeats so they're not literally identical runs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_gen = FakeLatentGenerator(
            duration_s=DURATION_S, mutation_strength=mutation_strength, output_dir=tmp_dir
        )
        fake_gen._counter = seed_offset * 1000
        target_latent = FakeLatentGenerator(duration_s=DURATION_S)._random_latent(
            seed=TARGET_SEED_BASE + seed_offset
        )
        runtime_ = FakeTribeRuntime()
        config = SearchConfig(population_size=INNER_POPULATION, n_generations=INNER_GENERATIONS)
        search = EvolutionarySearch(
            runtime=runtime_,
            generator=fake_gen,
            target=target_latent,
            fitness_fn=latent_distance_fitness,
            config=config,
        )
        search.run()
        result = analyze_manifest(Path(tmp_dir) / "manifest.csv")
        return result["overall_best"] - result["gen0_best"]


@dataclass
class _ScalarPrediction:
    activity: torch.Tensor


def scalar_fitness(prediction: _ScalarPrediction, target) -> float:
    return prediction.activity.item()


class MetaRuntime:
    """The outer search's "runtime": evaluating a mutation_strength
    candidate means actually running REPEATS_PER_CANDIDATE inner fake
    searches at that value and averaging their improvement."""

    model = None

    def predict(self, stimulus: Stimulus) -> _ScalarPrediction:
        value = stimulus.metadata["value"]
        improvements = [
            _measure_improvement(value, seed_offset=i) for i in range(REPEATS_PER_CANDIDATE)
        ]
        return _ScalarPrediction(activity=torch.tensor(statistics.mean(improvements)))

    def unload(self):
        pass


class MutationStrengthGenerator(StimulusGenerator):
    """Genome is a single float (mutation_strength), mutated
    multiplicatively in log-space."""

    # Stimulus only allows "audio"/"text" (tribe_core/stimulus.py) --
    # "text" here is arbitrary and unused as such, just needs to not be
    # "audio" so evaluate() skips its audio-only special-casing
    # (_skip_audio_transcription etc), which doesn't apply to a scalar.
    modality = "text"

    def _wrap(self, value: float) -> Stimulus:
        value = max(VALUE_RANGE[0], min(VALUE_RANGE[1], value))
        return Stimulus(
            identifier=f"mut_{value:.6f}",
            modality="text",
            source="<no-file>",  # never "" -- see generators/fake_audio.py's comment on why
            metadata={"value": value},
        )

    def initial_population(self, n: int) -> list[Stimulus]:
        # Log-uniform over the whole plausible range for the initial
        # spread -- not log-normal like mutate(), since we don't have a
        # "center" to start from yet.
        import math

        lo, hi = math.log(VALUE_RANGE[0]), math.log(VALUE_RANGE[1])
        return [self._wrap(math.exp(random.uniform(lo, hi))) for _ in range(n)]

    def mutate(self, parent: Candidate) -> Stimulus:
        import math

        parent_value = parent.stimulus.metadata["value"]
        child_value = parent_value * math.exp(random.gauss(0, LOG_MUTATION_SIGMA))
        return self._wrap(child_value)


def print_progress(message: str) -> None:
    print(message, flush=True)


def main():
    generator = MutationStrengthGenerator()
    runtime = MetaRuntime()
    config = SearchConfig(
        population_size=OUTER_POPULATION,
        n_generations=OUTER_GENERATIONS,
        elite_fraction=OUTER_ELITE_FRACTION,
    )

    print("=" * 70)
    print("Meta-search: evolving mutation_strength itself on the fake tier")
    print(
        f"outer: population={OUTER_POPULATION} generations={OUTER_GENERATIONS} "
        f"elites_kept={round(OUTER_POPULATION * OUTER_ELITE_FRACTION)}"
    )
    print(f"inner: population={INNER_POPULATION} generations={INNER_GENERATIONS} "
          f"repeats={REPEATS_PER_CANDIDATE}")
    print("=" * 70)
    print()

    search = EvolutionarySearch(
        runtime=runtime,
        generator=generator,
        target="unused",  # scalar_fitness ignores this; EvolutionarySearch requires non-None
        fitness_fn=scalar_fitness,
        config=config,
    )
    final_population = search.run(progress=print_progress)

    print()
    print("=== final population, best (highest avg improvement) first ===")
    for c in final_population:
        value = c.stimulus.metadata["value"]
        print(f"  gen={c.generation} mutation_strength={value:.5f} avg_improvement={c.fitness:+.4f}")


if __name__ == "__main__":
    main()
