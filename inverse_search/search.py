from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from inverse_search.candidate import Candidate
from inverse_search.fitness import network_delta_score
from inverse_search.generators.base import StimulusGenerator
from inverse_search.target import NeuralTarget
from tribe_core import BrainPrediction, TribeRuntime


@dataclass
class SearchConfig:
    population_size: int = 20
    n_generations: int = 10
    elite_fraction: float = 0.2


FitnessFn = Callable[[BrainPrediction, NeuralTarget], float]


class EvolutionarySearch:
    """Generic evolutionary loop: generate candidates, score them
    against a target via TribeRuntime, select, repeat.

    Deliberately knows nothing about TRIBE internals (only calls
    runtime.predict()), what the target means scientifically (that's
    the experiment's job, see target.py), or how to mutate a stimulus
    (that's the generator's job, see generators/). This class is just
    the selection loop that ties those three pieces together -- see
    mainStructure.md "Experiment Contract" for why that boundary matters.

    Pass runtime=None (and no target needed then) to test generator +
    selection mechanics alone, with random fitness instead of a real
    TRIBE prediction -- useful for checking mutation/selection actually
    behaves sensibly without paying TRIBE's load cost, before wiring in
    the real brain-based fitness. Not a permanent mode to build on top
    of; swap in a real runtime + target once the mechanics look right.
    """

    def __init__(
        self,
        runtime: TribeRuntime | None,
        generator: StimulusGenerator,
        target: NeuralTarget | None = None,
        fitness_fn: FitnessFn = network_delta_score,
        config: SearchConfig | None = None,
    ):
        if runtime is not None and target is None:
            raise ValueError("target is required when runtime is given")
        self.runtime = runtime
        self.generator = generator
        self.target = target
        self.fitness_fn = fitness_fn
        self.config = config or SearchConfig()

    def evaluate(self, candidate: Candidate) -> float:
        if self.runtime is None:
            candidate.fitness = random.random()
            return candidate.fitness
        prediction = self.runtime.predict(candidate.stimulus)
        candidate.fitness = self.fitness_fn(prediction, self.target)
        return candidate.fitness

    def run(self) -> list[Candidate]:
        """Run the full search, returning the final population sorted
        by fitness descending."""
        population = [
            Candidate(stimulus=s, generation=0)
            for s in self.generator.initial_population(self.config.population_size)
        ]
        for candidate in population:
            self.evaluate(candidate)

        for generation in range(1, self.config.n_generations):
            population.sort(key=lambda c: c.fitness, reverse=True)
            n_elite = max(1, round(len(population) * self.config.elite_fraction))
            elites = population[:n_elite]

            next_population = list(elites)
            while len(next_population) < self.config.population_size:
                parent = random.choice(elites)
                child_stimulus = self.generator.mutate(parent)
                child = Candidate(
                    stimulus=child_stimulus,
                    generation=generation,
                    parent_id=parent.stimulus.identifier,
                )
                self.evaluate(child)
                next_population.append(child)

            population = next_population

        population.sort(key=lambda c: c.fitness, reverse=True)
        return population
