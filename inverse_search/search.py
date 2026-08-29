from __future__ import annotations

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

    Not yet runnable: generators are stubs (see generators/audio.py,
    generators/text.py) until a mutation strategy is chosen.
    """

    def __init__(
        self,
        runtime: TribeRuntime,
        generator: StimulusGenerator,
        target: NeuralTarget,
        fitness_fn: FitnessFn = network_delta_score,
        config: SearchConfig | None = None,
    ):
        self.runtime = runtime
        self.generator = generator
        self.target = target
        self.fitness_fn = fitness_fn
        self.config = config or SearchConfig()

    def evaluate(self, candidate: Candidate) -> float:
        prediction = self.runtime.predict(candidate.stimulus)
        candidate.fitness = self.fitness_fn(prediction, self.target)
        return candidate.fitness

    def run(self) -> list[Candidate]:
        """Run the full search, returning the final population sorted
        by fitness descending."""
        raise NotImplementedError(
            "Selection/mutation loop not yet implemented -- "
            f"{self.generator.__class__.__name__} has no mutation strategy yet."
        )
