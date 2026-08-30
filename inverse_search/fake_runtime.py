"""
FakeTribeRuntime: instant stand-in for TribeRuntime.

Skips the real model entirely -- .predict() returns a lightweight
object exposing the candidate's own latent as `.activity`, so a fitness
function can score a candidate by its raw latent content instead of a
real brain prediction. Paired with generators/fake_audio.py's
FakeLatentGenerator and latent_distance_fitness below for a near-zero-
cost mechanism test of EvolutionarySearch -- see
experiments/psyche_search/src/run_fake_search_cli.py and DESIGN.md's
"testing pyramid" section for what this can and can't tell you (in
short: whether the search algorithm itself is sound, not whether
TRIBE's real fitness landscape behaves the same way).

Duck-typed against search.py's evaluate(), not a subclass of
TribeRuntime -- .model is accessed there as a force-load side effect
for audio candidates and just needs to not raise, .predict()/.unload()
are the only calls actually made.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tribe_core import Stimulus


@dataclass
class FakePrediction:
    activity: torch.Tensor


class FakeTribeRuntime:
    model = None

    def predict(self, stimulus: Stimulus) -> FakePrediction:
        return FakePrediction(activity=stimulus.metadata["latent"])

    def unload(self) -> None:
        pass


def latent_distance_fitness(prediction: FakePrediction, target_latent: torch.Tensor) -> float:
    """Higher is better (closer to target_latent), matching every other
    fitness_fn's maximize convention -- negative L2 distance. Only
    meaningful for comparing candidates against the same fixed target
    within one run; not normalized, so don't compare this value across
    different targets/durations."""
    return -torch.norm(prediction.activity.float() - target_latent.float()).item()
