from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
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


def _skip_audio_transcription() -> None:
    """Generated/searched audio candidates are always synthetic
    instrumental content with no real speech, but TRIBE's audio
    pipeline still tries to transcribe by default -- costing real time
    (~2+ minutes/candidate, measured) and risking a Whisper
    hallucination (observed: "Thank you.", a famous Whisper artifact on
    silence/music/non-speech input) getting fed through the text/LLaMA
    branch as if it were real spoken content, silently contaminating an
    audio-only prediction with phantom text.

    Forcing an empty transcript instead makes every candidate always
    take tribev2's own pre-existing "no speech found" code path --
    already proven correct on a real silent-audio test, nothing new to
    break. Empty DataFrame shape (`pd.DataFrame([])`, no columns)
    deliberately matches the original function's own empty-words return
    exactly, not just "an empty dataframe", to avoid any schema
    mismatch downstream.

    tribe_core's _windows_patches already overrides this same function
    (for an unrelated CPU/device bug), so this must be applied *after*
    that -- hence called from evaluate() right before predict() (once
    the model, and tribe_core's own patch, are guaranteed already
    loaded) rather than once at import time.
    """
    import pandas as pd
    from tribev2 import eventstransforms as _et

    def _empty_transcript(wav_filename, language):
        return pd.DataFrame([])

    _et.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(_empty_transcript)


def _organize_candidate_file(candidate: Candidate, generator: StimulusGenerator) -> None:
    """Cosmetic, for browsing: renames the candidate's file to show
    generation + fitness once known (e.g. cand_0003.wav ->
    g01_fit+0.423_cand_0003.wav), and appends a manifest.csv row with
    the full structured record (identifier, generation, parent, fitness,
    path) since lineage beyond one hop doesn't fit readably in a
    filename. Doesn't touch stimulus.identifier -- that's the stable id
    used for cache keys and parent_id links, only the display file.

    No-op if the generator doesn't expose output_dir (not all
    generators need one) or the file's already gone.
    """
    output_dir = getattr(generator, "output_dir", None)
    if output_dir is None or candidate.fitness is None:
        return
    output_dir = Path(output_dir)

    old_path = Path(candidate.stimulus.source)
    if old_path.exists():
        new_path = old_path.with_name(
            f"g{candidate.generation:02d}_fit{candidate.fitness:+.3f}_{old_path.name}"
        )
        old_path.rename(new_path)
        candidate.stimulus.source = str(new_path)

    manifest_path = output_dir / "manifest.csv"
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["identifier", "generation", "parent_id", "fitness", "source"])
        writer.writerow([
            candidate.stimulus.identifier,
            candidate.generation,
            candidate.parent_id or "",
            f"{candidate.fitness:.4f}",
            candidate.stimulus.source,
        ])


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
        else:
            if candidate.stimulus.modality == "audio":
                self.runtime.model  # force-load first so our patch below applies last
                _skip_audio_transcription()
            prediction = self.runtime.predict(candidate.stimulus)
            candidate.fitness = self.fitness_fn(prediction, self.target)
        _organize_candidate_file(candidate, self.generator)
        return candidate.fitness

    def _evaluate_batch(self, candidates: list[Candidate]) -> None:
        """Generator and TRIBE together don't fit on hardware like this
        project's 8GB dev card (DESIGN.md's sequential-load plan), so
        this always unloads the generator before evaluating and unloads
        TRIBE after -- never both resident at once. A no-op on either
        side costs nothing (generators/base.py's default load/unload are
        no-ops; unload() is a no-op with no model loaded)."""
        self.generator.unload()
        for candidate in candidates:
            self.evaluate(candidate)
        if self.runtime is not None:
            self.runtime.unload()

    def run(self) -> list[Candidate]:
        """Run the full search, returning the final population sorted
        by fitness descending."""
        self.generator.load()
        initial_stimuli = self.generator.initial_population(self.config.population_size)
        population = [Candidate(stimulus=s, generation=0) for s in initial_stimuli]
        self._evaluate_batch(population)

        for generation in range(1, self.config.n_generations):
            population.sort(key=lambda c: c.fitness, reverse=True)
            n_elite = max(1, round(len(population) * self.config.elite_fraction))
            elites = population[:n_elite]

            self.generator.load()
            children = []
            while len(elites) + len(children) < self.config.population_size:
                parent = random.choice(elites)
                child_stimulus = self.generator.mutate(parent)
                children.append(Candidate(
                    stimulus=child_stimulus,
                    generation=generation,
                    parent_id=parent.stimulus.identifier,
                ))
            self._evaluate_batch(children)

            population = elites + children

        population.sort(key=lambda c: c.fitness, reverse=True)
        return population

    def run_random_baseline(self) -> list[Candidate]:
        """Same total evaluation budget as run() (population_size x
        n_generations candidates), but every candidate is a fresh
        random draw -- no selection, no mutation. Comparison baseline:
        per restructure.md section 15, "fitness improved" during a real
        search doesn't by itself show evolution helped -- it could just
        be lucky sampling. Run both and compare best-fitness-so-far
        curves; if random keeps pace with evolutionary, the mutation/
        selection loop isn't earning its cost.
        """
        all_candidates: list[Candidate] = []
        for generation in range(self.config.n_generations):
            self.generator.load()
            stimuli = self.generator.initial_population(self.config.population_size)
            batch = [Candidate(stimulus=s, generation=generation) for s in stimuli]
            self._evaluate_batch(batch)
            all_candidates.extend(batch)

        all_candidates.sort(key=lambda c: c.fitness, reverse=True)
        return all_candidates
