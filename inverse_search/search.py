from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from inverse_search import checkpoint
from inverse_search.candidate import Candidate
from inverse_search.fitness import network_delta_score
from inverse_search.generators.base import StimulusGenerator
from inverse_search.target import NeuralTarget
from tribe_core import BrainPrediction, TribeRuntime


@dataclass
class SearchConfig:
    # Defaults sized for this project's actual 8GB dev card, not
    # textbook GA values: population_size=20/n_generations=10 (the
    # previous defaults) measured out to ~7-8 hours end to end here,
    # because generator+TRIBE fully reload every generation (see
    # EvolutionarySearch._evaluate_batch's docstring) -- a ~2-2.5min
    # fixed tax per round regardless of population size, plus
    # ~2-2.5min/candidate. 6x5 = 30 evaluations, ~80min, a size someone
    # will actually wait for. Override for a real overnight run.
    population_size: int = 6
    n_generations: int = 5
    elite_fraction: float = 0.2


FitnessFn = Callable[[BrainPrediction, NeuralTarget], float]
ProgressFn = Callable[[str], None]


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

    def _evaluate_batch(
        self, candidates: list[Candidate], progress: ProgressFn | None = None
    ) -> None:
        """Generator and TRIBE together don't fit on hardware like this
        project's 8GB dev card (DESIGN.md's sequential-load plan), so
        this always unloads the generator before evaluating and unloads
        TRIBE after -- never both resident at once. A no-op on either
        side costs nothing (generators/base.py's default load/unload are
        no-ops; unload() is a no-op with no model loaded)."""
        self.generator.unload()
        for i, candidate in enumerate(candidates):
            self.evaluate(candidate)
            if progress:
                progress(
                    f"  evaluated {i + 1}/{len(candidates)}: "
                    f"{candidate.stimulus.identifier} fitness={candidate.fitness:+.4f}"
                )
        if self.runtime is not None:
            self.runtime.unload()

    def run(
        self,
        checkpoint_path: str | Path | None = None,
        progress: ProgressFn | None = None,
    ) -> list[Candidate]:
        """Run the full search, returning the final population sorted
        by fitness descending.

        checkpoint_path, if given, enables pause/resume:
          - if a checkpoint already exists there, resumes from it
            instead of starting fresh (population_size/n_generations/
            etc for the resumed run come from *this* SearchConfig, not
            anything stored in the checkpoint -- only progress is
            persisted, not parameters).
          - saves after every fully-completed generation, so a crash or
            an ungraceful kill (e.g. closing the terminal window rather
            than Ctrl+C) loses at most the in-progress generation, never
            everything already done.
          - on a clean KeyboardInterrupt (Ctrl+C), saves immediately and
            returns the current population instead of raising.

        progress, if given, is called with short human-readable status
        lines -- for a CLI or any other live display. Purely a UI hook.
        """

        def notify(msg: str) -> None:
            if progress:
                progress(msg)

        start_generation = 0
        population: list[Candidate] = []

        if checkpoint_path is not None and checkpoint.exists(checkpoint_path):
            start_generation, population = checkpoint.load(checkpoint_path)
            self.generator.resume_from(population)
            notify(
                f"resumed from checkpoint: generation {start_generation}, "
                f"{len(population)} candidates"
            )

        try:
            if not population:
                notify(f"generation 0/{self.config.n_generations}: generating initial population")
                self.generator.load()
                initial_stimuli = self.generator.initial_population(self.config.population_size)
                initial_batch = [Candidate(stimulus=s, generation=0) for s in initial_stimuli]

                notify(f"generation 0/{self.config.n_generations}: evaluating")
                self._evaluate_batch(initial_batch, notify)

                population = initial_batch
                start_generation = 1
                if checkpoint_path is not None:
                    checkpoint.save(checkpoint_path, start_generation, population)

            for generation in range(start_generation, self.config.n_generations):
                population.sort(key=lambda c: c.fitness, reverse=True)
                n_elite = max(1, round(len(population) * self.config.elite_fraction))
                elites = population[:n_elite]
                n_children = self.config.population_size - len(elites)

                notify(
                    f"generation {generation}/{self.config.n_generations}: "
                    f"generating {n_children} children from {len(elites)} elite(s)"
                )
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

                notify(f"generation {generation}/{self.config.n_generations}: evaluating")
                self._evaluate_batch(children, notify)

                population = elites + children
                if checkpoint_path is not None:
                    checkpoint.save(checkpoint_path, generation + 1, population)

                best = max(c.fitness for c in population)
                notify(
                    f"generation {generation}/{self.config.n_generations}: "
                    f"done, best fitness so far = {best:+.4f}"
                )

        except KeyboardInterrupt:
            if checkpoint_path is not None and population:
                checkpoint.save(checkpoint_path, start_generation, population)
                notify(f"\nPaused (Ctrl+C). Checkpoint saved to {checkpoint_path}.")
                notify("Run the same command again to resume.")
            else:
                notify("\nInterrupted before any generation finished -- nothing to resume from.")
            return sorted(population, key=lambda c: c.fitness, reverse=True)

        population.sort(key=lambda c: c.fitness, reverse=True)
        return population

    def run_random_baseline(self, progress: ProgressFn | None = None) -> list[Candidate]:
        """Same total evaluation budget as run() (population_size x
        n_generations candidates), but every candidate is a fresh
        random draw -- no selection, no mutation. Comparison baseline:
        per restructure.md section 15, "fitness improved" during a real
        search doesn't by itself show evolution helped -- it could just
        be lucky sampling. Run both and compare best-fitness-so-far
        curves; if random keeps pace with evolutionary, the mutation/
        selection loop isn't earning its cost.

        No checkpoint/resume support -- less critical for a comparison
        run than for the main search; add it the same way as run() if
        that changes.
        """

        def notify(msg: str) -> None:
            if progress:
                progress(msg)

        all_candidates: list[Candidate] = []
        for generation in range(self.config.n_generations):
            notify(f"baseline generation {generation}/{self.config.n_generations}: generating")
            self.generator.load()
            stimuli = self.generator.initial_population(self.config.population_size)
            batch = [Candidate(stimulus=s, generation=generation) for s in stimuli]
            notify(f"baseline generation {generation}/{self.config.n_generations}: evaluating")
            self._evaluate_batch(batch, notify)
            all_candidates.extend(batch)

        all_candidates.sort(key=lambda c: c.fitness, reverse=True)
        return all_candidates
