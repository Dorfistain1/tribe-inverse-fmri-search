from __future__ import annotations

from abc import ABC, abstractmethod

from inverse_search.candidate import Candidate
from tribe_core import Stimulus


class StimulusGenerator(ABC):
    """Turns candidates into new stimuli for the next generation.

    This is the open design question that shapes the whole search
    engine: what does "slightly different audio" or "slightly different
    text" actually mean? Options range from raw waveform/token
    perturbation to mutating parameters of a generative model to
    latent-space interpolation in something like a VAE or diffusion
    model -- each implies a very different search space and is a design
    decision on its own, not something to guess at here. See
    concrete subclasses (audio.py, text.py) -- currently stubs.
    """

    modality: str

    @abstractmethod
    def initial_population(self, n: int) -> list[Stimulus]:
        """Produce n starting stimuli (generation 0)."""
        raise NotImplementedError

    @abstractmethod
    def mutate(self, parent: Candidate) -> Stimulus:
        """Produce one new stimulus derived from a parent candidate."""
        raise NotImplementedError

    def load(self) -> None:
        """Load any heavy model this generator needs. No-op by default
        -- override if there's VRAM to manage (see generators/audio.py).
        EvolutionarySearch calls this/unload() around generation phases
        so a generator and TribeRuntime never need to be resident at
        the same time on hardware too small for both (DESIGN.md)."""

    def unload(self) -> None:
        """Free whatever load() loaded. No-op by default."""

    def resume_from(self, population: list[Candidate]) -> None:
        """Called once, right after loading a checkpoint (search.py's
        run()), before generating anything new. Lets a generator avoid
        identifier collisions with candidates from before the pause --
        e.g. a fresh instance's internal counter starting back at 0 and
        reusing an identifier some *different*, already-discarded
        candidate from before the pause also had. That's not just
        cosmetic: TribeRuntime's prediction cache is keyed on
        stimulus.identifier, so a reused identifier can silently return
        a cached prediction for the wrong audio. No-op by default --
        override if the generator has internal state like a counter."""

    def record_result(self, candidate: Candidate) -> None:
        """Called by EvolutionarySearch right after a candidate is
        evaluated (candidate.fitness is already set). No-op by default
        -- override to build up history for something like a surrogate
        model (see generators/watcher_fake.py's SurrogateFakeLatentGenerator)."""

    def on_generation_result(self, improved: bool) -> None:
        """Called once per generation by EvolutionarySearch.run(),
        right after that generation's best-so-far is known, with
        whether it beat the best-so-far from before this generation.
        No-op by default -- override for e.g. an adaptive
        mutation_strength schedule (shrink the step size when a
        generation fails to improve -- the "1/5 success rule" pattern
        from evolution strategies)."""
