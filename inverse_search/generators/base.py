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
