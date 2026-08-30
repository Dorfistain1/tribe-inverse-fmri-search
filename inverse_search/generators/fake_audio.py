"""
FakeLatentGenerator: mechanism-only stand-in for AudioGenerator.

Same latent shape and mutation math as AudioGenerator (see audio.py)
but never touches Stable Audio Open -- no decode, no GPU, no disk
write. Exists purely to test EvolutionarySearch's selection/mutation
logic at near-zero cost, paired with FakeTribeRuntime (see
../fake_runtime.py) and a real AudioGenerator only for the handful of
reference clips actually worth listening to. See
experiments/psyche_search/src/run_fake_search_cli.py and DESIGN.md's
"testing pyramid" section for how this fits into the overall workflow.

Identifiers are content-hashed, same reasoning as AudioGenerator's
fix -- see its _identifier_for docstring.
"""

from __future__ import annotations

import hashlib

import torch

from inverse_search.candidate import Candidate
from inverse_search.generators.audio import LATENT_CHANNELS, LATENT_FRAMES_PER_SECOND
from inverse_search.generators.base import StimulusGenerator
from tribe_core import Stimulus


class FakeLatentGenerator(StimulusGenerator):
    modality = "audio"

    def __init__(
        self,
        duration_s: float = 5.0,
        mutation_strength: float = 0.5,
        device: str = "cpu",
        output_dir: str | None = None,
        mutation_decay: float | None = None,
        min_mutation_strength: float = 1e-4,
    ):
        self.duration_s = duration_s
        self.mutation_strength = mutation_strength
        self.device = device
        # Adaptive step size a la evolution strategies' "1/5 success
        # rule": shrink mutation_strength when a generation fails to
        # beat the previous best, instead of a fixed value for the
        # whole run. None (default) disables this -- mutation_strength
        # stays fixed, matching every generator's behavior before this
        # was added.
        self.mutation_decay = mutation_decay
        self.min_mutation_strength = min_mutation_strength
        # Optional, matching AudioGenerator's attribute of the same
        # name: if set, search.py's _organize_candidate_file logs a
        # manifest.csv row per candidate (identifier/generation/parent/
        # fitness) even though there's no real audio file to rename --
        # useful for e.g. computing parent-child fitness correlation on
        # the fake landscape the same way as a real run's data.
        self.output_dir = output_dir
        self._counter = 0

    @property
    def n_frames(self) -> int:
        return round(self.duration_s * LATENT_FRAMES_PER_SECOND)

    def _random_latent(self, seed: int) -> torch.Tensor:
        generator = torch.Generator(self.device).manual_seed(seed)
        return torch.randn(
            1, LATENT_CHANNELS, self.n_frames,
            generator=generator, device=self.device, dtype=torch.float16,
        )

    def _identifier_for(self, latent: torch.Tensor) -> str:
        digest = hashlib.sha256(latent.detach().cpu().numpy().tobytes()).hexdigest()[:12]
        return f"fake_{digest}"

    def _wrap(self, latent: torch.Tensor) -> Stimulus:
        # NOT "" -- Path("") resolves to "." (the current directory) and
        # Path("").exists() is True on Windows, so an empty source would
        # make _organize_candidate_file's rename() target the working
        # directory itself once output_dir is set. This placeholder
        # never resolves to a real path.
        identifier = self._identifier_for(latent)
        source = f"{self.output_dir}/{identifier}.fake" if self.output_dir else "<no-file>"
        return Stimulus(
            identifier=identifier,
            modality="audio",
            source=source,
            metadata={"latent": latent},
        )

    def initial_population(self, n: int) -> list[Stimulus]:
        stimuli = []
        for _ in range(n):
            seed = self._counter
            self._counter += 1
            stimuli.append(self._wrap(self._random_latent(seed)))
        return stimuli

    def mutate(self, parent: Candidate) -> Stimulus:
        parent_latent = parent.stimulus.metadata["latent"]
        noise = torch.randn_like(parent_latent) * self.mutation_strength
        return self._wrap((parent_latent + noise).to(torch.float16))

    def on_generation_result(self, improved: bool) -> None:
        if self.mutation_decay is not None and not improved:
            self.mutation_strength = max(
                self.min_mutation_strength, self.mutation_strength * self.mutation_decay
            )
