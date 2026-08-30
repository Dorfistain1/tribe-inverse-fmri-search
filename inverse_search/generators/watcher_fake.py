"""
First iteration of "the Watcher" (see inverse_search/DESIGN.md's
deferred v2 section) -- built and tested on the free fake tier first,
not real GPU time, per the testing-pyramid principle the rest of this
project follows.

Instead of blindly perturbing a parent's latent with isotropic
Gaussian noise (AudioGenerator/FakeLatentGenerator's approach), this:

1. Keeps a running history of every (latent, fitness) pair evaluated
   so far, via the record_result() hook (generators/base.py).
2. Once there's enough history, fits PCA down to a handful of
   components -- raw ~164,000-dim latents are hopeless for any
   surrogate model directly (FINDINGS.md's curse-of-dimensionality
   entry is exactly why) -- and a simple ridge regression predicting
   fitness from the PCA-reduced latent.
3. When mutating, generates several candidate offsets (plain Gaussian
   noise, same as the blind approach), but picks whichever the
   surrogate predicts will score highest -- before ever spending a
   real evaluation on it. This is the cheap part: evaluating the
   surrogate on N candidates costs nothing, so it's free filtering
   before the one (expensive, real) evaluation.

Falls back to plain random mutation until there's enough history
(min_history_for_surrogate) to fit anything meaningful.

This is a real, if crude, guided-mutation strategy -- not the
acquisition-function/Bayesian-optimization version DESIGN.md
eventually sketches, but the smallest version that actually tests "does
knowing the local fitness landscape shape help pick better mutations
than blind noise," cheaply, before deciding whether the real version
is worth building.
"""

from __future__ import annotations

import numpy as np
import torch

from inverse_search.candidate import Candidate
from inverse_search.generators.fake_audio import FakeLatentGenerator
from tribe_core import Stimulus


class SurrogateFakeLatentGenerator(FakeLatentGenerator):
    def __init__(
        self,
        *args,
        min_history_for_surrogate: int = 10,
        n_pca_components: int = 10,
        n_candidates: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_history_for_surrogate = min_history_for_surrogate
        self.n_pca_components = n_pca_components
        self.n_candidates = n_candidates
        self._history_latents: list[np.ndarray] = []
        self._history_fitness: list[float] = []

    def record_result(self, candidate: Candidate) -> None:
        latent = candidate.stimulus.metadata.get("latent")
        if latent is None or candidate.fitness is None:
            return
        self._history_latents.append(latent.detach().cpu().numpy().ravel())
        self._history_fitness.append(candidate.fitness)

    def _fit_surrogate(self):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge

        X = np.stack(self._history_latents)
        y = np.array(self._history_fitness)
        n_components = min(self.n_pca_components, len(X) - 1, X.shape[1])
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        model = Ridge(alpha=1.0)
        model.fit(X_reduced, y)
        return pca, model

    def mutate(self, parent: Candidate) -> Stimulus:
        if len(self._history_latents) < self.min_history_for_surrogate:
            return super().mutate(parent)

        pca, model = self._fit_surrogate()
        parent_latent = parent.stimulus.metadata["latent"]

        candidates = []
        for _ in range(self.n_candidates):
            noise = torch.randn_like(parent_latent) * self.mutation_strength
            candidates.append((parent_latent + noise).to(torch.float16))

        flat = np.stack([c.detach().cpu().numpy().ravel() for c in candidates])
        predicted = model.predict(pca.transform(flat))
        best_idx = int(np.argmax(predicted))
        return self._wrap(candidates[best_idx])
