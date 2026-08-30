"""
Watcher prototypes (see inverse_search/DESIGN.md's deferred v2
section) -- built and tested on the free fake tier first, not real GPU
time, per the testing-pyramid principle the rest of this project
follows.

Both keep a running history of every (latent, fitness) pair evaluated
so far (record_result(), generators/base.py's hook) and, once there's
enough of it, fit PCA down to a handful of components (raw ~164,000-dim
latents are hopeless for any surrogate directly -- FINDINGS.md's
curse-of-dimensionality entry is exactly why) plus a ridge regression
predicting fitness from the PCA-reduced latent. They differ in what
they DO with that surrogate:

- SurrogateFakeLatentGenerator (v1): generates several candidate
  mutations via plain random noise, and picks whichever the surrogate
  predicts will score highest. Only as good as its luck landing a good
  candidate in the batch -- tested in compare_watcher_fake.py and lost
  to plain blind mutation (FINDINGS.md).
- GradientSurrogateFakeLatentGenerator (v2): instead of filtering
  random guesses, uses the *fitted linear direction* itself -- since
  fitness ~= w . PCA(latent) + b, the gradient of that w.r.t. the raw
  latent is components.T @ w (a real, computable direction, not a
  guess) -- and steps the mutation along it (plus, still, some
  isotropic noise for exploration/history diversity). This is the
  more standard "surrogate-assisted" pattern: use the model to pick a
  direction, not to referee random guesses.

Both fall back to plain random mutation until there's enough history
(min_history_for_surrogate) to fit anything meaningful.
"""

from __future__ import annotations

import numpy as np
import torch

from inverse_search.candidate import Candidate
from inverse_search.generators.fake_audio import FakeLatentGenerator
from tribe_core import Stimulus


class _SurrogateBase(FakeLatentGenerator):
    def __init__(
        self,
        *args,
        min_history_for_surrogate: int = 10,
        n_pca_components: int = 5,
        ridge_alpha: float = 10.0,
        max_history: int = 200,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_history_for_surrogate = min_history_for_surrogate
        self.n_pca_components = n_pca_components
        self.ridge_alpha = ridge_alpha
        # PCA cost on wide data grows with history size even fit once
        # per generation -- capped to the most recent max_history
        # samples so a run of hundreds of generations doesn't have
        # fit cost growing unboundedly. Also arguably reasonable on its
        # own terms: older mutations came from a less-informed
        # surrogate, so recency-weighting isn't purely a speed hack.
        self.max_history = max_history
        self._history_latents: list[np.ndarray] = []
        self._history_fitness: list[float] = []
        # Refitting PCA+Ridge on the full history is expensive (roughly
        # quadratic in history size for wide data like this) -- caching
        # by history length means it only actually refits when new data
        # has arrived since the last fit, not once per mutate() call.
        # Without this, every child within the same generation redundantly
        # refits on identical data -- measured to make even a 30-
        # generation run impractically slow.
        self._surrogate_cache: tuple[int, tuple] | None = None

    def record_result(self, candidate: Candidate) -> None:
        latent = candidate.stimulus.metadata.get("latent")
        if latent is None or candidate.fitness is None:
            return
        self._history_latents.append(latent.detach().cpu().numpy().ravel())
        self._history_fitness.append(candidate.fitness)

    def _fit_surrogate(self):
        history_len = len(self._history_latents)
        if self._surrogate_cache is not None and self._surrogate_cache[0] == history_len:
            return self._surrogate_cache[1]

        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge

        X = np.stack(self._history_latents[-self.max_history:])
        y = np.array(self._history_fitness[-self.max_history:])
        # Few PCA components, strong ridge regularization -- deliberate,
        # not the defaults: with only tens to low hundreds of samples in
        # a space this large, a richer model just fits noise (see v1's
        # underperformance in FINDINGS.md, likely from exactly this).
        n_components = min(self.n_pca_components, len(X) - 1, X.shape[1])
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        model = Ridge(alpha=self.ridge_alpha)
        model.fit(X_reduced, y)

        self._surrogate_cache = (history_len, (pca, model))
        return pca, model

    def _has_enough_history(self) -> bool:
        return len(self._history_latents) >= self.min_history_for_surrogate


class SurrogateFakeLatentGenerator(_SurrogateBase):
    """v1: pick the best-predicted of several random candidates."""

    def __init__(self, *args, n_candidates: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_candidates = n_candidates

    def mutate(self, parent: Candidate) -> Stimulus:
        if not self._has_enough_history():
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


class GradientSurrogateFakeLatentGenerator(_SurrogateBase):
    """v2: step along the surrogate's fitted gradient direction instead
    of filtering random guesses.

    fitness ~= w . PCA(latent - mean) + b (Ridge is linear in the
    reduced space), so d(fitness)/d(latent) = components.T @ w -- a
    real direction in the full ~164,000-dim space, computed once per
    mutation from whatever data exists so far. child = parent +
    step_scale * unit_gradient_direction + (still) isotropic noise at
    mutation_strength, so there's always some exploration even if the
    surrogate's direction is wrong.
    """

    def __init__(self, *args, step_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_scale = step_scale

    def mutate(self, parent: Candidate) -> Stimulus:
        if not self._has_enough_history():
            return super().mutate(parent)

        pca, model = self._fit_surrogate()
        parent_latent = parent.stimulus.metadata["latent"]

        gradient = pca.components_.T @ model.coef_  # shape: (n_features,)
        norm = np.linalg.norm(gradient)
        if norm < 1e-12:
            return super().mutate(parent)
        gradient = gradient / norm

        step = torch.from_numpy(gradient).to(dtype=parent_latent.dtype).reshape(
            parent_latent.shape
        )
        noise = torch.randn_like(parent_latent) * self.mutation_strength
        child_latent = (parent_latent + self.step_scale * step + noise).to(torch.float16)
        return self._wrap(child_latent)
