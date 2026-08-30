"""
A fake target designed to match the REAL task's structure, not just
its dimensionality -- unlike every other fake target used so far
today (a "match this one exact point in the full ~164,000-dim latent
space" objective, see FINDINGS.md), which turned out to be a much
harder, differently-shaped problem than the real one.

network_delta_score (fitness.py) doesn't reward matching a specific
point at all -- it rewards pushing a FEW specific, bounded summary
values (correlations between specific brain networks, each naturally
in [-1, 1]) as far as possible in a given direction. Many different
audios could score well under that objective (anything that happens to
push those few readouts far enough), not one exact optimum buried in
an enormous space -- a fundamentally easier shape of problem than
distance-to-a-point.

This fake target mirrors that: a small number of FIXED random
"readout" directions in latent space (N_RELEVANT, matching how few
network pairs a real target usually specifies -- see
build_psychedelic_target in experiments/psyche_search/src/run_search.py,
2-3 pairs), each projected and squashed through tanh to bound it to
[-1, 1] like a real correlation value, rewarded via the same
direction-times-value structure network_delta_score actually uses.

Everything else in the latent (all ~164,000 - N_RELEVANT other
directions) is pure noise as far as fitness is concerned -- sparse,
low-effective-dimensional structure, unlike the dense target-matching
test (every one of the 164,000 dimensions mattered equally there).
This also directly tests something that test couldn't: can a
PCA-based surrogate actually recover a genuinely low-rank relevant
subspace from noisy evaluation history -- exactly the structure it's
suited for, and exactly what the real problem plausibly looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SparseReadoutTarget:
    directions: list[np.ndarray]  # each a unit vector, shape (n_features,)
    signs: list[int]  # +1 or -1, matching network_delta_score's "direction"
    scale: float = 1.0  # see make_sparse_readout_target's docstring for why 1.0 is principled


def make_sparse_readout_target(
    n_features: int, n_relevant: int = 3, scale: float = 1.0, seed: int = 12345
) -> SparseReadoutTarget:
    """scale=1.0 isn't arbitrary: the latent has ~unit variance per
    coordinate, so a random unit direction dotted with it is itself
    ~N(0,1) -- scale=1 means a typical, totally unrelated latent
    already produces a moderately-saturated readout (tanh(N(0,1))),
    which is also realistic: real Pearson correlations of finite noisy
    signals can look substantial by chance too, not just near zero."""
    rng = np.random.default_rng(seed)
    directions = []
    for _ in range(n_relevant):
        v = rng.normal(size=n_features)
        v /= np.linalg.norm(v)
        directions.append(v)
    signs = [1 if rng.random() < 0.5 else -1 for _ in range(n_relevant)]
    return SparseReadoutTarget(directions=directions, signs=signs, scale=scale)


def sparse_readout_fitness(prediction, target: SparseReadoutTarget) -> float:
    """Mean of sign * tanh(direction . latent / scale) over each
    relevant direction -- same "push it as far as possible in a
    direction" shape as network_delta_score, bounded to roughly
    [-1, 1] like real correlation values. 0 is what an unrelated random
    latent typically scores; +1 is the theoretical ceiling (every
    readout fully saturated in its rewarded direction)."""
    latent_flat = prediction.activity.detach().cpu().numpy().ravel().astype(np.float64)
    total = 0.0
    for direction, sign in zip(target.directions, target.signs):
        projection = np.dot(direction, latent_flat) / target.scale
        readout = np.tanh(projection)
        total += sign * readout
    return total / len(target.directions)
