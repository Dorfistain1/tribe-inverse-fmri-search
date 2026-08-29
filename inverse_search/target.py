from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NeuralTarget:
    """A desired pattern of network-level brain organization to search
    toward -- not a single-vertex activation map. Psychedelic states
    are most reliably characterized by changes in *relationships
    between* brain networks (functional connectivity), not amplitude at
    individual locations -- see experiments/psyche_search/restructure.md
    section 1. This is entirely experiment-defined: inverse_search only
    knows how to score a candidate against one of these, never what it
    represents scientifically (e.g. "psychedelic-like").

    network_deltas: {(network_a, network_b): direction}, where
    direction is +1 (this pair's connectivity should be higher) or -1
    (should be lower). Network names must be from
    brain_utils.YEO7_NETWORKS. This is a literature-encoded placeholder
    target (hand-specified from published findings, not measured from
    real psychedelic fMRI data) -- see restructure.md's "target source"
    discussion. A future version can replace this with deltas computed
    directly from a real dataset (e.g. PsiConnect's psilocybin+music
    minus baseline+music contrast) without changing this class's shape.
    """

    name: str
    network_deltas: dict[tuple[str, str], float]
    metadata: dict = field(default_factory=dict)
