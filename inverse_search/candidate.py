from __future__ import annotations

from dataclasses import dataclass

from tribe_core import Stimulus


@dataclass
class Candidate:
    """One stimulus in the search population, plus enough lineage to
    reconstruct how the search got here."""

    stimulus: Stimulus
    generation: int
    parent_id: str | None = None
    fitness: float | None = None
