from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BrainPrediction:
    """Standard prediction representation. See mainStructure.md
    "Standard Prediction Representation" / "Raw vs Derived Data" --
    this object is the raw TRIBE output and should never be mutated by
    experiment code; compute derived metrics separately."""

    stimulus_id: str
    modality: str
    activity: np.ndarray  # shape (time, vertices)
    model_identity: str
    timing: list[dict] = field(default_factory=list)  # per-row segment info from TRIBE
    metadata: dict = field(default_factory=dict)

    @property
    def n_timesteps(self) -> int:
        return self.activity.shape[0]

    @property
    def n_vertices(self) -> int:
        return self.activity.shape[1]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self.activity)
        sidecar = {
            "stimulus_id": self.stimulus_id,
            "modality": self.modality,
            "model_identity": self.model_identity,
            "timing": self.timing,
            "metadata": self.metadata,
        }
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "BrainPrediction":
        path = Path(path)
        activity = np.load(path.with_suffix(".npy"))
        sidecar = json.loads(path.with_suffix(".json").read_text())
        return cls(activity=activity, **sidecar)
