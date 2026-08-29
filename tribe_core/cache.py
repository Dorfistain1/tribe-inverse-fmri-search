from __future__ import annotations

import hashlib
from pathlib import Path

from tribe_core.prediction import BrainPrediction
from tribe_core.stimulus import Stimulus


class PredictionCache:
    """Reuses predictions keyed on stimulus identity + model/config
    identity, so a config change never silently returns a stale result.
    See mainStructure.md "Prediction Cache"."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, stimulus: Stimulus, model_identity: str) -> str:
        payload = f"{stimulus.identifier}|{stimulus.modality}|{model_identity}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(self, stimulus: Stimulus, model_identity: str) -> BrainPrediction | None:
        path = self.root / self.key(stimulus, model_identity)
        if not path.with_suffix(".npy").exists():
            return None
        return BrainPrediction.load(path)

    def put(self, prediction: BrainPrediction, stimulus: Stimulus) -> None:
        path = self.root / self.key(stimulus, prediction.model_identity)
        prediction.save(path)
