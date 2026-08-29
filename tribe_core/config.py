from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "tribe.yaml"


@dataclass
class TribeConfig:
    repo_id: str
    checkpoint_name: str
    supported_modalities: list[str]
    device: str
    num_workers: int
    model_root: str
    prediction_cache: str

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "TribeConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            repo_id=raw["model"]["repo_id"],
            checkpoint_name=raw["model"]["checkpoint_name"],
            supported_modalities=raw["supported_modalities"],
            device=raw["device"],
            num_workers=raw["num_workers"],
            model_root=raw["paths"]["model_root"],
            prediction_cache=raw["paths"]["prediction_cache"],
        )

    @property
    def model_identity(self) -> str:
        """Identifies model + checkpoint for cache keys and provenance."""
        return f"{self.repo_id}:{self.checkpoint_name}"

    def hf_cache_dir(self) -> str:
        return os.path.join(self.model_root, "huggingface")

    def torch_cache_dir(self) -> str:
        return os.path.join(self.model_root, "torch")

    def uv_cache_dir(self) -> str:
        return os.path.join(self.model_root, "uv_cache")

    def pip_cache_dir(self) -> str:
        return os.path.join(self.model_root, "pip_cache")

    def tribev2_cache_dir(self) -> str:
        return os.path.join(self.model_root, "tribev2_cache")

    def nilearn_data_dir(self) -> str:
        return os.path.join(self.model_root, "nilearn")
