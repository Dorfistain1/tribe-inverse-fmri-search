from __future__ import annotations

from dataclasses import dataclass, field

SUPPORTED_MODALITIES = ("audio", "text")


@dataclass
class Stimulus:
    """Standard stimulus representation. See mainStructure.md
    "Standard Stimulus Representation" -- tribe_core preserves
    `metadata` but never interprets it; that's for experiments."""

    identifier: str
    modality: str
    source: str  # path to an audio file or a .txt file
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality '{self.modality}'. "
                f"Supported: {SUPPORTED_MODALITIES}"
            )
