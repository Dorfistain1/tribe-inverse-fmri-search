"""
AudioGenerator: Stable Audio Open, mutated by direct latent perturbation.

See inverse_search/DESIGN.md for why this model and this mutation
strategy (v1 -- deferred: surrogate-model-guided mutation).

Shape constants below were measured empirically, not read from docs:
loaded stabilityai/stable-audio-open-1.0, generated a 2s clip, and
inspected the actual latent tensor passed through the diffusion
callback -- shape (1, 64, 1024), i.e. 64 channels and 512 latent frames
per second of audio. If a future diffusers/model update changes this,
re-derive it the same way rather than guessing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from inverse_search.candidate import Candidate
from inverse_search.generators.base import StimulusGenerator
from tribe_core import Stimulus
from tribe_core.env import configure_hf_cache

LATENT_CHANNELS = 64
LATENT_FRAMES_PER_SECOND = 512
SAMPLE_RATE = 44100
MODEL_ID = "stabilityai/stable-audio-open-1.0"


class AudioGenerator(StimulusGenerator):
    """Candidates are decoded from a (1, 64, duration_s * 512) latent
    tensor. `initial_population` samples random latents; `mutate` adds
    small Gaussian noise to a parent's latent and redecodes -- the
    "dumb" evolutionary approach, deliberately, see DESIGN.md.

    The latent tensor is stashed in the returned Stimulus's `metadata`
    dict (tribe_core preserves but never interprets that field) so
    `mutate` has something to perturb later. This only works for
    candidates created by this class in the same process -- it's not
    serialized to disk, so a Stimulus loaded from a saved prediction
    file has no latent to mutate from.

    Sequential VRAM use, not concurrent with TRIBE (DESIGN.md): call
    `load()` before generating a batch, `unload()` before predicting
    that batch with TribeRuntime.

    mutation_strength default (0.5) picked by ear, not derived: 0.15
    was inaudibly close to identical, 0.3 barely distinguishable, 0.5
    clearly a related-but-different take, 0.8+ starting to feel like a
    fresh draw rather than a mutation. Re-tune if the generation prompt
    or duration changes substantially -- this was measured on one
    3s "ambient drone music" clip, not a general constant.
    """

    modality = "audio"

    def __init__(
        self,
        model_root: str = "G:/AI_Models",
        prompt: str = "ambient music",
        duration_s: float = 4.0,
        num_inference_steps: int = 50,
        mutation_strength: float = 0.5,
        output_dir: str | Path | None = None,
        device: str = "cuda",
    ):
        self.model_root = model_root
        self.prompt = prompt
        self.duration_s = duration_s
        self.num_inference_steps = num_inference_steps
        self.mutation_strength = mutation_strength
        self.device = device
        self.output_dir = Path(
            output_dir or "experiments/psyche_search/data/candidates"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._pipe = None

    @property
    def n_frames(self) -> int:
        return round(self.duration_s * LATENT_FRAMES_PER_SECOND)

    def load(self):
        """Loads the ~5.9GB-VRAM pipeline if not already resident.
        Call unload() before loading TRIBE in the same process."""
        if self._pipe is None:
            configure_hf_cache(self.model_root)
            from diffusers import StableAudioPipeline

            self._pipe = StableAudioPipeline.from_pretrained(
                MODEL_ID, dtype=torch.float16
            ).to(self.device)
        return self._pipe

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()

    def _random_latent(self, seed: int) -> torch.Tensor:
        generator = torch.Generator(self.device).manual_seed(seed)
        return torch.randn(
            1,
            LATENT_CHANNELS,
            self.n_frames,
            generator=generator,
            device=self.device,
            dtype=torch.float16,
        )

    def _decode(self, latents: torch.Tensor, identifier: str) -> Stimulus:
        import soundfile as sf

        pipe = self.load()
        result = pipe(
            prompt=self.prompt,
            audio_end_in_s=self.duration_s,
            num_inference_steps=self.num_inference_steps,
            latents=latents,
            generator=torch.Generator(self.device).manual_seed(0),
        )
        audio = result.audios[0].float().cpu().numpy().T  # (samples, channels)
        path = self.output_dir / f"{identifier}.wav"
        sf.write(str(path), audio, SAMPLE_RATE)

        return Stimulus(
            identifier=identifier,
            modality="audio",
            source=str(path),
            metadata={"latent": latents.detach().to("cpu")},
        )

    def initial_population(self, n: int) -> list[Stimulus]:
        return [
            self._decode(self._random_latent(seed=i), identifier=f"gen0_{i:04d}")
            for i in range(n)
        ]

    def mutate(self, parent: Candidate) -> Stimulus:
        parent_latent = parent.stimulus.metadata.get("latent")
        if parent_latent is None:
            raise ValueError(
                f"Stimulus '{parent.stimulus.identifier}' has no stored latent "
                "to mutate -- it wasn't created by AudioGenerator."
            )
        parent_latent = parent_latent.to(self.device)
        noise = torch.randn_like(parent_latent) * self.mutation_strength
        child_latent = (parent_latent + noise).to(torch.float16)

        digest = hashlib.sha256(child_latent.cpu().numpy().tobytes()).hexdigest()[:8]
        identifier = f"{parent.stimulus.identifier}_m{digest}"
        return self._decode(child_latent, identifier=identifier)
