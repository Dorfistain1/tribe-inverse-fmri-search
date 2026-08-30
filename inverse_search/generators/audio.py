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

    duration_s default (5.0) reflects a real hardware ceiling on this
    8GB card, well short of Stable Audio Open's documented 47s max:
    generation time is fine through ~8s, but something (likely
    attention cost) scales badly beyond that -- 12s measured at ~8
    minutes for one clip, 20s+ hard OOMs. enable_tiling() "fixes" the
    OOM but made it worse (20+ minutes and still running), not a real
    fix. Keep clips short; that's also what keeps a full search
    generation fast enough to be usable.

    prompt default (plain "ambient music") reflects a real finding, not
    just a placeholder: prompts calling for crisp, precisely-timed
    drums (tested: "upbeat electronic dance music") produced noticeably
    worse, more incoherent output than sustained/melodic prompts
    (tested: "country acoustic guitar song" -- clearly better) at the
    same settings. Likely a general diffusion-audio-model weakness
    (precise transients are harder than sustained texture), not
    specific to this prompt wording. Conveniently, sustained/atmospheric
    material is also the natural fit for psyche_search's actual target
    anyway -- so favor ambient/sustained prompts over drum-heavy ones
    when choosing what to search over.
    """

    modality = "audio"

    def __init__(
        self,
        model_root: str = "G:/AI_Models",
        prompt: str = "ambient music",
        duration_s: float = 5.0,
        num_inference_steps: int = 40,
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
        # Only used to vary _random_latent's seed across calls -- NOT
        # part of the identifier (see _identifier_for). A plain counter
        # was tried for identifiers too (cand_0000, cand_0001, ...) but
        # caused a real bug: it resets to 0 in every fresh process,
        # while TribeRuntime's prediction cache is disk-persisted and
        # keyed on identifier alone (tribe_core/cache.py) -- so two
        # separate script runs (e.g. an evolutionary run followed by a
        # baseline run) starting their own counters at 0 would silently
        # return each other's cached predictions for unrelated audio
        # whenever the counters lined up. Discovered when a "random
        # baseline" run reproduced the evolutionary run's exact fitness
        # values almost everywhere.
        self._counter = 0

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
            # The VAE decoder upsamples ~86x (512 latent frames/s ->
            # 44100 samples/s) in one shot by default -- fine at 4s, but
            # OOMs on an 8GB card well before Stable Audio Open's
            # documented 47s max (measured: fails around 20s).
            # enable_tiling() "fixes" the OOM but is a bad trade here --
            # measured 20+ minutes and still running on a 20s clip, vs.
            # ~15s without it at durations that fit. Not worth it; stick
            # to whatever duration fits without tiling (see
            # experiments/psyche_search's notes for the measured ceiling
            # on this 8GB card).
            self._pipe.vae.enable_slicing()
        return self._pipe

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()

    def decode_latent(self, latent: torch.Tensor, identifier: str) -> Stimulus:
        """Public entry point for decoding a latent you already have in
        hand (e.g. a reference clip from a mechanism-testing script,
        see generators/fake_audio.py) -- unlike mutate()/
        initial_population(), doesn't touch self._counter or require a
        parent Candidate. Caller is responsible for matching shape/
        dtype to what the pipeline expects (float16,
        (1, LATENT_CHANNELS, duration_s * LATENT_FRAMES_PER_SECOND))."""
        return self._decode(latent, identifier)

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

    def _identifier_for(self, latent: torch.Tensor) -> str:
        """Content-derived, not sequential -- see the _counter comment
        in __init__ for the cross-run cache-collision bug this fixes.
        Same latent content always maps to the same identifier (a
        correct cache hit); different content is guaranteed a different
        identifier, regardless of process or run history."""
        digest = hashlib.sha256(latent.detach().cpu().numpy().tobytes()).hexdigest()[:12]
        return f"cand_{digest}"

    def initial_population(self, n: int) -> list[Stimulus]:
        # Seed from the running counter, not range(n) -- otherwise every
        # call to initial_population() produces the exact same n latents
        # (seed=0..n-1 every time), silently duplicating candidates
        # across repeated calls (e.g. a random-search baseline that
        # calls this once per "generation" instead of mutating).
        stimuli = []
        for _ in range(n):
            seed = self._counter
            self._counter += 1
            latent = self._random_latent(seed=seed)
            stimuli.append(self._decode(latent, identifier=self._identifier_for(latent)))
        return stimuli

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
        return self._decode(child_latent, identifier=self._identifier_for(child_latent))
