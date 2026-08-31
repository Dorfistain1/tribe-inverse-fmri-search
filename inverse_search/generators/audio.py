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

    prompt default is "acoustic guitar song with a clear, memorable
    melody" -- NOT the earlier "ambient music", which turned out to
    deterministically produce near-silent output on this model (peak
    amplitude ~6e-5, i.e. digital silence, reproduced twice on the same
    latent). Discovered via a fake-search reference-clip decode that
    came out with nothing audible; swapping only the prompt on the
    identical latent fixed it (peak jumped to 0.28). Root cause
    unconfirmed -- plausibly the model's training data tags some very
    quiet/droning tracks as "ambient", so it's not necessarily a bug in
    the model so much as a bad prompt choice on our part. Whatever the
    cause, don't use "ambient music" verbatim as a prompt with this
    model.

    More generally: prompts calling for crisp, precisely-timed drums
    (tested: "upbeat electronic dance music") produced noticeably
    worse, more incoherent output than sustained/melodic prompts
    (tested: "country acoustic guitar song" -- clearly better) at the
    same settings. Likely a general diffusion-audio-model weakness
    (precise transients are harder than sustained texture), not
    specific to this prompt wording.
    """

    modality = "audio"

    def __init__(
        self,
        model_root: str = "G:/AI_Models",
        prompt: str = "acoustic guitar song with a clear, memorable melody",
        duration_s: float = 5.0,
        num_inference_steps: int = 40,
        mutation_strength: float = 0.5,
        output_dir: str | Path | None = None,
        device: str = "cuda",
        mutation_decay: float | None = None,
        min_mutation_strength: float = 1e-4,
        stall_patience: int = 5,
        mutation_mode: str = "latent",
        redo_fraction: float = 0.3,
    ):
        self.model_root = model_root
        self.prompt = prompt
        self.duration_s = duration_s
        self.num_inference_steps = num_inference_steps
        self.mutation_strength = mutation_strength
        self.device = device
        # "latent" (default): mutate()'s original raw-initial-noise
        # perturbation. "rediffusion": mutate_by_rediffusion()'s
        # partial-re-diffusion img2img-style edit instead -- see that
        # method's docstring for why it was added (untested as of
        # this constructor change; verify before relying on it).
        self.mutation_mode = mutation_mode
        self.redo_fraction = redo_fraction
        # Adaptive step size (evolution strategies' "1/5 success
        # rule"): shrink mutation_strength when a generation fails to
        # beat the previous best. None (default) keeps mutation_strength
        # fixed, matching this class's behavior before this was added --
        # see generators/fake_audio.py's FakeLatentGenerator, where this
        # was tested first (cheap, no GPU) before adding it here too.
        self.mutation_decay = mutation_decay
        self.min_mutation_strength = min_mutation_strength
        # Only decay after this many CONSECUTIVE non-improving
        # generations, not on every single one -- measured directly
        # (watcher_fake.py's step_scale_decay, FINDINGS.md) that
        # decaying on every stall collapses the value within ~18
        # generations even during normal exploration.
        self.stall_patience = stall_patience
        self._consecutive_stalls = 0
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
        if self.mutation_mode == "rediffusion":
            return self.mutate_by_rediffusion(parent, redo_fraction=self.redo_fraction)
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

    def on_generation_result(self, improved: bool) -> None:
        if self.mutation_decay is None:
            return
        if improved:
            self._consecutive_stalls = 0
            return
        self._consecutive_stalls += 1
        if self._consecutive_stalls >= self.stall_patience:
            self.mutation_strength = max(
                self.min_mutation_strength, self.mutation_strength * self.mutation_decay
            )
            self._consecutive_stalls = 0

    @torch.no_grad()
    def mutate_by_rediffusion(self, parent: Candidate, redo_fraction: float = 0.3) -> Stimulus:
        """Real partial-re-diffusion mutation (SDEdit/img2img-style),
        as an alternative to mutate()'s raw-initial-noise perturbation.

        UNTESTED as of the commit that added this -- built at the end
        of a long session specifically to be verified next time this
        project is picked up. Ported by hand from diffusers' standard
        img2img pattern (e.g. StableDiffusionImg2ImgPipeline) since
        StableAudioPipeline doesn't expose a `strength` parameter of
        its own. Its own `initial_audio_waveforms` input does NOT do
        this despite the superficial similarity -- it adds encoded
        audio as a bias under FULL-strength noise and still runs every
        step (see diffusers' pipeline_stable_audio.py's
        prepare_latents), which is really the audio-continuation
        mechanism DESIGN.md's duration-extension plan already uses,
        not a controllable small-edit tool. Confirmed by reading that
        source directly, not assumed.

        Why this exists: mutate() adds Gaussian noise to the INITIAL
        noise latent (x_T) and re-runs the FULL num_inference_steps
        denoising trajectory from scratch. Diffusion sampling is a
        long, highly nonlinear iterative process, so even a small
        change to x_T can produce a wildly different, seemingly
        unrelated final result -- the empirical pattern across this
        whole project (curse-of-dimensionality mutation tuning, the
        sparse-readout "best" sounding nearly identical to "start"
        despite 99% fitness, real fitness not correlating with basic
        acoustic features) is consistent with this being the actual
        root cause, not a search-algorithm weakness (see FINDINGS.md).

        This method instead: takes the PARENT'S ACTUAL DECODED AUDIO
        (not its stored initial-noise latent), encodes it back into
        latent space with the same VAE the pipeline already uses, adds
        a controlled amount of noise corresponding to a specific point
        partway through the denoising schedule, and re-runs only the
        REMAINING steps from there. redo_fraction is a direct,
        controllable dial on edit magnitude that raw-noise mutation
        never had: near 0 keeps the result close to the parent, near 1
        is close to a fresh generation (all num_inference_steps steps
        redone from full noise).

        Returned Stimulus's metadata still has a "latent" key, but it
        is now the CLEAN, decoded-space latent this method produced
        (not an initial-noise latent) -- mixing outputs of this method
        and mutate() in the same lineage would be mutating
        incompatible things. Pick one mutation strategy per run.
        """
        import soundfile as sf
        from diffusers.models.embeddings import get_1d_rotary_pos_embed

        pipe = self.load()
        device = self.device
        guidance_scale = 7.0  # matches _decode()'s pipe() calls, which never override this default
        do_cfg = guidance_scale > 1.0

        # This card is already tight (8GB, ~5.9GB resident for the
        # loaded pipe -- see load()'s docstring). gc.collect() +
        # empty_cache() alone were NOT enough -- measured directly:
        # three separate attempts (with/without these, at two
        # durations) all OOM'd with "0 bytes free" and a near-identical
        # ~22.5GB "allocated by PyTorch" figure, unmoved by cache-
        # clearing. That ruled out fragmentation as the cause -- this
        # is genuine peak-memory pressure. Real fix: the transformer
        # (this pipe's largest resident component) isn't needed for a
        # VAE-only encode, so move it off the GPU for just this step
        # and bring it back before the denoising loop needs it.
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        pipe.transformer.to("cpu")
        torch.cuda.empty_cache()

        # 1. Encode the parent's actual audio back to latent space --
        # the real content being edited, not the abstract initial-noise
        # latent stored in metadata (that's x_T, this needs real x_0-ish
        # content). Padding/channel handling mirrors prepare_latents'
        # handling of initial_audio_waveforms in pipeline_stable_audio.py.
        audio_np, _sr = sf.read(parent.stimulus.source)
        if audio_np.ndim == 1:
            audio_np = audio_np[:, None]
        audio_tensor = torch.from_numpy(audio_np.T.copy()).to(device=device, dtype=pipe.vae.dtype)
        audio_tensor = audio_tensor.unsqueeze(0)  # (1, channels, samples)

        audio_channels = pipe.vae.config.audio_channels
        if audio_tensor.shape[1] == 1 and audio_channels == 2:
            audio_tensor = audio_tensor.repeat(1, 2, 1)
        elif audio_tensor.shape[1] == 2 and audio_channels == 1:
            audio_tensor = audio_tensor.mean(1, keepdim=True)

        # sample_size still readable from transformer.config even while
        # its weights are on CPU -- config is just plain Python data.
        audio_vae_length = int(pipe.transformer.config.sample_size) * pipe.vae.hop_length
        padded = audio_tensor.new_zeros((1, audio_channels, audio_vae_length))
        length = min(audio_tensor.shape[-1], audio_vae_length)
        padded[:, :, :length] = audio_tensor[:, :, :length]

        seed_generator = torch.Generator(device).manual_seed(0)
        encoded_latents = pipe.vae.encode(padded).latent_dist.sample(seed_generator)
        del audio_tensor, padded  # done with these -- free before the transformer moves back to GPU
        torch.cuda.empty_cache()
        pipe.transformer.to(device)  # back for the denoising loop below, which does need it

        # 2. Conditioning -- reuses the pipeline's own encode_prompt/
        # encode_duration rather than reimplementing them, specifically
        # to reduce the chance of a subtle mismatch bug in a method
        # that can't be tested until tomorrow. batch_size=1,
        # num_waveforms_per_prompt=1 always here, so this skips
        # __call__'s repeat/view calls for multi-waveform batches --
        # those are no-ops at batch size 1 anyway.
        prompt_embeds = pipe.encode_prompt(self.prompt, device, do_cfg, negative_prompt=None)
        seconds_start_hidden, seconds_end_hidden = pipe.encode_duration(
            0.0, self.duration_s, device, False, 1  # False: matches __call__ when no negative prompt is used
        )
        text_audio_duration_embeds = torch.cat(
            [prompt_embeds, seconds_start_hidden, seconds_end_hidden], dim=1
        )
        audio_duration_embeds = torch.cat([seconds_start_hidden, seconds_end_hidden], dim=2)
        if do_cfg:
            negative_text_audio_duration_embeds = torch.zeros_like(text_audio_duration_embeds)
            text_audio_duration_embeds = torch.cat(
                [negative_text_audio_duration_embeds, text_audio_duration_embeds], dim=0
            )
            audio_duration_embeds = torch.cat([audio_duration_embeds, audio_duration_embeds], dim=0)

        # 3. Truncated schedule: only denoise the last
        # redo_fraction*num_inference_steps steps, starting from the
        # encoded content plus noise matched to that starting point --
        # the actual img2img mechanism (add_noise + set_begin_index is
        # diffusers' own documented pattern for this, confirmed by
        # reading scheduling_edm_dpmsolver_multistep.py directly, not
        # assumed).
        pipe.scheduler.set_timesteps(self.num_inference_steps, device=device)
        full_timesteps = pipe.scheduler.timesteps
        init_timestep = min(round(self.num_inference_steps * redo_fraction), self.num_inference_steps)
        t_start = max(self.num_inference_steps - init_timestep, 0)
        timesteps = full_timesteps[t_start:]
        pipe.scheduler.set_begin_index(t_start)

        noise = torch.randn(
            encoded_latents.shape, generator=seed_generator, device=device, dtype=encoded_latents.dtype
        )
        latents = pipe.scheduler.add_noise(encoded_latents, noise, timesteps[:1])

        # 4. Denoising loop over the truncated schedule only -- same
        # per-step logic as StableAudioPipeline.__call__, minus the
        # progress bar/callback machinery this doesn't need.
        rotary_embedding = get_1d_rotary_pos_embed(
            pipe.rotary_embed_dim,
            latents.shape[2] + audio_duration_embeds.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )
        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = pipe.transformer(
                latent_model_input,
                t.unsqueeze(0),
                encoder_hidden_states=text_audio_duration_embeds,
                global_hidden_states=audio_duration_embeds,
                rotary_embedding=rotary_embedding,
                return_dict=False,
            )[0]
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # 5. Decode and trim to the exact requested duration, matching
        # __call__'s own post-processing (waveform_start/end).
        audio = pipe.vae.decode(latents).sample
        waveform_end = int(self.duration_s * pipe.vae.config.sampling_rate)
        audio = audio[:, :, :waveform_end]

        identifier = self._identifier_for(latents)
        audio_out = audio[0].float().cpu().numpy().T
        path = self.output_dir / f"{identifier}.wav"
        sf.write(str(path), audio_out, SAMPLE_RATE)

        return Stimulus(
            identifier=identifier,
            modality="audio",
            source=str(path),
            metadata={"latent": latents.detach().to("cpu")},
        )
