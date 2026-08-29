# inverse_search Design Notes

This documents *how the search engine works and why*, and what's
deliberately deferred. Lives here (not in experiments/psyche_search/)
because everything below is about the generic search machinery --
reusable by any future target, not specific to psychedelics or music.

## Pipeline

```
Candidate (Stimulus)
      |
StimulusGenerator.mutate()      <- generators/audio.py, generators/text.py
      |
TribeRuntime.predict()          <- tribe_core, unchanged
      |
BrainPrediction
      |
fitness_fn(prediction, target)  <- fitness.py, e.g. network_delta_score
      |
Candidate.fitness
      |
selection (search.py's loop)
      |
      +--> next generation's parents
```

`EvolutionarySearch` (search.py) only ever calls `runtime.predict()`,
`generator.mutate()`, and `fitness_fn(...)` -- it doesn't know what a
psychedelic target is, what Stable Audio Open is, or how mutation
actually works. See mainStructure.md "Experiment Contract" for why that
boundary is kept strict.

## Generator: Stable Audio Open (built -- see generators/audio.py)

Chosen over MusicGen (needs 12GB VRAM at fp16, doesn't fit this
machine's 8GB card) and a fully custom parametric synth (less
interesting, per direct user preference). Continuous latent diffusion
model -- the latent is literally a tensor of numbers that can be
perturbed directly, unlike an autoregressive token model where
"mutation" would mean editing a text prompt instead.

License: Stability AI Community License -- free for non-commercial use
and for orgs under $1M/year revenue. Fine for a free/open-source
research project; would need a real license for anything commercial.
Gated on HuggingFace (auto-approved -- accept the license on the model
page once, then `hf auth login`).

Latent shape (1, 64, duration_s * 512) was measured empirically (loaded
the pipeline, generated a clip, inspected the actual tensor via a
diffusion-step callback) rather than read from docs -- see
generators/audio.py's module docstring if this needs re-deriving after
a model/library update.

### VRAM strategy: sequential load/unload, not concurrent

Stable Audio Open (~5.9GB measured peak) and TRIBE (~3-6GB depending on
modality) together don't fit in 8GB. They never need to be resident at
once:

```
load generator -> generate candidate batch -> unload generator
load TRIBE     -> predict + score batch     -> unload TRIBE
                 (repeat next generation)
```

`AudioGenerator.load()`/`.unload()` implement this on the generator
side (confirmed VRAM actually drops to ~0 after unload). `TribeRuntime`
itself still only loads lazily and stays resident once loaded -- fine
today since nothing calls generator and runtime in the same process
yet, but `search.py`'s `run()` will need to explicitly interleave
`generator.unload()` before the first `runtime.predict()` call once
the full loop is wired up.

### A real gotcha hit building this: HF_HOME redirection breaks gated-model auth

Both TRIBE and Stable Audio Open needed HF_HOME redirected off C: (see
mainStructure.md's shared-storage principle). Doing that naively broke
auth for *both* Llama and Stable Audio Open, silently, no error until
the download itself 401s. Root cause: some `huggingface_hub` versions
only look for the token at `$HF_HOME/token`, not the real default
`~/.cache/huggingface/token`, once HF_HOME points elsewhere -- and
fixing that by calling `huggingface_hub.get_token()` before redirecting
is *itself* broken, because merely importing huggingface_hub bakes
HF_HOME into an internal constant at that exact moment. Fixed in
`tribe_core/env.py`'s `configure_hf_cache()`: read the token file
directly (plain file I/O, no huggingface_hub import) before touching
HF_HOME at all. Both `TribeRuntime` and `AudioGenerator` call this same
helper, since they don't share a process lifecycle.

## Mutation: direct latent perturbation (v1, built)

Take a candidate's latent tensor, add small random noise, decode. This
is the "dumb" evolutionary approach: random mutation + selection, no
model of which directions in latent space actually matter.

Chosen deliberately as the *first* version despite being aware of two
more sophisticated options, given time > complexity as the stated
priority, and because the research notes (experiments/psyche_search/
restructure.md, sections 14-15) call for a simple-GA / random-search
baseline before anything fancier anyway -- this isn't wasted work, it's
the thing v2 needs to be compared against to prove it's actually better.

`mutation_strength` default (0.5) was tuned by ear on one prompt/
duration, not derived -- see generators/audio.py's docstring for the
comparison points (0.15/0.3 too subtle, 0.5 clearly related-but-
different, 0.8+ starting to feel unrelated).

## Deferred: surrogate-model-guided search (v2)

Instead of randomly perturbing the latent, build a model of "these
latent settings -> this fitness score" from every candidate evaluated
so far, and use that model to pick the next candidate to try --
Bayesian optimization, in the same spirit as "The Automatic
Neuroscientist" (restructure.md section 18, a real closed-loop fMRI
optimization paper that did exactly this instead of random search).

Why deferred, not abandoned: real added complexity (a Gaussian process
or small neural surrogate, an acquisition function, etc.) versus "add
noise to a tensor." Given each evaluation costs real GPU time
(generate audio + run TRIBE), this is very likely worth building once
the dumb version proves the pipeline works end to end -- fewer expensive
evaluations needed to find good candidates is exactly the point of
Bayesian optimization, and it directly pairs with dimensionality
reduction of the latent (PCA or similar) if the raw latent turns out
too large/sparse for either approach to navigate well (restructure.md
sections 7-9's warning about huge unconstrained search spaces).

Trigger to revisit: once v1 (random mutation) runs end to end and
produces a fitness curve, per restructure.md section 15 -- compare
against a pure-random baseline first, *then* decide if the surrogate
model is worth building.

## Deferred: extending the winning candidate past the duration ceiling

`AudioGenerator`'s duration_s is capped around 5-8s by a real hardware
wall (see its docstring -- the VAE decoder's cost blows up well before
Stable Audio Open's documented 47s max on this 8GB card). That's a
property of the generation model + this GPU, independent of search
strategy -- a smarter search (the surrogate model above) still only
ever searches in the cheap short-clip regime, and should: there's no
reason to pay for long-duration generation on every candidate just to
evaluate it.

The plan for getting a longer *final* piece once search converges on a
good candidate: `StableAudioPipeline` accepts `initial_audio_waveforms`
(a clip to condition on, i.e. continue from) -- generate the winning
clip, feed its tail back in as the seed for the next chunk, repeat.
Each individual call stays in the cheap regime; only the assembly of
several such calls produces something longer. This is a one-time cost
applied to the winning candidate at the end, not something the search
loop itself needs to do.

Not built yet -- noted here so it isn't lost. When building it: the
surrogate/search's job is entirely "find a good seed to extend," it
never needs to know long clips exist.

## Target: intentionally out of scope here

See `experiments/psyche_search/src/run_search.py` and
`restructure.md`'s "target source" discussion for the psychedelic
target's own literature-vs-real-data roadmap. `inverse_search` just
consumes whatever `NeuralTarget` it's handed.
