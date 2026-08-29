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

## Generator: Stable Audio Open

Chosen over MusicGen (needs 12GB VRAM at fp16, doesn't fit this
machine's 8GB card) and a fully custom parametric synth (less
interesting, per direct user preference). Continuous latent diffusion
model -- the latent is literally a tensor of numbers that can be
perturbed directly, unlike an autoregressive token model where
"mutation" would mean editing a text prompt instead.

License: Stability AI Community License -- free for non-commercial use
and for orgs under $1M/year revenue. Fine for a free/open-source
research project; would need a real license for anything commercial.

### VRAM strategy: sequential load/unload, not concurrent

Stable Audio Open (~5.9GB) and TRIBE (~3-6GB depending on modality)
together don't fit in 8GB. They never need to be resident at once:

```
load generator -> generate candidate batch -> unload generator
load TRIBE     -> predict + score batch     -> unload TRIBE
                 (repeat next generation)
```

Not yet implemented in code -- `TribeRuntime` currently loads lazily
and keeps the model resident once loaded (fine when it's the only heavy
model in the process). Once the generator exists, `TribeRuntime` will
need an explicit `unload()`/reload path, or the generator and TRIBE
should run as two separate short-lived processes.

## Mutation: direct latent perturbation (v1)

Take a candidate's latent tensor, add small random noise, decode. This
is the "dumb" evolutionary approach: random mutation + selection, no
model of which directions in latent space actually matter.

Chosen deliberately as the *first* version despite being aware of two
more sophisticated options, given time > complexity as the stated
priority, and because the research notes (experiments/psyche_search/
restructure.md, sections 14-15) call for a simple-GA / random-search
baseline before anything fancier anyway -- this isn't wasted work, it's
the thing v2 needs to be compared against to prove it's actually better.

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

## Target: intentionally out of scope here

See `experiments/psyche_search/src/run_search.py` and
`restructure.md`'s "target source" discussion for the psychedelic
target's own literature-vs-real-data roadmap. `inverse_search` just
consumes whatever `NeuralTarget` it's handed.
