# Contributing / where to start

This is a solo side project, not a maintained library or a lab with a
roadmap. If you found this repo and want to poke at it, extend it, or
just understand what's actually been done -- this page is the map.

## What this repo actually is

A shared local runtime for [TRIBE v2](https://github.com/facebookresearch/tribev2)
(Meta's brain-response prediction model), plus a generic evolutionary
search engine (`inverse_search/`) built on top of it, plus one concrete
experiment using that engine (`experiments/psyche_search/`): searching
for audio whose *predicted* brain response resembles psychedelic-state
connectivity patterns.

Read that last sentence carefully -- this is not a claim that any
audio here produces a psychedelic *experience*. It's optimizing
against a brain-response *model's prediction*, which is a much weaker
and more honest claim. See
[experiments/psyche_search/README.md](experiments/psyche_search/README.md)'s
"Neural Similarity vs. Subjective Experience" section for why that
distinction matters.

No peer review, no multi-subject studies, no lab -- one person running
real experiments on a personal GPU and writing down what actually
happened, including the wrong turns. That's the whole methodology, and
it's a deliberate choice, not a limitation to apologize for.

## Reading order

1. **[README.md](README.md)** -- setup, requirements, what each top-level folder is.
2. **[inverse_search/DESIGN.md](inverse_search/DESIGN.md)** -- how the generic search engine works (candidate generation, mutation, fitness, selection), what's built vs. still deferred, and *why* each design choice was made -- most of them exist because something simpler was tried first and broke.
3. **[experiments/psyche_search/README.md](experiments/psyche_search/README.md)** -- the concept and hypothesis behind the psychedelic-audio experiment specifically. Framed before any results existed -- it's the theory, not the status report.
4. **[experiments/psyche_search/FINDINGS.md](experiments/psyche_search/FINDINGS.md)** -- **the actual state of things.** A dated, chronological log of every real experiment run, including the ones that failed, the ones that turned out to be measuring the wrong thing, and every bug found along the way. This is the single most important document if you want to know what's actually been established versus what's still just a hypothesis. Long, and deliberately not cleaned up into a highlight reel.
5. **[experiments/psyche_search/restructure.md](experiments/psyche_search/restructure.md)** -- research notes and open decisions specific to this experiment.

## Current status (short version -- see FINDINGS.md for the real thing)

- The pipeline works end to end: generate audio candidates (Stable
  Audio Open), mutate them (partial re-diffusion, not raw noise --
  see DESIGN.md for why the first approach didn't work), score them
  against a real TRIBE prediction, select, repeat.
- Several real, non-obvious bugs were found and fixed along the way
  (a memory bug from a missing `torch.no_grad()`, an amplitude/clipping
  drift in repeated mutation, a hidden GPU non-determinism bug that
  made "controlled" experiments not actually controlled, a scaling bug
  that silently zeroed out a fitness metric). All logged in FINDINGS.md
  with how they were found, not just the fix.
- With those bugs fixed, real runs are now trustworthy: fitness climbs
  for the first few generations, then plateaus. That plateau replicates
  across different starting populations and survived several attempts
  to escape it (adaptive decay of the mutation step size, heterogeneous
  mutation sizes within a generation) -- neither clearly helped. Current
  honest read: this specific mutation mechanism has a real ceiling for
  this setup, not that the whole approach is broken.
- At least one real result where a higher fitness score corresponded to
  an actual, described difference on listening ("very ominous") --
  not just a number moving.

## What's genuinely open

- A structurally different mutation approach (editing the decoded
  waveform directly -- timing, envelope, pitch -- instead of always
  re-invoking the diffusion model) hasn't been tried. Current best
  guess for what would meaningfully move past the plateau.
- No real listening/qualitative study beyond ad hoc reactions during
  development -- "does this actually sound different in a describable
  way as fitness changes" is still mostly untested at scale.
- The "convergent evolution across independent runs" question the
  concept doc raises (do independent searches find similar structure)
  hasn't been investigated at all yet.
- The text modality (`Stimulus(modality="text", ...)`) and any
  non-audio stimulus generator are unbuilt.

## If you want to run something

Every experiment script that costs real GPU time says so in its own
docstring, with a time estimate. If you're touching `inverse_search/`
or `experiments/psyche_search/src/`, skim FINDINGS.md first -- there's
a real chance whatever you're about to try has already been tried and
the result (including *why* it didn't work) is already written down.

## License

Original code and docs here are CC BY-NC-SA 4.0 (see [LICENSE](LICENSE)).
Dependencies carry their own separate terms -- see [NOTICE.md](NOTICE.md),
which also covers the required Llama 3.2 attribution for the text
modality.
