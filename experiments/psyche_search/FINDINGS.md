# Findings

Running log of actual results from this experiment, in the order they
happened. Each entry: what was run, what came out, what it does and
doesn't show. Negative/inconclusive results get logged here too, not
just wins -- the point of publishing this is the real record, not a
highlight reel.

---

## 2026-08-30: evolutionary search vs. random baseline (first comparison)

**Setup:** `compare_baseline.py`. Both methods given the same budget:
population 4, 3 generations (12 evaluations each). Prompt "ambient
drone music, sustained atmospheric pads". `mutation_strength=0.5`
(AudioGenerator default), `elite_fraction=0.2` -> only 1 elite kept
per generation for the evolutionary run. Fitness: `network_delta_score`
against the placeholder literature-encoded psychedelic target (see
README.md's "Psychedelic States as Neural Targets" -- not real
PsiConnect data yet).

**Result:**

| | gen 0 | gen 1 | gen 2 | best |
|---|---|---|---|---|
| Evolutionary | +0.0515 | (unchanged) | +0.0521 | **+0.0521** |
| Random | +0.0515 | +0.0521 | +0.1474 | **+0.1474** |

Random baseline won. Evolutionary search stalled near its starting
value across all 3 generations; random happened to draw a much better
candidate on its last generation.

**What this does NOT show:** that evolutionary search is useless for
this problem. 12 evaluations per method is far too small a sample to
conclude anything about which approach is generally better -- a single
lucky/unlucky draw dominates the result at this scale.

**What this probably does show:** a real problem with *this specific
configuration*, not with the approach itself:

- `elite_fraction=0.2` on a population of 4 rounds to 1 elite. Every
  child in a generation is a mutation of the *same single parent* --
  no diversity from a second lineage, so if that one elite sits in a
  flat/bad region of the fitness landscape, every child that
  generation inherits the same dead end.
- `mutation_strength=0.5` was tuned by ear for audible-but-related
  variation (see `AudioGenerator`'s docstring), not for search
  effectiveness. If the resulting latent perturbation is large enough
  that a child's fitness is nearly uncorrelated with its parent's,
  mutation-from-elite stops being meaningfully different from a fresh
  random draw -- which would exactly produce a result like this one.
- Both curves share the identical gen-0/gen-1 value (+0.0515 ->
  +0.0521) purely by coincidence of shared early seeds in this run,
  which shrinks the effective sample further.

**Next steps that would actually answer the question:** run at the
real budget (population 6, 5 generations -- `run_evolution_cli.py`'s
current defaults, ~30 evaluations, ~80min) and/or repeat this same
comparison a few times to see if evolutionary reliably catches up past
gen 2, or systematically tune `elite_fraction` (try 2 elites instead
of 1) and `mutation_strength` (try 0.2-0.3) before concluding either
way.

---

## 2026-08-30: first real-budget evolutionary run (no baseline yet)

**Setup:** `run_evolution_cli.py`, default `SearchConfig` (population 6,
5 generations, elite_fraction 0.2 -> 1 elite/generation), same prompt
and target as above. 30 evaluations total.

**Result -- best-fitness-so-far by generation:**

| gen | best this gen | best-so-far |
|---|---|---|
| 0 | +0.0515 | +0.0515 |
| 1 | +0.0642 | +0.0642 |
| 2 | +0.1474 | +0.1474 |
| 3 | **+0.1580** | +0.1580 |
| 4 | +0.0801 | +0.1580 (no improvement) |

Full lineage in `data/evolution_run/manifest.csv`.

Unlike the tiny 12-eval comparison above, this run shows a real,
monotonic climb across generations 0-3 -- each generation's elite
parent produced at least one child that beat it, which is evolution
actually doing something, not a lucky single draw. Generation 4 then
stalled: all 5 children of the generation-3 elite (cand_0016, +0.1580)
scored *worse* than their parent, spanning -0.0856 to +0.0801 -- a wide
spread from one parent in one mutation step.

That gen-4 spread reinforces the `mutation_strength=0.5` suspicion from
the earlier entry: if children this far from a good parent are common,
mutation isn't "refining" a local peak so much as frequently jumping
past it, which would show up exactly as this kind of stall right after
a strong generation.

**Not yet a win for evolutionary over random** -- this run has no
matched-budget random baseline (the earlier baseline entry above used
a different, much smaller budget: 12 evals vs. this run's 30). Next
step: run `run_random_baseline()` at this same population=6/
generations=5 budget before drawing any real conclusion about which
approach is better here.

---

## 2026-08-30: matched-budget baseline run INVALID -- cache-collision bug

**What happened:** ran `run_baseline_cli.py` (population 6, 5
generations) right after the real evolutionary run above, expecting a
fair comparison. Instead its printed fitness values for `cand_0000`
through `cand_0025` reproduced the evolutionary run's values almost
exactly (e.g. `cand_0016` = +0.1580 in both, `cand_0003` = +0.0515 in
both). Only `cand_0026`-`cand_0029` (identifiers past the evolutionary
run's highest) had genuinely new values.

**Root cause:** `AudioGenerator` identifiers were a plain counter
(`cand_0000`, `cand_0001`, ...) that restarts at 0 in every fresh
process. `TribeRuntime`'s prediction cache is disk-persisted and keyed
only on `stimulus.identifier` + modality + model identity
(`tribe_core/cache.py`) -- not on actual audio content. So the
baseline script's fresh generator, also starting at `cand_0000`,
collided identifier-for-identifier with the *previous* (evolutionary)
run's already-cached predictions and silently got those back instead
of evaluating its own, different, freshly-generated audio. This is the
same class of bug the checkpoint `resume_from()` hook fixed for
pause/resume *within* one script run -- but that fix did nothing for
two separate script invocations, which is what happened here.

**This means:** the baseline run's results above are void, not just
"inconclusive" -- most of it wasn't actually a baseline evaluation at
all. The apples-to-apples comparison this run was meant to produce
still doesn't exist yet.

**Fix:** `AudioGenerator` identifiers are now derived from a hash of
the actual latent content (`_identifier_for`), not a counter. Same
content -> same identifier -> correct cache hit; different content is
now guaranteed a different identifier regardless of process or run
history, so this specific collision can't recur. Fixed in the same
commit as this entry; `run_baseline_cli.py` needs to be re-run against
the fixed code to get a real matched-budget comparison.

---

## 2026-08-30: real matched-budget comparison (post cache-bug fix)

**Setup:** `run_baseline_cli.py` re-run after the identifier fix above,
same population 6 / 5 generations / prompt / target as the real
evolutionary run. All identifiers this time are fresh content hashes
with no overlap with the evolutionary run's -- confirmed no
cache-collision reuse (unlike the void run earlier today). Full
lineage in `data/baseline_run/manifest.csv`.

**Result -- best-fitness-so-far, evolutionary vs. random:**

| gen | evolutionary | random |
|---|---|---|
| 0 | +0.0515 | +0.0786 |
| 1 | +0.0642 | +0.1474 |
| 2 | +0.1474 | +0.1474 |
| 3 | +0.1580 | +0.1474 |
| 4 | +0.1580 | +0.1474 |

Evolutionary: **+0.1580** final. Random: **+0.1474** final. Evolutionary
wins by a modest margin (~7% relative) -- not dramatic, but for the
first time in this project a like-for-like, bug-free comparison
favors selection/mutation over pure random draws.

Wall time: random baseline run measured 78.6 min (4717s) for its 30
evaluations -- roughly double the ~42 min the evolutionary run took
for the same count (per file timestamps; that run predates the
elapsed-time printout, so it's a proxy, not a direct measurement).
Both scripts reload generator+TRIBE every generation identically (see
`_evaluate_batch`'s docstring), and both used the same duration_s/
num_inference_steps, so this gap is unexplained -- not attributed to
evolutionary vs. random here, just flagged as worth checking (system
load during the run, disk cache state, etc.) before reading anything
into per-candidate timing differences between the two approaches.

More notable than the final numbers: the *shape* of the two curves.
Evolutionary climbed steadily for 4 generations before stalling.
Random peaked early (gen 1) and never improved again -- generations
2-4 were flat-to-negative (4 of its last 12 candidates scored below
zero, vs. 1 of evolutionary's last 11). Evolutionary appears to hold a
fitness "floor" near its elite's quality across generations; random
has no such floor and keeps re-rolling the full range every generation.

**Still not strong evidence either way** -- one run each, small
population, and the earlier gen-4 stall in the evolutionary run (see
above) suggests `mutation_strength=0.5` may still be too coarse to
reliably refine a good elite. A real conclusion needs repeats (several
runs of each, not one) before treating "evolutionary wins" as settled
rather than "evolutionary wins today."

---

## 2026-08-30: prompt changed to structured music; random-baseline runs deprioritized

Two methodology changes going forward, both from direct user feedback
after listening to actual output:

**Prompt.** "Ambient drone music, sustained atmospheric pads" (used in
every run above) is genuinely hard for a human to tell candidates
apart by ear regardless of fitness score -- there isn't enough
structure in ambient drone for a listener to anchor on. Switched the
experiment's default prompt to something with real melody/rhythm to
judge by ear, same reasoning as "I can tell Ariana Grande from Rihanna
far better than one drone pad from another." First draft included
vocals, but that was dropped before ever running it -- this model
isn't built for voice mimicry (expect generic, possibly mumbled
quasi-singing, not anything resembling a real singer), so it wasn't
worth the added risk versus an instrumental prompt that still has
melody to anchor on. No named artist in the prompt either way (avoids
likeness issues on a public repo; artists were only ever an example of
"things a human can tell apart by ear," not a genre request). Also
dropped "upbeat"/drums from the first draft on a second look -- that's
exactly the failure mode AudioGenerator's docstring already documents
(precise/crisp transients, tested: EDM drums, came out worse than
sustained/melodic content on this model). Settled prompt:
`"acoustic guitar song with a clear, memorable melody"` -- reuses the
one prompt already known to render well (docstring: "country acoustic
guitar song -- clearly better" in earlier ad-hoc testing) instead of
guessing at a new one.

**Random-baseline comparisons deprioritized.** The next planned work is
"the Watcher" (see inverse_search/DESIGN.md's deferred v2 section) --
a surrogate model that learns from real evaluated candidates to guide
mutation. Watching a random-number generator teaches it nothing, so
there's no more value in spending a ~70-90min run on `run_random_baseline()`
just to prove evolution beats random -- that question was already
answered directionally in the entry above (evolutionary won, modestly).
`run_baseline_cli.py` is kept working in case a comparison is ever
wanted again, just no longer part of the regular workflow. This does
mean future evolutionary-only results can't be checked against a
random floor going forward -- a deliberate tradeoff, not an oversight.

---

## 2026-08-30: full-fake mechanism test -- curse of dimensionality, and a cache bug

**Setup:** `run_fake_search_cli.py` (see `inverse_search/DESIGN.md`'s
"testing pyramid" section) -- same population/generations/mutation as
a real run, but fitness is negative L2 distance to a real, valid
hidden target latent instead of a TRIBE prediction. No GPU cost except
decoding 3 reference clips (target/start/best) at the very end.

**Result:** the search never improved past its own initial random
population. Best-so-far stayed frozen at generation 0's value through
all 4 following generations. Every single mutation (80 parent-child
pairs across 4 runs while debugging) shifted fitness by almost exactly
the same amount: mean delta -34.6, stdev under 0.5. That's the curse of
dimensionality -- the latent is ~164,000 numbers, and Gaussian noise at
`mutation_strength=0.5` moves almost exactly the same total Euclidean
distance regardless of direction at that dimensionality, so raw L2
distance can't tell "toward the target" from "away from it." This is a
property of the fake objective (raw latent distance), not proof
TRIBE's real landscape behaves the same way -- but it's a real
warning that this particular fake design isn't a trustworthy stand-in
for tuning `mutation_strength` against.

**Real bug caught in the process:** the fake run itself worked fine
(~15s for 30 evaluations), but the final decode step tried to
re-download the entire Stable Audio Open model over the network,
eating real bandwidth, instead of using the already-complete cache on
the G: drive. Root cause: `EvolutionarySearch.evaluate()` imports
`tribev2.eventstransforms` for every audio candidate, and that import
alone pulls in `huggingface_hub` -- which bakes in the *default* cache
location the first time it happens, permanently, for the rest of the
process (same class of bug as the original HF_HOME gotcha in
`tribe_core/env.py`). Real runs are protected because `TribeRuntime`
calls `configure_hf_cache()` at construction, before anything else
runs. This fake-runtime script never constructs a real `TribeRuntime`
-- that's the point of "fake" -- so nothing called it early enough.
Fixed by calling `configure_hf_cache()` explicitly as the very first
thing in the script, before any other project import. Caught before
wasting more than a few minutes of bandwidth, but worth remembering
for any future fake-runtime script: this doesn't error, it just
silently uses the wrong cache.

**Listening check:** with the bug fixed, `data/fake_search/` has 3 real
clips (`reference_target.wav`, `reference_start.wav`,
`reference_best.wav`). Given the finding above, "best" is really just
the better of 6 random draws from generation 0, not something the
search actually earned through mutation -- worth keeping that in mind
when judging whether it sounds any closer to the target than "start"
does.

---

## 2026-08-30: AudioGenerator's default prompt produced silent audio

**What happened:** the 3 fake-search reference clips above came out
completely silent (peak amplitude ~6e-5, i.e. digital silence) --
caught by the user, not by any check on my end. Confirmed with
`soundfile`: not a playback/player issue, the files themselves had no
signal.

**Root cause:** `AudioGenerator.__init__`'s own default,
`prompt="ambient music"`, deterministically produces near-silent
output on this model. Verified directly: decoded the exact same latent
twice with that prompt (peak 6e-5 both times, reproducible), then
decoded it a third time with a different prompt and got peak 0.28 --
normal levels. Every other script in this project (run_evolution_cli,
run_baseline_cli, compare_baseline) always passes an explicit prompt,
so nothing had hit this bare default before; `run_fake_search_cli.py`
forgot to override it for its reference-clip decode, exposing it for
the first time. Root cause of *why* this specific prompt is
near-silent on this model is unconfirmed -- plausibly the model's
training data tags some genuinely very quiet/droning tracks as
"ambient," so it may not be a bug in the model so much as a bad prompt
choice on our part.

**Fix:** changed `AudioGenerator`'s default prompt to
`"acoustic guitar song with a clear, memorable melody"` (already the
project's standard working prompt everywhere else), and made
`run_fake_search_cli.py` pass it explicitly too so it can't silently
regress if the class default ever changes again for an unrelated
reason. Re-decoded the 3 reference clips -- now peak ~0.28 each, real
audio.

**Process note:** this took far longer to track down than it should
have, partly because of repeated bad time estimates on background GPU
work (see also the earlier "your machine does not support symlinks"/
network-stall investigation the same day) -- worth being more
conservative about promising specific completion times for anything
involving model loading or decoding on this hardware.

---

*(next entries go here)*
