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

## 2026-08-30: mutation_strength sweep on the fake tier

**Setup:** `sweep_mutation_strength.py` -- same fake search as before,
run once per `mutation_strength` in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02,
0.01, 0.005], reporting best-so-far improvement and parent-child
fitness correlation for each. Pure CPU, no audio/GPU, whole sweep runs
in seconds.

**Result:**

| mutation_strength | improvement over gen0 | parent-child correlation |
|---|---|---|
| 0.5 -- 0.02 | ~0.00 | ~0 or undefined |
| 0.005 | +0.01 | 0.614 |

Correlation only becomes real somewhere below 0.02 -- roughly 100x
smaller than the current default. But the improvement even at 0.005 is
tiny (0.01 against a -572 baseline), and this fake objective (raw
distance to one point in a ~164,000-dim space) is a much harder,
more artificial landscape than TRIBE's real fitness (a coarser
connectivity summary, plausibly much smoother). So this sweep confirms
0.5 is clearly too large, but its exact numbers don't transfer to
picking a real mutation_strength value directly.

**Follow-up -- two false starts on a full 80min real run, then a
smaller step instead:** first launched at `mutation_strength=0.1`,
killed almost immediately on re-reading the sweep table above -- 0.1
sat in the same degenerate cluster as the default 0.5, not actually a
different test. Relaunched at `mutation_strength=0.02`, then killed
that too on request before it produced anything: a single ~80min,
30-eval real run per candidate value is a lot to commit to based on
fake-tier numbers alone, however well-reasoned, without seeing any
real data point first.

Built `real_mutation_sweep_cli.py` instead -- small-budget (3
population x 2 generations = 5 evals, not 30) real runs (real audio +
real TRIBE) at `mutation_strength` in [0.5, 0.02, 0.005], meant to be
run and watched directly rather than summarized after the fact. Not
statistically solid on its own (5 evals is too few), but gives an
actual real-TRIBE data point at the small-mutation regime before
anyone commits real GPU time to a full-budget run. `run_evolution_cli.py`
reverted to its plain defaults (no mutation_strength override) in the
meantime -- the full run is on hold pending what this smaller sweep
shows.

## 2026-08-30: extended fake sweep, then meta-search finds the same answer twice

Redirected effort back to the free fake tier instead of spending real
GPU time -- two further steps, both CPU-only:

**Extended grid sweep** (`sweep_mutation_strength.py`, updated): 20
generations instead of 5, values zoomed in around 0.005 instead of
spanning 0.5-0.005. Correlation climbs steadily below ~0.01 and stays
strong (0.72-0.99) across the whole 0.001-0.01 band -- a real trend
across many values, not a single lucky point. Best single-run
improvement: 0.004 (+0.07). 0.015 scored *worse* than both its
neighbors (0.02 and 0.01) -- a reminder that any one value's number
here is noisy (one run, no repeats).

**Meta-search** (`meta_search_mutation_strength.py`, new): rather than
manually picking sweep points, ran `EvolutionarySearch` itself over
`mutation_strength` as the thing being evolved -- each "candidate" is
a value, evaluated by averaging 3 independent 6x20 fake searches
(reduces the single-run noise the grid sweep showed), mutated
multiplicatively in log-space (values span 3 orders of magnitude, so
additive noise doesn't work), keeping 3 elites per generation ("multiple
bests"). Converged from a wide random start (0.0001-0.022) to a tight
cluster at 0.003-0.007 within 6 generations; best found: 0.00597
(avg improvement +0.0685 over 3 repeats). ~3 minutes total, no GPU.

Two independent methods (grid sweep, meta-search) landing on the same
~0.004-0.006 region is a real cross-validation, not a coincidence of
one lucky sweep point -- the strongest evidence yet for where the fake
landscape's usable region is. Still the same standing caveat as every
entry above: this narrows down a real-run candidate, it doesn't
replace needing one.

**How the ~0.005 number was actually derived** (for anyone re-deriving
or doubting it later): not read off a single sweep row. Two
independent methods agreed:
1. Grid sweep, 10 values from 0.02 down to 0.001, 20 generations each,
   one run per value -- correlation only becomes non-degenerate below
   ~0.01 and stays strong (0.72-0.99) across 0.001-0.01, with 0.004
   giving the best single-run improvement (+0.07).
2. Meta-search (`meta_search_mutation_strength.py`), which evolves the
   value itself, averaging 3 repeats/candidate specifically to avoid
   trusting single noisy runs -- started from a random spread across
   0.0001-0.022 and collapsed to a stable 0.002-0.008 band within one
   generation, staying there for the rest of a 6-generation run. Best
   found: 0.00597.

**Two follow-up ideas, both tested on the fake tier before touching real
GPU time again:** (1) an adaptive mutation_strength schedule -- start
big, shrink on stall (evolution strategies' "1/5 success rule"),
implemented as a generic `on_generation_result(improved)` hook on
`StimulusGenerator` (base.py), wired into `search.py`'s generation
loop; (2) a first "Watcher" prototype
(`generators/watcher_fake.py`'s `SurrogateFakeLatentGenerator`) --
PCA + ridge regression fit on every (latent, fitness) pair seen so
far (via a new `record_result()` hook), used to pick the best-
predicted of several candidate mutations before spending a real
evaluation on any of them.

`compare_watcher_fake.py`: 30 generations, same hidden target, three
conditions --

| condition | final improvement |
|---|---|
| blind, fixed mutation_strength=0.005 | **+0.0841** (best) |
| Watcher, fixed mutation_strength=0.005 | +0.0445 |
| blind, adaptive from 0.5 (decay 0.7/stall) | +0.0357 (worst) |

**Adaptive schedule:** works exactly as designed, but the cost is
visible and real -- fitness stays completely frozen for the first 11
generations (0.5 is degenerate on the fake landscape, as already
known), only starting to climb once decay has shrunk it to ~0.007
(0.5 x 0.7^12), right where the sweep already said the useful region
starts. In a real ~5-generation budget this exact schedule would
likely show zero improvement the whole run -- it wouldn't have time to
decay into the useful range. Caveat: the *real* run at fixed 0.5
already showed genuine improvement (unlike fake), so this "wasted
generations" cost may be smaller or absent on TRIBE's real landscape --
untested.

**Watcher v1 underperformed plain blind mutation** -- a real, if
disappointing, negative result. Likely cause: PCA + ridge fit on very
few samples (even after reduction) isn't predicting well enough to
beat picking randomly, and may be actively steering toward worse
candidates from overfit noise rather than neutral. Not evidence the
Watcher concept is dead -- DESIGN.md always described this crude
version as a first iteration, not the intended final design (a real
acquisition-function/Bayesian-optimization approach) -- but it is
evidence this specific cheap version isn't earning its added
complexity yet. Don't invest further here without a better surrogate
or more data per fit; plain blind mutation at a known-good
mutation_strength remains the best fake-tier result so far.

## 2026-08-30: Watcher v2 (gradient-directed) -- a real breakthrough on the fake tier

After v1 underperformed, rebuilt as `GradientSurrogateFakeLatentGenerator`
(`generators/watcher_fake.py`): instead of picking the best of several
random guesses, use the fitted linear surrogate's own gradient
(`components.T @ ridge.coef_`) as an actual direction to step in,
scaled by a `step_scale` parameter, plus a smaller amount of ordinary
random noise for continued exploration/data diversity.

Also fixed a real performance bug found while testing this: the
surrogate was being refit from scratch on the entire history for
*every* mutation call, not once per generation -- refitting is cached
by history length now (only refits when new data actually arrived),
and history is capped at 200 samples so PCA cost stays bounded across
hundreds of generations instead of growing unboundedly.

**step_scale calibration** (`tune_watcher_v2.py`, 30 generations each,
same target, single run per value -- narrowed in three passes):

| step_scale | % of gap closed |
|---|---|
| 0.5 - 3 | ~0% (too small to matter -- a unit gradient step is tiny across 164,000 dims) |
| 10 - 300 | 1.15% -> 8.17% (climbing) |
| 400 - 500 | 17-19%, peaking at **475: 19.49%** |
| 550+ | falling |
| 1000+ | back to ~0% -- catastrophic overshoot |

The collapse above ~750 makes sense given the objective is a smooth
bowl (L2 distance to a point): the gradient direction is only valid
*locally* -- step far enough along it and you sail straight past the
target's basin into territory the linear surrogate never saw, landing
back at effectively-random. There's a real, sharp optimum, not a
"more is better" relationship.

For comparison, every blind-mutation result measured earlier in this
file peaks around 0.01-0.02% of the same gap. 19.49% is roughly
1000-2000x better than anything blind mutation (with or without
adaptive decay) achieved. This is the first result in the whole
project that looks like more than noise relative to the actual goal
(sounding meaningfully similar to a target), not just "beats a
randomized baseline by a small margin."

Caveats, still standing: (1) single run per calibration point, some
noise likely, though the smooth rise-then-fall shape across the final
7-point sweep is a real curve, not scatter; (2) this is still the fake
landscape (raw L2 distance in latent space), not TRIBE's real fitness
-- a real run is the next real test, once the fake tier's long-run
behavior (see next entry) is understood; (3) step_scale=475 is
specific to this mutation_strength (0.005) and this target/duration --
not verified to transfer if those change.

## 2026-08-30: the long Watcher run plateaus hard -- a fixed step size can't refine

Launched `long_watcher_run.py` (1000 generations, step_scale=475 fixed)
to see the real shape of the climb, not just a 30-generation endpoint.
Caught by the user watching the status file directly: it "got closest
in the first few gens." Verified independently from the raw
manifest.csv (recomputing running-best via cummax, not trusting the
displayed number blindly): true best hit **19.5% of the gap at
generation 4**, then sat completely flat for **347 straight
generations** with zero improvement before the run was killed.

Root cause: a fixed step_scale big enough to jump to a good region
fast (the calibration's whole point) is *also* too big to refine once
there -- every subsequent mutation is still that same huge step,
overshooting past whatever local improvement exists regardless of how
good the estimated direction is. The calibration's 19.49% was
therefore never a point on a rising curve -- it was the ceiling of one
lucky early jump, not evidence of continued climbing.

**Fix, not yet validated:** `step_scale_decay` added to
`GradientSurrogateFakeLatentGenerator` (same `on_generation_result`
pattern as `mutation_decay` elsewhere) -- shrinks step_scale by 30%
whenever a generation fails to improve, so the search can switch from
"leap toward a good area" to "refine locally" automatically instead of
a fixed value forcing one behavior for the whole run.

Also improved `long_watcher_run.py`'s status reporting on this same
occasion (the user's direct request after spotting the plateau): now
tracks `batch_best` (this generation's best *new* candidate) alongside
`running_best` and a stall counter, not just running_best alone --
batch_best staying far below running_best for many generations in a
row is exactly the plateau signal that was invisible before.

**Patience fix, then the real ceiling.** First patience attempt
(decay after 1 stall, i.e. none) collapsed step_scale to its floor
within ~18 generations and scored *worse* (13.76% by gen 333) than not
decaying at all -- fixed by only decaying after 5 *consecutive*
stalled generations (`stall_patience`, added to all three decay
implementations: `GradientSurrogateFakeLatentGenerator`'s step_scale,
`FakeLatentGenerator`/`AudioGenerator`'s mutation_strength).

With that fix, the full 1000-generation run: **23.36% of the gap
closed at generation 999** -- better than the un-decayed plateau
(19.5%), but it settled into this new plateau by roughly generation
300 (23.32% there) and the remaining ~700 generations bought only
+0.04 percentage points. Never crossed the 25% milestone. 26 minutes
total, all CPU.

**Real conclusion, not just a tuning note:** this specific
approach -- a linear (ridge regression) surrogate providing a gradient
direction, stepped with a decaying scale -- has a genuine ceiling
around 20-25% of this gap, not a slow-but-still-climbing trajectory.
More generations of the same method don't help past that point. Likely
cause: the surrogate is linear, but the true objective (L2 distance to
a point) curves -- a straight-line model can point roughly toward the
target from far away, but can't represent *when to slow down* as you
approach it, so refinement stalls once linear approximation stops
being locally accurate. Getting further would need either a surrogate
that captures curvature (not just a gradient), or a fundamentally
different step strategy (e.g. plain small-step gradient descent redone
every generation, rather than one big leap followed by decay). Neither
is built yet. 20-25% is real, dramatic progress over blind mutation's
~0.01-0.02% -- but still far short of anything resembling "sounds
similar to the target."

## 2026-08-30: a better-designed fake target changes the picture completely

Every fake test above used the same target shape: minimize raw L2
distance to one specific, arbitrary point in the full ~164,000-dim
latent space. On reflection (prompted directly by the user questioning
whether the experiment was a dead end), that target doesn't actually
match the real task's structure. `network_delta_score` (fitness.py)
doesn't reward matching a specific point at all -- it rewards pushing
a FEW bounded summary values (network correlations, each naturally in
[-1, 1]) as far as possible in a direction. Many different audios
could score well under that objective; it isn't a single needle in an
enormous haystack.

Built `inverse_search/sparse_readout.py`: a target using the *real*
objective's shape instead -- a small number (3) of fixed random
"readout" directions in latent space, each passed through tanh
(bounded like a real correlation) and rewarded via the same
sign-times-value structure network_delta_score actually uses. All
other ~163,837 dimensions are pure noise to this fitness function --
sparse, low-effective-dimensional structure, unlike the fully-dense
distance-to-a-point target where every one of the 164,000 dimensions
mattered equally.

`compare_sparse_readout.py`, 200 generations, same target both
conditions:

| generation | blind_fixed | watcher_decay |
|---|---|---|
| 0 | 0.23 | 0.23 |
| 90 | 0.30 | 0.34 (barely ahead) |
| 100 | 0.30 | 0.39 |
| 120 | 0.32 | 0.92 |
| 190 | 0.36 | 1.00 (99.67%) |
| final | **+0.36** | **+0.9978** |

Blind mutation climbs slowly and roughly linearly the whole time.
Watcher tracks it closely (barely ahead) through generation ~90, then
**explodes** -- 0.39 -> 0.66 -> 0.92 -> 0.98 in the next 30
generations, finishing essentially perfect (99.78% of the theoretical
ceiling). The flat stretch before ~gen 90 is most likely the
surrogate accumulating enough history to correctly identify which 3 of
163,840 directions actually matter; once it does, it converges almost
immediately. This is a completely different result from every
distance-to-a-point test above (~20-25% ceiling, plateauing hard) --
strong evidence that the *earlier target's shape*, not the search
mechanism itself, was the real bottleneck.

**What this changes:** if TRIBE's real fitness landscape has a
similarly sparse structure (plausible -- it only cares about a handful
of specific network-pair correlations, very possibly driven by a
relatively low-dimensional subset of real audio features, not
literally all 164,000 latent numbers independently), the Watcher could
perform far better on the real task than the earlier pessimistic
ceiling suggested. This makes testing the Watcher against real TRIBE
(even briefly) considerably more worth doing than it looked a few
entries ago.

**Standing caveats:** still the fake tier -- this doesn't prove TRIBE's
real landscape is actually this sparse, only that *if* it is, the
Watcher can exploit it dramatically well. One run per condition, no
repeats yet. The specific generation where the breakthrough happens
(~100 here) is plausibly random (depends on when the right history
samples happen to accumulate) -- not yet verified as reproducible
across different random seeds/targets.

## 2026-08-31: does higher fitness even correspond to an audible difference? Checked on real data.

Prompted by a direct user concern after the sparse-readout listening
test: a 99% fitness score meant almost nothing audible changed
(FINDINGS.md's sparse-readout entries) -- is fitness ever going to
track something a human would actually notice, or could the whole
project converge on changes too subtle to matter to anyone listening?

Built `inverse_search/acoustic_features.py` (spectral centroid =
brightness, RMS = loudness, short-window dynamic range, zero-crossing
rate = roughness/noisiness -- standard, dependency-free DSP metrics)
and `audio_feature_report.py`, then ran it immediately on the *real*
`evolution_run` data (26 real, TRIBE-scored candidates, zero
additional compute -- the audio already existed).

**Result: essentially no correlation between fitness and any of the
four features** (spectral centroid +0.095, RMS -0.066, dynamic range
+0.027, zero-crossing rate +0.015 -- all near zero). Values are
tightly clustered across the whole fitness range (dynamic range sits
at ~104.5-105.1 dB whether fitness is -0.09 or +0.16).

This is real evidence, not a hypothetical: on the one real run we
have, improving TRIBE-based fitness does not correspond to any of
these obvious, human-relevant acoustic properties changing. Two
readings, not yet distinguished: (1) TRIBE's real driver is something
more subtle/complex than these 4 basic descriptors capture (plausible
-- brain response could depend on spectral/temporal patterns finer
than "brighter" or "louder"), or (2) the actual acoustic differences
really are close to imperceptible in any human-relevant sense. Either
way, this directly validates the user's concern about the sparse-
readout test generalizing to the real experiment: chasing a fitness
number is not the same as producing something a listener can tell
apart, and this is the first real (not synthetic) data point showing
that gap might be real here too.

**Proposed next test (not yet run, costs real GPU time):** build a
partial-fake target that explicitly rewards a large, obviously
audible acoustic change (e.g. maximize spectral centroid/brightness),
computed from real decoded audio rather than TRIBE or a synthetic
latent-space proxy. If the Watcher can push that to a perceptible
extreme, it proves guided search *can* find audible structure when the
target actually has it -- directly testing whether the search itself
is capable of producing perceptible change, isolated from the separate
question of whether TRIBE's real landscape rewards anything perceptible
at all.

## 2026-08-31: diagnosing WHY fitness doesn't map to audible change -- and a real fix, untested

Direct user pushback tied together everything from the last several
entries: the sparse-readout "best" sounding 99.9%-identical to "start"
despite 99% fitness, and real fitness not correlating with acoustic
features, both point at the same root cause -- not a weak search
algorithm, but *what's being mutated*. All mutation so far (`mutate()`)
perturbs the diffusion model's INITIAL noise latent (x_T) and re-runs
the full ~40-step denoising trajectory from scratch. Diffusion sampling
is a long, highly nonlinear process -- there's no reliable relationship
between "how far you moved in x_T" and "how different the final audio
sounds." Small moves can do nothing (curse of dimensionality, matches
the mutation_strength tuning entries), and once a move is big enough to
matter, it does so unpredictably (matches the step_scale overshoot
collapse entry). This is a property of mutating x_T specifically, not
evidence mutation/evolution can't work at all.

**The fix** (built, not yet tested -- it was late, verification is
next session's first task): `AudioGenerator.mutate_by_rediffusion()`,
a real, standard img2img/SDEdit-style partial re-diffusion, ported by
hand from diffusers' own pattern since `StableAudioPipeline` doesn't
expose a `strength` parameter. Confirmed by reading the actual
diffusers source (not assumed) that the pipeline's own
`initial_audio_waveforms` input does NOT do this despite looking
similar -- it adds encoded audio as a bias under full-strength noise
and still runs every step, which is really the audio-continuation
mechanism, not a controllable edit. The real mechanism: encode the
parent's actual decoded audio back to latent space, add noise
corresponding to a specific point partway through the schedule
(`scheduler.add_noise` + `set_begin_index`, the scheduler's own
documented img2img pattern), and only redo the remaining steps.
`redo_fraction` becomes a direct, controllable knob on edit size --
something raw-noise mutation never had at all.

Also built `test_rediffusion_mutation.py` -- generates one real
candidate and mutates it at several `redo_fraction` values, ready to
run and listen to first thing next session. Expected signature of
success: edit size visibly/audibly scales with `redo_fraction` (small
= subtle variation, large = bigger change) -- something no version of
raw-latent mutation ever demonstrated all session. If it's broken, the
manually-ported conditioning/denoising loop is the most likely place
(shape mismatches, wrong scheduler state) -- flagged clearly in the
method's own docstring for whoever debugs it next.

## 2026-08-31: partial re-diffusion works -- real bug was a missing torch.no_grad()

First real GPU test of `mutate_by_rediffusion()` from the previous
entry. Took several rounds of genuine debugging, not one-line fixes --
worth recording the actual path since the first two "fixes" were
wrong guesses that happened to look plausible:

1. **First failure**: `CUDA out of memory`, "0 bytes free", inside the
   VAE encode step. Assumed fragmentation -- added `gc.collect()` +
   `torch.cuda.empty_cache()`. Retested: identical failure, same
   ~22GB "allocated" figure. Wrong guess.
2. **Second attempt**: assumed genuine peak-memory pressure instead,
   and moved `pipe.transformer` to CPU during the encode (it isn't
   needed for a VAE-only forward pass) to free real headroom. Retested:
   identical failure again. Also wrong, or at least not sufficient --
   confusingly, since a *standalone* test of the same offload call
   (outside the buggy method) proved it really does free ~6GB.
3. **Real diagnosis**: stopped guessing, instrumented
   `test_rediffusion_mutation.py` with direct
   `torch.cuda.memory_allocated()`/`memory_reserved()`/`mem_get_info()`
   calls at each stage (real APIs, not the OOM error message's
   possibly-misleading figure). This showed the actual bug: after a
   normal ~2.8GB load+decode, calling `mutate_by_rediffusion()` pushed
   *real, confirmed* `memory_allocated()` to **22.7GB** -- physically
   impossible on an 8GB card unless something is retaining far more
   than it should. Root cause: `StableAudioPipeline.__call__` is
   decorated `@torch.no_grad()`, but the hand-written
   `mutate_by_rediffusion()` had no such guard -- every tensor op
   (VAE encode, every transformer call in the denoising loop) was
   being tracked by autograd for a backward pass that would never
   happen, retaining the full computation graph across every step.
   Added `@torch.no_grad()` to the method. Fixed on the first retest:
   memory stayed bounded (~2.9GB allocated) across all 4 mutations,
   all real, non-silent audio (peaks 0.53-0.93), exit code 0.

Lesson for next time debugging GPU memory in this codebase: don't
trust the OOM error message's "X GiB allocated by PyTorch" figure at
face value, and don't guess-patch (fragmentation, peak pressure) --
instrument with the real `torch.cuda.memory_*()` APIs at each stage
first. The two wrong guesses cost real time; the actual bug was found
in one diagnostic pass once measured directly instead of assumed.

**Checked with `visualize_audio_comparison.py`** (waveform +
spectrogram, all 5 clips side by side,
`data/rediffusion_test/comparison.png`): a clean, monotonic
progression. `redo_fraction=0.1` is nearly indistinguishable from the
original -- same rhythm, same peak timing. `0.3` keeps the overall
structure but gets denser/fuzzier. `0.6` changes substantially --
smoother envelope, less like distinct notes, closer to a sustained
pad, runs slightly longer. `1.0` is the most different by far --
visibly shorter, fewer and denser bursts, a genuinely different
rhythm shape.

This is the first mutation approach all session with any reliable
relationship between "how much you told it to change" and "how much
actually changed." Every raw-latent-noise approach (mutate_strength,
step_scale tuning, all of it) never showed this -- small nudges did
nothing, big nudges did something unpredictable, and nothing in
between behaved as a controllable dial. This one does. Directly
addresses the user's own diagnosis from the previous session ("the
issue is with the diffuser that makes it seemingly random") with a
real fix, not just a smarter search algorithm layered on the same
broken substrate.

Still open: this is one qualitative visual check on one candidate,
not yet run through real TRIBE fitness, and not yet tested at the
project's normal 5s duration (tested at 3s to sidestep the VRAM issue,
now resolved -- worth retrying at 5s next).

## 2026-08-31: first real run with rediffusion mutation -- real win, same old plateau

`run_evolution_rediffusion_cli.py`, real target, `redo_fraction=0.3`
fixed, 10 generations (51 evaluations), real 5s duration (no VRAM
issue this time -- the no_grad fix held). 57 minutes.

**Result: +0.3708 best fitness** -- more than 2.3x the old raw-noise-
mutation record (+0.1580, FINDINGS.md's first real run). Real,
apples-to-apples improvement against the same TRIBE target.

**But the trajectory has the same shape as the fake-tier gradient-
surrogate result**: climbed fast (gen0 0.079 -> gen1 0.280 -> gen3
0.371), then completely flat for the remaining 6 generations -- same
elite (`cand_2f4766604e46`) parented every candidate from gen4 through
gen9, never beaten once. A fixed edit size finds a good jump fast and
then can't refine further, again. `mutate_by_rediffusion()` has no
adaptive `redo_fraction` decay yet (the `stall_patience`-gated decay
pattern exists for `mutation_strength`/`step_scale`, never ported to
this mutation mode) -- likely next fix, same shape as the earlier
adaptive-decay work.

Net read: the fixed mechanism alone is already a real, substantial win
over raw-noise mutation on actual TRIBE fitness -- not just fake
proxies. Adding decay should very plausibly push past 0.37, based on
the exact same pattern already solved once on the fake tier.

## 2026-08-31: redo_fraction decay, first run -- correcting a wrong attribution

Ran `run_evolution_rediffusion_decay_cli.py` (`stall_patience=5`, same
target/budget as the entry above). Result: **+0.5725**, higher than
the non-decay run's +0.3708. Initially reported as evidence decay
helps -- **wrong, caught by direct user pushback**, worth recording the
correction plainly.

Checked the actual manifest: the +0.5725 jump happened at **generation
3**, mutated from generation 2's own jump, mutated from generation 1's
-- three lucky mutations in a row off the same fixed redo_fraction=0.3,
*before* decay had triggered even once (`stall_patience=5` needs 5
consecutive stalled generations first, and generation 3 was still
climbing). Decay only started acting around generation 8, on a value
already found and already flat for 4 generations by then -- and once
active, it produced zero further improvement over the remaining
stalled generations anyway.

So +0.5725 and the earlier +0.3708 are the same phenomenon: elitism
preserving a lucky mutation, at a fixed (undecayed) redo_fraction, in
both cases. Decay hasn't demonstrated any actual benefit yet -- in the
one run where it engaged, it didn't produce further improvement
either. Real lesson: a higher number in a later run isn't evidence a
change helped unless the improvement happens to occur *after* that
change was actually active. Lowered `stall_patience` to 3 for the next
attempt (6 stalled generations only triggered one decay cycle at
patience=5 -- see the script's own comment) -- but this still needs an
actual test where decay is active *before* any further improvement
happens, or a repeat/statistical comparison, before crediting it with
anything.

**patience=3 rerun: same problem, checked directly.** Result +0.427 --
lower than the patience=5 run's +0.5725. Manifest shows the +0.427
elite was found at generation 4, decay didn't engage until *after*
generation 7 (three generations later), and the only two generations
that actually ran with a decayed redo_fraction (gen 8-9) never beat
it (best 0.358). Identical shape to every run so far: the headline
number comes from a lucky mutation chain on the fixed, undecayed
starting value, before decay gets a turn. 0.427 being lower than
0.5725 is consistent with this being ordinary variance in that early
phase, not a decay effect.

**Honest state across all three real runs (0.3708 no-decay, 0.5725
patience=5, 0.427 patience=3): zero of them have actually tested
whether decay helps.** In every case decay engaged, if at all, only
after the run's peak was already found, with at most 2-6 generations
of runway afterward -- never enough to show a trend either way. Real
test would need either much longer runs (more post-engagement runway)
or repeats at each setting to separate signal from the lucky-jump
noise that's dominated every result so far.

## 2026-08-31: the 27-generation decay run answered decay -- and found a bigger problem

`run_evolution_rediffusion_decay_long_cli.py`, 27 generations, ~2.5hr.
**Result: +0.8599**, by far the highest real fitness seen. Status file:
`first_decay_generation: 9`, `best_ever_generation: 21`,
`best_found_after_decay: true` -- the first of four real decay-related
runs where the peak was actually found *after* decay engaged. That's
the real answer to the question the last three runs failed to test:
decay does not block continued improvement, and this run improved
substantially (0.37ish range up to 0.86) well past generation 9.

**But a bigger, more serious problem surfaced checking a user
observation** ("gen 26's 0.214 candidate sounds virtually identical to
gen 21's 0.860 one despite a 0.5+ fitness gap," and separately "audio
sounds very loud, parts cut off"). Checked peak amplitude and clipping
across all 27 generations of this run's actual wav files -- not noise,
a clean monotonic trend:

| generation | peak | % samples clipped |
|---|---|---|
| 0 | 0.28-0.32 | 0% |
| 2-5 | hits 1.0000 | 0-0.02% |
| 17 | 1.0000 | ~1.5% |
| 22-26 | 1.0000 | ~2%+ |

Audio amplitude climbs steadily and starts hard-clipping by generation
~2, worsening every generation after. Root cause: `mutate_by_rediffusion()`
re-encodes the PREVIOUS decoded audio through the VAE and decodes
again on every single mutation -- a repeated round-trip, and each
cycle appears to inject a bit more energy that's never corrected,
compounding like repeatedly re-saving a lossy file. This is a
structural property of the mutation mechanism, not a one-off bug.

**Why this matters beyond audio quality**: it casts real doubt on
whether the fitness climb (0.37 -> 0.86 across all rediffusion runs so
far) reflects genuine progress toward the target, or is partly/wholly
an artifact of increasingly loud, distorted audio that TRIBE's audio
branch happens to score differently. The "sounds the same despite a
huge fitness gap" observation is consistent with fitness tracking
clipping-driven spectral change rather than anything meaningfully
target-directed. This isn't yet proven (correlation between generation
number and both fitness and clipping doesn't establish which drives
which, or whether both are just responding to the same underlying
mutation accumulation) -- but it's a serious enough confound that every
rediffusion-mutation fitness number to date (0.37, 0.57, 0.43, 0.86)
should be treated as suspect until checked.

**Likely fix, not yet built**: renormalize amplitude (e.g. peak-
normalize to a fixed level like 0.9) after every `mutate_by_rediffusion()`
decode, before saving/returning the candidate -- stops the round-trip
energy drift from accumulating generation over generation. Should be
done before trusting any further rediffusion-based real run's fitness
numbers.

Per-generation range of the meta-search population (shows the
collapse happening, not just the end state):

| gen | range (min-max) | mean | best fitness |
|---|---|---|---|
| 0 | 0.00011 - 0.02212 | 0.0065 | +0.052 |
| 1 | 0.00103 - 0.00674 | 0.0047 | +0.057 |
| 2 | 0.00330 - 0.00821 | 0.0059 | +0.052 |
| 3 | 0.00169 - 0.00677 | 0.0050 | +0.066 |
| 4 | 0.00179 - 0.00823 | 0.0055 | +0.069 |
| 5 | 0.00332 - 0.01139 | 0.0058 | +0.053 |

**Important tension this raises, not yet resolved:** the *real* TRIBE
run at the old default (0.5) -- the value the fake tier calls
catastrophically degenerate -- actually produced genuine, monotonic
improvement for 4 generations (+0.0515 -> +0.1580, see the "first
real-budget evolutionary run" entry above), reaching more total
improvement in 3 generations than 20 fake-tier generations at 0.005
managed (+0.07). A low/negative aggregate parent-child correlation
doesn't rule out occasional big beneficial jumps driving real progress
under elitism -- which is exactly what large mutation_strength can do
that small mutation_strength structurally can't in a short run. So
"smaller mutation_strength is better" is proven for *local search
validity* (correlation), not yet for *actually finding good candidates
faster in a limited generation budget* -- those can point in opposite
directions, and only a real test at ~0.005 can settle which one wins
on TRIBE's actual landscape.

---

## 2026-09-01: prompt/guidance_scale probe -- dropping the prompt breaks the model, doesn't free it

`probe_prompt_guidance.py`: same fixed latent, 5 decodes varying only
prompt text and `guidance_scale` (7.0 is `StableAudioPipeline`'s own
default, confirmed by reading `pipeline_stable_audio.py` directly --
not a value we picked, matches the library). Listened to all 5:

| combo | result |
|---|---|
| guitar prompt, cfg=7.0 (current default) | normal, coherent music |
| guitar prompt, cfg=3.0 | AI-noisy garbage |
| empty prompt, cfg=7.0 / 3.0 / 1.0 | ~0.5s of inaudible speech-like sound, then silence, for all three |

**This falsifies the working theory from earlier today** that
`guidance_scale` was the main thing forcing "acoustic guitar song"
onto every candidate, and that backing it off would let some hidden,
more diverse prior show through. It doesn't -- lowering `guidance_scale`
with a real prompt just breaks audio quality, and empty prompt breaks
the same way "ambient music" did (see 2026-08-30 entry): this model
needs strong text conditioning + `guidance_scale=7.0` to produce
anything coherent. There's no diverse unconditioned prior hiding
underneath at accessible settings.

**Decision**: keep the prompt, keep `guidance_scale=7.0`, but treat the
prompt as a deliberate, disclosed style choice rather than an
incidental default -- switch from the placeholder "acoustic guitar
song" wording to an explicitly psychedelic-flavored prompt, and say so
plainly in any write-up (we choose the target *style* via prompt; the
search explores/shapes fitness within it, it doesn't invent style from
noise). `guidance_scale` was exposed as a real `AudioGenerator`
constructor param (was hardcoded in two places) so this and future
prompt work don't require code changes to test.

Two different follow-up tests, not to be conflated:
- **Cheap**: sweep candidate prompt wordings at fixed cfg=7.0, check
  which ones produce coherent, non-garbage audio and survive
  rediffusion mutation without collapsing -- no TRIBE, minutes, same
  shape as this probe.
- **Expensive**: which prompt actually shifts TRIBE's output most
  toward the real target -- needs a full real run *per* prompt tested,
  multiplies real-TRIBE time by however many prompts survive the cheap
  filter. Only worth it after that filter narrows the field.

## 2026-09-01: brainstorm -- is sibling diversity from a designed mutation operator, or GPU jitter?

Checked something during a brainstorm session (no code changes, just
looking at existing on-disk data from the already-completed
`run_evolution_rediffusion_decay_long_cli.py` 27-gen run):
`mutate_by_rediffusion()` seeds its RNG with `torch.Generator(device).manual_seed(0)`,
freshly, on *every single call* -- used for both the VAE-encode
sampling and the injected noise. For a fixed parent + fixed
`redo_fraction`, every explicit source of randomness in the function is
therefore pinned to the same value every time it's called.

And yet: checking real sibling candidates that share a parent and
generation in that run's `manifest.csv` (e.g. gen1's 5 children, all
`parent_id=cand_c0b154b4fb16`), their real TRIBE fitness values differ
(0.1674, 0.1247, 0.0401, -0.1249, 0.1836) -- confirmed via the
manifest, not assumed. So the function is *not* actually deterministic
in practice, despite every named seed being fixed.

**Hypothesis, not yet verified**: the actual source of variation
between "identical" mutation calls isn't a designed exploration
operator at all -- it's likely non-deterministic floating-point
behavior in the GPU forward pass (attention/matmul reduction order;
nothing in this codebase sets `torch.use_deterministic_algorithms` or
pins cuDNN to a deterministic mode), compounding over the ~12+ manual
denoising steps into a real, measurable audio/fitness difference. If
true, this would mean the "mutation" driving evolutionary diversity
all session has actually been uncontrolled hardware jitter, not a
tunable stochastic operator -- which would explain a lot: why
`redo_fraction` decay never reliably helped (shrinking the window
shrinks the *jitter's* opportunity to compound too, not just the
"edit size"), and why exploration has felt weak/plateau-prone despite
`redo_fraction` nominally being a real dial.

**Cheap way to check, next session**: call `mutate_by_rediffusion()`
twice back-to-back on the exact same parent + `redo_fraction` and
compare the raw output arrays for bit-exact equality. If they're
identical, the hypothesis is wrong and diversity comes from somewhere
else not yet identified. If they differ, it's confirmed, and the fix
is straightforward -- stop hardcoding `manual_seed(0)`, seed with a
real per-call random value instead, turning accidental jitter into a
deliberate, understood, appropriately-sized exploration operator.

**Checked (2026-09-02, `verify_mutation_determinism.py`): confirmed.**
Same parent, same `redo_fraction=0.3`, identical `manual_seed(0)` in
both calls -- output still diverged: max abs sample difference 0.6487
(peak amplitude of both clips was only 0.9), RMS difference 0.1137.
That's not floating-point-scale jitter, that's substantially different
audio content. GPU non-determinism in the forward pass (not a designed
exploration operator) is the confirmed real source of "sibling"
diversity in every rediffusion-mutation run this project has done so
far. `redo_fraction` was never actually controlling "how much
randomness gets injected" the way its name implies -- it controls how
many denoising steps are exposed to this uncontrolled jitter, which is
a different, much less legible thing to tune. This plausibly explains
the repeated plateau/decay-doesn't-help pattern: shrinking
`redo_fraction` shrinks the jitter's compounding window, not a
deliberate "edit size."

**Fix applied (2026-09-02)**: `mutate_by_rediffusion()` now draws a real
random seed per call (`torch.randint`) instead of hardcoding
`manual_seed(0)`, and stores it in the returned `Stimulus.metadata`
(`"mutation_seed"`) for traceability. Doesn't remove the underlying GPU
non-determinism, but turns it from a silent confound into an
acknowledged, logged source of variation, and stops implying a false
reproducibility that never actually held. Not yet re-verified with a
real run.

**Follow-up, same day: the jitter can be eliminated outright, not just
logged.** `verify_deterministic_mode.py`: set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA init, called
`torch.use_deterministic_algorithms(True, warn_only=False)` -- enabled
cleanly, no unsupported-op fallback needed. Pinned `torch.manual_seed`
before each of two back-to-back `mutate_by_rediffusion()` calls (same
parent, same `redo_fraction=0.3`, same resulting internal seed
1608637542 both times) -- **output was bit-exact** (`np.array_equal`
true) both times.

So the GPU non-determinism found earlier today is fully closeable on
this model/setup, not a structural dead end. Real path forward:
combine `torch.use_deterministic_algorithms(True)` with the new
explicit per-call seed (already logged in metadata) and
`redo_fraction` becomes what it was always supposed to be -- a single,
clean, fully-reproducible-given-the-seed exploration dial, no hidden
hardware-jitter confound riding along with it. Not yet wired into the
main generation path (`AudioGenerator`/real search scripts) -- this
was checked in isolation, in a throwaway script.

Practical caveat, not yet measured: deterministic algorithms can be
slower than the default (non-deterministic) CUDA kernels for some ops
-- this run's per-mutation wall time wasn't compared against a
non-deterministic baseline call, so the speed cost (if any) of turning
this on for real search runs is still unknown.

### Other brainstormed directions (not started, ranked by how cheap they are to test)

- **Prompt-blend genome**: make the *conditioning* itself part of what
  evolution controls -- each candidate carries a blend weight across 2-3
  anchor prompts (e.g. "psychedelic ambient" vs. "hypnotic driving
  rhythm"), mutation nudges the blend. Guarantees real, audible effect
  per mutation (conditioning directly changes), unlike latent-level
  edits that CFG keeps pulling back toward one fixed style. Real risk:
  interpolating text embeddings doesn't reliably produce
  semantically-in-between audio -- untested, would need a cheap
  listen-only check first (2-3 interpolation points, no TRIBE).
- **CFG scheduling**: high `guidance_scale` early in the denoising
  schedule (locks in target-consistent global structure), lower later
  (loosens local/texture detail so mutation has more room to move
  without breaking coherence). Standard trick in image diffusion;
  untried here.
- **Heterogeneous mutation strength across the population**: instead of
  one shared `redo_fraction` for every candidate in a generation, give
  some candidates larger and some smaller values simultaneously (mixed
  step sizes, standard in evolution strategies) -- more chances to
  escape the "final 20%" trap in a single generation, no rebuild, pure
  config/algorithm change.
- **Multi-family seeding**: seed gen0 from several different
  but-individually-coherent prompt variants instead of one, let
  selection pick which family wins on real fitness, mutate
  within-family afterward. Injects real diversity structurally at gen0
  rather than expecting rediffusion mutation to manufacture it later.
- **DSP-layer mutation** (raised earlier, still on the table): keep the
  diffuser only for gen0 (real, natural, TRIBE-in-distribution audio),
  do all mutation via direct waveform edits (silence insertion, timing,
  pitch shift, gain envelope) instead of repeated prompt-conditioned
  re-diffusion. Fully legible/controllable, and today's empty-prompt
  result strengthens the case for it: the diffuser's conditioning is
  fragile enough that *not* re-invoking it every mutation looks safer,
  not just simpler.

## 2026-09-01: prompt screening -- psy_ambient wins clearly, no switch needed

`prompt_screening_cli.py`: 3 candidate psychedelic-style prompts x 6
seeds each, un-mutated gen0-style candidates, real TRIBE fitness, no
evolution. Point: ground the prompt choice in the actual optimization
target instead of just coherence/listening (all that prior prompt
comparisons had checked).

| prompt | mean | best | worst |
|---|---|---|---|
| psy_ambient | -0.0125 | **+0.0545** | -0.0490 |
| psy_electronic | -0.1898 | -0.1305 | -0.2372 |
| psy_rock | -0.1456 | -0.0980 | -0.2019 |

Not close. `psy_ambient`'s entire distribution sits ~0.13-0.18 above
the other two -- it's the only prompt where any seed ever crossed into
positive fitness (1 of 6). This is also the prompt both real
evolutionary runs so far happened to use (picked by ear, not by this
data) -- so no prompt switch needed, and both existing real runs stay
comparable to whatever comes next. Multi-prompt-family seeding
(brainstormed 2026-09-01) is deprioritized -- the screening gap is too
lopsided to expect blending in the weaker two prompts would help.

## 2026-09-01: decay fair-test -- finally engages properly, still no clean evidence it helps

`run_evolution_decay_fair_test_cli.py`, `stall_patience=1` (was 3),
same prompt (`psy_ambient`) and budget (9 gens) as the previous
trustworthy run, isolating just this one variable. 65.05 min elapsed
(3903s vs. the previous run's 3746s -- consistent pace, good).

Decay engaged properly this time: `first_decay_generation: 3`,
`best_found_after_decay: true`. Best-ever trajectory: 0.0786 (gen0) ->
0.18 (gen1) -> 0.2505 (gen2) -> [stalls gen3-4, decay engages] -> 0.3519
(gen5) -> [stalls gen6-8, run ends]. The gen5 breakthrough genuinely
happened after 2 rounds of decay had already shrunk `redo_fraction`
(0.3 -> 0.21 -> 0.147), so on its face this looks like decay enabling a
refinement past a plateau -- the hypothesis this whole line of testing
was chasing.

**But**: `best_found_after_decay: true` being real doesn't prove decay
*caused* it, and the one comparison we actually have points the other
way. The immediately preceding trustworthy run
(`run_evolution_deterministic_hour_cli.py`, `stall_patience=3`, decay
barely engaged until gen8) reached +0.3852 at generation 5. This run
(more decay pressure, earlier) reached +0.3519 at the same generation
number -- slightly *lower*. So the one apples-to-apples comparison
available says more/earlier decay correlated with a slightly worse
peak, not a better one. Nowhere near enough sample (n=2 runs, different
random seeds throughout) to conclude decay hurts either -- just enough
to say it did NOT clearly help, which is itself the honest answer after
this many attempts to test it. A real controlled comparison (same
seeds, only `stall_patience` differing) would need paired runs, not
independent ones -- not done here.

User's listening reaction to `best_g05_fit+0.352.wav`: "very ominous."
First real qualitative description tied to a run we actually trust end
to end (deterministic mutation, validated prompt, fair decay test) --
worth remembering as a data point even though it's subjective, since
so much of this project's early "does fitness even mean anything
audible" concern is finally being checked against trustworthy
numbers instead of confounded ones.

## 2026-09-01: different starting seeds -- the plateau pattern generalizes, but timing/magnitude don't

`run_evolution_different_seeds_cli.py`, `initial_seed_offset=100`
(gen0 seeds 100-105 instead of the 0-5 every prior run shared), same
prompt/budget/settings as the original trustworthy baseline
(`stall_patience=3`). 79.9 min elapsed (4793.7s) -- notably longer than
the previous two runs (62.4min, 65.1min); resolved, not a mystery --
the PC was in active use for other things during this run, real GPU/CPU
contention, not a property of the code or settings.

This gen0 started markedly worse (mean -0.0098, max +0.0425 vs. the
shared population's +0.0353/+0.0786) and had a rough gen1 (all 5
children negative, min -0.3078). Then a real breakthrough at
**generation 2**: max +0.2890 -- earlier than either previous run's
peak (both at gen5). `best_ever_generation: 2`, final best **+0.2890**,
never beaten through gen8. `first_decay_generation: 5`,
`best_found_after_decay: false` -- decay still hasn't helped in any
fair test.

**Interpretation**: the climb-then-plateau *shape* replicates on a
genuinely different starting population -- not an artifact of the
shared gen0 seeds after all, real evidence the mechanism (or the
landscape it's searching) has a general tendency to find a quick early
win and then stop improving. But the *specifics* vary a lot by starting
draw: peak at generation 2 here vs. generation 5 in both prior runs,
final best +0.289 here vs. +0.385/+0.352 there. So any single run's
exact numbers ("peaks around gen5", "reaches ~0.35-0.39") were never a
stable fact -- only the general dynamic (fast climb, early plateau,
decay not clearly helping) is now supported by three independent data
points instead of one shared starting condition.

Three for three on the plateau, zero for three on decay helping --
worth treating this as evidence the limitation is in the mutation
mechanism itself (rediffusion off a fixed prompt), not something more
tuning of redo_fraction/decay is likely to fix. Points toward trying a
structurally different exploration idea next (heterogeneous
redo_fraction across a generation, or the DSP-layer-mutation idea) over
further decay-parameter tuning.

## 2026-09-01: heterogeneous redo_fraction -- clean negative result

`run_evolution_heterogeneous_redo_cli.py`, `redo_fraction_range=(0.1, 0.5)`
(each mutation call draws its own value, no shared/decaying redo_fraction),
same prompt and starting population (`initial_seed_offset=0`) as the
original two trustworthy baselines -- directly comparable, one variable
changed. 87.7 min elapsed (5262.8s), longer than the ~62-65min other
same-seed runs took; unconfirmed whether that's PC contention (like the
last run's measured cause) or genuinely more compute from the range's
upper bound exceeding the old fixed 0.3 default -- not checked either
way.

**Worse, not better.** Best fitness +0.2264 at generation 3, well below
both fixed/decaying-redo_fraction runs on this same starting population
(+0.3852, +0.3519). Per-generation mean fitness was negative in 7 of 9
generations (gen0's initial population aside) -- population quality
was consistently rougher throughout, not just a lower peak. Generation
mins repeatedly hit -0.26 to -0.30, notably worse than any generation
min seen in the fixed/decay runs on the same seeds.

Reasonable read: allowing redo_fraction up to 0.5 (vs. the old fixed
0.3) let some mutations make bigger, more destructive edits more often
than it let good candidates refine -- the downside outweighed the
upside here. Doesn't rule out a narrower range (e.g. closer to 0.3
either side) doing better, but this specific range clearly didn't help
and shouldn't be assumed to be a general improvement over fixed/decaying
redo_fraction.

**Four real experiments on this same starting population now** (fixed
0.3/patience=3: +0.3852; fixed 0.3/patience=1: +0.3519; heterogeneous
0.1-0.5: +0.2264) plus one on a different population (+0.2890) -- none
beat the original plain fixed-redo_fraction run. Real diminishing
returns on tuning this lever further; worth treating this as the
mechanism's rough ceiling for this prompt/setup rather than continuing
to search for a better redo_fraction schedule.

*(next entries go here)*
