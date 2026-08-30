"""
Fake, near-instant version of run_evolution_cli.py.

Tests EvolutionarySearch's mechanism (selection + mutation) against a
known, real target latent instead of the real (~80min) TRIBE pipeline.
See inverse_search/DESIGN.md's "testing pyramid" section for how this
fits in: this proves the search algorithm converges reliably and lets
you tune mutation_strength/elite_fraction in seconds, but says nothing
about whether TRIBE's real fitness landscape behaves the same way --
that still needs real runs (run_evolution_cli.py).

The hidden target is a REAL, valid latent (generated the same way any
candidate is, not an arbitrary invented vector) -- so "closer to the
target" has an actual sound to check against, not just a smaller
number. See FINDINGS.md for why that distinction matters.

Run:
    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\run_fake_search_cli.py

The only GPU work this script does is at the very end: decoding 3 real
audio clips with the actual AudioGenerator (target / start / best), so
you can judge by ear whether "closer in latent space" corresponds to
anything audible.
"""

import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tribe_core.env import configure_hf_cache

# Must run before ANY other project import, not just before
# AudioGenerator.load(). EvolutionarySearch.evaluate() imports
# tribev2.eventstransforms for every audio candidate (see search.py's
# _skip_audio_transcription), and that import alone pulls in
# huggingface_hub -- which bakes in the *default* cache location the
# first time it's imported, permanently for the rest of the process
# (see tribe_core/env.py's docstring). Real runs are protected because
# TribeRuntime.__init__ calls this early; this script deliberately
# never constructs a real TribeRuntime (that's the whole point of
# "fake"), so nothing else will call it in time. Discovered the hard
# way: without this, the fake search's 30 near-instant evaluations run
# fine, but the final real decode step then silently re-downloads the
# whole model into the wrong (default) cache instead of using the
# already-complete one at MODEL_ROOT.
MODEL_ROOT = "G:/AI_Models"
configure_hf_cache(MODEL_ROOT)

from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.fake_runtime import FakeTribeRuntime, latent_distance_fitness
from inverse_search.generators.audio import AudioGenerator
from inverse_search.generators.fake_audio import FakeLatentGenerator

OUTPUT_DIR = Path("experiments/psyche_search/data/fake_search")
# Explicit, not just AudioGenerator's own default -- its old default
# ("ambient music") turned out to deterministically produce near-
# silent output on this model (see audio.py's docstring and
# FINDINGS.md). The default's fixed now too, but being explicit here
# means this script can't go silent again just because that default
# changes for some other reason later.
PROMPT = "acoustic guitar song with a clear, memorable melody"
DURATION_S = 5.0
MUTATION_STRENGTH = 0.5  # matched to AudioGenerator's default, for a fair comparison
TARGET_SEED = 999_999  # arbitrary, just needs to be outside the run's own seed range


def print_progress(message: str) -> None:
    print(message, flush=True)


def report_parent_child_correlation(manifest_path: Path) -> None:
    """Same analysis as run against the real evolution_run's manifest.csv
    (see FINDINGS.md) -- lets you compare this fake landscape's
    smoothness against the real one's measured -0.115 correlation."""
    rows = list(csv.DictReader(manifest_path.open()))
    by_id = {r["identifier"]: float(r["fitness"]) for r in rows}
    pairs = [
        (by_id[r["parent_id"]], float(r["fitness"]))
        for r in rows if r["parent_id"] and r["parent_id"] in by_id
    ]
    if len(pairs) < 2:
        return
    parents = [p for p, c in pairs]
    children = [c for p, c in pairs]
    n = len(pairs)
    mean_p, mean_c = sum(parents) / n, sum(children) / n
    cov = sum((p - mean_p) * (c - mean_c) for p, c in pairs) / n
    sd_p, sd_c = statistics.pstdev(parents), statistics.pstdev(children)
    corr = cov / (sd_p * sd_c) if sd_p and sd_c else float("nan")
    deltas = [c - p for p, c in pairs]
    print(f"  parent-child correlation: {corr:.3f} ({n} pairs)")
    print(f"  mean delta (child-parent): {sum(deltas) / n:+.4f}, stdev: {statistics.pstdev(deltas):.4f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fake_gen = FakeLatentGenerator(
        duration_s=DURATION_S,
        mutation_strength=MUTATION_STRENGTH,
        output_dir=str(OUTPUT_DIR),
    )
    target_latent = FakeLatentGenerator(duration_s=DURATION_S)._random_latent(seed=TARGET_SEED)

    runtime = FakeTribeRuntime()
    config = SearchConfig()  # same population/generations as the real run

    search = EvolutionarySearch(
        runtime=runtime,
        generator=fake_gen,
        target=target_latent,
        fitness_fn=latent_distance_fitness,
        config=config,
    )

    print("=" * 60)
    print("Fake search: mechanism test against a known real target")
    print(f"population={config.population_size} generations={config.n_generations} "
          f"mutation_strength={MUTATION_STRENGTH}")
    print("=" * 60)
    print()

    start_time = time.monotonic()
    final_population = search.run(progress=print_progress)
    elapsed = time.monotonic() - start_time

    print()
    print("=== final population, best fitness first ===")
    for c in final_population:
        print(f"  gen={c.generation} fitness={c.fitness:+.4f} id={c.stimulus.identifier}")
    print()
    print(f"Elapsed: {elapsed:.2f}s")

    manifest_path = OUTPUT_DIR / "manifest.csv"
    if manifest_path.exists():
        print()
        print("=== fake-landscape smoothness (compare to real run's -0.115) ===")
        report_parent_child_correlation(manifest_path)

    best_latent = final_population[0].stimulus.metadata["latent"]
    start_latent = FakeLatentGenerator(duration_s=DURATION_S)._random_latent(seed=0)

    print()
    print("Decoding 3 reference clips with the real AudioGenerator "
          "(target / start / best)...")
    real_gen = AudioGenerator(
        model_root=MODEL_ROOT, prompt=PROMPT, duration_s=DURATION_S, output_dir=str(OUTPUT_DIR)
    )
    real_gen.load()
    target_stim = real_gen.decode_latent(
        target_latent.to(real_gen.device), identifier="reference_target"
    )
    start_stim = real_gen.decode_latent(
        start_latent.to(real_gen.device), identifier="reference_start"
    )
    best_stim = real_gen.decode_latent(
        best_latent.to(real_gen.device), identifier="reference_best"
    )
    real_gen.unload()

    print(f"  target (what the search was aiming for): {target_stim.source}")
    print(f"  start  (the actual first candidate, seed=0): {start_stim.source}")
    print(f"  best   (fitness={final_population[0].fitness:+.4f}): {best_stim.source}")
    print()
    print("Listen to start vs. best vs. target -- does 'best' actually")
    print("sound closer to 'target' than 'start' does?")


if __name__ == "__main__":
    main()
