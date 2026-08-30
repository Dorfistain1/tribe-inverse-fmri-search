"""
Decodes 3 real audio clips from the sparse-readout Watcher run
(compare_sparse_readout.py) so the result can actually be heard, not
just read as a number.

Unlike the earlier "match one point" fake target, this one has no
single "target latent" -- it's defined by 3 directions to push, not
one destination, so many different latents could count as "perfect."
"target" here is a constructed reference: the one latent built purely
from those 3 directions (fully satisfying all of them at once), not
something the search was ever trying to literally reach.

Re-runs the watcher_decay condition once (it's the interesting one --
blind_fixed only reached 0.36) to get the actual winning Candidate
object, since compare_sparse_readout.py's own run only kept a fitness
curve, not the latents themselves.

Run:
    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\decode_sparse_readout_result.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import torch

from tribe_core.env import configure_hf_cache

MODEL_ROOT = "G:/AI_Models"
configure_hf_cache(MODEL_ROOT)  # before any other project import -- see FINDINGS.md's HF cache gotcha

from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.fake_runtime import FakeTribeRuntime
from inverse_search.generators.audio import AudioGenerator, LATENT_CHANNELS, LATENT_FRAMES_PER_SECOND
from inverse_search.generators.fake_audio import FakeLatentGenerator
from inverse_search.generators.watcher_fake import GradientSurrogateFakeLatentGenerator
from inverse_search.sparse_readout import make_sparse_readout_target, sparse_readout_fitness

DURATION_S = 5.0
N_GENERATIONS = 200
POPULATION = 6
N_RELEVANT = 3
MUTATION_STRENGTH = 0.005
STEP_SCALE = 475
STEP_SCALE_DECAY = 0.7
MIN_STEP_SCALE = 1.0
STALL_PATIENCE = 5
PROMPT = "acoustic guitar song with a clear, memorable melody"  # AudioGenerator's fixed-good default

OUT_DIR = Path("experiments/psyche_search/data/sparse_readout_reference")
N_FEATURES = LATENT_CHANNELS * round(DURATION_S * LATENT_FRAMES_PER_SECOND)


def print_progress(message: str) -> None:
    print(message, flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "search_manifest").mkdir(parents=True, exist_ok=True)
    target = make_sparse_readout_target(N_FEATURES, n_relevant=N_RELEVANT)

    # The actual first (gen0) candidate any fresh generator produces --
    # deterministic, so no need to have saved it during the original run.
    start_latent = FakeLatentGenerator(duration_s=DURATION_S)._random_latent(seed=0)

    # Constructed reference: a latent built purely from the 3 target
    # directions, normalized to a realistic overall magnitude (real
    # latents have norm ~sqrt(n_features) since each entry is ~unit
    # variance) -- not something the search ever aimed at directly,
    # just "what fully satisfying all 3 readouts looks like."
    combined = np.zeros(N_FEATURES)
    for direction, sign in zip(target.directions, target.signs):
        combined += sign * direction
    combined = combined / np.linalg.norm(combined) * np.sqrt(N_FEATURES)
    target_latent = torch.from_numpy(combined).reshape(1, LATENT_CHANNELS, -1).to(torch.float16)

    print("Re-running the Watcher search to get the actual winning candidate...", flush=True)
    generator = GradientSurrogateFakeLatentGenerator(
        duration_s=DURATION_S, mutation_strength=MUTATION_STRENGTH,
        step_scale=STEP_SCALE, step_scale_decay=STEP_SCALE_DECAY,
        min_step_scale=MIN_STEP_SCALE, stall_patience=STALL_PATIENCE,
        output_dir=str(OUT_DIR / "search_manifest"),
    )
    runtime = FakeTribeRuntime()
    config = SearchConfig(population_size=POPULATION, n_generations=N_GENERATIONS)
    search = EvolutionarySearch(
        runtime=runtime, generator=generator, target=target,
        fitness_fn=sparse_readout_fitness, config=config,
    )
    final_population = search.run(progress=print_progress)
    best_candidate = max(final_population, key=lambda c: c.fitness)
    best_latent = best_candidate.stimulus.metadata["latent"]

    print(flush=True)
    print(f"Best found: fitness={best_candidate.fitness:+.4f} "
          f"({best_candidate.fitness * 100:.2f}% of ceiling)", flush=True)
    print("Decoding 3 reference clips with the real AudioGenerator "
          "(target / start / best)...", flush=True)

    real_gen = AudioGenerator(
        model_root=MODEL_ROOT, prompt=PROMPT, duration_s=DURATION_S, output_dir=str(OUT_DIR)
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

    print(flush=True)
    print(f"  target (constructed, fully satisfies all 3 readouts): {target_stim.source}")
    print(f"  start  (actual first candidate, seed=0): {start_stim.source}")
    print(f"  best   (fitness={best_candidate.fitness:+.4f}): {best_stim.source}")


if __name__ == "__main__":
    main()
