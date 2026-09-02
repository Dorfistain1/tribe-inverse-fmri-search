"""
Cheap real-TRIBE prompt screening -- no evolution, no mutation. For
each candidate prompt, generate N different-seed gen0-style candidates
and score every one with real TRIBE fitness. Point: ground the prompt
choice in the actual optimization target, not just coherence/listening
(which is all prompt choice has been based on so far -- see
probe_psychedelic_prompts.py).

Deliberately reports best-of-N and spread per prompt, not just mean --
evolution with elitism doesn't care about a prompt's average fitness,
it cares whether there's a good outlier to latch onto and refine. A
prompt with a modest mean but wide spread could be a better
evolutionary starting point than one with a higher but rigid mean.

Same caveat as everything else this project has found: a higher score
doesn't guarantee it's higher for a reason that maps to anything
perceptually meaningful. Real number, worth having, not a substitute
for a sanity listen.

Reuses EvolutionarySearch._evaluate_batch directly (not .run()) --
that's the exact tested code path for generator-unload -> TRIBE-load ->
evaluate-with-transcription-patch -> manifest logging -> TRIBE-unload,
just fed a flat batch of un-mutated candidates instead of a generation
loop. One generator load + one TRIBE load total (not one pair per
generation like a real search run), so this should be considerably
cheaper per-candidate than the evolutionary runs' measured pace.

Time estimate (rough, real uncertainty flagged): audio model load ~3min
+ 18 candidates (3 prompts x 6 seeds) x ~26s generation = ~8min
generation phase, then one TRIBE load + 18 evaluations. TRIBE's own
load time and per-candidate eval time aren't cleanly separable from any
existing measurement (every real run mixed them with repeated per-
generation reload overhead this script doesn't pay) -- rough guess
~20-35min total, not a confident number.

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\prompt_screening_cli.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tribe_core.env import configure_deterministic_mode, configure_hf_cache

MODEL_ROOT = "G:/AI_Models"
configure_hf_cache(MODEL_ROOT)  # before any other project import -- see FINDINGS.md's HF cache gotcha
configure_deterministic_mode()  # same rule -- before any other project import

from experiments.psyche_search.src.run_search import build_psychedelic_target
from inverse_search import EvolutionarySearch, SearchConfig
from inverse_search.candidate import Candidate
from inverse_search.generators.audio import AudioGenerator
from tribe_core import TribeRuntime

OUTPUT_DIR = Path("experiments/psyche_search/data/prompt_screening")
DURATION_S = 5.0
GUIDANCE_SCALE = 7.0
N_SEEDS = 6

PROMPTS = {
    "psy_ambient": "psychedelic ambient soundscape with swirling, shifting layers and hypnotic texture",
    "psy_electronic": "trippy electronic music with warped, morphing textures and a hypnotic pulse",
    "psy_rock": "hypnotic psychedelic rock with driving rhythm and swirling guitar effects",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generator = AudioGenerator(
        model_root=MODEL_ROOT, duration_s=DURATION_S, output_dir=str(OUTPUT_DIR), guidance_scale=GUIDANCE_SCALE
    )
    target = build_psychedelic_target()
    search = EvolutionarySearch(
        runtime=TribeRuntime(), generator=generator, target=target, config=SearchConfig()
    )

    start_time = time.monotonic()

    print("Generating candidates...", flush=True)
    generator.load()
    by_prompt: dict[str, list[Candidate]] = {label: [] for label in PROMPTS}
    for label, prompt_text in PROMPTS.items():
        generator.prompt = prompt_text
        for seed in range(N_SEEDS):
            latent = generator._random_latent(seed=seed)
            stimulus = generator.decode_latent(latent, identifier=f"{label}_seed{seed}")
            candidate = Candidate(stimulus=stimulus, generation=0)
            by_prompt[label].append(candidate)
            print(f"  {label} seed={seed}: {stimulus.source}", flush=True)

    all_candidates = [c for candidates in by_prompt.values() for c in candidates]
    print(flush=True)
    print(f"Evaluating {len(all_candidates)} candidates against real TRIBE...", flush=True)

    def progress(message: str) -> None:
        print(f"  {message}", flush=True)

    search._evaluate_batch(all_candidates, progress)
    elapsed = time.monotonic() - start_time

    print(flush=True)
    print("=" * 70, flush=True)
    print(f"{'prompt':<16} {'n':>3} {'mean':>8} {'best':>8} {'worst':>8} {'stdev':>8}", flush=True)
    for label, candidates in by_prompt.items():
        fitnesses = [c.fitness for c in candidates]
        mean = statistics.mean(fitnesses)
        best = max(fitnesses)
        worst = min(fitnesses)
        stdev = statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0
        print(f"{label:<16} {len(fitnesses):>3} {mean:>+8.4f} {best:>+8.4f} {worst:>+8.4f} {stdev:>8.4f}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print(f"Elapsed: {elapsed / 60:.1f} min ({elapsed:.0f}s)", flush=True)
    print("Best mean = safest average starting point. Best 'best' + high stdev =", flush=True)
    print("most room for evolution to find and exploit a lucky outlier.", flush=True)


if __name__ == "__main__":
    main()
