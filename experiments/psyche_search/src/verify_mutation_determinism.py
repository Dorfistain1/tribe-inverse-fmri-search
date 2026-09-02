"""
Checks last night's brainstorm hypothesis (FINDINGS.md, 2026-09-01):
`mutate_by_rediffusion()` seeds its RNG with `manual_seed(0)` fresh on
every call, so a fixed parent + fixed redo_fraction SHOULD be fully
deterministic -- and yet real sibling candidates sharing a parent in
an already-completed run had different fitness. Two live hypotheses:

1. GPU floating-point non-determinism (attention/matmul reduction
   order -- nothing in this codebase pins deterministic algorithms)
   compounding over the denoising loop into a real output difference,
   despite every named seed being fixed.
2. Something else not yet identified.

Directly settles which: call mutate_by_rediffusion() twice, back to
back, same parent + same redo_fraction, and diff the raw output
arrays. Bit-exact -> hypothesis wrong, look elsewhere. Different ->
confirmed, and the fix is straightforward (stop hardcoding seed 0,
seed randomly per call instead -- turns accidental jitter into a real,
understood, appropriately-sized exploration operator).

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\verify_mutation_determinism.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tribe_core.env import configure_hf_cache

MODEL_ROOT = "G:/AI_Models"
configure_hf_cache(MODEL_ROOT)  # before any other project import -- see FINDINGS.md's HF cache gotcha

import numpy as np
import soundfile as sf

from inverse_search.candidate import Candidate
from inverse_search.generators.audio import AudioGenerator

DURATION_S = 5.0
REDO_FRACTION = 0.3
OUT_DIR = Path("experiments/psyche_search/data/mutation_determinism_check")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gen = AudioGenerator(model_root=MODEL_ROOT, duration_s=DURATION_S, output_dir=str(OUT_DIR))
    gen.load()

    print("Generating parent candidate...", flush=True)
    parent_stimulus = gen.initial_population(1)[0]
    parent_candidate = Candidate(stimulus=parent_stimulus, generation=0)
    parent_renamed = Path(parent_stimulus.source).with_name("parent.wav")
    parent_renamed.unlink(missing_ok=True)
    Path(parent_stimulus.source).rename(parent_renamed)
    parent_stimulus.source = str(parent_renamed)
    print(f"  parent: {parent_renamed}", flush=True)

    outputs = []
    for i in range(2):
        print(f"Mutating (call {i + 1}/2, redo_fraction={REDO_FRACTION})...", flush=True)
        mutated = gen.mutate_by_rediffusion(parent_candidate, redo_fraction=REDO_FRACTION)
        renamed = Path(mutated.source).with_name(f"mutation_call_{i + 1}.wav")
        renamed.unlink(missing_ok=True)
        Path(mutated.source).rename(renamed)
        audio, sr = sf.read(renamed)
        outputs.append(audio)
        print(f"  saved: {renamed} (peak={np.abs(audio).max():.4f})", flush=True)

    gen.unload()

    a, b = outputs
    bit_exact = np.array_equal(a, b)
    max_abs_diff = float(np.abs(a - b).max()) if a.shape == b.shape else None
    rms_diff = float(np.sqrt(np.mean((a - b) ** 2))) if a.shape == b.shape else None

    print(flush=True)
    print("=" * 60, flush=True)
    if bit_exact:
        print("BIT-EXACT: the two calls produced identical audio.", flush=True)
        print("Hypothesis WRONG -- something else is the real source of", flush=True)
        print("sibling diversity in real runs, not GPU non-determinism.", flush=True)
    else:
        print("NOT bit-exact -- the two calls produced different audio", flush=True)
        print("despite identical seeds, identical parent, identical redo_fraction.", flush=True)
        print(f"  max abs sample difference: {max_abs_diff:.6f}", flush=True)
        print(f"  RMS difference: {rms_diff:.6f}", flush=True)
        print("Hypothesis CONFIRMED -- diversity is coming from GPU float", flush=True)
        print("non-determinism, not a designed exploration operator.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
