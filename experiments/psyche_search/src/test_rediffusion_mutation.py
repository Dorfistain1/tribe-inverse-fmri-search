"""
Quick, cheap sanity check for AudioGenerator.mutate_by_rediffusion()
(new, UNTESTED as of when this script was written) -- built end of a
long session so tomorrow can start with "run this and listen" instead
of "write test infrastructure first."

Generates ONE real gen0 candidate, then mutates it via re-diffusion at
a few different redo_fraction values (how much of the denoising
schedule gets redone). Listen to all of them against the original:

- If this works as intended, small redo_fraction (0.1) should sound
  like a subtle variation of the original, and larger values (0.6)
  should drift further -- a *controllable* dial on edit size, unlike
  mutate()'s raw-noise perturbation which never had one (see
  FINDINGS.md and mutate_by_rediffusion's docstring for why).
- If something's broken in the port (wrong conditioning shapes, wrong
  scheduler state, etc.), likely symptoms: an exception, silent/
  garbage audio (check peak amplitude the same way the silent-prompt
  bug was caught earlier -- FINDINGS.md), or every redo_fraction
  sounding identical (would mean the "how much changed" dial isn't
  actually doing anything).

Run:
    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\test_rediffusion_mutation.py
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

PROMPT = "acoustic guitar song with a clear, memorable melody"
DURATION_S = 5.0
REDO_FRACTIONS = [0.1, 0.3, 0.6, 1.0]
OUT_DIR = Path("experiments/psyche_search/data/rediffusion_test")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gen = AudioGenerator(model_root=MODEL_ROOT, prompt=PROMPT, duration_s=DURATION_S, output_dir=str(OUT_DIR))
    gen.load()

    print("Generating original candidate...", flush=True)
    original_stimulus = gen.initial_population(1)[0]
    original_candidate = Candidate(stimulus=original_stimulus, generation=0)
    audio, sr = sf.read(original_stimulus.source)
    print(f"  original: {original_stimulus.source} (peak={np.abs(audio).max():.4f})", flush=True)

    for redo_fraction in REDO_FRACTIONS:
        print(f"Mutating via re-diffusion, redo_fraction={redo_fraction}...", flush=True)
        try:
            mutated = gen.mutate_by_rediffusion(original_candidate, redo_fraction=redo_fraction)
        except Exception as e:
            print(f"  FAILED at redo_fraction={redo_fraction}: {type(e).__name__}: {e}", flush=True)
            raise
        audio, sr = sf.read(mutated.source)
        print(f"  redo_fraction={redo_fraction}: {mutated.source} (peak={np.abs(audio).max():.4f})", flush=True)

    gen.unload()
    print(flush=True)
    print("Done. Listen to original vs. each redo_fraction -- does edit size scale with", flush=True)
    print("redo_fraction the way it should (small = subtle, large = bigger change)?", flush=True)


if __name__ == "__main__":
    main()
