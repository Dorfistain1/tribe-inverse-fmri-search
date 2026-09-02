"""
Follow-up to verify_mutation_determinism.py (confirmed: GPU float
non-determinism, not a designed operator, was the real source of
"mutation" diversity -- FINDINGS.md 2026-09-02). Question this answers:
can `torch.use_deterministic_algorithms(True)` force real bit-exact
reproducibility, or does this model hit an op with no deterministic
CUDA implementation?

If yes: redo_fraction can become a real, clean, single-purpose dial
(pair it with an explicit seed instead of the current random-per-call
one) -- worth doing. If no: the jitter is structurally unavoidable
while mutation goes through the diffuser at all, which is a real point
in favor of the DSP-layer-mutation idea from the brainstorm (mutate
the decoded waveform directly, skip re-invoking the diffuser).

CUBLAS_WORKSPACE_CONFIG must be set before CUDA initializes -- hence
before the torch import, not just before use_deterministic_algorithms().

Same probe shape as verify_mutation_determinism.py (parent + two
mutation calls, same redo_fraction, diff the output), but this time
also pinning torch's global RNG (torch.manual_seed) immediately before
each call, so mutate_by_rediffusion's own internal random per-call seed
draws the same value both times -- isolates the determinism question
from the (now-intentional) random-seed change.

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\verify_deterministic_mode.py
"""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # must precede CUDA init

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tribe_core.env import configure_hf_cache

MODEL_ROOT = "G:/AI_Models"
configure_hf_cache(MODEL_ROOT)  # before any other project import -- see FINDINGS.md's HF cache gotcha

import numpy as np
import soundfile as sf
import torch

from inverse_search.candidate import Candidate
from inverse_search.generators.audio import AudioGenerator

DURATION_S = 5.0
REDO_FRACTION = 0.3
CONTROL_SEED = 42  # pins mutate_by_rediffusion's internal random seed draw to the same value both calls
OUT_DIR = Path("experiments/psyche_search/data/deterministic_mode_check")


def main():
    print("Enabling torch.use_deterministic_algorithms(True)...", flush=True)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        strict_mode_ok = True
    except Exception as e:
        print(f"  Could not enable strict deterministic mode: {type(e).__name__}: {e}", flush=True)
        print("  Falling back to warn_only=True so the run can still complete.", flush=True)
        torch.use_deterministic_algorithms(True, warn_only=True)
        strict_mode_ok = False

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gen = AudioGenerator(model_root=MODEL_ROOT, duration_s=DURATION_S, output_dir=str(OUT_DIR))
    outputs = []
    mutation_error = None

    try:
        gen.load()

        print("Generating parent candidate...", flush=True)
        parent_stimulus = gen.initial_population(1)[0]
        parent_candidate = Candidate(stimulus=parent_stimulus, generation=0)
        parent_renamed = Path(parent_stimulus.source).with_name("parent.wav")
        parent_renamed.unlink(missing_ok=True)
        Path(parent_stimulus.source).rename(parent_renamed)
        parent_stimulus.source = str(parent_renamed)
        print(f"  parent: {parent_renamed}", flush=True)

        for i in range(2):
            print(f"Mutating (call {i + 1}/2, redo_fraction={REDO_FRACTION})...", flush=True)
            torch.manual_seed(CONTROL_SEED)  # pins mutate_by_rediffusion's internal seed draw
            try:
                mutated = gen.mutate_by_rediffusion(parent_candidate, redo_fraction=REDO_FRACTION)
            except Exception as e:
                mutation_error = e
                print(f"  FAILED (this itself is informative): {type(e).__name__}: {e}", flush=True)
                break
            renamed = Path(mutated.source).with_name(f"mutation_call_{i + 1}.wav")
            renamed.unlink(missing_ok=True)
            Path(mutated.source).rename(renamed)
            audio, sr = sf.read(renamed)
            outputs.append(audio)
            print(
                f"  saved: {renamed} (peak={np.abs(audio).max():.4f}, "
                f"seed={mutated.metadata['mutation_seed']})",
                flush=True,
            )
    finally:
        gen.unload()

    print(flush=True)
    print("=" * 60, flush=True)
    if mutation_error is not None:
        print("Strict deterministic mode broke the mutation forward pass --", flush=True)
        print("this model hits an op with no deterministic CUDA implementation.", flush=True)
        print("Real answer: jitter is structurally unavoidable while mutation", flush=True)
        print("goes through the diffuser. Point in favor of DSP-layer mutation.", flush=True)
    elif len(outputs) == 2:
        a, b = outputs
        bit_exact = np.array_equal(a, b)
        if bit_exact:
            print("BIT-EXACT: deterministic mode + a pinned seed produced", flush=True)
            print("identical audio both times. redo_fraction can become a real,", flush=True)
            print("clean dial again (with an explicit seed alongside it).", flush=True)
        else:
            max_abs_diff = float(np.abs(a - b).max())
            rms_diff = float(np.sqrt(np.mean((a - b) ** 2)))
            print("NOT bit-exact, even with deterministic mode + a pinned seed.", flush=True)
            print(f"  max abs sample difference: {max_abs_diff:.6f}", flush=True)
            print(f"  RMS difference: {rms_diff:.6f}", flush=True)
            print("Deterministic mode didn't fully close the gap -- something", flush=True)
            print("else (or a partially-enforced op) is still injecting variation.", flush=True)
    if not strict_mode_ok:
        print(flush=True)
        print("NOTE: strict mode (warn_only=False) could not even be enabled --", flush=True)
        print("this run used warn_only=True as a fallback, see the error above.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
