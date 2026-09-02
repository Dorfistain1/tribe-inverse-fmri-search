"""
Quick coherence check for candidate psychedelic-style prompts, at the
now-known-necessary guidance_scale=7.0 (yesterday's probe: lower CFG or
empty prompt both break on this model, FINDINGS.md 2026-09-01). Point:
pick ONE prompt to commit to before an expensive real-TRIBE run, not
after -- same landmine-check discipline as every prompt tried so far
("ambient music" deterministically produced near-silence).

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\probe_psychedelic_prompts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tribe_core.env import configure_deterministic_mode, configure_hf_cache

MODEL_ROOT = "G:/AI_Models"
configure_hf_cache(MODEL_ROOT)  # before any other project import -- see FINDINGS.md's HF cache gotcha
configure_deterministic_mode()  # same rule -- before any other project import

import numpy as np
import soundfile as sf

from inverse_search.generators.audio import AudioGenerator

OUT_DIR = Path("experiments/psyche_search/data/psychedelic_prompt_probe")
DURATION_S = 5.0
SEED = 0
GUIDANCE_SCALE = 7.0

PROMPTS = {
    "psy_ambient": "psychedelic ambient soundscape with swirling, shifting layers and hypnotic texture",
    "psy_electronic": "trippy electronic music with warped, morphing textures and a hypnotic pulse",
    "psy_rock": "hypnotic psychedelic rock with driving rhythm and swirling guitar effects",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gen = AudioGenerator(
        model_root=MODEL_ROOT, duration_s=DURATION_S, output_dir=str(OUT_DIR), guidance_scale=GUIDANCE_SCALE
    )
    gen.load()

    latent = gen._random_latent(seed=SEED)

    print(f"{'label':<16} {'prompt':<75} {'peak':>8}", flush=True)
    for label, prompt in PROMPTS.items():
        gen.prompt = prompt
        stimulus = gen.decode_latent(latent, identifier=label)
        audio, _sr = sf.read(stimulus.source)
        peak = np.abs(audio).max()
        flag = "  <-- near-silent, same shape as the 'ambient' bug" if peak < 1e-3 else ""
        print(f"{label:<16} {prompt!r:<75} {peak:>8.4f}{flag}", flush=True)

    gen.unload()
    print(flush=True)
    print(f"Clips: {OUT_DIR}", flush=True)
    print("Listen to all 3 -- which one is most coherent AND most distinctly", flush=True)
    print("'psychedelic' rather than just generic music?", flush=True)


if __name__ == "__main__":
    main()
