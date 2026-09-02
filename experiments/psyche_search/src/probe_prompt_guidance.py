"""
Cheap probe: how much does the text prompt vs. guidance_scale actually
shape output? User's complaint: the fixed "acoustic guitar song..."
prompt feels like it decides ~80% of the character, and evolution is
just nudging the remaining ~20%.

Architecture fact checked directly against diffusers' source
(pipeline_stable_audio.py): guidance_scale (7.0 here, matches the
library's own default) is what extrapolates the output 7x away from a
zero-conditioning baseline toward the prompt's embedding -- that
push, not the prompt text alone, is most of what "acoustic guitar
song" forces onto every candidate.

Known landmine (FINDINGS.md): "ambient music" as a prompt
deterministically produced near-silent garbage on this model (peak
~6e-5). A prompt like "" sits in similar "no specific content"
territory -- might be fine, might hit the same wall. Check peak
amplitude for every clip before listening to any of them.

One fixed latent (seed 0) across every combo -- isolates the effect of
prompt/guidance_scale alone, same as swapping only the prompt did when
the "ambient" bug was first found.

Run this yourself, directly, in your own terminal window:

    .venv\\Scripts\\python.exe experiments\\psyche_search\\src\\probe_prompt_guidance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tribe_core.env import configure_hf_cache

MODEL_ROOT = "G:/AI_Models"
configure_hf_cache(MODEL_ROOT)  # before any other project import -- see FINDINGS.md's HF cache gotcha

import numpy as np
import soundfile as sf

from inverse_search.generators.audio import AudioGenerator

OUT_DIR = Path("experiments/psyche_search/data/prompt_guidance_probe")
DURATION_S = 5.0
SEED = 0

# (label, prompt, guidance_scale) -- current default first, as a baseline
# to compare everything else against.
COMBOS = [
    ("current_default", "acoustic guitar song with a clear, memorable melody", 7.0),
    ("guitar_lowcfg", "acoustic guitar song with a clear, memorable melody", 3.0),
    ("empty_prompt_cfg7", "", 7.0),
    ("empty_prompt_cfg3", "", 3.0),
    ("empty_prompt_nocfg", "", 1.0),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gen = AudioGenerator(model_root=MODEL_ROOT, duration_s=DURATION_S, output_dir=str(OUT_DIR))
    gen.load()

    latent = gen._random_latent(seed=SEED)

    print(f"{'label':<22} {'prompt':<45} {'cfg':>4}  {'peak':>8}", flush=True)
    for label, prompt, guidance_scale in COMBOS:
        gen.prompt = prompt
        gen.guidance_scale = guidance_scale
        stimulus = gen.decode_latent(latent, identifier=label)
        audio, _sr = sf.read(stimulus.source)
        peak = np.abs(audio).max()
        flag = "  <-- near-silent, same shape as the 'ambient' bug" if peak < 1e-3 else ""
        print(f"{label:<22} {prompt!r:<45} {guidance_scale:>4.1f}  {peak:>8.4f}{flag}", flush=True)

    gen.unload()
    print(flush=True)
    print(f"Clips: {OUT_DIR}", flush=True)
    print("Listen to all 5 -- does dropping the prompt / lowering guidance_scale actually", flush=True)
    print("open up variety, or does it just get quieter/more textureless?", flush=True)


if __name__ == "__main__":
    main()
