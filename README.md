# tribe-lab

A shared local runtime for [TRIBE v2](https://github.com/facebookresearch/tribev2) (Meta's brain-response prediction model), plus tools and experiments built on top of it -- including an evolutionary search for stimuli whose predicted brain activity resembles a chosen target neural state.

See [mainStructure.md](mainStructure.md) for the full architecture and design philosophy. In short:

- **`tribe_core/`** -- the only thing that knows how to load TRIBE and run predictions. Everything else calls `TribeRuntime.predict(stimulus)` and never touches TRIBE directly.
- **`tools/brain_viewer/`** -- a local Gradio app for poking at TRIBE without writing code: text/audio in, cortical activation numbers, named brain regions, and rotatable 3D brain renders out.
- **`brain_utils/`** -- shared neuroscience utilities (network/atlas mapping) used across tools and experiments.
- **`inverse_search/`** -- a generic "search for a stimulus matching a target brain state" engine. See [inverse_search/DESIGN.md](inverse_search/DESIGN.md) for its pipeline and what's built vs. still planned.
- **`experiments/psyche_search/`** -- the first concrete experiment: searching for audio whose predicted brain response resembles psychedelic-state connectivity patterns. See its [README](experiments/psyche_search/README.md) and [research notes](experiments/psyche_search/restructure.md).

This is a solo research project, not a maintained library -- expect rough edges. The audio generator (Stable Audio Open, `inverse_search/generators/audio.py`) is built and has run real evolutionary searches against TRIBE; see [experiments/psyche_search/FINDINGS.md](experiments/psyche_search/FINDINGS.md) for what's actually been found so far, and [CONTRIBUTING.md](CONTRIBUTING.md) if you want to dig in or contribute.

## Requirements

- **Windows** (this repo works around several Windows-only bugs in TRIBE's dependencies -- see `tribe_core/_windows_patches.py`. It may work on Linux/Mac with those patches simply being no-ops, but that's untested.)
- **Python 3.12** (TRIBE requires >=3.11)
- **NVIDIA GPU with CUDA**, 8GB+ VRAM recommended for the audio modality. Text modality (LLaMA 3.2-3B) needs more headroom, has only been tested on the same 8GB card.
- Plenty of disk space (few GB+) on whatever drive you point model caches at -- see below.

## Setup

1. **Create a venv with Python 3.12** and activate it:
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install PyTorch with CUDA** (adjust the CUDA version to match your GPU driver -- check with `nvidia-smi`):
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Install TRIBE v2** from its GitHub repo (not on PyPI):
   ```
   pip install git+https://github.com/facebookresearch/tribev2.git
   ```

4. **Install the rest of the dependencies** used by the tools in this repo:
   ```
   pip install nilearn seaborn colorcet pyvista scikit-image plotly gradio mne static-ffmpeg
   ```

5. **Pick where model weights and caches live.** These get large (multiple GB) -- point them somewhere with room, not necessarily your system drive. Edit `config/tribe.yaml`'s `paths.model_root` (defaults to `G:/AI_Models` -- change this to wherever makes sense on your machine).

6. **(Text modality only) Get HuggingFace access to Llama 3.2.** The text modality loads `meta-llama/Llama-3.2-3B`, a gated model:
   - Request access at https://huggingface.co/meta-llama/Llama-3.2-3B (usually approved within minutes)
   - Create a **Read** access token at https://huggingface.co/settings/tokens
   - Run `hf auth login` and paste the token when prompted

   Audio modality doesn't need this step.

## Running things

**Brain Viewer** (the interactive tool -- start here):
```
python tools/brain_viewer/app.py
```
Opens a local Gradio app. Text or audio in, predicted brain activation out, including 3D rotatable cortical renders and named-region breakdowns.

**Sanity-check scripts** (`scripts/`) -- smaller, standalone checks used while building this out, not a maintained test suite:
- `check_vram.py` -- loads TRIBE, runs one prediction, reports real VRAM usage
- `test_runtime.py` -- exercises `tribe_core`'s runtime + caching
- `test_target.py` -- scores existing cached predictions against `inverse_search`'s current target definition

First run of anything will trigger some one-time downloads (TRIBE checkpoint, the relevant feature extractor for whichever modality you use, brain atlases) -- expect it to be slow the first time and fast after.

## License

Original code and documentation in this repository are licensed under **CC BY-NC-SA 4.0** -- see [LICENSE](LICENSE). Several dependencies (TRIBE v2 itself, Llama 3.2, planned use of Stable Audio Open, and the brain atlases used for region/network analysis) carry their own separate license terms -- see [NOTICE.md](NOTICE.md) for the full list before using this for anything beyond personal/non-commercial research.
