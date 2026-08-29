# Third-Party Notices

This repository's own original code and documentation are licensed
under CC BY-NC-SA 4.0 (see LICENSE). Several dependencies and models it
uses carry their own, separate licenses and terms, listed below. Using
this repository means also complying with those where applicable.

## facebookresearch/tribev2 (the TRIBE v2 model + library)

`tribe_core` depends on `tribev2` (installed from
`github.com/facebookresearch/tribev2`), which is licensed under
**CC BY-NC 4.0** (Attribution-NonCommercial) by Meta Platforms, Inc.
This repository does not vendor or redistribute tribev2's source --
it's installed as a normal pip dependency -- but credit and license
terms are noted here regardless. Non-commercial use only; see
https://github.com/facebookresearch/tribev2/blob/main/LICENSE.

## Llama 3.2 (via tribev2, text modality only)

The text modality path (`Stimulus(modality="text", ...)`) loads
`meta-llama/Llama-3.2-3B` as tribev2's text feature extractor. That
model is distributed under the Llama 3.2 Community License.

Per that license, any product or service built using Llama Materials
must display the attribution:

> Built with Llama

This notice satisfies that requirement for this repository. If a future
experiment fine-tunes or redistributes a derivative of the Llama weights
(not just calling the pretrained model at inference time), the derivative
must also be named with a "Llama" prefix per the license -- see
https://huggingface.co/meta-llama/Llama-3.2-3B for full terms.

The audio modality (Wav2Vec-BERT) and the TRIBE encoder itself are not
subject to this license.

## Stable Audio Open (inverse_search's audio generator)

`inverse_search`'s audio candidate generator (`generators/audio.py`)
uses `stabilityai/stable-audio-open-1.0`,
licensed under the **Stability AI Community License**: free for
research and non-commercial use, and for commercial use by
organizations with under $1,000,000 in annual revenue. See
https://huggingface.co/stabilityai/stable-audio-open-1.0 and
https://stability.ai/license for full terms.

## HCP-MMP1 / Glasser parcellation (tools/brain_viewer/regions.py)

The named cortical regions shown in the Brain Viewer tool use the
HCP-MMP1 parcellation (Glasser et al., 2016, "A multi-modal
parcellation of human cerebral cortex," *Nature* 536:171-178), fetched
via MNE-Python. Use of Human Connectome Project data/derivatives
requires acknowledging WU-Minn HCP in any public presentation of
results -- see https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms.

## Schaefer-2018 / Yeo-7 network atlas (brain_utils/networks.py)

Network-level connectivity analysis uses the Schaefer-2018 cortical
parcellation (Schaefer et al., 2018, "Local-Global Parcellation of the
Human Cerebral Cortex from Intrinsic Functional Connectivity MRI,"
*Cerebral Cortex* 29:3095-3114), grouped into the Yeo-7 networks (Yeo et
al., 2011). Distributed by the Computational Brain Imaging Group (CBIG)
at https://github.com/ThomasYeoLab/CBIG -- citation requested per that
repository's terms.

## Everything else (PyTorch, nilearn, Gradio, matplotlib, etc.)

The many other pip dependencies this project uses each retain their
own (generally permissive: BSD/MIT/Apache-style) open-source licenses
as declared on PyPI. Not individually enumerated here -- standard
practice for a Python project's dependency tree.
