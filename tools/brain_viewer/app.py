"""
Brain Viewer -- shared dev tool, not an experiment.

Text or audio in, predicted cortical activation numbers out, plus a
real cortical-surface render. Exists so tribe_core can be poked at
without writing code. Any experiment can grow on top of this later; it
makes no scientific claims of its own.

Usage:
    python tools/brain_viewer/app.py
Then open the printed local URL.
"""

import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import gradio as gr
import numpy as np
import pandas as pd

from tribe_core import Stimulus, TribeRuntime
from tools.brain_viewer.regions import top_regions
from tools.brain_viewer.surface import (
    render_brain_interactive,
    render_brain_png,
    render_video,
    render_video_interactive,
)

TOP_K = 50
N_VERTICES = 20484

print("Loading TRIBE runtime...")
runtime = TribeRuntime()
TMP_DIR = Path(tempfile.gettempdir()) / "brain_viewer"
TMP_DIR.mkdir(exist_ok=True)
VIDEO_DIR = TMP_DIR / "videos"
VIDEO_DIR.mkdir(exist_ok=True)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _summarize(activity: np.ndarray, header_lines: list[str]) -> tuple[str, pd.DataFrame]:
    mean_per_vertex = activity.mean(axis=0)
    top_idx = np.argsort(np.abs(mean_per_vertex))[::-1][:TOP_K]

    summary_lines = header_lines + [
        f"shape (time x vertices): {activity.shape}",
        f"mean / std:   {activity.mean():.4f} / {activity.std():.4f}",
        f"min / max:    {activity.min():.4f} / {activity.max():.4f}",
        "",
        f"top {TOP_K} vertices by |mean activation| shown in the table below.",
        "Use the 3D Brain View section below to render any timestep.",
    ]

    table = pd.DataFrame(activity[:, top_idx], columns=[f"v{i}" for i in top_idx])
    table.insert(0, "timestep", range(len(table)))
    return "\n".join(summary_lines), table


STAGE_PROGRESS = {
    "loading model": 0.05,
    "model ready": 0.1,
    "building events (transcription/TTS)": 0.15,
    "predicting brain activity": 0.6,
    "cache hit": 0.95,
}


def _run(stimulus: Stimulus, progress: gr.Progress):
    def on_stage(label: str) -> None:
        progress(STAGE_PROGRESS.get(label, 0.5), desc=label)

    prediction = runtime.predict(stimulus, use_cache=True, on_stage=on_stage)
    progress(0.98, desc="formatting result")
    activity = prediction.activity  # (time, vertices)

    # already saved to disk by runtime.predict()'s cache -- no separate
    # download step needed, just point at the real file.
    cache_key = runtime.cache.key(stimulus, prediction.model_identity)
    saved_path = (runtime.cache.root / cache_key).with_suffix(".npy").resolve()

    header = [
        f"stimulus_id:  {prediction.stimulus_id}",
        f"modality:     {prediction.modality}",
        f"model:        {prediction.model_identity}",
        f"segments kept: {len(prediction.timing)}",
    ]
    summary, table = _summarize(activity, header)
    summary += f"\nfull raw array saved to: {saved_path}"
    return summary, table, activity


def predict_from_text(text, progress=gr.Progress()):
    if not text or not text.strip():
        return "Enter some text first.", None, None
    path = TMP_DIR / f"text_{_hash(text.encode())}.txt"
    path.write_text(text, encoding="utf-8")
    stimulus = Stimulus(identifier=path.stem, modality="text", source=str(path))
    return _run(stimulus, progress)


def predict_from_audio(audio_path, progress=gr.Progress()):
    if audio_path is None:
        return "Upload an audio file first.", None, None
    data = Path(audio_path).read_bytes()
    stimulus = Stimulus(identifier=f"audio_{_hash(data)}", modality="audio", source=audio_path)
    return _run(stimulus, progress)


def load_npy(file_path):
    if file_path is None:
        return "Upload a .npy file first.", None, None
    activity = np.load(file_path)
    if activity.ndim != 2 or activity.shape[1] != N_VERTICES:
        return (
            f"Unexpected shape {activity.shape}, expected (time, {N_VERTICES}).",
            None,
            None,
        )
    summary, table = _summarize(activity, [f"loaded from: {file_path}"])
    return summary, table, activity


def update_slider(activity):
    if activity is None:
        return gr.Slider(minimum=0, maximum=1, value=0)
    n_timesteps = activity.shape[0]
    return gr.Slider(minimum=0, maximum=max(n_timesteps - 1, 1), value=0, step=1)


def _clamp_timestep(activity, timestep):
    return int(min(max(timestep, 0), activity.shape[0] - 1))


def render(activity, timestep):
    if activity is None:
        return None
    return render_brain_png(activity[_clamp_timestep(activity, timestep)])


def render_interactive(activity, timestep):
    if activity is None:
        return None
    return render_brain_interactive(activity[_clamp_timestep(activity, timestep)])


def render_regions(activity, timestep):
    if activity is None:
        return ""
    row = activity[_clamp_timestep(activity, timestep)]
    lines = [f"{name}: {val:+.4f}" for name, val in top_regions(row, k=10)]
    return "\n".join(lines)


def render_video_from_state(activity, progress=gr.Progress()):
    if activity is None:
        return None
    progress(0.1, desc=f"rendering {activity.shape[0]} frames...")
    cache_key = _hash(activity.tobytes())
    out_path = VIDEO_DIR / f"{cache_key}.mp4"
    if not out_path.exists():
        render_video(activity, str(out_path))
    progress(1.0, desc="done")
    return str(out_path)


def render_video_interactive_from_state(activity, progress=gr.Progress()):
    if activity is None:
        return None
    progress(0.2, desc="computing frame colors...")
    html = render_video_interactive(activity)
    progress(1.0, desc="done")
    return html


with gr.Blocks(title="TRIBE Brain Viewer") as demo:
    gr.Markdown("# TRIBE v2 Brain Activation Viewer")

    activity_state = gr.State(value=None)

    with gr.Tab("Text"):
        text_in = gr.Textbox(label="Text", lines=4, placeholder="Type something...")
        text_btn = gr.Button("Predict")
    with gr.Tab("Audio"):
        audio_in = gr.Audio(label="Audio", type="filepath")
        audio_btn = gr.Button("Predict")
    with gr.Tab("Load Saved"):
        gr.Markdown(
            f"Load any previously saved prediction "
            f"(`outputs/tribe_predictions/*.npy`, or one saved elsewhere)."
        )
        npy_in = gr.File(label="Prediction .npy (time x 20484 vertices)", file_types=[".npy"])
        npy_btn = gr.Button("Load")

    summary_out = gr.Textbox(label="Summary", lines=12)
    table_out = gr.Dataframe(label=f"Top {TOP_K} vertices (time x vertex)")

    gr.Markdown("## 3D Brain View")
    timestep_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="Timestep")
    with gr.Row():
        interactive_btn = gr.Button("Render Brain (rotatable)")
        video_interactive_btn = gr.Button("Render Video (rotatable, subsampled)")
    regions_out = gr.Textbox(label="Top activated named regions (at selected timestep)", lines=10)
    gr.Markdown(
        "Each value is TRIBE's **predicted** brain activity for that region, "
        "averaged across its vertices -- not a measurement of a real brain (see "
        "mainStructure.md \"Scientific Boundary\"). TRIBE predicts BOLD signal "
        "(blood-oxygen level, the standard fMRI proxy for neural activity), trained "
        "on per-vertex *normalized* responses, so values behave like a z-score "
        "relative to that region's own typical level: **0 = normal/typical activity "
        "for that region, positive = more activated than normal, negative = LESS "
        "activated than normal** (not \"inactive,\" just below its own usual level). "
        "Magnitude is roughly \"how many standard deviations from typical.\" Sorted "
        "by |value|, so a region can top the list from being unusually *suppressed* "
        "just as easily as unusually active."
    )
    brain_interactive = gr.HTML(label="Rotatable cortical activation")
    brain_video_interactive = gr.HTML(label="Rotatable activation over time")

    with gr.Accordion("Static renders (non-interactive, faster)", open=False):
        with gr.Row():
            render_btn = gr.Button("Render Brain (static)")
            video_btn = gr.Button("Render Video (static, all timesteps)")
        brain_image = gr.Image(label="Cortical activation (left + right hemisphere)")
        brain_video = gr.Video(label="Activation over time")

    text_btn.click(
        predict_from_text, inputs=text_in, outputs=[summary_out, table_out, activity_state]
    ).then(update_slider, inputs=activity_state, outputs=timestep_slider)

    audio_btn.click(
        predict_from_audio, inputs=audio_in, outputs=[summary_out, table_out, activity_state]
    ).then(update_slider, inputs=activity_state, outputs=timestep_slider)

    npy_btn.click(
        load_npy, inputs=npy_in, outputs=[summary_out, table_out, activity_state]
    ).then(update_slider, inputs=activity_state, outputs=timestep_slider)

    render_btn.click(
        render, inputs=[activity_state, timestep_slider], outputs=brain_image
    ).then(render_regions, inputs=[activity_state, timestep_slider], outputs=regions_out)
    interactive_btn.click(
        render_interactive, inputs=[activity_state, timestep_slider], outputs=brain_interactive
    ).then(render_regions, inputs=[activity_state, timestep_slider], outputs=regions_out)
    video_btn.click(render_video_from_state, inputs=activity_state, outputs=brain_video)
    video_interactive_btn.click(
        render_video_interactive_from_state, inputs=activity_state, outputs=brain_video_interactive
    )


if __name__ == "__main__":
    demo.launch()
