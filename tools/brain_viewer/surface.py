"""
Cortical surface rendering for the Brain Viewer tool.

Uses tribev2's own PlotBrainNilearn (tribev2.plotting.cortical) rather
than hand-rolling a nilearn wrapper -- it already encodes the correct
fsaverage5 hemisphere-split convention TRIBE's output uses (first half
of the 20,484 vertices = left hemisphere, second half = right; see
tribev2/plotting/base.py get_stat_map / get_hemis) and produces the same
look as TRIBE's own demo plots.

Imports are lazy (inside functions) so the NILEARN_DATA/HF_HOME env vars
tribe_core.TribeRuntime sets are already in place before nilearn decides
where to cache the fsaverage5 mesh download (moot in practice -- fsaverage5
ships bundled inside the nilearn package itself, no network needed).
"""

from __future__ import annotations

import html as html_module
import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_plotter = None


def _centered(inner_html: str) -> str:
    return f'<div style="display:flex;justify-content:center">{inner_html}</div>'


def get_plotter():
    global _plotter
    if _plotter is None:
        import matplotlib

        matplotlib.use("Agg")  # headless server process, no display
        from tribev2.plotting.cortical import PlotBrainNilearn

        _plotter = PlotBrainNilearn()  # defaults to mesh="fsaverage5"
    return _plotter


def render_brain_png(
    activity_row: np.ndarray, vmin: float | None = None, vmax: float | None = None
) -> np.ndarray:
    """activity_row: 1D array of length 20484 (one timestep).
    Returns an RGB image array (for gr.Image). Pass vmin/vmax to share a
    color scale across multiple frames (see render_video).

    Calls nilearn's plot_surf_stat_map directly instead of going through
    tribev2's plot_surf wrapper: that wrapper hardcodes bg_on_data=True
    with no override, which multiplies each vertex's color by the (often
    dark, in sulci) grayscale sulcal-depth map -- fine with tribev2's
    "hot" default (saturated colors survive being darkened), but a
    near-white RdBu_r color multiplied by a dark value crushes to near
    black, which is most of the cortical surface (sulci dominate the
    area). nilearn's own default is bg_on_data=False, matching how
    render_brain_interactive (nilearn's view_surf) already looks.
    """
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_surf_stat_map
    from PIL import Image

    plotter = get_plotter()
    hemis = plotter.get_hemis(activity_row)
    views = {"left": (0, 180), "right": (0, 0)}  # tribev2.plotting.cortical.VIEW_DICT

    fig, axarr = plt.subplots(
        1, 2, figsize=(6, 3), subplot_kw={"projection": "3d"}, gridspec_kw={"wspace": 0}
    )
    for ax, hemi_name in zip(axarr, ["left", "right"]):
        h = hemis[hemi_name]
        plot_surf_stat_map(
            surf_mesh=h["surf_mesh"],
            stat_map=h["stat_map"],
            bg_map=h["bg_map"],
            hemi=hemi_name,
            view=views[hemi_name],
            axes=ax,
            figure=fig,
            cmap="RdBu_r",
            symmetric_cbar=True,
            vmin=vmin,
            vmax=vmax,
            bg_on_data=False,
            colorbar=(hemi_name == "right"),
        )
        ax.set_box_aspect(None, zoom=1.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def render_brain_interactive(
    activity_row: np.ndarray, vmin: float | None = None, vmax: float | None = None
) -> str:
    """activity_row: 1D array of length 20484 (one timestep). Returns an
    iframe HTML string (for gr.HTML) with a rotatable/zoomable plotly
    view of both hemispheres, using the same mesh + hemisphere split as
    render_brain_png -- just a different nilearn rendering backend
    (plotly instead of matplotlib) so it's draggable in the browser
    instead of a static image."""
    from nilearn.plotting import view_surf

    plotter = get_plotter()
    stat = plotter.get_stat_map(activity_row)
    mesh = plotter._mesh["both"]
    view = view_surf(
        surf_mesh=(mesh["coords"], mesh["faces"]),
        surf_map=stat["both"],
        bg_map=mesh["bg_map"],
        vmin=vmin,
        vmax=vmax,
    )
    view.resize(width=900, height=700)
    return _centered(view.get_iframe())


def render_video_interactive(
    activity: np.ndarray, vmin: float | None = None, vmax: float | None = None, max_frames: int = 60
) -> str:
    """activity: 2D array (time, 20484). Returns an HTML blob with a
    rotatable/zoomable plotly scene that animates over time (play button
    + scrubber slider) -- rotation still works mid-playback since it's
    one persistent 3D scene, just with per-frame vertex colors.

    Doesn't reuse nilearn's view_surf (single frame only) or its
    per-vertex-color helpers (colorscale/_get_vertexcolor) -- those are
    private, undocumented internals split oddly across its matplotlib
    and plotly backend modules. Recomputes vertex colors directly with
    plain matplotlib color mapping instead, which is also considerably
    faster: no per-frame 3D figure re-render, just a color array.

    Payload scales with n_timesteps x n_vertices (colors are embedded
    as literal data in the HTML), so long clips are subsampled to at
    most max_frames evenly-spaced timesteps to keep the page loadable.
    """
    import matplotlib
    import plotly.graph_objects as go

    from tools.brain_viewer.regions import top_regions

    plotter = get_plotter()
    mesh = plotter._mesh["both"]
    coords, faces = mesh["coords"], mesh["faces"]
    x, y, z = coords.T
    i, j, k = faces.T

    # RdBu_r + symmetric range, matching render_brain_png/interactive --
    # see the comment there. "hot" here made the whole brain look black.
    # No bg_map blending either (unlike an earlier version of this
    # function): multiplying color by a dark sulcal-depth map crushes a
    # light diverging cmap like RdBu_r toward black almost everywhere,
    # same root cause as render_brain_png's bg_on_data=False fix.
    if vmin is None or vmax is None:
        bound = max(abs(float(activity.min())), abs(float(activity.max())))
        vmin, vmax = -bound, bound
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps["RdBu_r"]

    def vertex_colors(row: np.ndarray) -> list[str]:
        rgb = (cmap(norm(row))[:, :3] * 255).astype(np.uint8)
        return [f"rgb({r},{g},{b})" for r, g, b in rgb]

    def frame_annotation(t: int) -> str:
        regions = top_regions(activity[t], k=5)
        lines = [f"<b>t={t}</b>"] + [f"{name}: {val:+.2f}" for name, val in regions]
        return "<br>".join(lines)

    n_timesteps = activity.shape[0]
    frame_indices = np.linspace(0, n_timesteps - 1, min(n_timesteps, max_frames)).astype(int)

    no_axes = dict(visible=False)
    scene_layout = dict(aspectmode="data", xaxis=no_axes, yaxis=no_axes, zaxis=no_axes)

    def side_annotation(text: str) -> dict:
        # Positioned just outside the 3D scene's right edge (paper coords,
        # independent of the scene's own camera/zoom) -- a title would be
        # one line and clip; this wraps and has room to breathe.
        return dict(
            text=text, xref="paper", yref="paper", x=1.02, y=0.98,
            xanchor="left", yanchor="top", showarrow=False, align="left",
            font=dict(size=12), width=210,  # forces word-wrap; a long
            # region name ("Temporo-Parieto-Occipital Junction") is one
            # unbroken run with no <br>, so without an explicit box width
            # it overflows past the margin instead of wrapping.
        )

    base_mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, vertexcolor=vertex_colors(activity[frame_indices[0]]))
    frames = [
        go.Frame(
            data=[go.Mesh3d(vertexcolor=vertex_colors(activity[t]))],
            layout=go.Layout(annotations=[side_annotation(frame_annotation(t))]),
            name=str(t),
        )
        for t in frame_indices
    ]

    fig = go.Figure(data=[base_mesh], frames=frames)
    fig.update_layout(
        scene=scene_layout,
        width=1100,
        height=700,
        margin=dict(l=0, r=260, t=10, b=0),
        annotations=[side_annotation(frame_annotation(frame_indices[0]))],
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play", method="animate", args=[
                    None, {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True}
                ]),
                dict(label="Pause", method="animate", args=[
                    [None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}
                ]),
            ],
        )],
        sliders=[dict(steps=[
            dict(
                method="animate",
                args=[[str(t)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                label=str(t),
            )
            for t in frame_indices
        ])],
    )
    # gr.HTML inserts content via innerHTML, and browsers never execute
    # <script> tags inserted that way -- so a raw fig.to_html() fragment
    # (a <div> plus a <script> that calls Plotly.newPlot) silently does
    # nothing: no error, just a blank component. Wrapping it in a real
    # <iframe srcdoc="..."> gives it its own document, where the script
    # executes normally -- the same trick nilearn's own get_iframe() uses
    # for the single-frame view (see render_brain_interactive).
    full_html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    escaped = html_module.escape(full_html, quote=True)
    iframe = (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:1100px;height:700px;border:none"></iframe>'
    )
    return _centered(iframe)


def render_video(activity: np.ndarray, out_path: str, fps: int = 2) -> str:
    """activity: 2D array (time, 20484). Renders one frame per timestep
    (both hemispheres, like render_brain_png) with a color scale shared
    across the whole clip, and stitches them into an mp4 via ffmpeg
    (already on PATH -- see tribe_core._windows_patches).

    Deliberately doesn't use tribev2's own plot_timesteps_mp4: that
    method builds a plain 2D axes instead of a 3D one before calling the
    surface plotter, which crashes (AttributeError: 'Axes' object has no
    attribute 'view_init') -- an upstream bug, not a Windows portability
    issue like the others in _windows_patches.py.
    """
    from PIL import Image

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = float(activity.min()), float(activity.max())

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(activity.shape[0]):
            img = render_brain_png(activity[i], vmin=vmin, vmax=vmax)
            # libx264 requires even width/height; matplotlib's tight
            # bbox crop can land on an odd pixel count either way.
            h, w, _ = img.shape
            if h % 2 or w % 2:
                img = np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
            Image.fromarray(img).save(f"{tmp_dir}/frame_{i:05d}.png")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", f"{tmp_dir}/frame_%05d.png",
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    return str(out_path)
