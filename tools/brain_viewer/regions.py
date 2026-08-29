"""
Named-region summary for the Brain Viewer tool.

tribev2.utils already has get_hcp_labels/get_topk_rois for exactly this
(HCP-MMP1 / Glasser atlas on fsaverage), but that function hardcodes its
subjects_dir via mne.datasets.sample.data_path() -- which resolves to
MNE's "sample" dataset (a multi-GB MEG/MRI tutorial dataset), unrelated
to what's actually needed here and far bigger than the ~240MB fsaverage
download this actually requires. Reimplemented against our own
subjects_dir instead; the label-parsing logic itself is copied from
tribev2.utils.get_hcp_labels since that part is correct, just the path
resolution differs.

Uses the *combined* HCP-MMP1 parcellation (23 regions per hemisphere)
rather than the full 360-region atlas -- those combined labels are
human-readable ("Early Visual Cortex", "Auditory Association Cortex",
etc, per Glasser et al. 2016 supplementary), unlike the raw atlas's
cryptic codes ("STSvp", "PGi"), so no separate code->description lookup
table is needed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

SUBJECTS_DIR = Path("G:/AI_Models/mne_data/subjects")


def ensure_atlas_downloaded() -> None:
    """One-time fetch (~240MB): fsaverage surfaces + HCP-MMP1 annotation.
    Safe to call every time -- mne no-ops if already present."""
    import mne

    SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    mne.datasets.fetch_fsaverage(subjects_dir=SUBJECTS_DIR, verbose=False)
    mne.datasets.fetch_hcp_mmp_parcellation(
        subjects_dir=SUBJECTS_DIR, accept=True, combine=True, verbose=False
    )


@lru_cache
def get_hcp_labels(hemi: str = "both", n_vertices_per_hemi: int = 10242) -> dict[str, np.ndarray]:
    """name -> array of vertex indices (into the 20,484-long TRIBE
    activation vector, right hemisphere already offset)."""
    import mne
    import neuralset.utils as ns_utils

    ensure_atlas_downloaded()

    if hemi in ("left", "right"):
        with ns_utils.ignore_all():
            labels = mne.read_labels_from_annot(
                "fsaverage", "HCPMMP1_combined", hemi="both", subjects_dir=SUBJECTS_DIR
            )
        label_to_vertices = {}
        for label in labels:
            name, vertices = label.name, np.array(label.vertices)
            # tribev2.utils.get_hcp_labels only slices name[2:] (stripping
            # a leading "L_"/"R_") for the uncombined 360-region atlas;
            # the combined 23-per-hemisphere labels don't have that
            # prefix, just an MNE-added "-lh"/"-rh" suffix.
            name = name.replace("_ROI", "")
            if (hemi == "right" and "-lh" in name) or (hemi == "left" and "-rh" in name):
                continue
            name = name.replace("-rh", "").replace("-lh", "")
            label_to_vertices[name] = np.array(vertices)
        index_offset = n_vertices_per_hemi if hemi == "right" else 0
        return {
            k: v[v < n_vertices_per_hemi] + index_offset for k, v in label_to_vertices.items()
        }

    left = get_hcp_labels(hemi="left", n_vertices_per_hemi=n_vertices_per_hemi)
    right = get_hcp_labels(hemi="right", n_vertices_per_hemi=n_vertices_per_hemi)
    return {k: np.concatenate([left[k], right[k]]) for k in left}


def top_regions(activity_row: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
    """activity_row: 1D array of length 20484 (one timestep, or a
    time-average). Returns (region name, mean activation) sorted by
    |mean activation| descending."""
    labels = get_hcp_labels(hemi="both")
    scored = [
        (name, float(activity_row[idx].mean()))
        for name, idx in labels.items()
        if name != "???"  # medial wall / unlabeled vertices, not a real region
    ]
    scored.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return scored[:k]
