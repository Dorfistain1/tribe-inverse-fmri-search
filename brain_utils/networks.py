"""
Shared brain-space utility: maps TRIBE's fsaverage5 cortical vertices
onto the Yeo-7 canonical resting-state networks, and computes
network-level functional connectivity -- the construct the psychedelic
fMRI literature actually reports findings in terms of (e.g. "Visual <->
DMN coupling"), rather than single-vertex activation amplitude.

See mainStructure.md "Brain-Space Utilities" -- this is exactly the
reusable neuroscience layer that section anticipated, sitting between
tribe_core's raw predictions and inverse_search's target/fitness logic.
Not tied to psyche_search specifically; any future experiment reasoning
about networks rather than raw vertices can reuse this.

The Yeo-7 atlas only ships as a volumetric MNI parcellation (via
nilearn's Schaefer-2018 fetcher, which groups its 400 cortical parcels
into the 7 canonical networks) -- there's no ready-made fsaverage5
surface version, so it's projected onto the surface via nilearn's own
vol_to_surf with nearest-neighbor interpolation (these are categorical
network labels, not continuous data -- averaging/interpolating them
would produce meaningless fractional "labels").
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Order matches how they're conventionally listed in Yeo et al. 2011 /
# the Schaefer-2018 parcellation labels (abbreviations in the raw data:
# Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default).
YEO7_NETWORKS = [
    "Visual",
    "Somatomotor",
    "Dorsal Attention",
    "Ventral Attention",
    "Limbic",
    "Frontoparietal",
    "Default Mode",
]
_ABBREVIATIONS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

N_VERTICES_PER_HEMI = 10242  # fsaverage5


@lru_cache
def get_yeo7_vertex_labels() -> np.ndarray:
    """int array, length 20484 (TRIBE's vertex order: left hemi then
    right). Values 0-6 index YEO7_NETWORKS; -1 = unassigned (medial
    wall / background, not part of any of the 7 networks)."""
    from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_surf_fsaverage
    from nilearn.surface import vol_to_surf

    atlas = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas.labels]
    # labels[0] is "Background" (parcel value 0); parcel N's name is
    # labels[N]. Map each parcel index -> one of the 7 network indices.
    parcel_to_network = np.full(len(labels), -1, dtype=int)
    for parcel_idx, name in enumerate(labels):
        if name == "Background":
            continue
        abbrev = name.split("_")[2]
        parcel_to_network[parcel_idx] = _ABBREVIATIONS.index(abbrev)

    fsaverage = fetch_surf_fsaverage(mesh="fsaverage5")
    hemi_labels = []
    for hemi in ("left", "right"):
        parcel_values = vol_to_surf(
            atlas.maps, fsaverage[f"pial_{hemi}"], interpolation="nearest_most_frequent"
        )
        parcel_idx = parcel_values.astype(int)
        hemi_labels.append(parcel_to_network[parcel_idx])
    return np.concatenate(hemi_labels)


def connectivity_matrix(activity: np.ndarray) -> np.ndarray:
    """activity: (time, 20484), a full stimulus's predicted response
    (not a single timestep -- connectivity is inherently a multi-
    timepoint construct, it's a correlation over time). Returns a 7x7
    Pearson correlation matrix between Yeo-7 network-averaged time
    series, indexed by YEO7_NETWORKS in both dimensions.
    """
    labels = get_yeo7_vertex_labels()
    n_timesteps = activity.shape[0]
    network_series = np.zeros((n_timesteps, len(YEO7_NETWORKS)))
    for i in range(len(YEO7_NETWORKS)):
        mask = labels == i
        if mask.any():
            network_series[:, i] = activity[:, mask].mean(axis=1)
    return np.corrcoef(network_series.T)
