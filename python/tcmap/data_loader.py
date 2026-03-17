"""Load experiment data from the MATLAB .mat files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from .mesh import Mesh


def load_experiment(mat_path: str | Path) -> tuple[list[Mesh], Mesh]:
    """Load an experiment .mat file.

    Parameters
    ----------
    mat_path : path to ``experiment_data_X.mat``

    Returns
    -------
    sources : list[Mesh] — sequence of deforming meshes (iS1)
    target  : Mesh        — constant target shape (iS2)
    """
    data = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    # --- Target (constant) ---
    s2_raw = data["iS2"]
    if hasattr(s2_raw, "__len__"):
        s2_struct = s2_raw[0] if len(s2_raw) > 0 else s2_raw
    else:
        s2_struct = s2_raw
    target = _struct_to_mesh(s2_struct)

    # --- Sources (time sequence) ---
    s1_raw = data["iS1"]
    sources = []
    if hasattr(s1_raw, "__len__"):
        for s in s1_raw:
            sources.append(_struct_to_mesh(s))
    else:
        sources.append(_struct_to_mesh(s1_raw))

    return sources, target


def _struct_to_mesh(s) -> Mesh:
    """Convert a MATLAB struct (from loadmat) to a Mesh object."""
    # Try different field names — the data uses VERT/TRIV or surface.VERT/surface.TRIV
    verts = _get_field(s, "VERT")
    faces = _get_field(s, "TRIV")

    if verts is None and hasattr(s, "surface"):
        verts = _get_field(s.surface, "VERT")
        faces = _get_field(s.surface, "TRIV")

    if verts is None:
        raise ValueError("Could not find vertex data in struct")

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    # MATLAB uses 1-based indexing
    if faces.min() >= 1:
        faces = faces - 1

    return Mesh(verts, faces)


def _get_field(s, name: str):
    """Get a field from a MATLAB struct, handling both attribute and dict access."""
    if hasattr(s, name):
        return getattr(s, name)
    if isinstance(s, dict) and name in s:
        return s[name]
    return None
