"""The recorded experiments of Sec. 4.3 (data/experiment_data_*.mat, stored with Git LFS).

Every file holds the meshes of one experiment, as built by objectsurfacemeshing.m from
the color-segmented RGB-D frames: iS1 (one mesh per iteration, the deforming object)
and iS2 (the target mesh). Coordinates are in meters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from .mesh import Mesh

DATA = Path(__file__).resolve().parents[2] / "data"

EXPERIMENTS = {
    1: "Mexican hat",
    2: "T-shaped pool noodle",
    3: "Pillow",
    4: "Rectangular foam sheet",
    5: "Foam cut-out",
}


def _mesh(s) -> Mesh:
    faces = np.asarray(s.TRIV, dtype=np.int64)
    return Mesh(np.asarray(s.VERT, dtype=np.float64), faces - 1 if faces.min() >= 1 else faces)


def load_experiment(number: int, folder: Path = DATA) -> tuple[list[Mesh], Mesh]:
    """The meshes of the experiment (one per iteration) and the target mesh."""
    path = Path(folder) / f"experiment_data_{number}.mat"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    if path.stat().st_size < 1024:
        raise RuntimeError(f"{path} is a Git LFS pointer: clone the repository with Git LFS (git lfs pull)")
    data = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    frames = [_mesh(s) for s in np.atleast_1d(data["iS1"])]
    target = _mesh(np.atleast_1d(data["iS2"])[0])
    return frames, target
