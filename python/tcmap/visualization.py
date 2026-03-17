"""Polyscope-based 3D visualisation of meshes and surface maps."""

from __future__ import annotations

import numpy as np
import polyscope as ps

from .mesh import Mesh


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise array values to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def xyz_color(mesh: Mesh) -> np.ndarray:
    """Generate an RGB colour per vertex from normalised XYZ position."""
    r = _normalise(mesh.vertices[:, 0])
    g = _normalise(mesh.vertices[:, 1])
    b = _normalise(mesh.vertices[:, 2])
    return np.column_stack([r, g, b])


def init_polyscope():
    """Initialise Polyscope (call once)."""
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("y_up")


def show_mesh(name: str, mesh: Mesh, color: np.ndarray | None = None, enabled: bool = True):
    """Register a mesh in Polyscope.

    Parameters
    ----------
    name  : str — display name
    mesh  : Mesh
    color : (M, 3) float array in [0,1] — per-vertex RGB colour
    """
    sm = ps.register_surface_mesh(name, mesh.vertices, mesh.faces)
    if color is not None:
        sm.add_color_quantity("map_color", color, enabled=enabled)
    return sm


def visualise_map(
    source: Mesh,
    target: Mesh,
    T12: np.ndarray,
    source_name: str = "source",
    target_name: str = "target",
):
    """Show the source mesh coloured by its correspondence to the target.

    The target's XYZ position colour is transferred to the source through
    the point-to-point map *T12*, making it easy to visually assess map quality.
    """
    target_colors = xyz_color(target)

    # Source mesh coloured by the transferred target colours
    mapped_colors = target_colors[T12]
    show_mesh(source_name, source, color=mapped_colors)
    show_mesh(target_name, target, color=target_colors)


def visualise_error(
    source: Mesh,
    target: Mesh,
    T12: np.ndarray,
    name: str = "error",
):
    """Show per-vertex mapping error as a scalar field on the source."""
    diff = source.vertices - target.vertices[T12]
    err = np.linalg.norm(diff, axis=1)
    sm = ps.register_surface_mesh(name, source.vertices, source.faces)
    sm.add_scalar_quantity("position_error", err, enabled=True, cmap="reds")
    return sm


def visualise_eigenfunctions(mesh: Mesh, name: str = "mesh", n: int = 4):
    """Show the first *n* Laplacian eigenfunctions on the mesh."""
    sm = ps.register_surface_mesh(name, mesh.vertices, mesh.faces)
    for i in range(min(n, mesh.evecs.shape[1])):
        sm.add_scalar_quantity(
            f"phi_{i}", mesh.evecs[:, i], enabled=(i == 1), cmap="coolwarm"
        )
    return sm
