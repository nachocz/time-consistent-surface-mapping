"""Colors of the surface maps, as in the figures of the paper: the target carries a smooth
coloring of its points (here from their coordinates along its two principal axes), and
every point of the current shape takes the color of its image Pi[m]."""

from __future__ import annotations

import colorsys

import numpy as np


def reference_colors(V: np.ndarray) -> np.ndarray:
    """RGB colors (n, 3) in [0, 1] of the target points."""
    X = V - V.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axes = Vt[:2] * np.sign(Vt[:2, [0, 1]].diagonal() + 1e-12)[:, None]   # fixed orientation
    s = X @ axes.T
    s = (s - s.min(axis=0)) / np.maximum(np.ptp(s, axis=0), 1e-12)
    return np.array([colorsys.hls_to_rgb((0.98 + 0.78 * u) % 1.0, 0.36 + 0.3 * v, 0.7) for u, v in s])


def pull_colors(reference: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """Colors of the source points through the map Pi."""
    return reference[Pi]
