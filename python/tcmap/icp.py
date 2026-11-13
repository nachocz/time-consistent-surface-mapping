"""Rigid iterative closest point, for the first map (Sec. 3.1)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def kabsch(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimizing sum ||R x_m + t - y_m||^2 (paired rows)."""
    cx, cy = X.mean(axis=0), Y.mean(axis=0)
    U, _, Vt = np.linalg.svd((X - cx).T @ (Y - cy))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cy - R @ cx


def icp(X: np.ndarray, Y: np.ndarray, iterations: int = 40, tol: float = 1e-7):
    """Align X to Y, starting from the alignment of the centroids.

    Returns (R, t, X_aligned, pairs): pairs[m] is the nearest point of Y to the aligned
    point m, the point map Pi of eq. (10).
    """
    tree = cKDTree(Y)
    R, t = np.eye(3), Y.mean(axis=0) - X.mean(axis=0)
    prev = np.inf
    for _ in range(iterations):
        Xa = X @ R.T + t
        dist, pairs = tree.query(Xa)
        rms = float(np.sqrt(np.mean(dist ** 2)))
        if prev - rms < tol * max(1.0, rms):
            break
        prev = rms
        R, t = kabsch(X, Y[pairs])
    Xa = X @ R.T + t
    return R, t, Xa, tree.query(Xa)[1].astype(np.int64)
