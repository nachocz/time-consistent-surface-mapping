"""Mesh processing: cotangent Laplacian, area weights, and Laplacian eigenbasis."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# Cotangent Laplacian & area weights
# ---------------------------------------------------------------------------

def cotangent_laplacian(vertices: np.ndarray, faces: np.ndarray):
    """Compute the cotangent Laplacian and lumped area weights.

    Parameters
    ----------
    vertices : (M, 3) array
    faces    : (F, 3) int array — triangle indices (0-based)

    Returns
    -------
    W : (M, M) sparse — positive semi-definite cotangent weight matrix
    A : (M,)  array   — per-vertex lumped (Voronoi-mixed) area
    """
    nv = vertices.shape[0]
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]

    v0 = vertices[f0]
    v1 = vertices[f1]
    v2 = vertices[f2]

    # Edge vectors
    e01 = v1 - v0
    e12 = v2 - v1
    e20 = v0 - v2

    # Cotangent of each angle via cross / dot
    def _cot_angle(a, b):
        cross_norm = np.linalg.norm(np.cross(a, b), axis=1)
        dot = np.sum(a * b, axis=1)
        # clamp to avoid division by zero
        cross_norm = np.maximum(cross_norm, 1e-12)
        return dot / cross_norm

    cot0 = _cot_angle(-e01, e20)   # angle at vertex 0
    cot1 = _cot_angle(-e12, e01)   # angle at vertex 1
    cot2 = _cot_angle(-e20, e12)   # angle at vertex 2

    # Build symmetric weight matrix (0.5 * cot)
    ii = np.concatenate([f1, f2, f0, f2, f0, f1])
    jj = np.concatenate([f2, f0, f1, f1, f2, f0])
    vals = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])

    W_off = sp.coo_matrix((vals, (ii, jj)), shape=(nv, nv)).tocsc()
    # Make exactly symmetric
    W_off = (W_off + W_off.T) / 2.0
    # Diagonal = negative row sums  →  L = D - W_off  →  we store L directly
    diag_vals = np.array(W_off.sum(axis=1)).ravel()
    W = sp.diags(diag_vals, 0, format="csc") - W_off

    # Lumped area: 1/3 of adjacent triangle areas
    tri_areas = 0.5 * np.linalg.norm(np.cross(e01, -e20), axis=1)
    A = np.zeros(nv)
    np.add.at(A, f0, tri_areas / 3.0)
    np.add.at(A, f1, tri_areas / 3.0)
    np.add.at(A, f2, tri_areas / 3.0)

    return W, A


# ---------------------------------------------------------------------------
# Laplacian eigenbasis
# ---------------------------------------------------------------------------

def compute_laplacian_basis(vertices: np.ndarray, faces: np.ndarray, num_eigs: int = 30):
    """Compute the first *num_eigs* eigenfunctions of the cotangent Laplacian.

    Parameters
    ----------
    vertices : (M, 3) array
    faces    : (F, 3) int array
    num_eigs : int

    Returns
    -------
    evecs : (M, num_eigs) — eigenvectors sorted by ascending eigenvalue
    evals : (num_eigs,)   — corresponding eigenvalues
    W     : sparse         — cotangent Laplacian
    A_diag : (M,)          — per-vertex area
    """
    W, A_diag = cotangent_laplacian(vertices, faces)
    A_sparse = sp.diags(A_diag, 0, format="csc")

    try:
        evals, evecs = eigsh(W, k=num_eigs, M=A_sparse, sigma=1e-6, which="LM")
    except Exception:
        # fallback: shift Laplacian to guarantee positive-definiteness
        evals, evecs = eigsh(W - 1e-8 * sp.eye(vertices.shape[0]),
                             k=num_eigs, M=A_sparse, which="SM")

    # Sort ascending
    order = np.argsort(np.abs(evals))
    evals = np.abs(evals[order])
    evecs = evecs[:, order]

    return evecs, evals, W, A_diag


# ---------------------------------------------------------------------------
# Mesh data container
# ---------------------------------------------------------------------------

class Mesh:
    """Lightweight container mirroring the MATLAB mesh struct."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.nv = self.vertices.shape[0]
        self.nf = self.faces.shape[0]
        # Laplacian basis (populated by compute_basis)
        self.evecs: np.ndarray | None = None
        self.evals: np.ndarray | None = None
        self.W = None
        self.area: np.ndarray | None = None

    def compute_basis(self, num_eigs: int = 30):
        """Compute and store the Laplacian eigenbasis."""
        self.evecs, self.evals, self.W, self.area = compute_laplacian_basis(
            self.vertices, self.faces, num_eigs
        )
        return self

    @property
    def total_area(self) -> float:
        if self.area is None:
            _, self.area = cotangent_laplacian(self.vertices, self.faces)
        return float(np.sum(self.area))

    def copy(self) -> "Mesh":
        m = Mesh(self.vertices.copy(), self.faces.copy())
        if self.evecs is not None:
            m.evecs = self.evecs.copy()
            m.evals = self.evals.copy()
            m.area = self.area.copy()
            m.W = self.W.copy()
        return m
