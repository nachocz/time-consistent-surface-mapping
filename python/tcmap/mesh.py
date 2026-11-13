"""Triangle meshes and their Laplace-Beltrami eigenbasis (Sec. 2.1 of the paper).

The basis of a mesh holds the eigenvectors of the cotangent Laplacian L with the lumped
area matrix A (a third of the area of the triangles around each vertex), with the
smallest eigenvalues first:

    L phi = lambda A phi,        Phi^T A Phi = I,

so that the pseudo-inverse of Phi is Phi^T A, as in (9) and (17). As in ZoomOut and in
the reference MATLAB code, the constant eigenvector (eigenvalue 0) is the first column.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


@dataclass
class Mesh:
    """A triangle mesh: vertices (n, 3) and faces (f, 3), 0-based."""

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64)

    @property
    def n(self) -> int:
        return len(self.vertices)


@dataclass
class Basis:
    """Eigenbasis of a mesh: evecs (n, K), evals (K), lumped areas (n), total area."""

    mesh: Mesh
    evecs: np.ndarray
    evals: np.ndarray
    area: np.ndarray
    total: float = field(init=False)

    def __post_init__(self) -> None:
        self.total = float(self.area.sum())

    @property
    def n(self) -> int:
        return self.evecs.shape[0]

    @property
    def K(self) -> int:
        return self.evecs.shape[1]

    @property
    def vertices(self) -> np.ndarray:
        return self.mesh.vertices


def cotangent_laplacian(V: np.ndarray, F: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    """Cotangent Laplacian (positive semi-definite) and lumped vertex areas."""
    n = len(V)
    P = [V[F[:, c]] for c in range(3)]
    cross = np.cross(P[1] - P[0], P[2] - P[0])
    twice = np.linalg.norm(cross, axis=1)                 # twice the triangle areas
    rows, cols, vals = [], [], []
    for c in range(3):                                     # angle at corner c, opposite edge (a, b)
        a, b = (c + 1) % 3, (c + 2) % 3
        u, v = P[a] - P[c], P[b] - P[c]
        cot = np.einsum("ij,ij->i", u, v) / np.maximum(twice, 1e-300)
        w = 0.5 * np.clip(cot, -1e3, 1e3) * (twice > 1e-14)
        rows += [F[:, a], F[:, b]]
        cols += [F[:, b], F[:, a]]
        vals += [-w, -w]
    off = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    L = (off - sp.diags(np.asarray(off.sum(axis=1)).ravel())).tocsr()   # L_ii = sum_j w_ij, L_ij = -w_ij
    area = np.zeros(n)
    for c in range(3):
        np.add.at(area, F[:, c], twice / 6.0)
    return L, area


def eigenbasis(mesh: Mesh, K: int, seed: int | None = None) -> Basis:
    """The K eigenpairs of L phi = lambda A phi with the smallest eigenvalues.

    The signs of the eigenvectors are arbitrary (they depend on the start vector of the
    solver), as with any eigensolver: nothing fixes the sign of phi_i from one mesh to
    the next. `seed` sets the start vector, for reproducible runs.
    """
    L, area = cotangent_laplacian(mesh.vertices, mesh.faces)
    K = min(K, mesh.n - 1)
    A = sp.diags(area)
    v0 = np.random.default_rng(seed).standard_normal(mesh.n) if seed is not None else None
    sigma = -1.0 / max(area.sum(), 1e-12)                 # shift below the spectrum
    evals, evecs = eigsh(L.tocsc(), k=K, M=A.tocsc(), sigma=sigma, which="LM", v0=v0)
    order = np.argsort(evals)
    evals, evecs = np.maximum(evals[order], 0.0), evecs[:, order]
    return Basis(mesh, evecs, evals, area)


def lumped_area(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """A third of the area of the triangles around each vertex."""
    twice = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    area = np.zeros(len(V))
    for c in range(3):
        np.add.at(area, F[:, c], twice / 6.0)
    return area
