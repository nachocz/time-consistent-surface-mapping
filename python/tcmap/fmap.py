"""Functional map computation: ZoomOut refinement with non-isometry support."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .mesh import Mesh


# ---------------------------------------------------------------------------
# Core nearest-neighbour search in eigenspace
# ---------------------------------------------------------------------------

def _knnsearch(target: np.ndarray, query: np.ndarray) -> np.ndarray:
    """For each row in *query*, find the index of the closest row in *target*.

    Parameters
    ----------
    target : (N, d)
    query  : (M, d)

    Returns
    -------
    indices : (M,) int — index into *target* for each query point
    """
    tree = cKDTree(target)
    _, idx = tree.query(query, k=1)
    return idx


# ---------------------------------------------------------------------------
# ZoomOut-style refinement (Algorithm 1 core loop)
# ---------------------------------------------------------------------------

def refine_nonisometric(
    T12: np.ndarray,
    S1: Mesh,
    S2: Mesh,
    k_init: int,
    k_step: int,
    k_final: int,
    use_slanted_diagonal: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Refine a point-to-point map via ZoomOut with optional non-isometry handling.

    Mirrors ``refinenonisometric.m``.

    Parameters
    ----------
    T12 : (M,) int — initial map: S1 vertex i  →  S2 vertex T12[i]
    S1, S2 : Mesh — source / target with precomputed basis
    k_init, k_step, k_final : int — refinement schedule
    use_slanted_diagonal : bool — enable area-ratio compensation (Eq. 19)

    Returns
    -------
    T12 : (M,) int — refined point-to-point map
    C21 : (k, r) array — final functional map matrix
    area_ratio : float — sqrt(area_S1 / area_S2)
    """
    evals1, evals2 = S1.evals, S2.evals
    B1_all, B2_all = S1.evecs, S2.evecs
    A_sparse = _area_sparse(S1)

    area_ratio = 1.0
    if use_slanted_diagonal:
        area_ratio = np.sqrt(np.sum(S1.area) / np.sum(S2.area))

    ks = list(range(k_init, k_final + 1, k_step))
    if ks[-1] != k_final:
        ks.append(k_final)

    C21 = None
    for k in ks:
        if use_slanted_diagonal:
            max_ev1 = evals1[k - 1] if k <= len(evals1) else evals1[-1]
            max_ev2 = max_ev1 * area_ratio
            r = int(np.sum(evals2 < max_ev2))
            r = max(r, 1)
        else:
            r = k

        B1 = B1_all[:, :k]
        B2 = B2_all[:, :r]

        # C21 = B1^T  A  B2[T12, :]          (k × r)
        C21 = B1.T @ (A_sparse @ B2[T12, :])

        # T12 = argmin_n ||B2[n,:] C21^T - B1[m,:]||   for each m
        T12 = _knnsearch(B2 @ C21.T, B1)

    return T12, C21, area_ratio


# ---------------------------------------------------------------------------
# Classic ZoomOut (identity initialisation, no time-consistency)
# ---------------------------------------------------------------------------

def classic_zoomout(
    S1: Mesh,
    S2: Mesh,
    n_initial: int = 4,
    n_max: int = 30,
    step: int = 4,
    use_slanted_diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard ZoomOut (Melzi et al., 2019) — for baseline comparison.

    Parameters
    ----------
    S1, S2 : Mesh — with precomputed basis of size >= n_max

    Returns
    -------
    T12 : (M,) int
    C21 : array
    """
    C21_ini = np.eye(n_initial)
    B1 = S1.evecs[:, :n_initial]
    B2 = S2.evecs[:, :n_initial]
    T12_ini = _knnsearch(B2 @ C21_ini.T, B1)

    T12, C21, _ = refine_nonisometric(
        T12_ini, S1, S2, n_initial, step, n_max,
        use_slanted_diagonal=use_slanted_diagonal
    )
    return T12, C21


# ---------------------------------------------------------------------------
# Initialise functional map (first frame — random probing)
# ---------------------------------------------------------------------------

def initialise_fmap(
    S1: Mesh,
    S2: Mesh,
    n_initial: int = 4,
    n_max: int = 30,
    step: int = 4,
    use_slanted_diagonal: bool = True,
    n_probes: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust initialisation by probing random diagonal matrices (``initialisefmap.m``).

    Returns
    -------
    T12 : (M,) int — point-to-point map
    C21 : array    — functional map
    """
    B1 = S1.evecs[:, :n_initial]
    B2 = S2.evecs[:, :n_initial]

    best_T12 = None
    best_residual = np.inf
    best_C21 = None
    rng = np.random.default_rng(42)

    for _ in range(n_probes):
        diag_vals = rng.uniform(-0.5, 0.5, size=n_initial)
        C21_ini = np.diag(diag_vals)
        T12_ini = _knnsearch(B2 @ C21_ini.T, B1)

        # Extrinsic residual (centred)
        V1c = S1.vertices - S1.vertices.mean(axis=0)
        V2c = S2.vertices[T12_ini] - S2.vertices[T12_ini].mean(axis=0)
        residual = np.sum(np.linalg.norm(V1c - V2c, axis=1))

        if residual < best_residual:
            best_residual = residual
            best_T12 = T12_ini
            best_C21 = C21_ini

    T12, C21, _ = refine_nonisometric(
        best_T12, S1, S2, n_initial, step, n_max,
        use_slanted_diagonal=use_slanted_diagonal,
    )
    return T12, C21


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _area_sparse(S: Mesh):
    """Return the diagonal area matrix as a sparse matrix."""
    import scipy.sparse as sp
    return sp.diags(S.area, 0, format="csc")
