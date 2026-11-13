"""Functional maps and their coarse-to-fine refinement (Secs. 2.2, 3.3 and 3.4).

A point map Pi: M -> N is an integer array of length M (Pi[m] = n means Pi_{m,n} = 1,
eq. 5). A functional map C(I) has I rows (source eigenvectors) and J columns (target
eigenvectors).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .mesh import Basis


def nearest(query: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Index of the nearest row of `data` for every row of `query`."""
    return cKDTree(data).query(query, k=1)[1].astype(np.int64)


def fmap_from_map(src: Basis, I: int, tgt: Basis, J: int, Pi: np.ndarray) -> np.ndarray:
    """C(I, J) = (Phi^M_I)^T A^M Pi Phi^N_J, eqs. (9), (10), (12), (17)."""
    return src.evecs[:, :I].T @ (src.area[:, None] * tgt.evecs[Pi, :J])


def map_from_fmap(src: Basis, I: int, tgt: Basis, J: int, C: np.ndarray) -> np.ndarray:
    """Pi = argmin || Pi Phi^N_J C^T - Phi^M_I ||_F, eqs. (8), (11), (16)."""
    return nearest(src.evecs[:, :I], tgt.evecs[:, :J] @ C.T)


def target_size(src: Basis, tgt: Basis, I: int, rho: float | None) -> int:
    """J = max{ r | lambda^N_r <= lambda^M_I rho }, eq. (19); J = I without rho (ZoomOut)."""
    if rho is None:
        return min(I, tgt.K)
    return max(1, int(np.sum(tgt.evals <= src.evals[I - 1] * rho)))


@dataclass
class Refinement:
    Pi: np.ndarray
    C: np.ndarray
    I: int
    J: int
    trace: list = field(default_factory=list)


def zoomout(src: Basis, tgt: Basis, C0: np.ndarray, I0: int, J0: int, Imax: int,
            step: int = 1, rho: float | None = None, keep_trace: bool = False) -> Refinement:
    """Alternate the point map (16) and the larger functional map (17) up to I = Imax.

    `step` is the increment of I (1 in Algorithm 1 of the paper, 4 in the reference
    code); `rho` enables the non-isometry rule (19). The last point map is computed from
    C(Imax), so that Pi and C correspond.
    """
    I, J, C, trace = I0, J0, C0, []
    while True:
        Pi = map_from_fmap(src, I, tgt, J, C)
        if keep_trace:
            trace.append((I, J, C, Pi))
        if I >= Imax:
            return Refinement(Pi, C, I, J, trace)
        I = min(I + step, Imax)
        J = target_size(src, tgt, I, rho)
        C = fmap_from_map(src, I, tgt, J, Pi)


def block(C: np.ndarray, I: int, J: int) -> np.ndarray:
    """Top-left I x J block of C (zero-padded if C is smaller)."""
    out = np.zeros((I, J))
    r, c = min(I, C.shape[0]), min(J, C.shape[1])
    out[:r, :c] = C[:r, :c]
    return out
