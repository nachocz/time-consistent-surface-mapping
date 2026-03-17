"""Time-consistent surface mapping pipeline (Algorithm 1 of the paper).

Combines time-consistent basis tracking, previous-state initialisation,
and non-isometry-robust ZoomOut refinement.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .fmap import (
    _knnsearch,
    _area_sparse,
    initialise_fmap,
    refine_nonisometric,
)


# ---------------------------------------------------------------------------
# Previous-state functional map  (Section 3.2 — basis tracking)
# ---------------------------------------------------------------------------

def previous_state_fmap(
    S1: Mesh,
    S1_prev: Mesh,
    n_max: int = 30,
    step: int = 4,
    use_slanted_diagonal: bool = True,
) -> tuple[np.ndarray, np.ndarray, Mesh]:
    """Map between consecutive frames (``previousstatefmap.m``).

    Computes a mapping from the newly acquired mesh (*S1*) to the
    previous tracked mesh (*S1_prev*), then updates the eigenbasis of
    *S1* for time-consistency (sign correction via C_sgn).

    Parameters
    ----------
    S1       : Mesh — current-frame mesh (basis must be precomputed)
    S1_prev  : Mesh — previous-frame mesh (with time-consistent basis)
    n_max    : int  — maximum basis size
    step     : int  — refinement step

    Returns
    -------
    T12      : (M_new,) int — map from current vertices to previous vertices
    C212     : array         — inter-frame functional map
    S1       : Mesh          — *S1* with updated (time-consistent) eigenbasis
    """
    # Initial map from centred extrinsic positions (nearest neighbour)
    V1c = S1.vertices - S1.vertices.mean(axis=0)
    V1pc = S1_prev.vertices - S1_prev.vertices.mean(axis=0)
    T12_ini = _knnsearch(V1pc, V1c)

    # Refine — start from a high k (k=15) because consecutive frames are close
    k_init = min(15, n_max)
    T12, C212, _ = refine_nonisometric(
        T12_ini, S1, S1_prev, k_init, step, n_max,
        use_slanted_diagonal=use_slanted_diagonal,
    )

    # Update eigenvectors for time-consistency: Φ_new ← Φ_new · C212
    # C212 is (k, r) — may be non-square due to slanted diagonal
    # MATLAB: S1.evecs(:,1:size(S1.evecs*C212,2)) = S1.evecs*C212
    n_rows = C212.shape[0]
    updated = S1.evecs[:, :n_rows] @ C212
    n_out = updated.shape[1]
    S1.evecs[:, :n_out] = updated

    return T12, C212, S1


# ---------------------------------------------------------------------------
# Update functional map (Section 3.3 — reuse previous solution)
# ---------------------------------------------------------------------------

def update_fmap(
    S1: Mesh,
    S2: Mesh,
    n_update: int,
    n_max: int,
    C21_prev: np.ndarray,
    step: int = 4,
    use_slanted_diagonal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Warm-start refinement from a previous functional map (``updatefmap.m``).

    Parameters
    ----------
    S1, S2     : Mesh — source / target (basis precomputed)
    n_update   : int  — sub-matrix size to reuse (I_update)
    n_max      : int  — maximum basis size
    C21_prev   : array — previous functional map matrix
    step       : int
    use_slanted_diagonal : bool

    Returns
    -------
    T12 : (M,) int
    C21 : array
    """
    # Clamp n_update to available dimensions
    n_update = min(n_update, *C21_prev.shape)

    C21_ini = C21_prev[:n_update, :n_update]
    B1 = S1.evecs[:, :C21_ini.shape[0]]
    B2 = S2.evecs[:, :C21_ini.shape[1]]
    T12_ini = _knnsearch(B2 @ C21_ini.T, B1)

    T12, C21, _ = refine_nonisometric(
        T12_ini, S1, S2, n_update, step, n_max,
        use_slanted_diagonal=use_slanted_diagonal,
    )
    return T12, C21


# ---------------------------------------------------------------------------
# Full time-consistent pipeline (stateful)
# ---------------------------------------------------------------------------

class TimeConsistentMapper:
    """Stateful mapper that processes a sequence of deforming meshes.

    Usage::

        mapper = TimeConsistentMapper(target_mesh)
        for frame_mesh in sequence:
            T12, error = mapper.step(frame_mesh)
    """

    def __init__(
        self,
        target: Mesh,
        n_initial: int = 4,
        n_update: int = 5,
        n_max: int = 30,
        step: int = 4,
        use_slanted_diagonal: bool = True,
        n_probes: int = 2000,
    ):
        self.target = target
        self.n_initial = n_initial
        self.n_update = n_update
        self.n_max = n_max
        self.refine_step = step
        self.use_slanted = use_slanted_diagonal
        self.n_probes = n_probes

        # State
        self._iteration = 0
        self._C21_prev: np.ndarray | None = None
        self._S1_prev: Mesh | None = None
        self._S0: Mesh | None = None  # first-frame mesh (basis reference)

    def step(self, source: Mesh) -> tuple[np.ndarray, np.ndarray, float]:
        """Process one frame.

        Parameters
        ----------
        source : Mesh — new point cloud mesh for this frame.
                  Basis will be computed internally if not already present.

        Returns
        -------
        T12       : (M,) int   — point-to-point map (source → target)
        T12_cons  : (M0,) int  — time-consistent map (first-frame indexed)
        error     : float      — sum-of-squared shape error
        """
        S1 = source
        S2 = self.target

        # Ensure bases are computed
        if S1.evecs is None:
            S1.compute_basis(self.n_max)
        if S2.evecs is None:
            S2.compute_basis(self.n_max)

        self._iteration += 1

        if self._iteration == 1:
            # --- First frame: random-probing initialisation ---
            T12, C21 = initialise_fmap(
                S1, S2,
                n_initial=self.n_initial,
                n_max=self.n_max,
                step=self.refine_step,
                use_slanted_diagonal=self.use_slanted,
                n_probes=self.n_probes,
            )
            self._C21_prev = C21
            self._S1_prev = S1
            self._S0 = S1.copy()

            error = _shape_error(S1, S2, T12)
            return T12, T12, error

        else:
            # --- Subsequent frames ---
            # 1. Time-consistent basis update
            T12_self, C212, S1 = previous_state_fmap(
                S1, self._S1_prev,
                n_max=self.n_max,
                step=self.refine_step,
                use_slanted_diagonal=self.use_slanted,
            )

            # Map first-frame basis to current frame (tracking)
            ncols = min(S1.evecs.shape[1], self._S0.evecs.shape[1])
            T0_to_cur = _knnsearch(S1.evecs[:, :ncols], self._S0.evecs[:, :ncols])

            # 2. Warm-start refinement using previous functional map
            T12, C21 = update_fmap(
                S1, S2,
                n_update=self.n_update,
                n_max=self.n_max,
                C21_prev=self._C21_prev,
                step=self.refine_step,
                use_slanted_diagonal=self.use_slanted,
            )

            # Time-consistent map: compose tracking and current map
            T12_cons = T12[T0_to_cur]

            # Update state
            self._C21_prev = C21
            self._S1_prev = S1

            error = _shape_error(S1, S2, T12)
            return T12, T12_cons, error


# ---------------------------------------------------------------------------
# Shape error
# ---------------------------------------------------------------------------

def _shape_error(S1: Mesh, S2: Mesh, T12: np.ndarray) -> float:
    """Compute ||X - Pi*Y||^2 (sum of squared per-vertex position errors)."""
    diff = S1.vertices - S2.vertices[T12]
    return float(np.sum(np.linalg.norm(diff, axis=1) ** 2))
