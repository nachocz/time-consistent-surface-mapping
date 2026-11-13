"""Time-consistent and non-isometry robust surface mapping (Sec. 3, Algorithm 1), and
the non-time-consistent baseline (ZoomOut computed anew at every frame, Sec. 4.2).

At every frame k the method receives a new mesh M_ret(k) (points X_ret(k), in any order
and number) and returns the tracked points X(k) (the same M object points at every
frame, eq. 15), the point map Pi(k) from the tracked points to the target (16), and the
functional map C(I_max, k), from which the shape error E(k) = X(k) - Pi(k) Y (20) is
defined.

Tracking of the basis and of the points (Sec. 3.2):

``anchored`` (default)
    As in the reference MATLAB code (previousstatefmap.m): the map between consecutive
    meshes is initialized by nearest neighbors and refined coarse-to-fine as in
    (11)-(12); the basis of the new mesh is aligned with the whole map C^M(k), which
    corrects the signs (13) and also rotations inside eigenspaces of nearly equal
    eigenvalues; and each tracked point (15) is the new point whose aligned embedding is
    nearest to the embedding of that point at k = 1.
``literal``
    (13)-(15) as written: the basis Pi^M(k) Phi_ret(k) C^M_sgn(k) and the points
    Pi^M(k) X_ret(k) on the rows that Pi^M(k) selects. Pi^M(k) is not one-to-one, so
    tracked points that select the same new point keep the same embedding from then on
    and merge: the tracking degrades over time.
``none``
    no tracking: the basis of every new mesh is used as it comes (for comparison).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .fmap import block, fmap_from_map, map_from_fmap, nearest, target_size, zoomout
from .icp import icp
from .mesh import Basis, Mesh, eigenbasis, lumped_area


@dataclass
class Options:
    I_initial: int = 4        # size of C(I, k = 1) before the refinement
    I_update: int = 5         # re-used block of C(I_max, k - 1) (Sec. 3.3)
    I_max: int = 30           # largest basis size
    step: int = 4             # increment of I (1 in Algorithm 1, 4 in the reference code)
    track_from: int = 15      # first I of the tracking refinement (11)-(12) (as in the code)
    tracking: str = "anchored"   # 'anchored' | 'literal' | 'none'
    warm_start: bool = True   # initialize C(I_update, k) from C(I_max, k - 1)
    slanted: bool = True      # non-isometry rule (18)-(19)
    init: str = "icp"         # first map (Sec. 3.1): 'icp' | 'identity'


@dataclass
class Frame:
    """The result of Algorithm 1 at one frame."""

    k: int
    X: np.ndarray             # tracked points X(k), (M, 3)
    Pi: np.ndarray            # tracked point m -> target vertex Pi[m]
    C: np.ndarray             # C(I_max, k), (I_max, J)
    rho: float
    E: np.ndarray             # E(k) = X(k) - Pi(k) Y, (M, 3)
    retrieved: Basis          # basis of the mesh of this frame
    rows: np.ndarray | None   # rows of the mesh of this frame that are the tracked points
    CM: np.ndarray | None = None      # tracking map C^M(k)
    seconds: dict = field(default_factory=dict)

    @property
    def error(self) -> float:
        """||e|| = sqrt(sum_m ||e_m||^2), eq. (21)."""
        return float(np.sqrt(np.sum(self.E ** 2)))


def shape_error(X: np.ndarray, Y: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """E = X - Pi Y, eq. (20)."""
    return X - Y[Pi]


def classic_zoomout(src: Basis, tgt: Basis, I_initial: int = 4, I_max: int = 30, step: int = 1):
    """ZoomOut (Melzi et al., 2019) from C(I_initial) = identity with square maps."""
    return zoomout(src, tgt, np.eye(I_initial), I_initial, I_initial, min(I_max, src.K), step)


def initial_map(src: Basis, tgt: Basis, opt: Options):
    """First map (Sec. 3.1): ICP between X and Y, nearest neighbors, and eq. (10)."""
    rho = np.sqrt(src.total / tgt.total) if opt.slanted else None
    I0 = opt.I_initial
    if opt.init == "identity":
        return zoomout(src, tgt, np.eye(I0), I0, I0, opt.I_max, opt.step, rho), rho
    _, _, _, pairs = icp(src.vertices, tgt.vertices)
    J0 = target_size(src, tgt, I0, rho)
    return zoomout(src, tgt, fmap_from_map(src, I0, tgt, J0, pairs), I0, J0, opt.I_max, opt.step, rho), rho


class TimeConsistentMapper:
    """Algorithm 1: call step() with the mesh of every frame."""

    def __init__(self, target: Mesh, options: Options | None = None, seed: int = 1):
        self.opt = options or Options()
        self.target = eigenbasis(target, max(2 * self.opt.I_max, 40), seed=seed)
        self.seed = seed + 100
        self.k = 0
        self.state = None

    def step(self, mesh: Mesh) -> Frame:
        o, tgt = self.opt, self.target
        self.k += 1
        t0 = time.perf_counter()
        ret = eigenbasis(mesh, o.I_max, seed=self.seed + self.k)
        t1 = time.perf_counter()
        rows, CM = None, None
        if self.k == 1:
            # X(1) = X_ret(1): the first retrieved points are the points to track
            src, X = ret, ret.vertices
            self.ref = ret
        elif o.tracking == "none":
            src, X = ret, ret.vertices
        elif o.tracking == "anchored":
            src, rows, CM = self._track_anchored(ret)
            X = ret.vertices[rows]
        else:
            src, CM = self._track_literal(ret)
            X = src.vertices
        t2 = time.perf_counter()
        rho = np.sqrt(src.total / tgt.total) if o.slanted else None
        if self.k == 1 or not o.warm_start:
            res, rho = initial_map(src, tgt, o)
        else:
            Cp = self.state["C"]
            Iu = min(o.I_update, *Cp.shape, src.K)
            res = zoomout(src, tgt, block(Cp, Iu, Iu), Iu, Iu, o.I_max, o.step, rho)
        t3 = time.perf_counter()
        Pi = res.Pi[rows] if rows is not None else res.Pi
        self.state = {"C": res.C, "src": src, "X": X}
        return Frame(self.k, X, Pi, res.C, rho or 1.0, shape_error(X, tgt.vertices, Pi), ret, rows, CM,
                     {"basis": t1 - t0, "tracking": t2 - t1, "mapping": t3 - t2, "total": t3 - t0})

    # Sec. 3.2 as in the reference code
    def _track_anchored(self, ret: Basis):
        o, prev = self.opt, self.state["src"]
        K = min(o.I_max, ret.K, prev.K)
        I0 = min(o.track_from, K)
        c = lambda V: V - V.mean(axis=0)
        T0 = nearest(c(ret.vertices), c(prev.vertices))           # new point -> previous point
        r = zoomout(ret, prev, fmap_from_map(ret, I0, prev, I0, T0), I0, I0, K, o.step)
        CM = fmap_from_map(ret, K, prev, K, r.Pi)                   # Phi_ret^T A_ret T Phi_prev
        aligned = Basis(ret.mesh, ret.evecs[:, :K] @ CM, ret.evals[:K], ret.area)
        k_ref = min(K, self.ref.K)
        rows = nearest(self.ref.evecs[:, :k_ref], aligned.evecs[:, :k_ref])   # embedding at k = 1
        return aligned, rows, CM

    # Sec. 3.2 as written, (11)-(15)
    def _track_literal(self, ret: Basis):
        o, prev = self.opt, self.state["src"]
        K = min(o.I_max, ret.K, prev.K)
        I0 = min(o.track_from, K)
        c = lambda V: V - V.mean(axis=0)
        PiM = nearest(c(self.state["X"]), c(ret.vertices))         # tracked point -> new point
        r = zoomout(prev, ret, fmap_from_map(prev, I0, ret, I0, PiM), I0, I0, K, o.step)
        CM = fmap_from_map(prev, K, ret, K, r.Pi)                   # (12) at I_max
        sgn = np.where(np.diag(CM) < 0, -1.0, 1.0)                  # (13)
        X = ret.vertices[r.Pi]                                      # (15)
        faces = self.ref.mesh.faces
        basis = Basis(Mesh(X, faces), ret.evecs[r.Pi, :K] * sgn, ret.evals[:K], lumped_area(X, faces))   # (14)
        return basis, CM


def ntc_step(ret: Basis, target: Basis, opt: Options):
    """ZoomOut from scratch on the mesh of one frame: Pi_NTC(k) and E_NTC(k)."""
    t0 = time.perf_counter()
    r = classic_zoomout(ret, target, opt.I_initial, opt.I_max, opt.step)
    return r.Pi, shape_error(ret.vertices, target.vertices, r.Pi), time.perf_counter() - t0
