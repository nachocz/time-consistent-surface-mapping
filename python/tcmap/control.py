"""Shape control law (Sec. 4.1).

The grippers act one at a time, so that the change of the tracked points
Delta x(k) = x(k) - x(k-1) is the response to the action u_g = (Delta t_g, Delta r_g) of a
single gripper (translation, Rodrigues vector). Each gripper keeps its last B actions
U_g (6 x B) and responses Delta X_g (3M x B), from which

    L_g = Delta X_g U_g^T (U_g U_g^T + gamma I_6)^(-1)            (23)
    u_g = -alpha L_g^+ e                                           (24)

Equation (23) is printed with -gamma I_6; as a Tikhonov regularization it adds gamma I_6,
as here. In U_g the rotations are multiplied by a lever (cm per rad) so that a single
gamma weights translations and rotations alike; (24) does not depend on that scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ControlOptions:
    alpha: float = 0.3             # gain of (24)
    B: int = 12                    # buffer size (B >= 6)
    gamma: float = 0.2             # Tikhonov term of (23), in units of the scaled actions squared
    lever: float = 10.0            # rotations in U_g as the displacement at this distance
    max_translation: float = 0.5   # step limits of the robots
    max_rotation: float = np.radians(4.0)


class SequentialShapeController:
    def __init__(self, G: int, options: ControlOptions | None = None):
        self.opt = options or ControlOptions()
        self.G = G
        self.U = [[] for _ in range(G)]
        self.dX = [[] for _ in range(G)]

    def _scale(self) -> np.ndarray:
        return np.array([1, 1, 1, self.opt.lever, self.opt.lever, self.opt.lever], dtype=float)

    def record(self, g: int, u: np.ndarray, dx: np.ndarray) -> None:
        """Store the action u of gripper g and the change dx (3M) of the tracked points."""
        self.U[g].insert(0, np.asarray(u, float) * self._scale())
        self.dX[g].insert(0, np.asarray(dx, float).ravel())
        del self.U[g][self.opt.B:], self.dX[g][self.opt.B:]

    def estimate(self, g: int) -> np.ndarray | None:
        """L_g of (23) (3M x 6, scaled actions), or None with fewer than 6 entries."""
        if len(self.U[g]) < 6:
            return None
        U, dX = np.array(self.U[g]).T, np.array(self.dX[g]).T
        return dX @ U.T @ np.linalg.inv(U @ U.T + self.opt.gamma * np.eye(6))

    def action(self, g: int, e: np.ndarray) -> np.ndarray:
        """u_g = -alpha L_g^+ e (24), within the step limits."""
        L = self.estimate(g)
        if L is None:
            return np.zeros(6)
        u = -self.opt.alpha * np.linalg.lstsq(L, np.asarray(e, float).ravel(), rcond=None)[0] / self._scale()
        nt, nr = np.linalg.norm(u[:3]), np.linalg.norm(u[3:])
        if nt > self.opt.max_translation:
            u[:3] *= self.opt.max_translation / nt
        if nr > self.opt.max_rotation:
            u[3:] *= self.opt.max_rotation / nr
        return u
