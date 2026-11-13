"""Time-consistent surface mapping for deformable object shape control.

Python implementation of the method of

    I. Cuiral-Zueco and G. López-Nicolás, "Time Consistent Surface Mapping for Deformable
    Object Shape Control," IEEE Transactions on Automation Science and Engineering,
    vol. 22, pp. 11099-11111, 2025, doi: 10.1109/TASE.2025.3529180.

Modules: mesh (Laplace-Beltrami eigenbasis), fmap (functional maps, ZoomOut, rule (19)),
icp (first map), mapping (Algorithm 1 and the per-frame ZoomOut baseline), control (the
shape control law of Sec. 4.1), data (the recorded experiments).
"""

from .control import ControlOptions, SequentialShapeController
from .data import EXPERIMENTS, load_experiment
from .fmap import fmap_from_map, map_from_fmap, target_size, zoomout
from .mapping import Frame, Options, TimeConsistentMapper, classic_zoomout, ntc_step
from .mesh import Basis, Mesh, eigenbasis

__all__ = [
    "Basis", "ControlOptions", "EXPERIMENTS", "Frame", "Mesh", "Options", "SequentialShapeController",
    "TimeConsistentMapper", "classic_zoomout", "eigenbasis", "fmap_from_map", "load_experiment",
    "map_from_fmap", "ntc_step", "target_size", "zoomout",
]
