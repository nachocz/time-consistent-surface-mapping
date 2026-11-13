#!/usr/bin/env python
"""Time-consistent surface mapping on the recorded experiments of the paper.

    python demo.py --experiment 5             3D viewer (Polyscope): play the sequence
    python demo.py --experiment 1 --no-gui    run the whole sequence, print the summary
    python demo.py --experiment 3 --no-gui --plot results.png

For every frame, the mesh perceived in the experiment is mapped to the target mesh by
the method of the paper (Algorithm 1) and by ZoomOut computed anew at every frame (the
non-time-consistent maps of Sec. 4.2). The viewer shows both maps as colors, the
target, and the two shape errors over time (as in Figs. 10-14 of the paper).

I. Cuiral-Zueco and G. López-Nicolás, "Time Consistent Surface Mapping for Deformable
Object Shape Control," IEEE Transactions on Automation Science and Engineering, 2025.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tcmap import EXPERIMENTS, Options, TimeConsistentMapper, load_experiment, ntc_step  # noqa: E402
from tcmap.colors import pull_colors, reference_colors  # noqa: E402


class Sequence:
    """The maps of an experiment, computed in order (each frame needs the previous one)."""

    def __init__(self, number: int, options: Options):
        self.frames, self.target_mesh = load_experiment(number)
        self.mapper = TimeConsistentMapper(self.target_mesh, options)
        self.ref = reference_colors(self.target_mesh.vertices)
        self.results = []          # per frame: (Frame, Pi_NTC, ||e_NTC||, seconds NTC)

    def __len__(self) -> int:
        return len(self.frames)

    def compute(self, upto: int) -> None:
        while len(self.results) <= min(upto, len(self.frames) - 1):
            out = self.mapper.step(self.frames[len(self.results)])
            Pi_ntc, E_ntc, seconds = ntc_step(out.retrieved, self.mapper.target, self.mapper.opt)
            self.results.append((out, Pi_ntc, float(np.sqrt((E_ntc ** 2).sum())), seconds))

    def errors(self):
        e = np.array([r[0].error for r in self.results])
        e_ntc = np.array([r[2] for r in self.results])
        return e, e_ntc


def summary(seq: Sequence, number: int) -> None:
    e, e_ntc = seq.errors()
    t = np.array([r[0].seconds["total"] for r in seq.results[1:]])
    t_ntc = np.array([r[3] for r in seq.results])
    sizes = [f.n for f in seq.frames]
    print(f"Experiment {number} ({EXPERIMENTS[number]}): {len(seq)} frames, {min(sizes)}-{max(sizes)} points per mesh")
    print(f"  time-consistent  ||e||: {e[0]:.3f} -> {e[-1]:.3f}, largest change between frames {np.abs(np.diff(e)).max():.3f}, "
          f"{1000 * t.mean():.0f} ms per frame ({1 / t.mean():.0f} Hz)")
    print(f"  ZoomOut / frame  ||e||: {e_ntc.min():.3f} to {e_ntc.max():.3f}, largest change between frames {np.abs(np.diff(e_ntc)).max():.3f}, "
          f"{1000 * t_ntc.mean():.0f} ms per frame")


def plot(seq: Sequence, number: int, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e, e_ntc = seq.errors()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(e_ntc, color="#1f5fd1", lw=1, ls="--", label=r"ZoomOut every frame, $\|\mathbf{e}_{NTC}\|$")
    ax.plot(e, color="#d62728", lw=2, label=r"time-consistent, $\|\mathbf{e}\|$")
    ax.set_xlabel("$k$"); ax.set_ylabel("shape error [m]"); ax.set_xlim(0, len(e) - 1); ax.set_ylim(bottom=0)
    ax.set_title(f"Experiment {number}: {EXPERIMENTS[number]}")
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  plot written to {path}")


def viewer(seq: Sequence, number: int) -> None:
    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    span = np.ptp(seq.target_mesh.vertices, axis=0).max()
    shift = np.array([1.4 * span, 0.0, 0.0])
    target = ps.register_surface_mesh("target N", seq.target_mesh.vertices + shift, seq.target_mesh.faces, edge_width=0.6)
    target.add_color_quantity("reference colors", seq.ref, enabled=True)
    state = {"k": 0, "play": True, "color": 0}

    def show(k: int) -> None:
        seq.compute(k)
        out, Pi_ntc, _, _ = seq.results[k]
        tracked = ps.register_surface_mesh("time-consistent: tracked mesh X(k)", out.X, seq.mapper.ref.mesh.faces, edge_width=0.6)
        perceived = ps.register_surface_mesh("ZoomOut every frame: mesh of frame k", out.retrieved.vertices - shift,
                                             out.retrieved.mesh.faces, edge_width=0.6)
        if state["color"] == 0:
            tracked.add_color_quantity("surface map", pull_colors(seq.ref, out.Pi), enabled=True)
            perceived.add_color_quantity("surface map", pull_colors(seq.ref, Pi_ntc), enabled=True)
        else:
            top = max(np.linalg.norm(out.E, axis=1).max(), 1e-9)
            tracked.add_scalar_quantity("||e_m||", np.linalg.norm(out.E, axis=1), enabled=True, vminmax=(0, top), cmap="viridis")
            e_ntc = np.linalg.norm(out.retrieved.vertices - seq.mapper.target.vertices[Pi_ntc], axis=1)
            perceived.add_scalar_quantity("||e_m||", e_ntc, enabled=True, vminmax=(0, top), cmap="viridis")

    def callback() -> None:
        k = state["k"]
        psim.TextUnformatted(f"Experiment {number}: {EXPERIMENTS[number]}")
        psim.TextUnformatted("left: ZoomOut every frame | middle: time-consistent | right: target")
        _, state["play"] = psim.Checkbox("play", state["play"])
        changed, k_new = psim.SliderInt("frame k", k, 0, len(seq) - 1)
        changed_c, state["color"] = psim.Combo("color", state["color"], ["surface map", "point error"])
        if state["play"] and not changed:
            k_new = (k + 1) % len(seq)
        if k_new != k or changed_c or not seq.results:
            state["k"] = k_new
            show(k_new)
        e, e_ntc = seq.errors()
        psim.PlotLines("||e||", e.astype(np.float32), graph_size=(320, 70))
        psim.PlotLines("||e_NTC||", e_ntc.astype(np.float32), graph_size=(320, 70))
        out = seq.results[state["k"]][0]
        psim.TextUnformatted(f"M_ret(k) = {out.retrieved.n}, ||e|| = {out.error:.3f}, ||e_NTC|| = {seq.results[state['k']][2]:.3f}, "
                             f"{1000 * out.seconds['total']:.0f} ms")

    show(0)
    ps.set_user_callback(callback)
    ps.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", type=int, default=5, choices=sorted(EXPERIMENTS))
    parser.add_argument("--no-gui", action="store_true", help="compute the whole sequence and print a summary")
    parser.add_argument("--plot", metavar="PNG", help="with --no-gui: save the two shape errors over time")
    parser.add_argument("--tracking", default="anchored", choices=["anchored", "literal", "none"])
    parser.add_argument("--step", type=int, default=4, help="increment of I in the refinements (1 in Algorithm 1, 4 in the code)")
    parser.add_argument("--Imax", type=int, default=30)
    args = parser.parse_args()
    seq = Sequence(args.experiment, Options(I_max=args.Imax, step=args.step, tracking=args.tracking))
    if args.no_gui:
        seq.compute(len(seq) - 1)
        summary(seq, args.experiment)
        if args.plot:
            plot(seq, args.experiment, args.plot)
    else:
        viewer(seq, args.experiment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
