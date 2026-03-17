#!/usr/bin/env python
"""Demo: Time-Consistent Surface Mapping on real experiment data.

Replicates the results from:
    I. Cuiral-Zueco and G. López-Nicolás, "Time Consistent Surface Mapping for
    Deformable Object Shape Control," IEEE T-ASE, 2025.

Usage
-----
    python demo.py                       # run all 5 experiments (live 3D)
    python demo.py --experiments 1 3     # run experiments 1 and 3
    python demo.py --no-gui              # headless — print errors only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the package is importable when running from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tcmap.data_loader import load_experiment
from tcmap.fmap import classic_zoomout
from tcmap.time_consistent import TimeConsistentMapper, _shape_error
from tcmap.mesh import Mesh
from tcmap.visualization import xyz_color

EXPERIMENT_NAMES = {
    1: "Mexican hat",
    2: "T-shaped noodle",
    3: "Pillow",
    4: "Foam rectangle",
    5: "Foam free-shape",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def run_experiment(exp_id: int, gui: bool = True):
    """Run one experiment and optionally visualise with Polyscope."""
    mat_path = DATA_DIR / f"experiment_data_{exp_id}.mat"
    if not mat_path.exists():
        print(f"  [!] Data file not found: {mat_path}")
        return

    sources, target = load_experiment(mat_path)
    n_frames = len(sources)
    print(f"  Loaded {n_frames} frames, target has {target.nv} vertices")

    NH = 30
    target.compute_basis(NH)
    tgt_colors = xyz_color(target)

    # Horizontal spacing for side-by-side display
    x_vals = target.vertices[:, 0]
    x_span = float(x_vals.max() - x_vals.min())
    offset = x_span * 1.5

    if gui:
        import polyscope as ps
        import polyscope.imgui as psim

        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.set_up_dir("y_up")
        ps.set_window_size(1600, 900)

        # Register target (static, centred to the right)
        sm_tgt = ps.register_surface_mesh(
            "Target N", target.vertices + np.array([offset, 0, 0]),
            target.faces, smooth_shade=True,
        )
        sm_tgt.add_color_quantity("xyz", tgt_colors, enabled=True)
        sm_tgt.set_edge_width(0.5)

        # Placeholders for source meshes (will be updated each frame)
        sm_ours = None
        sm_zo = None

    # ---------- State for computation ----------
    mapper = TimeConsistentMapper(
        target, n_initial=4, n_update=5, n_max=NH, step=4,
        use_slanted_diagonal=True, n_probes=2000,
    )
    errors_zo = np.zeros(n_frames)
    errors_tc = np.zeros(n_frames)
    times_zo = np.zeros(n_frames)
    times_tc = np.zeros(n_frames)

    # ---------- Playback state ----------
    frame_idx = [0]
    playing = [True]
    speed = [1]          # frames to advance per tick
    computed_up_to = [0]  # how many frames have been computed

    # Pre-allocate storage
    T12_zo_all = [None] * n_frames
    T12_tc_all = [None] * n_frames

    def _compute_frame(i: int):
        """Compute mappings for frame *i* (if not already done)."""
        if T12_zo_all[i] is not None:
            return  # already computed

        src = sources[i]
        S1 = Mesh(src.vertices.copy(), src.faces.copy())
        S1.compute_basis(NH)

        # Classic ZoomOut
        t0 = time.perf_counter()
        T12_zo, _ = classic_zoomout(S1, target, n_initial=4, n_max=NH, step=4,
                                     use_slanted_diagonal=False)
        times_zo[i] = time.perf_counter() - t0
        errors_zo[i] = _shape_error(S1, target, T12_zo)

        # Time-consistent (ours)
        S1_tc = Mesh(src.vertices.copy(), src.faces.copy())
        S1_tc.compute_basis(NH)
        t0 = time.perf_counter()
        T12_tc, _, err_tc = mapper.step(S1_tc)
        times_tc[i] = time.perf_counter() - t0
        errors_tc[i] = err_tc

        T12_zo_all[i] = T12_zo
        T12_tc_all[i] = T12_tc
        computed_up_to[0] = i + 1

    def _update_polyscope(i: int):
        """Update Polyscope meshes for frame *i*."""
        nonlocal sm_ours, sm_zo
        src = sources[i]

        # --- Ours (left-centre) ---
        sm_ours = ps.register_surface_mesh(
            "Ours - M(k)", src.vertices, src.faces, smooth_shade=True,
        )
        mapped_c = tgt_colors[T12_tc_all[i]]
        sm_ours.add_color_quantity("correspondence", mapped_c, enabled=True)
        err_v = np.linalg.norm(src.vertices - target.vertices[T12_tc_all[i]], axis=1)
        sm_ours.add_scalar_quantity("error", err_v, enabled=False, cmap="reds")
        sm_ours.set_edge_width(0.5)

        # --- ZoomOut (further left) ---
        sm_zo = ps.register_surface_mesh(
            "ZoomOut - M(k)", src.vertices - np.array([offset, 0, 0]),
            src.faces, smooth_shade=True,
        )
        mapped_zo_c = tgt_colors[T12_zo_all[i]]
        sm_zo.add_color_quantity("correspondence", mapped_zo_c, enabled=True)
        err_zo_v = np.linalg.norm(src.vertices - target.vertices[T12_zo_all[i]], axis=1)
        sm_zo.add_scalar_quantity("error", err_zo_v, enabled=False, cmap="reds")
        sm_zo.set_edge_width(0.5)

    if not gui:
        # Headless: compute all frames, print results
        for i in range(n_frames):
            _compute_frame(i)
            print(f"\r  Frame {i+1:3d}/{n_frames}"
                  f"  |  ZoomOut err={errors_zo[i]:.4f} ({times_zo[i]:.3f}s)"
                  f"  |  Ours err={errors_tc[i]:.4f} ({times_tc[i]:.3f}s)"
                  f"  |  #nodes={sources[i].nv}", end="")
        print()
        return

    # ---------- Live Polyscope callback ----------
    def callback():
        nonlocal sm_ours, sm_zo
        i = frame_idx[0]

        # --- UI panel ---
        psim.SetNextWindowPos((10, 10))
        psim.SetNextWindowSize((380, 260))
        psim.Begin("Playback")

        psim.Text(f"Experiment {exp_id}: {EXPERIMENT_NAMES.get(exp_id, '')}")
        psim.Separator()

        _, playing[0] = psim.Checkbox("Play", playing[0])
        psim.SameLine()
        if psim.Button("<"):
            frame_idx[0] = max(0, i - 1)
        psim.SameLine()
        if psim.Button(">"):
            frame_idx[0] = min(n_frames - 1, i + 1)
        psim.SameLine()
        if psim.Button("Reset"):
            frame_idx[0] = 0

        changed, new_i = psim.SliderInt("Frame", i, 0, n_frames - 1)
        if changed:
            frame_idx[0] = new_i
            i = new_i

        changed_s, new_s = psim.SliderInt("Speed", speed[0], 1, 10)
        if changed_s:
            speed[0] = new_s

        # Ensure frame is computed (sequential requirement for TC mapper)
        while computed_up_to[0] <= i:
            _compute_frame(computed_up_to[0])

        psim.Separator()
        psim.Text(f"Frame {i+1}/{n_frames}   |   Nodes: {sources[i].nv}")
        psim.Text(f"Ours  err: {errors_tc[i]:.4f}   time: {times_tc[i]:.3f}s")
        psim.Text(f"ZoomOut err: {errors_zo[i]:.4f}   time: {times_zo[i]:.3f}s")

        # Mini error bar
        if computed_up_to[0] > 1:
            max_err = max(errors_zo[:computed_up_to[0]].max(),
                          errors_tc[:computed_up_to[0]].max(), 1e-6)
            bar_tc = errors_tc[i] / max_err
            bar_zo = errors_zo[i] / max_err
            psim.Text(f"Ours:    {'|' * int(bar_tc * 30)}")
            psim.Text(f"ZoomOut: {'|' * int(bar_zo * 30)}")

        psim.End()

        # Update meshes
        _update_polyscope(i)

        # Auto-advance
        if playing[0]:
            frame_idx[0] = min(n_frames - 1, i + speed[0])
            if frame_idx[0] >= n_frames - 1:
                playing[0] = False

    # Precompute first frame so there's something to show immediately
    _compute_frame(0)
    _update_polyscope(0)

    ps.set_user_callback(callback)
    ps.show()
    ps.clear_user_callback()

    # After closing Polyscope, show matplotlib summary
    _plot_summary(exp_id, sources, errors_zo, errors_tc, times_zo, times_tc,
                  computed_up_to[0])


def _plot_summary(exp_id, sources, errors_zo, errors_tc, times_zo, times_tc, n_computed):
    """Matplotlib summary plots (saved to PNG)."""
    import matplotlib.pyplot as plt

    n = n_computed
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Experiment {exp_id}: {EXPERIMENT_NAMES.get(exp_id, '')}")

    ax = axes[0]
    ax.plot(errors_zo[:n], "b--", alpha=0.6, label="ZoomOut (NTC)")
    ax.plot(errors_tc[:n], "r-", linewidth=2, label="Ours (TC)")
    ax.set_xlabel("Frame k")
    ax.set_ylabel("Shape error")
    ax.set_title("Shape error")
    ax.legend()
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.plot(times_zo[1:n], "b--", alpha=0.6, label="ZoomOut")
    ax.plot(times_tc[1:n], "r-", linewidth=2, label="Ours")
    ax.set_xlabel("Frame k")
    ax.set_ylabel("Time (s)")
    ax.set_title("Processing time")
    ax.legend()
    ax.set_ylim(bottom=0)

    ax = axes[2]
    ax.plot([sources[i].nv for i in range(n)], "k-")
    ax.set_xlabel("Frame k")
    ax.set_ylabel("Mesh nodes")
    ax.set_title("Mesh size")

    plt.tight_layout()
    plt.savefig(f"experiment_{exp_id}_results.png", dpi=150)
    print(f"  Saved experiment_{exp_id}_results.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Time-Consistent Surface Mapping Demo")
    parser.add_argument(
        "--experiments", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="Experiment IDs to run (1-5)",
    )
    parser.add_argument("--no-gui", action="store_true", help="Disable visualisation")
    args = parser.parse_args()

    for exp_id in args.experiments:
        print(f"\n{'='*60}")
        print(f"Experiment {exp_id}: {EXPERIMENT_NAMES.get(exp_id, 'Unknown')}")
        print(f"{'='*60}")
        run_experiment(exp_id, gui=not args.no_gui)


if __name__ == "__main__":
    main()
