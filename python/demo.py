#!/usr/bin/env python
"""Demo: Time-Consistent Surface Mapping on real experiment data.

Replicates the results from:
    I. Cuiral-Zueco and G. López-Nicolás, "Time Consistent Surface Mapping for
    Deformable Object Shape Control," IEEE T-ASE, 2025.

Usage
-----
    python demo.py                       # run all 5 experiments
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

    # Precompute target basis once
    NH = 30
    target.compute_basis(NH)

    # --- Classic ZoomOut (baseline) ---
    errors_zo = np.zeros(n_frames)
    times_zo = np.zeros(n_frames)

    # --- Time-consistent (ours) ---
    mapper = TimeConsistentMapper(
        target, n_initial=4, n_update=5, n_max=NH, step=4,
        use_slanted_diagonal=True, n_probes=2000,
    )
    errors_tc = np.zeros(n_frames)
    times_tc = np.zeros(n_frames)

    T12_zo_all = []
    T12_tc_all = []

    for i, src in enumerate(sources):
        S1 = Mesh(src.vertices.copy(), src.faces.copy())
        S1.compute_basis(NH)

        # -- Classic ZoomOut --
        t0 = time.perf_counter()
        T12_zo, _ = classic_zoomout(S1, target, n_initial=4, n_max=NH, step=4,
                                     use_slanted_diagonal=False)
        times_zo[i] = time.perf_counter() - t0
        errors_zo[i] = _shape_error(S1, target, T12_zo)

        # -- Time-consistent (ours) --
        S1_tc = Mesh(src.vertices.copy(), src.faces.copy())
        S1_tc.compute_basis(NH)
        t0 = time.perf_counter()
        T12_tc, T12_cons, err_tc = mapper.step(S1_tc)
        times_tc[i] = time.perf_counter() - t0
        errors_tc[i] = err_tc

        T12_zo_all.append(T12_zo)
        T12_tc_all.append(T12_tc)

        print(f"\r  Frame {i+1:3d}/{n_frames}"
              f"  |  ZoomOut err={errors_zo[i]:.4f} ({times_zo[i]:.3f}s)"
              f"  |  Ours err={errors_tc[i]:.4f} ({times_tc[i]:.3f}s)"
              f"  |  #nodes={S1.nv}", end="")

    print()  # newline after progress

    # ---- Visualisation ----
    if gui:
        _visualise_results(
            exp_id, sources, target,
            T12_zo_all, T12_tc_all,
            errors_zo, errors_tc, times_zo, times_tc,
        )


def _visualise_results(
    exp_id, sources, target,
    T12_zo_all, T12_tc_all,
    errors_zo, errors_tc, times_zo, times_tc,
):
    """Interactive Polyscope visualisation + matplotlib error/timing plots."""
    import polyscope as ps
    from tcmap.visualization import (
        init_polyscope,
        show_mesh,
        xyz_color,
        visualise_error,
    )
    import matplotlib.pyplot as plt

    # ---- Matplotlib: error & timing curves ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Experiment {exp_id}: {EXPERIMENT_NAMES.get(exp_id, '')}")

    ax = axes[0]
    ax.plot(errors_zo, "b--", label="ZoomOut (NTC)")
    ax.plot(errors_tc, "r-", linewidth=2, label="Ours (TC)")
    ax.set_xlabel("Frame k")
    ax.set_ylabel("Shape error")
    ax.set_title("Shape error over time")
    ax.legend()
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.plot(times_zo[1:], "b--", label="ZoomOut")
    ax.plot(times_tc[1:], "r-", linewidth=2, label="Ours")
    ax.set_xlabel("Frame k")
    ax.set_ylabel("Time (s)")
    ax.set_title("Processing time")
    ax.legend()
    ax.set_ylim(bottom=0)

    ax = axes[2]
    ax.plot([s.nv for s in sources], "k-")
    ax.set_xlabel("Frame k")
    ax.set_ylabel("Number of mesh nodes")
    ax.set_title("Mesh size")

    plt.tight_layout()
    plt.savefig(f"experiment_{exp_id}_results.png", dpi=150)
    plt.show(block=False)

    # ---- Polyscope: 3D visualisation (last frame) ----
    init_polyscope()

    last = len(sources) - 1
    src = sources[last]
    tgt_colors = xyz_color(target)

    # Our method — source coloured by target correspondence
    sm1 = ps.register_surface_mesh(
        "Source (ours)", src.vertices, src.faces, transparency=0.0
    )
    sm1.add_color_quantity("map_color", tgt_colors[T12_tc_all[last]], enabled=True)

    # Target mesh
    offset = np.array([target.vertices[:, 0].ptp() * 1.3, 0, 0])
    sm2 = ps.register_surface_mesh(
        "Target", target.vertices + offset, target.faces
    )
    sm2.add_color_quantity("xyz_color", tgt_colors, enabled=True)

    # ZoomOut baseline — source coloured
    sm3 = ps.register_surface_mesh(
        "Source (ZoomOut)", src.vertices - offset, src.faces
    )
    sm3.add_color_quantity("map_color", tgt_colors[T12_zo_all[last]], enabled=True)

    # Error visualisation
    diff_tc = src.vertices - target.vertices[T12_tc_all[last]]
    err_tc_per_v = np.linalg.norm(diff_tc, axis=1)
    sm1.add_scalar_quantity("error", err_tc_per_v, enabled=False, cmap="reds")

    diff_zo = src.vertices - target.vertices[T12_zo_all[last]]
    err_zo_per_v = np.linalg.norm(diff_zo, axis=1)
    sm3.add_scalar_quantity("error", err_zo_per_v, enabled=False, cmap="reds")

    ps.show()


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
