# Time-consistent surface mapping in Python

A Python implementation of the method of

> I. Cuiral-Zueco and G. López-Nicolás, "Time Consistent Surface Mapping for Deformable Object Shape Control," *IEEE Transactions on Automation Science and Engineering*, vol. 22, pp. 11099-11111, 2025, [doi: 10.1109/TASE.2025.3529180](https://doi.org/10.1109/TASE.2025.3529180),

with a 3D viewer of the five recorded experiments. It follows the equations of the paper and, where they differ, the reference MATLAB code of this repository (see below). The same method runs in the browser in the [interactive demos](https://ignaciocuiral.com/time-consistent-surface-mapping/).

## Install and run

```bash
python -m venv .venv
.venv/Scripts/activate              # Windows; on Linux and macOS: source .venv/bin/activate
pip install -r python/requirements.txt

python python/demo.py --experiment 5                          # 3D viewer: play the experiment
python python/demo.py --experiment 1 --no-gui                 # whole sequence, printed summary
python python/demo.py --experiment 3 --no-gui --plot out.png  # and the two shape errors over time
```

The recorded meshes (`data/experiment_data_*.mat`) are stored with Git LFS: clone the repository with Git LFS installed, or run `git lfs pull`.

The viewer shows, for every frame, the mesh of the experiment colored by the map of ZoomOut computed at that frame (left), the tracked mesh colored by the time-consistent map (middle), and the target with its reference colors (right), with the two shape errors over time. With `--no-gui`, the summary on this computer (Python 3.12, NumPy and SciPy) is:

| Experiment | Points per mesh | Time-consistent: ‖e‖ (largest change between frames) | ZoomOut every frame: ‖e_NTC‖ (largest change) | Time per frame |
|---|---|---|---|---|
| 1 Mexican hat | 527-559 | 2.02 → 0.19 (0.22) | 0.23 to 8.70 (8.11) | 37 ms |
| 2 T-shaped pool noodle | 297-374 | 3.25 → 0.47 (0.19) | 0.48 to 7.52 (6.78) | 21 ms |
| 3 Pillow | 390-438 | 2.06 → 0.96 (0.24) | 0.39 to 9.21 (8.77) | 30 ms |
| 4 Rectangular foam sheet | 573-727 | 4.79 → 0.79 (0.30) | 0.81 to 11.51 (10.58) | 59 ms |
| 5 Foam cut-out | 416-470 | 3.37 → 0.80 (0.23) | 0.47 to 11.04 (10.30) | 31 ms |

Errors in meters, ‖e‖ = (Σ_m ‖e_m‖²)^½ as in (21). The recorded sequences are the experiments of the paper, whose robots were driven by the MATLAB implementation; this code recomputes the maps from the recorded meshes.

## Package

| Module | Content |
|---|---|
| `tcmap/mesh.py` | `Mesh`, the cotangent Laplacian and lumped areas, and the Laplace-Beltrami eigenbasis (Sec. 2.1) |
| `tcmap/fmap.py` | functional maps from point maps (9), (17), point maps from functional maps (8), (16), the rule (19) for J, and the coarse-to-fine refinement (ZoomOut) |
| `tcmap/icp.py` | rigid ICP for the first map (Sec. 3.1) |
| `tcmap/mapping.py` | `TimeConsistentMapper` (Algorithm 1: initialization, tracking of the basis and of the points, warm start, non-isometry rule), and ZoomOut computed anew at every frame |
| `tcmap/control.py` | `SequentialShapeController`: the buffers, the interaction matrix (23), and the control law (24) of Sec. 4.1 |
| `tcmap/data.py` | the recorded experiments |
| `tcmap/colors.py` | the colors of the surface maps |
| `demo.py` | the viewer and the command-line summary |

```python
from tcmap import TimeConsistentMapper, Options, load_experiment

frames, target = load_experiment(5)
mapper = TimeConsistentMapper(target, Options(I_initial=4, I_update=5, I_max=30, step=4))
for mesh in frames:
    out = mapper.step(mesh)     # out.X: tracked points X(k); out.Pi: their images on the target;
                                # out.E = X(k) - Pi(k) Y (20); out.error = ||e|| (21); out.C = C(I_max, k)
```

## Choices where the paper and the code differ

- **Tracking (Sec. 3.2).** Read literally, (14)–(15) re-read the basis and the points on the rows that Π^M(k) selects. Π^M(k) is not one-to-one, so tracked points that select the same new point keep the same embedding afterwards and merge. The reference code (`previousstatefmap.m`, `timeconsistentsurfacemapping.m`) instead aligns the basis of every new mesh with the whole map C^M(k) between consecutive meshes, and finds the tracked points by their embedding at k = 1. `Options(tracking="anchored")` (the default) does the latter; `tracking="literal"` applies (13)–(15) as written.
- **Refinement step.** Algorithm 1 increases I by one; the reference code by 4 (`Cstep`). The default is 4; `Options(step=1)` follows Algorithm 1.
- **First map (Sec. 3.1).** The paper initializes with ICP and nearest neighbors (10), as here; `initialisefmap.m` instead probes random diagonal functional maps and keeps the one with the smallest extrinsic residual.
- **Basis.** As in ZoomOut and the code, the constant eigenvector is the first column of the basis.
- **Control law (23).** Printed with −γI₆; as a Tikhonov regularization it adds γI₆. The rotations of the actions are scaled by a lever (10 cm by default) so that one γ weights translations and rotations alike.
