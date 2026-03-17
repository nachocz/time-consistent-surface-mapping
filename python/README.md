# Time-Consistent Surface Mapping — Python Implementation

Python implementation of the method described in:

> I. Cuiral-Zueco and G. López-Nicolás, "Time Consistent Surface Mapping for Deformable Object Shape Control," *IEEE Transactions on Automation Science and Engineering*, 2025. DOI: [10.1109/TASE.2025.3529180](https://doi.org/10.1109/TASE.2025.3529180)

## Overview

This package implements the **time-consistent surface mapping** algorithm for computing point-to-point correspondences between deforming 3D surfaces. The method combines:

- **Functional maps** via Laplace-Beltrami eigenbasis
- **ZoomOut** coarse-to-fine refinement (Melzi et al., 2019)
- **Time-consistent basis tracking** across frames
- **Non-isometry robustness** through slanted-diagonal eigenvalue matching

## Installation

```bash
cd python
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `scipy` | Sparse matrices, eigendecomposition, I/O |
| `polyscope` | Interactive 3D mesh visualisation |
| `robust-laplacian` | (optional) Alternative Laplacian computation |
| `matplotlib` | 2D plots (error curves, timing) |

## Quick start

```bash
# Run all 5 experiments with 3D visualisation
python demo.py

# Run specific experiments
python demo.py --experiments 1 3

# Headless mode (no GUI, prints errors only)
python demo.py --experiments 1 --no-gui
```

## Package structure

```
python/
├── demo.py                     # Main demo script
├── requirements.txt
└── tcmap/
    ├── __init__.py
    ├── mesh.py                 # Cotangent Laplacian, eigenbasis, Mesh class
    ├── fmap.py                 # Functional maps: ZoomOut, initialisation
    ├── time_consistent.py      # Time-consistent pipeline (Algorithm 1)
    ├── data_loader.py          # Load MATLAB .mat experiment data
    └── visualization.py        # Polyscope 3D visualisation
```

## API usage

```python
from tcmap.data_loader import load_experiment
from tcmap.time_consistent import TimeConsistentMapper

# Load data
sources, target = load_experiment("../data/experiment_data_1.mat")

# Create mapper
mapper = TimeConsistentMapper(
    target,
    n_initial=4,    # Initial basis size (I_initial)
    n_update=5,     # Reuse sub-matrix size (I_update)
    n_max=30,       # Maximum basis size (I_max)
    step=4,         # Refinement step
)

# Process each frame
for frame_mesh in sources:
    frame_mesh.compute_basis(30)
    T12, T12_consistent, error = mapper.step(frame_mesh)
    # T12: point-to-point map (source vertex i → target vertex T12[i])
    # T12_consistent: time-consistent map (indexed by first-frame vertices)
    # error: shape error ||X - Π·Y||²
```

## Experiments

The `data/` folder contains 5 real experiments from RGB-D acquisitions:

| ID | Object | Description |
|---|---|---|
| 1 | Mexican hat | Hat-shaped deformable object |
| 2 | T-shaped noodle | Elongated deformable object |
| 3 | Pillow | Soft pillow deformation |
| 4 | Foam rectangle | Rectangular foam piece |
| 5 | Foam free-shape | Free-form foam cutout |

## Method summary

1. **Frame k=1**: Initialise functional map via random probing of 2000 diagonal matrices, selecting the one minimising extrinsic alignment error. Refine with ZoomOut.

2. **Frame k>1**:
   - **Basis tracking**: Map new mesh eigenvectors to previous frame for sign/order consistency.
   - **Warm start**: Reuse the top-left sub-matrix of the previous functional map.
   - **Non-isometry**: Dynamically adjust target basis size using area ratio $\rho(k) = \sqrt{A(\mathcal{M}(k))/A(\mathcal{N})}$.

## Citation

```bibtex
@article{cuiral2025time,
  title={Time Consistent Surface Mapping for Deformable Object Shape Control},
  author={Cuiral-Zueco, Ignacio and L{\'o}pez-Nicol{\'a}s, Gonzalo},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2025},
  doi={10.1109/TASE.2025.3529180}
}
```
