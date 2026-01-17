# PHYSCLIP v0

Repository structure:

```
.
├── README.md                    # Project overview and vision
├── requirements.txt             # Dependencies (PyTorch, NumPy, etc)
├── train_physclip_v0.py         # Complete training + evaluation script
│
├── data/                        # Physics data generation
│   ├── __init__.py
│   ├── burgers_solver.py        # Spectral solver for 1D Burgers equation
│   └── dataset.py               # Dataset generation script
│
├── models/                      # Neural network encoders
│   ├── __init__.py
│   ├── encoders.py              # FieldEncoder + TextEncoder
│   └── losses.py                # ContrastiveLoss (CLIP-style)
│
├── analysis/                    # Dataset and results analysis
│   ├── __init__.py
│   └── inspect_dataset.py       # Dataset validation script
│
└── results/                     # Output directory (generated during training)
    ├── physclip_v0_embeddings.png
    └── ...
```

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Generate dataset:
   ```
   cd data
   python dataset.py
   ```

3. Inspect dataset:
   ```
   cd analysis
   python inspect_dataset.py
   ```

4. Train PHYSCLIP v0:
   ```
   python train_physclip_v0.py
   ```

## Key Files

- **burgers_solver.py**: Spectral solver for the 1D viscous Burgers equation
  - Fourier pseudospectral method
  - RK4 time integration
  - 2/3 dealiasing
  - Periodic boundary conditions on [0, 2π]

- **dataset.py**: Generates synthetic physics data
  - Multiple sine-wave initial conditions per viscosity
  - Physics-text alignment: many trajectories → one physics description
  - Frozen dataset format (do not regenerate)

- **encoders.py**: Dual-encoder architecture
  - FieldEncoder: 1D CNN for field snapshots
  - TextEncoder: Pretrained sentence-transformer (frozen base)
  - Shared L2-normalized latent space

- **losses.py**: CLIP-style contrastive loss
  - InfoNCE with cosine similarity
  - Symmetric loss (field→text + text→field)
  - Temperature scaling

- **train_physclip_v0.py**: Complete experiment
  - Training loop with contrastive loss
  - Evaluation: PCA visualization + nearest-neighbor validation
  - Success criteria: embeddings cluster by viscosity, not initial condition

## Design Principles

PHYSCLIP v0 tests whether physics **regimes** are representationally learnable:

- NO physics enforcement (no PDE residuals, no PINNs)
- NO prediction or solving
- ONLY representation alignment: field behavior ↔ physics description
- Single testbed: 1D viscous Burgers equation
- Clean separation: solver / dataset / encoders / loss / training

This is a foundation for later integration with physics-informed solvers.
