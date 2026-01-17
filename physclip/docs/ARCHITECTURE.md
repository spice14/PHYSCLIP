# PHYSCLIP: System Architecture

## Overview

PHYSCLIP (Physics-informed Contrastive Learning for Initial condition Prediction) is a contrastive learning framework for learning robust physics representations from temporal trajectories of differential equations.

## Directory Structure

```
physclip/
├── src/
│   ├── models/              # Neural network architectures
│   │   ├── encoders.py           # Field and text encoders
│   │   ├── trajectory_models.py   # TrajectoryEncoder (temporal aggregation)
│   │   └── contrastive_loss.py    # InfoNCE-style contrastive loss
│   ├── data/                # Dataset and PDE solvers
│   │   ├── pde_solver.py         # Burgers equation spectral solver
│   │   ├── dataset_generation.py  # Generate synthetic training data
│   │   └── trajectory_dataset.py  # PyTorch dataset with windowing
│   └── utils/               # Utility functions
│       └── environment_check.py   # GPU and environment setup
├── scripts/                 # Executable training scripts
│   ├── train_baseline.py         # Snapshot-level contrastive learning (v0)
│   └── train_contrastive.py      # Trajectory-level with windowing (v1)
├── tests/                   # Unit and integration tests
│   └── test_trajectory_encoder.py
├── experiments/             # Experiment tracking and results
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md      # This file
│   ├── USAGE.md
│   └── METHODS.md
├── data/                    # Datasets (gitignored for large files)
├── results/                 # Trained models and visualizations
├── README.md
├── requirements.txt
└── setup.py
```

## Core Components

### 1. Models (`src/models/`)

#### Encoders (`encoders.py`)
- **FieldEncoder**: CNN-based spatial encoder for physics fields
  - Input: (batch, nx) spatial field
  - Output: (batch, latent_dim) normalized embedding
  
- **TextEncoder**: Transformer-based text encoder
  - Input: Physics description strings
  - Output: (batch, latent_dim) normalized embedding

#### Trajectory Models (`trajectory_models.py`)
- **TrajectoryEncoder**: Two-stage temporal encoder
  - Stage 1: Apply FieldEncoder to each timestep (weight sharing)
  - Stage 2: Mean pooling over time
  - Input: (batch, K, nx) trajectory
  - Output: (batch, latent_dim) temporal embedding

#### Contrastive Loss (`contrastive_loss.py`)
- **ContrastiveLoss**: InfoNCE-style symmetric contrastive loss
  - Aligns field/trajectory embeddings with text embeddings
  - Temperature-scaled softmax
  - Batch-level negatives

### 2. Data Pipeline (`src/data/`)

#### PDE Solver (`pde_solver.py`)
- **BurgersSpectralSolver**: Spectral method for 1D Burgers equation
  - RK2 temporal integration
  - FFT-based spatial derivatives
  - Periodic boundary conditions

#### Dataset Generation (`dataset_generation.py`)
- Generates synthetic trajectories for various physics regimes
- Configurable viscosity ranges, initial conditions
- Produces metadata mapping trajectories to descriptions

#### Trajectory Dataset (`trajectory_dataset.py`)
- **BurgersTrajectoryDataset**: PyTorch dataset with multi-scale windowing
  - Extracts overlapping temporal windows at multiple time scales
  - Preserves physics semantics (each window from same trajectory shares ν)
  - Deterministic windowing (no randomness in __getitem__)

### 3. Training (`scripts/`)

#### Baseline Training (`train_baseline.py`)
- Snapshot-level contrastive learning
- Each snapshot u(x, t_i) is one sample
- 352 samples from 32 trajectories
- Hyperparameters: BS=8, LR=1e-3, NT=100

#### Contrastive Training (`train_contrastive.py`)
- Trajectory-window-level learning
- Multi-scale temporal windows: sizes [5, 7, 9]
- 360+ samples from 32 trajectories
- Hyperparameters: BS=8, LR=1e-3, NT=200
- Learning rate scheduling with 0.5× decay every 50 epochs

## Data Flow

### Training Pipeline

```
Raw Trajectories (.npy files)
        ↓
BurgersTrajectoryDataset (multi-scale windowing)
        ↓
DataLoader (collate temporal windows)
        ↓
Batch: (windows, texts, viscosities)
        ↓
Encoders: {
    TrajectoryEncoder(windows) → z_field
    TextEncoder(texts) → z_text
}
        ↓
ContrastiveLoss(z_field, z_text)
        ↓
Backprop & Optimize
```

### Inference Pipeline

```
Test Trajectory → TrajectoryEncoder → z_field
Physics Description → TextEncoder → z_text
Similarity: z_field · z_text
```

## Key Design Decisions

### 1. Multi-Scale Temporal Windowing
- **Why**: Captures dynamics at different time scales
- **Implementation**: Extract windows of sizes [5, 7, 9] with stride=2
- **Justification**: Physics-valid data augmentation (not synthetic)

### 2. Temperature-Scaled Contrastive Loss
- **Why**: Controls similarity distribution between modalities
- **Value**: temperature=0.07 (same as CLIP)
- **Effect**: Sharper alignment, better generalization

### 3. Weight Sharing in TrajectoryEncoder
- **Why**: Reduces parameters, encodes temporal structure
- **Implementation**: Same FieldEncoder applied to all timesteps
- **Result**: 21.5K parameters (tractable for limited data)

### 4. Frozen Text Encoder
- **Why**: Stable reference embeddings from pre-trained BERT
- **Implementation**: Only train TrajectoryEncoder
- **Benefit**: Faster convergence, interpretable text space

## Physics Semantics

### Preserved Invariants

1. **Governing Equation**: All trajectories at ν satisfy same Burgers PDE
2. **Viscosity**: Each trajectory has ONE ν (no mixing)
3. **Temporal Order**: Windows are contiguous (no shuffling)
4. **Causal Structure**: t_i < t_j in windows (no time reversal)

### Contrastive Positive Pairs

```
(trajectory_i, description_i) are positive if:
  - Both derived from same viscosity ν
  - Both share same physics description
  - Encode same governing equation
```

## Performance Characteristics

### Training
- **Data**: ~360-2000 samples (depending on windowing)
- **Parameters**: ~22.8M total (22.76M text + 21.5K field)
- **Runtime**: ~2 hours (200 epochs, 1 GPU)
- **Convergence**: Loss decreases 2.77 → 2.55 (8% improvement)

### Evaluation
- **Nearest Neighbor Accuracy**: 40-60% (4-8 viscosity regimes)
- **PCA Separation**: Visual clustering by viscosity
- **Similarity Scores**: 0.02-0.05 (expected for L2-normalized embeddings)

## Future Extensions

1. **Heterogeneous Physics**: Multiple PDEs (Burgers, KdV, NLS)
2. **Longer-Horizon**: Higher-resolution temporal windows
3. **Noisy Data**: Robustness to measurement noise
4. **Transfer Learning**: Pre-train on synthetic, fine-tune on real
5. **Attention-Based**: Replace mean pooling with attention

## References

- CLIP: Radford et al. (2021)
- Burgers Equation: https://en.wikipedia.org/wiki/Burgers_equation
- Spectral Methods: Boyd (2001)
- Contrastive Learning: Chen et al. (2020), He et al. (2020)
