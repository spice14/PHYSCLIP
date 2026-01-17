# PHYSCLIP: Usage Guide

## Quick Start

### 1. Installation

```bash
cd physclip
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
# Generate synthetic training data
python -m src.data.dataset_generation

# Custom configuration
python -c "
from src.data.dataset_generation import generate_dataset
import numpy as np

generate_dataset(
    output_dir='./data/burgers_data',
    nu_values=np.linspace(0.01, 0.2, 20),
    n_trajectories_per_viscosity=10,
    t_final=1.0,
    nx=256,
    seed=42
)
"
```

### 3. Train Model

#### Baseline (Snapshot-Level)
```bash
python scripts/train_baseline.py
```

#### Contrastive (Trajectory-Level)
```bash
python scripts/train_contrastive.py
```

### 4. Evaluate

Results are automatically saved to `results/`:
- `physclip_v1_embeddings.png`: PCA visualization
- Nearest neighbor accuracy printed to console

## Dataset Generation

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nu_values` | `[0.01, 0.2]` (20 pts) | Viscosity coefficients |
| `n_trajectories_per_viscosity` | 10 | ICs per viscosity |
| `phase_shift_range` | `(0, 2π)` | Initial condition phase variation |
| `amplitude_variation_range` | `(0.8, 1.2)` | IC amplitude scaling |
| `t_final` | 1.0 | Integration time |
| `nx` | 256 | Spatial grid points |
| `dt` | 0.0005 | Temporal step size |
| `n_snapshots` | 11 | Time snapshots per trajectory |

### Output Format

Each trajectory saved as:
- **File**: `trajectory_XXXXXX.npy` (shape: 11×256)
- **Metadata**: `metadata.txt` (trajectory→description→ν mapping)

### Physics Coverage

| Configuration | # Base Traj | # Windowed Samples |
|---------------|-------------|-------------------|
| Test (10 ν × 5) | 50 | ~500 |
| Default (20 ν × 10) | 200 | ~2000 |
| Dense (30 ν × 20) | 600 | ~6000 |

## Training

### Baseline Model (v0)

Snapshot-level contrastive learning.

```python
python scripts/train_baseline.py \
    --dataset_dir ./data/burgers_data \
    --batch_size 8 \
    --learning_rate 1e-3 \
    --num_epochs 100
```

**Key Features**:
- Individual snapshots as training samples
- 352 samples from 32 trajectories
- Fast convergence (100 epochs)

**Expected Results**:
- Loss: 2.074 → 1.961 (5.5% decrease)
- Nearest neighbor accuracy: ~60%
- Training time: ~30 min

### Contrastive Model (v1)

Trajectory-level with multi-scale windowing.

```python
python scripts/train_contrastive.py \
    --dataset_dir ./data/burgers_data \
    --window_sizes 5 7 9 \
    --stride 2 \
    --batch_size 8 \
    --learning_rate 1e-3 \
    --num_epochs 200
```

**Key Features**:
- Temporal windows at multiple scales
- 360+ samples from 32 trajectories
- Extended training with LR scheduling
- Better physics coverage (denser ν manifold)

**Expected Results**:
- Loss: 2.01 → 2.65 (lower is better with more windows)
- Cleaner PCA clustering by viscosity
- More robust representations

## Hyperparameter Tuning

### Learning Rate

| Scenario | Recommended |
|----------|-------------|
| Small dataset (<100 samples) | 1e-2 |
| Medium dataset (100-1000) | 1e-3 |
| Large dataset (>1000) | 1e-4 |

### Batch Size

| Data Regime | Recommended |
|------------|-------------|
| <100 samples | 4 |
| 100-500 | 8 |
| 500-2000 | 16 |

### Number of Epochs

| Model | Recommended |
|-------|-------------|
| Baseline (352 samples) | 100 |
| Contrastive (360+ samples) | 200 |
| Large (>1000 samples) | 300+ |

## Evaluation Metrics

### 1. PCA Visualization

Points colored by viscosity should cluster by ν value.

**Good**: Clear separation by viscosity bands  
**Bad**: Random scatter (model hasn't learned ν structure)

### 2. Nearest Neighbor Accuracy

For random field embeddings, find nearest text embedding.

```
Sample i:
  True viscosity: ν=0.100
  Predicted (nearest text): ν=0.100 ✓
  Similarity: 0.0234
```

Accuracy = (# matches) / (# samples)

### 3. Loss Convergence

Monitor for:
- ✓ Smooth decrease (good gradient flow)
- ✓ Plateau by epoch 50-100 (sufficient data)
- ✗ Noisy/non-monotonic (bad learning rate)
- ✗ NaN (numerical instability)

## Troubleshooting

### Problem: Loss doesn't decrease

**Causes**:
- Learning rate too low (try 1e-3)
- Batch size too large (try BS=8)
- Insufficient gradient updates per epoch

**Solution**:
```python
LEARNING_RATE = 1e-3  # Increase from 1e-4
BATCH_SIZE = 8        # Decrease from 16
NUM_EPOCHS = 200      # Increase training
```

### Problem: Loss becomes NaN

**Causes**:
- Numerical instability in PDE solver
- Gradient explosion in loss

**Solution**:
- Reduce dt in dataset generation (0.0005)
- Add gradient clipping (already enabled)
- Check initial condition amplitude range

### Problem: Poor nearest neighbor accuracy

**Causes**:
- Too few viscosity regimes (ν ∈ {0.01, 0.05, 0.1, 0.2} only 4)
- Insufficient training samples per ν
- Text descriptions not diverse enough

**Solution**:
- Generate more viscosity values: `np.linspace(0.01, 0.2, 20)`
- Increase `n_trajectories_per_viscosity`
- Vary initial condition phases more

## Advanced Usage

### Custom Physics

Modify `src/data/pde_solver.py` to solve different PDEs:

```python
class KdVSolver(PDESolver):
    """Korteweg-de Vries equation solver"""
    def _rhs(self, u_hat):
        u = np.fft.ifft(u_hat).real
        du_dx = np.fft.ifft(1j * self.k * u_hat).real
        d3u_dx3 = np.fft.ifft((1j * self.k)**3 * u_hat).real
        return -u * du_dx - d3u_dx3
```

### Custom Architecture

Modify `src/models/trajectory_models.py`:

```python
class AttentionTrajectoryEncoder(nn.Module):
    """Use attention instead of mean pooling"""
    def __init__(self, field_encoder, latent_dim):
        super().__init__()
        self.field_encoder = field_encoder
        self.attention = nn.MultiheadAttention(latent_dim, num_heads=4)
    
    def forward(self, trajectories):
        # Shape: (B, K, nx) → (B, K, latent_dim)
        encoded = self._encode_snapshots(trajectories)
        # Apply attention over time
        out, _ = self.attention(encoded, encoded, encoded)
        # Aggregate
        return out.mean(dim=1)  # or take CLS token
```

## Performance Profiling

### Memory Usage

```bash
python -c "
import torch
from src.models.encoders import TextEncoder
from src.models.trajectory_models import TrajectoryEncoder

encoder = TextEncoder(latent_dim=128)
print(f'Text encoder: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M params')
"
```

### Training Speed

Monitor with:

```python
import time
start = time.time()
for epoch in range(num_epochs):
    train_epoch(...)
elapsed = time.time() - start
print(f"Time per epoch: {elapsed / num_epochs:.1f}s")
```

## Citation

If you use PHYSCLIP in your research, please cite:

```bibtex
@software{physclip2026,
  title={PHYSCLIP: Contrastive Learning for Physics Representations},
  author={Your Name},
  year={2026},
  url={https://github.com/spice14/PHYSCLIP}
}
```
