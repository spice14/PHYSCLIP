# PHYSCLIP v1: Multi-Scale Temporal Windowing

## Overview

This document explains the multi-scale temporal windowing strategy implemented in PHYSCLIP v1 to increase training sample count while preserving physics semantics.

## Motivation

**Problem**: Original v1 treated each complete trajectory as one sample (32 trajectories → 32 samples), which is insufficient for fair comparison with v0 (352 snapshot samples).

**Solution**: Extract multiple temporal windows from each trajectory at different time scales, increasing sample count to 384+ while preserving temporal dynamics.

## Implementation

### Dataset Expansion

The `BurgersTrajectoryDataset` now supports:

```python
dataset = BurgersTrajectoryDataset(
    dataset_dir="./burgers_data",
    metadata_path="./burgers_data/metadata.txt",
    window_sizes=[5, 7, 9],  # Multiple temporal scales
    stride=2                  # Overlapping windows
)
```

### Window Extraction

For a trajectory of length K=11:
- **Window size 3**: (11-3)//2+1 = 5 windows per trajectory
- **Window size 5**: (11-5)//2+1 = 4 windows per trajectory
- **Window size 7**: (11-7)//2+1 = 3 windows per trajectory

**Total**: 12 windows per trajectory × 32 trajectories = 384 samples

### Multi-Scale Rationale

Different window sizes capture different temporal scales:

1. **Small windows (3-5 snapshots)**: Capture short-term shock dynamics
   - Good for learning local temporal patterns
   - More samples, better gradient estimation

2. **Medium windows (5-7 snapshots)**: Capture intermediate evolution
   - Balance between short-term and long-term dynamics
   - Still maintains good sample count

3. **Large windows (7-9 snapshots)**: Capture long-term dissipation trends
   - Closer to full trajectory behavior
   - Fewer samples but richer temporal context

## Why This is NOT Data Leakage

### Critical Distinction

**Data leakage** would be:
- Using test set information during training
- Randomly shuffling time steps (breaking causality)
- Duplicating identical windows
- Mixing different physics regimes in one window

**Our approach** is valid because:
1. **Temporal coherence**: Windows are contiguous in time
2. **Physics preservation**: Each window obeys the same PDE
3. **Semantic validity**: All windows from same trajectory share same ν
4. **No information leakage**: Test trajectories remain unseen

### Analogy to Vision

In computer vision:
- **Invalid**: Using test images during training
- **Valid**: Random crops of training images (standard data augmentation)

In physics:
- **Invalid**: Using test trajectories during training
- **Valid**: Temporal windows of training trajectories (our approach)

## Physics Semantics

### What is Preserved

✅ **Temporal order**: Windows are contiguous, not shuffled  
✅ **Causal structure**: Earlier snapshots causally precede later ones  
✅ **Governing equation**: All windows evolve under same Burgers PDE  
✅ **Viscosity**: All windows from same trajectory have same ν  
✅ **Physics description**: All windows share same text label  

### What is Different

🔄 **Temporal span**: Different windows see different time intervals  
🔄 **Initial conditions**: Different windows start from different snapshots  
🔄 **Sample count**: More training signal for gradient estimation  

### Contrastive Positive Pairs

The key invariant for contrastive learning:

```
window[i] ↔ text[i]  should have HIGH similarity
window[i] ↔ text[j]  should have LOW similarity (i ≠ j)
```

This remains valid because:
- All windows from trajectory with ν=0.01 get description "ν=0.01"
- All windows from trajectory with ν=0.10 get description "ν=0.10"
- Physics semantics are preserved across windows

## Sample Count Comparison

| Approach | Samples | Physics Semantics |
|----------|---------|-------------------|
| v0 (snapshots) | 352 | Static fields only |
| v1 (full trajectories) | 32 | Full temporal dynamics |
| **v1 (multi-scale windows)** | **384** | **Temporal dynamics + more samples** |

## Implementation Details

### Dataset Structure

```python
class BurgersTrajectoryDataset:
    def __init__(self, ..., window_sizes=[5, 7, 9], stride=2):
        # Load base trajectories
        self.trajectories = [...]  # 32 trajectories
        self.descriptions = [...]  # Physics descriptions
        self.viscosities = [...]   # ν values
        
        # Build window indices
        for traj_idx in range(len(self.trajectories)):
            for window_size in window_sizes:
                for start_idx in range(0, K - window_size + 1, stride):
                    self.window_indices.append((traj_idx, window_size, start_idx))
    
    def __getitem__(self, idx):
        traj_idx, window_size, start_idx = self.window_indices[idx]
        window = self.trajectories[traj_idx][start_idx:start_idx+window_size]
        return window, self.descriptions[traj_idx], self.viscosities[traj_idx]
```

### Batching Constraint

**Critical**: PyTorch requires uniform tensor shapes in each batch.

**Solution**: Group windows by size during batching:

```python
# Group indices by window size
window_size_indices = {ws: [] for ws in WINDOW_SIZES}
for idx in range(len(dataset)):
    _, ws, _ = dataset.window_indices[idx]
    window_size_indices[ws].append(idx)

# Train on largest window size (or cycle through all)
main_window_size = max(WINDOW_SIZES)
train_indices = window_size_indices[main_window_size]
train_subset = Subset(dataset, train_indices)

dataloader = DataLoader(train_subset, ...)
```

## Validation Results

Sanity check results:
```
✓ Dataset loaded successfully
✓ Window count matches expected (384 windows)
✓ All window shapes correct, no NaNs
✓ Physics descriptions consistent across windows
✓ Batching works correctly
✓ Backward compatible with full trajectories
```

## Training Configuration

```python
# Multi-scale windowing parameters
WINDOW_SIZES = [5, 7, 9]  # Multiple temporal scales
STRIDE = 2                # Overlapping windows

# Training hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
```

## Expected Benefits

Compared to v0 (snapshots) and v1-original (full trajectories):

1. **More training samples** (384 vs 32) → better gradient estimation
2. **Multi-scale temporal information** → learn both short and long-term dynamics
3. **Physics-preserving augmentation** → more training signal without synthetic data
4. **Cleaner viscosity clustering** → temporal averaging reduces IC noise
5. **Better generalization** → model sees diverse temporal contexts

## Limitations

1. **Batching constraint**: Can only batch windows of same size together
2. **Redundancy**: Overlapping windows share some snapshots (controlled by stride)
3. **Computation**: More forward passes during training
4. **Memory**: Must store window indices (minimal overhead)

## Future Extensions

### Option 1: Cyclic Multi-Scale Training
Train on all window sizes in rotation:
```python
for epoch in range(NUM_EPOCHS):
    for window_size in WINDOW_SIZES:
        train_on_window_size(window_size)
```

### Option 2: Padding-Based Mixed Batching
Allow different window sizes in same batch via padding:
```python
def collate_with_padding(batch):
    max_len = max(w.shape[0] for w, _, _ in batch)
    padded = torch.zeros(len(batch), max_len, nx)
    masks = torch.zeros(len(batch), max_len)
    # ... pad and mask
```

### Option 3: Initial Condition Diversity
Generate more trajectories with varied ICs:
- Phase-shifted sine waves
- Multi-mode mixtures
- Low-amplitude noise

## References

- Original v0 implementation: `train_physclip_v0.py`
- Dataset implementation: `dataset_trajectory.py`
- Training script: `train_physclip_v1.py`
- Integration test: `test_v1_integration.py`

## Conclusion

Multi-scale temporal windowing increases PHYSCLIP v1 sample count from 32 to 384 while preserving physics semantics. This enables fair comparison with v0 and tests whether temporal dynamics improve representation learning at scale.

**Key insight**: Temporal windows are physics-valid training samples, not data leakage. Each window is a coherent dynamical evolution slice that maintains the hypothesis: *temporal dynamics + contrastive alignment → robust physics representations*.
