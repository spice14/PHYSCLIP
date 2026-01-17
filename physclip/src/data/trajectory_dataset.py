"""
Trajectory-level dataset for PHYSCLIP v1 with multi-scale temporal windowing.

CRITICAL SEMANTIC DIFFERENCE FROM v0:
- v0: Each snapshot u(x, t_i) is an independent training sample
- v1: Each temporal window is ONE sample (preserving temporal dynamics)

MULTI-SCALE WINDOWING RATIONALE:
- Original v1: Each trajectory (K, nx) → 1 sample (too few for training)
- Windowed v1: Each trajectory → multiple windows at different time scales
- Physics-preserving: windows are contiguous in time, no temporal shuffling
- NOT data leakage: encoder sees coherent dynamical evolution, not isolated frames
- Increases sample count while maintaining temporal semantics

Design rationale:
- Contrastive learning at window level: align temporal windows with descriptions
- Multiple window sizes capture multi-scale temporal dynamics
- Overlapping windows provide more training signal without data augmentation
- All windows from same trajectory share same physics (ν, equation, BC)

Data format:
- Each .npy file contains one trajectory of shape (K, nx)
- Metadata maps trajectory_id -> (filepath, physics_description)
- Viscosity ν is stored as a float field (no string parsing in training loops)
- Windows extracted deterministically (no randomness in __getitem__)

Why this is NOT data leakage:
- Each window is a coherent temporal slice with valid physics
- The model learns from temporal evolution (what v1 is designed for)
- Unlike image crops, temporal windows preserve causal structure
- Positive pairs remain semantically correct: (window_i ↔ text_i)
"""

import numpy as np
import csv
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


class BurgersTrajectoryDataset(Dataset):
    """
    Dataset for PHYSCLIP v1: trajectory-level contrastive learning with multi-scale windowing.
    
    Each sample is a temporal window from a trajectory: a contiguous slice of u(x, t).
    Multiple windows are extracted from each trajectory at different time scales.
    All windows from the same trajectory share the same physics description and viscosity.
    
    Multi-scale windowing increases sample count while preserving physics semantics:
    - Small windows (e.g., 3 snapshots): capture short-term dynamics
    - Medium windows (e.g., 5 snapshots): capture intermediate evolution
    - Large windows (e.g., 7 snapshots): capture long-term trends
    
    Design constraints:
    - Deterministic and reproducible (no randomness in __getitem__)
    - All file I/O happens in __init__ (O(1) access in __getitem__)
    - Viscosity stored explicitly as float (no string parsing)
    - Compatible with v0 data structure without regeneration
    - Windows are contiguous in time (no temporal shuffling)
    - No duplicate windows (controlled by stride)
    
    Parameters
    ----------
    dataset_dir : str
        Directory containing trajectory .npy files and metadata.txt
    metadata_path : str
        Path to metadata CSV file
    window_sizes : List[int], optional
        List of temporal window sizes to extract (default: [11] = full trajectory)
        Example: [3, 5, 7] extracts windows of 3, 5, and 7 snapshots
    stride : int, optional
        Stride for sliding window extraction (default: 1)
        stride=1: maximum overlap, more samples
        stride=window_size: no overlap, fewer samples
    
    Returns
    -------
    Each __getitem__(idx) returns:
        window : torch.Tensor
            Shape (window_size, nx) - a temporal slice of the trajectory
        description : str
            Physics description (same for all windows from same trajectory)
        viscosity : float
            Viscosity coefficient ν (same for all windows from same trajectory)
    
    Notes
    -----
    Dataset length increases multiplicatively with number of window sizes and stride.
    For a trajectory of length K:
        num_windows_per_size = (K - window_size) // stride + 1
        total_windows = sum(num_windows_per_size for each window_size)
    
    Example:
        K=11, window_sizes=[3, 5, 7], stride=1
        Windows of size 3: (11-3)//1+1 = 9
        Windows of size 5: (11-5)//1+1 = 7
        Windows of size 7: (11-7)//1+1 = 5
        Total per trajectory: 9+7+5 = 21 windows
        For 32 trajectories: 32*21 = 672 samples
    """
    
    def __init__(self, dataset_dir, metadata_path, window_sizes=None, stride=1):
        """Load all trajectory files and prepare windowing indices."""
        self.dataset_dir = dataset_dir
        self.window_sizes = window_sizes if window_sizes is not None else [11]  # Default: full trajectory
        self.stride = stride
        
        # Storage for base trajectories
        self.trajectories = []
        self.descriptions = []
        self.viscosities = []
        
        # Storage for window indices (trajectory_idx, window_size, start_idx)
        self.window_indices = []
        
        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trajectory_file = os.path.join(dataset_dir, row['filepath'])
                physics_desc = row['physics_description']
                
                # Extract viscosity from description (done ONCE at load time, not in training loop)
                # This is fragile but unavoidable given v0 data format
                # v1 data should store viscosity in metadata directly
                try:
                    nu_str = physics_desc.split("ν=")[1].strip(".")
                    nu = float(nu_str)
                except (IndexError, ValueError) as e:
                    raise ValueError(
                        f"Failed to parse viscosity from description: {physics_desc}\n"
                        f"Error: {e}\n"
                        "v1 dataset requires viscosity to be extractable from description."
                    )
                
                # Load ENTIRE trajectory as one sample
                # Shape: (n_snapshots, nx)
                if not os.path.exists(trajectory_file):
                    raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
                
                trajectory = np.load(trajectory_file)
                
                # Validate shape
                if trajectory.ndim != 2:
                    raise ValueError(
                        f"Trajectory file {trajectory_file} has invalid shape {trajectory.shape}. "
                        f"Expected 2D array (n_snapshots, nx)."
                    )
                
                # Store trajectory (entire temporal sequence)
                self.trajectories.append(trajectory)
                self.descriptions.append(physics_desc)
                self.viscosities.append(nu)
        
        # Validate dataset is non-empty
        if len(self.trajectories) == 0:
            raise ValueError("Dataset is empty. No trajectories loaded.")
        
        # Validate all trajectories have same shape (for batching)
        traj_shapes = [traj.shape for traj in self.trajectories]
        if len(set(traj_shapes)) > 1:
            raise ValueError(
                f"All trajectories must have the same shape for batching. "
                f"Found shapes: {set(traj_shapes)}\n"
                "To support variable-length trajectories, implement padding in collate_fn."
            )
        
        self.n_snapshots, self.nx = self.trajectories[0].shape
        
        # Validate window sizes
        for ws in self.window_sizes:
            if ws > self.n_snapshots:
                raise ValueError(
                    f"Window size {ws} exceeds trajectory length {self.n_snapshots}. "
                    f"All window sizes must be <= {self.n_snapshots}."
                )
            if ws < 1:
                raise ValueError(f"Window size must be positive, got {ws}")
        
        if self.stride < 1:
            raise ValueError(f"Stride must be positive, got {self.stride}")
        
        # Build window indices
        # For each trajectory, extract all windows at all scales
        for traj_idx in range(len(self.trajectories)):
            for window_size in self.window_sizes:
                # Slide window across time dimension
                num_windows = (self.n_snapshots - window_size) // self.stride + 1
                for i in range(num_windows):
                    start_idx = i * self.stride
                    # Store (trajectory_idx, window_size, start_idx)
                    self.window_indices.append((traj_idx, window_size, start_idx))
        
        # Report statistics
        print(f"Loaded {len(self.trajectories)} base trajectories")
        print(f"Trajectory shape: ({self.n_snapshots}, {self.nx})")
        print(f"Window sizes: {self.window_sizes}")
        print(f"Stride: {self.stride}")
        print(f"Total windows: {len(self.window_indices)}")
        print(f"Unique physics descriptions: {len(set(self.descriptions))}")
        print(f"Viscosity range: [{min(self.viscosities):.3f}, {max(self.viscosities):.3f}]")
        
        # Window distribution
        window_counts = {}
        for _, ws, _ in self.window_indices:
            window_counts[ws] = window_counts.get(ws, 0) + 1
        print(f"Window distribution: {window_counts}")
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        """
        Return one temporal window from a trajectory.
        
        Parameters
        ----------
        idx : int
            Window index (not trajectory index)
        
        Returns
        -------
        window : torch.Tensor
            Shape (window_size, nx) containing temporal window
        description : str
            Physics description (same for all windows from same trajectory)
        viscosity : float
            Viscosity coefficient ν (same for all windows from same trajectory)
        """
        # Get window metadata
        traj_idx, window_size, start_idx = self.window_indices[idx]
        
        # Extract trajectory
        trajectory = self.trajectories[traj_idx]
        
        # Extract window
        end_idx = start_idx + window_size
        window = trajectory[start_idx:end_idx]
        
        # Normalize window snapshots for stability
        # Each snapshot gets zero mean, unit std
        # This preserves temporal structure while stabilizing training
        window_normalized = np.zeros_like(window)
        for t in range(window.shape[0]):
            snapshot = window[t]
            window_normalized[t] = (snapshot - snapshot.mean()) / (snapshot.std() + 1e-8)
        
        window_tensor = torch.tensor(window_normalized, dtype=torch.float32)
        description = self.descriptions[traj_idx]
        viscosity = self.viscosities[traj_idx]
        
        return window_tensor, description, viscosity


def collate_trajectory_batch(batch):
    """
    Collate function for trajectory window batching with variable-length support.
    
    Windows may have different temporal sizes ([5, 7, 9]). This function
    pads them to a uniform size for batching.
    
    Parameters
    ----------
    batch : list of tuples
        Each tuple: (window_tensor, description_str, viscosity_float)
    
    Returns
    -------
    windows : torch.Tensor
        Shape (batch_size, max_window_size, nx) - padded windows
    texts : list of str
        Physics descriptions (batch_size,)
    viscosities : torch.Tensor
        Shape (batch_size,) viscosity values
    """
    windows, texts, viscosities = zip(*batch)
    
    # Find max temporal size for padding
    max_window_size = max(w.shape[0] for w in windows)
    
    # Pad windows to (max_window_size, nx)
    padded_windows = []
    for w in windows:
        if w.shape[0] < max_window_size:
            # Pad along temporal dimension
            padding = max_window_size - w.shape[0]
            padded = torch.nn.functional.pad(w, (0, 0, 0, padding), mode='constant', value=0)
            padded_windows.append(padded)
        else:
            padded_windows.append(w)
    
    # Stack padded windows: (batch_size, max_window_size, nx)
    windows_tensor = torch.stack(padded_windows)
    
    # Viscosities: (batch_size,)
    viscosities_tensor = torch.tensor(viscosities, dtype=torch.float32)
    
    # Texts remain as list (for text encoder)
    texts_list = list(texts)
    
    return windows_tensor, texts_list, viscosities_tensor


# ============================================================================
# Sanity Check
# ============================================================================

if __name__ == "__main__":
    """
    Multi-scale windowing sanity check.
    
    Tests:
    1. Dataset loads correctly with multiple window sizes
    2. Window extraction is deterministic
    3. All windows have correct shapes
    4. No NaNs after normalization
    5. Windows from same trajectory share same physics description
    6. Collate function handles batching correctly
    
    Run with:
        python dataset_trajectory.py
    """
    import sys
    
    DATASET_DIR = "./burgers_data"
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.txt")
    
    print("="*70)
    print("PHYSCLIP v1: Multi-Scale Trajectory Windowing Sanity Check")
    print("="*70)
    print()
    
    # Test 1: Load dataset with multiple window sizes
    print("Test 1: Loading dataset with multi-scale windows...")
    try:
        dataset = BurgersTrajectoryDataset(
            DATASET_DIR, 
            METADATA_PATH,
            window_sizes=[3, 5, 7],
            stride=2
        )
    except Exception as e:
        print(f"ERROR: Failed to load dataset")
        print(f"  {type(e).__name__}: {e}")
        sys.exit(1)
    
    print("✓ Dataset loaded successfully")
    print()
    
    # Test 2: Verify dataset statistics
    print("Test 2: Dataset Statistics")
    print(f"  Base trajectories: {len(dataset.trajectories)}")
    print(f"  Total windows: {len(dataset)}")
    print(f"  Trajectory shape: ({dataset.n_snapshots}, {dataset.nx})")
    print(f"  Window sizes: {dataset.window_sizes}")
    print(f"  Stride: {dataset.stride}")
    print(f"  Unique descriptions: {len(set(dataset.descriptions))}")
    
    # Calculate expected number of windows
    expected_windows = 0
    for ws in dataset.window_sizes:
        num_windows_per_traj = (dataset.n_snapshots - ws) // dataset.stride + 1
        expected_windows += num_windows_per_traj * len(dataset.trajectories)
        print(f"  Windows of size {ws}: {num_windows_per_traj} per trajectory")
    
    print(f"  Expected total: {expected_windows}")
    print(f"  Actual total: {len(dataset)}")
    assert len(dataset) == expected_windows, "Window count mismatch!"
    print("✓ Window count matches expected")
    print()
    
    # Test 3: Sample windows and check shapes
    print("Test 3: Window Shape Validation")
    window_size_samples = {ws: [] for ws in dataset.window_sizes}
    
    # Collect samples for each window size
    for idx in range(len(dataset)):
        _, ws, _ = dataset.window_indices[idx]
        if len(window_size_samples[ws]) < 3:  # Collect 3 samples per size
            window, desc, nu = dataset[idx]
            window_size_samples[ws].append((window, desc, nu))
    
    for ws, samples in window_size_samples.items():
        print(f"  Window size {ws}:")
        for i, (window, desc, nu) in enumerate(samples):
            assert window.shape == (ws, dataset.nx), f"Shape mismatch: {window.shape}"
            assert not torch.isnan(window).any(), f"NaNs found in window {i}"
            print(f"    Sample {i}: shape={window.shape}, ν={nu:.3f}, no NaNs ✓")
    
    print("✓ All window shapes correct, no NaNs")
    print()
    
    # Test 4: Verify windows from same trajectory share same description
    print("Test 4: Physics Description Consistency")
    trajectory_windows = {}  # traj_idx -> [(window_idx, desc, nu), ...]
    
    for idx in range(min(50, len(dataset))):  # Check first 50 windows
        traj_idx, _, _ = dataset.window_indices[idx]
        window, desc, nu = dataset[idx]
        
        if traj_idx not in trajectory_windows:
            trajectory_windows[traj_idx] = []
        trajectory_windows[traj_idx].append((idx, desc, nu))
    
    for traj_idx, windows in trajectory_windows.items():
        descriptions = [desc for _, desc, _ in windows]
        viscosities = [nu for _, _, nu in windows]
        
        assert len(set(descriptions)) == 1, f"Description mismatch for traj {traj_idx}"
        assert len(set(viscosities)) == 1, f"Viscosity mismatch for traj {traj_idx}"
    
    print(f"  Checked {len(trajectory_windows)} trajectories")
    print("✓ All windows from same trajectory share same physics")
    print()
    
    # Test 5: DataLoader batching
    print("Test 5: DataLoader Batching")
    from torch.utils.data import DataLoader
    
    # Create subset with uniform window size for batching
    # (In practice, use filtered datasets or custom samplers)
    indices_ws3 = [i for i, (_, ws, _) in enumerate(dataset.window_indices) if ws == 3]
    from torch.utils.data import Subset
    
    subset = Subset(dataset, indices_ws3[:16])  # 16 windows of size 3
    
    dataloader = DataLoader(
        subset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_trajectory_batch,
        num_workers=0
    )
    
    batch_windows, batch_texts, batch_nus = next(iter(dataloader))
    print(f"  Batch windows shape: {batch_windows.shape}")
    print(f"  Batch texts length: {len(batch_texts)}")
    print(f"  Batch viscosities shape: {batch_nus.shape}")
    print(f"  Batch viscosities: {batch_nus.tolist()}")
    
    assert batch_windows.shape == (4, 3, dataset.nx), "Batch shape mismatch"
    assert len(batch_texts) == 4, "Text count mismatch"
    assert batch_nus.shape == (4,), "Viscosity shape mismatch"
    print("✓ Batching works correctly")
    print()
    
    # Test 6: Test with full trajectories (backward compatibility)
    print("Test 6: Backward Compatibility (Full Trajectories)")
    dataset_full = BurgersTrajectoryDataset(
        DATASET_DIR,
        METADATA_PATH,
        window_sizes=[11],  # Full trajectory length
        stride=1
    )
    
    print(f"  Full trajectory dataset size: {len(dataset_full)}")
    assert len(dataset_full) == len(dataset_full.trajectories), "Should have one sample per trajectory"
    
    traj, desc, nu = dataset_full[0]
    print(f"  Sample shape: {traj.shape}")
    assert traj.shape[0] == 11, "Full trajectory should have all 11 snapshots"
    print("✓ Backward compatible with full trajectories")
    print()
    
    print("="*70)
    print("All tests PASSED ✓")
    print("="*70)
    print()
    print("Summary:")
    print(f"  Multi-scale windowing works correctly")
    print(f"  {len(dataset)} windows from {len(dataset.trajectories)} trajectories")
    print(f"  Window sizes: {dataset.window_sizes}")
    print(f"  All windows preserve physics semantics")
    print(f"  Ready for v1 training with increased sample count")

