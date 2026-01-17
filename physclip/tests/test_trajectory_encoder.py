"""
End-to-end integration test for PHYSCLIP v1 trajectory pipeline.

Tests the full data flow:
    BurgersTrajectoryDataset → TrajectoryEncoder → embeddings

Verifies:
- Dataset and encoder shape compatibility
- Batch processing works correctly
- Real data (not synthetic) encodes without errors
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath('..'))

import torch
from torch.utils.data import DataLoader

# Import from root-level modules
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataset_trajectory import BurgersTrajectoryDataset, collate_trajectory_batch
from models.encoders import FieldEncoder
from models.trajectory_encoder import TrajectoryEncoder


def main():
    print("="*70)
    print("PHYSCLIP v1: End-to-End Integration Test")
    print("="*70)
    print()
    
    # Configuration
    DATASET_DIR = "./burgers_data"
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.txt")
    BATCH_SIZE = 4
    LATENT_DIM = 128
    
    # Load dataset
    print("Loading trajectory dataset...")
    try:
        dataset = BurgersTrajectoryDataset(DATASET_DIR, METADATA_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Make sure you run this from the PHYSCLIP root directory.")
        return 1
    print()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_trajectory_batch,
        num_workers=0  # 0 for Windows compatibility
    )
    
    # Create encoder
    print("Creating trajectory encoder...")
    field_encoder = FieldEncoder(
        nx_input=dataset.nx,
        latent_dim=LATENT_DIM,
        normalize_output=True
    )
    trajectory_encoder = TrajectoryEncoder(
        field_encoder=field_encoder,
        normalize_output=True
    )
    
    total_params = sum(p.numel() for p in trajectory_encoder.parameters())
    print(f"  Parameters: {total_params:,}")
    print()
    
    # Process one batch
    print("Processing batch...")
    trajectories, descriptions, viscosities = next(iter(dataloader))
    
    print("Batch shapes:")
    print(f"  Trajectories: {trajectories.shape}")
    print(f"  Descriptions: {len(descriptions)} strings")
    print(f"  Viscosities: {viscosities.shape}")
    print()
    
    # Encode trajectories
    with torch.no_grad():
        embeddings = trajectory_encoder(trajectories)
    
    print("Encoded embeddings:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Expected: ({BATCH_SIZE}, {LATENT_DIM})")
    print(f"  Match: {embeddings.shape == (BATCH_SIZE, LATENT_DIM)}")
    print()
    
    # Verify L2 normalization
    norms = embeddings.norm(dim=-1)
    print("L2 Normalization:")
    print(f"  Min norm: {norms.min().item():.6f}")
    print(f"  Max norm: {norms.max().item():.6f}")
    print(f"  Mean norm: {norms.mean().item():.6f}")
    print(f"  All unit norm: {torch.allclose(norms, torch.ones_like(norms), atol=1e-5)}")
    print()
    
    # Check correspondence with viscosities
    print("Sample alignment:")
    for i in range(min(3, BATCH_SIZE)):
        print(f"  Sample {i}: ν={viscosities[i].item():.3f}")
        print(f"    Embedding norm: {embeddings[i].norm().item():.6f}")
        print(f"    Description: {descriptions[i][:60]}...")
    print()
    
    # Test multiple batches
    print("Processing full dataset...")
    all_embeddings = []
    all_viscosities = []
    
    for trajectories_batch, descs_batch, nus_batch in dataloader:
        with torch.no_grad():
            emb_batch = trajectory_encoder(trajectories_batch)
        all_embeddings.append(emb_batch)
        all_viscosities.append(nus_batch)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_viscosities = torch.cat(all_viscosities, dim=0)
    
    print(f"  Total trajectories encoded: {len(all_embeddings)}")
    print(f"  Embedding matrix shape: {all_embeddings.shape}")
    print(f"  Viscosity range: [{all_viscosities.min().item():.3f}, {all_viscosities.max().item():.3f}]")
    print()
    
    # Verify no NaNs
    has_nan = torch.isnan(all_embeddings).any()
    print(f"NaN check: {'FAILED - contains NaN!' if has_nan else 'PASSED'}")
    print()
    
    if (embeddings.shape == (BATCH_SIZE, LATENT_DIM) and 
        not has_nan and
        torch.allclose(norms, torch.ones_like(norms), atol=1e-5)):
        print("="*70)
        print("Integration test PASSED")
        print("="*70)
        return 0
    else:
        print("="*70)
        print("Integration test FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
