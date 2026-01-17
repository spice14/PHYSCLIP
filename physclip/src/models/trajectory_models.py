"""
Trajectory encoder for PHYSCLIP v1.

CRITICAL SEMANTIC DIFFERENCE FROM v0:
- v0 FieldEncoder: encodes individual snapshots u(x, t_i)
- v1 TrajectoryEncoder: encodes ENTIRE trajectories {u(x, t_1), ..., u(x, t_K)}

This enables learning from DYNAMICAL EVOLUTION, not just static field patterns.

Architecture rationale:
- Stage A (Snapshot Encoding): Apply FieldEncoder to each time step with weight sharing
- Stage B (Temporal Aggregation): Pool encoded snapshots via mean pooling

Why mean pooling?
- Permutation-invariant baseline (minimal inductive bias)
- No positional encoding needed
- Easy to swap with attention/transformers later
- Captures "average physics behavior" over trajectory

What this encoder learns:
- Field patterns that persist across time
- Regime-level features (e.g., viscosity effects on smoothing rate)
- Temporal invariances (what matters regardless of when)

What it does NOT do:
- Predict future states (no temporal decoder)
- Solve PDEs (no residual enforcement)
- Model sequential dependencies (no RNN/attention in v1)

Future extensions (not implemented yet):
- Attention pooling over time (learned importance weights)
- Temporal transformers (self-attention across snapshots)
- Physics-aware pooling (e.g., weight early vs late time differently)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryEncoder(nn.Module):
    """
    Encode entire trajectories into fixed-dimensional latent representations.
    
    Converts a temporal sequence of field snapshots into a single vector
    suitable for CLIP-style contrastive learning with physics descriptions.
    
    Architecture:
        1. Snapshot Encoder: FieldEncoder applied to each time step (weight sharing)
        2. Temporal Aggregation: Mean pooling over time dimension
        3. L2 Normalization: For contrastive loss stability
    
    Parameters
    ----------
    field_encoder : nn.Module
        Pre-configured FieldEncoder instance for snapshot encoding.
        Weight sharing: same encoder applied to all time steps.
    normalize_output : bool
        If True, L2-normalize final trajectory embedding (recommended).
        Note: Snapshot embeddings are already normalized by FieldEncoder.
    
    Input Shape
    -----------
    trajectories : torch.Tensor
        Shape (batch_size, K, nx) where:
        - batch_size: number of trajectories
        - K: number of time snapshots
        - nx: spatial grid points
    
    Output Shape
    ------------
    z_traj : torch.Tensor
        Shape (batch_size, latent_dim)
        L2-normalized trajectory embeddings
    
    Design Constraints
    ------------------
    - Temporal order is preserved during encoding but discarded during pooling
    - No learnable temporal weights (v1 baseline = uniform mean)
    - Assumes all trajectories have same length K (no padding/masking)
    - Snapshot normalization handled by dataset, not encoder
    
    Extension Points
    ----------------
    - Replace _aggregate_temporal() to use attention, transformers, RNNs
    - Add positional encoding before snapshot encoding
    - Implement weighted pooling based on physics priors
    """
    
    def __init__(self, field_encoder, normalize_output=True):
        """
        Initialize trajectory encoder.
        
        Parameters
        ----------
        field_encoder : nn.Module
            FieldEncoder instance (already configured with nx_input, latent_dim).
        normalize_output : bool
            Whether to L2-normalize final trajectory embeddings.
        """
        super().__init__()
        
        self.field_encoder = field_encoder
        self.normalize_output = normalize_output
        self.latent_dim = field_encoder.latent_dim
        
    def _encode_snapshots(self, trajectories):
        """
        Apply snapshot encoder to each time step with weight sharing.
        
        Parameters
        ----------
        trajectories : torch.Tensor
            Shape (batch_size, K, nx)
        
        Returns
        -------
        snapshot_embeddings : torch.Tensor
            Shape (batch_size, K, latent_dim)
            Encoded snapshots at each time step
        
        Notes
        -----
        Weight sharing: same FieldEncoder applied to all K snapshots.
        This is critical for:
        - Parameter efficiency (one encoder, not K encoders)
        - Temporal consistency (snapshots treated uniformly)
        - Transfer learning (encoder generalizes across time)
        """
        batch_size, K, nx = trajectories.shape
        
        # Reshape: (batch_size, K, nx) → (batch_size * K, nx)
        # This allows batch processing all snapshots at once
        snapshots_flat = trajectories.view(batch_size * K, nx)
        
        # Encode all snapshots: (batch_size * K, nx) → (batch_size * K, latent_dim)
        embeddings_flat = self.field_encoder(snapshots_flat)
        
        # Reshape back: (batch_size * K, latent_dim) → (batch_size, K, latent_dim)
        snapshot_embeddings = embeddings_flat.view(batch_size, K, self.latent_dim)
        
        return snapshot_embeddings
    
    def _aggregate_temporal(self, snapshot_embeddings):
        """
        Aggregate snapshot embeddings across time dimension.
        
        Parameters
        ----------
        snapshot_embeddings : torch.Tensor
            Shape (batch_size, K, latent_dim)
        
        Returns
        -------
        trajectory_embedding : torch.Tensor
            Shape (batch_size, latent_dim)
        
        Notes
        -----
        v1 baseline: Simple mean pooling.
        
        Mean pooling properties:
        - Permutation invariant (order doesn't matter after encoding)
        - No learnable parameters (minimal inductive bias)
        - Captures "average behavior" across trajectory
        
        Future extensions (replace this method):
        - Attention pooling: weighted_sum = Σ_t α_t * z_t
        - Max pooling: select most salient time step
        - Learnable pooling: trainable aggregation weights
        - Temporal transformer: self-attention across time
        
        Why mean pooling works for physics:
        - Physics descriptions refer to governing equations (time-averaged behavior)
        - Viscosity effects manifest consistently across trajectory
        - Regime-level features are temporally persistent
        """
        # Mean over time dimension: (batch_size, K, latent_dim) → (batch_size, latent_dim)
        trajectory_embedding = snapshot_embeddings.mean(dim=1)
        
        return trajectory_embedding
    
    def forward(self, trajectories):
        """
        Encode trajectories to latent representations.
        
        Parameters
        ----------
        trajectories : torch.Tensor
            Shape (batch_size, K, nx)
            Temporal sequences of field snapshots
        
        Returns
        -------
        z_traj : torch.Tensor
            Shape (batch_size, latent_dim)
            L2-normalized trajectory embeddings
        
        Pipeline
        --------
        1. Encode snapshots: (B, K, nx) → (B, K, latent_dim)
        2. Aggregate temporal: (B, K, latent_dim) → (B, latent_dim)
        3. L2 normalize: (B, latent_dim) → (B, latent_dim) with unit norm
        """
        # Stage A: Snapshot encoding with weight sharing
        # Shape: (batch_size, K, nx) → (batch_size, K, latent_dim)
        snapshot_embeddings = self._encode_snapshots(trajectories)
        
        # Stage B: Temporal aggregation via mean pooling
        # Shape: (batch_size, K, latent_dim) → (batch_size, latent_dim)
        trajectory_embedding = self._aggregate_temporal(snapshot_embeddings)
        
        # L2 normalization for contrastive learning
        # Note: Snapshot embeddings are already normalized by FieldEncoder,
        # but we renormalize after pooling to ensure unit norm
        if self.normalize_output:
            trajectory_embedding = F.normalize(trajectory_embedding, p=2, dim=-1)
        
        return trajectory_embedding


def create_trajectory_encoder(nx_input=256, latent_dim=128, normalize_output=True):
    """
    Factory function to create a TrajectoryEncoder with default FieldEncoder.
    
    Parameters
    ----------
    nx_input : int
        Spatial grid size for field encoder.
    latent_dim : int
        Dimension of latent representation.
    normalize_output : bool
        Whether to L2-normalize outputs.
    
    Returns
    -------
    encoder : TrajectoryEncoder
        Ready-to-use trajectory encoder.
    """
    from encoders import FieldEncoder
    
    # Create snapshot encoder
    field_encoder = FieldEncoder(
        nx_input=nx_input,
        latent_dim=latent_dim,
        normalize_output=True  # Normalize snapshots before pooling
    )
    
    # Wrap in trajectory encoder
    trajectory_encoder = TrajectoryEncoder(
        field_encoder=field_encoder,
        normalize_output=normalize_output
    )
    
    return trajectory_encoder


# ============================================================================
# Sanity Check
# ============================================================================

if __name__ == "__main__":
    """
    Minimal sanity check for TrajectoryEncoder.
    
    Verifies:
    - Input/output shapes are correct
    - Forward pass executes without errors
    - Output embeddings are L2-normalized
    """
    import sys
    sys.path.append('..')
    
    from encoders import FieldEncoder
    
    print("="*70)
    print("TrajectoryEncoder Sanity Check")
    print("="*70)
    print()
    
    # Configuration
    batch_size = 2
    K = 11  # Number of time snapshots
    nx = 256  # Spatial grid points
    latent_dim = 128
    
    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Time snapshots: {K}")
    print(f"  Spatial points: {nx}")
    print(f"  Latent dim: {latent_dim}")
    print()
    
    # Create encoder
    field_encoder = FieldEncoder(
        nx_input=nx,
        latent_dim=latent_dim,
        normalize_output=True
    )
    trajectory_encoder = TrajectoryEncoder(
        field_encoder=field_encoder,
        normalize_output=True
    )
    
    print("Model Statistics:")
    total_params = sum(p.numel() for p in trajectory_encoder.parameters())
    trainable_params = sum(p.numel() for p in trajectory_encoder.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()
    
    # Create dummy trajectory batch
    # Shape: (batch_size, K, nx)
    trajectories = torch.randn(batch_size, K, nx)
    
    print("Input:")
    print(f"  Shape: {trajectories.shape}")
    print(f"  Min: {trajectories.min().item():.4f}")
    print(f"  Max: {trajectories.max().item():.4f}")
    print(f"  Mean: {trajectories.mean().item():.4f}")
    print()
    
    # Forward pass
    with torch.no_grad():
        embeddings = trajectory_encoder(trajectories)
    
    print("Output:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Expected shape: ({batch_size}, {latent_dim})")
    print(f"  Shape match: {embeddings.shape == (batch_size, latent_dim)}")
    print()
    
    # Verify L2 normalization
    norms = embeddings.norm(dim=-1)
    print("L2 Normalization Check:")
    print(f"  Embedding norms: {norms.tolist()}")
    print(f"  All close to 1.0: {torch.allclose(norms, torch.ones_like(norms), atol=1e-6)}")
    print()
    
    # Check for NaNs
    has_nan = torch.isnan(embeddings).any()
    print("NaN Check:")
    print(f"  Contains NaN: {has_nan}")
    print()
    
    if embeddings.shape == (batch_size, latent_dim) and not has_nan:
        print("="*70)
        print("Sanity check PASSED")
        print("="*70)
    else:
        print("="*70)
        print("Sanity check FAILED")
        print("="*70)
        sys.exit(1)
