"""
PHYSCLIP v0: Contrastive Physics Representation Learning

Train encoders to align physical field snapshots with physics text descriptions
using a CLIP-style contrastive objective. This tests whether physics regimes
are representationally learnable.

Success criteria:
- Field embeddings cluster by viscosity regardless of initial condition
- Nearest text neighbor corresponds to correct viscosity regime
- Contrastive loss decreases consistently during training
"""

import numpy as np
import csv
import os
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no window creation
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import PHYSCLIP components
from src.models import FieldEncoder, ContrastiveLoss
from src.models.encoders import TextEncoder, create_encoders


# ============================================================================
# Configuration
# ============================================================================

DATASET_DIR = os.path.join(os.path.dirname(__file__), "../data/burgers_data")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.txt")

# Model hyperparameters
NX_INPUT = 256
LATENT_DIM = 128
NORMALIZE_OUTPUT = True

# Training hyperparameters
NUM_EPOCHS = 100  # Longer training to reach convergence
BATCH_SIZE = 8  # Smaller batch for better positive:negative ratio
LEARNING_RATE = 2.5e-3  # Higher learning rate for faster convergence
TEMPERATURE = 0.07  # Temperature for contrastive loss

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Dataset
# ============================================================================

class BurgersDataset(Dataset):
    """
    Dataset for PHYSCLIP v0.
    
    Each sample is a single field snapshot u(x) paired with its physics
    description string. Multiple snapshots may share the same description.
    
    Viscosity ν is stored explicitly to avoid brittle string parsing.
    """
    
    def __init__(self, dataset_dir, metadata_path):
        """Load all trajectories and flatten into individual snapshots."""
        self.dataset_dir = dataset_dir
        self.snapshots = []
        self.descriptions = []
        self.viscosities = []  # Store viscosity explicitly, not via parsing
        
        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trajectory_file = os.path.join(dataset_dir, row['filepath'])
                physics_desc = row['physics_description']
                
                # Extract viscosity from description (done once at load time)
                nu_str = physics_desc.split("ν=")[1].strip(".")
                nu = float(nu_str)
                
                # Load trajectory (shape: n_snapshots x nx)
                trajectory = np.load(trajectory_file)
                
                # Each snapshot becomes an independent sample
                for snapshot in trajectory:
                    self.snapshots.append(snapshot)
                    self.descriptions.append(physics_desc)
                    self.viscosities.append(nu)
        
        assert len(self.snapshots) > 0, "Dataset is empty"
        print(f"Loaded {len(self.snapshots)} snapshots from {len(self.snapshots) // 11} trajectories")
    
    def __len__(self):
        return len(self.snapshots)
    
    def __getitem__(self, idx):
        """Return (field_snapshot, description_string, viscosity)."""
        snapshot = self.snapshots[idx]
        # Normalize field snapshot for stability: zero mean, unit std
        snapshot = (snapshot - snapshot.mean()) / (snapshot.std() + 1e-8)
        snapshot = torch.tensor(snapshot, dtype=torch.float32)
        description = self.descriptions[idx]
        viscosity = self.viscosities[idx]
        return snapshot, description, viscosity


def collate_fn(batch):
    """
    Custom collate to handle text strings and viscosity values.
    
    Returns:
        fields: tensor of shape (batch_size, nx)
        texts: list of strings
        viscosities: tensor of shape (batch_size,)
    """
    fields, texts, viscosities = zip(*batch)
    fields = torch.stack(fields)
    viscosities = torch.tensor(viscosities, dtype=torch.float32)
    return fields, list(texts), viscosities


# ============================================================================
# Training
# ============================================================================

def train_epoch(field_encoder, text_encoder, criterion, dataloader, optimizer, device):
    """
    Train for one epoch.
    
    INVARIANTS:
    - Encoders output L2-normalized embeddings (enforced in encoder forward pass)
    - Contrastive loss assumes (field[i] ↔ text[i]) positive pairs
    - This trains representation alignment, NOT prediction or physics solving
    - Text embeddings are FROZEN — only field encoder is trained
    """
    field_encoder.train()
    text_encoder.eval()  # Text encoder always in eval mode (frozen)
    
    total_loss = 0.0
    num_batches = 0
    num_nan_batches = 0
    
    for fields, texts, viscosities in dataloader:
        # Move fields to device (viscosity unused during training but kept for consistency)
        fields = fields.to(device)
        
        # Encode
        z_phys = field_encoder(fields)
        z_text = text_encoder(texts)
        
        # Sanity check: batch sizes must match for contrastive loss
        assert z_phys.shape[0] == z_text.shape[0], "Batch size mismatch"
        
        # Debug: check for issues in embeddings before loss
        if torch.isnan(z_phys).any():
            print("WARNING: NaN in z_phys. Skipping batch.")
            continue
        if torch.isnan(z_text).any():
            print("WARNING: NaN in z_text. Skipping batch.")
            continue
        if (z_phys.norm(dim=-1) == 0).any():
            print("WARNING: Zero-norm z_phys. Skipping batch.")
            continue
        if (z_text.norm(dim=-1) == 0).any():
            print("WARNING: Zero-norm z_text. Skipping batch.")
            continue
        
        # Compute loss
        loss = criterion(z_phys, z_text)
        
        # Check for NaN
        if torch.isnan(loss):
            num_nan_batches += 1
            continue
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability (field encoder only)
        torch.nn.utils.clip_grad_norm_(field_encoder.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    # If ALL batches produced NaN, training has failed
    if num_batches == 0:
        raise RuntimeError(
            f"All {num_nan_batches} batches produced NaN loss. "
            "Training failed. Check model initialization and learning rate."
        )
    
    if num_nan_batches > 0:
        print(f"  Warning: {num_nan_batches} batches skipped due to NaN loss")
    
    avg_loss = total_loss / num_batches
    return avg_loss


# ============================================================================
# Evaluation
# ============================================================================

def extract_embeddings(field_encoder, text_encoder, dataloader, device, max_samples=500):
    """
    Extract embeddings for evaluation.
    
    Uses cached text embeddings to avoid redundant computation of identical descriptions.
    Multiple field snapshots share the same physics description; we encode each unique
    description only once.
    
    Returns:
        z_phys_all: field embeddings (n_samples, latent_dim)
        z_text_all: text embeddings (n_samples, latent_dim)
        viscosities: viscosity values for each sample
        descriptions: text descriptions for each sample
    """
    field_encoder.eval()
    text_encoder.eval()
    
    z_phys_list = []
    z_text_list = []
    viscosities_list = []
    descriptions_list = []
    
    # Cache to avoid re-encoding identical text descriptions
    text_embedding_cache = {}
    
    with torch.no_grad():
        n_samples = 0
        for fields, texts, viscosities_batch in dataloader:
            if n_samples >= max_samples:
                break
            
            # Move to device
            fields = fields.to(device)
            
            # Encode fields
            z_phys = field_encoder(fields)
            
            # Encode text with caching
            z_text_batch = []
            for text in texts:
                if text not in text_embedding_cache:
                    # Encode once and cache
                    z_text_single = text_encoder([text])
                    text_embedding_cache[text] = z_text_single[0].cpu()
                z_text_batch.append(text_embedding_cache[text])
            
            z_text = torch.stack(z_text_batch).to(device)
            
            # Store
            z_phys_list.append(z_phys.cpu())
            z_text_list.append(z_text.cpu())
            viscosities_list.extend(viscosities_batch.numpy())
            descriptions_list.extend(texts)
            
            n_samples += len(fields)
    
    z_phys_all = torch.cat(z_phys_list, dim=0).numpy()
    z_text_all = torch.cat(z_text_list, dim=0).numpy()
    viscosities = np.array(viscosities_list)
    
    # Check for NaN in embeddings
    if np.isnan(z_phys_all).any() or np.isnan(z_text_all).any():
        print("ERROR: NaN detected in embeddings. Training failed.")
        print(f"  NaN in z_phys: {np.isnan(z_phys_all).sum()} / {z_phys_all.size}")
        print(f"  NaN in z_text: {np.isnan(z_text_all).sum()} / {z_text_all.size}")
        raise ValueError("Training produced NaN embeddings. Check learning rate and model stability.")
    
    print(f"Text embedding cache: {len(text_embedding_cache)} unique descriptions")
    
    return z_phys_all, z_text_all, viscosities, descriptions_list


def visualize_embeddings(z_phys, viscosities, output_path):
    """Visualize field embeddings in 2D, colored by viscosity."""
    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    z_2d = pca.fit_transform(z_phys)
    
    # Plot
    _, ax = plt.subplots(figsize=(10, 8))
    
    # Color by viscosity
    scatter = ax.scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=viscosities,
        cmap='viridis',
        s=20,
        alpha=0.6
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Viscosity ν', fontsize=12)
    
    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_title('PHYSCLIP v0: Field Embeddings Colored by Viscosity', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved embedding visualization: {output_path}")


def nearest_neighbor_check(z_phys, z_text, viscosities, n_samples=5):
    """
    For random field embeddings, find nearest text embedding.
    
    Print predicted vs true physics descriptions.
    Uses explicit viscosity values (no string parsing).
    """
    print("\n" + "="*70)
    print("NEAREST NEIGHBOR SANITY CHECK")
    print("="*70)
    
    # Randomly sample field embeddings
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(z_phys), size=min(n_samples, len(z_phys)), replace=False)
    
    for idx in indices:
        z_phys_sample = z_phys[idx]
        true_nu = viscosities[idx]
        
        # Compute cosine similarity with all text embeddings
        # Embeddings are already L2-normalized, so dot product = cosine similarity
        similarities = z_phys_sample @ z_text.T
        
        # Find nearest text
        nearest_idx = np.argmax(similarities)
        pred_nu = viscosities[nearest_idx]
        
        match = "match" if np.isclose(true_nu, pred_nu) else "mismatch"
        
        print(f"\nSample {idx}:")
        print(f"  True viscosity:      nu={true_nu:.3f}")
        print(f"  Predicted viscosity: nu={pred_nu:.3f} ({match})")
        print(f"  Similarity:          {similarities[nearest_idx]:.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("PHYSCLIP v0: Contrastive Physics Representation Learning")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_DIR}")
    print()
    
    # Load dataset
    dataset = BurgersDataset(DATASET_DIR, METADATA_PATH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # 0 for Windows compatibility; increase on Linux/Mac
    )
    
    # Create models
    print("Creating models...")
    field_encoder, text_encoder = create_encoders(
        nx_input=NX_INPUT,
        latent_dim=LATENT_DIM,
        normalize_output=NORMALIZE_OUTPUT
    )
    field_encoder = field_encoder.to(DEVICE)
    text_encoder = text_encoder.to(DEVICE)
    
    criterion = ContrastiveLoss(temperature=TEMPERATURE)
    
    # Optimizer: train ONLY the field encoder
    # CRITICAL: Text embeddings are FROZEN in PHYSCLIP v0.
    # The TextEncoder provides fixed semantic anchors from pretrained language model.
    # We learn to align field behavior with frozen physics semantics.
    # This is intentional — v0 tests whether regimes are representationally learnable
    # WITHOUT fine-tuning language understanding.
    optimizer = optim.Adam(
        field_encoder.parameters(),  # ONLY field encoder is trained
        lr=LEARNING_RATE,
        weight_decay=1e-5  # Small L2 regularization
    )
    
    print(f"Field encoder parameters: {sum(p.numel() for p in field_encoder.parameters()):,}")
    print(f"Text encoder parameters:  {sum(p.numel() for p in text_encoder.parameters()):,}")
    print()
    
    # Training loop
    print("="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_epoch(
            field_encoder, text_encoder, criterion,
            dataloader, optimizer, DEVICE
        )
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")
    
    print()
    
    # Evaluation
    print("="*70)
    print("EVALUATION")
    print("="*70)
    
    z_phys, z_text, viscosities, _ = extract_embeddings(
        field_encoder, text_encoder, dataloader, DEVICE, max_samples=500
    )
    
    print(f"Extracted {len(z_phys)} embeddings for evaluation")
    print(f"Unique viscosities: {np.unique(viscosities)}")
    print()
    
    # Visualize
    viz_path = os.path.join(OUTPUT_DIR, "physclip_v0_embeddings.png")
    visualize_embeddings(z_phys, viscosities, viz_path)
    
    # Nearest neighbor check
    nearest_neighbor_check(z_phys, z_text, viscosities, n_samples=5)
    
    print("\n" + "="*70)
    print("PHYSCLIP v0 training complete!")
    print("="*70)
    print("\nSuccess criteria:")
    print("- Field embeddings should cluster by viscosity (see visualization)")
    print("- Nearest text neighbor should match correct viscosity (see above)")
    print("- Contrastive loss should decrease consistently (see training log)")


if __name__ == "__main__":
    main()
