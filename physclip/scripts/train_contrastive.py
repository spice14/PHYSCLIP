"""
PHYSCLIP v1: Fixed Training Script with Proper Convergence

CRITICAL FIXES APPLIED:
1. Use ALL windows with simple collate (no padding issues)
2. Increased learning rate: 1e-3 (was 1e-4)
3. Smaller batch size: 8 (was 16, gives more updates)
4. Extended epochs: 200 (was 20)
5. Added learning rate scheduling for gradual decay
6. Removed padding complexity - use all windows as-is
"""

import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no window creation
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from physclip package
from src.models import FieldEncoder, TrajectoryEncoder, ContrastiveLoss
from src.models.encoders import TextEncoder
from src.data import BurgersTrajectoryDataset
from src.data.trajectory_dataset import collate_trajectory_batch

# ============================================================================
# Configuration
# ============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_DATA = (_SCRIPT_DIR.parent / "data" / "burgers_data").resolve()
_ROOT_DATA = (_SCRIPT_DIR.parent.parent / "burgers_data").resolve()

# Prefer package-local data; fall back to repo-root burgers_data
if (_PKG_DATA / "metadata.txt").exists():
    DATASET_DIR = str(_PKG_DATA)
elif (_ROOT_DATA / "metadata.txt").exists():
    DATASET_DIR = str(_ROOT_DATA)
else:
    # Default to package path; loader will raise with clear path
    DATASET_DIR = str(_PKG_DATA)

METADATA_PATH = os.path.join(DATASET_DIR, "metadata.txt")

# Model hyperparameters
NX_INPUT = 256
LATENT_DIM = 128
NORMALIZE_OUTPUT = True

# Training hyperparameters - FIXED FOR CONVERGENCE
NUM_EPOCHS = 200  # Extended for better convergence
BATCH_SIZE = 8    # Smaller batch = more gradient updates
LEARNING_RATE = 1e-3  # Increased from 1e-4
WEIGHT_DECAY = 1e-5
TEMPERATURE = 0.07
GRADIENT_CLIP = 1.0

# Multi-scale windowing
WINDOW_SIZES = [5, 7, 9]
STRIDE = 2

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(trajectory_encoder, text_encoder, criterion, dataloader, optimizer, device):
    """Train for one epoch."""
    trajectory_encoder.train()
    text_encoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for windows, texts, viscosities in dataloader:
        windows = windows.to(device)
        
        # Forward pass
        z_traj = trajectory_encoder(windows)
        z_text = text_encoder(texts)
        
        # Loss
        loss = criterion(z_traj, z_text)
        
        # Check for NaN
        if torch.isnan(loss):
            print("WARNING: NaN loss, skipping batch")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trajectory_encoder.parameters(), max_norm=GRADIENT_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    if num_batches == 0:
        raise RuntimeError("All batches produced NaN. Training failed.")
    
    return total_loss / num_batches


def extract_embeddings(trajectory_encoder, text_encoder, dataloader, device):
    """Extract embeddings for all windows."""
    trajectory_encoder.eval()
    text_encoder.eval()
    
    z_traj_list = []
    z_text_list = []
    viscosities_list = []
    
    text_cache = {}
    
    with torch.no_grad():
        for windows, texts, viscosities_batch in dataloader:
            windows = windows.to(device)
            
            # Encode windows
            z_traj = trajectory_encoder(windows)
            
            # Encode text with caching
            z_text_batch = []
            for text in texts:
                if text not in text_cache:
                    text_cache[text] = text_encoder([text])
                z_text_batch.append(text_cache[text])
            z_text = torch.cat(z_text_batch, dim=0)
            
            z_traj_list.append(z_traj.cpu())
            z_text_list.append(z_text.cpu())
            viscosities_list.append(viscosities_batch)
    
    z_traj_all = torch.cat(z_traj_list, dim=0).numpy()
    z_text_all = torch.cat(z_text_list, dim=0).numpy()
    viscosities = torch.cat(viscosities_list, dim=0).numpy()
    
    print(f"Text embedding cache: {len(text_cache)} unique descriptions")
    
    return z_traj_all, z_text_all, viscosities


def visualize_embeddings(z_traj, viscosities, save_path):
    """Visualize trajectory embeddings via PCA."""
    pca = PCA(n_components=2, random_state=42)
    z_2d = pca.fit_transform(z_traj)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=viscosities,
        cmap='viridis',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    plt.colorbar(scatter, label='Viscosity nu')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.title('PHYSCLIP v1: Trajectory Window Embeddings (PCA)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Saved visualization: {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("PHYSCLIP v1: Fixed Training with All Windows")
    print("="*70)
    print(f"Device: {DEVICE}")
    print()
    
    # Load dataset
    dataset = BurgersTrajectoryDataset(
        DATASET_DIR,
        METADATA_PATH,
        window_sizes=WINDOW_SIZES,
        stride=STRIDE
    )
    
    print()
    print(f"Total windows: {len(dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches per epoch: {len(dataset) // BATCH_SIZE}")
    print()
    
    # DataLoader with all windows
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_trajectory_batch,
        num_workers=0
    )
    
    # Create models
    print("Creating models...")
    field_encoder = FieldEncoder(
        nx_input=NX_INPUT,
        latent_dim=LATENT_DIM,
        normalize_output=True
    )
    trajectory_encoder = TrajectoryEncoder(
        field_encoder=field_encoder,
        normalize_output=NORMALIZE_OUTPUT
    )
    trajectory_encoder = trajectory_encoder.to(DEVICE)
    
    text_encoder = TextEncoder(
        latent_dim=LATENT_DIM,
        normalize_output=NORMALIZE_OUTPUT
    )
    text_encoder = text_encoder.to(DEVICE)
    
    traj_params = sum(p.numel() for p in trajectory_encoder.parameters())
    text_params = sum(p.numel() for p in text_encoder.parameters())
    print(f"Trajectory encoder: {traj_params:,} params")
    print(f"Text encoder: {text_params:,} params")
    print()
    
    # Loss and optimizer
    criterion = ContrastiveLoss(temperature=TEMPERATURE)
    optimizer = optim.Adam(
        trajectory_encoder.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler: decay by 0.5x every 50 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Training
    print("="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_epoch(
            trajectory_encoder, text_encoder, criterion,
            dataloader, optimizer, DEVICE
        )
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
    
    # Evaluation
    print()
    print("="*70)
    print("EVALUATION")
    print("="*70)
    
    eval_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_trajectory_batch,
        num_workers=0
    )
    
    z_traj, z_text, viscosities = extract_embeddings(
        trajectory_encoder, text_encoder, eval_dataloader, DEVICE
    )
    
    print(f"Extracted {len(z_traj)} embeddings")
    print(f"Unique viscosities: {len(np.unique(viscosities))}")
    print(f"Viscosity range: [{viscosities.min():.3f}, {viscosities.max():.3f}]")
    print()
    
    # Visualize
    viz_path = os.path.join(OUTPUT_DIR, "physclip_v1_embeddings.png")
    visualize_embeddings(z_traj, viscosities, viz_path)
    
    # Nearest neighbor check
    print()
    print("="*70)
    print("NEAREST NEIGHBOR CHECK")
    print("="*70)
    
    for i in range(min(5, len(z_traj))):
        similarities = z_traj[i] @ z_text.T
        nearest_idx = np.argmax(similarities)
        match = "MATCH" if np.isclose(viscosities[i], viscosities[nearest_idx]) else "MISMATCH"
        print(f"Sample {i}: true_nu={viscosities[i]:.3f} -> pred_nu={viscosities[nearest_idx]:.3f} [{match}]")
    
    print()
    print("="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
