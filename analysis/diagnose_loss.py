"""
Diagnostic script to verify loss computation and embeddings.
"""
import torch
import torch.nn.functional as F
from models.losses import ContrastiveLoss

# Simulate batch of normalized embeddings
batch_size = 8
latent_dim = 128

# Perfect alignment case
z_phys = F.normalize(torch.randn(batch_size, latent_dim), p=2, dim=-1)
z_text = z_phys.clone()  # Perfect match

criterion = ContrastiveLoss(temperature=0.07)
loss_perfect = criterion(z_phys, z_text)
print(f"Loss (perfect alignment): {loss_perfect.item():.6f}")
print("  Expected: ~0 (all positives have highest similarity)")

# Random case (no alignment)
z_text_random = F.normalize(torch.randn(batch_size, latent_dim), p=2, dim=-1)
loss_random = criterion(z_phys, z_text_random)
log_batch = torch.tensor(batch_size).float().log().item()
print(f"Loss (random alignment): {loss_random.item():.6f}")
print(f"  Expected: ~log(batch_size)={log_batch:.6f}")

# Check logits structure
logits = torch.matmul(z_phys, z_text.t()) / 0.07
print("Logits (perfect, T=0.07):")
print(f"  Diagonal mean: {logits.diag().mean().item():.4f} (should be high)")
print(f"  Off-diag mean: {logits[~torch.eye(batch_size, dtype=torch.bool)].mean().item():.4f} (should be lower)")
print(f"  Max logit: {logits.max().item():.4f}")
print(f"  Min logit: {logits.min().item():.4f}")

# Now with larger temperature
criterion_large_t = ContrastiveLoss(temperature=0.5)
loss_perfect_large_t = criterion_large_t(z_phys, z_text)
print(f"\nLoss (perfect alignment, T=0.5): {loss_perfect_large_t.item():.6f}")
print("  (Softer softmax, slower convergence)")

logits_large_t = torch.matmul(z_phys, z_text.t()) / 0.5
print("Logits (perfect, T=0.5):")
print(f"  Diagonal mean: {logits_large_t.diag().mean().item():.4f}")
print(f"  Off-diag mean: {logits_large_t[~torch.eye(batch_size, dtype=torch.bool)].mean().item():.4f}")
