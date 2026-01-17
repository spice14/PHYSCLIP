"""
Contrastive loss for PHYSCLIP v0.

Implements CLIP-style InfoNCE loss to align physical field embeddings
with physics text embeddings in a shared latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    CLIP-style contrastive loss for physics-text alignment.
    
    Aligns physical field representations z_phys with physics description
    representations z_text by maximizing similarity of matched pairs and
    minimizing similarity of mismatched pairs within each batch.
    
    Loss is symmetric: computed both phys→text and text→phys.
    
    Parameters
    ----------
    temperature : float
        Temperature parameter τ for scaling logits. Lower values make
        the model more confident; higher values soften the distribution.
        Typical range: [0.01, 0.1].
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_phys, z_text):
        """
        Compute contrastive loss between field and text embeddings.
        
        Parameters
        ----------
        z_phys : torch.Tensor
            Physical field embeddings, shape (batch_size, latent_dim).
            Must be L2-normalized.
        z_text : torch.Tensor
            Physics text embeddings, shape (batch_size, latent_dim).
            Must be L2-normalized.
        
        Returns
        -------
        loss : torch.Tensor
            Scalar contrastive loss (symmetric).
        
        Notes
        -----
        Positive pairs are diagonal: (z_phys[i], z_text[i]).
        All off-diagonal pairs are treated as negatives.
        """
        batch_size = z_phys.shape[0]
        
        # Compute cosine similarity matrix (dot product since inputs are normalized)
        # Shape: (batch_size, batch_size)
        logits = torch.matmul(z_phys, z_text.t()) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=z_phys.device)
        
        # Symmetric loss:
        # 1. phys → text: for each z_phys[i], predict which z_text it matches
        loss_phys_to_text = F.cross_entropy(logits, labels)
        
        # 2. text → phys: for each z_text[i], predict which z_phys it matches
        loss_text_to_phys = F.cross_entropy(logits.t(), labels)
        
        # Average the two directions
        loss = (loss_phys_to_text + loss_text_to_phys) / 2.0
        
        return loss
