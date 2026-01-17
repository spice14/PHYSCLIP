"""
Encoder architectures for PHYSCLIP v0.

Two encoders map physical fields and text descriptions into a shared latent space:
- FieldEncoder: maps field snapshots u(x) → z_phys
- TextEncoder: maps physics description strings → z_text

Both encoders produce fixed-dimensional vectors in the same latent space.
This enables contrastive alignment between physical behavior and physics semantics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class FieldEncoder(nn.Module):
    """
    Encode 1D physical field snapshots into latent representations.
    
    Maps a spatial field u(x) to a fixed-dimensional latent vector z_phys.
    Designed to capture physical structure (e.g., smoothness, shock formation).
    
    Architecture: 1D CNN with progressive downsampling followed by global pooling.
    
    Parameters
    ----------
    nx_input : int
        Number of spatial grid points in input field.
    latent_dim : int
        Dimension of output latent vector.
    hidden_channels : list of int
        Number of channels in each convolutional layer.
    normalize_output : bool
        If True, L2-normalize latent vectors (recommended for contrastive learning).
    """
    
    def __init__(self, nx_input=256, latent_dim=128, hidden_channels=None, normalize_output=True):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [16, 32, 64]
        
        self.nx_input = nx_input
        self.latent_dim = latent_dim
        self.normalize_output = normalize_output
        
        # Build 1D convolutional layers
        layers = []
        in_channels = 1  # Single field u(x)
        
        for out_channels in hidden_channels:
            # Use GroupNorm instead of BatchNorm to avoid mixing across samples
            # num_groups = min(out_channels, 8) ensures divisibility
            num_groups = min(out_channels, 8)
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(num_groups, out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global average pooling followed by linear projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels[-1], latent_dim)
    
    def forward(self, u):
        """
        Encode field snapshot to latent vector.
        
        Parameters
        ----------
        u : torch.Tensor
            Input field of shape (batch_size, nx_input) or (batch_size, 1, nx_input).
        
        Returns
        -------
        z_phys : torch.Tensor
            Latent representation of shape (batch_size, latent_dim).
        """
        # Ensure input has channel dimension: (batch, 1, nx)
        if u.dim() == 2:
            u = u.unsqueeze(1)
        
        # Debug: check input
        if torch.isnan(u).any():
            print(f"FieldEncoder: NaN in input u")
            print(f"  u stats: min={u.min().item()}, max={u.max().item()}, mean={u.mean().item()}")
        
        # Apply convolutional layers
        x = self.conv_layers(u)
        
        # Debug: check after convolutions
        if torch.isnan(x).any():
            print(f"FieldEncoder: NaN after conv_layers")
            print(f"  x stats: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}")
        
        # Global pooling: (batch, channels, spatial) → (batch, channels, 1)
        x = self.global_pool(x)
        
        # Debug: check after pooling
        if torch.isnan(x).any():
            print(f"FieldEncoder: NaN after global_pool")
        
        # Flatten: (batch, channels, 1) → (batch, channels)
        x = x.squeeze(-1)
        
        # Project to latent space
        z_phys = self.fc(x)
        
        # Debug: check after linear layer
        if torch.isnan(z_phys).any():
            print(f"FieldEncoder: NaN after fc layer")
            print(f"  z_phys stats: min={z_phys.min().item()}, max={z_phys.max().item()}, mean={z_phys.mean().item()}")
        
        # L2 normalization for contrastive learning
        if self.normalize_output:
            z_phys = F.normalize(z_phys, p=2, dim=-1)
        
        # Debug: check after normalization
        if torch.isnan(z_phys).any():
            print("FieldEncoder: NaN after L2 normalization!")
            print("  This may indicate zero-norm vectors")
        
        return z_phys


class TextEncoder(nn.Module):
    """
    Encode physics description strings into latent representations.
    
    Maps physics text (equation, regime, parameters) to a fixed-dimensional
    latent vector z_text using a pretrained sentence embedding model.
    
    This encoder is strictly semantic, not generative. The pretrained model
    is frozen and provides only embeddings. A learned linear projection maps
    these embeddings into the shared latent space for contrastive alignment.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of output latent vector.
    model_name : str
        Name of pretrained sentence-transformers model.
    freeze_base : bool
        If True, freeze the pretrained model weights (only train projection).
    normalize_output : bool
        If True, L2-normalize latent vectors (recommended for contrastive learning).
    """
    
    def __init__(
        self,
        latent_dim=128,
        model_name='all-MiniLM-L6-v2',
        freeze_base=True,
        normalize_output=True
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.normalize_output = normalize_output
        
        # Load pretrained sentence embedding model
        self.base_model = SentenceTransformer(model_name)
        
        # Freeze pretrained weights if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Get embedding dimension from base model
        base_dim = self.base_model.get_sentence_embedding_dimension()
        
        # Linear projection to target latent dimension
        self.projection = nn.Linear(base_dim, latent_dim)
    
    def forward(self, text_list):
        """
        Encode physics description strings to latent vectors.
        
        CRITICAL: Text embeddings are FROZEN semantic anchors in PHYSCLIP v0.
        No gradients flow through the SentenceTransformer. Only the projection
        layer is trainable, but even that should not receive gradients in v0.
        
        Parameters
        ----------
        text_list : list of str
            Physics description strings (batch).
        
        Returns
        -------
        z_text : torch.Tensor
            Latent representations of shape (batch_size, latent_dim).
        """
        # Encode text using pretrained model (frozen, semantic only)
        # MUST be wrapped in no_grad to prevent autograd issues
        with torch.no_grad():
            embeddings = self.base_model.encode(
                text_list,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            # Clone to convert inference tensor to normal tensor
            # The error message explicitly says to use clone, not detach
            embeddings = embeddings.clone()
        
        # Project to shared latent dimension and immediately detach
        # No gradients should flow through text encoder in v0
        z_text = self.projection(embeddings).detach()
        
        # L2 normalization for contrastive learning
        if self.normalize_output:
            z_text = F.normalize(z_text, dim=-1)
        
        # Defensive NaN check
        if torch.isnan(z_text).any():
            raise RuntimeError("TextEncoder produced NaN embeddings. Check projection layer.")
        
        return z_text


def create_encoders(nx_input=256, latent_dim=128, normalize_output=True):
    """
    Factory function to create matched field and text encoders.
    
    Parameters
    ----------
    nx_input : int
        Spatial grid size for field encoder.
    latent_dim : int
        Shared latent dimension for both encoders.
    normalize_output : bool
        If True, L2-normalize encoder outputs (recommended for contrastive learning).
    
    Returns
    -------
    field_encoder : FieldEncoder
        Encoder for physical fields.
    text_encoder : TextEncoder
        Encoder for physics descriptions.
    """
    field_encoder = FieldEncoder(
        nx_input=nx_input,
        latent_dim=latent_dim,
        normalize_output=normalize_output
    )
    text_encoder = TextEncoder(
        latent_dim=latent_dim,
        normalize_output=normalize_output
    )
    
    return field_encoder, text_encoder
