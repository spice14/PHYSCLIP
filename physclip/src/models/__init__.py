"""Neural encoders and contrastive loss for PHYSCLIP."""

from .encoders import FieldEncoder, TextEncoder, create_encoders
from .trajectory_models import TrajectoryEncoder
from .contrastive_loss import ContrastiveLoss

# Alias for backward compatibility
InfoNCELoss = ContrastiveLoss

__all__ = [
    "FieldEncoder",
    "TextEncoder",
    "TrajectoryEncoder",
    "ContrastiveLoss",
    "InfoNCELoss",
    "create_encoders",
]
