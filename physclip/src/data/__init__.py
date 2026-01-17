"""Data generation and solvers for PHYSCLIP."""

from .dataset_generation import generate_dataset
from .trajectory_dataset import BurgersTrajectoryDataset, collate_trajectory_batch
from .pde_solver import BurgersSpectralSolver

__all__ = [
    "generate_dataset",
    "BurgersTrajectoryDataset",
    "collate_trajectory_batch",
    "BurgersSpectralSolver",
]
