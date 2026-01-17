"""
Dataset generation for PHYSCLIP v0.

Generates solution trajectories of the 1D viscous Burgers equation for varying
viscosity using sine-wave initial conditions. Each trajectory is associated
with a physics description string.

Output format:
- Trajectories saved as NumPy .npy files.
- Physics descriptions stored as metadata strings.
- Suitable for later contrastive alignment training.
"""

import numpy as np
import os
from pathlib import Path
from burgers_solver import BurgersSpectralSolver


def generate_dataset(
    output_dir="./burgers_data",
    nu_values=None,
    n_trajectories_per_viscosity=1,
    n_modes_list=None,
    t_final=1.0,
    nx=256,
    dt=0.0005,
    n_snapshots=11,
    seed=42
):
    """
    Generate synthetic Burgers equation data for PHYSCLIP.
    
    Parameters
    ----------
    output_dir : str
        Directory to save trajectories and metadata.
    nu_values : list of float
        Viscosity coefficients to sample. Default: [0.01, 0.05, 0.1, 0.2].
    n_trajectories_per_viscosity : int
        Number of independent initial conditions per viscosity.
    n_modes_list : list of int
        Sine-wave mode numbers for initial conditions. Default: [1, 2, 3, 4].
    t_final : float
        Integration time.
    nx : int
        Number of spatial grid points.
    dt : float
        Time step size.
    n_snapshots : int
        Number of time snapshots to save from each trajectory.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    None
        Saves trajectories to output_dir.
    """
    
    # Set random seed
    np.random.seed(seed)
    
    # Default parameter values
    if nu_values is None:
        nu_values = [0.01, 0.05, 0.1, 0.2]
    if n_modes_list is None:
        n_modes_list = [1, 2, 3, 4]

    # Diffusive stability guard: dt must satisfy dt <= dx^2 / (2*max_nu)
    max_nu = max(nu_values) if len(nu_values) > 0 else 0.0
    dx = 2 * np.pi / nx
    if max_nu > 0:
        dt_limit = (dx ** 2) / (2.0 * max_nu)
        if dt >= dt_limit:
            raise ValueError(
                f"Time step dt={dt} is too large for diffusion stability (limit {dt_limit:.6f} for nu={max_nu}). "
                "Reduce dt or increase nx."
            )
    
    # Numerical safety check
    dx = 2 * np.pi / nx
    if dt >= dx:
        raise ValueError(
            f"Time step dt={dt} is not small relative to dx={dx:.6f}. "
            "Consider reducing dt or increasing nx."
        )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.txt")
    
    # Open metadata file with UTF-8 encoding (handles Unicode math symbols)
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        meta_file.write("trajectory_id,physics_description,filepath\n")
        
        trajectory_id = 0
        
        # Loop over viscosity values
        for nu in nu_values:
            # Physics description is determined only by the equation and viscosity
            # All trajectories at this viscosity share this description
            physics_description = (
                f"1D viscous Burgers equation: "
                f"∂u/∂t + u·∂u/∂x = ν·∂²u/∂x². "
                f"Domain: [0, 2π] periodic boundary conditions. "
                f"Viscosity: ν={nu:.3f}."
            )
            
            # Loop over initial condition modes (nuisance variability)
            for mode in n_modes_list:
                # Loop over independent trajectories per (nu, mode) pair
                for _ in range(n_trajectories_per_viscosity):
                    
                    # Create initial condition: u(x, 0) = sin(mode·x)
                    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
                    u0 = np.sin(mode * x)
                    
                    # Initialize solver and integrate
                    solver = BurgersSpectralSolver(nu=nu, nx_grid=nx, dt=dt)
                    _, trajectory = solver.solve(u0, t_final, return_trajectory=True)
                    
                    # Select evenly-spaced snapshots
                    indices = np.linspace(0, len(trajectory)-1, n_snapshots, dtype=int)
                    snapshots = np.array([trajectory[i] for i in indices])
                    
                    # Save trajectory
                    filename = f"trajectory_{trajectory_id:06d}.npy"
                    filepath = os.path.join(output_dir, filename)
                    np.save(filepath, snapshots)
                    
                    # Write to metadata
                    # Multiple trajectories share the same physics description
                    meta_file.write(
                        f"{trajectory_id},\"{physics_description}\",{filename}\n"
                    )
                    
                    trajectory_id += 1
    
    print(f"Dataset generation complete. Saved {trajectory_id} trajectories to {output_dir}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    # Generate default dataset
    generate_dataset(
        output_dir="./burgers_data",
        nu_values=[0.01, 0.05, 0.1, 0.2],
        n_trajectories_per_viscosity=2,
        n_modes_list=[1, 2, 3, 4],
        t_final=1.0,
        nx=256,
        dt=0.0005,
        n_snapshots=11,
        seed=42
    )
