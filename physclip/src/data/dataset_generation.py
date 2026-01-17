"""
Dataset generation for PHYSCLIP v1 with dense physics-parameter coverage.

Generates solution trajectories of the 1D viscous Burgers equation for varying
viscosity using sine-wave initial conditions. Each trajectory is associated
with a physics description string.

CRITICAL DESIGN CHOICES (v1 update):
- Dense viscosity grid (20 values from 0.01 to 0.2) for better physics coverage
- Multiple trajectories per viscosity (≥10) with varied initial conditions
- Sample count increase is physics-driven, not data augmentation
- Each trajectory still obeys ONE viscosity (no mixing)

Output format:
- Trajectories saved as NumPy .npy files (K, nx)
- Physics descriptions stored as metadata strings
- Compatible with multi-scale temporal windowing
- Suitable for contrastive alignment training
"""

import numpy as np
import os
from pathlib import Path

# Support running as a module (preferred) and as a direct script (fallback)
try:
    from .pde_solver import BurgersSpectralSolver  # type: ignore
except ImportError:
    # If executed directly (python dataset_generation.py), ensure repo root is on sys.path
    import sys
    ROOT = Path(__file__).resolve().parents[3]  # points to repo root (PHYSCLIP)
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from physclip.src.data.pde_solver import BurgersSpectralSolver


def generate_dataset(
    output_dir="./burgers_data",
    nu_values=None,
    n_trajectories_per_viscosity=10,
    phase_shift_range=(0, 2*np.pi),
    amplitude_variation_range=(0.8, 1.2),
    t_final=1.0,
    nx=256,
    dt=0.0005,
    n_snapshots=11,
    seed=42
):
    """
    Generate synthetic Burgers equation data for PHYSCLIP v1.
    
    CRITICAL PHYSICS INVARIANTS:
    - Each trajectory evolves under ONE viscosity ν (no mixing)
    - Each trajectory has ONE physics description
    - Sample count increases via physics-parameter coverage, not time oversampling
    - Initial condition variations are physics-preserving (sine waves with varied phase/amplitude)
    
    Parameters
    ----------
    output_dir : str
        Directory to save trajectories and metadata.
    nu_values : list of float or None
        Viscosity coefficients to sample. 
        Default: np.linspace(0.01, 0.2, 20) - dense grid for better physics coverage
    n_trajectories_per_viscosity : int
        Number of independent initial conditions per viscosity.
        Default: 10 (provides statistical diversity while maintaining physics semantics)
    phase_shift_range : tuple of float
        Range for phase shifts in initial conditions: u0 = sin(kx + φ)
        Default: (0, 2π) - full phase space coverage
    amplitude_variation_range : tuple of float
        Range for amplitude variations: u0 = A·sin(kx)
        Default: (0.8, 1.2) - small variations to preserve shock structure
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
        
    Notes
    -----
    Sample count scaling:
        - v0 original: 4 viscosities × 2 traj/ν × 4 modes = 32 trajectories → 352 snapshots
        - v1 default: 20 viscosities × 10 traj/ν = 200 trajectories → ~2400 windows
    
    This is a physics-driven expansion (denser ν manifold), not data augmentation.
    """
    
    # Set random seed
    np.random.seed(seed)
    
    # Default viscosity grid: dense coverage of physics parameter space
    if nu_values is None:
        nu_values = np.linspace(0.01, 0.2, 20)
    
    # Convert to list if numpy array
    if isinstance(nu_values, np.ndarray):
        nu_values = nu_values.tolist()

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
    
    # Logging: dataset statistics
    print("="*70)
    print("PHYSCLIP v1: Dataset Generation")
    print("="*70)
    print(f"Viscosity values: {len(nu_values)}")
    print(f"  Range: [{min(nu_values):.4f}, {max(nu_values):.4f}]")
    print(f"Trajectories per viscosity: {n_trajectories_per_viscosity}")
    print(f"Total trajectories: {len(nu_values) * n_trajectories_per_viscosity}")
    print(f"Snapshots per trajectory: {n_snapshots}")
    print(f"Spatial resolution: nx={nx}")
    print(f"Temporal resolution: dt={dt}, t_final={t_final}")
    print()
    
    # Estimate windowed sample count (for reference)
    # Assuming multi-scale windowing with [5, 7, 9] and stride=2
    total_trajectories = len(nu_values) * n_trajectories_per_viscosity
    windows_per_traj_approx = 10  # Rough estimate: (11-5)//2+1 + (11-7)//2+1 + (11-9)//2+1 ≈ 10
    estimated_windows = total_trajectories * windows_per_traj_approx
    print(f"Estimated windowed samples (with multi-scale):")
    print(f"  ~{estimated_windows} windows (assuming window_sizes=[5,7,9], stride=2)")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.txt")
    
    # Open metadata file with UTF-8 encoding (handles Unicode math symbols)
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        meta_file.write("trajectory_id,physics_description,filepath\n")
        
        trajectory_id = 0
        
        # Loop over viscosity values (physics parameter)
        for nu in nu_values:
            # Physics description is determined ONLY by the equation and viscosity
            # All trajectories at this viscosity share this description
            physics_description = (
                f"1D viscous Burgers equation: "
                f"∂u/∂t + u·∂u/∂x = ν·∂²u/∂x². "
                f"Domain: [0, 2π] periodic boundary conditions. "
                f"Viscosity: ν={nu:.3f}."
            )
            
            # Loop over independent trajectories with varied initial conditions
            # This provides statistical diversity while preserving physics
            for traj_idx in range(n_trajectories_per_viscosity):
                
                # Generate physics-preserving initial condition variations
                # Strategy: phase-shifted and amplitude-scaled sine waves
                
                # Base mode (fixed for simplicity, can be randomized)
                mode = 1
                
                # Phase shift: u0 = sin(kx + φ)
                phase_shift = np.random.uniform(*phase_shift_range)
                
                # Amplitude variation: u0 = A·sin(kx)
                amplitude = np.random.uniform(*amplitude_variation_range)
                
                # Create initial condition
                x = np.linspace(0, 2*np.pi, nx, endpoint=False)
                u0 = amplitude * np.sin(mode * x + phase_shift)
                
                # Initialize solver and integrate
                # CRITICAL: Each trajectory uses ONLY ONE viscosity ν
                solver = BurgersSpectralSolver(nu=nu, nx_grid=nx, dt=dt)
                _, trajectory = solver.solve(u0, t_final, return_trajectory=True)
                
                # Select evenly-spaced snapshots
                indices = np.linspace(0, len(trajectory)-1, n_snapshots, dtype=int)
                snapshots = np.array([trajectory[i] for i in indices])
                
                # Validate no NaNs (physics instability check)
                if np.isnan(snapshots).any():
                    raise RuntimeError(
                        f"NaN detected in trajectory {trajectory_id} (nu={nu:.4f}). "
                        "Physics integration is unstable. Check dt, dx, or initial conditions."
                    )
                
                # Save trajectory
                filename = f"trajectory_{trajectory_id:06d}.npy"
                filepath = os.path.join(output_dir, filename)
                np.save(filepath, snapshots)
                
                # Write to metadata
                # INVARIANT: Multiple trajectories share the same physics description
                # only if they have the same viscosity
                meta_file.write(
                    f"{trajectory_id},\"{physics_description}\",{filename}\n"
                )
                
                trajectory_id += 1
            
            # Progress logging
            if (nu == nu_values[0]) or (nu == nu_values[-1]) or (trajectory_id % 50 == 0):
                print(f"  Generated {n_trajectories_per_viscosity} trajectories for nu={nu:.4f} (total: {trajectory_id})")
    
    print()
    print("="*70)
    print(f"Dataset generation complete!")
    print(f"  Total trajectories: {trajectory_id}")
    print(f"  Saved to: {output_dir}")
    print(f"  Metadata: {metadata_path}")
    print("="*70)


if __name__ == "__main__":
    # Generate PHYSCLIP v1 dataset with dense physics coverage
    # 
    # DESIGN RATIONALE:
    # - 20 viscosity values: dense sampling of physics parameter space
    # - 10 trajectories per ν: statistical diversity with varied initial conditions
    # - Total: 200 base trajectories (vs v0's 32)
    # - With windowing (stride=2, sizes=[5,7,9]): ~2000 training samples
    # 
    # This is a physics-driven expansion, NOT data augmentation.
    # Each trajectory represents a unique solution to the Burgers equation.
    
    generate_dataset(
        output_dir="./burgers_data",
        nu_values=np.linspace(0.01, 0.2, 20),  # Dense viscosity grid
        n_trajectories_per_viscosity=10,        # Multiple ICs per viscosity
        phase_shift_range=(0, 2*np.pi),        # Full phase space
        amplitude_variation_range=(0.8, 1.2),  # Small amplitude variations
        t_final=1.0,
        nx=256,
        dt=0.0005,
        n_snapshots=11,
        seed=42
    )
