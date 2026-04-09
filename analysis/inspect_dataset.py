"""
Dataset inspection for Burgers equation trajectories.

Validates the structure and physics alignment of the generated dataset.
No modification of data; analysis only.
"""

import numpy as np
import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt


dataset_dir = "./burgers_data"
metadata_path = os.path.join(dataset_dir, "metadata.txt")


# Load metadata and count trajectories
descriptions_to_trajectories = defaultdict(list)
trajectory_files = []

with open(metadata_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        trajectory_id = row['trajectory_id']
        physics_desc = row['physics_description']
        filepath = row['filepath']
        
        descriptions_to_trajectories[physics_desc].append(trajectory_id)
        trajectory_files.append((trajectory_id, filepath, physics_desc))

n_total_trajectories = len(trajectory_files)
n_unique_descriptions = len(descriptions_to_trajectories)

# Load one example trajectory to determine array shape
first_file = os.path.join(dataset_dir, trajectory_files[0][1])
first_data = np.load(first_file)
n_snapshots_per_trajectory, nx_spatial = first_data.shape

# Print summary statistics
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)
print(f"Total trajectories:           {n_total_trajectories}")
print(f"Snapshots per trajectory:     {n_snapshots_per_trajectory}")
print(f"Spatial grid points:          {nx_spatial}")
print(f"Unique physics descriptions:  {n_unique_descriptions}")
print()

# Print many-to-one mapping
print("=" * 70)
print("PHYSICS DESCRIPTION MAPPING")
print("=" * 70)
for physics_desc, traj_ids in descriptions_to_trajectories.items():
    print(f"Count: {len(traj_ids):3d} trajectories")
    print(f"Desc:  {physics_desc}")
    print()

# Extract viscosity values from descriptions and identify low/high examples
viscosities_set = set()
viscosity_to_first_trajectory = {}

for physics_desc, traj_ids in descriptions_to_trajectories.items():
    nu_str = physics_desc.split("ν=")[1].strip(".")
    nu_value = float(nu_str)
    viscosities_set.add(nu_value)
    if nu_value not in viscosity_to_first_trajectory:
        viscosity_to_first_trajectory[nu_value] = (traj_ids[0], physics_desc)

viscosities_sorted = sorted(viscosities_set)
nu_low = viscosities_sorted[0]
nu_high = viscosities_sorted[-1]

low_traj_id, low_desc = viscosity_to_first_trajectory[nu_low]
high_traj_id, high_desc = viscosity_to_first_trajectory[nu_high]

# Load trajectory data for plotting
low_traj_file = os.path.join(dataset_dir, f"trajectory_{int(low_traj_id):06d}.npy")
high_traj_file = os.path.join(dataset_dir, f"trajectory_{int(high_traj_id):06d}.npy")

low_data = np.load(low_traj_file)
high_data = np.load(high_traj_file)

# Spatial grid
x = np.linspace(0, 2*np.pi, nx_spatial, endpoint=False)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Low viscosity: early vs late time
ax = axes[0]
u_early_low = low_data[0, :]
u_late_low = low_data[-1, :]
ax.plot(x, u_early_low, "b-", linewidth=2, label=f"t=0 (ν={nu_low})")
ax.plot(x, u_late_low, "r-", linewidth=2, label=f"t=final (ν={nu_low})")
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("u(x,t)", fontsize=12)
ax.set_title(f"Low Viscosity (ν={nu_low})", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# High viscosity: early vs late time
ax = axes[1]
u_early_high = high_data[0, :]
u_late_high = high_data[-1, :]
ax.plot(x, u_early_high, "b-", linewidth=2, label=f"t=0 (ν={nu_high})")
ax.plot(x, u_late_high, "g-", linewidth=2, label=f"t=final (ν={nu_high})")
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("u(x,t)", fontsize=12)
ax.set_title(f"High Viscosity (ν={nu_high})", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("burgers_trajectories.png", dpi=150, bbox_inches="tight")
print("=" * 70)
print(f"Plot saved: burgers_trajectories.png")
print("=" * 70)

# Spatial gradient analysis (shock steepening)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Low viscosity gradients
ax = axes[0]
grad_early_low = np.gradient(u_early_low, x)
grad_late_low = np.gradient(u_late_low, x)
ax.plot(x, grad_early_low, "b-", linewidth=2, label=f"∂u/∂x at t=0")
ax.plot(x, grad_late_low, "r-", linewidth=2, label=f"∂u/∂x at t=final")
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("∂u/∂x", fontsize=12)
ax.set_title(f"Spatial Gradient (ν={nu_low})", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# High viscosity gradients
ax = axes[1]
grad_early_high = np.gradient(u_early_high, x)
grad_late_high = np.gradient(u_late_high, x)
ax.plot(x, grad_early_high, "b-", linewidth=2, label=f"∂u/∂x at t=0")
ax.plot(x, grad_late_high, "g-", linewidth=2, label=f"∂u/∂x at t=final")
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("∂u/∂x", fontsize=12)
ax.set_title(f"Spatial Gradient (ν={nu_high})", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("burgers_gradients.png", dpi=150, bbox_inches="tight")
print(f"Plot saved: burgers_gradients.png")
print()
