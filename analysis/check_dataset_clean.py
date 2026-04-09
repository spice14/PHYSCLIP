import numpy as np
import pathlib

files = sorted(pathlib.Path('burgers_data').glob('trajectory_*.npy'))
bad = []
mn = float('inf')
mx = float('-inf')
for f in files:
    arr = np.load(f)
    if np.isnan(arr).any():
        bad.append(f.name)
    mn = min(mn, np.nanmin(arr))
    mx = max(mx, np.nanmax(arr))

print(f"checked {len(files)} files")
print(f"has_nan: {len(bad) > 0}")
print(f"min: {mn}, max: {mx}")
print(f"nan_files: {bad}")
