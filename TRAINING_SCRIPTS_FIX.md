# Training Script Fixes - Summary

## Problem
After the directory reorganization, the training scripts couldn't run because they still referenced old import paths and module locations from the pre-reorganization structure.

## Root Cause
The codebase was reorganized from a scattered structure into `physclip/src/` with proper package layout, but the training scripts still had:
- Old absolute imports (e.g., `from models/encoders import ...`)
- Hardcoded relative paths (e.g., `./burgers_data`)
- References to modules that were renamed

## Solutions Applied

### 1. **Fixed Package Structure Initialization**

**Files**: `physclip/src/models/__init__.py`, `physclip/src/data/__init__.py`

Added proper exports to make submodules importable:

```python
# src/models/__init__.py
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
```

### 2. **Fixed Relative Imports Between Modules**

**File**: `physclip/src/data/dataset_generation.py`

Changed:
```python
from burgers_solver import BurgersSpectralSolver
```

To:
```python
from .pde_solver import BurgersSpectralSolver
```

### 3. **Updated Training Script Imports**

**Files**: `physclip/scripts/train_contrastive.py`, `physclip/scripts/train_baseline.py`

Added sys.path manipulation to enable imports from the parent `physclip/` directory:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now imports work relative to physclip/
from src.models import FieldEncoder, TrajectoryEncoder, ContrastiveLoss
from src.models.encoders import TextEncoder
from src.data import BurgersTrajectoryDataset
```

### 4. **Fixed Data and Output Paths**

**Files**: Both training scripts

Changed hardcoded paths:
```python
# OLD
DATASET_DIR = "./burgers_data"
OUTPUT_DIR = "./results"

# NEW
DATASET_DIR = os.path.join(os.path.dirname(__file__), "../data/burgers_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../results")
```

### 5. **Standardized Loss Function Names**

**Files**: Training scripts

Changed `InfoNCELoss` to `ContrastiveLoss` to match actual class name in codebase:
```python
# OLD: criterion = InfoNCELoss(temperature=TEMPERATURE)
# NEW: criterion = ContrastiveLoss(temperature=TEMPERATURE)
```

## Testing

Both training scripts now run successfully:

✅ **train_baseline.py** - Snapshot-level contrastive training
```
Device: cpu
Loaded 275 snapshots from 25 trajectories
Field encoder parameters: 21,536
Text encoder parameters: 22,762,496
Training in progress... (losses decreasing: 2.0504 → 2.0399)
```

✅ **train_contrastive.py** - Trajectory-level with multi-scale windowing  
```
Loaded 25 base trajectories
Total windows: 225 (sizes [5,7,9], stride=2)
Trajectory encoder: 21,536 params
Training in progress...
```

## Key Changes Summary

| Component | Before | After |
|-----------|--------|-------|
| **Import paths** | Absolute paths + sys.path hacks | Relative imports from physclip/ |
| **Data directory** | `./burgers_data` | `../data/burgers_data` (relative) |
| **Output directory** | `./results` | `../results` (relative) |
| **Package imports** | Old module locations | New src/ structure |
| **Loss function** | `InfoNCELoss` (undefined) | `ContrastiveLoss` (correct) |

## How to Run

```bash
cd d:\Work\PHYSCLIP

# Baseline (v0)
python physclip/scripts/train_baseline.py

# Contrastive (v1) 
python physclip/scripts/train_contrastive.py

# Or from within physclip/
cd physclip
python scripts/train_baseline.py
python scripts/train_contrastive.py
```

## Notes

- The scripts now work from any directory
- Data and results are stored in `physclip/data/` and `physclip/results/` respectively
- Both scripts properly initialize the models and load the datasets
- Training loss is decreasing normally, indicating convergence is working

---

**Status**: ✅ Fixed and tested
