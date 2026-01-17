# PHYSCLIP Quick Reference

## 🚀 Getting Started

### Installation
```bash
cd physclip
pip install -e .
# or: pip install -r requirements.txt
```

### Verify Environment
```bash
python -c "from physclip.src.utils import check_gpu; check_gpu()"
```

---

## 📁 Project Layout

```
physclip/
├── src/                  ← Install with: pip install -e .
│   ├── models/          ← Neural network modules
│   ├── data/            ← Data generation and loading
│   └── utils/           ← Helper functions
├── scripts/             ← Training scripts
├── tests/               ← Unit tests
├── docs/                ← Documentation
├── data/                ← Data storage (git-ignored)
└── results/             ← Training outputs (git-ignored)
```

---

## 🔧 Common Tasks

### Generate Data
```python
from physclip.src.data import BurgersDataset

# Create dataset
dataset = BurgersDataset(
    nu_values=np.linspace(0.01, 0.2, 20),
    n_trajectories_per_nu=10,
    output_dir="physclip/data/burgers_data"
)
```

### Load Trajectories
```python
from physclip.src.data import BurgersTrajectoryDataset

dataset = BurgersTrajectoryDataset(
    trajectories_dir="physclip/data/burgers_data",
    window_sizes=[5, 7, 9],
    stride=2
)
```

### Train Baseline (v0)
```bash
python physclip/scripts/train_baseline.py
```

### Train Contrastive (v1)
```bash
python physclip/scripts/train_contrastive.py
```

### Run Tests
```bash
pytest physclip/tests/ -v
```

---

## 📊 Key Files

| File | Purpose |
|------|---------|
| `src/models/encoders.py` | FieldEncoder - spatial encoding |
| `src/models/trajectory_models.py` | TrajectoryEncoder - temporal encoding |
| `src/models/contrastive_loss.py` | InfoNCE loss function |
| `src/data/pde_solver.py` | Burgers equation solver |
| `src/data/dataset_generation.py` | Dataset classes |
| `scripts/train_baseline.py` | Snapshot-level training |
| `scripts/train_contrastive.py` | Trajectory-level training |

---

## 📚 Documentation

- **[ARCHITECTURE.md](physclip/docs/ARCHITECTURE.md)** - System design
- **[USAGE.md](physclip/docs/USAGE.md)** - Detailed API reference
- **[METHODS.md](physclip/docs/METHODS.md)** - Experimental results
- **[STRUCTURE.md](physclip/docs/STRUCTURE.md)** - Directory organization

---

## 🎯 Hyperparameters

### Baseline (v0)
```python
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
TEMPERATURE = 0.07
```

### Contrastive (v1)
```python
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
WINDOW_SIZES = [5, 7, 9]
WINDOW_STRIDE = 2
TEMPERATURE = 0.07
```

---

## 🐛 Troubleshooting

### Import Error: "No module named 'physclip'"
```bash
cd physclip
pip install -e .
```

### GPU Not Available
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Data Not Found
Ensure `physclip/data/burgers_data/` exists with `.npy` files

---

**Status**: Research-ready | **Python**: 3.10+ | **PyTorch**: 2.0+
