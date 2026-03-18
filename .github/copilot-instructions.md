# BirdCLEF+ 2026 — Copilot Instructions

**Task**: Multi-label audio classification of 234 wildlife species from PAM recordings in the Brazilian Pantanal. Metric: macro-averaged ROC-AUC on 5-second windows.

See [plan.md](../plan.md) for the full competition plan.

---

## Environment

- **Conda env**: `kaggle` — always activate before running Python
- **Hardware**: NVIDIA GB10 (Blackwell, 128 GB unified memory) — use BF16 for training
- **Project root**: `/home/swatson/work/MachineLearning/kaggle/BirdCLEF`

---

## Script Execution Policy

All Python scripts **must** run with `nohup` in the background with timestamped logs:

```bash
conda activate kaggle && cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF
rm -f log/train_*.log   # clean before each new training run
nohup python -u src/<script>.py [args] > log/<script>_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f log/<script>_*.log
```

- Core implementation in `.py` scripts only — **not** notebooks (notebooks are for Kaggle submission only)
- Shell scripts in `scripts/` must use **absolute paths**

---

## Key Pitfalls

- **Insect sonotypes** (e.g., `47158son16`) are valid class labels — treat as unique species
- **Secondary labels**: mask their loss to 0 (not 0.5) — location within clip is unknown
- **iNat clips**: no quality filter; XC clips: prefer rating ≥ 3.5
- **`torch.compile` + ONNX export** are incompatible — export un-compiled weights
- **BF16** does not need `GradScaler`
- **Shell scripts**: always use absolute paths
- **Logs**: clean `log/train_*.log` before each new training run
- **OpenVINO**: ~2× faster than ONNX but ~0.01 accuracy drop; `eca_nfnet_l0` fails — use ONNX only
- **Aves/BirdAves model**: +0.01 public LB but hurts private LB — don't include without LB gate
- **Minimum 10 samples per class**: duplicate rare clips before fold splitting
- **Cap external XC data at 500 records/class**

---

## Detailed Instructions (auto-loaded per file type)

- **Model architecture** → `.github/instructions/model.instructions.md` (`src/model.py`)
- **Dataset & audio pipeline** → `.github/instructions/dataset.instructions.md` (`src/dataset.py`, `src/utils.py`)
- **Training** → `.github/instructions/training.instructions.md` (`src/train*.py`, `src/self_train.py`)
- **Inference & ONNX** → `.github/instructions/inference.instructions.md` (`src/export_onnx.py`, `jupyter/**`)
