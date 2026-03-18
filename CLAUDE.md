# CLAUDE.md - Execution Guidelines

> Canonical execution policy is in `.github/copilot-instructions.md`. This file retains extended examples for reference.

## Environment Setup

### Conda Environment
This project uses the **kaggle** conda environment. Before executing any Python scripts, activate this environment:

```bash
conda activate kaggle
```

## Python Script Execution Policy

All Python scripts executed for this project **MUST** be run in the background using `nohup` with log files written to the project log directory. The `kaggle` conda environment must be active.

### Log Directory
```
/home/swatson/work/MachineLearning/kaggle/BirdCLEF/log
```

### Standard Execution Format

#### Prerequisites:
```bash
conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF
```

#### For any Python script, use:
```bash
nohup python -u src/<script_name>.py [arguments] > /home/swatson/work/MachineLearning/kaggle/BirdCLEF/log/<script_name>_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Or more concisely from the project directory (with kaggle active):
```bash
nohup python -u src/<script_name>.py [arguments] > log/<script_name>_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Examples

#### Setup and run EDA:
```bash
conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF
nohup python -u src/eda.py > log/eda_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Setup and run Stage 1 training:
```bash
conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF
nohup bash scripts/train_stage1.sh > log/train_stage1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Setup and run pseudo-label generation:
```bash
conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF
nohup python -u src/pseudo_label.py > log/pseudo_label_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Setup and run self-training:
```bash
conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF
nohup bash scripts/self_train_stage2.sh > log/self_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Setup and run ONNX export:
```bash
conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF
nohup python -u src/export_onnx.py > log/export_onnx_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Why This Approach?

1. **nohup** - Ensures process continues even if terminal disconnects
2. **Background execution** - Frees terminal for other tasks
3. **Timestamped logs** - Each run creates unique log file with timestamp
4. **Centralized logging** - All logs in `/log` directory for easy tracking
5. **Both stdout & stderr** - `2>&1` captures all output

### Monitoring Execution

#### View logs in real-time:
```bash
tail -f log/<log_file_name>.log
```

#### Check background processes:
```bash
jobs -l
ps aux | grep python
```

#### Stop a running process:
```bash
kill <PID>
# or force kill if needed:
kill -9 <PID>
```

### Log Directory Structure

The log directory will contain timestamped files like:
```
log/
├── preprocessing_20260202_120000.log
├── training_seq2seq_20260202_130000.log
├── training_mbart_20260202_140000.log
├── inference_20260202_150000.log
└── ...
```

### Important Notes

- Always create log files with timestamps to avoid overwriting previous runs
- Check log files regularly for errors or unexpected behavior
- Keep log files for reference and debugging
- Clean up old logs periodically if disk space becomes an issue
- The log directory is already created in the project structure

---

## ⚠️ CRITICAL: Clean Log Directory Before Restarting Training

**Every time you restart training, ALWAYS clean up the old log files first.**

This prevents log file confusion and ensures you're tracking the correct training run.

### Clean Logs Before Training

Before executing any training scripts, run:

```bash
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF
rm -f log/*.log
```

### Complete Workflow for Training Restart

```bash
# Step 1: Activate environment
conda activate kaggle

# Step 2: Navigate to project
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF

# Step 3: Clean ALL logs
rm -f log/*.log

# Step 4: Start ByT5 training
nohup python -u src/train_byt5.py --epochs 20 --output-dir models/byt5-akkadian > log/train_byt5_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Step 5: Verify logs were created
tail -f log/train_byt5_*.log
```

### Why Clean Logs?

1. **Avoid confusion** - Old logs from previous runs won't interfere
2. **Accurate monitoring** - `tail -f log/train_seq2seq_*.log` shows current run only
3. **Cleaner tracking** - Each training session has fresh logs
4. **Prevent misinterpretation** - No mixing of loss curves or metrics from old runs
5. **Easier debugging** - If training fails, you know which log to check

### Quick Commands

```bash
# Clean all training logs
rm -f log/train_*.log

# Clean all logs
rm -f log/*.log

# View cleaned log directory
ls -la log/
```

---

**Effective Date**: February 2, 2026  
**Status**: Active  
**Last Updated**: February 24, 2026

## Training Script Logging Conventions

All training scripts **MUST** include a per-epoch summary callback with the following format:

```
========================================
Epoch  N/20: train_loss=X.XXXX val_loss=X.XXXX BLEU=XX.XX chrF++=XX.XX GeoMean=XX.XX time=Xm XXs  YYYY-MM-DD HH:MM:SS ★ BEST
========================================
```

**Required fields:**
1. **Epoch time in `Xm XXs` format** — always show elapsed time per epoch as minutes and seconds (e.g. `8m22s`), not raw seconds
2. **Date/time stamp** — always include `time.strftime('%Y-%m-%d %H:%M:%S')` at epoch end so logs are self-documenting and finish times can be estimated
3. **`★ BEST` marker** — append ` ★ BEST` when the current epoch achieves a new best validation loss

These must be implemented via a `TrainerCallback` (or equivalent) that:
- Tracks `best_val_loss = float("inf")` across epochs
- Records `epoch_start = time.time()` in `on_epoch_begin`
- Computes `elapsed`, `mins, secs = divmod(elapsed, 60)` in `on_evaluate`
- Compares `val_loss < best_val_loss` and sets the best marker accordingly

See `src/train_toda.py` → `EpochSummaryCallback` as the reference implementation.
