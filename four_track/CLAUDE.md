# CLAUDE.md - Execution Guidelines (four_track workspace)

> Canonical execution policy is in `.github/copilot-instructions.md`. This file
> retains extended examples for reference, scoped to the **four_track**
> workspace at `/home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track/`.
>
> The four_track directory is the active project home as of 2026-04-06.
> All new code, scripts, logs, models, and notebooks live under `four_track/`.
> The parent `BirdCLEF/src/` tree is read-only legacy and should only be
> imported from, never modified.

## ⚠️ ALWAYS Consult plan documents Before Any Action

**Before suggesting or starting any training run, experiment, or code change,
read both `four_track/new_plan.md` and (for legacy LB context) the parent
`BirdCLEF/plan.md`.**

`new_plan.md` is the single source of truth for the four-track strategy:
- The four tracks (A: SED on raw audio, B: second Perch consumer, C:
  ProtoSSM-as-teacher pseudo-labels, D: recalibration & stacking)
- Per-track gates and kill criteria
- Sequencing and Kaggle-slot budget
- What's intentionally NOT in the plan and why

`plan.md` (parent) is the historical LB submission log and experiment ledger
that motivated `new_plan.md`. Consult it whenever you need to know what has
already been tried.

Never propose the next step from memory or inference alone — always verify
against `new_plan.md` first.

## Environment Setup

### Conda Environment
This project uses the **kaggle** conda environment. **ALWAYS activate it
before running ANY command** — training, inference, `kaggle` CLI pushes, or
any Python script:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
```

⚠️ Plain `conda activate kaggle` fails in non-interactive shells (Bash tool).
Always use the `source` prefix above.

## Python Script Execution Policy

All Python scripts executed for this project **MUST** be run in the background
using `nohup` with log files written to the four_track log directory. The
`kaggle` conda environment must be active.

### ⚠️ NEVER use `conda run` for scripts that write log files

`conda run` buffers stdout/stderr internally — the log file will remain
**empty** while the process runs, making monitoring impossible. Always
activate the environment directly with `conda activate kaggle` before using
`nohup`:

```bash
# ✅ CORRECT — log file receives output immediately
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track
nohup bash scripts/train_a1_5fold.sh > log/train_a1_5fold_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ❌ WRONG — log file stays empty; no way to monitor progress
conda run -n kaggle nohup bash scripts/train_a1_5fold.sh > log/....log 2>&1 &
```

### Log Directory
```
/home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track/log
```

### Standard Execution Format

#### Prerequisites:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track
```

#### For any Python script, use:
```bash
nohup python -u src/<script_name>.py [arguments] \
  > /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track/log/<script_name>_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Or more concisely from the four_track directory (with kaggle active):
```bash
nohup python -u src/<script_name>.py [arguments] \
  > log/<script_name>_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Examples

#### Run Track A1 5-fold SED training:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track
nohup bash scripts/train_a1_5fold.sh > log/train_a1_5fold_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Run Track A1 single-fold smoke test:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track
nohup python -u src/train_a1.py --fold 0 --epochs 1 --smoke-test \
  > log/train_a1_smoke_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Run Track C2 pseudo-label generation:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track
nohup python -u src/c2_pseudo_label.py \
  > log/c2_pseudo_label_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Run ONNX export:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track
nohup python -u src/export_onnx_a1.py \
  > log/export_onnx_a1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Why This Approach?

1. **nohup** - Ensures process continues even if terminal disconnects
2. **Background execution** - Frees terminal for other tasks
3. **Timestamped logs** - Each run creates unique log file with timestamp
4. **Centralized logging** - All logs in `four_track/log/` for easy tracking
5. **Both stdout & stderr** - `2>&1` captures all output

### Monitoring Execution

#### View logs in real-time:
```bash
tail -f /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track/log/<log_file_name>.log
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
four_track/log/
├── train_a1_5fold_20260406_120000.log
├── train_a1_smoke_20260406_115500.log
├── c2_pseudo_label_20260407_080000.log
├── export_onnx_a1_20260408_100000.log
└── ...
```

### Important Notes

- Always create log files with timestamps to avoid overwriting previous runs
- Check log files regularly for errors or unexpected behavior
- Keep log files for reference and debugging
- Clean up old logs periodically if disk space becomes an issue
- The `four_track/log/` directory is created at workspace setup; do not
  delete it

---

## ⚠️ CRITICAL: Clean Log Directory Before Restarting Training

**Every time you restart training, ALWAYS clean up the old log files first.**

This prevents log file confusion and ensures you're tracking the correct
training run.

### Clean Logs Before Training

Before executing any training scripts, run:

```bash
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track
rm -f log/*.log
```

### Complete Workflow for Training Restart

```bash
# Step 1: Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle

# Step 2: Navigate to four_track project
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track

# Step 3: Clean ALL logs for this workspace
rm -f log/*.log

# Step 4: Start the chosen training run (A1 5-fold shown here)
nohup bash scripts/train_a1_5fold.sh > log/train_a1_5fold_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Step 5: Verify logs were created
tail -f log/train_a1_5fold_*.log
```

### Why Clean Logs?

1. **Avoid confusion** - Old logs from previous runs won't interfere
2. **Accurate monitoring** - `tail -f log/train_a1_*.log` shows current run only
3. **Cleaner tracking** - Each training session has fresh logs
4. **Prevent misinterpretation** - No mixing of loss/AUC curves from old runs
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

**Effective Date**: April 6, 2026
**Status**: Active
**Last Updated**: April 6, 2026 (workspace migrated to four_track/)

## Training Script Logging Conventions

All training scripts **MUST** include a per-epoch summary line with the
following format:

```
========================================
Fold F  Epoch  N/25: train_loss=X.XXXX  val_roc_auc=X.XXXX  time=Xm XXs  YYYY-MM-DD HH:MM:SS ★ BEST
========================================
```

**Required fields:**
1. **Fold index** when training a multi-fold loop
2. **Epoch time in `Xm XXs` format** — always show elapsed time per epoch as
   minutes and seconds (e.g. `8m22s`), not raw seconds
3. **Date/time stamp** — always include
   `time.strftime('%Y-%m-%d %H:%M:%S')` at epoch end so logs are
   self-documenting and finish times can be estimated
4. **`★ BEST` marker** — append ` ★ BEST` when the current epoch achieves a
   new best validation metric (val_roc_auc for SED branches, val_loss for
   loss-driven branches)

These must be implemented inline in the training loop or via a callback that:
- Tracks `best_metric` across epochs
- Records `epoch_start = time.time()` at the top of each epoch
- Computes `elapsed`, `mins, secs = divmod(elapsed, 60)` after validation
- Compares the current metric vs `best_metric` and sets the marker accordingly

See `four_track/src/train_a1.py` as the reference implementation for the
four_track workspace.
