# CLAUDE.md - Execution Guidelines (four_track workspace)

> Canonical execution policy is in `.github/copilot-instructions.md`. This file
> retains extended examples for reference, scoped to the **four_track**
> workspace at `/home/swatson/work/kaggle/BirdCLEF/four_track/`.
>
> The four_track directory is the active project home as of 2026-04-06.
> All new code, scripts, logs, models, and notebooks live under `four_track/`.
> The parent `BirdCLEF/src/` tree is read-only legacy and should only be
> imported from, never modified — **with one documented exception**:
> portability fixes required for cross-machine dispatch (e.g.
> `config.py:22` ROOT was patched 2026-05-02 to honor `BIRDCLEF_ROOT`
> env var + dynamic `__file__`-based resolution so deepthought sees the
> mirror path, not skynet's hardcoded one). Behavior on skynet is unchanged.
> Don't extend this exception to add features; it's for portability only.

## 🖥️🖥️🖥️ Three machines — parallel execution is an option

This project has **three GPU-capable machines** available via passwordless SSH (one is currently driver-broken; see below):

- **skynet** (local): NVIDIA GB10, 119 GB unified — aarch64. Local commands run here directly. Conda env: `kaggle-arch` (the `kaggle` env is broken — see §14.22.8.1 of `new_plan.md`).
- **deepthought** (remote): NVIDIA RTX 4080, 16 GB — x86_64. Dispatch via `runon deepthought <cmd>`; pull results with `syncback deepthought`. Conda env: `kaggle`.
- **hal9000** (remote): AMD Ryzen 5 2400G + **NVIDIA GeForce GTX 1650 (TU116, ~4 GB VRAM, compute capability 7.5)**, 8 threads, 16 GB RAM, ~657 GB free disk — x86_64, RHEL 9.7. LAN IP 192.168.1.150. **Driver UP** (570.181, repaired 2026-05-13 via DKMS rebuild with `--no-drm`; see hal9000 section for the fix details). CUDA 12.8 toolkit at `/usr/local/cuda` (nvcc on disk; not on PATH by default). Modest third GPU lane; **GTX 1650 is ~10× slower than the 4080** so it's only useful for small models, inference probes, or development.

When you have **independent workloads** (different experiments, different folds, code-side script + GPU training), consider running them **in parallel** on the GPU machines. Default to deepthought for GPU-heavy training (~3-4× faster than skynet on most workloads); use skynet for CPU-bound work, I/O-heavy tasks, or whenever deepthought is already committed. hal9000 is a third CPU/light-GPU lane.

See "⚙️ Two-GPU workflow (deepthought + skynet)" further down this file for GPU routing rules, the 4:1 fold-split heuristic, and caveats; see "🖥️ hal9000 (third machine — driver fix in progress)" for hal9000-specific state. Don't dispatch long-running jobs to deepthought without first checking `ssh deepthought nvidia-smi` (multi-tenant).

## ⚠️ ALWAYS Consult plan documents Before Any Action

**Before suggesting or starting any training run, experiment, or code change,
read `four_track/new_plan.md` (active current state, ~200 lines).**

`new_plan.md` is the single source of truth for the four-track strategy:
- Current LB high + production baseline
- **⛔ Killed directions table** — grep this BEFORE recommending any new
  direction; each entry has a kill reason + memory pointer
- Active in-flight work + decision gates
- PICK UP HERE handoff for the next session

**For pre-A3 historical context (§1-§28, all probes/kills through 2026-05-16
morning, ~14k lines):** see `four_track/docs/new_plan_history.md`. Grep that
file when you need detail on a past experiment, but don't read it whole —
the active plan summarizes everything load-bearing.

`BirdCLEF/plan.md` (parent) is the older historical LB submission log
predating new_plan.md. Consult only if neither `new_plan.md` nor
`new_plan_history.md` answers your question.

Never propose the next step from memory or inference alone — always verify
against `new_plan.md`'s Killed Directions table and the PICK UP HERE section
first.

## NotebookLM Cross-Source Synthesis

When a task calls for synthesizing multiple external sources — competitor
notebook audits, paper digests, multi-document Q&A, cross-source technique
gap analysis — use the `notebooklm` skill rather than manually re-reading
or re-grepping.

**Good fits:**
- "What techniques from competitors haven't we tried?" (see §14.15 of
  `new_plan.md` for a worked example — and the caveat that pre-digested
  memory summaries yield shallow synthesis; raw competitor notebooks yield
  sharper results)
- "Does technique X appear in any of the reference writeups?"
- Summarizing a paper before applying its method

**Not a fit:**
- Numerical reasoning over LB deltas, fold metrics, or val gates — these
  stay local.
- Grep-able questions answerable directly against `new_plan.md` or the
  repo. Don't round-trip a one-line grep through NotebookLM.

**Invocation pattern** (all commands need `-n <notebook_id>` in parallel
or multi-session workflows to avoid context-file races):

```bash
notebooklm create "Title" --json                         # capture id
notebooklm source add -n <id> /path/to/file.md --json
notebooklm source list -n <id> --json                    # wait for ready
notebooklm ask -n <id> "question" --json                 # one-shot
notebooklm ask -n <id> -c <conv_id> "follow-up" --json   # continue
```

When a round returns generic or already-known results, sharpen the query
(strict "not in plan" filter, single-source allowed) before discarding
the exercise.

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

### Weights & Biases (opt-in experiment tracking)

W&B is installed on both training machines (skynet `kaggle-arch` env,
deepthought `kaggle` env) and authenticated for user `swatson1000000`
(default entity `swatson1000000-lumen`, project `birdclef-2026`). API
keys live in `~/.netrc` on each host.

**Wired only into `src/a2_train.py`** (and reachable via the
`src/a3_train.py` wrapper). Off by default. To enable for a run:

```bash
python -u src/a2_train.py --folds 0,1,2,3,4 --epochs 25 --loss asl \
  --pseudo-ratio 0.4 \
  --use-wandb --wandb-group my-sweep-tag --wandb-tag a2 --wandb-tag iter2
```

CLI args: `--use-wandb`, `--wandb-project` (default `birdclef-2026`),
`--wandb-group` (default `<prefix>-YYYYMMDD-HHMMSS`),
`--wandb-tag` (repeatable), `--wandb-run-prefix` (default `a2`;
per-fold runs land as `<prefix>-foldN` under a shared group).

Per-epoch logged: `epoch`, `train_loss`, `val_roc_auc`,
`best_val_roc_auc`, `epoch_seconds`, `lr`. Final summary:
`final_best_val_roc_auc`. Full config (backbone, hyperparams,
pseudo paths, save_dir, etc.) logged at run start.

**When to actually enable it:** hyperparameter sweeps (≥5 runs varying
LR / mixstyle-p / pseudo-ratio), cross-experiment val-curve plots, or
sharing run state. For one-off 5-fold runs, plaintext logs in `log/`
+ the per-epoch summary lines are usually enough.

**NOT wired into** `train_a1.py`, `train_a4_ast.py`, `c2_student_train.py`.
Mirror the a2_train.py pattern (lazy `import wandb` inside an
`if use_wandb:` guard) if you need it elsewhere.

**Smoke-test safety:** `--smoke-test` appends `_smoke` to the saved ckpt
filename so smoke runs cannot overwrite production ckpts at the same
fold's path (fixed 2026-05-16 after a near-miss).

See memory `reference_wandb_optin` for full usage notes.

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
cd /home/swatson/work/kaggle/BirdCLEF/four_track
nohup bash scripts/train_a1_5fold.sh > log/train_a1_5fold_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ❌ WRONG — log file stays empty; no way to monitor progress
conda run -n kaggle nohup bash scripts/train_a1_5fold.sh > log/....log 2>&1 &
```

### Log Directory
```
/home/swatson/work/kaggle/BirdCLEF/four_track/log
```

### Standard Execution Format

#### Prerequisites:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/kaggle/BirdCLEF/four_track
```

#### For any Python script, use:
```bash
nohup python -u src/<script_name>.py [arguments] \
  > /home/swatson/work/kaggle/BirdCLEF/four_track/log/<script_name>_$(date +%Y%m%d_%H%M%S).log 2>&1 &
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
cd /home/swatson/work/kaggle/BirdCLEF/four_track
nohup bash scripts/train_a1_5fold.sh > log/train_a1_5fold_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Run Track A1 single-fold smoke test:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/kaggle/BirdCLEF/four_track
nohup python -u src/train_a1.py --fold 0 --epochs 1 --smoke-test \
  > log/train_a1_smoke_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Run Track C2 pseudo-label generation:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/kaggle/BirdCLEF/four_track
nohup python -u src/c2_pseudo_label.py \
  > log/c2_pseudo_label_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Run ONNX export:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/kaggle/BirdCLEF/four_track
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
tail -f /home/swatson/work/kaggle/BirdCLEF/four_track/log/<log_file_name>.log
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
cd /home/swatson/work/kaggle/BirdCLEF/four_track
rm -f log/*.log
```

### Complete Workflow for Training Restart

```bash
# Step 1: Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle

# Step 2: Navigate to four_track project
cd /home/swatson/work/kaggle/BirdCLEF/four_track

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

## ⚠️ CRITICAL: GPU Memory Hygiene Between Epochs

All training scripts in this workspace **MUST** call
`torch.cuda.empty_cache()` and `gc.collect()` after every epoch and after
every fold. This is non-negotiable on this machine.

### Why

The host is an NVIDIA DGX Spark (GB10, compute capability 12.1, unified
memory: host RAM and GPU RAM share the same 119 GB pool). Without
explicit cleanup, the CUDA caching allocator and Python's reference graph
let fragmented blocks accumulate across epochs. On this hardware that
fragmentation has caused **silent kernel-level hangs** mid-training —
no Python traceback, no OOM message, system-level freeze that requires
a hard reboot. Empirically observed on `train_b2.py` 2026-04-25:
crash mid-epoch-8 with no error captured, journal goes silent until
manual reboot.

This is not the same as ordinary peak-memory pressure. Memory may report
healthy (e.g. 18% used) immediately before the hang.

### How

In every training script, at the bottom of the per-epoch loop (after the
epoch summary print), and at the bottom of the per-fold loop in `main`,
add:

```python
import gc  # at the top of the file

# … per-epoch loop …
print("=" * 40, flush=True)  # epoch summary
gc.collect()
torch.cuda.empty_cache()

# … per-fold loop in main …
fold_results.append((f, best_auc, save_path))
gc.collect()
torch.cuda.empty_cache()
```

`empty_cache()` returns cached but unused blocks to the OS so the next
epoch's allocations don't fight the previous epoch's fragmentation.
`gc.collect()` drops dangling Python references to tensors first, so
`empty_cache` actually has reclaimable blocks to release.

### Reference implementations

- `four_track/src/train_b2.py` — added 2026-04-25 after the silent hang
- `four_track/src/train_a1.py` — should match (apply this rule when next
  touching the file)

## ⚙️ Two-GPU workflow (deepthought + skynet)

This workspace has access to **two GPU machines**, both reachable from
skynet with passwordless SSH:

| Host | GPU | VRAM | Arch | Default role |
|---|---|---|---|---|
| `deepthought` (remote) | NVIDIA RTX 4080 | 16 GB | x86_64, CC 8.9 | **GPU workhorse for training** |
| `skynet` (local) | NVIDIA GB10 | 119 GB unified | aarch64, CC 12.1 | reserve: >16 GB or I/O-bound only |

### Default routing rule (effective 2026-05-03)

For any GPU training (arch swap, pretrain, finetune, fold runs),
**default to deepthought**. Only use skynet when one of:

1. The workload genuinely needs **>16 GB** GPU memory (large batch
   contrastive, ProtoSSM/large-LM, etc.).
2. The workload is **I/O-dominated** (downloads, dataset prep, audio
   transcoding) — compute speed is irrelevant so the slower GPU is fine.
3. Deepthought is **already busy** with another canonical run.

**Why:** Empirically validated 2026-05-03 on V2-S Phase 2 — GB10 is
**~4.5× slower** than the 4080 on V2-S training (13.4 vs 3.2 min/epoch
on the same code/seed/data). Two structural reasons: (a) GB10's
LPDDR5X unified memory has roughly **2.6× lower bandwidth** than the
4080's GDDR6X, which dominates throughput on memory-bound depthwise
and inverted-bottleneck convs; (b) GB10 is **sm_121** (CC 12.1,
GB10/DGX-Spark Blackwell) — a different SM revision from sm_120
(CC 12.0, RTX 50-series consumer Blackwell). Stable PyTorch ships
sm_120 binaries since 2.7.0, so on skynet we run sm_120 binaries on
an sm_121 device via fallback (verified locally: `arch_list` ends at
sm_120/compute_120; `device cap` reports (12,1)) and cuDNN/Triton
heuristics use generic paths instead of sm_121-tuned kernels. NVRTC
JIT compilation for sm_121 fails outright — the same trap as
`feedback_gb10_nvrtc_jit.md`. **No public PyTorch/cuDNN roadmap
targets sm_121** as of 2026-05-05; community heat is on sm_120
(RTX 5090). NVIDIA acknowledged the DGX Spark training-throughput
regression on their dev forum but committed no timeline. Don't plan
around "kernel tuning will catch up" — it's not on the public
backlog. NVIDIA pitches DGX Spark as a unified-memory development /
large-model-inference workstation, not a training accelerator.
Don't fight that.

Practical implication for fold parallelism: 4 folds *sequentially on
deepthought* (~5.5 h) finishes faster than folds 2,4 on deepthought +
folds 0,1 on skynet *in parallel* (~2.5 h DT + ~10 h skynet = 10 h
wall-clock, bounded by skynet). The "use both machines" framing in
prior sections of `new_plan.md` priced skynet ≈ 4080; that assumption
is wrong.

### How to dispatch (already installed)

The `runon` infrastructure handles code rsync + SSH + conda + nohup +
log capture. Three commands, all in `~/bin`:

```bash
runon-setup deepthought                              # one-time per project
runon deepthought python -u src/train.py --fold 2    # dispatch a job
syncback deepthought                                 # pull models/ + log/ back
```

Full doc: `~/work/MachineLearning/DOCUMENTATION/RUNON.md`. Per-host
config in `~/.runon.conf`. Memory entry:
`reference_runon_multi_machine.md`.

### When to actually use both machines in parallel

The default is "deepthought sequential". Add skynet only when the
parallel speedup beats waiting for deepthought, which is rarer than it
looks because skynet's per-job wall-clock is 4-5× deepthought's.

USE BOTH when:
- Deepthought is **already committed** to a long canonical run, and a
  *separate* I/O-bound or large-memory job needs to run now (e.g. an
  XC bulk download, a CPU inference probe).
- A workload that genuinely needs >16 GB sits alongside one that fits
  on the 4080.
- Deepthought has unrelated tenants on the GPU and we can't wait —
  skynet absorbs the GPU-bound job at its own slow pace.

DON'T use both when:
- A single arch / fold sweep can run sequentially on deepthought in
  reasonable time. **Default to this case for typical ML training.**
- Cross-machine DDP/DDP-NCCL would be needed (LAN bandwidth too low
  for sane gradient sync).
- Reproducibility of a *specific* fold matters (cross-machine cuDNN
  nondeterminism + arch differences mean fold-N on the two machines
  are not numerically identical; fine for ensembles, not for
  replication).
- The model is small enough that the parallelism overhead (rsync,
  syncback, env churn) eats most of the wall-clock gain.

### Allocation ratio when you DO split (4:1 favoring deepthought)

When the use-both criteria above are met and you're dividing N
independent tasks across both GPUs, allocate **~4 tasks to deepthought
for every 1 task to skynet**. The ratio comes from the empirical 4.5×
per-epoch speed gap — it balances wall-clock so neither machine sits
idle while the other finishes.

| Total tasks (folds, seeds, archs) | deepthought | skynet | Comment |
|---|---|---|---|
| 1–2 | all | 0 | overhead exceeds parallel gain — DT-only |
| 3–4 | all | 0 | DT-only within ~1% of split — DT-only |
| 5 | 4 | 1 | ~20% wall-clock saving |
| 8 | 6–7 | 1–2 | ~25% wall-clock saving |
| 10 | 8 | 2 | ~30% wall-clock saving |

Rounding rule: when the 4:1 split doesn't divide cleanly, give the
extra task to deepthought. The single slowest skynet task is always
the wall-clock floor, so adding more skynet load doesn't help — keep
it at the smallest count the ratio allows.

Concrete decision rule: for a fold sweep of N independent runs, only
queue skynet at all if N ≥ 5; below that, run all N sequentially on
deepthought.

### ⚠️ deepthought storage layout — there is no NVMe filesystem (read before placing data)

Verified by `lsblk` on 2026-05-11 after a 6-h wasted run that assumed
"/home is on NVMe":

| Device | Type | Size | ROTA | Backing FSs |
|---|---|---|---|---|
| `sda` | TOSHIBA HDWG440 | 3.6 T | **1 (HDD)** | LVM `rhel-pool00_tdata` → `/`, `/home`, `/var` |
| `sdb` | ST6000VX009 | 5.5 T | **1 (HDD)** | `/mnt/mytoshiba` (ext4) |
| `sdd` | WDC WD50NDZM | 4.5 T | **1 (HDD)** | unmounted |
| `nvme0n1` | PCIe SSD | 465.8 G | **0 (NVMe)** | LVM `rhel-pool00_tmeta` (3.8 G) + **`rhel-nvme_cache` (214 G, mounted at `/mnt/nvme`)** — swap removed 2026-05-11 |

**Practical consequence: every standard data path on deepthought is on
rotational disk.** This includes:

- `/home/swatson/...` → rhel-home → backed by **sda HDD**
- `/` and `/var` → also rhel-* → backed by **sda HDD**
- `/home/swatson/work/MachineLearning` is a **symlink → `/mnt/mytoshiba/MachineLearning`** (`sdb` HDD)
- Therefore `runon`'s remote root `/home/swatson/work/MachineLearning/_runon/BirdCLEF/...` lives on `sdb` HDD
- Anything written to `four_track/data/processed/` on deepthought lands on HDD, **regardless of what the path looks like**

Phrases like "free space on rhel-root LVM" refer to thin-pool capacity,
not NVMe. The 87 GB / 182 GB margin numbers you may see in older plan
sections are correct as free space but **wrong as NVMe.**

### ✅ `/mnt/nvme` — the only real NVMe filesystem on deepthought

**Current state (2026-05-11, post-expansion): 214 GB xfs**, mounted at
`/mnt/nvme`, persisted in `/etc/fstab`. Use this for any I/O-sensitive
data on deepthought — training caches, scratch tensors, dataset subsets.
Anything else (`/home`, `/mnt/mytoshiba`, `/mnt/mypassport`) is HDD.

```bash
ssh deepthought "df -hT /mnt/nvme"
# /dev/mapper/rhel-nvme_cache xfs 214G ... 122G ... /mnt/nvme
```

Currently houses the iNat mel cache (`/mnt/nvme/inat_mels/`, train + val
splits, ~120 GB total). Build history:
- 2026-05-11 ~17:00 — Initial 95 GB LV from 100 GB unallocated PFree on
  `nvme0n1p1`, no swap surgery.
- 2026-05-11 ~20:30 — Expanded by +119 GB by **removing the 120 GB
  `rhel-swap` LV** and adding its extents to `rhel-nvme_cache`. Final
  214 GB. After this change deepthought has **zero swap**; 61 GB RAM
  with no paging fallback. OOM-killer fires immediately on memory
  pressure.

#### Reversing the swap-removal (≠ trivial — XFS can't shrink in place)

`xfs_growfs` is grow-only. To get swap back, you must destroy the
214 GB `/mnt/nvme` LV (losing its contents) and rebuild a smaller one
alongside a recreated swap LV. Plan compute around this.

```bash
# 1) Save anything important off /mnt/nvme first (rsync to /mnt/mytoshiba
#    or wherever). The cache here can be REBUILT, but it takes ~3 h.
ssh deepthought "rsync -a /mnt/nvme/inat_mels /mnt/mytoshiba/.../inat_mels_backup/"

# 2) Tear down nvme_cache
ssh deepthought "sudo umount /mnt/nvme \\
  && sudo lvremove -y /dev/rhel/nvme_cache"

# 3) Recreate swap (120 GB pinned to NVMe PV)
ssh deepthought "sudo lvcreate -L 120G -n swap rhel /dev/nvme0n1p1 \\
  && sudo mkswap /dev/rhel/swap \\
  && sudo swapon /dev/rhel/swap"

# 4) Optionally recreate a smaller nvme_cache LV with remaining NVMe extents
#    (PFree on nvme0n1p1 will be ~95 GB after step 3)
ssh deepthought "sudo lvcreate -L 94G -n nvme_cache rhel /dev/nvme0n1p1 \\
  && sudo mkfs.xfs /dev/rhel/nvme_cache \\
  && sudo mount /dev/rhel/nvme_cache /mnt/nvme \\
  && sudo chown swatson:swatson /mnt/nvme"

# 5) Restore /etc/fstab swap line (original is saved at /etc/fstab.bak2,
#    the pre-removal backup)
ssh deepthought "sudo cp /etc/fstab.bak2 /etc/fstab \\
  && grep -E 'swap|nvme_cache' /etc/fstab"
```

If you don't need swap back (current state has plenty of RAM headroom),
just leave the layout as-is. The reversal is only worth doing if you
hit a memory-pressure incident that swap would have softened.

Don't write large datasets to `/home/swatson/work/...` expecting NVMe
speed — see the storage layout table above.

**Before placing any cache/mel/dataset where I/O matters:**

```bash
ssh deepthought "df --output=target,fstype,source <path>; lsblk -d -o NAME,ROTA,MODEL"
```

ROTA=1 on the backing device = HDD = expect random-read bottleneck. The
v3 (raw audio, ~30 h/epoch) and v4 (mel cache, also on HDD) failures
were both this misdiagnosis — the "NVMe optimization" was writing 94 GB
to the same spinning disk it was meant to escape.

#### Stage audio to `/mnt/nvme` before large-dataset training on DT

When a training run reads >~30 GB of audio per epoch (e.g. A2's 90K-row
pseudo-augmented manifest, or any future >2× baseline dataset), stage
the audio to `/mnt/nvme` *before* launching, not after. The runon-staged
default (`/mnt/mytoshiba/.../four_track/data/raw/...`) is HDD and the
random-read bottleneck shows up as per-epoch slowdown.

Measured 2026-05-15 on A2 (90K-row B0-SED retrain):
- B0 expected DT:skynet ratio ~3× (per
  `reference_b0_sed_skynet_dt_ratio.md`, measured 2026-05-10 at 28K rows)
- Actual A2 ratio: **2.72×** (DT 12.5 min/epoch vs skynet 34 min/epoch)
- Gap attributable to HDD random-read tax on the larger dataset

Staging pattern (one-time, ~30 min for ~120 GB of train_audio):

```bash
# 1) Check /mnt/nvme has room (need dataset size + 10% headroom)
ssh deepthought "df -h /mnt/nvme"

# 2) rsync the audio tree from the HDD path to /mnt/nvme
ssh deepthought "mkdir -p /mnt/nvme/birdclef_audio \\
  && rsync -a /mnt/mytoshiba/.../four_track/data/raw/birdclef_2026/train_audio/ \\
       /mnt/nvme/birdclef_audio/train_audio/"

# 3) Symlink (or pass --data-root) the runon-staged data dir to the NVMe copy
ssh deepthought "ln -sfn /mnt/nvme/birdclef_audio/train_audio \\
  /home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/data/raw/birdclef_2026/train_audio"
```

Expected payoff: ~20% per-epoch speedup on large-dataset DT runs.
**Not worth doing mid-run** — only stage when launching a new job that
will run for >4 hours. Skip for baseline-size (~28K rows) training,
where the I/O bottleneck is small and rsync overhead dominates.

#### Audit `/mnt/nvme` before staging — ask, then reclaim

Before staging any new data to `/mnt/nvme`, **audit the existing
contents first**. The 214 GB filesystem fills up quickly with leftover
caches from previous experiments (mel caches, scratch tensors, dataset
subsets). Stale content from killed/exhausted experiments is the most
common reason a staging step runs out of room.

Mandatory pre-staging check:

```bash
ssh deepthought "df -h /mnt/nvme; ls -la /mnt/nvme/; du -sh /mnt/nvme/* 2>/dev/null"
```

For each entry > a few GB, identify the originating experiment, then
check `new_plan.md` for its status. If the experiment is marked
exhausted/killed/superseded, the cache is reclaimable — **but always
ask the user before deleting**, even when the plan unambiguously
greenlights it. Disk reclaim is destructive; user confirmation is the
default per CLAUDE.md's "executing actions with care" policy.

Worked example (2026-05-15): `/mnt/nvme/inat_mels` (118 GB) was
identified as the iNat 2024 Sounds mel cache. `new_plan.md` §11941 and
§12575 mark the iNat pretrain lever exhausted. Raw audio source
(150 GB) preserved on DT HDD makes the cache regenerable in ~4h. User
confirmed, cache deleted, /mnt/nvme reclaimed from 92 GB free → 210 GB
free.

Patterns of what's commonly reclaimable:
- Mel/feature caches for experiments marked exhausted in the plan
- Pretrain ckpts already syncback'd to skynet (keep one canonical copy)
- Scratch tensors from probes that gate-failed
- Stale `.log` files from rsync-resume sessions

Patterns of what's NOT reclaimable without explicit user sign-off:
- Caches for in-flight experiments (check running PIDs first)
- Anything the user has flagged as "keep" in a recent session
- Data whose regeneration cost is unknown or > 1 day of compute

If the audit shows /mnt/nvme is already largely free (e.g. >150 GB
available), skip the audit step and proceed to stage.

### Mandatory pre-flight before dispatching long jobs to deepthought

deepthought is **multi-tenant** — 40+ unrelated conda envs (HunyuanVideo,
gemma3, opensora, etc.) belong to other work. Always check the GPU is
idle before queuing a multi-hour job:

```bash
ssh deepthought nvidia-smi
```

If something is using the 4080, either wait or run on skynet only. The
`runon` wrapper does NOT block on this — it's the caller's
responsibility. A surprise OOM mid-fold-2 because someone else started a
diffusion model burns hours of compute.

### Result collation

Results land on the machine that produced them. After all parallel jobs
complete, pull deepthought's results back to skynet for a single canonical
location:

```bash
syncback deepthought models/ log/
```

Then JIT export / Kaggle dataset push happens on skynet as usual.

### Caveats — already documented, don't re-derive

- **CC mismatch on JIT artifacts.** `torch.jit.trace`d ckpts may not
  cross-load between CC 12.1 (GB10) and CC 8.9 (4080). For Kaggle
  CPU export it doesn't matter; for cross-machine inference it does.
- **Different effective BATCH_SIZE.** skynet fits BS=64 trivially;
  4080 may need BS=32 or 24 for larger backbones. Document the BS
  per-fold; small bias washes out in the fold-mean ensemble.
- **Spark's broken `kaggle` env.** Has torchvision/torchaudio version
  skew (will be rebuilt from `BirdCLEF/environment.yml` after the live
  L2-redux download finishes ~May 7). Until then, deepthought is the
  ONLY machine where new training/probe runs can launch cleanly. See
  `new_plan.md` §14.20.3.



---

## 🖥️ hal9000 (third machine — driver fix in progress)

Added 2026-05-13. Probed via SSH; specs read from the live host. **Earlier in-session writeup said "no NVIDIA GPU" — that was wrong, based on `nvidia-smi` failing and `lscpu` only showing the integrated Vega. A full `lspci` enumeration on the user's pushback revealed a discrete GTX 1650.** Corrected below; the diagnostic is in §"Driver state" further down.

### Hardware + OS
| Field | Value |
|---|---|
| OS | Linux RHEL 9.7 (kernel 5.14.0-611.35.1.el9_7) |
| Arch | x86_64 |
| CPU | AMD Ryzen 5 2400G — 4 cores × 2 threads = **8 threads**, max 3.6 GHz. Integrated Radeon Vega iGPU drives the console/display. |
| Discrete GPU | **NVIDIA GeForce GTX 1650 (TU116)** — VID:DID `10de:2188`, on PCIe `01:00.0`. ~896 CUDA cores, ~4 GB VRAM, compute capability 7.5 (Turing). |
| GPU driver | **UP, driver 570.181 (repaired 2026-05-13).** dkms-3.4.0 from EPEL + `.run` installer with `--silent --dkms --no-drm` rebuilt and registered the kernel modules (`nvidia.ko`, `nvidia-modeset.ko`, `nvidia-uvm.ko`, `nvidia-peermem.ko`) at `/lib/modules/$(uname -r)/extra/`. **nvidia-drm.ko intentionally omitted** — driver 570.181 was built before the kernel's `drm_client_setup` symbol changed, so DRM-KMS won't link. This costs only kernel-mode display on the NVIDIA card; the iGPU drives console, and compute paths (nvidia, nvidia-uvm) work fully. `nvidia-smi` lists the GTX 1650 with 4 GB VRAM. `lsmod` shows `nvidia` + `nvidia_uvm` loaded. `dkms status` reports `nvidia/570.181, 5.14.0-611.35.1.el9_7.x86_64: installed` — auto-rebuilds on future kernel updates. |
| CUDA toolkit | CUDA 12.8 at `/usr/local/cuda` → `/etc/alternatives/cuda`. `nvcc --version`: `release 12.8, V12.8.61` (built Jan 2025). **Not on default PATH** — add `export PATH=$PATH:/usr/local/cuda/bin` to `~/.bashrc` if needed. `libcuda.so.1` resolves on `/usr/lib64` via ldconfig — Python can `ctypes.CDLL("libcuda.so.1")` from any env. |
| RAM | **16 GB total** |
| Disk | ~657 GB free on `/home` (800 GB filesystem), 420 GB free on `/`, 43 GB free on `/var` |
| LAN | 192.168.1.150 — sub-ms ping from skynet (0.275 ms avg) |
| sudo | **Passwordless full root** (verified via `sudo -nl` — `(ALL) NOPASSWD: ALL`). |
| kernel-devel | `kernel-devel-5.14.0-611.35.1.el9_7` is installed (matches running kernel). `kernel-headers` is slightly newer at `5.14.0-611.55.1.el9_7` — non-issue for module build but worth flagging. |
| Conda | **Installed 2026-05-13** at `~/miniconda3` (matches skynet's path). conda 26.3.2, Python 3.13.13, libmamba solver, conda-forge as the sole channel (strict priority). Base env auto-activates via `~/.bashrc`. 841 MB on disk. |
| Project env | **`kaggle-cpu` built 2026-05-13.** 7.1 GB at `~/miniconda3/envs/kaggle-cpu` (name retained for historical reasons; env contains CUDA torch). Python 3.11.15, numpy 2.4.4, pandas 3.0.3, soundfile 0.13.1 (pip), Kaggle CLI 2.1.2. `~/.kaggle/kaggle.json` copied from skynet — `kaggle` CLI calls authenticate cleanly. **torch 2.11.0+cu128 + torchaudio 2.11.0+cu128** installed via pip with bundled NVIDIA CUDA libs (cublas, cudnn-9.19, cufft, curand, nccl, nvjitlink, triton). Smoke test verified: `torch.cuda.is_available()=True`, GTX 1650 (sm_75), 2048×2048 fp32 matmul at ~1 TFLOPS (~10× slower than the 4080). 3.7 GB usable VRAM — tight for SED training (A1 needs BS≤16-32 on this card), reasonable for inference probes. |

### Known constraints — read before dispatching

1. **Memory pressure on arrival.** On first probe, `SetroubleshootPrivileged.py`
   (root-owned SELinux audit process, running since 2026-04-18) was
   consuming **13.3 GB / 16 GB RAM**. Available RAM was ~1.2 GB. Until
   that's resolved (root needs to restart the service or restart the
   machine), the practical memory budget for our jobs is **~1-2 GB**.
   Anything larger will OOM-kill on hal9000. Not a hardware ceiling, but
   a current-state ceiling.

2. **Conda + `kaggle-cpu` env are installed and working.** Both base
   (conda 26.3.2) and the `kaggle-cpu` env (Python 3.11.15 + numpy +
   pandas + soundfile + kaggle CLI) were built 2026-05-13 inside the
   1.2-2.5 GB available-RAM envelope. Phased install (one solve for
   `python+kaggle`, one for `numpy+pandas`, pip for `soundfile`) kept
   each solve below the memory ceiling. **A full `environment.yml`
   build from the project root would still OOM** — that pulls torch,
   torchaudio, librosa, scipy, scikit-learn, transformers, timm, etc.,
   and the bulk solve needs 2-4 GB. Add packages incrementally if
   needed; don't `conda env create -f environment.yml` until the RAM
   hog in #1 is cleared. The broken `kaggle` env on skynet (per
   `new_plan.md` §14.20.3) is a separate warning that the full env
   builds in this project aren't trivial even with plenty of RAM.

3. **No GPU.** Anything that needs CUDA fails immediately. Don't try
   to send training, mel-cache build with torch, or Perch ONNX inference
   here (Perch ONNX on CPU would also be slow + memory-heavy).

4. **System Python is RHEL-stock.** Likely 3.9 with limited site-packages.
   Don't assume `numpy`, `pandas`, `librosa` are available globally.
   Probe before dispatching any one-liner.

### What hal9000 is good for (narrow list)

Treat hal9000 as a **third CPU lane for genuinely independent CPU
work** — not a replacement for skynet's CPU or deepthought's CPU when
their GPUs are idle. Right uses:

- **Kaggle CLI pushes** (`kaggle kernels push`, `kaggle datasets version`)
  once the env is provisioned. These are I/O-bound on the upload side
  and don't compete with GPU work.
- **Manifest joins / CSV transforms** that fit in ~1-2 GB RAM.
- **JIT/ONNX inspection** with system tools (lightweight read-only).
- **Long-running file downloads** that the GPU machines shouldn't be
  tied up on (e.g., XC v3 incremental fetch). Network throughput from
  this host is unknown; characterize before relying on it.
- **A shell to run `kaggle competitions submissions` polling, leaderboard
  fetches, deadline checks** — sub-ms LAN to skynet, lightweight.

### What hal9000 is NOT good for

- ❌ ML training of any kind (no GPU; 8 threads + 16 GB RAM ≈ 100×
  slower than the GPU machines on conv backbones).
- ❌ Mel-spectrogram cache builds — even the smallest BirdCLEF cache
  builds peak at >4 GB RAM and shuttle ~100 GB of audio. hal9000 has
  neither the RAM headroom nor the disk locality.
- ❌ Perch / ProtoSSM ONNX inference on real data — high memory + slow
  on CPU. Use Kaggle for those (the embedding-mismatch constraint in
  `new_plan.md` says Perch inference must happen Kaggle-side anyway).
- ❌ `runon` dispatch via the existing wrapper. `~/.runon.conf` has no
  hal9000 entry, and the conda-env-default pattern doesn't apply
  (no `kaggle` env exists). Use plain `ssh hal9000 <cmd>` with explicit
  paths until provisioning is done.

### Dispatch pattern

```bash
# kaggle-cpu env activation (project env), matching skynet's
# source ~/miniconda3/etc/profile.d/conda.sh pattern:
ssh hal9000 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-cpu && python -c 'import numpy, pandas, soundfile; print(numpy.__version__, pandas.__version__, soundfile.__version__)'"

# Kaggle CLI (authenticated via ~/.kaggle/kaggle.json copied from skynet):
ssh hal9000 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-cpu && kaggle competitions list --search birdclef"

# Manifest joins, CSV transforms, lightweight Python one-liners — fit in 1-2 GB:
ssh hal9000 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-cpu && python -c '<your script>'"
```

For non-interactive shells, the explicit `source ~/miniconda3/etc/profile.d/conda.sh`
prefix is required (same as skynet). Login shells get conda on PATH via
`~/.bashrc` auto-activation, but `ssh hal9000 "<cmd>"` uses a non-login
shell.

### Provisioning state (2026-05-13, end of session)

What works today:
- ✅ Conda 26.3.2 at `~/miniconda3`, conda-forge as the sole channel
- ✅ `kaggle-cpu` env (Python 3.11.15, numpy, pandas, soundfile, kaggle CLI, **torch 2.11.0+cu128 + torchaudio 2.11.0+cu128** with bundled CUDA libs + cudnn 9.19)
- ✅ Kaggle credentials at `~/.kaggle/kaggle.json` (auth verified)
- ✅ NVIDIA driver 570.181 + DKMS auto-rebuild registered
- ✅ CUDA 12.8 toolkit at `/usr/local/cuda`
- ✅ `~/.runon.conf` has hal9000 entries (SSH target, remote root `/home/swatson/work/MachineLearning/_runon`, conda env `kaggle-cpu`, conda.sh path). `runon hal9000 <cmd>` and `syncback hal9000` work the same as for deepthought. Smoke test 2026-05-13 18:43 EDT confirmed end-to-end: `runon hal9000 python -u src/foo.py` activates kaggle-cpu, imports torch, reaches the GPU.
- ✅ **RAM hog resolved.** During the torch+CUDA pip install, OOM-killer reaped the old SetroubleshootPrivileged.py PID 2487522 (which had grown to 13 GB over 3 weeks of accumulated SELinux denials). Current state: 13 GB RAM available. New PID 1643464 spawned on demand per denial, exits cleanly. `setroubleshootd` itself is healthy (110 MiB current, 267 MiB peak).

Known issues / future-work items:
1. **mongod-vs-snapd SELinux denial flood.** `setroubleshootd` is still at 40% sustained CPU because mongod keeps tripping denials on `/var/lib/snapd`. Not currently a memory issue — short-lived helper processes — but the underlying noise could re-leak someday. Real fix would be writing a local SELinux policy module to allow mongod's snapd accesses, or stopping mongod if it's not needed. Not in scope until it causes problems again.
2. **`runon-setup hal9000` partially fails on env update.** `runon-setup` runs `conda env update -f environment.yml` against the existing env; the project's `environment.yml` references the `defaults` channel which hits Anaconda's commercial ToS gate (`CondaToSNonInteractiveError` on `pkgs/main` and `pkgs/r`). The smoke test for actual `runon hal9000` dispatch is unaffected — `runon-setup` is only used for one-time provisioning and the env is already built. To make `runon-setup` clean, either (a) accept Anaconda's commercial ToS via `conda tos accept`, or (b) edit `environment.yml` to use conda-forge exclusively, or (c) remove the `runon-setup` env-update step (it's optional for an already-provisioned env).
3. **Optional**: extend `kaggle-cpu` with scipy + sklearn + librosa if a workload needs them. Currently env has the basics + torch GPU stack but not the wider scientific stack. Add incrementally.

Don't half-ship by running serious workloads in `base` expecting project
deps — `base` only has the conda toolchain. Activate `kaggle-cpu` for
anything that imports numpy/pandas/soundfile/kaggle/torch.

### Pre-flight checks (mandatory before any dispatch)

```bash
ssh hal9000 "free -h | head -2; nproc; df -h /home | tail -1"
```

If available RAM is < 2 GB, the SetroubleshootPrivileged.py issue
hasn't been cleared — defer or escalate to root.
