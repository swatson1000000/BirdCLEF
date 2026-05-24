"""§14.19 Phase 1.5 — CPU inference budget probe for EffNetV2-S arch swap.

One-off measurement script (NOT for production). Loads the V2-S smoke ckpt,
JIT-traces it, then times forward passes on CPU constrained to 4 threads
(mimicking Kaggle's CPU-only kernel constraint).

Decision rule (from new_plan.md §14.19.5 Phase 1.5):
  Hard kill if extrapolated A1 inference time > 78 min on the hidden test
  (~700 files × 12 chunks × 4 folds = ~33,600 forwards).
"""

import sys
import time
from pathlib import Path

import torch

HERE    = Path(__file__).resolve().parent       # four_track/src/
FT_ROOT = HERE.parent                           # four_track/
ROOT    = FT_ROOT.parent                        # BirdCLEF/

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config                                   # noqa: E402
from model_a1 import BirdSEDModelA1            # noqa: E402

# ── Constants matching the production inference notebook (Cell 41) ────────────
BATCH       = 16            # A1_BATCH in the notebook
N_MELS      = 224
N_FRAMES    = 512           # 20 sec at HOP=1252, SR=32000
HIDDEN_TEST_FILES = 700     # estimated; production uses MODE-detected count
CHUNKS_PER_FILE   = 12      # 60 sec / 5 sec window
N_FOLDS     = 4
N_FORWARDS_TOTAL = HIDDEN_TEST_FILES * CHUNKS_PER_FILE * N_FOLDS

# A1 budget = 90 min total - Perch+ProtoSSM stack overhead. Updated 2026-05-02
# from new_plan.md §14.19.4 risk #2: stack overhead ~12 min, A1 budget ~78 min.
A1_BUDGET_SEC = 78 * 60

# ── Constrain CPU threads to mimic Kaggle 4-vCPU kernel ───────────────────────
# (Kaggle's free CPU-only kernel exposes 4 vCPUs.)
N_THREADS = 4
torch.set_num_threads(N_THREADS)
torch.set_num_interop_threads(1)

print("=" * 60, flush=True)
print("§14.19 Phase 1.5 — CPU inference budget probe (EffNetV2-S)")
print("=" * 60, flush=True)
print(f"  threads        : {N_THREADS} (taskset constraint should match)")
print(f"  batch          : {BATCH}")
print(f"  input shape    : (B, 3, {N_MELS}, {N_FRAMES})")
print(f"  total forwards : {N_FORWARDS_TOTAL:,} for {HIDDEN_TEST_FILES} files")
print(f"  budget         : {A1_BUDGET_SEC} sec ({A1_BUDGET_SEC/60:.0f} min)")
print(f"  per-batch budget : {A1_BUDGET_SEC / (N_FORWARDS_TOTAL / BATCH):.3f} sec")
print()

# ── Instantiate V2-S BirdSEDModelA1 with random init (weights don't affect ────
# inference time — only FLOPs/topology matter for this CPU timing benchmark).
BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"
print(f"Instantiating BirdSEDModelA1({BACKBONE}) on CPU (random init) …", flush=True)
model = BirdSEDModelA1(backbone_name=BACKBONE, mixstyle_p=0.0)
model.eval()
model.mixstyle.active = False


class A1Wrapper(torch.nn.Module):
    """Strip dict return → tensor (matches export_a1_jit.py:A1Wrapper)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)["clip_logits"]


print("JIT-tracing on CPU …", flush=True)
wrapper = A1Wrapper(model).eval()
example = torch.zeros(1, 3, N_MELS, N_FRAMES)
with torch.no_grad():
    traced = torch.jit.trace(wrapper, example, strict=False)

# Sanity: verify output shape
out = traced(example)
print(f"  trace OK: input {example.shape} → output {out.shape}", flush=True)
print(f"  params : {sum(p.numel() for p in model.parameters())/1e6:.1f} M", flush=True)

# ── Benchmark ────────────────────────────────────────────────────────────────
print()
print("Warming up (3 forwards, BS=16) …", flush=True)
batch_input = torch.zeros(BATCH, 3, N_MELS, N_FRAMES)
with torch.no_grad():
    for _ in range(3):
        traced(batch_input)

N_TIMING_BATCHES = 10
print(f"Timing {N_TIMING_BATCHES} forward passes at BS={BATCH} …", flush=True)
times = []
with torch.no_grad():
    for i in range(N_TIMING_BATCHES):
        t0 = time.time()
        traced(batch_input)
        dt = time.time() - t0
        times.append(dt)
        print(f"  pass {i+1:2d}: {dt:.3f} sec", flush=True)

import statistics
mean_t = statistics.mean(times)
median_t = statistics.median(times)
stdev_t = statistics.stdev(times) if len(times) > 1 else 0.0

print()
print("=" * 60)
print("Per-batch timing (BS=16):")
print(f"  mean    : {mean_t:.3f} sec  ± {stdev_t:.3f}")
print(f"  median  : {median_t:.3f} sec")
print()

# Extrapolation
n_batches = N_FORWARDS_TOTAL / BATCH
extrap_total = mean_t * n_batches
print("Hidden-test extrapolation:")
print(f"  forwards needed  : {N_FORWARDS_TOTAL:,}  ({n_batches:.0f} batches @ BS={BATCH})")
print(f"  mean total time  : {extrap_total:.0f} sec  ({extrap_total/60:.1f} min)")
print(f"  budget           : {A1_BUDGET_SEC} sec ({A1_BUDGET_SEC/60:.0f} min)")
print(f"  margin           : {A1_BUDGET_SEC - extrap_total:.0f} sec  "
      f"({100 * (A1_BUDGET_SEC - extrap_total) / A1_BUDGET_SEC:+.1f}%)")
print()

if extrap_total <= A1_BUDGET_SEC:
    print(f"VERDICT: PASS  (under budget by {(A1_BUDGET_SEC - extrap_total)/60:.1f} min)")
    print("  → §14.19 Phase 1.5 gate cleared; arch swap CPU-feasible.")
else:
    print(f"VERDICT: FAIL  (over budget by {(extrap_total - A1_BUDGET_SEC)/60:.1f} min)")
    print("  → §14.19 Phase 1.5 gate FAILED; arch swap structurally infeasible.")
    print("  → Same fate as B2 (kernel timeout on hidden-test re-run).")
print("=" * 60)
