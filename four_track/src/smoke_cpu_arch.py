"""CPU inference timing smoke for cross-arch backbone candidates.

Measures per-file forward-pass time on 4-thread CPU (mimics Kaggle's
submission-scoring environment), extrapolates to 700-file hidden test.

Required gate: extrapolated wall-clock < 30 min per arch (leaves headroom
for the other pipeline cells under the 90-min Kaggle cap).

Input shape mirrors production A1 cell 41: (12_windows, 3_channels,
N_MELS=224, T=512). One file = one batch of 12.

Usage:
  python -u src/smoke_cpu_arch.py [--backbone <timm_name>] ...
  (multiple --backbone flags can be passed)

Defaults to the three candidates from §29.7 + B0 baseline.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))

import timm  # noqa: E402

DEFAULT_BACKBONES = [
    "tf_efficientnet_b0.ns_jft_in1k",   # baseline reference (currently ships)
    "convnext_pico.d1_in1k",            # ~10M, ConvNeXt block (depthwise)
    "mobilevit_s.cvnets_in1k",          # ~6M, hybrid conv+transformer
    # tf_efficientnetv2_s killed §14.19; do not re-add without revisiting
]

INPUT_SHAPE = (12, 3, 224, 512)  # batch=12 windows per file, mel (224, 512)
N_WARMUP_BATCHES = 2
N_TIMED_FILES = 5
KAGGLE_HIDDEN_TEST_FILES = 700
KAGGLE_BUDGET_MIN = 90.0
PER_ARCH_KILL_MIN = 30.0  # leave 60 min for other pipeline cells


def _bench(backbone_name: str) -> dict:
    print(f"\n{'='*70}", flush=True)
    print(f"[arch] {backbone_name}", flush=True)
    print(f"{'='*70}", flush=True)

    try:
        # Plain backbone (no SED head) — gives a clean ceiling on inference
        # cost. Real production path wraps this in BirdSEDModelA1; head adds
        # only ~5-10% to the forward time, so the CPU gate decision is driven
        # by the backbone.
        model = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=234,
            in_chans=3,
        ).eval()
    except Exception as e:
        return {"backbone": backbone_name, "error": f"create_model: {e}"}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.1f}M", flush=True)

    x = torch.randn(*INPUT_SHAPE)
    try:
        with torch.no_grad():
            _ = model(x)
    except Exception as e:
        # Some backbones need different default input sizes
        return {"backbone": backbone_name, "params_m": n_params/1e6,
                "error": f"forward at {INPUT_SHAPE}: {e}"}

    # Warm-up
    with torch.no_grad():
        for _ in range(N_WARMUP_BATCHES):
            _ = model(x)

    # Timed
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N_TIMED_FILES):
            _ = model(x)
    elapsed = time.time() - t0
    per_file = elapsed / N_TIMED_FILES
    extrap_min = per_file * KAGGLE_HIDDEN_TEST_FILES / 60.0
    headroom_frac = extrap_min / KAGGLE_BUDGET_MIN

    verdict = (
        "PASS (CPU-fittable)" if extrap_min < PER_ARCH_KILL_MIN
        else f"FAIL (>{PER_ARCH_KILL_MIN:.0f} min projected for 700 files; "
             f"can't fit Kaggle 90-min cap)"
    )
    print(f"  {N_TIMED_FILES} files: {elapsed:.1f}s  →  {per_file:.2f}s/file",
          flush=True)
    print(f"  extrapolated to {KAGGLE_HIDDEN_TEST_FILES} files: "
          f"{extrap_min:.1f} min ({100*headroom_frac:.0f}% of {KAGGLE_BUDGET_MIN:.0f}-min cap)",
          flush=True)
    print(f"  verdict: {verdict}", flush=True)

    return {
        "backbone": backbone_name,
        "params_m": n_params / 1e6,
        "sec_per_file": per_file,
        "extrap_min_700": extrap_min,
        "verdict": verdict,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", action="append", default=None,
                   help="Repeatable. Defaults to {B0, ConvNeXt-Pico, MobileViT-S}.")
    p.add_argument("--threads", type=int, default=4,
                   help="torch.set_num_threads (default 4 to mimic Kaggle).")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    print(f"[setup] torch threads: {torch.get_num_threads()}", flush=True)
    print(f"[setup] input shape: {INPUT_SHAPE}", flush=True)
    print(f"[setup] Kaggle test: {KAGGLE_HIDDEN_TEST_FILES} files × "
          f"{INPUT_SHAPE[0]} windows; cap {KAGGLE_BUDGET_MIN:.0f} min", flush=True)

    backbones = args.backbone or DEFAULT_BACKBONES
    results = []
    for bb in backbones:
        results.append(_bench(bb))

    # Summary table
    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'backbone':<40s} {'params':>8s} {'sec/file':>10s} {'700-file min':>14s} {'verdict':>20s}",
          flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['backbone']:<40s} ERROR: {r['error']}", flush=True)
            continue
        verdict_short = "PASS" if "PASS" in r["verdict"] else "FAIL"
        print(f"{r['backbone']:<40s} {r['params_m']:>6.1f}M {r['sec_per_file']:>9.2f}s "
              f"{r['extrap_min_700']:>13.1f}m {verdict_short:>20s}",
              flush=True)


if __name__ == "__main__":
    main()
