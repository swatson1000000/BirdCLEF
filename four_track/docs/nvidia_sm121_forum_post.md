# DRAFT — NVIDIA Developer Forum post requesting sm_121-tuned cuDNN/cuBLAS kernels

**Target forum:** https://forums.developer.nvidia.com/ — either the "DGX Spark" or "cuDNN" subforum (cuDNN is probably the better fit since the ask is specifically about kernel tuning).

**Suggested title:** Request for sm_121-tuned kernels in cuDNN/cuBLAS — DGX Spark training throughput gap

---

## Draft body

Hi NVIDIA team,

I'm running training workloads on a DGX Spark (GB10, compute capability 12.1 / sm_121) and observing a significant per-epoch throughput gap compared to a discrete consumer GPU (RTX 4080, sm_89) on the same code, dataset, and PyTorch version. I'd like to formally request that cuDNN/cuBLAS ship sm_121-tuned kernel paths, and ask whether there is any public roadmap commitment for this.

### Hardware and software

- DGX Spark — NVIDIA GB10 Blackwell, compute capability **12.1 (sm_121)**, 119 GB unified LPDDR5X
- Reference comparator — NVIDIA GeForce RTX 4080, compute capability 8.9 (sm_89), 16 GB GDDR6X
- Driver: NVIDIA 570.x series
- CUDA toolkit: 12.8
- PyTorch: stable channel (built with `TORCH_CUDA_ARCH_LIST` ending at sm_120 / compute_120 — no sm_121 binaries shipped as of this writing)
- cuDNN: 9.x bundled with PyTorch
- Workload: medium-scale audio classification training (multi-label CNN, EfficientNet-B0 backbone via timm, bfloat16 amp, batch size 64)

### Observed behavior

| GPU | Per-epoch wall-clock | Ratio vs sm_89 4080 |
|---|---|---|
| RTX 4080 (sm_89) | ~18.5 min | 1.0× (reference) |
| DGX Spark GB10 (sm_121) | ~52 min projected | **~2.8× slower** |

A second prior workload (EfficientNetV2-S, same family) measured an even wider gap of **~4.2× slower** on the same DGX Spark (13.4 min/epoch vs 3.2 min/epoch on the same 4080, same code, same seed, same data).

The GPU itself appears healthy — `nvidia-smi` shows brief utilization spikes to 96% during kernel execution windows, indicating the GPU IS doing real work when batches arrive. The gap is not a hardware fault.

### What I believe is happening

Per public discussion (and PyTorch's own arch-list at build time), **PyTorch stable currently ships sm_120 binaries**. GB10 is sm_121 — a different SM revision from sm_120 (RTX 50-series consumer Blackwell). On GB10:
- cuDNN convolution kernels and cuBLAS GEMMs fall back to generic / sm_120 paths rather than sm_121-tuned ones.
- NVRTC JIT compilation targeting sm_121 fails outright (no compiler support visible at our toolchain version).
- Triton heuristics use generic paths.

The combination of these — plus the GB10's lower memory bandwidth (~273 GB/s LPDDR5X vs ~717 GB/s GDDR6X on the 4080) — appears to dominate throughput on memory-bound depthwise / inverted-bottleneck convolutions, which are common in modern CNN backbones.

### My ask

1. **Is there a committed roadmap for sm_121-tuned kernels in cuDNN/cuBLAS?** Public posts I've seen from NVIDIA representatives acknowledged a DGX Spark training-throughput regression but stopped short of committing a timeline. A roadmap commitment — even a vague quarter — would help DGX Spark customers plan around this.

2. **Is there a near-term workaround on the toolchain side?** For example: an NGC PyTorch container with internal cuDNN tunings that aren't yet in the public release, a CUDA 13.x preview with broader compute-capability coverage, or a recommended environment configuration.

3. **Should DGX Spark customers expect the existing sm_120 fallback path to be the long-term plan?** If yes, that's a useful clarification — it would inform purchasing and capacity decisions. If no, what's the rough horizon?

### Context

I understand DGX Spark is positioned as a unified-memory development workstation and large-model inference rig, not primarily as a training accelerator. That's a reasonable product framing, but the current gap means many real-world training workloads (CV backbones, audio classifiers, anything memory-bound) run materially slower than on a consumer 4080 — and a sm_120-tuned cuDNN release would presumably close most of that gap without any hardware change.

A formal roadmap commitment, or an NGC-container path, would let customers like me plan deployment and budgeting decisions with less uncertainty.

Thanks in advance for any update you can share.

---

## Notes (for your eyes only, not for the post)

- I've kept the workload description generic ("audio classification training, EfficientNet-B0 backbone") — no mention of BirdCLEF / Kaggle.
- The two benchmark data points (B0-SED 2.8×, V2-S 4.2×) both came from CLAUDE.md's documented measurements. If you have any newer numbers you'd rather use, swap them in before posting.
- The "ask" section is intentionally three distinct asks. NVIDIA reps are more likely to respond to one of three than to a single broad complaint.
- Per `feedback_check_actual_lb_not_plan_lb` style discipline, the benchmark numbers above are reproducible from current code; I haven't introduced any speculative comparisons.
- If you have a DGX support contract, post this to the support portal as well — typically faster response than the forum.
