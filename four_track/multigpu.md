# Multi-GPU plan for the iNat pretrain (and similar workloads)

## TL;DR — don't

For the current workload (B0 backbone, 50-epoch iNat pretrain) on the
current hardware (one 4080 on deepthought, one GB10 on skynet, no fast
interconnect), **multi-GPU makes the run slower, not faster**. Run the
pretrain single-GPU on deepthought as planned. This document records
*why*, *what we'd do if hardware changed*, and *what to actually
implement if the user insists despite the math*.

This is the same conclusion CLAUDE.md already reached for fold-parallel
training (sequential on deepthought beats split deepthought+skynet by
~2× wall-clock); DDP only makes the gap wider because it adds
synchronous gradient AllReduce on top of the same speed mismatch.

## Hardware constraints

| Host | GPU | VRAM | Arch | Per-step throughput (V2-S, ref) |
|---|---|---|---|---|
| deepthought | RTX 4080 | 16 GB GDDR6X | sm_89 | 1.0× (3.2 min/ep) |
| skynet | GB10 (DGX Spark) | 119 GB LPDDR5X unified | sm_121 | **0.22×** (13.4 min/ep) |

Per CLAUDE.md, the GB10 is structurally slow on training:
- LPDDR5X bandwidth ≈ 2.6× lower than 4080's GDDR6X — dominates
  memory-bound conv layers.
- sm_121 has no PyTorch/cuDNN/Triton kernel parity (PyTorch ships
  sm_120; runs via fallback). NVRTC JIT for sm_121 fails outright.
- Not on the public PyTorch/cuDNN roadmap as of 2026-05-05.

**Each machine has exactly one GPU** (no PCIe scale-up). So "multi-GPU"
here means *cross-machine over LAN* — there is no single-host DDP path.

LAN: assume 1 GbE/10 GbE Ethernet (no Infiniband, no NVLink).

## The four candidate strategies

### 1. Cross-machine DDP via NCCL/Gloo

Synchronous data-parallel training, gradient AllReduce per step.

**Why it loses:**

- **Slowest-GPU bottleneck.** Synchronous DDP forces the fast GPU to
  wait for the slow one at every gradient sync. Effective per-step
  time ≥ skynet's per-step time = **4.5× slower** than deepthought
  alone, before sync overhead.
- **AllReduce over Ethernet.** B0 has ~5 M params; one fp32 AllReduce
  per step = ~20 MB. At 1 GbE (~110 MB/s effective), that's ~180 ms
  per step. At 10 GbE, ~18 ms. Per-step compute on deepthought is
  ~250 ms (B0 BS=64), so 1 GbE adds ~70% to step time on top of the
  4.5× wait — net **5–6× slower** than deepthought alone.
- **NCCL doesn't actually like Ethernet.** It works (with
  `NCCL_IB_DISABLE=1` + `NCCL_SOCKET_IFNAME=<eth>`), but is tuned
  for Infiniband/NVLink. Gloo is the supported Ethernet path; even
  slower on this size of payload.

**Verdict:** strictly worse than deepthought-only single-GPU.

### 2. Independent seed-parallel runs (NOT multi-GPU)

Train two separate checkpoints — one per machine — with different
seeds, ensemble at finetune-init time.

**This is not really multi-GPU**; it's two single-GPU jobs.

- Wall-clock for the *first* checkpoint: deepthought finishes in
  ~2–3 days (canonical).
- Wall-clock for the *second* checkpoint: ~10–14 days on skynet
  (4.5× slower).
- Bundle 2 step 5 needs one backbone init for step 6's finetune;
  a 2nd init delivered 10 days later would change the experimental
  unit and confound the gate.

**When this would help:** if Phase 6 finetune validates the encoder
hypothesis and we want a 2nd seed for ensemble at inference. That's
a *follow-up* decision, not a multi-GPU strategy.

**Verdict:** not a multi-GPU answer. Belongs in a downstream ensemble
plan, not this one.

### 3. Pipeline / model parallelism (split layers across hosts)

Forward pass: deepthought computes layers 1–N₁, ships activations
to skynet, skynet computes N₁₊₁–N. Backward in reverse.

**Why it loses:**

- B0 fits in 16 GB on a single GPU at BS=64. There's no memory
  pressure motivating model parallelism.
- Activation transfer per step over LAN dwarfs compute — full mel
  feature maps for BS=64 are ~50–100 MB depending on chunk length;
  even at 10 GbE that's 50–100 ms per direction × forward + backward
  per step = **~400 ms of pure I/O wait per step**.
- Adds engineering overhead (must hand-partition the model) for
  zero throughput benefit.

**Verdict:** never makes sense for this workload size at this
interconnect speed.

### 4. ZeRO / FSDP across hosts

Partition optimizer state / gradients / params across two GPUs.
Same Ethernet bandwidth problem as DDP plus extra sharded parameter
gathers per step.

**Verdict:** same loss as DDP, more complexity.

## What would have to change for multi-GPU to be net-positive

- **Add a 2nd GPU to deepthought (or another single host).** A
  2-GPU PCIe-connected box would make DDP viable — NVLink or
  PCIe gen4 x16 has ~2 orders of magnitude more bandwidth than
  Ethernet, so AllReduce overhead drops to single-digit-ms per step.
  Hardware cost ≈ $1–2 K for a used 4080/4090.
- **Replace the GB10 with a 4080/4090-class GPU**, or get sm_121
  kernel parity in PyTorch + cuDNN + Triton. The latter is not on
  any public roadmap.
- **Add Infiniband or 25/40 GbE between the two hosts.** Plus the
  per-host NIC cost, plus the skynet speed mismatch is still there.
  The mismatch alone kills synchronous DDP regardless of LAN.

If exactly one of these changes happens, revisit. None of them are
in flight.

## If we still want to do it — concrete implementation plan

This is the part to read **only if** the user is overriding the
recommendation above and wants the work done anyway. None of these
steps deliver a net wall-clock win on current hardware; they only
deliver a working DDP setup that can be used the day hardware
changes.

### Phase A — make the script DDP-aware (one session, ~2 h)

1. **Add `torch.distributed` init.** In `pretrain_inat_sounds.py`,
   after argparse, before `main()` body:
   ```python
   if args.ddp:
       import torch.distributed as dist
       dist.init_process_group(
           backend="nccl",  # or "gloo" for Ethernet-only fallback
           init_method=args.dist_url,    # tcp://<master>:<port>
           world_size=args.world_size,
           rank=args.rank,
       )
       torch.cuda.set_device(0)  # always 0 — each host has one GPU
   ```

2. **Wrap model in DDP.** Replace the `model = …` line with:
   ```python
   model = model.to(device)
   if args.ddp:
       model = torch.nn.parallel.DistributedDataParallel(
           model, device_ids=[0], output_device=0,
           find_unused_parameters=False,  # B0 has no unused params
       )
   ```

3. **Switch sampler to `DistributedSampler` when DDP is on.**
   Replace the `WeightedRandomSampler` with a custom sampler that
   composes weighted-random with rank-sharding, OR drop the
   weighted sampler under DDP and accept the 82% Aves bias (not
   acceptable per §14.22.10.4 — see Risk below).

4. **Rank-0-only logging + checkpointing.**
   ```python
   is_main = (not args.ddp) or dist.get_rank() == 0
   if is_main:
       ...print/save...
   ```

5. **Per-epoch sampler reseed.**
   `train_sampler.set_epoch(epoch)` if using DistributedSampler.

6. **Add `--ddp`, `--world-size`, `--rank`, `--dist-url` argparse.**

### Phase B — dispatch wrapper (one session, ~1 h)

Need a launcher that fires `pretrain_inat_sounds.py` on both hosts
with consistent args.

1. Pick a master host (deepthought, since it's faster — sync
   barrier is at the master).
2. Pick a TCP port (e.g. 29500). Open in firewall on master.
3. Modify `~/bin/runon` (or write `runon-ddp`) to launch on rank 0
   first, capture its hostname/IP, then launch rank 1 on the other
   host with `--dist-url tcp://<master>:29500 --rank 1`.
4. NCCL env (NCCL backend over Ethernet):
   ```bash
   export NCCL_IB_DISABLE=1
   export NCCL_SOCKET_IFNAME=eth0  # adjust per host
   export NCCL_DEBUG=WARN
   ```
   If NCCL refuses to negotiate over the LAN (likely on first try),
   fall back to `backend="gloo"`.

### Phase C — smoke test (~30 min compute, hours of debug)

1. Smoke-test DDP with `--epochs 1 --smoke-test --ddp` on both hosts.
   Expected first-time issues:
   - NCCL handshake timeout → switch to Gloo.
   - DataLoader worker pickling errors with DistributedSampler →
     reduce `num_workers` to debug.
   - cuDNN nondeterminism between sm_89 and sm_121 → check both
     ranks compute the same loss after the first step's AllReduce
     (they will, modulo float epsilon).
2. Measure wall-clock per step. Compare against deepthought-only
   baseline. Expected outcome per the math above: 5–6× slower.
3. Decide whether to proceed with full 50-ep run or abort.

### Phase D — full run (canonical 50 ep)

If steps A–C cleared and the user still wants to proceed:
- Estimated wall-clock: **~10–14 days** (skynet-bound + sync
  overhead) vs ~2–3 days deepthought-only.
- Save deepthought-only as a parallel run for direct comparison
  (it will finish first by ~7 days).

## Risks specific to this script

- **WeightedRandomSampler is mandatory for non-Aves balance.**
  Per §14.22.10.4 of `new_plan.md`: without it, the encoder sees
  82% Aves clips and the +0.27 Insecta probe-v3 ceiling becomes
  structurally unreachable. The naive DDP swap to `DistributedSampler`
  drops the weighted draw. Need a `DistributedWeightedRandomSampler`
  (write it from scratch — `torch.utils.data.distributed.DistributedSampler`
  doesn't expose weights). About 50 lines of code.
- **Cross-arch numerical drift.** sm_89 vs sm_121 cuDNN/Triton paths
  differ; bf16 reductions across hosts will produce a checkpoint
  that's not bit-exact reproducible from either host alone. Probably
  fine for a backbone init (downstream finetune dominates), but
  formally not the same model two single-host runs would produce.
- **Single point of failure.** If skynet hangs (per
  `CLAUDE.md` GPU-memory-hygiene note — the GB10 has had silent
  kernel-level hangs mid-training), the DDP run hangs too. With
  single-host on deepthought, skynet hangs are not your problem.

## Recommendation

**Run the iNat pretrain single-GPU on deepthought as planned.**

If — after Phase 6 finetune validates the encoder hypothesis — we
want a 2nd-seed pretrained checkpoint for inference ensemble, queue
it as a *separate single-GPU run* on skynet (estimated ~14 days)
**after** the canonical run finishes. That's not multi-GPU; it's
just running two jobs on two machines, which is what the existing
`runon` workflow already supports.

Revisit this document if a 2nd local GPU is added to either host or
if NVIDIA ships sm_121 kernel parity. Until then, this is sealed.
