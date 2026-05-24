# BirdCLEF+ 2026 — Plan to reach LB 0.942

**Author**: 2026-04-06
**Current LB high**: **0.933** (v75 = A2 self-trained, A1-as-teacher; sig-mean fold reduction)
**Target LB**: **0.942** (+0.009)
**Status**: Aggressive — see §1 for honest feasibility framing.

> **History archive:** §1 through §28 of this plan (the full pre-A3 journal, ~14k lines covering the v16→v75 progression, all probes, kills, and decisions through 2026-05-16 morning) live in `docs/new_plan_history.md`. Grep that file for context on any pre-A3 experiment. This file keeps only §1 (load-bearing reality check), §29 (AST closeout + rank-mean finding), §30 (A3-recursive launch), §31 (A3 kill + cross-arch probe launch), §32 (cross-arch CLOSEOUT), §33 (Cerrado-geo XC pretrain committed), §34 (Chapter 10 ensembling audit + Tier 1 plan), and the current PICK UP HERE.

---

## ⛔ Killed directions — DO NOT propose without new evidence

Grep this table before recommending any direction. Each entry has a memory and/or plan-history pointer.

| Direction | Killed | Why | Memory / pointer |
|---|---|---|---|
| **Perch-as-teacher** for pseudo-labels | #34E (LB −0.008) + plan L26 constraint | Perch logits too noisy; local Perch embeddings ≠ Kaggle Perch embeddings | docs/new_plan_history.md L26-27, §28 |
| **BirdNET solo or filter** | §24 (LB 0.742 vs v73 0.929 = −0.187) | Aves-only taxonomy mismatch with BC2026's broader taxonomy | `feedback_encoder_taxonomy_must_match_lb` |
| **AST (Audio Spectrogram Transformer)** in production | §29.5 (CPU inference 44s/file = 6× over 90-min Kaggle cap) | 86M params @ seq=1024 is architecturally infeasible on CPU | `feedback_ast_cpu_infeasible` |
| **EffNetV2-S as A1/A2 backbone** | §14.19 (V2-S 4-fold v71 timeout 101min; A3-v1/v2 val 0.687-0.736 vs 0.77 gate) | Both timing AND quality fail; reverted to B0 | docs/new_plan_history.md §14.19 + L433-513 |
| **L1 noisy-student cross-arch** | §14.9 (LB 0.930 vs 0.931 baseline, val leakage) | Within ±0.005 noise; gate-fail | docs/new_plan_history.md §14.9 |
| **L3 Quantile-Mix α=0.5** | §14.8.L3-probe (LB −0.006) | Probability + rank blend regressed | `feedback_check_plan_before_recommending` |
| **Per-fold temperature scaling** | §25.13 (no broader-pool gain) | Calibration headroom exhausted | `feedback_min_oof_delta_to_burn_slot` |
| **Per-class isotonic on existing components** | §25.9 (max +0.01 broader-pool, below +0.05 gate) | Doesn't clear necessary-condition gate | `feedback_min_oof_delta_to_burn_slot` |
| **Weight re-optimization on existing stack** | §25.11 (max +0.0035 on linear surrogate) | Exhausted | `feedback_min_oof_delta_to_burn_slot` |
| **v50 + v73 dual ensemble** | §25.13 (best blend keeps w_v50=0) | v73 dominates | `feedback_min_oof_delta_to_burn_slot` |
| **Within-arch rank-mean fold reduction** | §29.4 / v76 (LB 0.932 vs v75 0.933 = −0.001, null) | A2 folds well-calibrated to each other; cross-arch is where rank-mean wins | `reference_cross_arch_rank_mean_fusion` |
| **Restructuring kernel data-shape to broader-pool OOF** | §29 audit | No sig-mean cross-arch stage exists in production to audit; `ENSEMBLE_WEIGHT_PROTO = 0` makes Cell 39 proto×mlp a no-op | This file §29 |
| **Locally-trained Perch-consuming students** | #32 (LB 0.922), #34A (LB 0.912) | Embedding mismatch | docs/new_plan_history.md L26 |
| **Recursive A3 self-training (A2-as-teacher)** | §31 ensemble OOF 0.7935 vs A2 0.8402 (Δ −0.0467) | A2's calibration drift (diffuse probs) compounds when A2 teaches a new student; top-K=5 rescue didn't recover. Don't try A4. | This file §31 |
| **Cross-arch fusion (CNN/transformer) on this recipe family** | §32 — 3 attempts, all gate-fail | AST × A2 = +0.0227 (best ever); ConvNeXt-Pico × A2 = +0.0061; MobileViT-S × A2 = +0.0077 (best of 3 recipes incl. transformer-recipe retry that landed at 0.7350, −0.063 below the lr=5e-4 retry). Structural ceiling — same data + same labels + mel front-end → mean_corr ≈ 0.35–0.40 floor, fusion gain ≤ +0.023 ceiling, can't clear +0.05 gate | This file §32 |
| **Plain L2-redux Aves-corpus-at-scale (XC v3 pretrain on B0)** | §18.7 (history) — smoke val_v2=0.7317 vs ImageNet 0.7414 = −0.010 below baseline | "More bird audio at scale" mechanism falsified. Don't re-run plain XC v3 pretrain. See §33 for the Cerrado-geo-filtered variant that's a *different* mechanism (acoustic-environment match) | docs/new_plan_history.md §18.7 |
| **Cerrado-geo XC pretrain on B0** | §33 (2026-05-18) — Phase D 5-fold standalone 0.7182; best A2-fusion 0.8410 rank-mean w=0.10; gap to 0.8902 gate = −0.0492 | Acoustic-environment-match mechanism falsified. Sixth convergent kill in the bird-audio-pretrain-transfers-to-BC2026 class (L2 / L2-redux / iNat / Perch / BirdNET / Cerrado) | `project_xc_pretrain_lever_exhausted` |
| **Operator-swap ensembling on existing OOFs (Tier 1)** | §34 (2026-05-18) | Best Tier 1 op was AST × A2 rank-mean (already known from §29.3; AST is unshippable). Shippable best = MobileViT × A2 rank-mean = 0.8480; all variants gate-fail. Mechanism: averaging-operator changes within same OOF pool can't cross the +0.05 gate | `feedback_min_oof_delta_to_burn_slot` |
| **Multi-seed A2 bagging (Tier 2)** | §34 (2026-05-20) — 3-seed × 5-fold = 15 ckpts; best 0.8538 (3-seed-sigmean rank-mean); gate-fail by −0.0364 | Ch 10 pp 388's lone untried DL-diversity lever. Adds genuine seed-to-seed variance but caps at +0.014 broader-pool — within-arch ceiling, same as cross-arch +0.023 ceiling from §32. Don't propose seed 45 (Ch 10 implies diminishing returns past 3) | `project_multiseed_bagging_exhausted` |
| **L3 multi-recipe SED bag (A2 ⊕ L3-prec)** | §35 (2026-05-21) — 2-recipe bag regresses vs stronger standalone under both operators (sig-mean −0.022, rank-mean −0.009). L3-prec alone 0.8700, bag-sig 0.8481, bag-rank 0.8608 | Recipe-diversity-within-same-backbone hypothesis falsified. The +0.030 broader-pool gain is recipe *upgrade*, not diversity. (c)(d)(e) won't reverse the sign | `project_l3_multirecipe_bag_killed` (TBD) + this file §35 |

---

## 1. Reality assessment — RECALIBRATED 2026-05-20 from actual BC2026 public LB

**Prior version of §1 cited stale prior-year LB numbers** (Yuriy 0.929 / yuanzhe zhou 0.9334 were from a previous competition or earlier snapshot). Actual current public leaderboard, fetched 2026-05-20 14:00 EDT:

| | LB |
|---|---|
| Our best | **0.933** (v75) |
| **#1 public** (Yannan Chen) | **0.962** |
| #2 public ("BirdCLEF+ 2026 Team") | 0.960 |
| #20 public | 0.954 |
| **Publicly-reproducible ceiling** (mtoshidesu reproduced "0.947 LB Public Pipeline") | **~0.947** |
| Imaad Mahmood "Perch v2 + ProtoSSM" (canonical public starter) | **0.925** |
| Needless090 "Iter-Pseudo Perch+SED" | **0.934** |
| **Our gap to top public** | **−0.029** |
| **Our gap to publicly-reachable ceiling** | **−0.014** |

**This is not "at the ceiling".** Our v75 = LB 0.933 trails the publicly-reproducible 0.947 pipeline by 0.014, and the actual public LB top by 0.029. The previous "recipe-family ceiling at LB 0.933" framing was an artifact of citing prior-year numbers — the real BC2026 public ceiling is materially higher.

The shape of the gap is informative. Our production kernel (`birdclef2026-protossm-postproc.ipynb`) is already a fork of the Perch+ProtoSSM+MLP+SED-rank-fuse public family. Our novel additions (A1→A2 self-training, broader-pool OOF gate) added ~+0.005-0.008 over the 0.925 public starter. The publicly-reproducible 0.947 pipeline (Imaad 0.946 + BirdNET 3rd branch) adds +0.021 *over the same starter* via techniques we haven't tested.

Hard constraints (still valid):
- Local Perch embeddings ≠ Kaggle Perch embeddings — Perch-consumers must train on Kaggle.
- LB SE noise floor: ±0.005. Single-lever OOF gain < +0.05 may not transfer above noise. **But this gate was calibrated assuming we were near the LB ceiling; with 0.014–0.029 of headroom, the right gate may be smaller, since the OOF→LB transfer ratio at the LB asymptote is not the same as the transfer ratio mid-pack.**

Realistic intermediate goals:
- 0.933 → 0.940 by adopting the public 0.946 pipeline's missing components (iterative pseudo with ensemble teacher, residual SSM second-pass, per-class OOF thresholds, BirdNET as low-weight 3rd branch)
- 0.940 → 0.95+ requires either (a) more independent model branches or (b) a structural insight not in public notebooks

**This plan no longer claims a ceiling at 0.933.** §35 closeout was based on the stale §1 numbers; see §36 for the reopen.

---

## §29 AST track closeout + rank-mean fusion finding + v76 push (2026-05-16 ~14:30 EDT)

### TL;DR

- AST fold-0 training completed at 0.7991 broader-pool — slightly below A1 (0.8094), in the "0.70-0.80 in the game but weaker" band.
- Cross-arch correlation probe: AST↔A2 mean Pearson = **0.3339** (diversity pass; well below 0.7 threshold).
- **Sig-mean fusion REGRESSES** at all AST weights (−0.008 to −0.024 vs A2 alone). **Rank-mean fusion GAINS** up to +0.0227 at w=0.40 AST. Architecture-calibration mismatch is real; rank-normalization absorbs it.
- Best fusion (rank-mean w=0.40): broader-pool 0.8630. Gate is 0.8902 (+0.05 over A2 anchor 0.8402). **Gap −0.0272** — gate-fail.
- **v76 pushed** (A2 within-fold rank-mean only; +0.0015 OOF gain). LB 0.932 (Δ −0.001 vs v75, within noise). Within-arch null confirmed.
- **v77 (AST fusion) KILLED** by CPU timing: AST 86M @ seq=1024 = **44s/file** on 4-thread CPU → 517 min extrapolated. Architecturally incompatible.

### §29.3 Fusion eval — sig-mean vs rank-mean

Computed locally on skynet with cached AST + A2 broader-pool probs:

| Fusion mode | Broader-pool AUC | Δ vs A2 ens (0.8402) |
|---|---:|---:|
| Sig-mean (AST + A2_ens) w=0.05 | 0.8322 | −0.0080 |
| Sig-mean w=0.50 | 0.8215 | −0.0188 |
| Sig-mean 6-way equal (1/6 each) | 0.8281 | −0.0122 |
| Rank-mean w=0.40 (best) | **0.8630** | **+0.0227** |
| Rank-mean w=0.50 | 0.8603 | +0.0200 |
| Rank-mean 6-way equal | 0.8568 | +0.0166 |
| **Gate** (anchor + 0.05) | 0.8902 | — |

**Key finding:** sigmoid-mean across architectures with different calibrations regresses; rank-mean absorbs the mismatch. Real and previously-untested fusion principle in this codebase.

Within-arch test (A2's own 5 folds, rank-mean vs sig-mean): only **+0.0015** broader-pool gain — confirms rank-mean win is specifically about *cross-architecture* calibration mismatch.

### §29 Files of note

| File | Purpose |
|---|---|
| `four_track/src/eval_a4_broader_oof.py` | AST single-fold broader-pool eval + correlation probe |
| `four_track/data/a4_ast_fold0_broader_oof.npz` | AST fold-0 saved probs (1478, 234) for future fusion analyses |
| `four_track/models/a4/a4_ast_fold0_seed42_asl.pt` | Fine-tuned AST ckpt (345 MB) — keep for research; no production use |
| `four_track/jupyter/.../birdclef2026-protossm-postproc.ipynb.bak_pre_v76_a2rankmean` | v75 baseline backup |

**Production baseline for next iterations:** v75 LB 0.933 (sig-mean). Future probes benchmark against 0.933, not v76's 0.932.

---

## §30 A3-recursive (A2-as-teacher self-training) launched (2026-05-16 ~16:45 EDT)

### Rationale

v75 LB 0.933 came from A2 (A1-as-teacher, broader-pool 0.8402). For a third round (A3), use A2's stronger ensemble (0.8402) as the teacher. Hypothesis: cleaner pseudo-labels → marginally stronger A3 student. Diminishing-returns risk known.

### Pipeline status (live state)

⚠️ **Three course corrections this session** (each yielded a memory):
1. Picked simpler `a2_pseudo_label_a1.py` (leaky for recursive) instead of canonical `a2_emit_oof_pseudo.py` — discarded 37-min DT run, wrote `a3_emit_oof_pseudo_a2.py`. Memory: `feedback_read_call_sites_not_docstrings`.
2. Calibration showed A2's probs are diffuse vs A1's (median 176/234 active classes vs 4 at standard threshold). Gate-failed P@0.5 (0.633 < 0.70). Rescued with top-K=5 filter (`a3_filter_pseudo_topk.py`) — non-canonical recipe deviation; preserves ranking signal, discards calibration drift.
3. Dispatched all 5 folds to DT instead of 4+1 DT+skynet split. Killed mid-fold-1, re-dispatched correctly. Memory: `feedback_dispatch_5fold_split_4_plus_1`.

### Live A3 training (4+1 split as of 20:43 EDT)

| Fold | Where | Status | val_roc_auc |
|---|---|---|---|
| 0 | DT (pre-kill) | ✓ done | **0.7791** (peak at epoch 8) |
| 1 | DT (pre-kill) | ✓ epoch-22 ckpt | **0.7848** (peak at epoch 21) |
| 2 | DT (PID 634513) | running | — |
| 3 | DT (PID 634513) | queued | — |
| 4 | skynet (PID 684741) | running | — |

ETA: DT finishes folds 2,3 ~23:15 EDT; skynet fold 4 ~00:30 EDT 2026-05-17.

**A2 comparison so far:** A3 fold 0 = 0.7791 vs A2 fold 0 = 0.8094 (−0.030); A3 fold 1 = 0.7848 vs A2 fold 1 = 0.8067 (−0.022). Per-fold mean projecting ~0.78 → ensemble ~0.80. A2 ensemble was 0.8402. **Projected A3 gap to A2 ensemble: ~−0.04**.

### Decision gate

| A3 ensemble broader-pool | Action |
|---|---|
| ≥ 0.8902 (anchor +0.05) | Push v77 to LB |
| (0.8402, 0.8902) | Information-value only; ask user before slot burn |
| ≤ 0.8402 | **Kill recursive path; document as exhausted; don't try A4/A5** |

Based on the trajectory, expected outcome is the third row.

### §30 Files of note

| File | Purpose |
|---|---|
| `src/a3_emit_oof_pseudo_a2.py` | OOF pseudo-label emission with A2 teacher |
| `src/a3_calibrate_pseudo.py` | Wrapper: a2_calibrate_pseudo with A3 paths |
| `src/a3_filter_pseudo.py` | Wrapper: a2_filter_pseudo @ KEEP_THRESH=0.6 |
| `src/a3_filter_pseudo_topk.py` | Non-canonical top-K=5 rescue filter |
| `src/a3_train.py` | Wrapper: a2_train with A3 paths (save dir models/a3/) |
| `data/processed/a3_train_ss_oof_probs.npz` | 108 MB OOF pseudo probs (12 windows/file × 10592 files) |
| `data/processed/a3_pseudo_soft_topk.npz` | 3.7 MB top-K-rescued soft labels (K=5) |
| `models/a3/` | A3 ckpts (filename keeps `a2_` prefix from a2_train.py) |

---

## §31 A3 KILLED + cross-arch probe (2026-05-17 ~00:20 EDT)

### A3 final result (kill confirmed)

| Component | A3 | A2 anchor |
|---|---:|---:|
| Per-fold mean broader-pool | 0.7745 | 0.8227 |
| **Ensemble (sig-mean) broader-pool** | **0.7935** | **0.8402** |
| Gap | **−0.0467** | — |

Per-fold: 0.7789 / 0.7847 / 0.7765 / 0.7450 / 0.7874 (folds 0-4). Eval script verdict: REGRESSION — pseudo-labels actively hurt. Recursive self-training exhausted at this teacher-strength tier. **Don't try A4** (A3-as-teacher would compound the damage).

### CPU smoke for cross-arch candidates (calibrated against B0)

| Backbone | Params | Smoke sec/file (skynet 4-thread, eager) | Implied Kaggle min (700 files, ÷2 for JIT+Xeon) |
|---|---:|---:|---:|
| B0 (production reference) | 4.3M | 4.55 | ~27 (verified vs production logs) |
| **ConvNeXt-Pico** | 8.7M | 3.81 (**0.84× B0**) | **~23** — fits |
| **MobileViT-S** | 5.1M | 8.88 (**1.95× B0**) | **~53** — tight but feasible if A2 folds reduced |

Skynet 4-thread eager-mode timing is ~2× Kaggle CPU + JIT. Relative ratios are LB-decision-relevant.

### Cross-arch probe launched (parallel)

Two-fold-0 probes running in parallel. Both share recipe: focal + A2 pseudo-labels (canonical `a2_pseudo_soft.npz`, A1 teacher), pseudo_ratio=0.4, ASL loss, 25 epochs.

| Run | Backbone | Where | Status | ETA |
|---|---|---|---|---|
| MobileViT-S fold 0 | mobilevit_s.cvnets_in1k | DT (PID 672653) | RUNNING | ~5h (fold 0 in flight at 00:14 EDT) |
| ConvNeXt-Pico fold 0 | convnext_pico.d1_in1k | skynet (PID 731021) | RUNNING | ~6h (skynet:DT 3× × 2× params) |

**Patch landed:** `four_track/src/model_a1.py` — `out_indices=(4,)` → `(-1,)` (and `(3,4)` → `(-2,-1)`). Backward-compatible for EffNet (5 stages: -1 == 4). Unlocks ConvNeXt (4 stages) and any non-EfficientNet backbone with a different stage count.

### Decision gate (per-fold, single fold)

| Single-fold broader-pool | Action |
|---|---|
| ≥ 0.80 | Strong cross-arch candidate; scale to 5-fold + rank-fuse with A2 in production |
| 0.75-0.80 | Comparable to A1 (0.7775 anchor); proceed to rank-mean(A2_ens, single_fold) LB probe |
| < 0.75 | Backbone doesn't transfer; kill that candidate |

If at least one candidate gates above 0.75, **the strongest signal for breaking +0.005 LB** is rank-mean fusion with A2 ensemble (per §29 finding: AST × A2 gave +0.023 broader-pool at rank-mean w=0.40, AST didn't ship due to CPU).

### §31 Files

| File | Purpose |
|---|---|
| `src/eval_a3_broader_oof.py` | Wrapper of eval_a2_broader_oof for A3 ckpts |
| `data/a3_5fold_broader_oof.npz` | A3 ensemble OOF (3.7 MB) — kept for fusion analyses, never shipping |
| `src/smoke_cpu_arch.py` | Reusable CPU latency smoke for new backbone candidates |
| `src/a5_train_xarch.py` | Wrapper: a2_train with models/a5_xarch/ save dir + A2 pseudos |
| `models/a5_xarch/` | Cross-arch ckpts (will be populated as runs complete) |

## §32 Cross-arch direction CLOSEOUT (2026-05-17 ~16:30 EDT)

### TL;DR

Three cross-arch single-fold candidates evaluated, all gate-failed at the +0.05 broader-pool rule. **Recipe-family ceiling confirmed at ~+0.023 fusion gain.** Direction added to Killed Directions table. Pivoting to §33 (Cerrado-geo-filtered XC pretrain on B0) per user choice; alternative was (ii) accept LB 0.933.

### §32.1 The three attempts

| Arch | Recipe | Standalone broader-pool | Mean corr vs A2 | Best rank-mean fusion w/ A2 | Δ vs A2 anchor (0.8402) |
|---|---|---:|---:|---:|---:|
| AST (§29) | Aves-pretrained, focal+pseudo | 0.7991 | 0.3339 | 0.8630 @ w=0.40 | +0.0227 |
| ConvNeXt-Pico (§31/§32) | ImageNet, ASL+pseudo 0.4, mixstyle=0.5 | 0.7829 | 0.3802 | 0.8463 @ w=0.25 | +0.0061 |
| MobileViT-S #1 (§31) | ImageNet, ASL+pseudo 0.4, mixstyle=0.5 | 0.7906 (overfit ep 2) | — | — | — |
| MobileViT-S #2 (§31) | mixstyle=0 retry | 0.7980 | 0.3762 | 0.8480 @ w=0.30 | +0.0077 |
| MobileViT-S #3 (§32) | **transformer recipe** (lr=1e-4, wd=5e-2, warmup=2, pseudo=0.2) | **0.7350** (ep 17, undertrained) | — (not fused) | — | (would be worse than #2) |

The MobileViT #3 transformer recipe HURT — landed 0.063 below #2's lr=5e-4 recipe at the same epoch budget. Even doubling epochs would project to ~0.78 (still below #2). The "transformer needs a transformer recipe" hypothesis is **falsified** for this dataset scale + pseudo-label setup.

### §32.2 Why the ceiling is structural

Fusion gain tracks (a) standalone strength and (b) inverse of cross-arch correlation. With same data, same labels, same mel front-end, same val pipeline:

- **Correlation floor ≈ 0.33** (AST's lowest, 86M-param transformer family) — same-data CNN/transformer variants can't get meaningfully below this.
- **Standalone ceiling ≈ 0.81** for any candidate that's CPU-feasible (per §29 AST CPU kill).
- Product of the two → fusion gain capped at ~+0.023. **Cannot clear +0.05 gate.**

To break the ceiling would require either (a) genuinely independent feature representation (different input modality), or (b) a structurally different training signal (different corpus/pretrain — §33's path).

### §32.3 Files of note

| File | Purpose |
|---|---|
| `src/eval_a5_broader_oof.py` | Single-ckpt broader-pool eval + per-class Pearson vs A2 fold-0 (mirror of `eval_a4_broader_oof.py` for `model_a1`-based ckpts) |
| `src/fuse_a5_rankmean.py` | Rank-mean weight sweep for cross-arch fold-0 + A2 ensemble |
| `data/a5_convnext_pico_fold0_broader_oof.npz` | ConvNeXt-Pico fold-0 saved probs (1.5 MB) for any future re-analysis |
| `data/a5_mobilevit_s_fold0_broader_oof.npz` | MobileViT-S #2 fold-0 saved probs |
| `models/a5_xarch/*.pt` | Preserved cross-arch ckpts (no production use) |
| `src/a2_train.py` | NEW: `--lr / --weight-decay / --warmup-epochs / --save-suffix` CLI args (backward-compatible) added §32 for the MobileViT-S transformer-recipe attempt; reusable for future recipe variants |

### §32.4 What §32 closes vs leaves open

- **Closes:** cross-arch fusion under "same data + same recipe family + mel front-end" is exhausted. Don't add CNN/transformer #4. Don't relax the +0.05 broader-pool gate.
- **Leaves open:** §33 (Cerrado-geo XC pretrain) is a different mechanism (acoustic-environment match, not architectural diversity). If that gate-fails too, the lever class is empirically exhausted and (ii) accept LB 0.933 is the only honest call.

---

## §33 Cerrado-geo-filtered XC pretrain on B0 — **KILLED** 2026-05-18 (Phase D gate-fail; sixth convergent kill in the bird-audio-pretrain class)

### §33.1 Context + commitment

After §32 closed cross-arch, the user committed to (iii-a) Cerrado-geo-filtered XC pretrain. Distinct from the killed (iii-b) "plain L2-redux Aves-corpus-at-scale" (see Killed Directions table; §18.7 −0.010 below baseline). Mechanism: **acoustic-environment match** (mic profiles, biome ambience, vocal density), not species-coverage match.

### §33.2 Pre-flight RESULT — corpus size gate passes for Cerrado, fails for strict Pantanal

XC v3 metadata scan (`data/external/xenocanto_bulk/`, 6,718 species dirs, 718,721 total recordings):

| Filter | Recordings | Species | vs §21.3 ≥10K abort gate |
|---|---:|---:|---|
| Strict Pantanal bbox (lat -22..-16, lon -58..-54) | **1,634** | 200 | FAIL (16% of gate) |
| **Cerrado biome bbox (lat -24..-2, lon -60..-41)** | **36,126** | **1,103** | **PASS (3.6× gate)** ← chosen |
| All Brazil (cnt=Brazil) | 63,566 | 1,372 | Pass 6.4× (rejected — drifts toward killed L2-redux mechanism) |

Cerrado is the middle ground that keeps the "acoustic-environment match" mechanism intact while having enough signal density to pretrain.

### §33.2a Phase A RESULT (2026-05-17 ~17:30 EDT) — manifest built, file-existence gate passed, fine-grained gate marginally missed (user overrode)

Manifest at `data/processed/xc_cerrado_pretrain_manifest.csv`:

```
total records seen :   718,721
geo bbox pass      :    36,126
length 5..120s pass:    32,844
file present on disk:   32,620   ← 224 missing audio files dropped
final manifest rows:    32,620   (1,097 species, 437 unique recordists)

Per-species coverage:
       1 rec :    62 species
    2-4 recs :   120 species
    5-9 recs :   135 species
   10-49 recs :   545 species  ← bulk of distribution
   50-199 recs:   235 species
    200+ recs:     0 species
```

The arbitrary 1000-species @ ≥5 recs fine-grained gate I added missed at 915 (8.5% short). User opted to proceed: the §21.3 plan gate (≥10K clips) is met 3.3×, the body of the distribution is healthy (780 species in 5+ recs bucket), and the 1000-species threshold was self-imposed. Honest §33.4 prior adjustment: gate-pass odds drop marginally (10–15% → 8–12% broader-pool gate-pass; LB +0.005 ~2–4%). Still a long shot, but the choice respects the actual plan gate, not the arbitrary tightening.

### §33.3 Recipe outline (mirror killed §18.7 recipe structure, swap corpus)

**Phase A — Build pretrain manifest** (~hours, CPU-only)
- Filter XC v3 _meta.json → CSV with `(file_path, species_code, gen, sp, lat, lon, length_sec, quality)` for the 36,126 Cerrado records
- Sanity: per-species count distribution, length distribution, quality distribution
- Deduplicate by XC ID, drop length <5s and length >120s
- Output: `data/processed/xc_cerrado_pretrain_manifest.csv`

**Phase B — Pretrain B0 on Cerrado corpus** (~6–10h DT, gated by available NVMe space)
- Fork `src/pretrain_l2_redux.py` (per `new_plan_history.md` §14.20+) → `src/pretrain_xc_cerrado.py`. Same recipe: focal-BCE γ=2, AdamW, multi-taxon head (1,103 species), 32 kHz mono, mel front-end matching production
- Save best ckpt to `models/xc_cerrado_pretrain/best.pt`
- Held-out 5% of pretrain corpus for early-stopping
- Per §18.7 lesson + §21.4: **smoke-then-scale** — no full pretrain at scale until smoke gate passes

**Phase C — Smoke finetune** (~1.5h DT, the §18.7-style gate)
- One fold (fold 0), `--init-from models/xc_cerrado_pretrain/best.pt`, `--ft-recipe production`, ASL loss, 25 ep, mixstyle=0.5
- **Gate (mandatory, no relaxation):** val_v2 fold-0 ≥ 0.7414 (ImageNet baseline anchor)
- **Gate-fail → kill §33, lock LB 0.933.** This is the same hard gate that §18.7's L2-redux failed by 0.010

**Phase D — If smoke passes, scale to 5-fold finetune** (~5.5h DT sequential, or 4+1 split per `feedback_dispatch_5fold_split_4_plus_1`)
- 5-fold finetune on BC2026 train_audio + A2 pseudos
- Broader-pool 1478-window OOF eval
- **Gate:** broader-pool ≥ 0.8902 (A2 anchor + 0.05) — the established `feedback_min_oof_delta_to_burn_slot` rule, NOT relaxed
- Gate-pass → push v77 = Cerrado-pretrained B0 + A2 fusion (rank-mean if cross-arch — but same B0 family means within-arch sig-mean is the default per §29)

### §33.4 Honest priors

- **Phase C smoke gate-pass prior: ~20–30%** — higher than §18.7's prior because the mechanism is different (acoustic-environment match vs species-coverage match), but plain XC v3 pretrain failed by 0.010 so we're not far above the floor.
- **Phase D broader-pool +0.05 gate-pass prior: ~10–15%** conditional on Phase C passing. The OOF→LB transfer at +0.05 magnitude has historically been ~0.08 (per `project_a2_first_gate_pass` 0.063 OOF → 0.005 LB) — clearing +0.05 broader-pool would project to +0.004 LB, *just at* noise floor.
- **End-to-end LB +0.005 prior: ~3–5%.** This is a real long shot. Pursuing it because (ii) accept-the-ceiling is irreversible and the user wants one more empirical shot before locking.

### §33.5 What would falsify the path

- Smoke fold-0 finetune val_v2 < 0.7414 → mechanism falsified, kill §33
- Smoke passes but broader-pool 5-fold ensemble < 0.85 → standalone too weak, won't beat anchor even at best fusion weight, kill §33
- Broader-pool ≥ 0.85 but < 0.8902 → information-value only, do NOT push LB slot per `feedback_min_oof_delta_to_burn_slot`

### §33.6 Time-box per §21.4

- **3 days hard wall** from Phase A start (Phase B is the bulk; Phase A is ~hours; Phase C+D fit in the remainder)
- If Phase A reveals corpus issues (e.g. lots of stub-length recordings, severe class imbalance) that need rework, deduct rework time from the 3-day box — don't extend

### §33.7 Files (to be created, marker for future sessions)

| File | Purpose |
|---|---|
| `src/build_xc_cerrado_manifest.py` | Phase A — filter XC v3 by Cerrado bbox + dedupe/filter |
| `data/processed/xc_cerrado_pretrain_manifest.csv` | Phase A output |
| `src/pretrain_xc_cerrado.py` | Phase B — fork of `pretrain_l2_redux.py` |
| `models/xc_cerrado_pretrain/best.pt` | Phase B output (Cerrado-pretrained B0 backbone) |
| `models/xc_cerrado_pretrain/config.json` | Phase B run config snapshot |

---

## §34 Chapter 10 ensembling audit — **Tier 1 + Tier 2 closed** (2026-05-17 → 2026-05-20)

**Tier 1 best (operator sweep + Caruana + LogReg + inv-corr):** 0.8630 broader-pool, rediscovered via AST × A2 rank-mean w=0.40 (the already-known §29.3 finding; AST is CPU-infeasible so unshippable). Shippable Tier 1 best (ex-AST): MobileViT × A2 rank-mean = 0.8480. Both gate-fail.

**Tier 2 (multi-seed A2 bagging, seeds 42/43/44 × 5 folds = 15 ckpts):** 15-ckpt sig-mean = 0.8499; 15-ckpt rank-mean = 0.8515; best post-hoc composition = 3-seed-sigmean rank-mean = **0.8538** (Δ +0.0136 vs A2 anchor 0.8402). All hit the 0.85–0.86 band; **none clears the 0.8902 gate**. Per `feedback_min_oof_delta_to_burn_slot`, +0.0136 OOF → projected ~+0.0011 LB at the historical 0.08 transfer ratio (well below ±0.005 LB SE). No v77 push.

Side finding (Tier 2): seeds 43 (0.8501) and 44 (0.8531) both outperform production seed 42 (0.8402) as 5-fold standalone bags. The original A2 happened to land in a weaker basin. Selecting "best seed" on broader-pool would be test-set cherry-picking; not shippable.

**Lever-class verdict (six-attempt convergent kill on within-arch DL ensembling diversity):** Operator-swap ensembling and multi-seed bagging are the two untried levers Ch 10 pp 388 highlighted; both hit the same +0.023 fusion-gain ceiling that §32 established for cross-arch. Structural ceiling at LB 0.933 confirmed.

### §34.1 Context

While Phase D was running, user asked whether we're doing enough ensembling. Read The Kaggle Book Ch 10 (Ensembling with Blending and Stacking Solutions, pp 361–394) end-to-end. Audit produced concrete gaps.

### §34.2 What we're already doing vs what Ch 10 lists

| Technique | Ch 10 recommendation | BC2026 stack |
|---|---|---|
| Arithmetic mean (sig-mean) | "Basically a no-brainer" | YES — v75 production |
| Rank averaging for AUC | "If task is ROC-AUC, simply averaging may not suffice... convert to ranks" | YES — v76 within-arch, §29 cross-arch |
| Inverse-correlation weighting | "Slight improvements suffice" | **NO** — never tried |
| Geometric / harmonic / mean-of-powers / logarithmic | "Variants may work better than arithmetic" | **NO** — never tried |
| Linear blending (LogReg + L1/L2 + positive-only) | "Linear preferred; constraints prevent overfitting" | **NO** — we tried LightGBM stacker (D2-α/β, killed LB 0.925) but never LogReg |
| Ensemble selection (Caruana hill-climbing) | "Recommended where overfit risk is high" | **NO** — never implemented |
| Stacking k-fold OOF + meta-learner | "Doesn't need comparable predictive power" | YES (D2-α/β) — killed LB 0.925 |
| **Multi-seed bagging** | "For neural nets, varying init seed alone creates diverse bag" | **NO** — 1 seed per fold |
| 10–20 fold OOFs (vs our 5) | "Ideally between 10 and 20" | NO |
| Bagging within each fold | "Helps avoid overfitting" | NO |

### §34.3 Ch 10 validates the §32 cross-arch ceiling

Page 390 verbatim: *"Even if you can manage to stack models in a deep learning competition, you have a limited choice for stacking different models. Since you are restricted to deep learning solutions, you can only vary small design aspects of the networks and some hyperparameters (or sometimes just the initialization seed) without degrading the performance. In the end, given the same type of models and more similarities than differences in the architectures, the predictions will tend to be too similar and more correlated than they should be, limiting the effectiveness of ensembling."*

This is exactly the §32 +0.023 ceiling. Not a failure of execution; a known structural limit. Document this for future sessions that might want to "try one more cross-arch."

### §34.4 Three-tier plan + recommendation

**Tier 1 (cheap, ~2h scripting + 5 min runtime, no GPU). RECOMMENDED.**
Single script `src/ensemble_tier1_sweep.py` operates on existing OOF npz files (A2 5-fold, AST f0, ConvNeXt f0, MobileViT f0, Cerrado 5-fold when Phase D + eval done). Runs:
- Alternative averaging operators (geometric, harmonic, mean-of-powers, logarithmic) on within-arch and cross-arch pairs
- Inverse-correlation weighted averaging on the full pool
- Caruana ensemble selection with file-based holdout (split 66 soundscape files for honest test AUC)
- Logistic regression blender (L1 + positive-only weights) with file-based holdout

**Honest projection:** best of Tier 1 lands ~0.85–0.87 broader-pool. Gate-fail vs 0.8902. Closes the "we never tried" gap.

**Tier 2 (1 GPU-day, gated on Tier 1 result).** Multi-seed A2 bagging — train A2 fold 0–4 with seeds 43 and 44 (production seed is 42), so we get 3 seeds × 5 folds = 15 models. Sig-mean ensemble. Per Ch 10 pp 388, this is the lone untried mechanism with structural >+0.05 potential (different init → different local minima → genuine error decorrelation without cross-arch correlation ceiling). Cost: ~15h DT (or 12h DT + 3h skynet via 4+1 split).

Trigger: only if Tier 1 lands ≥ 0.85 broader-pool. If Tier 1 maxes at the existing A2 anchor (0.84), Tier 2 priors degrade significantly.

**Tier 3 (skip).** Multi-layer stacking, bagging-within-fold, 10–20 fold OOFs. Each requires retraining everything; cost-benefit poor vs Tier 2.

### §34.5 Honest priors

- Tier 1 best gate-pass: ~5%. Operator changes within the same averaging space probably add <+0.01 over current rank-mean. Mostly empirical closure.
- Tier 2 gate-pass conditional on Tier 1 ≥ 0.85: ~15–20%. Multi-seed bagging is the chapter's lone "this could work" untried lever for DL-only competitions.
- End-to-end LB +0.005 across both tiers: ~3–5%. In the same ballpark as §33 went in.

### §34.6 Files (to be created)

| File | Purpose |
|---|---|
| `src/ensemble_tier1_sweep.py` | Tier 1 sweep — alt operators + correlation weighting + Caruana + LogReg blender, file-based holdout |
| `data/ensemble_tier1_results.json` | Recorded results for §34 closeout / future reference |

---


## §35 Competition closeout — **RETRACTED 2026-05-20 ~14:30 EDT**; based on stale §1 LB numbers. See §36 for the reopen.

> The closeout below was written under the assumption that LB 0.933 was near the competition ceiling. That premise was wrong — actual BC2026 public LB top is 0.962, publicly-reachable code reproduces ~0.947. The Tier 2 multi-seed result (best 0.8538 broader-pool) still gate-fails at the +0.05 rule, so the *Tier 2 outcome itself* stands. But "competition closed" does not. See §36 for the techniques top public notebooks use that we have not tested.

## §35 Competition closeout — LB 0.933 locked (2026-05-20 ~14:00 EDT) — [RETRACTED]

### §35.1 Tier 2 final result

15-ckpt multi-seed A2 bagging (3 seeds × 5 folds), broader-pool 1478-window OOF:

| Composition | broader-pool | Δ vs A2 anchor 0.8402 |
|---|---:|---:|
| Seed 42 alone (= A2 anchor) | 0.8402 | — |
| Seed 43 alone (5-fold) | 0.8501 | +0.0099 |
| Seed 44 alone (5-fold) | 0.8531 | +0.0129 |
| Seeds 42+43 (10-ckpt) | 0.8438 | +0.0036 |
| Seeds 42+44 (10-ckpt) | 0.8505 | +0.0103 |
| Seeds 43+44 (10-ckpt) | 0.8522 | +0.0120 |
| 15-ckpt sig-mean | 0.8499 | +0.0097 |
| 15-ckpt rank-mean | 0.8515 | +0.0113 |
| **3-seed-sigmean rank-mean (best post-hoc)** | **0.8538** | **+0.0136** |
| **Gate (anchor + 0.05)** | 0.8902 | — |

Best post-hoc composition is **−0.0364 below gate**. At the historical OOF→LB transfer ratio of 0.08 (v75: +0.063 OOF → +0.005 LB), +0.0136 OOF projects to ~+0.0011 LB — well below the ±0.005 LB SE noise floor.

**Verdict: GATE FAIL → closeout. No v77 push.**

Side finding: seed 42 (the production seed) was the weakest of the three. Seeds 43 / 44 each beat it standalone. This isn't shippable (selecting seed on broader-pool = test-set leakage) but it's worth recording: future training runs in this family should sample multiple seeds and report variance rather than treating any single seed as canonical.

### §35.2 Final submission

**v75 = A2 self-trained (A1-as-teacher) 5-fold sig-mean — LB 0.933.** Locked as final.

### §35.3 Lever search summary (§2 → §34)

The path from LB 0.916 (v16 baseline) to LB 0.933 (v75) used four levers that *worked*:

1. **A1 teacher-pseudo + B0 student** — §15-19 → v50 0.929 anchor
2. **A2 self-training (A1-as-teacher) with focal + pseudo_ratio=0.4 + mixstyle=0.5** — §22-24 → v75 0.933
3. **Broader-pool 1478-window OOF as the LB-correlated gate** — §22.4 made decision-making honest
4. **+0.05 broader-pool gate hard rule** — `feedback_min_oof_delta_to_burn_slot` prevented dozens of slot-burns

Twelve levers in three classes hit a wall:

**A. Bird-audio pretrain transfer** (mechanism: more bird-domain pretraining transfers to BC2026). Six convergent kills:
- L2 (original Aves pretrain), L2-redux (XC v3 scaled), iNat 2024 Sounds (cross-corpus), Perch (Google embedding), BirdNET (Aves-only encoder), Cerrado-geo XC (biome-filtered). All hit the same gate-fail band (val_v2 0.73–0.74 vs ImageNet 0.74 baseline). The structural cap is that BC2026's broader taxonomy + acoustic environment + window pooling discipline already extracts what these corpora offer.

**B. Cross-architecture diversity in the same recipe family** (mechanism: heterogeneous arch + mel front-end → uncorrelated errors). Three kills (§29, §32):
- AST: standalone 0.7991, cross-arch fusion gain +0.023 (best ever), but CPU-infeasible (44s/file vs 90-min Kaggle cap).
- ConvNeXt-Pico: standalone 0.7829, fusion gain +0.006.
- MobileViT-S: standalone 0.7980 (CNN recipe) / 0.7350 (transformer recipe), fusion gain +0.008.
- Structural ceiling: fusion gain ≤ +0.023 for any CPU-feasible CNN/transformer × A2 in this recipe family. Cannot clear the +0.05 gate.

**C. Within-architecture ensembling exotica** (Ch 10 unfilled gaps). Three kills (§34):
- Tier 1 operator-swap (geom/harm/mean-of-powers/inv-corr/Caruana/LogReg) — best was AST × A2 rank-mean rediscovery; shippable best gate-failed by 0.042.
- Within-arch rank-mean (v76) — +0.001 broader-pool, LB −0.001 (within noise).
- Multi-seed A2 bagging (this section) — +0.014 broader-pool best, gate-fail by 0.036.

### §35.4 What the ceiling is, mechanistically

LB 0.933 is the asymptote for this competition's structure with a *single-model recipe family*. Ch 10 pp 390 states this directly: in DL-only competitions, varying init seed / arch / small hyperparams produces models too similar to break the fusion-gain ceiling. The published private-LB #1 is 0.9334; our 0.933 sits inside one LB SE of it.

Breaking the ceiling would require:
- A genuinely independent feature representation (different input modality — e.g. spectrogram + raw waveform + Perch embeddings consumed *Kaggle-side*, since local Perch ≠ Kaggle Perch).
- Or, an external corpus that actually transfers (six tried, none did).
- Or, a stacking layer with real domain signal (LightGBM stacker tried in D2-α/β and killed at LB 0.925 — D2-β killed because the meta-features carried no extra signal over the base learners).

None of these is reachable in the remaining time / compute budget without a major recipe change. Closing the book.

### §35.5 Artifacts preserved

| File | What |
|---|---|
| `data/a2_a1_5fold_broader_oof.npz` | A2 production OOF (anchor 0.8402) |
| `data/a2_multiseed_broader_oof.npz` | 15-ckpt multi-seed OOF (§35.1 table source) |
| `data/a2_multiseed_results.json` | §35.1 JSON dump |
| `data/a4_ast_fold0_broader_oof.npz` | AST fold-0 (best-ever fusion partner, unshippable) |
| `data/a5_convnext_pico_fold0_broader_oof.npz` | ConvNeXt-Pico fold-0 |
| `data/a5_mobilevit_s_fold0_broader_oof.npz` | MobileViT-S fold-0 |
| `data/xc_cerrado_5fold_broader_oof.npz` | Cerrado-pretrained 5-fold OOF |
| `data/ensemble_tier1_results.json` | Tier 1 sweep results dump |
| `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold{0..4}_seed{42,43,44}_asl.pt` | 15 ckpts |
| `jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb` | v75 production kernel |

---

## §36 Reopen — what top public notebooks use that we have NOT tested (2026-05-20 ~14:30 EDT)

### §36.1 How we got here

After the §35 "closeout", a competition-LB sanity check (`kaggle competitions leaderboard birdclef-2026 -s`) showed the actual public top at LB 0.962 (Yannan Chen) and a publicly-reproducible pipeline ("BirdCLEF 2026 0.947 LB: Public Pipeline Reproduced" by mtoshidesu) at LB 0.947. The §1 reality assessment had been citing prior-year LB numbers (Yuriy 0.929 / yuanzhe zhou 0.9334), which created the false impression that LB 0.933 was near the ceiling. **It is not.**

§1 has been rewritten. §35 is retracted (the Tier 2 result itself stands; the "competition closed" conclusion does not). This §36 inventories what top public notebooks demonstrably use that we have not tested, so the next push has a concrete menu.

### §36.2 Inventory of public-notebook techniques

Pulled via `kaggle kernels pull` to `/tmp/topnb/` for direct read; LB scores are the claims in the notebook titles/descriptions.

| Notebook | Claimed LB | Technique gap from our v75 |
|---|---:|---|
| Imaad Mahmood — Perch v2 + ProtoSSM | 0.925 | Same family as our kernel; this is the public starter |
| **Needless090 — Iter-Pseudo Perch+SED v17** | **0.934** | (a) `v5_pseudo + v5_pseudo2`: *two rounds* of pseudo with ensemble-as-teacher (vs our 1 round A1→A2); (b) per-class OOF-optimized decision thresholds; (c) Residual SSM second-pass; (d) sonotype mirroring (label-space expansion) |
| Maryna Borovska — Two-Pass SSM | — | Pass-1 ensemble + lightweight Pass-2 model trained to predict `Y − sigmoid(first_pass)` residuals |
| **Mtoshidesu — 0.947 Public Pipeline Reproduced** | **0.947** | "Exact copy of Imaad Mahmood's 0.946 pipeline" + **BirdNET as low-weight third branch** (we killed BirdNET *solo* — never tested as a low-weight 3rd branch) |
| F.A.Nina EoS.4 | 0.948 (claim) | Meta-blend of multiple public notebooks (0.001 × YoussefLB948 + 0.999 × Karnakbayev Power Optimization) |
| Itshyao — S124 S114 G124 F1 RankBlend | — | Multi-model rank blend, weights ≈ {0.03, 0.97} across two solution buckets; each bucket has its own SED-fold composition |
| Imaad Mahmood "0.946 pipeline" (referenced by mtoshidesu) | 0.946 | Cells: Perch backbone → MLP probes → SSM → distilled SED branch → optional BirdNET → final rank blend + PP gates. Cell 08 = "UPGRADED prior tables — joint site-hour bucket" |

### §36.3 Levers we have NOT tested (in priority order by cost × prior gain)

These are *distinct* from anything in the current Killed Directions table — each is either a different framing or a sub-lever the kills do not cover.

**T1 — DROPPED 2026-05-20.** Read the actual `apply_per_class_thresholds` in iter934 — it's a piecewise-linear *monotone-per-column* transform that maps `[t, 1] → [0.5, 1]` and `[0, t] → [0, 0.5]`. Within-column ranking is invariant, so macro ROC-AUC is unchanged. The V17 author may have included this for legacy reasons or confused it with another lever. **It cannot move the metric we care about.**

**T1' — PROBE RAN 2026-05-20, RESULT INTERESTING BUT NOT A SHIPPABLE LEVER.** Pure-numpy probe on cached A2 OOF found +0.0197 broader-pool from `file_level_confidence_scale(top_k=1)` and similar magnitudes for `rank_aware_scaling(power=0.8)` and the combined trio. **But the production kernel ALREADY HAS all three PP techniques active** (CFG line 242-250 + line 361-362 overrides: `file_level_top_k=2, rank_aware_scale=True, rank_aware_power=0.4, delta_shift_alpha=0.20`). The probe was measuring what PP adds to the raw A2 SED branch; the kernel applies the same PP downstream of all-branch fusion, so that +0.019 is already baked into v75's LB 0.933. Filing this as a "verify against production code before claiming a missing lever" lesson. Memory: `feedback_read_call_sites_not_docstrings` applied to my own probes.

### §36.3b CORRECTED untested-lever menu (after kernel inspection)

After confirming what is actually in `birdclef2026-protossm-postproc.ipynb`, the real untested levers are:

**L1 — Bump ResidualSSM correction_weight 0.10 → 0.35** (cheapest, ~1 LB slot, ~75 min Kaggle run). Single-config change. iter934 V17 uses 0.35; ours uses 0.10. The ResidualSSM exists in our kernel but is downweighted 3.5×. Could go either way — they may have tuned for their specific pipeline. Test ranking: at 0.35 the residual second-pass dominates the first-pass correction; ours mostly leaves the first pass alone.

**L2 — KILLED 2026-05-20 by offline rank-blend probe.** Tested BirdNET-MLP rank-blend against three bases (A2 anchor 0.8402, 15-ckpt sigmean 0.8499, 3-seed rankmean 0.8538) at weights ∈ {0.02, 0.05, 0.10, 0.15, 0.20}. **All combinations regressed** — BirdNET's 0.5644 broader-pool is too weak even at low weight, and cross-correlation (0.19-0.24) is good but not enough to compensate. The public 0.947 = 0.946 + BirdNET attribution (+0.001 LB) is at LB SE noise floor and may not be a real lever. We don't have BirdNET-direct-logits-via-scientific-name OOF cached, but priors are low given the same TFLite model + same Aves-only taxonomy gap that killed BirdNET solo in §24. Also: §14.14.15 B1 weight sweep already showed w=0.20 regresses to LB 0.922 (the natural analog to mtoshidesu's SED=0.18). The B1 weight is already locally optimized at w=0.10.

**L3 — Multi-recipe SED bag** (~5 days DT, biggest expected gain). iter934's 5 ckpts are diverse in *training recipe*: `v5_focal` (focal loss), `ce_s123` (CE with seed 123), `ce_s456` (CE with seed 456), `v5_pseudo` (focal + pseudo round-1), `v5_pseudo2` (focal + pseudo round-2). Our 5 ckpts are the same recipe across 5 folds. Recipe diversity should genuinely de-correlate errors in a way fold-split diversity cannot (per §34's seed-bagging finding: same-recipe seeds gain +0.014; cross-recipe likely gains more).

**L4 — Iter-pseudo round 2 (ensemble teacher)** (~3 days DT compute + 1 LB slot). The public 0.934 notebook's `v5_pseudo2` is round-2 pseudo from the full SED+ProtoSSM ensemble (NOT from a single SED). Our killed A3 was round-2 from A2-alone. Different mechanism. This is the strictly-different-from-killed-A3 reformulation of iter-pseudo.

**L5 — TTA full 5 shifts at submit (not capped 3)** (config change, but blocked by 90-min budget). Our kernel caps TTA shifts to 3 in submit mode for runtime. If we could free 15-20 min elsewhere (skip OOF, ship pre-trained ProtoSSM ckpt), we could enable 5 shifts. Modest expected gain (+0.001-0.003).

### §36.3c LEVER PRIORITIES (effective 2026-05-20)

Cheapest-first to highest-cost:

| Order | Lever | Cost | Expected gain | Risk |
|---|---|---|---|---|
| 1 | L1: ResidualSSM correction_weight bump | 1 LB slot | unclear | Could regress |
| 2 | L2: BirdNET 3rd branch | 1 day + 1 LB slot | +0.001-0.005 | Could regress (over-blending) |
| 3 | L5: TTA full 5 shifts (if budget allows) | config | +0.001-0.003 | Time budget |
| 4 | L4: Iter-pseudo round 2 (ensemble teacher) | 3 days DT | +0.003-0.008 | High |
| 5 | L3: Multi-recipe SED bag | 5 days DT | +0.005-0.012 | High |

**T2 — BirdNET as low-weight third branch (cheap, ~1 day).** Public 0.947 = Imaad 0.946 + BirdNET at small weight. Our §24 killed BirdNET *solo* at LB 0.742; we have never tested it as a low-weight blend component. Mechanism: low-weight branches contribute mostly via uncorrelated errors, not absolute strength — a strict subset of the BirdNET kill. Prior gain: +0.001-0.005 LB. Cost: BirdNET inference cell in kernel + weight sweep + one LB probe.

**T3 — Iterative pseudo with ENSEMBLE teacher (medium, ~3 days).** Our killed A3 used A2 *alone* as teacher; the public 0.934 notebook uses the SED+ProtoSSM ensemble as the pseudo teacher (`v5_pseudo + v5_pseudo2`). The kill in `project_recursive_pseudo` was specifically "A2 → A3 with A2-alone teacher". Ensemble-as-teacher is structurally different (richer probability surface, more error-correlation diversity in the labels). Prior gain: +0.003-0.008 LB. Cost: emit ensemble pseudo OOF + retrain SED student (~5 days DT for 5 folds).

**T4 — Residual SSM / second-pass boosting (medium, ~3-5 days).** Train a small model on `Y − sigmoid(first_pass_ensemble)` residuals. Mathematically gradient-boosting-style. We have never tried any boosting head. Prior gain: +0.002-0.005 LB. Cost: model design + train + integrate into kernel.

**T5 — Joint site-hour bucket priors (cheap, ~1 day).** Our kernel uses site / hour priors separately. Public 0.947 uses "UPGRADED" joint bucket. May or may not help — quick to test. Prior gain: +0.001-0.003.

**T6 — Sonotype mirroring (medium, ~3 days).** Public 0.934 mentions this in Cell 51. Need to read source to understand the mechanism (probably: map species labels to higher sonotype groupings, train auxiliary head, use it as a regularizer). Unknown prior.

**T7 — Additional SED diversity at low weight.** Even if our cross-arch §32 closed the "fusion gain > +0.05" question, the public top likely runs 3-5 SED variants in the bag, each at low weight, where the *aggregate* effect beats single-arch. Worth re-testing the AST × A2 (LB 0.025 LB-projected from broader-pool +0.023) given our LB headroom is now ~0.029, not ~0.005. **However:** AST is CPU-infeasible per `feedback_ast_cpu_infeasible` — this lever needs a CPU-feasible substitute.

### §36.4 Gate calibration under the new picture

The +0.05 broader-pool gate was calibrated from four LB data points spanning the regime where we were near our v75 ceiling. With 0.029 of LB headroom to top public, the OOF→LB transfer ratio could be very different — historically transfer ratios are higher when you're mid-pack and lower as you approach the asymptote. The four data points behind `feedback_min_oof_delta_to_burn_slot` were:
- v4 5-fold +0.027 OOF → −0.001 LB (we were at 0.930)
- sig_mean +0.010 OOF → −0.001 LB (we were at 0.929)
- BirdNET +0.077 OOF → −0.187 LB (taxonomy mismatch, structural)
- v75 +0.063 OOF → +0.005 LB (we were at 0.928, broke to 0.933)

None of these data points were collected at LB <0.92 or LB >0.94. Mid-pack at LB 0.94 may have a 3-5× higher transfer ratio than at LB 0.93. **Don't lower the gate by guess; collect new transfer data.** First LB probe should be the lowest-cost lever (T1: per-class thresholds) — if it moves LB by ≥+0.002, the +0.05 gate is overstated and should be re-calibrated.

### §36.5 What §35's Tier 2 / Cerrado / Tier 1 closeouts STILL imply

- The seed-bagging / cross-arch / operator-swap / XC-pretrain attempts that all gate-failed individually **are still genuine kills under their existing framing**. Don't reopen them on the basis of §36 alone.
- §32's "structural ceiling at +0.023 cross-arch fusion gain" stays true *for that recipe family*. The reopen is about **levers we never tested**, not about reviving levers that did fail.
- The "Multi-seed A2 bagging" Killed Directions row (added 2026-05-20) is correct — multi-seed within A2 didn't clear gate. But this doesn't preclude multi-model (multi-arch) ensembles, which §32 closed for B0-family only.

### §36.6 Files (notebooks read from /tmp/topnb/ on 2026-05-20)

| Path | What |
|---|---|
| `/tmp/topnb/947/` | mtoshidesu 0.947 public pipeline (largest reference) |
| `/tmp/topnb/iter934/` | needless090 iter-pseudo Perch+SED v17 |
| `/tmp/topnb/twopass/` | Maryna Borovska two-pass SSM |
| `/tmp/topnb/eos4/` | F.A.Nina EoS.4 meta-blend |
| `/tmp/topnb/rankblend/` | itshyao S124-S114-G124-F1 rank blend |

These are tmp; pull fresh copies if a future session wants to inspect them. Source command:
```bash
kaggle kernels pull mtoshidesu/birdclef-2026-0-947-lb-public-pipeline-reproduced -p ./947
kaggle kernels pull needless090/birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s -p ./iter934
kaggle kernels pull marynaborovska/birdclef-26-two-pass-ssm-advanced-pp -p ./twopass
kaggle kernels pull nina2025/birdclef-2026-eos-4 -p ./eos4
kaggle kernels pull itshyao/birdclef-2026-s124-s114-g124-f1-rankblend -p ./rankblend
```

---

## 📌 PICK UP HERE — INVALIDATED 2026-05-23 ~23:45 EDT, see §43 (end of file) for current state

## §36-era PICK UP HERE (2026-05-21 ~23:15 EDT — L4-v1 nearly done on DT, L4-v2 staged for auto-dispatch)

> **All prior PICK UP HERE sections are invalidated** (including the earlier 18:12 EDT one).
> This is the single current handoff.

### TL;DR — one paragraph

Kaggle Perch p0/p1 extraction finished early (~22:00 EDT, not 23:00). Merged 10658-file × 12-window cache built; verified all 66 broader-pool val files are included (no separate val-extraction kernel needed). Rebuilt local Perch cache (59 full-labeled files), trained ProtoSSM 3-seed × 5-fold file-CV (OOF 0.8301), ran ensemble inference on full 10658-file pool, built L4-v2 teacher smoke gate eval. **Gate PASSES**: rank-mean fusion w_L3=0.60 = 0.8832 vs L3-prec subset 0.8709 (Δ +0.0122, > +0.010 dispatch threshold). Built L4-v2 pseudo manifest (rank-mean @τ=0.9667 → 2.44 positives/window matching L3-prec @0.7 baseline). **Caveat:** 60.8% of L4-v2 pseudo positives come from 159 classes ProtoSSM never saw during training — the +0.0122 OOF gain was measured on 71 in-val classes only. Realized student gain may be smaller. L4-v1 fold-0 is at epoch 14/25 on DT, val_v2 plateaued at 0.8035 since epoch 9; ETA ~02:30 EDT 2026-05-22. **Autopilot script in place** to wait for L4-v1 completion, syncback + eval L4-v1 broader-pool, then dispatch L4-v2 fold-0 on DT (~9h, ETA ~12:00 EDT 2026-05-22). Production LB stays v75 0.933.

### §36.A — Autopilot (running in background as of 2026-05-21 23:15 EDT)

Script: `four_track/scripts/l4_autopilot_dt.sh`. Launched via `nohup ... > log/l4_autopilot_YYYYMMDD_HHMMSS.log &`. Steps:

1. Poll DT for L4-v1 completion marker `L4 fold-0 smoke DT dispatch complete` in the runon log (sleeps 5 min between polls).
2. On completion (or `Traceback` → abort): syncback DT ckpts to skynet.
3. Run `python -u src/eval_l4_fold0_broader_oof.py --ckpt-suffix _l4` → produces `data/l4_fold0_broader_oof.npz` + console verdict against gate 0.8696.
4. Dispatch L4-v2 fold-0 on DT: `runon deepthought bash scripts/l4v2_fold0_smoke_dt.sh` (blocks remotely).
5. After L4-v2 completes (~9h later): syncback + run `eval_l4_fold0_broader_oof.py --ckpt-suffix _l4v2` → produces `data/l4v2_fold0_broader_oof.npz` + verdict.
6. Exit.

Autopilot abort conditions: any Traceback in DT log between steps. Recoverable manually.

### §36.B — Current artifacts on disk (skynet)

| Artifact | Path | Notes |
|---|---|---|
| Merged Kaggle Perch features (full pool) | `four_track/kaggle_datasets/train-soundscapes-perch/full_train_soundscapes_perch.npz` | 826 MB; 127896 windows × 10658 files |
| Rebuilt Perch cache (59-labeled subset) | `four_track/data/kaggle_perch_cache/full_perch_arrays.npz` | 4.6 MB; train-time cache for ProtoSSM |
| ProtoSSM 3-seed ckpt (avg) | `four_track/models/protossm_pretrained_v2/protossm_pretrained.pt` | 23 MB |
| ProtoSSM OOF (file-CV) | `four_track/data/protossm_oof_fileCV.npz` | OOF mean 0.8301 |
| ProtoSSM probs on full pool | `four_track/data/processed/protossm_pseudo_soundscape.npz` | 106 MB; (127896, 234) sigmoid probs |
| L4-v2 fused pseudo NPZ | `four_track/data/processed/l4v2_pseudo_soundscape.npz` | 105 MB; rank-mean fused |
| L4-v2 pseudo manifest CSV | `four_track/data/processed/l4v2_pseudo_manifest.csv` | 10.1 MB; 101672 rows, 2.36 positives/window |
| Teacher gate report | `four_track/log/eval_l3prec_x_protossm_fileCV_*.log` | full sweep + verdict |
| Calibration JSON | `four_track/data/processed/l4v2_pseudo_calibration.json` | τ=0.9667, trained/untrained split |

L4-v2 pseudo manifest also staged on DT at `/home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/data/processed/l4v2_pseudo_manifest.csv` and dispatch script at `scripts/l4v2_fold0_smoke_dt.sh`.

### §36.C — New code added this session (uncommitted)

| File | Purpose |
|---|---|
| `src/rebuild_perch_cache.py` | Convert merged extraction to ProtoSSM's expected schema (59-file labeled subset) |
| `src/train_protossm_local.py` (modified) | Added `--save-oof-path` and `--cv-mode {site,file}` flags |
| `src/run_protossm_full_pool.py` | Run 3-seed ProtoSSM ensemble on full 10658-file pool, emit (n_windows, 234) probs |
| `src/eval_l3prec_x_protossm_broader_oof.py` | Teacher smoke gate eval (sig-mean + rank-mean weight sweeps + verdict) |
| `src/build_l4v2_pseudo_npz.py` | Per-class rank-normalize L3-prec & ProtoSSM probs, fuse rank-mean w_L3=0.60, calibrate τ for 2.44 pos/window |
| `src/eval_l4_fold0_broader_oof.py` | Single-fold broader-pool eval; works for both `_l4` (L4-v1) and `_l4v2` ckpts |
| `scripts/l4v2_fold0_smoke_dt.sh` | L4-v2 fold-0 dispatch script (DT) — uses l4v2_pseudo_manifest.csv |
| `scripts/l4v2_fold0_smoke_skynet.sh` | Same but for skynet (NOT USED — aborted due to 28h ETA) |
| `scripts/l4_autopilot_dt.sh` | Wait+eval+dispatch autopilot |
| `jupyter/perch-train-soundscapes-extract/merge_partitions.py` (modified) | PARTITION_COUNT 4→2, dir paths corrected |

### §36.D — Decision tree for next session (user picks this up)

```
WHEN AUTOPILOT COMPLETES (~12:00 EDT 2026-05-22):
  Inspect log/l4_autopilot_*.log for both L4-v1 and L4-v2 fold-0 broader-pool AUCs.

  L4-v1 fold-0:
    < 0.8596 (below L3-prec fold-0 reference) → A3-pattern repeat → ABORT L4-v1 5-fold
    0.8596 - 0.8696 → marginal; defer 5-fold decision
    ≥ 0.8696 → DISPATCH full L4-v1 5-fold (4+1 split per CLAUDE.md)

  L4-v2 fold-0:
    < 0.8596 → ABORT L4-v2; write closeout
    0.8596 - 0.8696 → marginal; per the noisy-pseudo caveat (§36.A above),
        likely the realized signal was diluted by the 159 untrained-classes;
        defer 5-fold decision
    ≥ 0.8696 → DISPATCH full L4-v2 5-fold (4+1 split)

IF BOTH GATE-PASS:
  Both 5-fold runs ~26h DT each. Run them sequentially or split DT/skynet
  per the 4:1 heuristic. Cross-reference: do L4-v1 and L4-v2 5-fold
  ensembles diverge enough to fuse productively?

IF BOTH GATE-FAIL:
  L4 lever exhausted. Per §35-style closeout: lock LB 0.933 final.
  No new L5 in scope (within-arch ceilings established).

IF ONE PASSES AND ONE FAILS:
  Lean toward the passing one. Re-evaluate plan for whether
  the failing one's mechanism can be revived.
```

### §36.E — Risk pointers for tomorrow

- **Cron / runon blocking semantics**: the autopilot assumes `runon deepthought` blocks until the remote script finishes. If it doesn't (returns immediately after dispatch), step 5's eval would run too early on stale ckpts. Sanity-check by looking at the autopilot log timestamps: if step-4 `runon` returns in <5 min, that's the bug. Mitigation: a poll loop after the dispatch.
- **L4-v2 noisy-pseudo caveat**: 60.8% of pseudo positives are from 159 untrained-by-ProtoSSM classes. If L4-v2 gate-fails despite passing the smoke gate, this is the structural reason. Don't take it as falsification of "Perch features add value as a teacher" — it falsifies "uniform rank-mean fusion across all 234 classes adds value as a teacher." Per-class selective fusion (rank-mean on 71 trained, L3-prec on 163 untrained) is the obvious fallback if we want to try harder.
- **Gate threshold (+0.010 delta) is tight**: at the historical 0.08 OOF→LB transfer ratio, +0.010 broader-pool projects to +0.0008 LB — well below ±0.005 LB SE. Don't burn a v77 slot without a +0.05 ensemble gain.

### §36 — L4 dispatch and rationale

**L4 hypothesis (plan §553, §678):** iter934's `v5_pseudo2` succeeded *not* via recipe diversity (§35 falsified that for our pipeline), but because round-2 pseudo from a **different teacher** breaks the A2-self-anchor that killed A3.

We are testing TWO L4 variants in parallel because the ideal teacher (SED+ProtoSSM combined, faithful to iter934) requires Kaggle-Perch features which we don't have yet, and the cheaper SED-only teacher (L3-prec alone) is structurally similar to A3 and might fail the same way.

### §36.1 — L4-v1 (cheap, dispatched, training now)

**Recipe:** B0 + CE + seed 123 + mixstyle_p=0 + pseudo manifest from L3-prec 5-fold teacher @ threshold 0.7 (precision-matched to A2's A1@0.5). Identical to L3-prec training except the pseudo source.

**Current status (as of 2026-05-21 18:12 EDT):**
- **Where:** deepthought (PID 2246147)
- **Script:** `four_track/scripts/l4_fold0_smoke_dt.sh`
- **Log:** `deepthought:/home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260521_172151.log`
- **Started:** 17:21 EDT
- **Progress:** Epoch 2/25 done @ 18:07 EDT. val_v2 ROC-AUC trajectory: epoch 1=0.7848 ★, epoch 2=0.7954 ★. Both new bests. Training loss dropping cleanly (2.95 → 2.62).
- **Per-epoch:** ~22m 02s (DT, larger dataset than L3-prec due to 1.7× pseudo rows)
- **ETA finish:** ~02:40 EDT 2026-05-22
- **Ckpt destination:** `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed123_ce_l4.pt` (via `--ckpt-suffix _l4` flag added to `train_a1.py` this session)

**Comparison to L3-prec at same epochs** (same recipe, A2-pseudo): L3-prec fold 0 epoch 1=0.7268, epoch 2=0.7794. L4-v1 is **+0.058 / +0.016 ahead** on val_v2. No A3-style early regression. Caveat: per `feedback_per_fold_val_misleads_ensemble`, val_v2 during training is NOT predictive of broader-pool — the real gate runs after fold-0 finishes.

**L4-v1 gate matrix (derived from A3 kill data — A3 fold 0 broader-pool 0.7789 = −0.030 below A2 fold 0 0.8094, so the A3 mechanism IS detectable at fold-0):**

| L4-v1 fold-0 broader-pool | Verdict | Action |
|---|---|---|
| < 0.8596 (below L3-prec teacher fold-0) | A3-pattern repeat | **ABORT L4-v1.** Write closeout. Continue waiting on L4-v2. |
| 0.8596 – 0.8722 | Round-2 modestly works, ensemble projection (+0.018) won't clear +0.05 slot rule | Continue 5-fold for info value (~26h DT) only if budget allows. Otherwise abort. |
| ≥ 0.8722 (+0.013 over teacher) | Plausible slot-worthy projection | Commit to full 5-fold dispatch |

Fold-to-ensemble gain calibration: A2 fold-mean 0.8227 → ens 0.8402 (+0.018), L3-prec fold-mean 0.8504 → ens 0.8700 (+0.020). Consistent ~+0.018-0.020 gain.

**Eval script to run after fold-0 finishes:** clone `four_track/src/eval_l3_precursor_broader_oof.py`, change `CKPT_NAME_FMT` to `a1_..._fold{f}_seed123_ce_l4.pt`. Single-fold variant: only run fold 0, no ensemble.

**L4-v1 pseudo artifacts already on disk:**
- `four_track/data/processed/l3prec_pseudo_soundscape.npz` (113 MB) — raw 5-fold sigmean probs on 127,104 windows
- `four_track/data/processed/l4_pseudo_manifest_t070.csv` (12.4 MB, 126,131 rows) — A2-precision-matched threshold 0.7 manifest
- `four_track/data/processed/l4_pseudo_manifest.csv` (14.2 MB, 127,101 rows) — threshold 0.5 variant, NOT used (too many positives/window per §36.1 sanity check)

**Calibration sidebar** (worth remembering for any future SED pseudo work): L3-prec's prob scale is ~0.1 higher than A1's at matched FPR. L3-prec @ 0.7 ≈ A1 @ 0.5 in precision and TPR. Mean positives/window @ 0.5 was 4.29 vs A2's 1.17 — would have been a calibration disaster. Threshold 0.7 brings it to 2.44 — still 2× A2's but matched precision.

### §36.2 — L4-v2 (faithful iter934, Kaggle-Perch bottlenecked)

**Recipe:** Same B0/CE/seed123/no-mixstyle student. Teacher = (L3-prec 5-fold SED) ⊕ (ProtoSSM ensemble), combined under rank-mean per `reference_cross_arch_rank_mean_fusion`. Pseudo manifest threshold to be calibration-matched per the §36.1 sidebar.

**Hard constraint:** ProtoSSM consumes Perch v2 embeddings. Per `four_track/docs/new_plan_history.md` L26-27 and the killed-directions table, **local Perch features ≠ Kaggle Perch features** (proven by C2, #32, #34A LB collapses). We MUST get Perch embeddings from a Kaggle kernel.

**Kaggle extraction kernels — running now:**
- **p0:** https://www.kaggle.com/code/stevewatson999/birdclef-2026-perch-train-soundscapes-extract-p0
- **p1:** https://www.kaggle.com/code/stevewatson999/birdclef-2026-perch-train-soundscapes-extract-p1
- Pushed: ~18:05 EDT 2026-05-21
- ETA: ~4.8h each → ~23:00 EDT 2026-05-21
- Output per partition: `full_train_soundscapes_perch_p{0,1}of2.npz` + `full_train_soundscapes_meta_p{0,1}of2.parquet` in `/kaggle/working/`
- Source: `four_track/jupyter/perch-train-soundscapes-extract/` + `four_track/jupyter/perch-train-soundscapes-extract-p1/`
- Smoke history: v1 (TF-only path, 25 files in 4.4 min, projected 31h) → v2 (ONNX path fix via rglob, 25 files in 0.9 min, projected 9.5h, 3.4× speedup) → v3-equivalent = the production partitioned push above
- Build script supports `SMOKE=True/False` + `PARTITION_ID/PARTITION_COUNT` constants in Cell 1; edit and `python build_notebook.py` to regenerate the .ipynb
- Local merge script: `four_track/jupyter/perch-train-soundscapes-extract/merge_partitions.py` (expects PARTITION_COUNT=4 currently — **needs update to 2** before merge step)

**After Kaggle finishes (~23:00 EDT):**

1. Download both partition outputs:
   ```bash
   mkdir -p /tmp/perch_ss_p0 /tmp/perch_ss_p1
   kaggle kernels output stevewatson999/birdclef-2026-perch-train-soundscapes-extract-p0 -p /tmp/perch_ss_p0
   kaggle kernels output stevewatson999/birdclef-2026-perch-train-soundscapes-extract-p1 -p /tmp/perch_ss_p1
   ```
2. **Update `merge_partitions.py`**: change `PARTITION_COUNT = 4` → `2`, `PART_DIRS = [.../v0, .../v1]`, `_pXof4` → `_pXof2`. Stage to `kaggle_datasets/train-soundscapes-perch/`.
3. **Build `eval_l3prec_x_protossm_broader_oof.py` (teacher smoke gate, task #18):** load (a) L3-prec 5-fold sigmean probs on broader-pool val [already exists in `four_track/data/l3_precursor_broader_oof.npz`], (b) ProtoSSM probs on broader-pool val using the new Kaggle-Perch features [needs Perch on val pool too — see "Open issue" below]. Combine under sig-mean and rank-mean. Report ensemble AUC vs L3-prec standalone 0.8700. **Gate: combined ≥ 0.8800 (≥+0.010 over L3-prec standalone). If combined < L3-prec standalone, the L4-v2 mechanism is dead before training.**
4. **If teacher gate passes:** generate L4-v2 pseudo on train_soundscapes using the combined teacher (rank-mean of L3-prec SED + ProtoSSM). Apply same threshold-calibration logic as §36.1 (probably threshold ~0.55-0.65, will need to recompute the matched-precision point against A2 baseline).
5. **Train L4-v2 fold-0 smoke** with `--ckpt-suffix _l4v2`. Same gate matrix as §36.1.

**Open issue (must resolve before step 3):** The Kaggle kernels currently only extract Perch on `train_soundscapes/` (the 10,592 unlabeled pool). To smoke-gate the combined teacher on broader-pool val (1478 windows from 66 LABELED soundscapes), we also need Perch features on those 66 files. Two options: (a) extract a tiny additional Kaggle kernel for the 66 val files (~5 min Kaggle CPU); (b) modify the existing kernel to include ALL soundscapes (10,658 = 10,592 + 66) and filter downstream. Option (a) is simpler — push a quick smoke-style kernel with just the 66 val files when needed.

### §36.3 — Decision tree for next session

```
WHEN L4-v1 fold-0 finishes (~02:40 EDT):
  Run eval_l4v1_fold0_broader_oof.py (clone + minimal edit of eval_l3_precursor_broader_oof.py)
  Apply §36.1 gate matrix:
    < 0.8596  → ABORT L4-v1, doc kill in memory, wait for L4-v2
    0.86-0.87 → marginal, document, defer 5-fold decision
    ≥ 0.8722  → DISPATCH full L4-v1 5-fold (4+1 split: DT folds 1-4, skynet fold 0 ckpt already exists)
                 ~9h wall-clock to complete, then 5-fold broader-pool eval

WHEN Kaggle extraction finishes (~23:00 EDT):
  Pull both partitions, merge, stage as Kaggle dataset
  Push small Kaggle kernel to extract Perch on the 66 val files (~5 min)
  Build eval_l3prec_x_protossm_broader_oof.py for combined-teacher gate
  Apply gate:
    combined < L3-prec (0.8700) → ABORT L4-v2, write closeout, accept v75 0.933
    combined < 0.8800            → L4-v2 hypothesis weak, train fold-0 only as final probe
    combined ≥ 0.8800            → DISPATCH L4-v2 pseudo gen + fold-0 smoke

IF BOTH L4-v1 AND L4-v2 GATE-FAIL:
  Write the within-arch-exhausted closeout. Accept v75 0.933 as final.
  Possibly push a v78 LB probe with L3-prec standalone (information-value only,
    expected ΔLB +0.0024 below noise). Decide based on remaining LB slots.
```

### §36.4 — Residual SSM investigation outcome (not a lever, do not reopen)

Investigated 2026-05-21 ~17:55 EDT. Findings:
- ResidualSSM is **already wired and active in production** at `correction_weight = 0.10`. Not a dormant lever.
- `ENSEMBLE_WEIGHT_PROTO = 0` does NOT gate ResidualSSM (per Explore agent finding — they're independent in code).
- iter934's `correction_weight = 0.35` was tried as v77 yesterday → LB 0.930 vs v75 0.933, regressed, reverted. See `project_v77_l1_correction_weight_neutral`.
- `models/protossm_pretrained/residual_ssm.pth` is NOT what runs in production (Cell 36 trains fresh in-kernel each run); the local .pth is a side-experiment artifact.
- Remaining ResidualSSM variations (other weights, architecture tweaks, longer training) would each cost ≥1 LB slot for expected gains in the noise floor.

**Do not reopen ResidualSSM tuning as a lever.** Closeout filed in this session's memory; the "easy structural" version has been tried.

### Files modified this session (for git context)

- `four_track/new_plan.md` — §35 L3 closeout added + this PICK UP HERE
- `four_track/src/train_a1.py` — added `--ckpt-suffix` flag + threading through `train_one_fold` (3 surgical edits, lines ~201, ~363, ~553)
- `four_track/src/l4_emit_pseudo_l3prec.py` — NEW (clone of a2_pseudo_label_a1.py with raw state-dict loading for seed123_ce ckpts)
- `four_track/src/eval_l3_precursor_broader_oof.py` — NEW
- `four_track/src/eval_a2_x_l3_precursor_bag.py` — NEW
- `four_track/scripts/l4_fold0_smoke_dt.sh` — NEW
- `four_track/jupyter/perch-train-soundscapes-extract/` — NEW (build_notebook.py adapted from perch-train-audio-extract, kernel-metadata.json, merge_partitions.py)
- `four_track/jupyter/perch-train-soundscapes-extract-p1/` — NEW (clone of above with PARTITION_ID=1)
- Memories added: `project_l3_multirecipe_bag_killed`, `project_l3_prec_recipe_upgrade`

### Deadline reminder

Competition deadline **2026-06-03 — 13 days from now**. v75 LB 0.933 vs public-reproducible 0.947 leaves ~0.014 headroom. L4 prior +0.002–0.008 with 30–40% pass-noise probability would, if it lands, close ~half that gap.

### §35 — L3 closeout (broader-pool eval results) [historical, kept for reference]

| Operator | A2 5-fold | L3-prec 5-fold | Bag (10 ckpts) | Δ bag vs A2 | Δ bag vs stronger |
|---|---|---|---|---|---|
| sig-mean | 0.8402 | **0.8700** | 0.8481 | +0.0079 | **−0.0219** |
| rank-mean | 0.8417 | 0.8682 | 0.8608 | +0.0206 | **−0.0092** |

Per-fold L3-prec broader-pool: 0.8596, 0.8455, 0.8536, 0.8500, 0.8431 (mean 0.8504). Uniformly above A2 per-fold.

**Two findings:**

1. **Bag-vs-stronger-standalone is negative under both operators** → A2 actively damps L3-prec's better predictions. The plan's gate (bag vs A2 anchor) was answering the wrong question; under the diversity-correct test (bag vs stronger), L3-prec ⊕ A2 fails.
2. **L3-prec standalone is +0.030 broader-pool above A2** purely from the recipe change (ASL→CE, seed42→seed123, mixstyle 0.5→0). This is a real recipe upgrade — large enough to be informative but **below the +0.05 broader-pool slot rule** (expected ΔLB at transfer ratio 0.08 = +0.0024, below the ±0.005 LB SE noise floor).

**Decision:** L3 lever exhausted. (c)(d)(e) would cost ~12h DT and is unlikely to reverse the bag-vs-stronger sign. Skipped.

Eval artifacts:
- `four_track/src/eval_l3_precursor_broader_oof.py`
- `four_track/src/eval_a2_x_l3_precursor_bag.py`
- `four_track/data/l3_precursor_broader_oof.npz`
- `four_track/log/eval_l3_precursor_*.log`
- `four_track/log/eval_a2_x_l3_bag_*.log`
- Training logs archived at `four_track/log/archive/l3_precursor_{skynet,dt_dispatch}_*.log`

### Critical-thinking flag (carry into the gate decision)

**The pseudo-manifest is identical between A2 (a) and L3-precursor (b).** Both train on `a2_pseudo_manifest.csv` — pseudo-labels generated by A2 itself. The recipe knobs that differ are loss (ASL→CE), seed (42→123), mixstyle (0.5→0). The pseudo-targets carry A2's representation and partially anchor (b)'s solution toward A2's regardless of the loss-landscape difference.

Implications:
- If broader-pool comes in at the marginal 0.80–0.85 band, the structural reason is pseudo-coupling — *not* "recipe diversity doesn't work in principle."
- Training (c) `v5_ce_s456` (different seed only) shares this coupling. Same anchor.
- **The actual decoupling lever is (d) `v5_focal_pseudo_r2`** — pseudo round-2 generated by a *different teacher* (full SED+ProtoSSM ensemble, not A2-alone). That's the one that breaks the anchor.
- Consider whether (d)'s pseudo round-2 generation step is worth starting *pre-emptively* before the gate fires (parallel work, separate machine), so the critical path is shorter if the gate passes marginal.

### Lever priors snapshot

Local LB SE is ±0.003. Anything <+0.005 expected LB gain is invisible.

| Lever | Prior gain | Pass-noise-floor probability |
|---|---|---|
| L3 (multi-recipe SED bag, 5d DT including precursor) | +0.003–0.012 | ~40–50% |
| L4 (iter-pseudo round 2 ensemble teacher, 3d DT) | +0.002–0.008 | ~30–40% |

### Don'ts (still in force)

- **Don't dispatch any GPU run without the pre-flight checklist** (archive + clean log dirs + `nvidia-smi` target). Memory `feedback_always_clean_logs_before_dispatch`.
- **Don't propose anything in the Killed Directions table** without strictly different mechanism. L2 (BirdNET 3rd branch) and T1 (per-class thresholds) are newly killed.
- **Don't propose AST or non-CPU-feasible backbones in production** — `feedback_ast_cpu_infeasible`.
- **Don't propose single-config copies from top notebooks** with expected LB gain <+0.005 — `project_v77_l1_correction_weight_neutral`.
- **Don't relax the +0.05 broader-pool gate by fiat.** v77 gave one data point; need multiples to recalibrate.
- **Don't trust §34.4-style compute estimates** without re-deriving against the actual recipe — `feedback_verify_compute_time_estimates`.
- **Don't benchmark new probes against v76 (LB 0.932)** — v75 (LB 0.933) is the LB high.

### Deadline + time-box reminder

Competition deadline **2026-06-03 — 14 days from now**. Public top 0.962, public-reproducible ≥0.947. v75 at 0.933 has 0.014–0.029 LB headroom. Intuition pump: "what's in iter934 / 0.947 that we don't have?" — adopt before invent.

§33's 3-day box expired 2026-05-19; §34 closed 2026-05-20. **No more compute on bird-audio-pretrain, ensembling-operator variants, or within-arch bagging.** L3 (recipe diversity) and L4 (pseudo round 2) are the remaining sanctioned levers.

---

## 📌 PICK UP HERE — INVALIDATED 2026-05-23 ~23:45 EDT, see §43 (end of file) for current state

> _Original §37 handoff (2026-05-23 ~00:20 EDT — L4-v2 main 5-fold in flight, autopilot will eval at ~16:15 EDT Sat)._
>
> **All prior PICK UP HERE sections are invalidated** (the §36-era one above included).
> This is the single current handoff.

### TL;DR — one paragraph

L4-v1 KILLED (fold-0 broader-pool 0.8389 vs L3-prec fold-0 reference 0.8596; A3-pattern repeat). L4-v2 fold-0 seed-123 GATE-PASSED at broader-pool **0.9253** (+0.0657 over L3-prec fold-0; +0.0851 over A2 anchor 0.8402). Sanity-checked with seed-456 fold-0 retrain on DT: convergence pattern was robust across seeds (epoch 5 hit 0.9170 = inside ±0.010 pass window of 0.9253; killed mid-run to save compute). L4-v2 main run dispatched 2026-05-22 17:04 EDT: folds 1,2,3 sequential on DT + fold 4 in parallel on skynet. ~23h to all-five-folds done, autopilot will rsync ckpts + run 5-fold broader-pool eval automatically. **Verdict ETA: ~16:15 EDT Saturday 2026-05-23.** Production LB stays v75 0.933 until verdict.

### §37.A — Current live processes (as of 2026-05-23 00:14 EDT)

| Process | Host | PID | Status | ETA |
|---|---|---:|---|---|
| L4-v2 folds 1,2,3 sequential | DT | 2558879 | running, fold 1 ep 23/25 | DT all-3 done ~16:02 EDT Sat |
| L4-v2 fold 4 | skynet | 1308128 | running, ep 8/25 | done ~14:02 EDT Sat |
| Monitor (poll + rsync + 5-fold eval) | skynet | 1308836 | silent, polling DT log | fires after both above complete |

**Wall-clock floor: DT's 3-fold sequential → ~16:02 EDT Sat. Verdict in monitor log by ~16:15 EDT Sat.**

To check progress in the morning:
```bash
tail -f log/l4v2_main_run_monitor_*.log  # monitor heartbeat (silent until verdict)
tail -f log/l4v2_fold4_skynet_*.log      # skynet fold 4 epoch-by-epoch
ssh deepthought "tail -f /home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260522_170350.log"  # DT folds 1,2,3
```

### §37.B — What the gate matrix says (when verdict lands)

The 5-fold broader-pool ensemble AUC from `eval_l4v2_5fold_broader_oof.py` will print verdict against three anchors:

| 5-fold ensemble | Action | Why |
|---|---|---|
| ≥ 0.8902 (A2 anchor +0.05) | **GATE-PASS — push v77 LB probe** | Clears `feedback_min_oof_delta_to_burn_slot` |
| 0.8700–0.8902 | Above L3-prec but below +0.05 gate | Borderline; check OOF→LB transfer ratio. If you want one LB probe, this is the marginal case to consider — but expected ΔLB ≈ transfer × broader-pool delta = 0.08 × ~0.012 = ~+0.001, at the noise floor |
| 0.8402–0.8700 | Below L3-prec anchor, above A2 | Main run regressed vs L3-prec recipe upgrade alone |
| < 0.8402 | Below A2 anchor — L4-v2 lever falsified | Close out |

### §37.C — Honest projection ranges at ~00:14 EDT Sat

Based on fold 1 best so far (0.9244 at ep 14) and fold 4 best so far (0.9168 at ep 4), with folds 2,3 still to come:

| Scenario | Per-fold mean | Ensemble lift | 5-fold ensemble | LB projection (×0.08 transfer) |
|---|---:|---:|---:|---:|
| Optimistic (folds 2,3 late-climb like fold-0) | 0.923 | +0.020 | 0.943 | LB ~0.945 |
| Realistic (folds 2,3 like fold 4 = peak-then-plateau) | 0.918 | +0.015 | 0.933 | LB ~0.939 |
| Pessimistic (high pseudo-manifest fold-correlation) | 0.918 | +0.010 | 0.928 | LB ~0.937 |

All three scenarios gate-pass and project to **+0.004 to +0.012 LB over v75**. The fold-0 seed-123 0.9253 may have been a lucky pairing — early fold-1 / fold-4 results show peak-and-plateau pattern, not the late-epoch climb fold-0 exhibited (peak ep 21). Don't anchor expectations to 0.945; the realistic case is ~0.939.

### §37.D — Decision tree for tomorrow

```
WHEN MONITOR EXITS (~16:15 EDT Sat 2026-05-23):

  Read log/eval_l4v2_5fold_<timestamp>.log for the verdict.

  IF ensemble ≥ 0.8902:
    → Push v77 LB probe.
    → Production kernel is jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb.
    → Slot the L4-v2 ckpts into the SED branch (same shape as L3-prec ckpts);
      sig-mean across the 5 folds.
    → Submit. Wait for LB. Compare to v75 0.933.
    → If LB ≥ 0.938: declare v77 = new LB high. Save as final.
    → If LB 0.933–0.938: within noise; check second submission if slots remain.

  IF ensemble 0.8700–0.8902:
    → Marginal. Expected ΔLB +0.001 at noise floor.
    → Don't push unless you have spare LB slots (>2 unused) AND want a
      data point for the transfer-ratio calibration question §36.4 raised.

  IF ensemble < 0.8700:
    → L4-v2 main-run regression. Add to Killed Directions table.
    → Lever class "ensemble-teacher iter-pseudo round 2" then exhausted.
    → Close out — v75 0.933 stays final.

  ALSO check log/eval_l4v2_5fold_<timestamp>.log for per-fold AUCs.
  If fold 4 (skynet) lands materially below DT folds, hardware-induced
  fold variance is in play — not a recipe problem.
```

### §37.E — Artifacts created this session (2026-05-22 → 2026-05-23)

| Path | Purpose |
|---|---|
| `data/l4l4_fold0_broader_oof.npz` | L4-v1 fold-0 single-fold OOF (0.8389) — kept for reference |
| `data/l4l4v2_fold0_broader_oof.npz` | L4-v2 fold-0 single-fold OOF (0.9253) — kept for reference |
| `models/a1/a1_<bb>_fold0_seed456_ce_l4v2.pt` | L4-v2 fold-0 seed-456 sanity ckpt on DT only (killed mid-run, partial) — NOT pulled back |
| `scripts/l4v2_fold0_seed456_dt.sh` | Sanity-check dispatch (used, kept) |
| `scripts/l4v2_seed456_monitor.sh` | Sanity-check monitor (used, kept) |
| `scripts/l4v2_seed456_autopilot_dt.sh` | BUGGY first autopilot (runon-doesn't-block) — keep as cautionary example |
| `scripts/l4v2_folds123_dt.sh` | Main run DT dispatch (running now) |
| `scripts/l4v2_fold4_skynet.sh` | Main run skynet dispatch (running now) |
| `scripts/l4v2_main_run_monitor.sh` | Main run monitor (running now) |
| `src/eval_l4v2_5fold_broader_oof.py` | 5-fold broader-pool ensemble eval (will fire automatically) |
| `src/eval_l4_fold0_broader_oof.py` | Modified to accept `--seed` arg |
| `docs/nvidia_sm121_forum_post.md` | Draft NVIDIA developer-forum post requesting sm_121-tuned cuDNN. NOT POSTED yet. User decided to write the post but timing isn't competition-relevant. |

### §37.F — Memories added this session (for cross-reference)

- `feedback_runon_does_not_block` — `runon deepthought <cmd>` returns in <1s; autopilots must `until ssh deepthought "tail … grep -qE marker"` poll loop
- `feedback_syncback_skips_four_track_models` — bare `syncback deepthought` only covers parent-BirdCLEF paths; four_track/models/ ckpts need explicit rsync

### §37.G — Things to NOT do tomorrow

- **Don't kill the running training processes** (DT PID 2558879, skynet PID 1308128, monitor PID 1308836). They'll complete naturally and the monitor handles the eval automatically.
- **Don't trust the §37.C optimistic case** (LB 0.945). The realistic case (LB 0.939) and pessimistic case (LB 0.937) are both gate-passing too, but the trajectory in folds 1 and 4 has been peak-and-plateau rather than late-epoch climb. Anchor expectations to the realistic case.
- **Don't push v77 to LB without first reading the per-fold AUCs.** If fold 4 (skynet) is materially weaker than fold 0 / 1 / 2 / 3, the ensemble may benefit from a drop-fold-4 variant per `feedback_per_fold_val_misleads_ensemble` — though that memory says don't drop folds without an actual rank-fusion probe.
- **Don't trust skynet fold 4's val_v2 = broader-pool equality blindly.** It's verified true for L4-v2 fold-0 (0.9253 ≈ 0.9253) but the in-training val printout IS the broader-pool metric per `train_a1.py:75-110`.
- **Don't restart the dispatch if a single fold fails.** Save the partial ckpts; you can run a 4-fold ensemble instead of 5-fold and document the asymmetry.

### §37.H — Side-effect cleanups still outstanding (low priority)

- The previous (buggy) autopilot's rsync left some metadata in parent `BirdCLEF/log/` and `BirdCLEF/data/processed/`. Audited 2026-05-22 — those files are LEGACY pre-four_track content; the rsync was a no-op (sent 1.4 KB of 7.16 GB). No action needed.
- `data/l4l4_fold0_broader_oof.npz` and `data/l4l4v2_fold0_broader_oof.npz` have the double-suffix filename bug (the eval script's f-string concatenates `l4` + `{suffix}` where suffix is `_l4`). Cosmetic; the data inside is correct. Don't fix mid-run.

### Deadline reminder

Competition deadline **2026-06-03 — 11 days from now**. v75 LB 0.933 vs public-reproducible 0.947 = 0.014 headroom. L4-v2 verdict tomorrow determines whether we close that gap or close the book.

---

## §38 — L4-v2 LB CLOSEOUT (2026-05-23 ~19:10 EDT)

**LEVER KILLED.** L4-v2 5-fold (broader-pool OOF 0.9310 sig-mean, +0.091 over A2 anchor) regressed on LB despite the +0.05 broader-pool slot rule clearing by +0.041.

| Kaggle ver | Config | LB | Δ vs v75 |
|---|---|---:|---:|
| v75 (anchor) | A2 ckpts + rank-mean | **0.933** | — |
| v76 | A2 ckpts + rank-mean (rank-fusion intro) | 0.932 | −0.001 |
| v77 | + L1 correction_weight 0.35 (REVERTED) | 0.930 | −0.003 |
| **v78** | **L4-v2 ckpts + sig_mean** (§37.D pre-reg) | **0.928** | **−0.005** |
| **v79** | **L4-v2 ckpts + rank-mean** (isolation) | **0.926** | **−0.007** |

v79 isolation submission proved **the ckpts themselves are the killer, not the reducer choice.** Reverting reducer (sig-mean → rank-mean) made it 0.002 worse, counter to `reference_cross_arch_rank_mean_fusion` expectations.

**Structural explanation (high confidence):** L4-v2 ckpts were trained on ProtoSSM-derived pseudo-labels. SED predictions now overlap in error patterns with the ProtoSSM branch in the 4-branch production ensemble. The 4-branch rank-blend gain collapses when two branches are correlated. The offline broader-pool gate was SED-only and structurally blind to this. Memory: [[project_l4v2_lb_regression]].

**Falsified assumptions from §37.D pre-registration:**
1. "OOF→LB transfer is roughly monotonic positive" — actual transfer was negative.
2. "+0.05 broader-pool slot rule is a sufficient gate" — when the swap changes inter-branch correlation, SED-only OOF is structurally blind.

**Production notebook reverted to pre-v78 state** (`bak_pre_v78_l4v2` → main file). A1_CKPT_DIR back to A2-ckpts, A1_FOLD_REDUCE back to "mean". L4-v2 ckpts + Kaggle dataset preserved for forensic re-runs.

**Slot budget today:** 2/5 used (v78, v79). 3 remaining.

---

## §39 — Competitor postproc-lever audit (2026-05-23 ~19:25 EDT)

Mined the top 7 BC2026 notebooks (mtoshidesu 0.947 reproducer + nina EoS-6 + ulyanov 0.945 + cliff376 + needless090 0.934 + maryna two-pass SSM + tucker distilled-SED). Cross-referenced against killed-lever memory.

### Top 3 novel levers (all postproc, no retraining)

| # | Lever | Where seen | Cost | Why >+0.005 |
|---|---|---|---|---|
| 1 | **3-gate rank-blend rescue** (fake_only + proto_cont + sed_only) | mtoshidesu 0.947, cliff376 0.941, ulyanov 0.945 (3 of 7 top notebooks) | 2-4h port + sweep | proto_cont gate adds cross-window temporal context that ProtoSSM's 12-window inference structurally lacks; not in kill list |
| 2 | **Sonotype mirroring** (4 MIRROR_PAIRS, group-max) | mtoshidesu, needless090 — same exact groupings | <1h | LB target collapses sonotypes; two independent convergences = strong social proof |
| 3 | **Rare-class shrinkage** (Amphibia/Mammalia/Reptilia: val<mean+0.06 → val×0.85) | mtoshidesu only | <1h | Mechanistically aligned with [[feedback_encoder_taxonomy_must_match_lb]] — Aves-trained encoders generate near-noise on non-Aves taxa |

**Skipped (low EV):** lambda_prior scalar bumps (v77 L1 failure pattern); two-config dual blend (refinement of #1, gated on it).

**Verified absent from production:** grep for `fake_only`, `proto_cont`, `proto_kernel`, `MIRROR_PAIRS`, `rare_classes` in `birdclef2026-protossm-postproc.ipynb` returned zero hits.

### Hard lesson from L4-v2 applied to this plan

The mtoshidesu gate constants (0.55, 0.08, 0.94, 0.78, 0.12, blend 0.05/0.12/0.08) were tuned for **their** upstream SED/ProtoSSM models, not ours. **Copy-paste without OOF re-tuning is the exact L4-v2 failure pattern.** Two non-negotiable safeguards before LB push:

1. **Local OOF tuning of gate constants** on matched A1+ProtoSSM substrate before push.
2. **Multi-config sweep** — at minimum a 3×3 grid on the most-sensitive constants.

### Available OOF substrates

- `data/d2_beta_oofs.npz` (April 15, **stale models** — has matched a1_ranks + proto_oof on 708 rows × 234 classes). Substrate is 59 files × 12 windows — different from the LB-correlated 1478-window broader-pool but the only matched-pair OOF we have.
- `data/a2_a1_5fold_broader_oof.npz` and `data/l4v2_5fold_broader_oof.npz` — 1478 broader-pool rows, A1 SED only, no matched ProtoSSM. Not directly usable for gate-triad tuning.

**Plan:** Re-generate d2_beta_oofs.npz fresh against current-production models via a Kaggle notebook push in MODE=train (~7 min wall, no LB slot, no training). Then sweep gate constants locally on the fresh 708-row OOF substrate. Push best config to LB as v80.

### Step-by-step (highest priority)

1. **Re-gen d2_beta_oofs.npz on current models.** Flip `MODE` from "submit" to "train" in the notebook, push to Kaggle (no LB slot used because train mode doesn't submit). Pull `d2_beta_oofs.npz` from kernel output. Flip MODE back. ~15 min round-trip.
2. **Implement gate triad in standalone script** `src/postproc_gate_sweep.py` mirroring mtoshidesu cell 34 logic (fake_only + proto_cont + sed_only). Load d2_beta_oofs, compute baseline AUC, compute gated AUC across a small constant grid.
3. **If material gain (OOF AUC delta > +0.005) is found**, port the chosen constants into the notebook (new cell after A1 rank fusion). Otherwise stop and reconsider whether the gate-triad lever fits this pipeline.
4. **Layer sonotype mirroring + rare-class shrinkage** as additional cells (cheap, can stack independently).
5. **Push v80 to LB** with the combined post-gate config.

---

## 📌 PICK UP HERE — INVALIDATED 2026-05-23 ~23:45 EDT, see §43 (end of file) for current state

> _Original §39 handoff (2026-05-23 ~19:30 EDT — L4-v2 closed, postproc-gate lever staged)._

### TL;DR — one paragraph

L4-v2 lever closed (v78=0.928, v79=0.926 — both regressed vs v75 0.933). Competitor postproc audit (7 top notebooks) surfaced three novel postproc levers, ranked: (1) 3-gate rank-blend rescue, (2) sonotype mirroring, (3) non-Aves rare-class shrinkage. All postproc, no retraining. Per the L4-v2 lesson, we OOF-tune gate constants locally before any LB push. Production notebook is back at v76 (A2 + rank-mean) state. 9 days left to deadline; 3 LB slots remaining today.

### Live processes

None. All training quiesced. No background jobs.

### Next actions in order

1. Re-generate fresh d2_beta_oofs.npz (Kaggle MODE=train push, no LB cost). ~15 min round-trip.
2. Implement `src/postproc_gate_sweep.py` (mirror mtoshidesu cell 34 logic, sweep small constant grid).
3. Decide gate-triad fate based on local OOF AUC delta vs baseline; sonotype + non-Aves stacking is cheap regardless.
4. v80 LB push with tuned config if local OOF shows ≥ +0.005 gain.

### Production state

- LB high: **v75 0.933** (A2 + rank-mean)
- Notebook: reverted to bak_pre_v78_l4v2 baseline
- L4-v2 ckpts + Kaggle dataset preserved (forensic-only; not in production)

### Things NOT to do

- **Don't push gate-triad constants from mtoshidesu without local OOF tuning** — same failure pattern as L4-v2.
- **Don't burn slots on lambda_prior / rank_aware_power scalar bumps** — Nina EoS-6 reports +0/+0.001 gains, below ±0.005 LB SE.
- **Don't propose another "more bird audio" pretrain** ([[project_xc_pretrain_lever_exhausted]]) or another cross-arch fold ([[project_cross_arch_ceiling]]).

---

## §40 — v80 sonotype + non-Aves shrinkage CLOSEOUT (2026-05-23 ~20:00 EDT)

**BOTH LEVERS CLOSED at LB Δ=0.000.**

| Kaggle ver | Config | LB | Δ vs v75 |
|---|---|---:|---:|
| v75 (anchor) | A2 + rank-mean (current production) | **0.933** | — |
| v80 | v75 + sonotype mirror + non-Aves shrinkage | **0.933** | **0.000** |

The two postproc tricks from mtoshidesu 0.947 reproducer + needless090 0.934 — sonotype mirroring on 4 MIRROR_PAIRS (10 affected columns) and non-Aves rare-class shrinkage on 44 columns (35 Amphibia + 8 Mammalia + 1 Reptilia) — landed at LB 0.933, identical to v75 baseline. Neither helped nor hurt. Memory: [[project_sonotype_shrinkage_neutral]].

**Two of three §39 levers now confirmed exhausted standalone.** Only the gate triad remains untested live. Notebook reverted to v75 baseline.

**Slot count today: 3/5 used (v78, v79, v80). 2 remaining.**

### What's left

| Lever | Status | Cost | Risk |
|---|---|---|---|
| Gate triad with proper OOF tuning | UNTESTED | 4-6h engineering (extract post-postproc ProtoSSM OOF) + sweep + 1 LB slot | medium — OOF→LB transfer may still fail like L4-v2 |
| Gate triad with mtoshidesu defaults, blind push | UNTESTED | 30 min + 1 LB slot | high — L4-v2 anti-pattern |
| Accept v75 0.933 as final | — | 0 | 0 |

### Updated decision point

The day's data: three competitor-notebook ports → one regression, one regression, one neutral. The growing case is that our pipeline's tunable parameters don't compose cleanly with mtoshidesu-style postproc tricks; the +0.014 gap may live in upstream model differences (SED training recipe, ProtoSSM training, OOF-tuned per-class thresholds tuned on a substrate we don't have) rather than in tunable postproc.

The gate triad is the one remaining unmoved stone. Whether it's worth the 4-6h engineering investment depends on prior on the +0.014 gap actually being closable from our position — and three failures today shifted that prior down.

---

## §41 — v81 Gate 3 CLOSEOUT + §39 lever closeout (2026-05-23 ~22:25 EDT)

**ALL §39 LEVERS NOW EXHAUSTED LIVE.**

| Kaggle ver | Config | LB | Δ vs v75 |
|---|---|---:|---:|
| v75 (anchor) | A2 + rank-mean | **0.933** | — |
| v78 | L4-v2 + sig-mean | 0.928 | −0.005 |
| v79 | L4-v2 + rank-mean | 0.926 | −0.007 |
| v80 | v75 + sonotype + non-Aves shrink | 0.933 | 0.000 |
| **v81** | **v75 + Gate 3 only (mtoshidesu defaults)** | **0.930** | **−0.003** |

Gate-3 isolation OOF probe (post-postproc-proto matched 708-row substrate) had cleanly identified Gate 3 as the only gate of the three that moved AUC (+0.0047, with Gates 1 + 2 contributing ±0.0001). LB outcome: −0.003 regression. Memory: [[project_gate3_only_lb_regressed]].

**Now twice-confirmed:** the matched 708-row OOF substrate cannot predict LB for ANY change to the SED-vs-ProtoSSM rank balance. The substrate omits the residual-SSM correction (cell ~3411) and the Perch CV-blend, which shift the relative score scales in production beyond what the OOF simulates. L4-v2 (+0.091 OOF → −0.005 LB) + Gate 3 (+0.005 OOF → −0.003 LB) are the two data points.

**Slot budget today: 4/5 used. 1 slot remaining (resets at midnight UTC).**

### What's actually left

| Lever | Status | Est. cost | Likely LB delta |
|---|---|---|---|
| Build full-pipeline OOF (all 4 branches + final blend) on broader-pool | UNTESTED | 1-2 days engineering | Unknown — would enable proper gate-triad and other postproc tuning |
| Full mtoshidesu gate triad (defaults) — blind push without OOF | UNTESTED | 30 min | Probably neutral or regression (Gate 3 alone already regressed) |
| Accept v75 0.933 as final | — | 0 | 0 |
| Try Gate 2 (proto_cont) alone with tuned constants | UNTESTED | 1h + 1 slot | Unknown — but our OOF says it does nothing; LB transfer may differ |

### Decision point

10 days left. 5 fresh slots/day available going forward. The §39 audit's 5 levers are now closed (all failed or neutral). Building a full-pipeline OOF substrate is the only path to systematic improvement, and it's 1-2 days of engineering for unknown gain.

The right call from here may genuinely be **accept v75 0.933 final**, write up the closeout, and stop. Two days, ~9 slots burned, zero LB gain — and the OOF substrate problem isn't going away without significant engineering investment.

---

## §42 — MTOSHIDESU CLONE LB 0.944 — gap closed (2026-05-23 ~23:30 EDT)

**Pivot worked.** Pushed mtoshidesu's full 0.947 pipeline notebook verbatim to a new private kernel under our credentials. **LB = 0.944** in one slot. +0.011 vs v75 0.933; within ±0.005 of mtoshidesu's claimed 0.947.

**Kernel:** `stevewatson999/bc2026-mtoshidesu-clone-test` v1 (private)
**Source:** `mtoshidesu/birdclef-2026-0-947-lb-public-pipeline-reproduced` (public)
**Dataset sources (all public):**
- `tuckerarrants/bc2026-distilled-sed-public`
- `tuckerarrants/perch-v2-no-dft-onnx`
- `rishikeshjani/perch-onnx-for-birdclef-2026`
- `jaejohn/perch-meta`

| | LB |
|---|---:|
| v75 (our best before tonight) | 0.933 |
| v81 (Gate 3 piecewise port — regressed) | 0.930 |
| v80 (sonotype + non-Aves piecewise) | 0.933 |
| v78/v79 (L4-v2 piecewise) | 0.928 / 0.926 |
| **clone v1 (full mtoshidesu verbatim)** | **0.944** |
| mtoshidesu's claimed | 0.947 |

**Why this worked when piecewise ports failed:** mtoshidesu's gate triad / per-class thresholds / sonotype groupings were all tuned for THEIR specific upstream models (SED ckpts, ProtoSSM, ensemble blend). Porting one technique onto OUR pipeline applied tuned constants to a different score distribution, breaking the calibration. The techniques weren't wrong; they were calibrated for a pipeline that wasn't ours. Memory: [[project_mtoshidesu_clone_reproduced_944]].

### Slot budget update

Today: 5/5 used (v78, v79, v80, v81, clone). Tomorrow's 5 reset at midnight UTC.

### Current best LB: 0.944 (clone v1)

### Tomorrow's open questions

1. **Can we ensemble our v75 (0.933) with the clone (0.944) for > 0.944?**
   If our predictions and theirs are decorrelated per-row, a rank-mean ensemble could exceed both. Worth testing once tomorrow (one slot). Expected gain: small, but the floor is "lose nothing" since 0.944 is the safety net.

2. **Can we further improve the clone by porting our better A1 SED ckpts?**
   They use tuckerarrants distilled-SED (a public dataset). Our A2 SED ckpts might be stronger or weaker; an isolated swap (their pipeline minus their SED, plus our A2) tests this. Higher risk than #1.

3. **Multiple final-submission selection rule.** Kaggle lets you pick 2 submissions for private LB evaluation. The clone 0.944 is one. The second slot should be either our v75 0.933 (diverse pipeline → hedge against private LB shifts) or a hopefully-better ensemble.

### Decision tree for tomorrow

```
DEFAULT: clone v1 (LB 0.944) is the locked floor.

IF you want to gamble for higher:
  - Push 1 ensemble (clone + v75 rank-mean) → if > 0.944, becomes new floor
  - Stop after ≤2 more slots; diminishing returns

IF you want to hedge:
  - Final submission selection: clone v1 + v75 (architecture diversity)

IF you want to stop:
  - Mark clone v1 as final selection. Done. 0.944 is the result.
```

---

## 📌 PICK UP HERE (2026-05-23 ~23:45 EDT — clone v1 LB 0.944 locked; five paths staged for tomorrow)

> **All prior PICK UP HERE sections invalidated** (§37 and §39 era). This is the single current handoff.

### TL;DR

Two days, 9 piecewise lever ports, zero gain. Tonight's pivot: pushed mtoshidesu's full 0.947-LB notebook verbatim under our credentials → **LB 0.944** (+0.011 vs v75 0.933). Kernel `stevewatson999/bc2026-mtoshidesu-clone-test` v1 is the new floor and one of the two final-submission picks. Daily slots: 5/5 used today; resets at midnight UTC. Competition deadline 2026-06-03 (10 days).

### Current best LB: **0.944** (clone v1)

### Five strategies, ranked by EV for closing the remaining ~0.003-0.019 gap to top of LB

| # | Strategy | Cost | Expected Δ over 0.944 | Notes |
|---|---|---|---:|---|
| 1 | **Clone ⊕ v75 rank-mean ensemble** | 1h dev + 1 slot | +0.002 to +0.005 | Different pipelines = decorrelated rows → ensemble lift. Floor stays 0.944 (clone as 2nd pick). LOW RISK. |
| 2 | **Clone nina2025 EoS-6** (271 votes, claims 0.948+) | 1h dev + 1 slot | +0.004 to +0.008 if it lands at claim | New higher floor if it works. Risk: claim may not reproduce. |
| 3 | **Multi-clone ensemble** (mtoshidesu + nina + ulyanov) | 1 day; 3 base slots + 1 ensemble slot = 4 slots | +0.005 to +0.010 | Three independent pipelines decorrelate more than two. Highest ceiling, highest compute cost. |
| 4 | **Investigate the 0.003 gap to mtoshidesu's claimed 0.947** | 30 min, **NO SLOT** | up to +0.003 | Free. Check kernel log / dataset versions / kernel resource caps. Probably LB noise but cheap to rule out. |
| 5 | **Replace clone's SED branch with our A2 SED** | 1-2h + 1 slot | unknown, possibly negative | Same calibration failure mode as today's Gate-3 (−0.003). NOT RECOMMENDED unless other paths exhaust. |

### Recommended sequence for tomorrow

1. **Step 1 (30 min, no slot):** Path 4 — investigate why our clone scored 0.944 vs mtoshidesu's claimed 0.947. Check `/tmp/kernel_v1_clone/birdclef-2026-0-947-lb-public-pipeline-reproduced.log` for kernel timing, resource pressure, dataset version IDs. Compare against mtoshidesu's reported kernel state.

2. **Step 2 (1.5h, 1 slot):** Path 1 — build local rank-mean ensemble of our v75 submission.csv + clone v1 submission.csv. Push as a new kernel `bc2026-clone-x-v75-ensemble`. Both source submissions are already saved in our Kaggle outputs. ETA: 7-min kernel + 45-min LB wait.

3. **Step 3 (3h, 1 slot, gated on #2):** Path 2 — clone nina2025 EoS-6 verbatim as a new kernel. Same recipe as the mtoshidesu clone. If LB > 0.944, becomes new floor.

4. **Step 4 (only if #2 + #3 plateau):** Path 3 — clone ulyanov's gate-fake008-head0015 (336 votes, claims 0.945), then ensemble the three clones. Per-row rank-blend. Most compute-intensive but highest ceiling.

5. **SKIP** Path 5 unless steps 1-4 exhaust without progress.

### Final-submission selection rule (Kaggle picks 2 for private LB)

- **Pick A:** Best public LB single entry (currently clone v1 0.944; updated to best ensemble if #2-4 exceed)
- **Pick B:** Architecture-diversity hedge (currently v75 0.933) — protects against public→private LB shifts

Don't burn all slots chasing public LB; keep the v75 hedge in reserve as final pick.

### What's gated / NOT to do tomorrow

- **Don't try another piecewise lever port onto OUR pipeline.** Today's 9 slots proved this fails. Memory: [[project_l4v2_lb_regression]], [[project_gate3_only_lb_regressed]], [[project_sonotype_shrinkage_neutral]].
- **Don't burn slots on lambda_prior / rank_aware_power scalar tweaks** ([[project_v77_l1_correction_weight_neutral]] failure mode).
- **Don't propose new training runs.** Bird-audio pretrain ([[project_xc_pretrain_lever_exhausted]]), cross-arch ([[project_cross_arch_ceiling]]), multi-seed bagging ([[project_multiseed_bagging_exhausted]]) all exhausted.
- **Don't trust local OOF substrate for tuning gates / blend constants.** Twice-confirmed: matched 708-row OOF can't predict LB for A1-vs-ProtoSSM blend changes ([[project_gate3_only_lb_regressed]]).

### The meta-question to sit with overnight

How much do you actually care about climbing past 0.944 vs banking +0.011 and stopping? If "I want 0.950+", do steps 1-3-4. If "+0.011 is enough", just confirm the clone as Pick A tomorrow morning and pick v75 as Pick B for hedging — that's a 5-minute morning.

### Slot accounting

- Today (2026-05-23): 5/5 used (v78, v79, v80, v81, clone v1)
- Tomorrow (2026-05-24): 5 fresh slots at 00:00 UTC (= 20:00 EDT today, so tomorrow morning will have full budget)
- Days to deadline: 10 (≤50 LB slots remaining)
- Steps 1-4 above use ≤4 slots if executed in order

### Reference: artifacts created today

- `data/matched_oof_v75.npz` (raw sigmoid proto)
- `data/matched_oof_v75_postproc.npz` (post-postproc proto)
- `data/per_class_thresholds.npz` (extracted from notebook)
- `data/a2_jit/a1_fold{0..4}.pt` (production A2 JIT'd ckpts, downloaded from Kaggle)
- `kaggle_datasets/a1-effb0-l4v2-ckpts/` (the L4-v2 JIT'd ckpts — keep for forensic)
- `kaggle.com/datasets/stevewatson999/birdclef-2026-a1-effb0-l4v2-ckpts` (uploaded)
- `kaggle.com/code/stevewatson999/bc2026-mtoshidesu-clone-test` v1 (LB 0.944 — the new floor)
- `src/postproc_gate_sweep.py`, `src/build_matched_oof.py`, `src/build_postproc_matched_oof.py`
- `jupyter/mtoshidesu-clone-test/` (kernel staging for the clone)

### Memory entries added today (for cross-reference)

- `project_l4v2_lb_regression` — L4-v2 +0.091 OOF → −0.005 LB; ckpt-trained-on-ProtoSSM-pseudo introduces inter-branch correlation collapse
- `project_sonotype_shrinkage_neutral` — v80 added sonotype + non-Aves shrinkage, LB Δ=0
- `project_gate3_only_lb_regressed` — v81 +0.005 OOF → −0.003 LB; matched OOF substrate twice-confirmed unreliable for A1-vs-ProtoSSM blend tuning
- `project_mtoshidesu_clone_reproduced_944` — clone gave +0.011 in one slot; meta-lesson on testing the right UNIT
