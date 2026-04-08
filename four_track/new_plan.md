# BirdCLEF+ 2026 — Plan to reach LB 0.942

**Author**: 2026-04-06
**Current LB**: 0.927 (`protossm-yuriy929` v16, ProtoSSM + Perch v2 + MLP probes)
**Target LB**: **0.942** (+0.015)
**Status**: Aggressive — see §1 for honest feasibility framing.

---

## 1. Reality assessment — what 0.942 means

| | LB |
|---|---|
| Our best | **0.927** |
| Top public notebook (Yuriy 0.929) | **0.929** |
| Top public ceiling we can adapt to | ~0.929 |
| **Our target** | **0.942** |
| Top **private** LB (yuanzhe zhou, #1) | **0.9334** |

**0.942 is +0.013 above the public ceiling and +0.0086 above the published #1 LB.** Hitting it means producing a solution stronger than every known submission in the competition. This is *winning territory*. No amount of post-processing tuning, hyperparameter sweeps, or notebook forking will get there — every such lever has been tried (see `plan.md` LB history rows #31–#36).

The only path with a realistic chance of +0.013 is **adding a model branch that is (a) comparably strong to ProtoSSM and (b) makes uncorrelated errors**. Output diversity from independent feature pipelines is the historical winner of these competitions.

Hard constraints we keep hitting in `plan.md` history:
- "Weak + strong" ensembles regress (#28C 0.901, #29 0.910). The new branch **must score ≥0.92 standalone** to help.
- Locally pre-trained Perch students collapse on Kaggle (#32 0.922, #34A 0.912). **Local Perch embeddings ≠ Kaggle Perch embeddings.** Anything that consumes Perch features must be trained on Kaggle.
- Perch logits as pseudo-labels are too noisy (#34E −0.008).
- Single-architecture post-proc is fully exhausted (#35A–#35H).

These constraints rule out almost everything cheap. What remains is one of:
- **(A)** Train a strong **non-Perch** model branch (raw audio CNN / BirdNET / log-mel SED) that doesn't share Perch's failure modes.
- **(B)** Train a heavier **second Perch-consumer** *on Kaggle* (Xformer / Perceiver / cross-modal head) and ensemble with ProtoSSM. Risk: high feature correlation → less diversity than (A).
- **(C)** Use ProtoSSM-as-teacher to pseudo-label `train_audio` (~46K focal clips), retrain a stronger student.

Realistic intermediate goals (not 0.942 in one shot):
- 0.927 → **0.930** via residual SSM unlock + correction_weight fix (already pushed in v16, expected this week)
- 0.930 → **0.933** via track A or B (≥1 month of work)
- 0.933 → **0.937+** via stacking + recalibration (decision pending track A/B results)
- 0.937 → **0.942** speculative — would require winning the competition

**This plan does not promise 0.942.** It lays out the only credible attempts and the gates that decide whether each track is alive.

---

## 2. Tracks

Each track is independent. They can run in parallel since they don't share Kaggle daily-submission slots until the final ensemble step.

### Track A — Independent SED branch on raw audio (highest expected lift)

**Hypothesis**: A CNN trained on raw audio / log-mel features makes errors uncorrelated with Perch-derived models. Even at standalone 0.90 LB it can lift a 0.929 Perch model into 0.935+ via rank-averaging.

**Why this is different from our 0.769 SED dead-end** (Mar 16–29 history):
- In Mar we hit 0.769 because our SED was the *primary* predictor on noisy soundscapes with too few labeled examples and a frozen pre-Perch backbone.
- In April Yuriy/Koushik public solutions (~0.929) use SED as a *secondary* branch. We don't need 0.92 SED *alone* — we need a SED that recovers signal Perch misses (insect sonotypes, the 31 missing species, etc.).
- The 31 species not in Perch's vocabulary are a guaranteed signal source: ProtoSSM/MLP probes can only learn them via embedding shape, but a raw-audio model can directly classify them.

**Concrete subtracks**:

| ID | Description | Risk | Expected lift |
|----|-------------|------|---------------|
| A1 | **PCEN+ASL+Freq-MixStyle EffNet-B0** trained on `train_audio` only (focal clips), 5-fold, 25 epochs. Inference on Kaggle as ONNX (`onnxruntime` *is* available via the `tf-wheels` style bundled wheel — verify). Standalone target: ≥0.85 LB. | MED | +0.005 to +0.010 if standalone ≥0.88 |
| A2 | **Train A1 with `train_soundscapes` pseudo-labels** from ProtoSSM ensemble (not Perch logits — those failed in #34E). Use only windows where ProtoSSM confidence > 0.7 *and* prior agrees. | MED | +0.002 to +0.004 |
| A3 | **Add a second backbone** (EffNetV2-S or NFNet-L0) for intra-SED diversity. Soup the checkpoints. | LOW | +0.001 to +0.003 |

**Gates**:
- A1 standalone LB **must be ≥0.85** before spending submission slots on the ensemble. If <0.85, kill A1 — the 0.769 lesson says weak SED dilutes strong Perch.
- A1 + ProtoSSM ensemble LB **must beat 0.929** at first attempt or kill the track.

**Estimated work**: A1 ~3–5 days (training + ONNX export + Kaggle integration). A2 ~2 days. A3 ~3 days.

#### A1 training results (2026-04-07)

5-fold × 25-epoch run complete (7h 45m total). Per-fold best val ROC-AUC on
the 1478-segment soundscape val (75/234 species present):

| Fold | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| val_roc_auc | 0.7414 | 0.7232 | 0.6970 | **0.6636** | 0.7250 | 0.7101 |

Ensemble analysis on the same val set (`four_track/src/eval_a1_ensemble.py`):

| Combiner | val AUC | lift vs per-fold mean |
|---|---|---|
| Mean-of-sigmoids, all 5 folds | 0.7017 | **−0.0084** |
| Rank-avg, all 5 folds | 0.7312 | +0.0211 |
| Mean-of-sigmoids, drop fold 3 | 0.7294 | +0.0193 |
| **Rank-avg, drop fold 3 (folds {0,1,2,4})** | **0.7431** | **+0.0330** |

**Key findings:**
1. **Fold 3 is miscalibrated, not just weak.** Mean-of-sigmoids across all 5
   folds gives *negative* bagging lift because fold 3's sigmoid scores live on
   a different scale and pull the average toward the wrong side. Drop-one
   analysis shows fold 3 is the only fold whose removal improves the
   ensemble. Every other fold contributes positively.
2. **Rank-averaging rescues A1.** Under rank-avg the folds DO add information
   to each other (+0.033 lift). This is the combiner to use for both
   intra-A1 bagging and cross-model fusion with ProtoSSM.
3. **Absolute AUC = 0.7431 is below the 0.85 standalone-LB gate,** but the
   val set only covers 75/234 species so absolute AUC is not directly
   comparable to LB scale. The binding test remains LB.
4. **A local cross-model OOF gate is impractical.** ProtoSSM's existing
   `oof_predictions.npz` covers the 708-file train_audio holdout, not the
   soundscape val; re-running ProtoSSM on the soundscape val requires
   re-implementing the entire Perch → embeddings → SSM → residual → TTA
   pipeline outside the notebook and risks drift vs the LB scorer. The plan
   already allots 2 LB attempts to A1 — spend the first attempt instead.

#### A1 → LB integration strategy

Track A1 will use **rank-space fusion at the very end of cell 36**, after the
residual SSM correction but before cell 37's per-class thresholds:

1. **Combiner**: rank-avg of A1 folds {0, 1, 2, 4} (drop fold 3). No
   score-space mixing — A1's sigmoids are demonstrably miscalibrated relative
   to ProtoSSM.
2. **Fusion point**: *after* the residual SSM, not before. The residual SSM
   was trained on ProtoSSM's score distribution; feeding it rank-transformed
   or A1-perturbed scores corrupts an already-tuned model.
3. **Fusion formula**:
   ```
   proto_rank = per-class CDF rank of final_test_scores (post-residual)
   a1_rank    = per-class rank-avg of A1 fold sigmoids
   fused_rank = (1 - w) * proto_rank + w * a1_rank
   final_test_scores = per-class CDF inverse mapping of fused_rank
   ```
   The round-trip through rank space and back preserves ProtoSSM's score
   distribution (and therefore cell 37's per-class threshold semantics), so
   the A1 experiment is isolated from threshold drift.
4. **Weight**: `A1_WEIGHT = 0.10` for the first LB attempt. 0.05 is too small
   to distinguish from noise at 0.927 scale; 0.10 is large enough for a
   ±0.003–0.005 swing either way but small enough that a total A1 collapse
   cannot drag LB below ~0.922.
5. **Checkpoint format**: raw `.pt` + torch, *not* ONNX for the first
   attempt. Torch is already imported in the notebook (cell 22 defines
   `SelectiveSSM(nn.Module)`, cell 36 uses `torch.no_grad()`). ONNX would
   save ~10 min of inference time at the cost of ~1h engineering +
   re-validation; save that for attempt 2 if attempt 1 justifies the effort.
6. **Inference compute budget**: ~20–30 min added wall time on CPU.
   EffNet-B0 forward over ~700 hidden test files × 12 windows × 4 folds ≈
   33,600 forwards at ~25 ms/sample batched = ~14 min of pure compute, +5
   min mel/PCEN preprocessing, +overhead. Fits the 9h Kaggle CPU budget.

**Gate update** (supersedes the original A1 gates where they conflict):
- First LB attempt uses w=0.10 rank fusion. If LB ≥ 0.929, A1 is alive and
  attempt 2 sweeps w ∈ {0.15, 0.20}. If LB drops by >0.003 from 0.927, A1 is
  dead — move to B1/C2 without using the second slot. If LB moves by
  ≤0.003 (inconclusive), attempt 2 uses w=0.15 to try to resolve the sign.
- The original "standalone LB ≥0.85" gate is retired — we never intended to
  use A1 standalone, and running standalone on LB would waste a slot.

#### A1 LB results

| Attempt | Notebook version | Config | LB | Δ vs prior best | Verdict |
|---|---|---|---|---|---|
| 1 (errored) | v17 | rank fusion w=0.10, mel pre-compute | OOM | — | Bug — pre-computing all mels held ~12 GB. Did not consume an A1 attempt. |
| 1 | v18 | rank fusion w=0.10, streaming mels | 0.929 | +0.002 | A1 alive — passes the ≥0.929 gate. |
| 2 | v19 | rank fusion w=0.15, streaming mels | 0.930 | +0.001 | Higher weight helps. |
| **3** | **v20** | **rank fusion w=0.20, streaming mels** | **0.932** | **+0.002** | **New best LB.** |
| 4 | v21 | rank fusion w=0.25, streaming mels | 0.932 | 0.000 | Flat vs w=0.20 — curve has plateaued. Freeze A1 at w=0.20. |

**Track A1 frozen at w=0.20 (LB 0.932).** Attempt 4 at w=0.25 stayed flat
vs attempt 3, satisfying the "regresses or stays flat" stop condition.
A1 is locked in at w=0.20 and the next action is **Track B1** (PerceiverIO
Perch consumer trained alongside ProtoSSM).

### Track B — Second Perch-consumer on Kaggle (medium lift)

**Hypothesis**: Different architectures on Perch features make different errors. A Perceiver/Xformer head trained on Kaggle (so no embedding domain shift) can ensemble with ProtoSSM for +0.003–0.005.

**Why this might work where #34A (chaneyma Xformer 0.912) failed**: chaneyma was pre-trained locally → embedding mismatch. If we *train on Kaggle* like ProtoSSM does, the features are consistent.

**Concrete subtracks**:

| ID | Description | Risk | Expected lift |
|----|-------------|------|---------------|
| B1 | **PerceiverIO over 12 windows** trained on Kaggle alongside ProtoSSM in the same notebook. d_latent=256, n_latents=16, 2 cross-attn + 4 self-attn layers, 30 epochs. Cost: ~10–15 min on top of ProtoSSM's 25–30 min. | MED | +0.002 to +0.004 |
| B2 | **GRU dual head** like #34C but with 2-fold ProtoSSM (we proved 2-fold ProtoSSM = 0.922 in #34C2; pair with GRU at full strength). Risk: same time bottleneck. | HIGH | +0.001 to +0.003 |

**Gates**:
- B1 must add ≥+0.002 to ProtoSSM OOF AUC before any LB submission. If <+0.002 OOF, kill — sub-noise.
- B1 wall time must stay <10 min added (else displaces ResidualSSM via the 35-min gate).

**Estimated work**: B1 ~2–3 days (architecture + training loop). B2 already prototyped but ruled out by 2-fold OOF weakness — only revisit if Track A succeeds and we have headroom.

#### B1 — concrete design (decided 2026-04-08, post-A1-freeze)

A1 is frozen at w=0.20 / LB 0.932; Track B1 is now active. Implementation
notes that pin down the design so the next session can resume cleanly:

- **Input contract** — same as ProtoSSM: per-file `(emb [N,T=12,1536], logits [N,T,234], site_ids, hours)`. B1 reads Perch logits as a side-channel via a simple front-end projection + concat with `emb`, **not** via a learned per-class fusion alpha (that's ProtoSSM's job). Same inputs, different *use* → architectural diversity at the model body without breaking the data plumbing.
- **Architecture** — `PerceiverIOHead` in `four_track/src/b1_perceiver.py`:
  - Input encoder: `Linear(1536+proj(234)+site_emb+hour_emb → 256)` per window.
  - Latent bank: 16 learned latents, `d_latent=256`.
  - 2 cross-attn (latents ← windows) + 4 self-attn (latents ↔ latents) blocks, 8 heads each, dropout 0.3.
  - Output decoder: 12 query tokens (one per window) cross-attend back into the latents → `Linear(256 → 234)` → `(B, T=12, 234)` logits.
  - **No taxonomy/family aux head.** ProtoSSM has one; dropping it in B1 increases architectural diversity (which is the whole reason B1 exists).
- **Training** — mirrors `train_proto_ssm_single` so the loss landscape is comparable: focal BCE w/ `pos_weight` (capped), MSE distill against raw Perch logits, file-level Mixup after epoch 5, AdamW + OneCycleLR (`pct_start=0.1`, cosine), grad clip 1.0, SWA from `swa_start_frac × n_epochs`, early stop on `val_loss`. Config lives in new `CFG["b1_perceiver"]` / `CFG["b1_perceiver_train"]` blocks (added to the V18 CFG cell).
- **OOF protocol** — `run_b1_perceiver_oof` reuses ProtoSSM's exact `file_groups` and `GroupKFold(n_splits=…)`, so per-fold splits match ProtoSSM 1:1 and the proto/B1 OOF correlation check is honest.
- **Notebook integration** — three new cells, all marker-anchored (no hard-coded indices), inserted by `four_track/src/inject_b1_cell.py` which mirrors `inject_a1_cell.py`:
  - **Cell 24b** — `# Cell 24b — Track B1 PerceiverIO training (def + OOF)` — function defs, inserted after cell 24 (ProtoSSM training defs).
  - **Cell 31b** — `# Cell 31b — Track B1 PerceiverIO instantiate + OOF + retrain on full` — runs OOF, logs `LOGS["oof_auc_b1"]` and the proto/B1 OOF correlation, then retrains on all soundscapes. Inserted after cell 31.
  - **Cell 36b** — `# Cell 36b — Track B1 inference + rank fusion` — runs `b1_model` on `(emb_test_files, logits_test_files, …)`, rank-fuses into `final_test_scores` via the same per-class CDF round-trip pattern as cell 37 (preserves ProtoSSM's marginals so cell 38's per-class thresholds keep their semantics). Inserted **before** cell 37 (the A1 cell), so the order is `ProtoSSM → B1 fusion → A1 fusion → postproc`.
- **Gates** (must all hold before any LB submission):
  - `oof_auc_proto+b1 ≥ oof_auc_proto + 0.002` after sweeping `B1_WEIGHT ∈ {0.10, 0.15, 0.20, 0.25}` on OOF.
  - `corrcoef(oof_proto_flat.flatten(), oof_b1_flat.flatten()) < 0.97` (else B1 is seeing the same signal — kill).
  - Notebook wall time stays under the 35-min ResidualSSM gate (B1 budget ≤10 min added).
- **Kill criterion** — if either OOF gate fails, freeze B1, do **not** burn an LB slot, and move to **Track C1** (Perch v2 embedding extraction for `train_audio` pseudo-labels).

#### B1 OOF protocol: structurally broken on this dataset (2026-04-08)

First train-mode dry-run (notebook v22) revealed the OOF lift gate is
**uninformative** and cannot be used as a go/no-go for B1 (or any branch):

- Cell 7 filters to files with all 12 windows labeled → only **59 files**
  (not the ~720 we'd assumed). 5 unique site groups → wildly imbalanced
  GroupKFold splits (39/5/5/5/5).
- ProtoSSM OOF AUC = **0.6468**, vs its known LB of **0.932**. The OOF
  number is disconnected from LB by ~0.28 — i.e. the protocol cannot rank
  branches at all on a 234-class problem with val folds of size 5.
- B1 standalone OOF AUC = 0.3878 (worse than chance, expected at this fold
  size). Diversity gate **passed** (corr=0.7115, well below 0.97).
- The lift gate naturally drives `B1_WEIGHT_FROZEN → 0.00` because any
  random perturbation looks bad on these tiny folds. The same gate would
  reject ProtoSSM itself.

**Decision**: bypass the OOF lift gate for B1 in submit mode and burn one
LB slot at a small `B1_WEIGHT = 0.10` (mirrors how A1's sweep started).
The diversity gate is still meaningful and B1 passes it. Set in
`b1_perceiver.py` as `CFG.setdefault("b1_frozen_weight_submit", 0.10)`.

**Carryover for Track C**: C2's "OOF AUC must improve vs C0" gate is
likewise unreliable on this dataset and needs reformulating before C2
runs. Reformulation candidates: (a) larger bootstrap on per-class AUCs,
(b) accept any non-regressing OOF + small LB probe, (c) drop OOF entirely
for C and budget LB slots instead.

#### B1 LB results

| Attempt | Notebook ver | B1_WEIGHT | LB    | Note                                      |
|---------|--------------|-----------|-------|-------------------------------------------|
| 1       | v23          | 0.10      | TBD   | First B1 LB probe; OOF gate bypassed      |

### Track C — ProtoSSM-as-teacher pseudo-labels on train_audio (medium-low lift)

**Hypothesis**: `train_audio` has ~46K focal clips. We currently use exactly **0** of them for ProtoSSM training (ProtoSSM only sees ~720 fully-labeled `train_soundscapes` files). Pseudo-labeling these with ProtoSSM and retraining gives the model 60× more data.

**Why this might work where #34E (Perch logits as pseudo-labels) failed**: Perch logits are too noisy. ProtoSSM predictions are calibrated, OOF-valid, and respect the soundscape distribution. They won't have the "always-on" failure mode of Perch on noisy audio.

**Concrete subtracks**:

| ID | Description | Risk | Expected lift |
|----|-------------|------|---------------|
| C1 | **Extract Perch v2 embeddings** for all 46K train_audio clips locally. Cache to a Kaggle dataset. | LOW | enables C2 |
| C2 | **Pseudo-label C1 embeddings** with current ProtoSSM. Filter to (max_conf > 0.6) ∧ (primary_label is in top-3). Retrain ProtoSSM on union of trusted soundscapes + filtered focal clips. | MED | +0.002 to +0.005 |
| C3 | **Iterate**: use the C2 model to re-pseudo-label, retrain again. 1–2 iterations max (diminishing returns). | LOW | +0.001 to +0.002 |

**Gates**:
- C2 OOF AUC on the original 720-file holdout **must improve** vs the C0 ProtoSSM baseline. If equal/worse, kill — pseudo-labels are corrupting the signal.
- After C1 cache exists, C2 is just retraining ProtoSSM with bigger input — minimal Kaggle integration risk.

**Estimated work**: C1 ~1 day (Perch embedding extraction is well-understood from `perch_v2/`). C2 ~2 days. C3 ~1 day.

### Track D — Recalibration & stacking (low lift, do last)

Apply *after* tracks A/B/C have produced their model branches. None of this works on a single-architecture pipeline — we already exhausted that surface.

| ID | Description | Risk | Expected lift |
|----|-------------|------|---------------|
| D1 | **Per-class isotonic calibration** of each branch on OOF, then rank-average. Done correctly (no train-leak), unlike #25B which overfit. | LOW | +0.001 to +0.002 |
| D2 | **Stacking meta-learner** (LightGBM) over branch predictions + metadata features (site, hour, file-level confidence). | MED | +0.002 to +0.004 |
| D3 | **Per-taxon ensemble weights** instead of one global blend weight. Taxon with strongest signal source dominates. | LOW | +0.001 to +0.003 |

**Gates**:
- D1/D2/D3 must each show OOF improvement before LB submission (we burned slots on #25B/#34E learning this).

---

## 3. Sequencing

Run tracks in parallel where possible. Hardware constraint: only one GB10 box, so local training serializes.

```
Week 1 (Apr 6–12)
  ├─ A1: Train PCEN+ASL EffB0 on train_audio (5-fold, 25ep)            [local, ~24h]
  ├─ C1: Extract Perch v2 embeddings for train_audio (~46K clips)      [local, ~12h]
  └─ Submit current v16 daily, monitor LB

Week 2 (Apr 13–19)
  ├─ A1: Export to ONNX, integrate into notebook, LB gate              [Kaggle slot]
  ├─ B1: Prototype PerceiverIO head locally on cached embeddings       [local, ~8h]
  └─ C2: Pseudo-label train_audio with ProtoSSM, retrain               [local, ~24h]

Week 3 (Apr 20–26)
  ├─ A1+ProtoSSM ensemble LB submission                                 [Kaggle slot]
  ├─ A2: Self-train A1 with ProtoSSM pseudo-labels                     [local]
  └─ B1: Integrate into notebook if OOF gate passes                    [Kaggle slot]

Week 4 (Apr 27 – May 3)
  ├─ A3: Second backbone (EffNetV2-S or NFNet-L0)                      [local]
  ├─ D1: Per-class isotonic calibration on combined ensemble           [local]
  └─ Final ensemble LB submission                                      [Kaggle slot]

Week 5+ (May 4 onward)
  ├─ D2/D3: Stacking + per-taxon weights if D1 lifts above 0.933       [local + LB]
  └─ Final tuning toward May 27 entry deadline
```

**Daily Kaggle submission slots**: 5/day. We have ~50 days to entry deadline (May 27) → ~250 slots. We will not run out — the constraint is *informativeness per slot*, not slot count. **Only submit when local OOF predicts an improvement.**

---

## 4. Decision gates (kill criteria)

A track is killed if any of the following triggers:

| Track | Kill criterion |
|-------|----------------|
| A1 | Standalone LB <0.85 after 2 LB attempts |
| A1 | A1 + ProtoSSM ensemble LB <0.928 (i.e. doesn't beat 0.927 baseline) |
| A2 | Pseudo-label retrain doesn't improve A1 OOF |
| A3 | Second backbone doesn't add to A1+A2 OOF |
| B1 | Adds >10 min wall time **OR** OOF improvement <+0.001 |
| B2 | Already abandoned (#34C2). Do not revisit unless headroom >15 min in submit budget. |
| C1 | Perch embedding extraction doesn't complete in 24h locally |
| C2 | C2 OOF on original holdout ≤ baseline ProtoSSM OOF |
| C3 | C3 OOF ≤ C2 OOF + 0.0005 (diminishing returns floor) |
| D1 | Doesn't add to held-out fold AUC |
| D2 | Stacker overfits OOF (test-OOF gap > 0.005) |

When a track is killed, **don't replace it with similar work**. Move to the next track in priority order.

---

## 5. What's NOT in this plan and why

- **More Perch-only post-proc tuning**: exhausted in #35A–#35H. Would not surprise me if the entire knob has ±0.001 left. Stop touching it.
- **Pre-training Perch students locally**: dead path (#32, #34A). Local ≠ Kaggle.
- **Replacing ProtoSSM**: ProtoSSM is the strongest Perch consumer we have at 0.927. Ablating it in favor of a hypothetical better head is not supported by the OOF data.
- **Bigger TTA / more folds**: time-budget bound. Already tried (#31B v1/v2), all timed out.
- **Retraining the 0.769 SED pipeline as primary**: dead path. SED is only useful as a secondary ensemble member now.

---

## 6. Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `onnxruntime` not available on Kaggle no-internet env | MED | Verify in a no-op submission first. Brucewu1200 0.911 used a bundled wheel (#28A) — copy that approach. |
| A1 standalone LB <0.85 (insufficient quality) | MED-HIGH | Train two backbones in parallel. The 2026 expert-labeled soundscapes are a much stronger supervision signal than 2025 had — A1 should be stronger than the Mar 0.769 ceiling. |
| Local C1 embedding extraction is too slow / disk | LOW | Cached on GB10 with 128GB unified memory — comfortable. |
| New ensemble pushes wall time over 90 min | HIGH | A1 inference is the new bottleneck. Budget: A1 inference ≤15 min, ProtoSSM training ≤25 min, Perch test ≤15 min, ResidualSSM ≤6 min, misc ≤8 min = 69 min. Build with explicit per-cell wall-time guards. |
| Honest answer: 0.942 turns out to be unreachable | MED-HIGH | Acceptable end states: 0.933 (top-private parity) is a respectable result. 0.937 would be a public-notebook-tied competition winner. Adjust expectations as data comes in. |

---

## 7. First action

> **Status update 2026-04-07**: A1 5-fold training is **complete**. Mean per-fold val ROC-AUC = 0.7101, best 4-fold rank-avg ensemble (drop fold 3) = 0.7431. See §2 Track A "A1 training results (2026-04-07)" for the full analysis. **Next action is the first A1 LB attempt** via rank-space fusion at w=0.10 in the `protossm-postproc` notebook (see "A1 → LB integration strategy" in Track A). C1 below remains complete and is still ready to feed C2 if A1 is killed.

> **Status update 2026-04-06**: C1 is **already complete**. `perch_v2/src/extract_embeddings.py` was previously run with `--per-window-audio`, producing both averaged and per-window Perch v2 embeddings for all 35,549 `train_audio` clips:
>
> | Asset | Path | Count |
> |---|---|---|
> | Averaged embeddings | `perch_v2/data/processed/perch_embeddings/train_audio/` | 35,549 |
> | Per-window embeddings | `perch_v2/data/processed/perch_embeddings/train_audio_pw/` | 256,490 (~2.0 GB) |
>
> Track C2 is therefore unblocked immediately — no embedding extraction step required.

The first action is now **A1** (highest expected lift): build a PCEN+ASL+Freq-MixStyle EffNet-B0 SED training script under `four_track/src/`, smoke-test it on one batch, then launch the 5-fold × 25-epoch run on the GB10.

Kickoff command (after smoke test passes):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track
nohup bash scripts/train_a1_5fold.sh > log/train_a1_5fold_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

While A1 trains, **Track C2** can run as a CPU-side data-prep job in parallel (it doesn't contend for the GPU once the pseudo-label dump is done).
