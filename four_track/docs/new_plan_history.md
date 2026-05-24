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

| Attempt | Notebook ver | B1_WEIGHT | LB    | Note                                       |
|---------|--------------|-----------|-------|--------------------------------------------|
| 1       | v23          | 0.10      | 0.933 | +0.001 vs A1-only 0.932                    |
| 2       | v24          | 0.15      | 0.927 | −0.006 regression → freeze at w=0.10       |

**Frozen weight: `B1_WEIGHT = 0.10` / LB 0.933.** Per the A1 stop rule
(freeze on first regression or plateau), B1 is locked in. The curve peaks
sharply at w=0.10; the diversity gate was honest but the lift surface is
narrow. `b1_frozen_weight_submit` reverted to 0.10 in
`four_track/src/b1_perceiver.py`.

**Next action**: Track **C1 is complete (2026-04-09)** — 35,549 / 35,549
train_audio clips extracted to
`four_track/data/processed/perch_train_audio_c2/<species>/<stem>.npz`
via `four_track/src/extract_train_audio_c2.py`, which is a faithful port
of the postproc notebook's ONNX Perch + `MAPPED_POS`/`MAPPED_BC_INDICES`
direct-indexing + genus-proxy logic (cells 2, 3, 5). Per-clip format is
`{emb: (T, 1536), scores: (T, 234)}` — exactly what ProtoSSM consumes.
253,101 total 5s windows, mean 7.12 windows/clip, 1.8 GB on disk, 206 /
234 competition classes have at least one training clip (28 classes have
no `train_audio` entries at all — mostly unmapped sonotypes). All clips
capped at the first 600 s (10 min) to bound per-worker RSS — the handful
of multi-hour field recordings would otherwise blow up memory.

**Next action is Track C2** — ProtoSSM-as-teacher pseudo-labeling: run
the current ProtoSSM (retrained on all 720 soundscapes) on the C1 cache,
filter to `(max_conf > 0.6) ∧ (primary_label in top-3)`, then retrain
ProtoSSM on the union of trusted soundscapes + filtered focal clips.
**C2 gate reformulation**: the plan's original "OOF AUC must improve
vs C0" gate is structurally broken (see "B1 OOF protocol" note above —
only 59 fully-labeled files, 5/5/5/5/39 GroupKFold). Replacement gate
candidates: (a) hold out 5–10 soundscape *files* as a fixed eval set
and require strict improvement on that, (b) skip OOF entirely and burn
one LB slot at small mix weights like B1 did, (c) filter aggressively
enough that a retrain on just the focal clips still beats ProtoSSM
standalone on the 59-file OOF. Default: **(b)** — matches how B1 landed,
and the OOF on 59 files is uninformative anyway. Target to beat: **0.946**.

#### C2 implementation: local-port path (2026-04-09) — ❌ KILLED 2026-04-09 afternoon

> ❌ **STRUCTURAL KILL — DO NOT REVIVE THIS PATH AS-IS.**
> The local-port C2 was built end-to-end (teacher, pseudo-labels, student, sweep) but is structurally incompatible with the constraint at line 26 of this plan ("Anything that consumes Perch features must be trained on Kaggle"). The three justifications below are real benefits but do not override the embedding-mismatch failure mode that killed #32 and #34A. See the "CURRENT STATE (2026-04-09 afternoon)" section below for the full kill rationale, the local sweep ledger, and the next concrete sequencing item. If you ever consider redoing C2, the only viable path is **(B-redo)**: pre-extract focal-clip Perch features in a one-shot Kaggle kernel, then retrain the student in-notebook on Kaggle-consistent embeddings.

We are pursuing the **local-port path** for C2, not the alt-path of
burning a Kaggle kernel slot to save the teacher checkpoint. Rationale:

1. **Kernel wall-time ceiling** — retraining ProtoSSM on soundscapes +
   focal clips inside the notebook eats minutes that already contend
   with the 90-min submit budget. Training locally removes that
   contention entirely.
2. **Multi-seed ensembling** — local training lets us run N seeds of
   ProtoSSM and soup the checkpoints, something the kernel cannot do
   without burning N submit slots.
3. **Knob sweeping without LB cost** — filter threshold, mix weight of
   focal vs soundscape, epoch count, SWA fraction, and distill weight
   can all be swept locally; only the final checkpoint hits LB.

**Plan:**

1. **Port ProtoSSMv2 → `four_track/src/protossm_model.py`** — verbatim
   lift of notebook cell 22 (`SelectiveSSM`, `TemporalCrossAttention`,
   `ProtoSSMv2`, `init_prototypes_from_data`, `init_family_head`). Pure
   module, no notebook globals. Extracted staging file:
   `/tmp/nb_cells/c22_proto_model.py`.
2. **Port training loop → `four_track/src/protossm_train.py`** — lift
   of cell 24 (`train_proto_ssm_single`, `run_proto_ssm_oof`,
   `build_taxonomy_groups`, `build_site_mapping`, `reshape_to_files`,
   `get_file_metadata`, `mixup_files`, `focal_bce_with_logits`,
   `optimize_ensemble_weight`). Staging file:
   `/tmp/nb_cells/c24_proto_train_loop.py`.
3. **C2 dataset loader** — new `four_track/src/c2_dataset.py` that
   consumes `four_track/data/processed/perch_train_audio_c2/<species>/<stem>.npz`
   (C1 output), tiles/random-slices per-clip windows to T=12 to match
   ProtoSSM's fixed temporal dim, and mixes focal clips with
   soundscape files at a configurable ratio.
4. **C2 teacher scoring** — `four_track/src/c2_pseudo_label.py` loads
   the current best ProtoSSM checkpoint, scores every C1 clip, writes
   per-clip soft pseudo-labels (NOT hard one-hot — avoids the #34E
   self-training trap) plus per-clip max-confidence. Filter to
   `(max_conf > 0.6) ∧ (primary_label in top-3)`.
5. **C2 training driver** — `four_track/src/train_c2.py` with CLI
   (`--seed`, `--focal-weight`, `--filter-threshold`,
   `--out`). Trains ProtoSSM on soundscapes + filtered focal clips,
   writes checkpoint to `four_track/models/protossm_c2/seed<N>.pt`.
6. **Kaggle integration** — final checkpoints shipped as a Kaggle
   dataset (`four_track/kaggle_datasets/protossm-c2-ckpts/`), loaded
   in the notebook via a new cell that replaces the local retrain
   with `torch.load(...)`. No training in-notebook at all for C2.

**Gates**: still (b) from above — skip OOF, burn one LB slot at the
default focal-weight / filter-threshold combo.

#### ⚠️ CURRENT STATE (2026-04-09 afternoon) — C2 KILLED on structural Perch-embedding-mismatch constraint

**TL;DR**: The local-port C2 pipeline ran cleanly end-to-end, but the entire approach was structurally incompatible with a hard constraint already documented in `../plan.md` lines 1824–1838: **locally-trained Perch consumers collapse on Kaggle**, because local Perch v2 features ≠ Kaggle Perch v2 features. C2 trained on `extract_train_audio_c2.py` + `extract_soundscapes_c2.py` outputs (both local) is a member of exactly the category proven DOA by #32 (LB 0.922) and #34A (LB 0.912). The local val_auc collapse from focal clips (recorded below) is real but moot — even if val_auc had been 0.95, LB would still collapse on the embedding mismatch. **No C2 checkpoint will be packaged or submitted.**

**The contradiction the local-port path created (see "C2 implementation: local-port path" section above, lines 278–327)**: that section justifies the local-port path on three grounds — kernel wall-time pressure, multi-seed ensembling, and free knob sweeping. All three benefits are real. None of them address the embedding-mismatch constraint at line 26 of this same plan ("Locally pre-trained Perch students collapse on Kaggle … Anything that consumes Perch features must be trained on Kaggle"). Whoever approved the local-port path on 2026-04-09 (the timestamp on the section header) did not reconcile against line 26. The 4-stage pipeline ran successfully but was building a checkpoint that the plan's own constraints had pre-classified as DOA.

**Lesson for future sessions**: when deciding to do "local training of an X-consumer", grep `../plan.md` and the top of `new_plan.md` for any constraint of the form "X must be trained on Kaggle" *before* writing the implementation. The C2 mistake cost ~1 day of local compute and pipeline-design work; reading line 26 of this file would have caught it in 30 seconds.

**Next concrete action**: see "Next concrete sequencing item" at the bottom of this section.

---

**4-stage local pipeline ran cleanly** overnight (preserved here as evidence of *what was built*, even though none of it ships):

| Stage                              | Output                                                           | Result                                                    |
|------------------------------------|------------------------------------------------------------------|-----------------------------------------------------------|
| Extract soundscape Perch v2        | `data/processed/perch_train_soundscapes_c2/train_soundscapes/`   | 10,658 clips → 127,896 windows (55.8 min)                 |
| Train ProtoSSM teacher (80 ep)     | `models/protossm_teacher/teacher.pt`                             | best @ ep 37: val_loss=0.8094 val_auc=0.9115 (SWA over 5) |
| Pseudo-label C1 cache              | `data/processed/c2_pseudo_labels/pseudo_manifest.csv` + per-clip | total=35549, max_conf>0.6=35549, in_top3=33597, both=33597|
| Train C2 student (seed 0, fm=5)    | `models/protossm_c2/seed0.pt`                                    | best @ ep 48: val_loss=0.7943 val_auc=0.8127 (SWA over 16)|

**Followup C2 sweep** (fm=focal-mult, T=label temperature, hard=use ground-truth one-hot instead of teacher soft):

| Run               | focal-mult | label-T | label source           | best val_loss | best val_auc | checkpoint                                |
|-------------------|------------|---------|------------------------|---------------|--------------|-------------------------------------------|
| Teacher           | —          | —       | (real labels)          | 0.8094        | **0.9115**   | `models/protossm_teacher/teacher.pt`      |
| **fm=0 control**  | **0.0**    | —       | (real labels only)     | **0.8192**    | **0.9070**   | `models/protossm_c2/seed0_fm0.pt`         |
| fm=2 T=1          | 2.0        | 1.0     | teacher soft           | 0.8033        | 0.8059       | (overwritten)                             |
| fm=2 T=4          | 2.0        | 4.0     | teacher soft, T=4      | 0.8645        | 0.8217       | `models/protossm_c2/seed0_fm2_T4.pt`      |
| **fm=2 hard**     | **2.0**    | —       | **ground-truth one-hot** | **0.8684** | **0.8276**   | **`models/protossm_c2/seed0_fm2_hard.pt`**|
| fm=5 T=1 (orig)   | 5.0        | 1.0     | teacher soft           | 0.7943        | 0.8127       | `models/protossm_c2/seed0.pt`             |

**Local OOF verdict** (val=10 soundscape files): every variant that injects focal clips drops val_auc by ~8 points vs the no-focal control, **regardless of focal weight, soft-vs-hard labels, or temperature softening**. Hard ground-truth one-hot labels (the cleanest possible signal — the focal clip's known species) drop just as much as teacher soft. The fm=0 control reproduces teacher val_auc within rounding (0.9070 vs 0.9115), demonstrating the val=10 metric is not pure noise — it responds correctly to model quality when the test variable is removed.

**Local OOF evidence (sideshow — moot under the structural kill, recorded for completeness)**: every variant that injects focal clips drops val_auc ~8 points vs the no-focal control on the val=10 split, regardless of focal weight, soft-vs-hard labels, or temperature softening. The fm=0 control reproduces teacher val_auc, so the metric is internally consistent. This *was* the basis for an earlier "decision M: burn one LB slot per the plan's pre-committed protocol (b) and let the LB arbitrate" recommendation, before the embedding-mismatch constraint was rediscovered. Once we re-grepped `../plan.md` for #32/#34A and confirmed the constraint applies, the OOF evidence stopped mattering: there's no point packaging a checkpoint the plan's own documented constraints classify as DOA.

**Why the structural kill supersedes the OOF debate**: the worst case (M-route) was "burn one slot, learn from LB". But the plan-level evidence (`../plan.md` lines 1824–1838) shows the LB outcome is **pre-determined**: locally-trained Perch consumers regress on Kaggle by ~0.005–0.020 LB. The slot is not a free experiment; it's a guaranteed regression, and it would also use a daily slot on a day where the actual A1/B1 ensemble at LB 0.933 is the current best. Decision M is structurally invalid; there is no salvageable variant of "ship the local C2 checkpoint" that escapes the constraint.

**What was built but won't ship**:
- `models/protossm_teacher/teacher.pt` — locally trained ProtoSSM-like teacher. Cannot be loaded into the Kaggle notebook for the same embedding-mismatch reason. (Possible local-only use as a regression-test reference, not for LB.)
- `models/protossm_c2/seed0.pt` and `seed0_fm0.pt`, `seed0_fm2_T4.pt`, `seed0_fm2_hard.pt` — locally trained student variants. Same reason. None ship.
- `data/processed/c2_pseudo_labels/` — locally generated pseudo-labels using the local teacher on local Perch features. Internally consistent, externally meaningless to a Kaggle-trained model.
- `data/processed/perch_train_soundscapes_c2/` — local Perch v2 cache over `train_soundscapes` (10,658 clips). **Useful only as a local development reference**; cannot be used to train any Perch-consuming model that will run on Kaggle.

**What is preserved as potentially-useful infrastructure**:
- `data/processed/perch_train_audio_c2/` (35,549 focal clip Perch v2 caches, 1.8 GB) — same caveat: local-only, not Kaggle-compatible. Keep on disk as development reference. **If a future track (e.g. A2) requires a Kaggle-extracted focal-clip Perch cache, that work will need to be redone in a one-shot Kaggle kernel — the local cache will not help directly, only as a sanity-check reference.**
- `four_track/src/extract_train_audio_c2.py`, `extract_soundscapes_c2.py`, `c2_pseudo_label.py`, `c2_dataset.py`, `train_c2.py`, `protossm_model.py`, `protossm_train.py`, `train_protossm_teacher.py` — all reusable as local reference implementations. The model code (protossm_model/protossm_train) is a faithful port of notebook cells 22/24 and can be useful for offline experimentation that doesn't touch LB.

**Files that should NOT be deleted yet** (pending C2 retrospective + possible future Kaggle-port reuse):
- All of the above outputs and src files. The cost to keep them is small (~3.5 GB of caches); the cost to recreate them is ~12 hours of local compute. Defer cleanup until at least one other track has progressed and we've had time to write a C2 retrospective.

---

#### Next concrete sequencing item (post-C2-kill, 2026-04-09)

With A1 frozen at LB 0.932, B1 frozen at LB 0.933, C1 done, C2 dead, the live tracks per §3 sequencing are:

| Track | Status | Blocker |
|-------|--------|---------|
| A2 (self-train A1 with ProtoSSM pseudo-labels) | candidate | needs ProtoSSM pseudo-labels for `train_soundscapes`. Generating these locally hits the same embedding mismatch — must be a one-shot Kaggle kernel that runs ProtoSSM in-budget and dumps soundscape pseudo-labels as a Kaggle dataset. ~1 day infra work. |
| A3 (second SED backbone — EffNetV2-S or NFNet-L0) | candidate | none — fully Perch-independent, raw audio only. ~2 day local train + ONNX integration mirroring A1's pattern. |
| D1 (per-class isotonic calibration) | candidate | none — pure post-processing on the existing A1+B1+ProtoSSM stack. Fastest, smallest, lowest-risk. ~half day local. Plan estimates +0.001 to +0.002 LB. |
| D2 (LightGBM stacking meta-learner) | gated on D1 | wait for D1 result to know if calibration alone closes the gap. |

**Recommended next item: D1 (per-class isotonic calibration)**, because:
1. Pure local post-processing on outputs we already have — no new training, no Kaggle submission until we have a calibrated stack to test.
2. Fastest path to a measurable LB read on a non-blocked track.
3. Validates whether the +0.001 to +0.002 plan estimate holds before committing to the longer A2/A3 work.
4. If D1 pays off, D2/D3 are natural follow-ons in the same workspace.
5. If D1 yields nothing, that's still useful information — it means the calibration headroom is gone and we should push directly to A3 (new backbone) for the next +0.005.

The candidate for the actual next session: **read `four_track/src/eval_a1_ensemble.py` (the rank-fusion eval already in the repo) and the OOF dump locations for A1/B1/ProtoSSM, then write `four_track/src/d1_isotonic.py` that fits per-class isotonic regression on each branch's OOF and re-evaluates the rank-fused ensemble.** No Kaggle slot consumed until we have a positive local result.

**Important re-grep before starting D1**: confirm there's no analogous "must be calibrated on Kaggle" constraint we've forgotten. The C2 mistake came from not re-grepping; the lesson should not be wasted.

---

#### D1-a result (2026-04-09) — **FAIL, pivot to A3**

Scoped D1 down to **A1 only** (per-class isotonic on each fold's sigmoids, then rank-fuse across folds {0,1,2,4}). Protocol hardened vs #25B:
1. **File-level held-out split** of the 66 `train_soundscapes` files into disjoint FIT and EVAL halves (segment-level splits were leaking — the 22-segments/file structure put same-file segments on both sides and inflated Δ by ~10×).
2. **Min-positives gate** per class (N_MIN ∈ {5, 10, 20, 50}).
3. **Multi-seed** (5 seeds), report median + sign stability.

Results (`src/d1_isotonic.py`, honest file-mode):

| N_MIN | auc_base | auc_cal | Δ_median | sign stable | verdict |
|---|---|---|---|---|---|
| 5 | 0.7434 | 0.7475 | +0.0041 | ❌ (one seed −0.0003) | fail |
| 10 | 0.7434 | 0.7462 | +0.0027 | ❌ (one seed −0.0053) | fail |
| 20 | 0.7434 | 0.7399 | −0.0035 | ❌ | fail |
| 50 | 0.7434 | 0.7419 | −0.0010 | ❌ | fail |

Leaky segment-mode reference (for comparison, NOT a valid result): Δ ranged from +0.0034 (N_MIN=50) up to **+0.0476 (N_MIN=5)**, all sign-stable. That's the #25B fingerprint: isotonic was memorizing per-file primary-label tie structure.

**Decision**: D1-a fails the gate (`median Δ ≥ +0.001 AND sign stable`). Per-class isotonic on A1 alone is exhausted. Extending D1 to also calibrate B1 and ProtoSSM before rank fusion would face the same tie-collapse pathology, on an even thinner per-class positive budget — expected ROI is zero or negative. **D1 is deprioritized.**

**Next sequencing item: A3 (second SED backbone — EffNetV2-S or NFNet-L0)**, the only remaining candidate not blocked by an infra prereq (A2 needs a Kaggle kernel to dump soundscape pseudo-labels first) and not exhausted (D1 just was). ~2 days local train + ONNX integration mirroring A1. Plan estimate +0.005 LB.

---

#### A3-v1 result (2026-04-10) — **KILL, pivot to A3-v2**

Launched A3-v1 with `tf_efficientnetv2_s.in21k_ft_in1k` via `scripts/train_a3_5fold.sh`, reusing the A1 recipe **unchanged** except for `--backbone` and `--save-dir`:
- hybrid BCE+ASL loss, `--mixstyle-p 0.5`
- LR/WD/T_0 from `config.py` (LR=A1 default, T_0=5 cosine warm restarts)
- Batch 32, 2 workers, no pin_memory/persistent_workers (GB10 unified-memory recipe)
- 25 epochs, fold 0 first

Fold 0 trajectory (killed at epoch 19/25 to save ~23h of the remaining folds):

| Epoch | train_loss | val_roc_auc | note |
|-------|-----------|-------------|------|
| 1 | 0.0367 | 0.6143 | cold start |
| 2 | 0.0251 | 0.6740 | ★ |
| 5 | 0.0169 | 0.6855 | ★ (first T_0 restart) |
| 7 | 0.0188 | 0.7172 | ★ |
| 9 | 0.0152 | 0.7260 | ★ |
| **10** | **0.0141** | **0.7362** | **★ PEAK (second T_0 restart)** |
| 11–15 | 0.012–0.017 | 0.71–0.73 | plateau; restart at 15 peaks at 0.7256, *below* ep-10 |
| 16–19 | 0.012–0.015 | 0.70–0.71 | downtrend, train_loss still falling → overfitting |

**Verdict**: A3-v1 peaked at **val_roc_auc = 0.7362**, which is **0.034 below the A1 gate (≥0.77 on this same 1478-segment soundscape val set)**. Gap is too large to close in the 6 remaining epochs of fold 0, and each successive T_0 cosine restart was peaking *lower* than the previous one, not higher. Train loss vs val divergence confirms overfit.

Root cause (hypothesis): the A1 recipe (LR, WD, mixstyle_p=0.5) is tuned for a 5.3M-param EffNet-B0. EffNetV2-S is ~4× larger (21.5M params) and typically wants **lower LR and higher regularization**; the current config lets V2-S memorize `train_audio` faster than it generalizes to `train_soundscapes`.

Per-epoch time: **12m 10s** on GB10 (kaggle env, batch 32). Full 5-fold run would have been ~25h.

**Pivot to A3-v2**: same V2-S backbone, recipe retune. Knobs to try (in order of expected impact):
1. **Lower base LR** — halve it (V2-S canonical recipe uses ~half B0's LR at this batch size).
2. **Longer T_0** (e.g. 10 instead of 5) — each full cosine cycle gets more time to settle; the observed "later restarts peak lower" pattern suggests current cycles are too short for a larger model.
3. **Pure ASL loss** (drop hybrid BCE term) — ASL's hard-negative down-weighting matters more at larger model capacity.
4. **Lower mixstyle_p** (0.25 instead of 0.5) — V2-S's BatchNorm statistics are more sensitive; 0.5 may be over-regularizing the wrong layer.
5. **Higher weight decay** (2× current) — if lower LR alone isn't enough.

**A3-v2 gate (unchanged)**: fold-0 val_roc_auc ≥ 0.77 before committing to 5-fold. If fold-0 v2 still caps below 0.77 after a 25-epoch run, **abandon EffNetV2-S entirely** and switch to **ECA-NFNet-L0** (the BirdCLEF 2025 2nd-place pairing, held in reserve per `scripts/train_a3_5fold.sh` comments).

**Artifacts retained**:
- `log/train_a3_5fold_20260410_151538.log` — full fold-0 epoch-by-epoch trajectory
- `models/a3/a1_tf_efficientnetv2_s.in21k_ft_in1k_fold0_seed42_hybrid.pt` — the ep-10 best checkpoint (0.7362), kept as a reference point for comparing against A3-v2
- `models/a3/a1_tf_efficientnetv2_s.in21k_ft_in1k_fold0_seed42_hybrid_last.pt` — resume checkpoint at ep-19 (not useful for v2, delete at cleanup)

**Infra upgrade landed during this session**: `src/train_a1.py` now saves a per-epoch `_last.pt` full-state checkpoint (model + optimizer + scheduler + epoch + best_auc) and supports `--resume`, `--start-epoch`, and `--end-epoch` flags. This means A3-v2 (and any future multi-day local run) can survive a crash without losing all progress.

---

#### A3-v2 result (2026-04-10) — **KILL at ep 10/25, pivot to NFNet-L0 (A3-v3)**

Launched A3-v2 with `tf_efficientnetv2_s.in21k_ft_in1k` and the retuned recipe from the v1 kill entry above: **pure ASL loss, LR 2.5e-4 (halved), T_0 10 (doubled), mixstyle-p 0.25**. Also landed `--lr`, `--weight-decay`, and `--t0` CLI overrides on `train_a1.py` so recipe retuning doesn't require touching `config.py`.

Fold 0 trajectory (killed at epoch 10/25 when the kill gate `ep-10 < 0.70` hit):

| Epoch | A3-v1 val | A3-v2 val | v2 train_loss | note |
|---|---|---|---|---|
| 1 | 0.6143 | 0.6065 | 0.0087 | ★ |
| 2 | 0.6740 | 0.6262 | 0.0061 | ★ |
| 3 | 0.6506 | 0.6451 | 0.0053 | ★ |
| 4 | 0.6695 | 0.6360 | 0.0048 | |
| 5 | 0.6855 | 0.6607 | 0.0043 | ★ |
| 6 | 0.6737 | 0.6622 | 0.0040 | ★ |
| 7 | 0.7172 | 0.6825 | 0.0038 | ★ |
| 8 | 0.7061 | 0.6851 | 0.0035 | ★ |
| 9 | 0.7260 | 0.6849 | 0.0034 | |
| **10** | **0.7362** | **0.6870** | **0.0033** | **★ PEAK — kill gate hit** |

**Verdict**: A3-v2 peaked at **val_roc_auc = 0.6870 at ep 10**, which is:
- **0.049 below A3-v1's ep-10 peak** (0.7362) — the retune made things *worse*, not better.
- **0.083 below the A1 gate** (≥0.77). Gap is structural, not close.

**Key diagnostics from the v2 trajectory:**
1. **The cosine terminal boost didn't happen.** Ep 7→10 gains were +0.003, −0.0002, +0.0021 — a plateau at the bottom of the first cosine cycle, not the steep descent we needed. LR hit `LR_MIN` with the model just sitting there.
2. **Train-val divergence starting at ep 7.** Train loss fell 0.0038→0.0033 between ep 7 and ep 10 while val only moved from 0.6825→0.6870. Pure ASL did **not** prevent overfit the way I hypothesized — easy-negative gradient removal didn't help this dataset.
3. **v2 was worse than v1 at every single matching epoch.** The shrinking gap at ep 5–6 (−0.025 → −0.012) that I called "catching up" was just noise around a consistent −0.035 deficit. At ep 10 the gap blew back out to −0.049.

**What v1+v2 jointly prove:**
- **Pure ASL was a bad call on this dataset.** Contradicts my theoretical argument that zeroing out easy negatives would concentrate gradient signal on hard positives. Empirically, hybrid BCE was the better loss.
- **The recipe retune was too aggressive.** Four knobs changed at once (LR, T_0, loss, mixstyle_p) makes attribution impossible, but the net direction was wrong.
- **EffNetV2-S is probably the wrong backbone for this pipeline.** Two failures in opposite recipe directions (v1: B0 recipe as-is; v2: more regularization) suggests it's the architecture, not the hyperparameters. ~25 compute-hours burned; best V2-S number produced anywhere = 0.7362 (still 0.034 below gate). Further V2-S retunes are not worth their compute cost.

**Decision**: Abandon V2-S. Go directly to the plan-documented fallback: **ECA-NFNet-L0** (2025 2nd-place SED pairing, held in reserve per `scripts/train_a3_5fold.sh` comments).

**Artifacts cleaned**: `models/a3_v2/` directory deleted (both `_last.pt` and best checkpoint were at val=0.687, strictly worse than v1's 0.7362 — no reference value).

---

#### A3-v3 (2026-04-10, pending) — **ECA-NFNet-L0**

Planned recipe (one-fold-first, retune-before-5-fold gate still applies):

| Setting | Value | Reason |
|---|---|---|
| backbone | `eca_nfnet_l0.ra2_in1k` | 2025 2nd-place SED pairing. 21.8M params, similar scale to V2-S but NormFree + WS-Conv architecture. Verified pretrained-available and `features_only out_indices=(4,)` compatible via timm. |
| loss | `hybrid` (0.5·BCE + 0.5·ASL) | **Revert** to A1's original loss. A3-v2 proved pure ASL hurts. |
| lr | 2.5e-4 | Same as v2. Proven not-too-high for ~22M params; no reason to go lower on the first attempt. |
| weight_decay | 1e-4 (default) | NFNet canonical recipe uses very low WD (~2e-5) but that's for SGD+Nesterov. For AdamW, standard 1e-4 is a safer starting point. |
| t0 | 10 | Longer cycles, same reasoning as v2. |
| mixstyle_p | **0.0 (disabled)** | NFNet has no `.blocks` or `.stages` attribute on the `features_only` wrapper — the existing hook in `model_a1.py` would fall through to the stem Conv2d, which the code comment explicitly warns is "too aggressive." Rather than hack the hook target, disable MixStyle entirely. NFNet's WS-Conv + scaled activations provide intrinsic regularization. |
| batch_size | 32 | Unchanged — GB10 unified-memory constraint. |
| epochs | 25 | Unchanged. |
| folds | 0 only | Gate check before 5-fold. |
| save-dir | `models/a3_v3` | Segregate NFNet artifacts from V2-S failures. |

**Known risks specific to NFNet that are NOT being addressed this run:**
- NFNet canonical training uses **AGC** (Adaptive Gradient Clipping, coefficient ~0.01). `train_a1.py` uses `clip_grad_norm_(5.0)`, which is less aggressive. Fine-tuning tolerates this, but peak accuracy may be lower than the NFNet paper suggests. If v3 clears the gate, AGC is a v4 knob.
- NFNet's original recipe uses **SGD+Nesterov with stochastic depth**; we're using AdamW. This is normal for fine-tuning but is another deviation from canonical.

**A3-v3 gate (unchanged from v1/v2)**: fold-0 val_roc_auc ≥ 0.77 before committing to 5-fold.

**A3-v3 kill criterion**: ep-10 val < 0.70, OR obvious train-val divergence with train_loss falling faster than val improves. If v3 also fails, **abandon Track A3 entirely** — three failed attempts across two backbones is sufficient evidence that the A1 pipeline doesn't accept a second SED backbone with this amount of recipe investment, and further spending on A3 is strictly dominated by working on A2 (ProtoSSM-self-train, once the Kaggle-kernel-dump infra is built) or D2 (LightGBM stacker on the existing A1+B1+ProtoSSM stack).

---

#### A3-v3 result (2026-04-10) — **KILL at ep 10/25, Track A3 ABANDONED**

Launched A3-v3 with `eca_nfnet_l0.ra2_in1k`, hybrid loss (reverted from v2's pure-ASL mistake), LR 2.5e-4, T_0 10, mixstyle_p=0.0 (MixStyle hook disabled — NFNet has no `backbone.blocks[1]` on the `features_only` wrapper, and the hook's Conv2d fallback would target the stem which is "too aggressive").

Fold 0 trajectory through the first cosine cycle:

| Epoch | A3-v1 val | A3-v3 val | v3 train_loss | note |
|---|---|---|---|---|
| 1 | 0.6143 | 0.5958 | 0.0348 | ★ |
| 2 | 0.6740 | 0.6704 | 0.0238 | ★ |
| 3 | 0.6506 | 0.6707 | 0.0207 | ★ |
| 4 | 0.6695 | 0.7326 | 0.0190 | ★ |
| 5 | 0.6855 | 0.7202 | 0.0175 | |
| 6 | 0.6737 | 0.6740 | 0.0162 | mid-cycle dip |
| 7 | 0.7172 | **0.7458** | 0.0153 | **★ ALL-TIME A3 PEAK** |
| 8 | 0.7061 | 0.7427 | 0.0143 | |
| 9 | 0.7260 | 0.7402 | 0.0136 | |
| 10 | 0.7362 | 0.7428 | 0.0133 | first cycle bottom |

**Verdict**: A3-v3 peaked at **val_roc_auc = 0.7458 at ep 7** and then spent ep 8/9/10 in a 0.7402–0.7428 plateau. The cosine-cycle bottom at ep 10 (0.7428) is **0.003 below the ep-7 mid-cycle spike** — meaning the terminal LR decay did **not** find a better basin, so ep 7's 0.7458 was almost certainly a one-step lucky sample from a mid-cycle fluctuation, not sustained. The "real" v3 plateau is 0.743–0.746.

**Gate comparison:**
- **v3 best (0.7458)** vs **A1 gate (≥0.77)** → gap = **−0.024**
- **v3 cycle-bottom (0.7428)** vs gate → gap = **−0.027**
- **v3 ep-10 (0.7428)** vs **v1 ep-10 (0.7362)** → +0.0066 (a real but tiny architectural advantage over V2-S; not a path to closing the gate)

**Why this is a kill, not a "wait for second cycle":**
1. **v3's plateau is 15% of the full training range the model has covered.** The model's entire val span from ep 1 to ep 10 is ~0.16 (0.59 to 0.75). Closing the remaining 0.024 to the gate in the second cycle would require a sustained improvement of comparable magnitude to ~15% of everything the model has achieved so far — on top of already-converging train loss, in a second cycle, with the same recipe. There's no mechanism to expect that.
2. **The plan's escalation rule was explicit**: "If A3-v3 also fails, abandon Track A3 entirely." v3 is failing (ep 10 < gate − 0.02, plateau shape, no second-cycle-productive hypothesis).
3. **Sunk cost hygiene.** Across v1 + v2 + v3 we've burned ~**32 hours** of GB10 compute on Track A3. Every additional hour is strictly dominated by working on A2 (ProtoSSM-self-train, blocked on Kaggle-kernel infra) or D2 (LightGBM stacker, unblocked and pure local post-processing).

## 🔴 Track A3 decision: ABANDONED 2026-04-10

Across three attempts (v1 EffNetV2-S B0 recipe, v2 EffNetV2-S retuned, v3 ECA-NFNet-L0 hybrid), the best fold-0 val_roc_auc achieved was **0.7458** — still 0.024 below the A1 gate of 0.77. Two different backbones, three different recipes, same plateau.

**Joint lessons from v1 + v2 + v3** (all three variants tried):

| Variant | Backbone | Loss | LR | T_0 | mixstyle | Best fold-0 val | Notes |
|---|---|---|---|---|---|---|---|
| v1 | EffNetV2-S | hybrid | 5e-4 | 5 | 0.5 | 0.7362 @ep10 | Overfit starting ep 11; each cosine restart peaked lower |
| v2 | EffNetV2-S | pure ASL | 2.5e-4 | 10 | 0.25 | 0.6870 @ep10 | Worse than v1 at every matching epoch — pure ASL was wrong |
| v3 | ECA-NFNet-L0 | hybrid | 2.5e-4 | 10 | 0.0 | 0.7458 @ep7 | Best A3 attempt; still 0.024 below gate; plateau confirmed at cycle bottom |

1. **The A1 pipeline (PCEN + 5s clips + soundscape val + MixStyle-on-blocks[1]) has a ceiling around val_roc_auc 0.74–0.75 for non-B0 SED backbones we've tried.** This is a *pipeline* ceiling, not a backbone choice — the recipe + data + MixStyle hook target are all tuned to EffNet-B0 specifically.
2. **Further Track A3 spending is strictly dominated** by A2 (needs Kaggle-kernel infra first) or D2 (unblocked, cheapest, no new training).
3. **v3's checkpoint was tested (D2-γ, 2026-04-11) and has no salvage value** — blend sweep with A1 was monotonic, not bowl-shaped, and the single-fold A1-vs-NFNet delta (+0.0044) is inside noise. Both `models/a3/` (V2-S v1 ep-10 best) and `models/a3_v3/` (NFNet fold-0 best) are scheduled for deletion. Track A3 is fully closed including salvage.
4. **v2 checkpoints already deleted** (`models/a3_v2/` fully cleaned 2026-04-10 pre-v3 launch).

**Kill gate on further A3 work**: Do not launch any A3-v4 attempt (AGC, different loss, different backbone, reduced LR, etc.) without *new* evidence. The three-attempt ceiling is sufficient. If a future D2 result reveals that v3's NFNet is contributing meaningfully to ensemble diversity, at that point a full 5-fold v3 run could become re-justified — but only on that specific evidence, not on general "maybe a fourth attempt will work" optimism.

---

#### A2 — Soundscape domain adaptation (2026-04-16)

**Context:** v46 (D2-β S1 stacker, LB 0.775 catastrophic regression) and
v47 (§10 pre-trained ProtoSSM checkpoint, LB 0.929 neutral) both failed.
D2-β S1 is killed, §10 is reverted. The D track (post-processing) is
exhausted. Pivoting to A2 as the next viable path to improve the A1 SED
branch itself.

**Design (simplified from original A2 spec):** Train A1 EffNet-B0 on
`train_audio` focal clips **plus** `train_soundscapes` labeled segments.
The original A2 spec called for ProtoSSM pseudo-labels generated via a
Kaggle kernel, but we simplified to use ground truth labels from
`train_soundscapes_labels.csv` directly:

- **Why ground truth instead of pseudo-labels:** The main A2 hypothesis is
  *domain adaptation* — A1 trains on focal clips but is tested on
  soundscapes. Exposing A1 to soundscape audio during training is the
  primary lever. Ground truth labels are available for 66 files / 1478
  segments / 75 species. ProtoSSM pseudo-labels would add coverage for
  the other 159 species, but at the cost of Kaggle kernel infra work.
  Ground truth is a clean, zero-infra starting point.
- **No embedding-mismatch risk:** A1 operates on raw audio → mel
  spectrograms. It does not consume Perch features. The constraint that
  killed C2 does not apply.

**Implementation:**
- `four_track/src/dataset_soundscape.py` — `SoundscapeTrainDataset` that
  loads labeled soundscape segments, pads/crops to 20s, returns
  `(mel, multi_hot_labels, mask)` matching `BirdTrainDataset`'s interface.
- `four_track/src/train_a2.py` — wraps A1's training loop, uses
  `ConcatDataset([focal_ds, soundscape_ds])`. Soundscapes oversampled by
  `--soundscape-mult` (default 10) to balance with ~28k focal clips.
- Same model (EffNet-B0), same hyperparameters (ASL, LR=5e-4, T_0=5,
  MixStyle 0.5), 25 epochs per fold.
- Folds 0,1,2,4 (fold 3 excluded, same as A1).
- Checkpoints saved to `four_track/models/a2/`.

**Fold data composition (fold 0):**
- 28,606 focal clips (A1 baseline)
- 14,780 soundscape segments (1,478 base × 10 oversample)
- 43,386 total per epoch (~52% more than A1)

**Training launched 2026-04-16:** folds {0,1,2,4}, 25 epochs each, ASL
loss, soundscape-mult=10. Estimated runtime ~11-12h (50% longer than A1's
7h45m due to larger dataset). PID 290673.

**Validation caveat:** val set is the same train_soundscapes segments that
are now in the training set, so local val_roc_auc is contaminated. LB is
the only honest evaluation. This is consistent with how B1 was handled
(OOF protocol structurally broken, LB as arbiter).

**Decision gate:**
- Compare A2 fold-0 val_roc_auc trajectory against A1 fold-0 (0.7414
  best). If A2 fold-0 matches or exceeds A1 fold-0: good sign (even
  with contaminated val, if it's NOT higher it means training is broken).
- Upload A2 checkpoints to `stevewatson999/birdclef-2026-a1-effb0-ckpts`
  (replace A1 checkpoints) and LB probe at same rank-fusion weight
  (w=0.20).
- LB ≥ 0.934 → A2 helps, lock in.
- LB 0.931–0.933 → neutral, revert to A1 checkpoints.
- LB < 0.931 → regression, revert to A1 checkpoints.

#### A2 LB result (2026-04-16) — **KILL, A1 restored**

Notebook v49 with A2 JIT ensemble replacing A1 at the same `A1_WEIGHT=0.20`
rank fusion returned **LB 0.926**, a **−0.007 regression** vs the 0.933
A1+B1 baseline. Triggers the "LB < 0.931 → regression, revert" gate cleanly.

Kaggle dataset `stevewatson999/birdclef-2026-a1-effb0-ckpts` version
history:

| Version | Contents | LB |
|---|---|---|
| v1 | A1 (train_audio only, hybrid loss) | **0.933 (A1+B1 baseline)** |
| v2 | A2 (train_audio + train_soundscapes, ASL loss) | 0.926 |
| **v3** | **A1 restored** (v1 JIT files) | **0.933 (restored baseline)** |

Recovery: v1's JIT files were recovered via the undocumented
`datasetVersionNumber=N` query param on the Kaggle REST download
endpoint (`GET /api/v1/datasets/download/{owner}/{slug}?datasetVersionNumber=1`).
The A1 source state_dicts in `models/a1/` were already absent from disk
pre-A2, and my export step overwrote the v1 JITs at
`four_track/kaggle_datasets/a1-effb0-ckpts/` — nearly unrecoverable without
the versioned-download workaround. Local archive now at
`four_track/models/a1_jit_archive/a1_fold{0,1,2,4}.pt` to prevent a repeat.

**Why A2 regressed** (hypotheses, not confirmed):
1. **Train-set contamination of val = false signal**. A2 was selected on
   val_roc_auc computed on the same soundscape segments that went into the
   training set. The selector optimized for memorization, not generalization.
2. **ASL vs hybrid loss**. A2 used pure ASL (vs A1's hybrid BCE+ASL). The
   A3-v2 ablation already showed pure ASL hurts on this pipeline (see
   "A3-v2 result (2026-04-10)"). I should have noticed before launching A2.
3. **Soundscape oversampling ratio**. `--soundscape-mult=10` may have
   over-amplified the 1478 soundscape segments relative to 28k focal clips,
   destabilizing batchnorm / mixstyle statistics.

**Track A2 verdict**: killed on first LB probe. Not a candidate for retry
with a different recipe — the contaminated-val selection problem is
structural, and rerunning A2 with cleaner loss/oversample settings would
still need LB arbitration (one more slot burned). Retired.

**A2 artifacts retained** for potential offline diagnostics:
- `four_track/models/a2/a2_tf_efficientnet_b0.ns_jft_in1k_fold{0,1,2,4}_seed42_asl.pt`
- Kaggle dataset v2 remains in version history (not deleted).

---

#### Next concrete sequencing item (post-A3-abandonment, 2026-04-10)

With A1 frozen at LB 0.932, B1 frozen at LB 0.933, C1 done, C2 dead, D1-a failed, Track A3 abandoned, the live tracks per §3 sequencing are:

| Track | Status | Blocker |
|---|---|---|
| **D2-β** (Kaggle-side OOF dump → local LightGBM stacker) | **candidate — RECOMMENDED NEXT** | needs a one-shot Kaggle kernel run that dumps B1 + ProtoSSM validation sigmoids on a common 5-fold split as a Kaggle dataset artifact. After that the stacker itself is pure local work. ~½ day Kaggle infra + ~½ day local. |
| D2-α (full Kaggle-side stacker, LightGBM trained in-notebook) | fallback | heavier Kaggle-side lift; burns submission slots for iteration. Only if β proves infeasible. |
| A2 (ProtoSSM-self-train with soundscape pseudo-labels) | candidate | needs a one-shot Kaggle kernel that runs ProtoSSM in-budget and dumps soundscape pseudo-labels as a Kaggle dataset (same Perch-embedding-mismatch constraint that killed C2). ~1 day infra work. |
| D3 (per-taxon ensemble weights) | gated on D2 | wait for D2 result — D3 is a refinement of D2's stacker, doesn't make sense to build before the base stacker exists. |

**⚠️ D2 is NOT purely local — correction logged 2026-04-11.** The original plan claimed D2 was "pure local post-processing on outputs we already have." That was wrong. Filesystem audit (2026-04-11) found:
- A1 5-fold OOFs: `four_track/data/a1_soundscape_preds.npz` — 1478 soundscape val rows.
- ProtoSSM OOFs: `models/protossm_pretrained/oof_predictions.npz` — **708 rows, different split**.
- 0911 teacher OOFs: `data/external/birdclef-0911/teacher_oof_predictions.npz` — **739 rows, different split**.
- B1 (PerceiverIO) OOFs: **do not exist on disk**. B1 is Kaggle-trained end-to-end on Perch embeddings and we never dumped val sigmoids locally.

No shared row substrate exists for a stacker. D2 therefore requires either (α) training the stacker inside the Kaggle notebook, or (β) a one-shot Kaggle kernel run that dumps `{A1_val, B1_val, ProtoSSM_val, y_true}` on a common 5-fold split as a dataset artifact for local development. **β is preferred** because it keeps the stacker iteration loop local and cheap; α is the fallback if β proves too complex.

##### D2-γ result (2026-04-11) — **NULL, A3 fully closed including salvage**

Before committing to D2-β, ran a degenerate local-only "D2-γ" to answer the narrow A3 salvage question: *does the retained v3 NFNet fold-0 checkpoint add anything to A1 on the 1478 soundscape val set?*

Script: `four_track/src/d2_gamma.py`. Loads A1 5-fold probs from the existing dump, runs NFNet v3 fold-0 inference on the same val set, grid-searches blend weight `w·A1_ens + (1-w)·NFNet`.

| metric | value |
|---|---|
| A1 fold 0 (best single fold on this set) | 0.7414 |
| A1 fold 1 | 0.7232 |
| A1 fold 2 | 0.6970 |
| A1 fold 3 | 0.6636 |
| A1 fold 4 | 0.7250 |
| **A1 5-fold soft-vote ensemble** | **0.7017** ← *worse than 4/5 individual folds* |
| NFNet v3 fold-0 | 0.7458 |
| Best blend (w=0.05) | 0.7462 (+0.0004 over NFNet alone) |

**Two findings:**

1. **NFNet has no salvage value.** The blend sweep is strictly monotonic from w=0 (0.7458) to w=1 (0.7017) with only a +0.0004 bump at w=0.05 — well inside noise on 1478 rows × 75 present classes. Apples-to-apples single-fold comparison is A1-f0 0.7414 vs NFNet-f0 0.7458 = +0.0044, also inside noise. A real stacker win would show a bowl-shaped AUC curve with a clear interior optimum; a monotonic sweep means the two models are not diverse, one is just slightly better in isolation. **Delete `models/a3/` and `models/a3_v3/` — Track A3 is fully closed including salvage.**

2. **A1's soft-vote ensemble exhibits severe cross-fold calibration drift.** The stored 5-fold soft-vote (0.7017) is *worse* than 4 of its 5 individual folds (fold-0 = 0.7414, fold-1 = 0.7232, fold-2 = 0.6970, fold-4 = 0.7250, only fold-3 at 0.6636 is worse). Averaging sigmoids across folds with different temperature/bias calibration is destroying signal. This is **a real D1-style finding independent of Track A3**: before the 5-fold ensemble averages sigmoids on Kaggle (which is what the current A1 submit path does via rank fusion with B1), the folds should be temperature-scaled to a common calibration — e.g. fit one scalar T per fold on a held-out slice, then soft-vote on `sigmoid(logit/T)`. Plan estimate on this finding alone: unknown, but the gap between per-fold and soft-vote here is 0.04 AUC, which is enormous relative to the +0.002–0.004 D2 stacker estimate. Worth exploring as **D1-b (per-fold temperature scaling)** separately from the D2 stacker work. Note: A1's Kaggle submit uses *rank fusion* with B1, not raw soft-vote, which may already implicitly sidestep this — verify before investing in D1-b. Also note the A1 "ensemble here = 0.7017 vs LB = 0.933" gap means this local soundscape val set is not a calibrated proxy for LB in absolute terms; any D1-b finding would need LB confirmation before committing.

**Recommended next item: D2-β (Kaggle-side OOF dump → local LightGBM stacker)**, because:
1. Answers the real stacker question (does combining A1+B1+ProtoSSM beat the current A1+B1 rank fusion?) that D2-γ could not.
2. Unblocks D3 (per-taxon ensemble weights) as a natural follow-on.
3. Hard constraint re-grep before starting: per the `feedback_regrep_constraints` memory, re-grep both `four_track/new_plan.md` and `../plan.md` for "must be trained on Kaggle" / DOA / embedding mismatch hazards **before** committing to a D2-β implementation design. The C2 kill established that constraint-oversight costs ~1 day; do not repeat it.

##### D2-β design (planned 2026-04-11)

**Prerequisite re-grep (done 2026-04-11):** Both plan files re-grepped for "must be trained on Kaggle" / DOA / embedding-mismatch hazards per `feedback_regrep_constraints`. Result: constraint is **live and binding** for both B1 and ProtoSSM — both consume Perch v2 embeddings, and local Perch ≠ Kaggle Perch (the canonical lesson from #32/#34A/C2). Any OOF dump for B1 or ProtoSSM **must** come from a Kaggle kernel run; a local dump would inject the same embedding-mismatch poison that killed C2.

**Key infrastructure findings (from 2026-04-11 exploration):**

1. **Canonical submit notebook**: `jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb` — single integrated notebook that runs ProtoSSM (baseline) → B1 rank fusion (cell 36b, injected via `inject_b1_cell.py`) → A1 rank fusion (cell 37, injected via `inject_a1_cell.py`). The injected cells live in `four_track/src/b1_perceiver.py` (B1 cell) and `four_track/src/a1_notebook_cell.py` (A1 cell). This is the notebook to modify for the D2-β OOF dump.
2. **Fold alignment**: B1 and ProtoSSM both use **GroupKFold(5) on identical `file_groups`** (unique labeled-soundscape file IDs). `four_track/src/b1_perceiver.py:29` explicitly states "Reuses ProtoSSM's `file_groups` so the splits [are identical]". A1, by contrast, uses `MultilabelStratifiedKFold(5)` **on train_audio focal clips** — a completely disjoint data distribution from soundscapes. **Crucial implication: A1 never trains on any soundscape file, so A1's predictions on any soundscape segment are automatically out-of-fold regardless of which GroupKFold split that segment falls into.** A1 can contribute a single non-fold-dependent "ensemble rank" feature per row; B1 and ProtoSSM contribute genuine OOF features that respect the GroupKFold holdout structure.
3. **Common substrate**: The B1+ProtoSSM OOF row space is **708 rows = 59 files × 12 windows/file** where each window is a fixed 5s slice at `t ∈ {0, 5, 10, …, 55}` in a 60s soundscape file. That 708-row substrate has `y_flat` labels (derived from `train_soundscapes_labels.csv`) and is the natural stacker training substrate. The 1478-segment count used by `build_soundscape_val` in `train_a1.py` is a different tiling strategy and is not needed for D2-β.
4. **A1 aggregation (verified 2026-04-11)**: `a1_notebook_cell.py:189-206` rank-averages the 4 A1 folds (`A1_FOLDS = {0,1,2,4}`, fold 3 intentionally excluded). A1's stacker feature should mirror this — use rank-averaged A1 output, not raw sigmoid mean, so the feature space the stacker learns on matches what it will see at inference.
5. **Local ProtoSSM OOF at `models/protossm_pretrained/oof_predictions.npz` is POISONED**. File date 2026-04-03 22:41 matches the #32 submission ("Pre-train ProtoSSM locally, inference-only on Kaggle, 0.922 LB"), which was the canonical case of the local-Perch-embedding trap. **This file must not be used as a D2-β stacker input.**

**Phase 1: Kaggle-side OOF dump kernel (≈½ day)** — **scoped and implemented 2026-04-11**

Goal: one-shot kernel run (`MODE="train"`) on the canonical `birdclef2026-protossm-postproc` notebook that produces a downloadable artifact `/kaggle/working/d2_beta_oofs.npz` containing `{a1_ranks, b1_oof, proto_oof, y_true, file_groups, fold_ids, n_windows, a1_folds}` on the shared 708-row substrate (59 labeled train_soundscapes files × 12 windows/file).

**Key scoping finding (2026-04-11):** Phase 1 is *much* smaller than originally sketched. In the existing canonical notebook, cell 31b (`b1_perceiver.py:CELL_31B`) **already computes `oof_proto_flat`, `oof_b1_flat`, `y_flat`, and `file_groups` on the 708-row substrate in every `MODE="train"` run** — these are inputs to the existing B1 OOF gate and correlation sweep, and they're already used by cell 31b for its own diagnostics. What was missing was just (a) an A1 pass on the same 59-file substrate and (b) a `np.savez` call to persist everything. No fold-loop retraining, no disjoint-holdout reimplementation, no notebook fork.

**Actual Phase 1 scope (implemented):**

1. **New cell source** at `four_track/src/d2_beta_oof_cell.py:CELL_SOURCE` (220 lines). Injects one new cell ("cell 31c") that runs only in `MODE="train"`:
   - Asserts `oof_proto_flat / oof_b1_flat / y_flat / file_groups / full_paths` exist with expected shapes.
   - Runs A1 streaming inference on `full_paths` (the same 59 labeled files B1/ProtoSSM OOF'd on) using the identical mel + PCEN pipeline as cell 37's submit-path A1 fusion (`a1_notebook_cell.py:106-124`), then rank-averages per class across the 4 A1 folds `{0,1,2,4}`. Produces `a1_ranks_oof` of shape `(708, 234)`.
   - `np.savez`s all eight arrays to `/kaggle/working/d2_beta_oofs.npz`.
   - Logs sanity numbers (`a1_ranks_mean`, `b1_oof_mean`, `proto_oof_mean`, `y_true_positives`, `present_classes`, `file_size_mb`, `wall_time_seconds`) into `LOGS["d2_beta_oof_dump"]` so run history shows dump health without needing to download first.
   - In `MODE="submit"` this cell is a single-line no-op — it's safe to leave injected permanently on the LB notebook without affecting LB 0.933.
2. **New injector** at `four_track/src/inject_d2_beta_oof_cell.py` mirroring `inject_a1_cell.py` / `inject_b1_cell.py`:
   - Idempotent: replaces in place on re-injection, otherwise inserts after the B1 OOF cell (anchor marker `"# Cell 31b — Track B1 PerceiverIO instantiate"` at `b1_perceiver.py:478`).
   - Does **not** add any new `kernel-metadata.json:dataset_sources` entry — the only external dataset the D2-β cell needs is `stevewatson999/birdclef-2026-a1-effb0-ckpts`, already added by `inject_a1_cell.py`.
3. **Validation (2026-04-11)**:
   - `ast.parse(CELL_SOURCE)` passes — cell source is syntactically valid Python.
   - B1 anchor marker is present in exactly one cell of the canonical notebook (index 33), so the injector will insert at index 34 unambiguously.
   - D2-β marker is not yet present in the notebook (as expected — injector has not been run yet).

**Runtime to execute Phase 1 on Kaggle (estimated):**
- Baseline train-mode kernel wall time (without this cell): ~30-45 minutes (ProtoSSM OOF + B1 OOF + rest of postproc pipeline).
- Additional cost of the D2-β cell: A1 streaming inference on 59 files × 12 windows × 4 folds. On the submit path the same operation on hidden-test (~700 files × 12 × 4) takes a few minutes — for 59 files it should be ≤1 minute. Negligible relative to the full kernel run.
- Total: well under the 9h Kaggle kernel cap. No risk.

**Phase 1 execution checklist** — **COMPLETE 2026-04-11**:

All steps done end-to-end in one session. Commit trail:

- [x] Write `four_track/src/d2_beta_oof_cell.py` (cell source).
- [x] Write `four_track/src/inject_d2_beta_oof_cell.py` (injector).
- [x] Validate `ast.parse(CELL_SOURCE)`; B1 anchor at unique index 33; D2-β marker initially absent.
- [x] `jupyter nbconvert --clear-output --inplace` → commit `1ec4c0c6d1` "protossm-postproc: clear notebook outputs".
- [x] Run injector → D2-β cell inserted at index 34, commit `9573e3aa40` "D2-β Phase 1: inject OOF-dump cell into protossm-postproc".
- [x] Flip cell 1 `MODE = "submit"` → `MODE = "train"`, commit `1b4c5eac7c` "D2-β Phase 1: flip MODE to train for OOF dump run".
- [x] `kaggle kernels push` → **v25**. Kernel ran ~1000s and crashed at final `np.savez` with `ValueError: invalid literal for int() with base 10: np.str_('S08')` — cell forced `dtype=np.int64` on `file_groups`, but the OOF substrate uses site-ID strings (`S03`, `S08`, …) not ints. All expensive work (ProtoSSM OOF, B1 OOF, A1 OOF inference 225.8s) succeeded; failure was only in the save step.
- [x] Fix: factorize `file_groups` into contiguous ints via `np.unique(..., return_inverse=True)`, keep string names as separate `file_group_names` side array. Commit `1240bea6b4` "D2-β Phase 1: fix file_groups int cast in OOF dump cell".
- [x] Re-push → **v26**. Ran successfully end-to-end in ~1000s. A1 OOF sub-pass wall time 195.7s.
- [x] Download `/kaggle/working/d2_beta_oofs.npz` → saved as `four_track/data/d2_beta_oofs.npz` (2.66 MB).
- [x] Revert `MODE = "submit"`, commit `125383ec71` "D2-β Phase 1: revert MODE to submit — LB 0.933 production state restored".
- [x] `kaggle kernels push` → **v27**. Canonical notebook back to LB-ready state; D2-β cell remains injected but is a no-op in submit mode.

**Kaggle-side cost actual:** 2 kernel runs (one failed at the np.savez step, one clean). No LB submission slots consumed.

**OOF artifact contents** (`four_track/data/d2_beta_oofs.npz`, 2.66 MB):

| Key | Shape | Dtype | Note |
|---|---|---|---|
| `a1_ranks` | (708, 234) | float32 | rank-averaged A1 over folds {0,1,2,4}, per-col rank ∈ [0,1]; macro AUC on 71 active classes = **0.7359** |
| `proto_oof` | (708, 234) | float32 | ProtoSSM OOF logits (range −5.9 … +5.9); macro AUC = **0.6353** |
| `b1_oof` | (708, 234) | float32 | PerceiverIO OOF logits (range −3.4 … +1.0); macro AUC = **0.4028** — near-random, consistent with the structurally-broken B1 OOF on this 59-file substrate |
| `y_true` | (708, 234) | float32 | 3087 positives, 71 active classes |
| `file_groups` | (59,) | int64 | 8 unique site groups (not 59!) |
| `file_group_names` | (8,) | \<U3 | `['S03','S08','S13','S15','S18','S19','S22','S23']` |
| `fold_ids` | (708,) | int64 | 12-tiled `file_groups` |
| `n_windows` | scalar | int64 | 12 |
| `a1_folds` | (4,) | int64 | `[0,1,2,4]` (fold 3 intentionally excluded) |

**Important structural finding**: `file_groups` yields only **8 unique site-level groups**, not 59 per-file groups. GroupKFold on this substrate is bounded at ≤8 splits; 5-fold GroupKFold is safe. This is a meaningfully smaller CV denominator than what the Phase 2 design below assumed ("59 file-level units"). The noise-floor comments in §Risks remain directionally right but slightly pessimistic — with only 8 groups, per-fold ΔAUC variance is dominated by which sites land together, not how many files are held out. Plan Phase 2 with 5 seeds as baseline and 10 seeds if marginal.

**Phase 2: Local LightGBM stacker (≈½ day)**

File: `four_track/src/d2_lgbm_stack.py`.

Design:
1. Load `d2_beta_oofs.npz`. Reshape into a long format: `708 rows × 234 classes = 165_672 samples`, each with features `[logit_a1, logit_b1, logit_proto, class_id]` (convert sigmoids back to logits for the blender; `class_id` is a categorical feature so the tree can learn per-class biases). Target: `y_true.flatten()`.
2. **5-fold GroupKFold on `file_groups`** — NOT segment-level, NOT sample-level. The D1-a postmortem proved segment-level splits leak via the per-file tie structure. Class-level and sample-level splits would also leak because each file contributes 12×234 samples with highly correlated labels. The only safe split is by `file_groups`.
3. Fit **one LightGBM binary classifier** over all classes (shared across classes via the `class_id` categorical feature). Alternative: start with a **simpler baseline** — a single-parameter logistic blender `P = σ(w_A · logit_a1 + w_B · logit_b1 + w_P · logit_proto + b)` fit per class or shared. LightGBM can overfit with only 59 file-level units in the outer CV; fit the logistic baseline first and only escalate to LightGBM if the logistic baseline shows positive signal.
4. **Evaluation**: macro ROC-AUC on the `y_true[class present in fold k]` subset, aggregated across 5 GroupKFold folds. Repeat 5 seeds, take median. Baseline to beat: the existing rank-fusion output (A1+B1+ProtoSSM with frozen weights w_A1=0.20, w_B1=0.10) on the same 708-row substrate — the kernel should dump that baseline output as `final_test_scores_baseline` in the same npz so we have an apples-to-apples reference.
5. **Gate before spending a Kaggle submission slot**: median Δ over 5 seeds ≥ **+0.001** macro ROC-AUC AND sign-stable across all 5 seeds (no seed regression). Stricter than the D1-a gate because (a) the stacker adds 3+ parameters vs D1-a's one-per-class isotonic and (b) the 708-row substrate is small enough that a +0.001 delta is only ~2x the sampling noise floor.

**Phase 3: Kaggle integration (≈½ day, only if gate passes)**

If the gate passes, inject the fitted stacker into `birdclef2026-protossm-postproc` as a new cell **after** the A1 rank fusion but **before** the post-processing thresholds (cell 38). Implementation: ship the fitted LightGBM model (or logistic weights) as a Kaggle dataset, load in the notebook, apply to the A1+B1+ProtoSSM per-class outputs, rewrite `final_test_scores` via the same inverse-CDF pattern the A1 and B1 cells already use so downstream thresholds remain valid. One submission. If LB ≥ 0.933 + 0.001, freeze the stacker; otherwise roll back and file the result in `new_plan.md`.

**Risks / failure modes to watch**:
- **59-file CV noise floor.** With only 59 files in the GroupKFold base pool, single-seed ΔAUC has a std of ~0.002-0.005; the +0.001 gate is aggressive and may demand more seeds to confirm sign. If the gate is marginal, upgrade to 10 seeds before deciding.
- **A1 rank vs A1 sigmoid mismatch.** A1's per-class rank is bounded in [0,1] while B1/ProtoSSM are sigmoid outputs. The stacker needs a consistent input representation — either rank-transform B1 and ProtoSSM per class to match A1, or convert A1 back to a logit-like scale. Prefer the former (rank-space) because the Kaggle submit path already lives in rank space for the fusion stage.
- **Label sparsity.** 708 rows × 234 classes with ~25-per-class median positive count means most classes will have 0-1 positives per fold. The macro AUC aggregation will collapse to a small set of "present" classes per fold. Document how many classes are present per fold in the kernel output for sanity checking.
- **LB ≠ local delta.** D1-a's failure reminded us that file-level local CV on 59 files is a noisy estimator of LB. The +0.001 local gate should be treated as a minimum, not a prediction — plan for the Kaggle LB to move somewhere in `[local Δ − 0.003, local Δ + 0.003]`.

**Explicit non-goals for D2-β**:
- No per-taxon weight tuning (that's D3, gated on a successful D2).
- No retraining of A1, B1, or ProtoSSM (D2-β is strictly post-processing).
- No attempt to reuse the poisoned `models/protossm_pretrained/oof_predictions.npz` — delete or ignore it in the D2-β workspace.
- No local evaluation on the 1478-segment `eval_a1_ensemble.py` val set — that val set is on a different row space and uses `sigmoid-mean` A1 which is the wrong baseline (per the `project_a1_calibration_drift` memory).

### Track C — ProtoSSM-as-teacher pseudo-labels on train_audio (medium-low lift)

**Hypothesis**: `train_audio` has ~46K focal clips. We currently use exactly **0** of them for ProtoSSM training (ProtoSSM only sees ~720 fully-labeled `train_soundscapes` files). Pseudo-labeling these with ProtoSSM and retraining gives the model 60× more data.

**Why this might work where #34E (Perch logits as pseudo-labels) failed**: Perch logits are too noisy. ProtoSSM predictions are calibrated, OOF-valid, and respect the soundscape distribution. They won't have the "always-on" failure mode of Perch on noisy audio.

**Concrete subtracks**:

| ID | Description | Risk | Expected lift |
|----|-------------|------|---------------|
| C1 | **Extract Perch v2 embeddings** for all 46K train_audio clips locally. Cache to a Kaggle dataset. | LOW | enables C2 — **DONE 2026-04-09** |
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
cd /home/swatson/work/kaggle/BirdCLEF/four_track
nohup bash scripts/train_a1_5fold.sh > log/train_a1_5fold_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

While A1 trains, **Track C2** can run as a CPU-side data-prep job in parallel (it doesn't contend for the GPU once the pseudo-label dump is done).

---

## 8. Pantanal reverse-ablation (active 2026-04-15)

**Problem**: `jupyter/protossm-pantanal/` was snapshotted at V19
(`bfceb494c4`) with the full pantanal-derived change set. That V19
notebook **timed out on the LB scorer**. A leaner variant (the working
tree at conversation start) scored **LB 0.931** without timing out.

**Goal**: identify which specific pantanal change(s) push the notebook
over the LB time budget (and/or regress score) by re-applying them one
at a time on top of the LB 0.931 baseline.

**Baseline commit** `b33ec1a7e2` — pure LB 0.931 state. Metadata:
`brucewu1200/birdclef-2026-cvlb-assets-0911` +
`stevewatson999/birdclef-2026-a1-effb0-ckpts` (not V19's
`protossm-b1-ckpts`).

**Workflow**:
1. One commit per pantanal candidate, applied on top of the baseline.
2. After each commit, push `stevewatson999/birdclef-2026-pantanal` to
   Kaggle via `kaggle kernels push` and measure LB + wall time.
3. If the commit regresses LB or causes a timeout: `git revert <sha>`
   and move to the next candidate. If it's neutral/positive: keep it
   and layer the next candidate on top.
4. Each commit is small and isolated so that `git revert` returns the
   tree to a clean known-good state (the baseline or the last
   accepted ablation), never to V19.

**Candidate order** (smallest-net-delta first):

| # | Candidate | HEAD cell | Δ lines | Status |
|---|---|---|---|---|
| P10 | Score Fusion `.to(DEVICE)` / `.cpu()` | Score Fusion | +1 | **applied** (`d0fc7119db`) |
| P5  | Cell 5 Perch inference | Cell 5 | +2 | pending |
| P4  | Cell 4 helpers (V16/V17 NEW) | Cell 4 | +8 | pending |
| P6  | Cell 6 Perch cache | Cell 6 | +11 | pending |
| P7  | Cell 24 ProtoSSM v4 training loop | Cell 24 | +18 | pending |
| P9  | Cell 31b PerceiverIO OOF+retrain | Cell 31b | +16 | pending |
| P11 | Cell 17 A1 SED fusion | Cell 17 | +17 | pending |
| P8  | Cell 31 Instantiate+train ProtoSSM v4 | Cell 31 | +27 | pending |
| P2  | V18 CFG UPGRADES (CPU-tuned) variant | V18 CFG block | (swap 47L → 37L) | pending |
| P3  | Cell 3 Load Perch TF-only path | Cell 3 | (swap) | pending |
| P1  | Cell 0 install (TF 2.20 only, no ONNXRuntime) | Cell 0 | (swap) | pending |

**Explicitly out of scope** (baseline-only edits, not pantanal-added
content — do not revert):
- Cell 2 Imports (WORK has +30L over HEAD)
- Cell 18 postproc (WORK +7L)
- Residual SSM (WORK +1L)

**Results log**:

| Commit | Candidate | LB | Wall time | Timeout? | Verdict |
|---|---|---|---|---|---|
| `b33ec1a7e2` | baseline | 0.931 | (known good) | no | pass |
| `d0fc7119db` | P10 | 0.916 | n/a | no | **fail** (−0.015) — reverted in `4d78564641` |
| `1c73f0f989` | P6 | (no push) | n/a | n/a | **pass by inspection** — additive fallback paths only, existing resolver already finds cache on Kaggle so new entries are unreachable; no behavior change expected |

**Closeout (2026-04-15):** reverse-ablation exhausted without finding a
score-positive candidate. Byte-for-byte comparison of baseline's "V18 CFG
UPGRADES" cell vs V19's version confirms that **the V18 architectural
upgrades (`d_model=320`, `n_ssm_layers=4`, `n_prototypes=2`) are already
in the LB 0.931 baseline**. Every remaining V19 delta falls into one of:

- **no-op plumbing** (P1, P3, P5, P6, P7, P9, P11a) — alternate Kaggle
  input layouts, `.to(DEVICE)` calls on a CPU-only kernel, ckpt-loader
  fallbacks that never resolve with baseline metadata
- **speed-for-score trades** (P2, P11b) — fewer TTA shifts, smaller MLP
  probe, fewer folds; all intended as timeout mitigations for V19's
  heavier training budget, not score wins
- **speed-only wins** (P11c PCEN vectorization) — neutral on LB unless
  paired with a "spend the saved time" change (more TTA / folds)
- **confirmed regressions** (P10) — `.to(DEVICE)/.cpu()` in Score Fusion,
  −0.015 on V22

Therefore pantanal cannot close the 0.931 → 0.951 leaderboard gap. The
+0.020 target must come from **Tracks A / B / C / D** in the upper
sections of this plan (new signal sources, not rearranging the existing
ProtoSSM stack).

---

## 9. Next action after §8 closeout (2026-04-15)

**Context:** public LB leader is at **0.951**; our production baseline is
**0.931**; minimum useful gain per submission is **+0.020**. §8 confirmed
the pantanal branch cannot deliver this, so the next submission has to
come from a track that introduces new signal, not one that rearranges
ProtoSSM/A1/B1 plumbing.

### Chosen next step: resume D2-β Phase 2 S1 stacker LB probe

**Why this first** (ahead of Track A SED or Track C pseudo-labels):

1. **Fully staged already.** Commit `ce2594d496` (2026-04-11) left the
   repo in a ready-to-push state:
   - `four_track/src/d2_lgbm_stack.py` has `fit_final_s1()` / `apply_s1()`
     and the fitted per-class logistic coefs live in
     `four_track/data/d2_beta_s1_coefs.npz`.
   - The kernel-side cell `four_track/src/d2_s1_kernel_cell.py` is
     already injected into the canonical `protossm-postproc` notebook
     and is a no-op in `MODE="submit"` unless the coef dataset is
     attached.
   - `kernel-metadata.json` changes + coef-dataset upload path were
     documented in §D2-β Phase 2. Only action left is
     `kaggle kernels push` and measure LB.
2. **Answers a live gate, not builds a new one.** Phase 1.5 found the
   708-row local OOF substrate is structurally broken (LB-production
   rank fusion scored 0.6699 locally vs A1-alone 0.7359). That makes
   the local S1 lift of +0.1211 **untrustworthy as a local gate**. The
   only way to learn whether S1 is real is one LB slot. Every day we
   don't push it, the question stays open.
3. **Cheap to revert.** If S1 regresses LB, a single commit revert
   returns us to LB 0.931 baseline. No code is replaced, only one cell
   added and one dataset attached.
4. **Unblocks D3.** If S1 passes, per-taxon ensemble weights (D3) become
   a natural follow-on using the same OOF substrate.

**Concrete steps for the next session:**

1. Re-grep both `four_track/new_plan.md` and `../plan.md` for
   "must be trained on Kaggle" / DOA / embedding-mismatch hazards per
   `feedback_regrep_constraints`. Verify the S1 coefs are a *local*
   post-processing fit over Kaggle-produced OOF scores (they are —
   fit on `four_track/data/d2_beta_oofs.npz` which came from Kaggle
   kernel v26), so the hazard doesn't apply; but document the check.
2. Confirm the `protossm-postproc` canonical notebook on disk matches
   the submit-mode production state (MODE="submit", D2-β OOF-dump cell
   present as no-op, S1 cell present as no-op without coefs dataset).
3. Upload `four_track/data/d2_beta_s1_coefs.npz` to a private dataset
   `stevewatson999/birdclef-2026-d2b-s1-coefs` and add it to
   `jupyter/protossm-postproc/kernel-metadata.json:dataset_sources`.
4. `kaggle kernels push` on `protossm-postproc`; submit to LB.
5. Record outcome in the D2-β Phase 2 section and in the top-of-file
   LB submission history.

**Status (2026-04-15):** Steps 1–4 complete. S1 coefs uploaded as
`stevewatson999/birdclef-2026-d2b-s1-coefs`, notebook pushed as **v46**.

**Result (2026-04-15): v46 LB = 0.775 — catastrophic regression (−0.156
from 0.931 baseline).** Decision gate → **regression → revert S1 +
document, pivot to Track A SED.**

Root cause (likely): the per-class logistic stacker was fitted on 708 OOF
rows but applied to ~200 test rows. Rank features on 200 rows have
completely different granularity/distribution than on 708 rows, and the
per-class logistic coefs massively overfit to the OOF rank structure.
The inverse-CDF remapping then amplified the distortion. S1-style
rank-based stacking is structurally incompatible with train/test size
mismatch — this approach is dead, not just poorly tuned.

**Action taken:** D2-β S1 is permanently killed. Revert notebook to
pre-S1 baseline (remove S1 cell + coefs dataset from kernel-metadata).
§10 pre-trained ProtoSSM checkpoint probe is next in line (independent
of S1), followed by Track A SED pivot.

Next action depends on the decision gate below (now resolved).

**Decision gate on the result:**

| LB outcome | Verdict | Next step |
|---|---|---|
| ≥ 0.951 | 🎉 single-slot close of the gap | lock S1 in production, resume other tracks for further gain |
| 0.951 > LB > 0.933 | partial win | keep S1, proceed to D3 (per-taxon weights) for the rest of +0.020 |
| 0.931 ≤ LB ≤ 0.933 | neutral | revert S1, conclude D2-β exhausted, pivot to Track A SED |
| < 0.931 | regression | revert S1 + document, pivot to Track A SED |

**If S1 is neutral/regressive**, the follow-on is **Track A (SED on
raw audio)**, not Track C. Rationale:

- Track A introduces a fundamentally new signal path (mel-spectrogram
  CNN/SED on raw audio, independent of Perch embeddings) — the class
  of model change that typically moves LB by 0.01–0.03 on BirdCLEF.
- Track C (pseudo-labels) requires a working student that can consume
  Kaggle-produced Perch features without the embedding-mismatch trap;
  C1 has the feature extraction staged (commit `a6ffebdeca`) but C2
  is still flagged KILLED in §Track C. Reviving it is a bigger design
  lift than starting Track A.

**Budget:** the S1 probe consumes one LB submission. Track A takes
~2–4 days of training work before its first LB slot. Plan for at most
one submission per LB-probe day until we know whether S1 lands.

---

## 10. Pre-trained ProtoSSM/B1 checkpoints (side-project, 2026-04-15)

**Problem:** ProtoSSM and B1 PerceiverIO train from scratch every
submission inside the 90-min Kaggle CPU budget. This caps them at
~30 epochs / small model / no hyperparameter sweeps. The 0.931 → 0.951
gap likely requires more model capacity or training than the budget
allows.

**Key discovery:** `jaejohn/perch-meta` is a Kaggle dataset containing
Perch v2 embeddings+logits for all 59 train_soundscape files (708 rows),
extracted *on Kaggle* on 2026-03-13. The production notebook already
loads these for training (Cell 6 `resolve_full_cache_paths`). Verified
2026-04-15 that the local cache (`data/processed/perch_cache/`) does
NOT match — max embedding diff = 0.118, confirming the local-vs-Kaggle
mismatch is real and the `jaejohn/perch-meta` copy is the authoritative
Kaggle-extracted version.

**Implication:** we can train ProtoSSM and B1 *locally* on the
Kaggle-extracted features with no time limit, upload the checkpoints,
and load them in the submission notebook — skipping in-kernel training
entirely. This sidesteps the 90-min bottleneck without hitting the
embedding-mismatch wall (because the training features *are* Kaggle
features, just downloaded).

**This is a side-project**: it does not modify the current production
pipeline until Step 5. Steps 1–4 are purely additive.

### Steps

**Step 1 — Local training script** (`four_track/src/train_protossm_local.py`)
- Loads features from the downloaded `jaejohn/perch-meta` dataset
  (copied to `four_track/data/kaggle_perch_cache/`)
- Loads labels from `train_soundscapes_labels.csv` via the competition
  data directory
- Replicates the exact ProtoSSM v5 architecture from the notebook
  (Cell 22/24) — same `d_model`, `d_state`, `n_ssm_layers`, etc.
- Trains with full freedom: 200+ epochs, LR sweep, multi-seed,
  optionally larger `d_model` / more layers
- GroupKFold OOF on the same 8-site split the notebook uses
- Saves best checkpoint as `protossm_pretrained.pt`
- Estimated local runtime: minutes (708 rows is tiny)

**Step 2 — Local B1 training** (same script or
`four_track/src/train_b1_local.py`)
- Same feature source, same labels, same GroupKFold
- Replicates PerceiverIO architecture from Cell 24b
- Saves `b1_pretrained.pt`

**Step 3 — Upload checkpoints**
- `kaggle datasets create` → `stevewatson999/birdclef-2026-pretrained-ckpts`
- Contains `protossm_pretrained.pt` + `b1_pretrained.pt`
- Add to `kernel-metadata.json:dataset_sources`

**Step 4 — Guarded load path in notebook**
- Cell 24 (ProtoSSM training): wrap in
  `if ckpt.exists(): model.load_state_dict(...) else: <current training>`
- Cell 31b (B1 training): same pattern (P9 in §8 already had this for
  B1 — reuse the structure but point at the new dataset)
- Without the checkpoint dataset attached, notebook behaves identically
  to today

**Step 5 — LB probe**
- `kaggle kernels push` with checkpoint dataset attached
- Compare LB vs 0.931 baseline
- If regression: detach dataset → instant revert to current behavior

### What this unlocks

| Parameter | Current (in-kernel) | With pre-trained ckpts |
|---|---|---|
| ProtoSSM epochs | 30 (submit caps) | 200+ |
| ProtoSSM d_model | 320 (time-limited) | 512+ if data supports it |
| B1 epochs | ~20 | 100+ |
| Hyperparameter sweeps | impossible (1 config/submission) | unlimited local |
| Multi-seed ensembling | impossible | average N seeds |
| Freed Kaggle wall time | 0 | ~25 min → more TTA / A1 folds |

### Risks

1. **Feature staleness**: if Kaggle updates TF/ONNX runtime, the
   `jaejohn/perch-meta` features may drift from live-extracted test
   features. Mitigation: re-run extraction kernel (~30 min). Low
   frequency risk — Kaggle env changes are rare mid-competition.
2. **Overfitting on 708 rows**: more epochs + bigger model on tiny data
   could overfit. Mitigation: OOF monitoring + early stopping + SWA.
   The notebook already has these; the local script replicates them.
3. **Architecture drift**: if the local script's model definition
   diverges from the notebook's, `load_state_dict` will fail with a
   key mismatch. Mitigation: import the model class from a shared
   source file used by both notebook cell and local script.

### Interaction with other tracks

- **D2-β S1 probe** (§9): killed — v46 LB 0.775 catastrophic regression.
- **Track A SED**: independent — A1 operates on raw mel spectrograms.
- **Track C**: if pre-trained ProtoSSM is better, C's pseudo-label
  teacher is also better. Complementary.

### §10 LB probe — v47 (2026-04-15)

**v47 pushed.** Combined operation: revert S1 (cell removed, coefs dataset
detached) + probe §10 pre-trained ProtoSSM checkpoint (dataset
`birdclef-2026-protossm-pretrained-v2` attached, guarded load in Cell 32).
If the checkpoint file is found at submit time, ProtoSSM skips in-kernel
training and loads the locally-trained weights. If not found, falls back
to current from-scratch training (no regression risk).

**Decision gate:**

| LB outcome | Verdict | Next step |
|---|---|---|
| > 0.932 | pre-trained ckpt helps | lock in, consider B1 pre-trained ckpt next |
| 0.930–0.932 | neutral (matches baseline) | ckpt didn't help or wasn't loaded; check logs |
| < 0.930 | regression | detach dataset → instant revert, pivot to Track A SED |

**Result (2026-04-16): v47 LB = 0.929 — slight regression (−0.002 from
0.931 baseline).** Checkpoint confirmed loaded (504s runtime vs ~25 min
baseline). The 200+ epoch locally-trained ProtoSSM did not outperform the
30-epoch in-kernel version. Likely overfitting on 708 rows or training
dynamics mismatch. Verdict: **neutral/slight regression — detach
pre-trained dataset, revert to in-kernel training.**

Next action: push v48 with `birdclef-2026-protossm-pretrained-v2` removed
from kernel-metadata to restore 0.931 baseline, then pivot to Track A SED.

---

## 11. D2-α LightGBM stacker (post-A2-kill, 2026-04-16)

**Context:** §9 ends with D2-β S1 killed on LB (v46 0.775, −0.156) and
notes "S1-style rank-based stacking is structurally incompatible with
train/test size mismatch — this approach is dead, not just poorly tuned."
With Track A2 also killed today (LB 0.926, −0.007 vs 0.933 baseline) and
§10 pre-trained ProtoSSM neutral/slightly regressive (LB 0.929), the
remaining live candidates per the §9 sequencing list are D2-α (LightGBM
fallback to D2-β), §10 B1 pre-trained ckpt, and Track C revival.

D2-α was chosen as the next attempt for two reasons:
1. **Reuses D2-β infrastructure**: the 708-row OOF substrate
   (`four_track/data/d2_beta_oofs.npz`) and the pre-injected kernel cell
   slot `d2_s1_kernel_cell.py` already exist. ~½ day infra spend.
2. **Different failure modes than S1**: trees are not as sensitive to
   feature scale, and sigmoid features are absolute rather than
   batch-size-dependent. The specific mechanism that killed S1 (per-class
   linear coefs amplifying rank distortion under 708→200 row mismatch)
   does not directly apply to a global LGBM with sigmoid features.

### Variant chosen: D2-α-(a) — conservative β swap-out

The user-confirmed variant on 2026-04-16:

- Swap S1's per-class logistic for **single global LightGBM** over
  `[a1_rank, sigmoid(b1_oof), sigmoid(proto_oof), class_id_categorical]`.
- Train and apply on the same fit-on-OOF / apply-on-test pattern as β.
- Mixed feature space (rank for A1, sigmoid for B1/proto) — keeps
  `a1_notebook_cell.py` unchanged at the cost of one extra design
  inconsistency. LGBM trees handle mixed feature scales fine.
- Local sweep: 5 seeds × 5-fold GroupKFold (8 site groups, max 5 splits),
  fixed 200 boost rounds (no early stopping on outer val to keep
  comparison vs BP fair), gate Δ ≥ +0.001 macro AUC sign-stable.

Implementation: `four_track/src/d2_lgbm_stack.py`
(`stacker_S2_global_lgbm`, `fit_final_lgbm`, `apply_lgbm`,
`build_flat_features`, `expand_row_idx_to_flat`).

### Local sweep result (2026-04-16) — **S2 with `class_id` FAILS catastrophically**

Run command: `nohup python -u src/d2_lgbm_stack.py > log/d2_alpha_sweep_*.log 2>&1 &`

| Stacker | OOF AUC (median, 5 seeds) | Δ vs BP=0.6699 | Sign-stable | Gate |
|---|---|---|---|---|
| BP prod-fused (baseline) | 0.6699 | — | — | — |
| B0 A1-only | 0.7359 | — | — | — |
| S1 per-class logreg | 0.7910 | +0.1211 | yes | local PASS (already known dead on LB) |
| **S2 global LGBM (D2-α)** | **0.3376** | **−0.3323** | no | **FAIL — worse than random** |

**Diagnostic gap**: in-sample S2 macro AUC = 0.9953 vs OOF S2 = 0.3376.
Textbook overfit. The two findings together:

1. **`class_id` as categorical is poison.** With 234 classes × 708 rows
   split by 8 site groups, the LGBM memorizes per-class site
   co-occurrence patterns from the training sites. The pattern inverts
   on held-out sites because each site has a different bird community.
   AUC < 0.5 means the trees are predicting in the *wrong direction* on
   holdout — not just overfitting noise but actively learning
   site-conditional priors that are anti-predictive cross-site.
2. **The 708-row OOF substrate is structurally non-predictive of LB.**
   S1 passed locally with +0.12 then died on LB by −0.16 — a 0.28
   gap. S2 in-sample → OOF gap is 0.66. Two independent signals that
   any stacker built on top of this 708-row substrate cannot be gated
   trustworthily by local AUC. The Phase 1.5 finding (BP local =
   0.6699 vs A1-alone = 0.7359; production rank fusion *loses* to
   A1-alone on this substrate) was the first signal of the same
   structural issue.

### Fork chosen: D2-α-A (drop `class_id` from S2)

Three forks were considered; user picked Fork 1:

1. **D2-α-A: drop `class_id` from S2.** Single global LGBM over
   `[a1_rank, b1_sigmoid, proto_sigmoid]` only. Removes the
   memorization vector. Same model applied to all 234 classes —
   essentially a learned non-linear blend of the three signals,
   class-agnostic. Cheap to try (~2 min sweep).
2. **D2-α-B: per-class LGBM.** Same scarcity as S1, trees instead of
   logistic. Likely to pass local (like S1) and die on LB (like S1).
   Skipped — unlikely to teach us anything new.
3. **Pivot away from D2.** Two failures (S1, S2-with-class-id) on the
   same 708-row substrate are strong evidence the substrate is the
   problem, not the stacker family. Pivot to §10 B1 pre-trained ckpt
   or Track C. Held in reserve if Fork 1 also fails.

**Decision criterion for Fork 1:**

- If S2-no-class-id passes the local gate AND its in-sample/OOF gap
  is small (<0.05 AUC), it's the cleanest D2-α candidate seen so far
  and worth a single LB slot. The 708-row substrate is still
  unreliable, so an LB regression is plausible — same revert path as
  S1 (detach dataset, instant rollback to 0.933).
- If S2-no-class-id also fails locally OR shows a large in-sample/OOF
  gap, that's the third independent signal that the 708-row substrate
  is non-predictive. **Declare D2 structurally dead** in §11 and
  pivot to §10 B1 pre-trained ckpt or Track C in §12.

### Fork 1 result (2026-04-16) — **S2A FAIL, D2 declared structurally dead**

Re-ran sweep with both S2 (with class_id, retained for forensic comparison)
and S2A (no class_id, the active D2-α candidate):

| Stacker | OOF AUC (median) | Δ vs BP=0.6699 | Sign-stable | In-sample | In-sample/OOF gap |
|---|---|---|---|---|---|
| BP prod-fused (gate baseline) | 0.6699 | — | — | — | — |
| B0 A1-only | 0.7359 | +0.066 | — | — | — |
| S1 per-class logreg | 0.7910 | +0.1211 | yes | — | — |
| S2 with class_id | 0.3376 | −0.3323 | no | 0.9953 | **0.66** |
| **S2A no class_id (Fork 1)** | **0.6419** | **−0.0280** | **no (all 5 seeds neg)** | 0.9385 | **0.30** |

S2A is *not* catastrophic the way S2-with-class-id was — its OOF AUC is
sign-stably 0.028 below BP across all 5 seeds, with a much smaller (but
still substantial) 0.30 in-sample/OOF gap. But it still fails the gate,
and it cannot beat A1-alone (0.7359) — adding B1 and Proto information
in a class-agnostic way actively dilutes A1's signal on this substrate.

**Three independent stackers, three failure modes on the same 708-row
substrate:**

1. **S1 per-class logreg** — passes local with +0.1211, dies on LB by
   −0.156 (v46 LB 0.775). 0.28 local→LB gap.
2. **S2 global LGBM with class_id** — catastrophic local fail (worse
   than random, OOF 0.34). Class identity poison on 8-site holdout.
3. **S2A global LGBM no class_id** — sign-stable −0.028 below BP locally.
   Class-agnostic blender cannot extract per-class calibration that S1
   could, can only smear A1 with weaker B1/Proto signals.

The unified explanation is **the 708-row OOF substrate is structurally
non-predictive of LB**. The Phase 1.5 finding (BP=0.6699 < A1-alone=0.7359
means production rank fusion *loses* to A1-alone here) was the first
signal of the same structural issue. Three stacker designs from three
different families have now confirmed it.

**D2 verdict: structurally dead.** No further D2-α/β/γ variants will be
attempted. The substrate is the problem, not the stacker family. Killing
the entire D2 track frees us to commit to a different lever in §12.

**D2-α artifacts retained** for potential future diagnostics:
- `four_track/data/d2_alpha_lgbm.txt` (S2A active, 1.38 MB)
- `four_track/data/d2_alpha_lgbm_with_class_id.txt` (S2 forensic, 1.41 MB)
- `four_track/data/d2_alpha_phase2_results.json` (sweep output)
- `four_track/data/d2_beta_s1_coefs.npz` (S1, retained from §9)
- `four_track/data/d2_beta_oofs.npz` (the substrate itself, retained for
  future ablations if we ever revisit the local-vs-LB calibration question)

No Kaggle dataset was created for D2-α (gate failed before upload). The
canonical notebook `birdclef-2026-protossm-postproc` is unaffected — the
D2-β S1 cell remains injected as a no-op-without-coefs cell, which is
how it has been in production since v48 (S1 dataset detached).

---

## 12. §10-pattern pre-trained B1 checkpoint (2026-04-16)

**Context:** D2 is structurally dead (§11). Per §9's sequencing list, the
two live candidates are §10-style B1 pre-trained ckpt and Track C
revival. User selected the B1-ckpt pivot ("ii — B1-ckpt"). Rationale:
B1-ckpt reuses the §10 ProtoSSM side-project infrastructure (local
trainer → Kaggle dataset → guarded load in a single notebook cell,
detach-to-revert safety) and is a smaller infra lift than Track C.

### Hypothesis

In the canonical notebook, **B1 retrains from scratch every submit** at
cell 31b's "final B1 model on all soundscapes" block — 30 epochs of
training inside the 90-min kernel. Three consequences follow:

1. **Seed variance = submit variance.** Each submit draws a different
   random initialisation and (via mixup + OneCycleLR) a different
   optimisation trajectory. The fused output on LB gets noise that a
   frozen pre-trained ckpt would eliminate.
2. **30 epochs may be under-trained for PerceiverIO.** ProtoSSM (§10)
   stayed neutral at extended budget (LB 0.929 vs 0.933, slight regression),
   but it also had a *stronger* in-kernel fit (AUC converges faster
   because of the prototype cosine warm-start from `init_prototypes_from_data`).
   B1 has no such warm-start — it learns its latent bank from zero — so
   the extended-budget payoff *might* be larger than it was for Proto.
3. **Multi-seed averaging should help.** PerceiverIO's learned latent
   bank is a flat loss landscape with many equivalent solutions
   (permutation symmetry on the latents). Averaging state dicts across
   seeds can only improve variance, not hurt it.

The base rate for this class of experiment is lukewarm: §10's ProtoSSM
pre-train result was LB 0.929 — a 0.004 regression against the 0.933
baseline. We're not assuming B1 will be a bigger win; we're trying it
because the cost is small and the three differences above (no warm-start,
heavier variance, averagable latents) plausibly make it a better fit.

### Design

Mirror `src/train_protossm_local.py` exactly:

- **Same data pipeline.** Load `four_track/data/kaggle_perch_cache/full_perch_arrays.npz`
  + `full_perch_meta.parquet`. Reshape to `(n_files, 12, d_input)`. Compute
  `file_groups`, `site_ids_all`, `hours_all` via the same helpers
  (`build_site_mapping`, `get_file_metadata`, filename groupkey).
- **Same arch keys as the notebook.** Use `CFG["b1_perceiver"]` defaults
  baked into `b1_perceiver.py` (`d_latent=256, n_latents=16,
  n_cross_layers=2, n_self_layers=4, n_heads=8, meta_dim=16,
  n_sites=20, dropout=0.3`) so the state dict loads cleanly into the
  `b1_model` that cell 31b instantiates. **Do not change architecture
  here** — any shape change forces an also-edit of `b1_perceiver.py`
  CFG defaults and the guarded-load becomes a two-file change.
- **Extended budget.** `--epochs 200`, `--patience 40`,
  `--seeds 3`, SWA on the last 35 % of epochs, `distill_weight=0.05`,
  `mixup_alpha=0.3`, `focal_gamma=2.0` (same training cfg as in-kernel
  except epochs + patience + multi-seed average).
- **OOF validation on the same 5-fold GroupKFold as ProtoSSM.** Print
  per-seed OOF AUC and mean/std across seeds; decision gate below.
- **Output.** `four_track/models/b1_pretrained/b1_seed{0,1,2}.pt` per-seed
  plus `b1_pretrained.pt` = multi-seed averaged state dict (the actual
  ckpt we upload).

### Decision gate (before Kaggle upload)

OOF macro AUC for B1 pre-trained multi-seed-avg must be **≥ the 30-epoch
in-kernel baseline** on the same GroupKFold splits. Two ways to measure:

1. **Rigorous**: run the in-kernel 30-epoch B1 OOF once using
   `run_b1_perceiver_oof` verbatim (seed 0), record it, then compare.
   Costs 1 run.
2. **Cheap fallback**: compare multi-seed-avg 200-epoch OOF vs the best
   single-seed 30-epoch OOF within the same sweep. If multi-seed-avg
   gives ≥ +0.002 over the 30-epoch baseline-like run, proceed.

If the multi-seed-avg OOF is **worse** than the 30-epoch baseline, do
not upload. That would mean extended training is actively degrading B1
(plausible — PerceiverIO without a warm-start can collapse latents if
over-trained on 59 files). In that case, pivot to Track C revival in §13.

### LB probe protocol (if gate passes)

1. `kaggle datasets create -p models/b1_pretrained`
   with slug `stevewatson999/birdclef-2026-b1-pretrained`.
2. Add `stevewatson999/birdclef-2026-b1-pretrained` to
   `jupyter/protossm-postproc/kernel-metadata.json:dataset_sources`.
3. Wire a guarded load in cell 31b: after the
   `b1_model = PerceiverIOHead(...)` instantiation and **before**
   `train_b1_perceiver_single(b1_model, emb_files, ...)`, check
   `MODE == "submit"` + candidate ckpt paths, `load_state_dict` if
   found, skip the in-kernel final-retrain, else fall through to the
   existing retrain. Exact mirror of cell 32's proto guard
   (notebook lines ~3317–3340).
4. Submit once. Compare LB to 0.933 baseline (current best is the
   post-§10-ProtoSSM-pretrain run at LB 0.929 — base is the pre-§10
   v44/v45 at 0.933).
5. **Revert path**: if LB regresses, detach the
   `birdclef-2026-b1-pretrained` dataset from the kernel. Cell 31b's
   guarded load sees no ckpt and falls through to in-kernel retrain.
   Zero code-side revert required.

### Artifacts and paths

| Artifact | Path |
|---|---|
| Local trainer | `four_track/src/train_b1_local.py` (new, mirror of `train_protossm_local.py`) |
| Per-seed ckpts | `four_track/models/b1_pretrained/b1_seed{0,1,2}.pt` |
| Averaged ckpt | `four_track/models/b1_pretrained/b1_pretrained.pt` |
| Config snapshot | `four_track/models/b1_pretrained/config.json` |
| Kaggle dataset | `stevewatson999/birdclef-2026-b1-pretrained` (create after gate) |
| Notebook cell | `four_track/src/b1_perceiver.py:CELL_31B` (edit to add guard) |

### Kill criterion

If the B1-ckpt LB probe is neutral-or-worse *and* §10's ProtoSSM-ckpt
probe was already neutral-or-worse, that's two §10-pattern experiments
with no payoff. Stop throwing pre-trained ckpts at this notebook and
commit the remaining Kaggle-slot budget to Track C revival in §13.

### Local training result (2026-04-16)

Ran `src/train_b1_local.py --epochs 200 --patience 40 --seeds 3`
(~1 min total on GB10). Artifacts landed at
`four_track/models/b1_pretrained/{b1_seed[0-2].pt, b1_pretrained.pt,
config.json}`. Then ran the 30-epoch baseline with identical arch +
train-cfg except `--epochs 30 --patience 8 --seeds 1` to anchor the
§12 gate.

| Run | OOF macro AUC |
|---|---|
| 30-epoch in-kernel baseline (seed 0) | 0.3430 |
| 200-epoch seed 0 | 0.3403 |
| 200-epoch seed 1 | 0.3514 |
| 200-epoch seed 2 | 0.4480 |
| 200-epoch mean (3 seeds) | **0.3799** (std 0.048) |

Two read-outs from this:

1. **Extended budget alone does nothing.** 200-epoch seed 0 (0.3403) is
   within 0.003 of 30-epoch seed 0 (0.3430) — the extra training did
   not move the needle. Consistent with §10 ProtoSSM pre-train result
   (LB 0.929 vs 0.933 baseline): the in-kernel 30-epoch budget is
   *not* the bottleneck on this dataset.
2. **Multi-seed average reduces estimator variance.** Per-seed spread
   is 0.048, and the 3-seed mean (0.3799) is +0.036 above the
   single-seed baseline (0.3430) — but this is a *smoothing* effect on
   a noisy OOF estimator, not evidence the averaged ckpt will LB
   better. And 0.3799 < 0.5 is still sub-random — the OOF substrate is
   confirmed uninformative, as the `b1_perceiver.py:87-92` comment
   already warned ("only ~59 fully-labeled files → 5 wildly imbalanced
   GroupKFold splits → OOF AUC is uninformative for *any* branch").

Technical note on what the gate measured vs what we ship: the OOF
numbers above come from per-fold training (5 GroupKFold folds). The
ckpt we would upload (`b1_pretrained.pt`) is a **seed-average of
full-data retrains**, which is a different object from the per-fold
OOF predictors. There's no cheap way to OOF the full-data averaged
ckpt without paying 15× the compute (5-fold × 3-seeds × new retrain).

### Decision point

The gate in §12 as written ("multi-seed-avg OOF must be ≥ 30-epoch
baseline") is ambiguous here: the per-seed OOF (0.3403) is not better
than baseline, but the seeds-averaged OOF estimator (0.3799) is. Per
the established wisdom that OOF is structurally uninformative on this
substrate, neither number is load-bearing.

**Three options:**

1. **Burn one LB probe.** Upload `b1_pretrained.pt` as Kaggle dataset
   `stevewatson999/birdclef-2026-b1-pretrained`, wire the guarded load
   into cell 31b (mirror of cell 32 proto guard), submit once. Base
   rate for §10-pattern experiments is neutral-or-worse (§10 proto:
   LB 0.929 vs 0.933 baseline), so expected value is small. Revert
   path = detach dataset, zero code change.
2. **Skip B1-ckpt, pivot to Track C.** Two §10-pattern attempts with
   no upside would constitute the §12 kill criterion. Pivot straight
   to Track C revival in §13 — ProtoSSM-as-teacher pseudo-labels on
   train_audio unlabeled crops, re-train A1 with pseudo labels.
3. **Variant B1 pre-train before uploading.** The current run used
   the exact in-kernel arch + train-cfg to guarantee ckpt compatibility.
   A variant that changes *something* (larger model with retrained cell
   31b, or stronger regularization, or distillation from the ProtoSSM
   OOF predictions) could give a real lift. High cost, harder to keep
   drop-in compatible with cell 31b.

User call required.

**Pessimistic verdict**: this result is strong evidence for option 2.
The §10 ProtoSSM pre-train was neutral-to-bad on LB, and here the
extended budget shows no per-seed signal over the 30-epoch baseline.
The two pieces of evidence rhyme — Perch-consumer models are not
bottlenecked by training time on this dataset. Track C is the lever
that has not been pulled yet.

### Decision (2026-04-16): KILL §12, pivot to §13 Track C

User selected option 2. §12 is closed without an LB probe.

**§12 kill rationale (for the log):**
- §10 ProtoSSM pre-train → LB 0.929 (−0.004 vs 0.933 baseline, slight regression).
- §12 B1 pre-train → extended 200-epoch budget shows no per-seed OOF
  signal over 30-epoch baseline (0.3403 vs 0.3430).
- Two §10-pattern attempts with no upside = the §12 kill criterion fires.
- Probing LB would consume a daily slot at near-zero expected value,
  confirming a base rate we already have from §10.

**Artifacts retained** (local, no Kaggle dataset was created):
- `four_track/models/b1_pretrained/b1_pretrained.pt` — 3-seed averaged
  ckpt from 200-epoch training. Architecturally drop-in for cell 31b.
  Keep on disk as a reference for a future "variant B1" experiment if
  we ever revisit this lever with a different arch or distillation
  target.
- `four_track/models/b1_pretrained/b1_seed{0,1,2}.pt` — individual seeds.
- `four_track/models/b1_pretrained/config.json` — arch + train cfg +
  OOF numbers.
- `four_track/src/train_b1_local.py` — the trainer itself. Kept as a
  working mirror of `train_protossm_local.py` for any future B1
  variant run.

No code changes landed in `b1_perceiver.py`, `jupyter/protossm-postproc/`,
or Kaggle dataset sources. The canonical notebook still retrains B1
from scratch every submit at `CFG["b1_frozen_weight_submit"] = 0.10`,
exactly as it did before §12 began.

---

## 13. Track C revival — ProtoSSM-as-teacher pseudo-labels on train_audio (2026-04-16)

**Context:** §12 killed per user decision. Per §9 sequencing, Track C is
the remaining live candidate. Purpose of this section: scope it honestly
before writing any code, because the original Track C was KILLED on
2026-04-09 for a structural reason that still applies today.

### Hard prereq: Kaggle-extracted focal-clip Perch features

The 2026-04-09 kill rationale (see §Track C / "C2 implementation:
local-port path" above, lines 278–381):

> Locally-trained Perch consumers regress on Kaggle by ~0.005–0.020 LB,
> because the local ONNX Perch extraction produces embeddings that are
> subtly different from the Kaggle-runtime Perch extraction. Any
> Perch-consuming model trained on one will underperform on the other.

This constraint is immutable and still applies. The **only viable
revival path** (called "B-redo" on line 281) is:

1. Run a one-shot Kaggle kernel that extracts Perch v2 features for all
   35,549 `train_audio` focal clips using the *same* in-kernel Perch
   call-path that the postproc notebook uses on soundscapes.
2. Save the extracted features as a Kaggle dataset
   (`stevewatson999/birdclef-2026-train-audio-perch` or similar),
   analogous to how `jaejohn/perch-meta` provides the soundscape cache.
3. Use those Kaggle-consistent features for all subsequent Track C work
   (teacher scoring → pseudo-labeling → student retrain), either on a
   Kaggle kernel or locally — but the features themselves must originate
   from Kaggle, not local ONNX.

**Infra cost estimate**: ~1 day to build the extraction kernel (it's a
slimmed-down version of the postproc notebook's cell 5, applied to
`train_audio` instead of `train_soundscapes`). The Kaggle CPU runtime
for extracting 35K clips is the binding uncertainty — a smoke-test on
~100 clips will tell us total wall time.

**Local artifacts that exist but cannot ship** (per §Track C kill):
- `data/processed/perch_train_audio_c2/` — 1.8 GB local Perch v2 cache
  over 35,549 focal clips. Known to have the embedding mismatch.
  **Useful as a regression-test reference**: once the Kaggle-extracted
  cache exists, we can compare a handful of clips between the two
  caches to quantify the embedding drift and sanity-check the Kaggle
  kernel produced sensible features.
- `four_track/src/extract_train_audio_c2.py` — the local extractor.
  Not useful as-is for Kaggle; ONNX runtime on Kaggle has different
  quirks. But the clip-iteration / window-slicing logic in this file
  is directly portable to a Kaggle kernel.

### §13 design (phased)

#### Phase 1 — Build the Kaggle Perch extraction kernel (~1 day)

- New kernel: `jupyter/perch-train-audio-extract/` (mirror of how
  `jupyter/protossm-postproc/` is laid out).
- Core code: port the postproc notebook's cells 2 (Perch model load),
  3 (ONNX path resolution), 5 (clip → 5s windows → Perch embedding +
  logits). Apply to `train_audio/` instead of `train_soundscapes/`.
- Output: `full_train_audio_perch.npz` with
  `{emb: (N, 1536), scores: (N, 234)}` + `full_train_audio_meta.parquet`
  with `{row_id, stem, species, window_idx, clip_duration_sec}`. Same
  schema as `jaejohn/perch-meta` so downstream code treats it uniformly.
- Smoke test: run on first 100 clips, verify extraction completes and
  the output schema matches. Bail early if Kaggle wall-time projection
  exceeds the 9-hour CPU cap.
- Ship: `kaggle datasets create` →
  `stevewatson999/birdclef-2026-train-audio-perch` (private).

**Phase 1 gate**: dataset uploaded, embeddings for ≥34,000 / 35,549
clips. If the kernel OOMs or times out on full extraction, split into
species-partitioned jobs and run the kernel multiple times (each output
becomes a dataset version).

#### Phase 2 — Teacher scoring + pseudo-labeling (~0.5 day)

Run locally (CPU or GPU) on the Kaggle-extracted focal features:

- Teacher: the **current in-kernel ProtoSSM** (cell 31's final retrained
  model). This requires a small Kaggle kernel to dump its state dict
  after the retrain — OR reuse §10's `birdclef-2026-protossm-pretrained-v2`
  ckpt (already Kaggle-compatible). Probable choice: use the §10 ckpt
  to avoid another Kaggle slot.
- Output: per-clip `{max_conf, pseudo_soft_labels (234,), top3_labels}`
  stored as parquet + npz at
  `four_track/data/processed/c2_pseudo_labels_kagglefeat/`.
- Filter gate: `(max_conf > 0.6) ∧ (primary_label ∈ top3)` — same as
  the 2026-04-09 C2 pipeline. Expected retention rate ~95 % (the prior
  local run retained 33,597 / 35,549).

#### Phase 3 — Student retrain on Kaggle-consistent features (~0.5 day)

Retrain a ProtoSSM student on the union of:
- 59 soundscape files (from `jaejohn/perch-meta`, same as current
  postproc cell 31).
- ~33K filtered focal clips (from the new Phase-1 dataset).

Student arch = current ProtoSSM (keeps drop-in compatibility with
notebook cell 31b if we decide to ship it). Train locally with multi-seed
averaging, mirroring `src/train_protossm_local.py` exactly.

**Output**: `four_track/models/c2_student/c2_student_pretrained.pt`
(multi-seed avg).

#### Phase 4 — Notebook integration + LB probe (~0.25 day)

- Upload `c2_student_pretrained.pt` as
  `stevewatson999/birdclef-2026-c2-student`.
- Wire guarded load in cell 32 of postproc notebook (mirror of §10's
  proto ckpt guard), but **instead** of loading into the fresh-init
  ProtoSSM, load into a parallel "student model" that feeds a second
  rank-fusion cell (say cell 36c, after 36b/B1). Or, simpler, **replace**
  the §10 proto pretrained ckpt entirely and measure.
- One LB submit.

### Decision gates

| Phase | Gate | If fail |
|---|---|---|
| Phase 1 | Kaggle kernel extracts ≥34k clips in <9h | partition and retry; if still fails after 2 attempts, kill §13 |
| Phase 2 | Pseudo-label retention ≥ 80% (28k+ clips after filter) | loosen filter or kill — too few surviving clips isn't worth the student-train cost |
| Phase 3 | Student OOF on 59-file soundscape ≥ teacher OOF (same substrate, known-uninformative but sign-preserving) | student has degraded from teacher — kill or rerun with different focal:soundscape mix ratio |
| Phase 4 | LB ≥ current production baseline 0.933 | revert (detach c2 dataset), declare Track C dead for this competition |

### Kill criterion

If any of Phase 1–3 fails hard, or if Phase 4 LB is a net-negative, declare
Track C dead. The remaining live track is A3 (second SED backbone — ~2
days local + ONNX). See §3 for A3 framing.

### Why this is the right next pivot

- D2 (§9, §11): dead. 708-row substrate non-predictive, three stackers tried.
- A2 (§A2 elsewhere in plan): killed at LB 0.926 (SED on audio only).
- §10 Proto pretrain: LB 0.929 (slight regression).
- §12 B1 pretrain: local kill (OOF uninformative).
- Track C: not yet attempted with Kaggle-consistent features. The
  2026-04-09 kill was infrastructural, not substantive — the hypothesis
  (pseudo-labeled focal clips add class coverage that soundscape-only
  training misses) was never falsified on real data.
- A3: feasible but bigger lift (~2 days). Good fallback if §13 Phase 1
  shows Kaggle extraction is unworkable.

### First concrete action

Build and smoke-test the Phase-1 Kaggle extraction kernel. Subtasks:

1. Create `jupyter/perch-train-audio-extract/` dir with
   `kernel-metadata.json` mounting `birdclef-2026` +
   `google/bird-vocalization-classifier/.../perch_v2_cpu/1` + `ashok205/tf-wheels`.
2. Port `extract_train_audio_c2.py` clip-iteration logic to a notebook
   cell that uses Kaggle's in-kernel Perch call-path (same as postproc
   cell 5). Start with smoke-test mode (first 100 clips) gated behind
   an env var.
3. Push kernel, run smoke, verify schema + wall-time projection.
4. If wall-time projection < 6h, run full extraction. If 6–9h, split
   into two species-partitioned runs. If > 9h even partitioned, bail.

Confirm before pushing anything to Kaggle — Phase 1 creates a publicly
visible (well, private-to-account) Kaggle dataset.

### Phase 1 execution log (2026-04-16)

Kernel: `stevewatson999/birdclef-2026-perch-train-audio-extract`
(`jupyter/perch-train-audio-extract/`). All versions SMOKE mode (first
100 clips) except as noted.

| ver | runtime | backend | outcome | rate (clips/s) | full projection |
|---|---|---|---|---|---|
| v1 | CPU | TF 2.20 (SavedModel) | brucewu dataset silently did not mount → TF fallback | 0.18 | 54.81 h |
| v2 | GPU | — | fail: `XlaCallModuleOp version 10 not supported` on Kaggle's TF 2.19 GPU image (perch_v2/2 is JAX-exported) | — | — |
| v3 | GPU | — | fail: same XlaCallModule error even with perch_v2_cpu/1 (also JAX-exported, different serialization) | — | — |
| v4 | CPU | TF 2.20 (SavedModel) | brucewu still did not mount despite being in `dataset_sources` (unresolved Kaggle-side mount mystery) | 0.18 | 54.81 h |
| v5 | CPU | **ONNXRuntime** | own ONNX dataset (`stevewatson999/birdclef-2026-perch-onnx`, 418 MB: `perch_v2_no_dft.onnx` + wheels/) mounted → ONNX path hit | **0.67** | **14.81 h** (3.7× speedup) |

**Key findings:**
- Kaggle GPU runtime is unusable for Perch v2 (both SavedModels are
  JAX-exported XlaCallModule v10; only TF 2.20+ deserializes, but
  installing TF 2.20 into the GPU image breaks CUDA bindings). Must use
  CPU runtime + ashok205/tf-wheels TF 2.20 wheels.
- brucewu's dataset `birdclef-2026-cvlb-assets-0911` is CC0-public
  and accessible via `kaggle datasets files` from our account, but
  consistently fails to mount under `/kaggle/input/` in our kernel runs.
  Worked around by re-uploading the needed ONNX + wheels as our own
  `stevewatson999/birdclef-2026-perch-onnx` dataset.
- ONNXRuntime 1.24.3 + deps wheels bundle: ~24 MB. Full dataset: 418 MB.
- 14.81 h > 9 h CPU slot cap → **3-partition split chosen** (≈5 h/run,
  zero-risk headroom; 2-partition was 7.4 h/run — too close for
  long-recording tail risk).

**Partition scheme** (v6 / v7 / v8):
- Clips sorted alphabetically by species path via `rglob("*.ogg")`
  then `sorted()`. This keeps each partition species-balanced.
- Keep clip i iff `i % 3 == PARTITION_ID` (0, 1, 2). No need for
  species-name hashing — alphabetical-modulo is deterministic and
  already interleaves species.
- Outputs: `full_train_audio_perch_p{0,1,2}of3.npz` +
  `full_train_audio_meta_p{0,1,2}of3.parquet`. Merge locally into
  `birdclef-2026-train-audio-perch` dataset after all 3 complete.

**Gate refinement**: v5 confirmed schema + call-path. The Phase 1 gate
(≥34k / 35,549 clips extracted) now reduces to: all three partitions
complete in <9h each, merged total ≥34,000 rows at expected window
schema `(N, 1536)` emb + `(N, 234)` scores.

### Phase 1 outcome (2026-04-17 12:14 UTC)

Autopilot `scripts/perch_extract_autopilot.sh` (PID 326614, started
2026-04-16 23:55) ran v6→v7→v8 extractions back-to-back, merged
locally, and uploaded the dataset cleanly. **Phase 1 gate PASSED.**

| metric | value |
|---|---|
| partitions extracted | v6 / v7 / v8 (all `[done]`) |
| merged rows | 250,719 windows |
| unique stems | **35,549 / 35,549** (100%) |
| emb shape | `(250719, 1536)` float32 |
| scores shape | `(250719, 234)` float32 |
| NPZ size | 1,619.9 MB |
| parquet size | 2.15 MB |
| dataset | `kaggle.com/datasets/stevewatson999/birdclef-2026-train-audio-perch` |
| local staging | `kaggle_datasets/train-audio-perch/` |

### Phase 2 execution log (2026-04-17 14:15 UTC)

Teacher scoring + pseudo-labeling on the Kaggle-consistent Perch
features. Runner: `four_track/src/c2_pseudo_label_kagglefeat.py`.

- **Teacher**: §10's `protossm_pretrained.pt` at
  `four_track/models/protossm_pretrained_v2/` (loaded with
  `strict=False` to ignore `family_head` keys — not needed for
  inference). Config per `config.json`: d_model=320, n_ssm_layers=4,
  cross_attn_heads=8, ~5.77 M params.
- **Window handling**: focal clips have 1–120 windows (mean ~7).
  ProtoSSM's `pos_enc` is sized (1,12,d), so T>12 would error. Solved
  by splitting each clip into ⌈T/12⌉ 12-window chunks with zero-padding
  and a validity mask; per-class max is taken over sigmoid(logits) of
  valid windows only across all chunks.
- **Site/hour metadata**: passed `None` (focal clips have no site
  identity); teacher's `site_emb`/`hour_emb` are skipped on the
  None-branch of forward().
- **Wall time**: 18 s end-to-end on GPU (GB10, cuda-cap warning but
  no numerical failure). Negligible cost.

**Phase 2 gate (≥80% retention): PASS (94.3%).**

| metric | value | notes |
|---|---|---|
| stems scored | 35,549 | full Phase-1 coverage |
| retained | **33,516 (94.3%)** | filter: `(max_conf>0.6) ∧ (primary_label∈top3)` |
| prior local-extract retention | 33,597 (94.5%) | delta −0.23% — essentially identical to the pre-Kaggle-consistency run |
| species unmapped | 0 | all 206 training species are in the 234-class space |
| mean `max_conf` | 0.991 | teacher is very confident on focal clips (expected; §10 was trained on soundscape-only, so focal distribution is in-domain-but-easier) |

**Artifacts** at `four_track/data/processed/c2_pseudo_labels_kagglefeat/`:
- `pseudo_soft_labels.npz` — `{stems: (35549,) <U32, species_id:
  (35549,) <U16, soft_labels: (35549, 234) float32}`.
- `pseudo_labels.parquet` — `{stem, species_id, primary_label_idx,
  max_conf, top1, top2, top3, retained}`.
- `summary.json` — run metadata.

**Key finding**: 33,516 retained vs the 33,597 the 2026-04-09 local-
ONNX run retained is a delta of only 81 clips (0.23%). This confirms
the teacher's behavior on Kaggle-consistent vs local-ONNX features is
essentially identical *at the pseudo-label level*, even though we know
the raw embeddings differ enough to cost ~0.005–0.020 LB when training
on one and evaluating on the other. The pseudo-label filter is
robust to that drift, so Phase 3's student will train on a set that's
(a) Kaggle-consistent end-to-end and (b) near-indistinguishable in
membership from what local-ONNX would have produced.

---

### Phase 3 execution log + KILL (2026-04-17 15:55 UTC)

**Outcome: Track C KILLED. The §10 teacher is the Track-C stacking model.**

Phase 3 `src/c2_student_train.py` was drafted and iterated through four
recipes. Every real gradient step degraded the teacher-init ceiling.

| Run | Recipe | Best val_auc | Notes |
|---|---|---|---|
| 1 | 50/50 mini-batch focal+sc, lr=3.2e-3, rand init | 0.8806 @ ep 2 | Peaked ep 2, degraded to ~0.85. |
| 2 | Soundscape-only, lr=3.2e-3, rand init | 0.9038 @ ep 3 | Confirmed focal contamination but still below teacher. |
| 3 | Soundscape-only, lr=8e-4, rand init | 0.9151 @ ep 1 | Lower lr peak; still ep-1 peak + degrade. |
| 4 (sanity) | Teacher-init, no focal, lr=2e-4, full-batch | **0.9734 @ ep 1** | Gate PASS by +0.0001 — pure teacher inheritance, no real gain. |
| 5 | Teacher-init + focal (64/step @ w=0.25), full-batch | **0.9734 @ ep 1** | Same ceiling; focal dragged AUC to 0.85-0.90 for mid-run, recovered to 0.9052 by ep 50, never beat ep 1. |

**Root cause:** The teacher already extracts all signal this val split
exposes; pseudo-labels derived from the teacher cannot add information it
doesn't have (standard self-distillation ceiling). Focal distribution
shift actively degrades weights tuned on soundscape — focal loss plateaus
at 3.4–4.7 while soundscape loss is 0.6, the model can't fit both.

**Asymmetry also discovered:** the initial draft used mini-batch (523
steps/epoch) with random init + no site/hour metadata vs teacher's
full-batch (1 step/epoch) with `init_prototypes_from_data()` + metadata.
Once the draft matched the teacher recipe exactly (run 4/5), the ceiling
was still the teacher.

**Per §13 kill criterion:** "student OOF < teacher OOF → kill or rerun
with different focal:soundscape mix ratio." Tried mix=0, mix=50/50,
mix=1:(47/64) via separate focal pass — every configuration degrades.
There is no focal mix ratio where the student exceeds the teacher.

**Decision:** Use the §10 pre-trained ProtoSSM (`models/protossm_pretrained_v2/protossm_pretrained.pt`)
directly as the Track-C model for stacking. No student ckpt will ship.
Redirect remaining Kaggle-slot budget to Tracks A/B/D.

**Artifacts to retain:**
- `src/c2_pseudo_label_kagglefeat.py` + `data/processed/c2_pseudo_labels_kagglefeat/`
  (pseudo-labels may be useful if a future track revisits self-training).
- `src/c2_student_train.py` + sanity ckpts at
  `models/c2_student_teacherinit_{no,}focal/` (reference for future
  teacher-ensemble work; not for submission).

---

## 14. Leader analysis + proposed next-phase levers (2026-04-17 16:30 UTC)

**Context.** After Track C (§13) was killed and D1-b was verified as a
mathematical no-op (rank-then-average in `a1_notebook_cell.py:203-206`
kills any monotonic recalibration), the §13 pickup anchor defaulted to
"Track A3 per §9 sequencing" — but Track A3 was already **ABANDONED
2026-04-10** after three attempts (see §2 Track A "A3 decision" at line
575). That made the next-action pointer stale.

This section does a **fresh external leader analysis** to identify
techniques the 0.951 public leader (vs our 0.933 production) is likely
using that our current plan does not cover, then proposes a ranked set
of next-phase levers for the user to pick from on return.

Research was done 2026-04-17 16:00-16:30 UTC via web search + fetch of
public BirdCLEF 2024/2025 writeups (BirdCLEF-2026 writeups don't exist
yet — competition is mid-flight, entry deadline 2026-05-27). The 2025
playbook is the closest available proxy because the 2025 task shared
the same soundscape/focal structure and had the same 90-min CPU
inference cap.

### 14.1 What top teams did in BirdCLEF 2024/2025

Sources (all consulted 2026-04-17):

- **1st 2025, Nikita Babych — "Multi-Iterative Noisy Student Is All
  You Need"** (writeup title only was retrievable; Kaggle writeup pages
  render client-side, the body wasn't in fetch output — but the title
  alone is the core technique disclosure).
- **2nd 2025, VSydorskyy — public repo** at
  `github.com/VSydorskyy/BirdCLEF_2025_2nd_place` (full README fetched).
  Public 0.925 / private 0.928.
- **38th 2025, Max Melichov — Medium writeup** (full post fetched).
  Public ~0.902. Most transparent LB-lift-per-lever breakdown publicly
  available for the 2025 task.
- **1st 2024, yuto_mo writeup (RegNet + EffB0 ensemble, min-reduction)**,
  Zenn article.
- **3rd 2024, Theo Viel + jfpuget repos** — two-level pseudo-label
  distillation, EffViT-B0 + MNasNet ensemble.
- **STSG paper (DS@GT 2025, arxiv 2507.08236)** — novel but too weak
  alone (LB 0.559), noted for completeness.

Recurring techniques across these writeups, ranked by how often they
show up in top-10 solutions:

| Rank | Technique | Evidence |
|---|---|---|
| 1 | **Iterative pseudo-labeling (noisy student loop)** on unlabeled soundscapes | 1st 2025, 2nd 2025, 3rd 2024, jfpuget 2024 |
| 2 | **Diverse backbone ensemble** (EffNetV2-S, ECA-NFNet-L0, EffViT, MNasNet, RegNet) | 2nd 2025, 3rd 2024, 1st 2024, all |
| 3 | **Multi-year pretraining** (BirdCLEF 2021-2024 corpus → fine-tune) | Max Melichov +0.009, 2nd 2025, 3rd 2024 |
| 4 | **External data** (Xeno-Canto, iNaturalist, CSA) for rare species | 2nd 2025, 3rd 2024 |
| 5 | **Quantile-Mix / min-reduction ensemble** (rank-avg + mean blend, or min of sigmoids) | Max +0.025 over plain mean, 1st 2024 |
| 6 | **Pseudo-label filtering** (F2 thresh 0.5, min_target 0.1, min_instance 0.4) | 2nd 2025 |
| 7 | **Silero-VAD to strip human speech from train_audio** | Max Melichov |
| 8 | **OpenVINO/fp16 inference + model compression** | 2nd 2025, 1st 2024 |
| 9 | **10-second windows with label-averaging of 2×5s** | 1st 2024 |
| 10 | **"nocall" class added as extra output** | 1st 2024 |

### 14.2 Gap analysis vs our stack

For each of the top-10 techniques above, what is our current coverage?

| # | Technique | Our coverage | Gap |
|---|---|---|---|
| 1 | Iterative noisy student | §13 tried **self**-distillation (ProtoSSM→ProtoSSM on focal). Killed. **We have NOT tried cross-architecture noisy student** (e.g. ProtoSSM→A1 EffB0 or A1→ProtoSSM). | **Large gap** — this is the 2025 winning lever and our kill memo only rules out self-distillation, not cross-arch. |
| 2 | Diverse backbone ensemble | A1 EffB0 (live), B1 PerceiverIO (live), ProtoSSM (live). 3 backbones. Top teams use 5-10. A3 abandoned on the A1 pipeline recipe but **NOT abandoned with different data/pretraining**. | Medium. |
| 3 | Multi-year pretraining | Not attempted anywhere in our plan. | **Large gap** — Max cited +0.009 single-model, 2nd-place 2025 cited as core lever. |
| 4 | External data (XC/iNat) | Not attempted. | Medium. |
| 5 | Quantile-Mix / min-reduction | We use **rank-average** (not Quantile-Mix α=0.5 blend, not min-reduction). | Small-medium — Quantile-Mix is a 1-line swap. Low cost, unknown lift. |
| 6 | Pseudo-label filtering thresholds | §13 Phase 1 used ProtoSSM teacher scores directly (no F2/confidence filter). Killed for other reasons. | Small. |
| 7 | Silero-VAD | Not attempted. train_audio quality assumed clean. | Small-medium. |
| 8 | OpenVINO/fp16 | **Not applied.** A1 JIT is fp32 torch-scripted. | No direct LB lift, but buys runtime to add more models. |
| 9 | 10s windows | A1 uses 5s clips matching 2025 baseline. | Small — 1st-place 2024 recipe, neutral on 2025 per Max Melichov. |
| 10 | "nocall" class | Not attempted. | Small. |

### 14.3 Important caveat on the 0.951 leader

The 2025 public #1 finished at **0.929** public / private ~0.92.
2026-04-15 memory says public #1 is **0.951** — already +0.022 above
the strongest known 2025 public solution. Plausible explanations:

1. **Scale difference:** the 2026 Pantanal task has a different class
   mix (65 species? confirm via `BirdCLEF_2026_metadata_exploration`).
   ROC-AUC is not comparable across years. 0.951 in 2026 ≠ 0.951 in
   2025 in difficulty terms.
2. **Public-LB overfit:** some 2024 leaders at 0.74 public dropped to
   0.69 private. The 0.951 may shrink on private. Plan conservatively.
3. **Someone applied noisy student + multi-year pretraining + diverse
   backbones simultaneously.** This is the most likely explanation by
   weight of evidence from 2025 writeups.

**Implication for gap target:** Do not take "+0.018 vs 0.951" as the
gate. Take "match 2025 1st-place technique stack" as the gate. If after
applying techniques 1+2+3 we're at ~0.940-0.945 LB, that's a successful
outcome regardless of where 0.951 ends up privately.

### 14.4 Ranked proposed levers

Constraints applying to every candidate:

- **Embedding-mismatch hazard** (per `feedback_regrep_constraints`):
  anything that consumes Perch v2 features must be trained on Kaggle or
  via Kaggle-extracted cache. Local Perch ≠ Kaggle Perch. This rules out
  local pseudo-labeling of anything that feeds B1 or ProtoSSM.
- **Ensemble-strength constraint** (§1): a new model branch must score
  **≥0.92 LB standalone** to help — weak branches regress (#28C, #29).
- **Slot budget**: 5/day, ~40 days until deadline = ~200 slots, but
  slots are cheap, informativeness is not. One LB probe per lever max.

| # | Lever | Expected lift | Cost (days) | Risk | Kill gate |
|---|---|---|---|---|---|
| L1 | **Cross-arch noisy student: ProtoSSM teacher → A1 EffB0 student on train_audio pseudo-labels (multi-label)** | +0.005 – +0.015 | 3-4 | MED-HIGH. Related to killed §13 but structurally different — student ≠ teacher. A1 can learn from ProtoSSM signals it doesn't itself extract. | A1 fold-0 val_roc_auc on soundscape held-out must beat the frozen A1 baseline (0.7414) by ≥+0.01, OR LB ≥0.935. Below either → kill. |
| L2 | **Multi-year pretraining: A1 EffB0 pretrained on BirdCLEF 2021-2024 → fine-tune on 2026** | +0.005 – +0.010 | 4-5 | LOW. Max Melichov cited +0.009 on 2025; easy to re-run path. Expensive only in data-download + train time. | A1 fold-0 val must beat baseline 0.7414 by ≥+0.005. Below → kill (pretraining is known good, failure = broken setup). |
| L3 | **Quantile-Mix ensemble blend (α=0.5 mean + rank)** replacing our pure rank fusion | +0.001 – +0.003 | <1 | LOW. One-line notebook change. Max cited +0.025 for this over plain mean — but he was mixing 3+ CNNs + SED; our A1+B1+ProtoSSM already rank-fused, delta likely smaller. | LB ≥ 0.934. Below → revert. |
| L4 | **Diverse backbone via A3-revival with multi-year pretraining**: ECA-NFNet-L0 (A3-v3 best at val 0.7458) re-trained with multi-year pretrained init | +0.002 – +0.008 | 3-4 | HIGH. A3 was abandoned at fold-0 val ceiling 0.7458. Different pretraining *might* break the pipeline ceiling hypothesis; no guarantee. | A3 fold-0 val must clear 0.77 gate (the original A1 gate). Failure = re-abandon. |
| L5 | **External data (Xeno-Canto) for Pantanal species** — download Xeno-Canto recordings for the 234 species, add to A1 focal training corpus | +0.002 – +0.008 | 2-3 | MED. Data-licensing, quality filtering, species-code mapping. 2nd-place 2025 used this. | A1 fold-0 val must beat baseline 0.7414 by ≥+0.005. |
| L6 | **OpenVINO fp16 inference for A1** — reduce A1 inference time, use freed budget for a 2nd backbone | 0 (indirect; enables L4 under budget) | 1-2 | LOW. Proven technique in 2nd-place 2025 solution. | Notebook wall time must stay <85 min. |
| L7 | **Silero-VAD speech cleaning on train_audio** — strip human narration before A1 retrain | +0.001 – +0.005 | 1 | LOW. Additive to L1/L2 if they happen. | A1 fold-0 val must match or beat baseline. Below → revert. |
| L8 | **"nocall" synthetic class** for A1 — add background-only segments as 235th class during training | 0 – +0.003 | 1 | LOW. 1st-place 2024 recipe. Small delta expected because our soundscape val isn't no-call dominated. | A1 fold-0 val must match or beat baseline. |

**Not recommended** (dominated by above):

- **Revisiting D1-b (per-fold temperature)** — verified no-op,
  monotonic-invariant (see 2026-04-17 Option-A exploration).
- **Re-opening §13 Track C / C2 student** — `project_track_c_killed.md`
  memory. Self-distill ceiling is real; don't retry mix ratios.
- **Retrying A2 (train_soundscape + focal)** — killed on contaminated-val
  selection. Would need clean-val redesign first; L1 covers the same
  "use pseudo-labels on A1" goal with a cleaner setup.
- **New post-proc knob sweeps** — §11/D2 and ancestors exhausted the
  knob. LB 0.775 regression (v46) and LB 0.929 neutral (v47) are
  evidence.

### 14.5 Recommended sequencing

**First concrete action: L2 (multi-year pretraining) before L1 (noisy student).**

Rationale:
1. L2 is **unconditionally good**: multi-year pretraining is known
   positive from two independent 2025 writeups. Even if L1 later
   kills, L2 survives. L1 has a non-trivial chance of falling into
   the same distribution-shift trap that killed §13 even with a
   different student arch — the focal-vs-soundscape gap is real.
2. L2's pretrained A1 becomes the **better student** for L1 when we
   run it. Doing L2 first makes L1 cheaper and more likely to pass.
3. L2 is self-contained: download BirdCLEF 2021-2024 training data
   (Kaggle datasets, all public), retrain A1 EffB0 pipeline, fine-tune
   on 2026. No new infrastructure, no Kaggle-side work until the LB
   probe.

After L2 → LB probe. Then:

- **L2 passes (LB +≥0.002)**: Launch L1 (noisy student with the
  multi-year-pretrained A1 as student).
- **L2 neutral (LB ±0.002)**: Launch L1 from current A1 anyway; the
  pretraining was cheap enough to treat as a no-op and move on.
- **L2 regresses**: Freeze current A1, debug recipe drift, skip to L3
  (Quantile-Mix) as a cheap independent test while diagnosing.

**Second action: L3 (Quantile-Mix blend)** — because it's a <1-day
notebook-only change with a well-defined kill gate, and it can run in
parallel with L2's training.

**Third action: L1 (cross-arch noisy student)** only after L2 is
either locked or rejected.

**Parked for now:** L4/L5/L6/L7/L8 — all are +0.001-+0.008 contenders
that can be queued after L1/L2/L3 land. Do not launch in parallel; the
GB10 serializes local training.

### 14.6 First concrete action (detailed)

**L2: Multi-year A1 pretraining.**

Prereq checks (do these first, in order):

1. **Confirm 2026 class list ⊆ 2021-2024 class list**. Compare
   `taxonomy.csv` (2026 234 species) against the union of 2021-2025
   class lists. Species not in the historic list will be zero-signal
   in pretraining; that's fine but worth knowing. If >50% overlap,
   pretraining is worthwhile; if <30%, reassess.
2. **Disk budget**. BirdCLEF 2021-2024 train_audio is roughly 60-90 GB
   total. Confirm `df -h` on GB10 can hold it.
3. **No conflict with §10/§13 side-project**. §10 teacher is frozen;
   this touches A1 EffB0 only, independent file tree.

**Prereq findings (2026-04-17 17:00 UTC):**

Three structural constraints discovered during prereq check that
reshape L2 expected lift from the §14.4 table's "+0.005 – +0.010" to
**+0.002 – +0.006**:

1. **2026 taxonomy is multi-class**, not bird-only:
   - 162 Aves (eBird-style labels: `ashgre1`, `banana`, …)
   - 35 Amphibia + 28 Insecta + 8 Mammalia + 1 Reptilia (iNat
     numeric labels: `1161364`, `22961`, …)
   Historic BirdCLEF 2021-2024 is **bird-only**. Pretraining on it can
   at best help the 162 Aves classes = 69% of 2026 focal classes.
2. **28 species have zero focal training clips** (25 unidentified
   "Insect son01-25" + 3 rare amphibians). No pretraining helps these;
   they need soundscape-only supervision or external data.
3. **Soundscape val (the single selector for A1 training) covers
   only 75 species** — 28 Aves + 25 Insecta + 17 Amphibia + 4
   Mammalia + 1 Reptilia. **Only 37% of val-visible species are
   birds.** A pretraining boost that helps only Aves is masked on the
   selector metric by 63% non-bird val classes. This does NOT mean
   the LB won't move — LB is private and may over-weight Aves — but
   local val is a poor predictor of LB lift for a bird-only pretrain.

**Additional finding** (affects L5 in §14.4): Xeno-Canto is already
**65% of our 2026 training corpus** (23,043 XC + 12,506 iNat = 35,549
total). 2nd-place 2025's "add Xeno-Canto" lever is already implicitly
in 2026's supplied data. Revise L5 expected lift to **+0 – +0.003**;
L5 demoted below L4.

**Revised L2 variant — pretrain on BirdCLEF 2025 only**, NOT
2021-2024:

- **2025 is the only historic year with class-scope match.** 2025 is
  Colombia (Middle Magdalena Valley) → Pantanal species overlap is
  plausibly high. 2025 is also multi-class (birds + amphibians +
  mammals + insects), using the **same iNat-ID taxonomy format** as
  2026 (confirmed by 2025 filename patterns: `train_audio/1139490/…`,
  matching 2026's scheme).
- 2021-2024 are Hawaii / E Africa / India — climatically and
  taxonomically distant from Pantanal, bird-only, eBird-coded. Low
  value per GB and high mapping cost (eBird → scientific_name →
  2026 primary_label).
- Smaller corpus (~30-40 GB vs 60-90 GB for 2021-2024) → faster
  (~6-8h pretrain vs 18-24h).
- Expected lift unchanged at +0.002 – +0.006 because class-scope
  match compensates for smaller corpus size.

**If the cheap 2025-only variant passes (LB ≥ 0.935):** the
full 2021-2024 add-on can be explored as a follow-up. If it fails or
is neutral: skip 2021-2024 entirely.

Training recipe (L2 concrete, revised):

- **Prereq on Kaggle side**: accept BirdCLEF-2025 competition rules
  via the Kaggle UI (10s action — `userHasEntered=False` for 2025 as
  of 2026-04-17). Without this, `kaggle competitions download -c
  birdclef-2025` returns 403.
- Pretraining: BirdCLEF 2025 focal audio at A1's current mel pipeline
  (PCEN+ASL, 5s clips, MixStyle 0.5), on the union of 2025+2026 class
  space (species appearing in 2025's `taxonomy.csv` ∪ 2026's
  `taxonomy.csv`). 10 epochs, `lr=1e-3`, cosine, no fold splits. Save
  `a1_pretrained_2025.pt`.
- Fine-tuning: A1's current 5-fold recipe (25 epochs, ASL+BCE hybrid,
  LR=5e-4, T_0=5), `--init-from` pointed at the pretrained ckpt, only
  the classifier head re-initialized to 234 2026 classes.
- **New script paths**:
  - `src/pretrain_a1_2025.py` (to be written; ~250 lines, forks
    `src/train_a1.py` with 2025 data loader and union-class head).
  - `scripts/pretrain_a1_2025.sh` (launcher).
- **Data prep**:
  - `kaggle competitions download -c birdclef-2025` (after rules
    accept). Extract into `data/raw/birdclef_2025/`.
  - Load `data/raw/birdclef_2025/taxonomy.csv` → build union class
    list vs 2026 taxonomy. Persist the union mapping so fine-tune
    can slice the pretrained head cleanly.
- **Validation**: use 2026 soundscape val (same 1478-segment set as
  A1) as the single selector. Pretraining is pretraining; no CV
  needed.

LB probe protocol when fine-tuning finishes:

1. Export A1 2025-pretrained JITs → replace v3 in Kaggle dataset
   `stevewatson999/birdclef-2026-a1-effb0-ckpts` (back up v3 first per
   `feedback_backup_ckpts_before_overwrite`).
2. Submit notebook v48+ at `A1_WEIGHT=0.20` (current production).
3. Gate: **LB ≥ 0.935** (baseline 0.933 + 0.002 minimum). Below 0.933 →
   revert v3 JITs immediately.

Estimated wall-clock: pretraining ~6-8 h, fine-tuning 25 epochs × 4
folds × ~25 min = ~8-10 h, total **~0.7 - 0.75 days** (down from the
~1.5-2 days of the original multi-year variant).

### 14.7 Explicit non-goals for this phase

- No re-opening of §13 Track C. Memory `project_track_c_killed.md` is
  authoritative; only cross-architecture L1 Noisy Student is in scope,
  and L1 is explicitly downstream of L2.
- No further D1/D2/D3 post-proc exploration — D2-γ, D2-α, D2-β, D1-b
  all exhausted their natural scope.
- No A2-v2 retry. The contaminated-val selection is structural; any
  train_soundscape-as-training-data approach needs a different val
  substrate first.
- No SED-as-primary retry (pre-2026-04 0.769 dead path stays dead).
- No switching backbones away from EffNet-B0 without L2 evidence.

---

## 14.8 Prereq measurements (2026-04-17 17:30 UTC) — L2 demoted

User accepted BirdCLEF-2025 rules. `taxonomy.csv` downloaded.
Species-overlap measurement:

| Class | 2026 total | 2026 species found in 2025 (any key: primary_label / inat_taxon_id / scientific_name) |
|---|---|---|
| Aves | 162 | 40 (24.7%) |
| Amphibia | 35 | 2 (5.7%) |
| Insecta | 28 | 0 (0%) |
| Mammalia | 8 | 1 (12.5%) |
| Reptilia | 1 | 0 |
| **Total** | **234** | **43 (18.4%)** |

The "Colombia-adjacent → high Pantanal overlap" hypothesis was wrong.
Only 18% of 2026 species appear in 2025.

**Revised L2 expected lift: +0.001 – +0.004** (third downward revision):

| Stage | Expected L2 lift |
|---|---|
| §14 initial | +0.005 – +0.010 |
| After class-scope constraint | +0.002 – +0.006 |
| After 18% overlap measurement | **+0.001 – +0.004** |

L2's expected lift is now **below L3's (Quantile-Mix, +0.001 – +0.003,
<1 day, no training).** L2 is demoted to fallback.

**Decision (2026-04-17 17:30 UTC): Pivot to L3 → L1 → L2 order.**

Rationale:
- L3 is dirt cheap (<½ day, notebook-only, 1 LB slot). Certain
  low-value information either way.
- L1 (cross-arch noisy student) is the highest-upside untried lever
  (+0.005 – +0.015). Starting L3 first cashes a cheap win and informs
  L1's blend-integration step.
- L2 kept alive as fallback if L1 and L3 both fail; at that point the
  1-2 day cost is a fair "run out the clock before deadline" bet.

---

## ⏸️ PICK UP HERE — previous (2026-04-18 02:30 UTC — L1 killed, L2 is next lever — SUPERSEDED)

**Starting state** (verified 2026-04-18 02:30 UTC):

- Production LB: **0.931** (A1+B1+ProtoSSM rank fusion via
  `birdclef2026-protossm-postproc` notebook v52 rollback state,
  `A1_WEIGHT=0.20`, `A1_QMIX_ALPHA=0.0`, `A1_VARIANT="baseline"`).
  Bit-identical to v50 LB-0.931 path.
- §13 Track C killed. §10 ProtoSSM teacher is the Track-C stacking
  model (see `project_track_c_killed.md`).
- D1-b verified no-op.
- **§14 sequencing status after kills:**
  - L3 (Quantile-Mix α=0.5) — KILLED 2026-04-17, LB 0.925 (−0.006).
    See `project_l3_killed.md` + §14.8.L3-probe RESULT.
  - L1 (cross-arch noisy student) — **KILLED 2026-04-18, LB 0.930**
    (−0.001). Val 0.9042 vs LB 0.930 — largest val/LB disagreement
    observed (+0.163 val → −0.001 LB). Confirms the teacher
    val-leakage hypothesis empirically. See §14.9.12 + new
    `project_l1_killed.md`.
  - L2 (multi-year pretrain) — **now the next lever**. Expected
    +0.001 to +0.004 (§14.4 + §14.8; 18% species overlap with 2025).
    Modest upside; lower-risk than L1 since it doesn't involve a
    contaminated teacher.
- Notebook rollback complete: cell 41 reads
  `A1_VARIANT = "baseline"  # L1 KILLED 2026-04-18 (LB 0.930 vs 0.931
  baseline); val-leakage confirmed`. `A1_NS_CKPT_DIR` / `A1_NS_FOLDS`
  / conditional swap logic left in place as dormant code (harmless
  when baseline; cheap re-probe path if teacher is retrained clean).
- Kaggle dataset `stevewatson999/birdclef-2026-a1-effb0-ns-ckpts`
  (fold-0 NS JIT) stays live as a no-cost artifact.

**First thing tomorrow — 30-second state check:**

Nothing is running overnight (no autopilot, no training process).
Before any new work, verify state is as described above:

```bash
# 1. No stray training processes (pilot ended 2026-04-17 21:59 EDT)
ps -ef | grep -E "train_a1_noisy_student|train_a1\\b|train_protossm" | grep -v grep
# Expected: empty.

# 2. Latest LB submission should be 0.930 (L1 probe, 2026-04-18 02:26 UTC)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
kaggle competitions submissions birdclef-2026 2>&1 | head -5
# Expected: top row is 0.930 (L1 probe); row below is 0.925 (L3 probe); then 0.931 (baseline).

# 3. Submit notebook is on rollback state
grep -nE 'A1_VARIANT\s*=' jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb
# Expected: one hit containing 'A1_VARIANT       = "baseline"  # L1 KILLED 2026-04-18'
```

If all three match, proceed. If anything differs, investigate before
drafting L2.

**Chosen next action: draft §14.10 (L2 — multi-year pretrain recipe).**

Before any training: this is a recipe draft, NOT a launch. Sketch
the spec in §14.10 with the same rigor as §14.9 (inputs on disk,
script path, hyperparameters, pilot vs full sequencing, rollback,
kill gate). Key open questions for the draft:

1. **Which years to include?** 2021/2022/2023/2024/2025 BirdCLEF
   datasets vary in species overlap with 2026 Pantanal — per §14.8,
   only 18% species overlap with 2025. Decide: (a) pretrain on the
   full multi-year union (maximum data, lowest overlap), (b) restrict
   to species ∩ 2026 (cleaner transfer, much less data), (c) use
   multi-year as pure representation-learning then fine-tune on 2026.
2. **Pretrain → finetune pipeline or co-training?** A recipe that
   first trains on all years then finetunes on 2026 is the standard
   interpretation; check §14.4 row L2 for the specific lever
   description.
3. **Which backbone gets the pretrain?** A1 (EffNet-B0) or B1
   (Perch-consumer). A1 is the L2 target per §14.4.
4. **Data logistics.** Multi-year audio may require another Kaggle
   dataset and local preprocessing pass. Time budget this before
   committing — L2's upside is small, so cost must stay proportional.

**After the recipe is drafted, return for a go/no-go gate
conversation before any training launch.** Do NOT auto-launch L2.

**Do NOT**:
- Launch additional folds (1/2/4) of the L1 NS student.
- Re-try L1 at different pseudo-label thresholds or MixUp settings —
  the teacher itself is the contamination source; no hyperparameter
  tweak fixes that. L1 v2 requires teacher retraining and is below
  L2 in the queue.
- Re-open Track C / §13.
- Edit `src/train_a1.py` or overwrite `kaggle_datasets/a1-effb0-ckpts/`.
- Re-try L3 Quantile-Mix at smaller α.
- Skip the clean-logs step (`rm -f log/*.log`) before any future
  training launch (CLAUDE.md requirement).

#### 14.8.L3-probe Local OOF gate is uninformative (2026-04-17 18:10 UTC)

Prototype script (`/tmp/l3_quantile_mix_probe.py`) run against
`four_track/data/d2_beta_oofs.npz`. Result — all alternatives beat
`prod_fused` on OOF, but the substrate is known-biased:

| Blend                                | Macro AUC | Δ vs prod_fused |
|--------------------------------------|-----------|-----------------|
| prod_fused (baseline)                | 0.6699    | —               |
| A1 ranks alone                       | 0.7359    | +0.066          |
| sigmoid(proto) alone                 | 0.6659    | −0.004          |
| sigmoid(b1) alone                    | 0.3959    | −0.274          |
| QMix(proto,b1,a1) α=0.5              | 0.6612    | −0.009          |
| QMix(proto,b1,a1) α=1.0 (mean only)  | 0.7150    | +0.045          |
| QMix(proto,a1 only) α=0.5            | 0.7325    | +0.063          |
| QMix(proto,a1 only) α=1.0            | 0.7561    | +0.086          |
| Weighted-QMix (prod w) α=0.25        | 0.6773    | +0.007          |
| min-reduce(proto,b1,a1)              | 0.5995    | −0.070          |

Two structural facts make these numbers unusable as a gate:
1. §5 "B1 OOF protocol structurally broken" (2026-04-08) documents a
   0.28 OOF→LB disconnect; ProtoSSM = 0.6468 OOF vs 0.932 LB.
2. `d2_beta_oof_cell.py:252-260` explicitly notes "A1-alone beats
   rank-mean blends on this small substrate" is an artifact of the
   59-file / 71-class / 5-way GroupKFold setup — not a real LB signal.

The probe *does* confirm one real fact: B1 logits are pathologically
left-skewed (AUC 0.40 raw, 0.40 after rank) on this OOF. Production's
inverse-CDF-restored rank-space fusion rescues it. **Do not change B1's
sigmoid→rank handling** — that is load-bearing for the LB-0.933 path.

**Revised L3 plan: skip the local gate, go straight to LB probe.** The
kill gate stays at LB ≥ 0.934 (baseline 0.933 + 0.001). Local OOF can
only detect catastrophic bugs (shape mismatch, NaN), not blend quality.

**Notebook edit applied 2026-04-17 18:35 UTC** (not yet pushed):

Scope — three-point edit to
`jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb`:
- **Cell 1 (Mode switch):** added `L3_TAG = "qmix_a50"` + echo to the
  MODE print line so the LB-run log is identifiable.
- **Cell 2 (CFG dict):** added `"l3_tag": L3_TAG` key.
- **Cell 41 (A1 fusion, "cell 37" position):** replaced the pure-rank
  `_fused_ranks = (1-w)*proto_ranks + w*a1_ranks` block with
  Quantile-Mix `α*prob_blend + (1-α)*rank_blend`, where
  - `rank_blend`  = the prior rank-space formula (unchanged)
  - `prob_blend` = `(1-w)*rescale01(proto_before_fusion) + w*a1_sigmoid_mean`
  - `A1_QMIX_ALPHA = 0.5` (probe value; α=0 degenerates to the prior
    pure-rank cascade → bit-identical LB-0.933 path, clean revert).
- B1 cell 40 unchanged; rank-space inverse-CDF rescue of B1's
  pathological logits preserved (see §14.8.L3-probe for why).
- Inverse-CDF round-trip preserved at the end → cell 38 per-class
  threshold semantics unchanged.

Backup of pre-edit notebook at `/tmp/postproc_backup.ipynb` (2026-04-17).
`git diff` on the notebook shows the edit net clearly; expect
`A1_QMIX_ALPHA`, `L3_TAG`, and the two logging string changes
("Quantile-Mix fusion: …", "A1 Quantile-Mix mean |Δ score|").

### 14.8.L3-probe RESULT — L3 KILLED 2026-04-17 23:48 UTC

**Kernel v51 pushed 2026-04-17 ~20:10 UTC, COMPLETE 22:46, auto-submitted.**
**LB 0.925** on hidden test.

| Metric | Value |
|---|---|
| LB score | 0.925 |
| Baseline (most recent production, 2026-04-16 21:29) | 0.931 |
| Δ vs baseline | **−0.006 regression** |
| Kill gate (§14.8.L3-probe) | LB ≥ 0.934 |
| Pass? | **NO — kill L3** |

**Revert completed 2026-04-17 23:50 UTC:**
- Cell 41 `A1_QMIX_ALPHA = 0.0` (bit-identical to pure-rank LB-0.933 path).
- `L3_TAG="qmix_a50"` left in place for audit trail — harmless, no
  effect when α=0.
- Not re-pushed; LB-0.933 production is unchanged on Kaggle (v50 still
  selected for final private score unless replaced).
- `/tmp/postproc_backup.ipynb` available for full rollback if needed.

**Interpretation:**
- Quantile-Mix at α=0.5 (canonical writeup value) actively regressed
  the fused output. The prob-space blend `rescale01(proto) + a1_sigmoid`
  likely introduces scale mismatch that the rank-space cascade's
  inverse-CDF absorbs but the prob-space side cannot.
- α=0.25 sweep not justified — writeups cite α=0.5 as the lift
  point; a weaker α is likely a smaller regression, not a win.
- L3 is complete: the hypothesis was tested at the writeups' stated
  value; it does not hold on this fusion stack.

**Reconciliation note:** Prior "LB 0.933" label in §B1 / §§14 was
from an earlier fusion state; the CURRENT production baseline as of
2026-04-17 is LB 0.931 (v50, 2026-04-16 21:29). All ±Δ gates going
forward should reference 0.931, not 0.933. §14.4 lever expected-lift
ranges remain valid in direction but absolute "beats" must clear 0.931.

**Next action: L1 (cross-arch noisy student) per §14.4 row L1 and
§14.5 sequencing.** Blockers:
- `train_audio` Perch features must be Kaggle-extracted (embedding
  mismatch constraint — see §§10/13 kill memos).
- Check §13 Phase 1 autopilot completion status: does
  `jupyter/perch-train-audio-extract/` have a completed run with the
  full train_audio Perch feature cache on Kaggle? That cache gates L1.

**Blocker for L1**: train_audio Perch feature availability from §13
Phase 1 autopilot (must be Kaggle-extracted, not local, per
embedding-mismatch constraint). Check autopilot completion status
before writing L1 scripts.

**Do NOT**:
- Re-open Track C / §13 under any pretext.
- Revisit D1-b or any monotonic per-fold calibration.
- Start L2 or L1 before L3 is resolved.
- Skip the clean-logs step (`feedback_clean_logs_before_training.md`)
  before any future training launch.

---

## 14.9 L1 recipe — cross-arch noisy student (2026-04-17 23:55 UTC)

**Concept.** Train a fresh A1 EffNet-B0 SED model (the student) on the
existing focal `train_audio` corpus, but with soft targets injected from
the §10 ProtoSSM teacher (pseudo-labels already generated at
`four_track/data/processed/c2_pseudo_labels_kagglefeat/pseudo_soft_labels.npz`).
This is structurally different from the killed §13 Track C because the
student (EffNet-B0 on mels) has a **different inductive bias** from the
teacher (ProtoSSM on Perch features) — the student can learn signals
from the teacher that its own arch doesn't extract from mels directly.
The self-distillation ceiling documented in `project_track_c_killed.md`
does NOT apply to cross-architecture distillation.

### 14.9.1 Inputs (already on disk, no new extraction needed)

| Artifact | Path | Shape / content |
|---|---|---|
| Pseudo-label soft targets | `four_track/data/processed/c2_pseudo_labels_kagglefeat/pseudo_soft_labels.npz` | `stems (35549,) <U32`, `soft_labels (35549, 234) float32`, `species_id (35549,) <U16` |
| Pseudo-label retain mask | `four_track/data/processed/c2_pseudo_labels_kagglefeat/pseudo_labels.parquet` | 33,516 of 35,549 clips have `retained=True` via `(max_conf > 0.6) ∧ (primary_label ∈ teacher_top3)`. Use the retained subset to drop teacher-confused clips. |
| Train folds | `data/processed/train_folds.csv` | `filename` = `"<species_id>/<stem>.ogg"`, `fold ∈ {0..4}` |
| A1 baseline checkpoints | `kaggle_datasets/a1-effb0-ckpts/a1_fold{0,1,2,4}.pt` | **Keep untouched.** These serve the current LB-0.931 path. Noisy-student checkpoints go to a parallel directory. |

**Stem-join key:** `Path(row["filename"]).stem` (e.g. `"1161364/iNat1216197.ogg"` → `"iNat1216197"`) lookup into the `stems` array in the NPZ. Build a dict once at dataset construction.

**No Kaggle blocker for L1.** The pseudo-label NPZ was already produced
from Kaggle-extracted Perch features (directory suffix `_kagglefeat`),
so the teacher side is embedding-consistent with the submit kernel.
The student itself reads raw mel spectrograms — no Perch features at
training time. The §13 Phase 1 train_audio Perch extraction autopilot
is therefore NOT a blocker for L1; it was a blocker for Track C's
ProtoSSM self-distill which needed Kaggle-side Perch feats for both
teacher and student.

### 14.9.2 Script: `src/train_a1_noisy_student.py`

New file, sibling of `src/train_a1.py`. **Do not modify `train_a1.py`.**
This keeps the baseline A1 pipeline untouched so rollback is
`rm -r four_track/models/a1_ns/ && rm four_track/src/train_a1_noisy_student.py`.

Structure (mirrors `train_a1.py`, overrides where marked `#NS#`):

```python
# train_a1_noisy_student.py — L1 cross-arch noisy student
#NS# Loads pseudo_soft_labels.npz and pseudo_labels.parquet at startup.
#NS# Subclasses BirdTrainDataset with stem→soft_label lookup.
#NS# Merges hard + soft targets per-sample via element-wise max.
#NS# MixUp also element-wise-maxes the two samples' merged targets.
#NS# Trains with BCE (not ASL — see §14.9.3 loss note).
#NS# Checkpoints to four_track/models/a1_ns/.

class NoisyStudentDataset(BirdTrainDataset):
    def __init__(self, df, soft_lookup, augment=True, ...):
        # Filter df to retained=True rows (drops 2033 low-conf clips).
        df = df[df["_ns_retained"]].reset_index(drop=True)
        super().__init__(df, augment=augment, ...)
        self.soft_lookup = soft_lookup   # dict[str, np.ndarray(234,)]

    def _build_targets(self, row):
        labels, mask = super()._build_targets(row)
        stem = Path(row["filename"]).stem
        soft = self.soft_lookup.get(stem)
        if soft is not None:
            # Soft target already sigmoid'd by teacher.
            # Element-wise max: hard labels dominate where present,
            # teacher leaks additional positive classes elsewhere.
            labels = np.maximum(labels, soft.astype(np.float32))
        return labels, mask   # mask unchanged (secondary-label handling identical)
```

MixUp's `_mixup` already does `np.maximum(labels1, labels2)` — that
naturally composes with the soft-merge above (the mix's labels are the
max of both samples' merged targets). No change to `_mixup` needed.

### 14.9.3 Loss — use BCE, not ASL

ASL (`AsymmetricLossOptimized`) aggressively down-weights easy negatives
by raising `(1 − p)^γ_neg`. Applied to soft targets, this interacts
badly: a target value of 0.4 from the teacher is neither a clear positive
nor a clear negative, and ASL's asymmetric γ will either over-confidently
suppress the teacher signal (treating 0.4 as a soft negative) or amplify
it (if γ_pos is small). Plain `BCEWithLogitsLoss(reduction="none")` on
soft targets is well-defined (cross-entropy against a Bernoulli with
parameter = soft target) and is the standard Noisy-Student loss.

Secondary-label masking is preserved via the existing `mask` tensor —
loss is `(loss_per * mask).sum() / mask.sum()`, same as `train_a1.py:252`.

### 14.9.4 Noise — three levers above the baseline

Noisy Student's empirical edge comes from making the student *harder
to converge than the teacher's target function*, forcing it to learn
more robust features. Three cheap knobs:

| Knob | Baseline | NS value | Where |
|---|---|---|---|
| `mixstyle_p` | 0.5 | **0.7** | `train_a1.py:313` CLI default |
| `SPEC_TIME_MASK_PROB` | config default | **×1.2** | override inside script via monkey-patch in `config` import |
| `SPEC_FREQ_MASK_PROB` | config default | **×1.2** | same |

MixUp and background-noise augmentation stay at their current settings
(already strong). Dropout stays at the model default. Do NOT add
stochastic depth without a smoke test — the EffNet-B0 backbone is not
wired for it.

### 14.9.5 Hyperparameters — identical to baseline except where noted

| Param | Value | Note |
|---|---|---|
| epochs | 25 | same as `train_a1.py --epochs 25` |
| optimizer | AdamW | same |
| `LR`, `LR_MIN`, `T_0` | same as parent `config.py` | |
| `BATCH_SIZE` | same | |
| `WEIGHT_DECAY` | same | |
| loss | **BCE** | §14.9.3 above |
| `mixstyle_p` | **0.7** | §14.9.4 |
| fold order | **0 first**, then 1, 2, 4 if pilot passes | matches existing a1 ckpt set |
| checkpoint dir | `four_track/models/a1_ns/a1_ns_fold{F}_seed42_bce.pt` | NEW path — do not overwrite `models/a1/` |

### 14.9.6 Pilot → full-fold sequencing

**Phase 1 (pilot, ~12h GB10 wall time):**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/kaggle/BirdCLEF/four_track
rm -f log/*.log   # CLAUDE.md pre-launch requirement
nohup python -u src/train_a1_noisy_student.py --fold 0 --epochs 25 \
  > log/train_a1_ns_fold0_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Pilot kill gate** (§14.4 row L1 + §14.8 revised baseline):

- PASS: fold-0 `val_roc_auc ≥ 0.7514` (baseline 0.7414 + 0.010)
- PARTIAL: `0.7414 ≤ val < 0.7514` — borderline; push LB probe with
  fold-0 only (single-fold LB ≈ baseline − 0.003 historically, so a
  borderline pilot is worth one LB slot to confirm). Kill if
  LB < 0.935.
- FAIL: `val < 0.7414` — kill. Do not run other folds. Remove
  `models/a1_ns/`. Pivot to L2 fallback.

**Phase 2 (full folds, ~36h total):**

Only if pilot PASSES. Run folds 1, 2, 4 sequentially (one nohup each,
serialize because GB10 is single-GPU):

```bash
for F in 1 2 4; do
  nohup python -u src/train_a1_noisy_student.py --fold $F --epochs 25 \
    > log/train_a1_ns_fold${F}_$(date +%Y%m%d_%H%M%S).log 2>&1
done
```

**Phase 3 (LB probe):**

Upload `models/a1_ns/a1_ns_fold{0,1,2,4}.pt` as a new Kaggle dataset
`stevewatson999/a1-effb0-ns-ckpts`. Add it to the submit kernel's
`dataset_sources`, add a `A1_VARIANT = "noisy_student"` flag to Cell 2
CFG, and swap the checkpoint directory in the A1 inference cell from
`a1-effb0-ckpts` → `a1-effb0-ns-ckpts` when the flag is set. Keep the
baseline path as the default fall-through so one `A1_VARIANT` flip
fully rolls back.

**LB kill gate**: LB ≥ 0.935 (baseline 0.931 + 0.004, mid of expected
range). Below → revert `A1_VARIANT` to baseline, keep NS checkpoints
archived for possible future ensemble.

### 14.9.7 Rollback (single command)

Baseline is untouched throughout:
- `four_track/src/train_a1.py` — never edited.
- `four_track/models/a1/*.pt` — never overwritten.
- `kaggle_datasets/a1-effb0-ckpts/` — never modified.
- Submit kernel reverts by setting `A1_VARIANT = "baseline"` (or
  removing the flag block entirely).

Full local rollback if NS fails:
```bash
rm -rf four_track/models/a1_ns/
rm four_track/src/train_a1_noisy_student.py
rm -f four_track/log/train_a1_ns_*.log
```

### 14.9.8 Known limitations (do not try to fix preemptively)

1. **Single teacher.** Using only §10 ProtoSSM, not a teacher ensemble
   (2025 winners used 3-5 teachers averaged). Revisit only if L1 passes
   and a further +0.002 is needed — NOT before.
2. **Clip-level soft labels on 5s windows.** The teacher scored the
   whole clip; a 5s window within it may be silence/noise while the
   stem-level soft label believes species X. Acceptable noise for NS —
   makes the student harder to fit, which is the point.
3. **No VAD pre-filter.** `project_track_c_killed.md` noted A2's focal
   vs. soundscape distribution shift. L1 inherits the focal-clip
   training distribution unchanged; the hope is the multi-target
   soft-label injection closes more of the gap than the mismatch widens.
   Kill gate handles the downside.
4. **Embedding-mismatch not triggered.** The student consumes mels;
   the teacher-signal is pre-baked into the NPZ offline. No Perch
   feature computation happens inside `train_a1_noisy_student.py`.

### 14.9.9 Why this is expected to lift LB

- Teacher's soft labels carry **multi-label context** the current A1
  hard-label training discards (hard labels are essentially single-
  primary-label one-hots; the teacher knows about co-occurring species
  implicit in the audio).
- **Regularization** against overfitting to focal-clip primary-label
  bias. A1's current fold-0 val 0.7414 on soundscapes (not focal) is
  evidence the focal prior hurts generalization.
- **Diversity to the fusion.** Current A1 and ProtoSSM rank-fuse well
  (LB 0.931). NS-A1 will be correlated with baseline A1 (same arch,
  same data) but shifted toward the teacher's multi-label mode —
  modest ensemble gain on top of the standalone lift.
- Expected single-fold lift: +0.003 to +0.008 val; ensemble-with-ProtoSSM
  lift: +0.005 to +0.015 LB (§14.4 row L1 range).

### 14.9.10 Fold-0 pilot mid-run check (2026-04-17 21:11 EDT) — val lift suspiciously large

At 42-minute mark of the fold-0 pilot (PID 365849), val ROC-AUC on
train_soundscapes was already tracking **~0.85** by epoch 6 and hit
best=**0.8603** at epoch 10 — **+0.119 above the frozen A1 baseline
0.7414, which is ~10× the +0.005-+0.015 expected range in §14.4.**

Per-epoch wall: ~3m 32s (25 epochs × 3.5min ≈ 88min total). Revised
fold-0 ETA: **~21:58 EDT**, not 08:28 tomorrow.

| Ep | val | Δ vs 0.7414 |
|---|---|---|
| 1 | 0.6795 | −0.062 |
| 2 | 0.7499 | +0.009 |
| 3 | 0.7782 | +0.037 |
| 4 | 0.8165 | +0.075 |
| 5 | 0.8179 | +0.077 |
| 6 | 0.8499 | +0.109 |
| 7 | 0.8523 | +0.111 |
| 8 | 0.8492 | +0.108 |
| 9 | 0.8519 | +0.111 |
| 10 | **0.8603** | **+0.119** |
| 11 | 0.8542 | +0.112 |

**Hypothesis for the magnitude (soundscape-leakage via teacher) — CONFIRMED 2026-04-17 21:30 EDT.**

Evidence: `four_track/src/train_protossm_local.py`:
- L468: `soundscape_labels = pd.read_csv(RAW_DATA / "train_soundscapes_labels.csv")`
- L505-513: builds `Y_FULL` FROM those labels
- L517: `labels_files, _ = reshape_to_files(Y_FULL, meta_full)` — these ARE the training labels
- L607: `gkf = GroupKFold(n_splits=n_splits)`
- L647-651: `train_proto_ssm_single(... labels_train=data["labels_files"][train_idx] ...)`

The §10 ProtoSSM teacher is a soundscape CV model — it trains **on**
train_soundscapes with GroupKFold held-out val. Across all CV folds its
weights have seen all train_soundscapes files at some point.

Leakage mechanism for L1 NS:
1. Teacher trained on train_soundscapes → encodes soundscape acoustics.
2. Teacher produces pseudo-labels on train_audio (`pseudo_soft_labels.npz`).
3. A1 NS student trains on train_audio with those pseudo-labels →
   inherits soundscape-aware supervision.
4. A1 NS validates on train_soundscapes → matches val distribution via
   teacher proxy.

The student never touches val directly, but its supervision was
generated by a model that did. **val_roc_auc on this val set is
invalid as a kill gate for distilled students; only LB is trustworthy.**

Baseline A1 0.7414 remains a CLEAN val measurement — baseline trains on
`train_audio` hard labels only with no teacher in the loop. The
asymmetric comparison (clean baseline vs. leakage-biased student) is
why pilot shows +0.1189.

Saved to memory: `project_protossm_teacher_val_leakage.md`.

### 14.9.11 Revised next steps — LB is the only arbiter (audit complete, val invalid)

**DO NOT auto-launch folds 1/2/4 on fold-0 completion.** Val is
leakage-biased (§14.9.10) — the gate is invalid for distilled students.

**Step 1 — let fold-0 finish** (ETA 21:58 EDT). Record the final
best_val_roc_auc in this section. *Do not treat it as a gate.* Even
if it hits 0.90, LB may show no real lift.

**Step 2 — teacher-lineage audit: DONE 2026-04-17 21:30 EDT — contaminated.**
See §14.9.10 for evidence. Teacher trained on train_soundscapes via
GroupKFold. Not a kill for L1 (student still might generalize), but
val numbers from this pilot cannot be used for a kill/greenlight decision.

**Step 3 — fold-0-only LB probe** (1 LB slot) — **the only arbiter**:
- Upload `models/a1_ns/a1_ns_*_fold0_seed42_bce.pt` as a single-file
  Kaggle dataset `stevewatson999/a1-effb0-ns-ckpts` (v1 = fold-0 only).
- Add `A1_VARIANT = "noisy_student"` flag + conditional checkpoint-dir
  swap in postproc notebook Cell 2 / Cell 30.
- Push + auto-submit one LB probe. Treat the baseline A1+B1+ProtoSSM
  fusion as the denominator (LB 0.931).
- **Decision:**
  - LB ≥ 0.935 → strong signal the lever works. Launch folds 1, 2, 4
    sequentially (~10h), upload combined dataset v2, push again.
  - 0.931 ≤ LB < 0.935 → weak signal. Might still win after 4-fold
    ensemble. Judgement call: launch remaining folds only if
    soundscape-leakage audit (Step 2) came back clean.
  - LB < 0.931 → leakage hypothesis confirmed. Kill L1, pivot to L2.
    Archive `models/a1_ns/` for possible future use if teacher is
    retrained.

**Step 4 — if Step 3 passes** (folds 1, 2, 4 launch):
- Single sequential nohup run: `--folds 1,2,4 --epochs 25`.
- Wall time: ~(88 min × 3) ≈ 4.4h on GB10.
- Gate per fold: the +0.010 fold-0 gate is now moot (pilot exceeded).
  Instead use the LB-probe result as the authoritative evidence;
  additional folds are ensemble strength, not a re-test.
- Upload combined ckpts as `a1-effb0-ns-ckpts` v2. Push final LB probe.

**Step 5 — final integration:**
- If fully successful (LB ≥ 0.935 with 4-fold): fold-0 NS pilot result
  + LB path documented in `project_l1_passed.md`, `A1_VARIANT` flag
  promoted to default, baseline `a1-effb0-ckpts` kept as rollback dataset.
- If partial (LB 0.931-0.935): document in `project_l1_partial.md`, keep
  as optional blend — may be worth ensembling with baseline A1 via
  weighted rank fusion, but outside current L1 scope.

### 14.9.12 RESULT — L1 KILLED 2026-04-18 (LB 0.930 vs 0.931 baseline)

**Fold-0 NS pilot complete** (PID 365849, 2026-04-17 20:28 → 21:59 EDT,
25/25 epochs). Best val_roc_auc = **0.9042** at epoch 23 — curve still
climbing at run end, no overfit rollover. +0.1628 vs clean baseline
0.7414.

**LB probe** (kernel v52 of `birdclef-2026-protossm`, pushed
2026-04-18 02:10 UTC, submitted 02:26 UTC):
- `A1_VARIANT = "ns_fold0"` — NS checkpoint served fold 0, baseline
  checkpoints for folds 1/2/4 (confirmed via kernel log:
  `loaded fold 0 [NS]`, `loaded fold {1,2,4} [base]`).
- Submit pipeline otherwise bit-identical (α=0.0, rank-space fusion,
  D2B OOF cell left on baseline so D2B coefs remain valid).
- **Public LB: 0.930** (baseline 0.931, **Δ = −0.001**).

**Verdict: KILL.** The +163 bp val lift evaporated to −1 bp on hidden
test — the largest val/LB disagreement we've seen on this pipeline.
Consistent with `project_protossm_teacher_val_leakage.md`: the
ProtoSSM teacher (trained on `train_soundscapes` via GroupKFold) bled
soundscape-distribution signal into the student via pseudo-labels,
which then matched held-out soundscape val folds but had no transfer
to the hidden Pantanal test set.

**Rollback applied same session:** Cell 41 of
`jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb` set
to `A1_VARIANT = "baseline"` (bit-identical to the LB-0.931 path).
`A1_NS_CKPT_DIR`, `A1_NS_FOLDS`, and the conditional checkpoint-dir
swap left in place as dormant code — harmless at `A1_VARIANT =
"baseline"`, cheap re-probe path if the teacher is ever retrained
clean.

**How to apply to future levers:**
- Never set val-based kill gates for distilled students when the
  teacher overlaps their val set, even indirectly. LB is the only
  valid arbiter. This now stands as an empirical, not just
  theoretical, result — val can overshoot LB by 160+ bp.
- Do NOT launch folds 1/2/4. The fold-0 ckpt on disk at
  `four_track/models/a1_ns/a1_ns_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_bce.pt`
  is kept for reference but is not in the production pipeline.
- The Kaggle dataset `stevewatson999/birdclef-2026-a1-effb0-ns-ckpts`
  stays live as a no-cost artifact for any future clean-teacher retry.
- L1 v2 (clean teacher) requires retraining the ProtoSSM teacher
  *excluding* `train_soundscapes` from its training set, which is a
  multi-day effort and is NOT the next action. Ranked below L2.

**Sequencing update:** L3 killed, L1 killed. **L2 (multi-year
pretrain, +0.001 to +0.004 expected per §14.4+§14.8) is now the next
lever.** See §14.10 when drafted.

---

## 14.10 L2 recipe — 2025-only A1 pretrain → 2026 finetune (2026-04-18 draft)

**Status: RECIPE DRAFT, not greenlit.** Per the 2026-04-18 02:30 UTC
pickup pointer, return for a go/no-go gate conversation before any
data download or training launch. Kill criteria below; do not auto-
launch on draft completion.

This recipe consolidates §14.6 (revised 2025-only variant) and §14.8
(prereq measurements) into a single executable spec, structured to
match §14.9.

### 14.10.1 Inputs (what's on disk vs. what needs downloading)

**Already on disk (verified 2026-04-18):**
- `four_track/src/train_a1.py` (375 lines) — production A1 EffB0
  trainer; entry point for the finetune phase. Untouched since L1.
- `four_track/scripts/train_a1_5fold.sh` — production 5-fold launcher.
- `four_track/kaggle_datasets/a1-effb0-ckpts/a1_fold{0,1,2,4}.pt` —
  current LB-0.931 production ckpts. **Roll-forward target.**
- `data/raw/taxonomy.csv` — 2026 taxonomy (234 species).
- `data/raw/train_audio/` — 2026 focal corpus (208 species dirs).
- 1.8 TB free on `/dev/nvme0n1p2`; 2025 corpus expected ~30-40 GB.

**Needs downloading (one-time, ~10-15 min on a fast link):**
- `data/raw/birdclef_2025/` — full 2025 competition dataset.
  - `kaggle competitions download -c birdclef-2025 -p data/raw/birdclef_2025/`
  - `cd data/raw/birdclef_2025 && unzip -q birdclef-2025.zip`
  - Rules already accepted by user 2026-04-17 per §14.8 — verify
    with a dry-run `kaggle competitions files birdclef-2025` first.
- After extraction, expect:
  - `data/raw/birdclef_2025/train_audio/` (focal clips, iNat-id
    directory layout matching 2026)
  - `data/raw/birdclef_2025/taxonomy.csv` (2025 species list,
    ~206 species per Colombia / Middle Magdalena Valley)
  - `data/raw/birdclef_2025/train.csv` (focal-clip metadata)
  - **DO NOT download** `train_soundscapes/` from 2025 — irrelevant
    to a focal-clip pretrain and adds disk pressure.

**Class-space mapping (one-time, in pretrain script):**
- Build `union_2025_2026 = sorted(set(tax_2025.primary_label) ∪
  set(tax_2026.primary_label))` — expected size ~400 (234 + ~206
  with 18% overlap = ~400 unique).
- Persist `data/processed/union_2025_2026_classes.json` for
  finetune to slice the pretrained head cleanly.

### 14.10.2 Scripts (paths, ownership)

Two new files, one new launcher; **no modifications** to
`src/train_a1.py` or `scripts/train_a1_5fold.sh`:

1. **`src/pretrain_a1_2025.py`** (new, ~250 lines).
   - Forks `src/train_a1.py`'s data loader & mel pipeline.
   - Replaces the 2026 train_audio path with 2025 train_audio.
   - Loads union class list from
     `data/processed/union_2025_2026_classes.json`.
   - Single-split (no folds): 95% train / 5% val on 2025 only,
     stratified by primary_label.
   - Saves `models/a1_pretrained_2025/a1_pretrained_2025.pt` with
     full state_dict + the union class list embedded as a buffer
     for finetune-side verification.

2. **`scripts/pretrain_a1_2025.sh`** (new, ~20 lines).
   - `nohup`-wrapped launcher matching CLAUDE.md conventions.
   - Single GPU, no resume logic, no fold loop.

3. **`src/train_a1.py` (REUSE, NOT MODIFY)** for finetune phase.
   - Add a thin CLI flag `--init-from PATH` to
     `src/train_a1.py` (the only mod). When set, loads the
     pretrained ckpt's backbone weights, drops/reinits the
     classifier head to 234 (2026 only). All other hyperparameters
     unchanged from the LB-0.931 baseline.
   - This keeps the production training loop untouched at runtime
     when `--init-from` is omitted (default = current behavior).

4. **`scripts/finetune_a1_from_2025.sh`** (new, ~25 lines).
   - Calls `src/train_a1.py --init-from
     models/a1_pretrained_2025/a1_pretrained_2025.pt --epochs 25
     --folds 0,1,2,4 ...` — i.e. the production 4-fold setup, just
     init'd from the pretrained backbone.

### 14.10.3 Loss / class-space (union-head, then slice)

- **Pretrain head**: BCE-with-logits over the **union class space**
  (~400 classes). 2025 audio's primary_label one-hot lives in the
  2025 slice; 2026 classes get all-zero supervision during
  pretrain (intentional — no leakage from 2026 labels).
- **Finetune head**: re-init a **234-class** linear layer from
  scratch on top of the pretrained backbone. Discard the union
  classifier weights entirely. Rationale: head re-init is standard
  for cross-corpus transfer; the union head only existed to give
  the pretrain a non-trivial loss signal.
- **Loss for finetune**: ASL+BCE hybrid (production setting),
  unchanged from current `train_a1.py`.

### 14.10.4 Hyperparameters

**Pretrain phase** (single run, no folds):

| Knob | Value | vs. baseline |
|---|---|---|
| Epochs | 10 | (baseline finetune is 25, but pretrain is shorter on purpose) |
| LR | 1e-3 | (baseline 5e-4 — higher because head is fresh and corpus is bigger) |
| Schedule | cosine, no warmup | (baseline cosine + 1ep warmup) |
| Batch | 32 (same as baseline) | unchanged |
| Mel pipeline | PCEN + ASL + Frequency-MixStyle 0.5 | unchanged from `train_a1.py` |
| Val split | 5% stratified, single split | (no fold loop) |
| MixUp | OFF | (baseline OFF too) |

**Finetune phase** (4-fold, identical to LB-0.931 baseline except for `--init-from`):

| Knob | Value |
|---|---|
| Epochs | 25 |
| LR | 5e-4 |
| Schedule | CosineAnnealingWarmRestarts, T_0=5 |
| Folds | 0, 1, 2, 4 (skipping fold 3 same as production) |
| Loss | ASL + BCE hybrid (unchanged) |
| Init | `--init-from models/a1_pretrained_2025/a1_pretrained_2025.pt` |
| Head | re-init to 234 classes |

**Wall-clock estimate (GB10):**
- Pretrain: 10 epochs × ~40 min/epoch (2025 corpus 1.5-2× 2026 size)
  = **~6.5 h**.
- Finetune: 4 folds × 25 epochs × ~3.5 min/epoch (same as L1 NS
  pilot, since the loop is unchanged) = **~5.8 h**.
- Combined: **~12.5 h** end-to-end. Fits in one overnight run.

### 14.10.5 Pilot → full sequencing

Two-stage gate to avoid wasting the full 12.5h on a broken pretrain:

**Stage 1 — pretrain smoke (1 epoch, ~40 min):**
1. Run `pretrain_a1_2025.py --epochs 1 --smoke-test`.
2. Verify: training loss decreases, val_roc_auc on the 5% 2025
   split is ≥ 0.55 (random = 0.50; need a non-trivial signal).
3. **Smoke gate:** below 0.55 → halt, debug data loader / class-
   space mapping / mel pipeline drift before launching full
   pretrain. Do NOT just push through.

**Stage 2 — full pretrain (10 epochs, ~6.5 h):**
1. Launch full pretrain from the smoke-checked ckpt OR from
   scratch (decide: from-scratch is cleaner and only +40 min).
2. Save `a1_pretrained_2025.pt` on epoch-best val_roc_auc.

**Stage 3 — finetune fold-0 only (1 fold, ~1.5 h):**
1. `train_a1.py --fold 0 --init-from a1_pretrained_2025.pt`.
2. **Local val gate (per §14.4 row L2):** fold-0 val_roc_auc on
   the 2026 soundscape val must beat the frozen baseline 0.7414
   by **≥ +0.005** (i.e. ≥ 0.7464). Below → kill L2 immediately.
   Above → proceed to Stage 4.

**Stage 4 — finetune folds 1, 2, 4 (3 folds, ~4.3 h):**
1. Sequential `--folds 1,2,4`.
2. Replace `kaggle_datasets/a1-effb0-ckpts/a1_fold{0,1,2,4}.pt`
   with the four new ckpts. **Per `feedback_backup_ckpts_before_overwrite`**:
   `cp -a kaggle_datasets/a1-effb0-ckpts kaggle_datasets/a1-effb0-ckpts.lb931_baseline_$(date +%Y%m%d)` first.

**Stage 5 — LB probe (1 LB slot):**
1. Push `stevewatson999/birdclef-2026-a1-effb0-ckpts` v(N+1) with
   the four new ckpts.
2. Submit notebook unchanged at A1_VARIANT="baseline",
   A1_WEIGHT=0.20.
3. **LB gate:** ≥ 0.932 (baseline 0.931 + 0.001 minimum, per the
   demoted +0.001-+0.004 estimate in §14.8). Below 0.931 → roll
   back ckpts to the .lb931_baseline_* backup, declare L2 dead.

### 14.10.6 Rollback (single command)

If LB ≤ 0.931 OR if Stage-3 fold-0 val gate fails:

```bash
cd /home/swatson/work/kaggle/BirdCLEF/four_track
rm -rf kaggle_datasets/a1-effb0-ckpts
mv kaggle_datasets/a1-effb0-ckpts.lb931_baseline_<date> kaggle_datasets/a1-effb0-ckpts
# Re-push the dataset to restore Kaggle to v(N) baseline:
kaggle datasets version -p kaggle_datasets/a1-effb0-ckpts -m "L2 rollback to LB-0.931 baseline"
```

Notebook needs no edit — `A1_VARIANT="baseline"` already points
at the dataset name; reverting the dataset content is sufficient.

### 14.10.7 Known limitations (do not try to fix preemptively)

1. **18% species overlap (§14.8).** Most of the 2025 pretraining
   signal helps non-overlapping species, which the 2026 finetune
   discards along with the union head. The transfer is via
   *low-level acoustic features in the backbone*, not class-
   level priors. This is exactly why L2's expected lift is so
   modest (+0.001 to +0.004).
2. **2025 val (5% split) is not predictive of 2026 LB.** Pretrain
   val just gates "is the loop training at all"; only Stage-3
   fold-0 2026-soundscape val is a meaningful kill gate, and
   even that overstates LB confidence (per L1's val-leakage
   lesson).
3. **No multi-year (2021-2024) extension in this recipe.** §14.6
   already ruled out 2021-2024 as low value per GB and high
   mapping cost (eBird → scientific_name → 2026 primary_label).
   Revisit only if 2025-only L2 passes with margin.
4. **Pretrained backbone is single-seed.** A multi-seed pretrain
   ensemble could squeeze another ~+0.001 but multiplies wall-
   clock; out of scope for the first try.

### 14.10.8 Why this is expected to lift LB (and why only modestly)

- Multi-year pretraining is the **only +-confirmed lever from 2025
  writeups we haven't tried**: Max Melichov reported +0.009 on
  2025 with 2021-2024 pretrain. Our class overlap is much smaller
  (18% vs the within-2025 case), so we discount the upside ~3×.
- Pretraining provides the A1 backbone with **broader acoustic
  exposure** — different recording gear, different forest types,
  different background-noise distributions. Even if the species
  don't overlap, the *encoder* benefits.
- Lift is independent of ProtoSSM and B1 — diversifies the LB
  story. Unlike L1/L3 which interact with fusion, L2 just makes
  the A1 branch better and lets the existing rank fusion do its
  job.
- This is a **safe** lever (LOW risk in §14.4): the worst-case
  outcome is "pretrained backbone is no better than ImageNet
  init", which costs 12.5h of GPU but doesn't damage anything.
  No teacher-leakage hazard (cf. L1).

### 14.10.9 Decision gate before launch (return for go/no-go)

Per the 2026-04-18 02:30 UTC pickup pointer:

> **After the recipe is drafted, return for a go/no-go gate
> conversation before any training launch. Do NOT auto-launch L2.**

Open questions for the gate conversation:

1. **Pretrain from scratch or from current A1 ckpt?**
   - From-scratch (recipe default): cleaner, longer (~6.5h),
     known recipe.
   - From current A1: warmer start, shorter (~3-4h), but the
     pretrain head has to coexist with 2026 weights — risk of
     catastrophic forgetting on the union-head re-init.
   - **Default = from scratch.** Override only if user prefers
     speed.
2. **Is +0.001-+0.004 worth ~13h of GPU + 1 LB slot?**
   - Slot budget is comfortable (~200 slots over 40 days,
     §14.4); GPU is idle overnight; L4/L5/L6/L7/L8 are all
     contingent on something else first. So opportunity cost is
     ~free.
   - But if user has a specific higher-EV lever in mind (e.g.
     external Xeno-Canto Pantanal data, L5 v2), that should
     pre-empt L2.
3. **Should we wait for a clean ProtoSSM teacher (L1 v2) instead?**
   - L1 v2 requires retraining the §10 teacher *excluding*
     train_soundscapes — multi-day, distinct workstream.
     Per §14.9.12 it's ranked below L2; no reason to invert.

**If user greenlights:** start at Stage 1 (pretrain smoke).
**If user pivots:** archive this recipe, jump to whichever lever
the user prefers.

---

## ⏸️ PICK UP HERE — previous (2026-04-18 — L2 recipe drafted, awaiting go/no-go — SUPERSEDED)

**State:** §14.10 recipe is drafted but **not greenlit**. No
training has started, no 2025 data downloaded, no script created.
The only on-disk side-effect of this drafting session was appending
this section to `four_track/new_plan.md`.

**Production state unchanged from §14.9.12 close-out:**
- LB 0.931 baseline (notebook v52 rollback, A1_VARIANT="baseline").
- L1 NS fold-0 ckpt archived in `models/a1_ns/` and Kaggle dataset
  `birdclef-2026-a1-effb0-ns-ckpts` (dormant).
- Submissions list top: 0.930 (L1 probe), 0.925 (L3 probe), 0.931
  (baseline).

**Next conversation should start with:** "Reviewed §14.10. {green-
light Stage 1 pretrain smoke / pivot to <other lever> / iterate on
recipe at <specific concern>}."

**Do NOT** in any new session:
- Skip the §14.10.5 Stage-1 smoke gate and go straight to full
  pretrain.
- Run pretrain without first backing up
  `kaggle_datasets/a1-effb0-ckpts/` per
  `feedback_backup_ckpts_before_overwrite.md`.
- Modify `src/train_a1.py` for anything beyond the single
  `--init-from` flag described in §14.10.2.
- Skip `rm -f log/*.log` before any training launch (CLAUDE.md).
- Begin downloading 2025 data before the go/no-go conversation —
  Kaggle download is fast but takes a slot of attention; cleanest
  to confirm intent first.

---

## ⏸️ PICK UP HERE — previous (2026-04-16 end of day — SUPERSEDED)

**Current state:** Standalone bash autopilot is running unattended via
`nohup` at PID 326614 (started 2026-04-16 23:55). It owns the entire
v6 → v7 → v8 → merge → upload chain. **Do not push v7 or v8 manually —
the autopilot will do it.**

- Driver script: `scripts/perch_extract_autopilot.sh`
- Merge step:   `jupyter/perch-train-audio-extract/merge_partitions.py`
- Live log:     `log/perch_extract_autopilot_20260416_235507.log`
- Kernel:       https://www.kaggle.com/code/stevewatson999/birdclef-2026-perch-train-audio-extract

**First thing tomorrow morning — check progress:**
```bash
tail -50 /home/swatson/work/kaggle/BirdCLEF/log/perch_extract_autopilot_20260416_235507.log
ps -p 326614 -o pid,etime,cmd        # is it still alive?
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
kaggle kernels status stevewatson999/birdclef-2026-perch-train-audio-extract
```

**Possible morning states:**

| State | Where to look | What to do |
|---|---|---|
| Autopilot still running, on v6/v7/v8 | tail of autopilot log shows recent `status=KernelWorkerStatus.RUNNING` | Nothing — let it finish |
| Autopilot complete, dataset uploaded | log ends with `§13 Phase 1 autopilot COMPLETE` and a kaggle.com URL | **Move to Phase 2** (local teacher scoring) |
| Autopilot died with `ABORT:` | log ends with `ABORT: <reason>` | Diagnose. Likely culprits: (a) Kaggle kernel error → check kernel log; (b) merge schema mismatch → inspect `/tmp/perch_extract_v{6,7,8}/`; (c) row count below gate → may need to re-extract a partition |
| Autopilot process dead but log ends mid-step | `ps` returns nothing, log doesn't end with COMPLETE or ABORT | Probably `set -uo pipefail` tripped on an unset var. Inspect last log lines and decide whether to resume manually from current PARTITION_ID state |

**What the autopilot does NOT do automatically:**
- Phase 2 (local teacher scoring) — that's the next pickup point
- LB submission — Phase 1 deliverable is a feature dataset, not a submission

**Known state of the code:**
- `jupyter/perch-train-audio-extract/kernel-metadata.json` has
  `dataset_sources: [jaejohn/perch-meta, stevewatson999/birdclef-2026-perch-onnx]`.
- `build_notebook.py` Cell 1 PARTITION_ID will be auto-incremented by
  the autopilot (0 → 1 → 2). Whatever value it ends on reflects the
  last partition pushed.
- Own ONNX dataset lives at
  `/home/swatson/work/kaggle/BirdCLEF/kaggle_datasets/perch-onnx/`
  (already live; do not re-upload).
- Final deliverable will be at
  `kaggle.com/datasets/stevewatson999/birdclef-2026-train-audio-perch`.
- Task list: #21 `in_progress` (v6); #22, #23, #24 `pending` —
  these are tracked by Claude's task system but the autopilot will
  not update them, so they may look stale. Use the autopilot log as
  ground truth.

### Autopilot reference (`scripts/perch_extract_autopilot.sh`)

**Why it exists:** Claude Code's `ScheduleWakeup` only fires while the
Claude session is open. To survive a logout, the autopilot is a plain
bash script run via `nohup` so it persists after the terminal closes.

**Behavior per phase:**
1. Poll `kaggle kernels status` every **600 s** (`POLL_INTERVAL`). On
   `COMPLETE`, advance. On `ERROR`/`CANCELLED`/`FAILED`, abort.
2. Pull outputs to `/tmp/perch_extract_v{6,7,8}/`. Verify:
   - log contains `PARTITION {0,1,2}/3` marker
   - log contains `[done]` (means extraction finished cleanly, not crashed)
   - both NPZ and parquet exist with the partitioned filename
3. Edit `build_notebook.py` Cell 1: regex-replace `PARTITION_ID = N` →
   `PARTITION_ID   = N+1`. Done by inline Python (sed avoided to keep
   regex behavior portable).
4. Rebuild the .ipynb (`python build_notebook.py`).
5. Push the new kernel version (`kaggle kernels push -p .`).
6. Sleep **180 s** (`POST_PUSH_GRACE`) — gives Kaggle time to register
   the new version before status polling resumes (else we might see
   the previous version's stale `COMPLETE` status).
7. Loop until partition 2 is done, then run `merge_partitions.py`
   (validates schema, dtype, row alignment, ≥34k unique stems), then
   `kaggle datasets create` to publish the final dataset.

**Key constants** (top of script — edit if needed):
```bash
KERNEL="stevewatson999/birdclef-2026-perch-train-audio-extract"
POLL_INTERVAL=600
POST_PUSH_GRACE=180
```

**Files the autopilot writes:**
- `log/perch_extract_autopilot_<timestamp>.log` (its own status log)
- `jupyter/perch-train-audio-extract/build_notebook.py` (PARTITION_ID
  bumps — the only mutated source file in the repo)
- `jupyter/perch-train-audio-extract/birdclef2026-perch-train-audio-extract.ipynb`
  (regenerated by build_notebook.py each push)
- `/tmp/perch_extract_v{6,7,8}/` (pulled kernel outputs)
- `kaggle_datasets/train-audio-perch/` (merged staging dir)
- Kaggle: kernel versions v7, v8 + the new dataset

**Files the autopilot does NOT touch:**
- `kernel-metadata.json` (frozen — the dataset_sources are correct as-is)
- Anything in `four_track/`, `data/`, `models/` outside the listed paths
- Git state (no commits, no pushes)

**If the autopilot dies and you need to resume manually:**
1. Read its last log line — that tells you what step it was on.
2. Read `build_notebook.py` Cell 1 PARTITION_ID — that's the partition
   currently in flight (or last completed if Kaggle status is COMPLETE).
3. Manually do whichever step the autopilot would do next:
   - Still waiting → re-launch the autopilot (it will resume polling).
   - Finished partition N, need to push N+1 → edit PARTITION_ID, rebuild,
     push, then optionally re-launch autopilot for the remaining partitions.
   - All 3 partitions extracted but merge failed → run
     `python jupyter/perch-train-audio-extract/merge_partitions.py`
     directly and inspect its non-zero exit code (1=missing files,
     2=row mismatch in input partition, 3=row mismatch after merge,
     4=emb schema fail, 5=scores schema fail, 6=stem count below 34k).

**To kill the autopilot cleanly:**
```bash
kill 326614    # PID from the start log; or `pgrep -f perch_extract_autopilot`
```
This will not stop the in-flight Kaggle kernel (Kaggle runs server-side);
it only stops local polling. Use `kaggle kernels status` to check the
remote kernel state separately.

---

### 14.10.10 RESULT — L2 KILLED 2026-04-18 21:47 UTC

**L2 fold-0 finetune fails the gate (need ≥0.7464; observed 0.6802).**

Three independent recipes were tested against the L2 gate (baseline A1
fold-0 val_roc_auc = 0.7414):

| attempt | pretrain split | finetune recipe | best val_auc | Δ vs baseline |
|--------|-----------------|-----------------|--------------|----------------|
| L2v1   | stratified-by-species (leaky) | LR=5e-4 + CosineAnnealingWarmRestarts(T_0=5) | 0.6614 | −0.080 |
| L2v2   | stratified (leaky) | LR=1e-4 + 2-ep linear warmup + single-cycle cosine | killed @ ep7 (0.5758) | trending to ~0.65 |
| L2v3   | **author-grouped (honest)** | LR=1e-4 + 2-ep warmup + single-cycle cosine | **0.6802** | **−0.061** |

All three miss the gate by a wide margin. Honest pretrain + gentle recipe
(v3) is +0.019 above leaky + hot (v1), confirming both fixes helped, but
**the remaining −0.061 gap is structural, not a recipe or split bug:**

1. **Pretrain val is honestly high (0.9908 author-grouped) but
   task-metric-irrelevant.** Focal single-species AUC saturates at ~0.99
   for any competent SED backbone — it measures "can you tell 2025 species
   apart in clean close-mic audio," which correlates poorly with
   "can you detect 234 Pantanal species in multi-source ambient soundscape."

2. **Train loss collapses immediately** (to ~0.006 by ep2) because the
   pretrained backbone already exploits its own discriminative features
   for the 234-class head. With near-zero gradient, the backbone **cannot
   rewrite** its specialist features into soundscape-general ones during
   finetune.

3. **Domain + species mismatch is catastrophic.** 2025 = Colombia
   Humboldt focal clips. 2026 = Pantanal (Brazil wetlands) soundscape,
   234 species with limited overlap. The pretrained specialist features
   don't generalize; ImageNet init's *generic* edge/texture features
   happen to transfer better via SpecAug + ASL + MixStyle forcing
   bird-general features during from-scratch training.

**Supporting evidence:**
- The leaky-split hypothesis was tested and falsified. Author-grouped
  pretrain (shared_authors=0) yielded essentially the same pretrain
  val_auc (0.9908) as the leaky split (0.9898) — leakage wasn't
  materially inflating the pretrain metric.
- v1 vs v3 convergence: v3 started −0.085 behind v1 at ep1, caught up
  by ep9 (v3=0.618 vs v1=0.620), and finished +0.019 ahead (0.6802 vs
  0.6614). Confirms honest pretrain + gentle recipe is strictly better
  than leaky + hot — but the effect size is too small to matter.

**Implementation changes made during this investigation (kept):**
- `src/pretrain_a1_2025.py:231-234` — StratifiedShuffleSplit replaced
  with GroupShuffleSplit on `author`. Prevents pretrain val leakage
  (cosmetic fix; didn't save L2 but is structurally correct).
- `src/train_a1.py:240-262` — `init_from` branch now uses a
  pretrain-aware recipe: LR=1e-4, 2-ep LinearLR warmup, single-cycle
  CosineAnnealingLR to LR_MIN=1e-6 (no warm restarts). Activates only
  when `--init-from` is passed; from-scratch path unchanged.
- Both changes are dormant code paths once L2 is killed; no need to
  revert unless L2 is ever revived.

**Artifacts:**
- `models/a1_pretrained_2025/a1_pretrained_2025.pt` — honest pretrain
  ckpt, val_auc 0.9908 @ ep7. Kept for reference.
- `models/a1_pretrained_2025/archive_leaky_v1/a1_pretrained_2025_leaky_valauc0p9898.pt`
  — prior leaky pretrain.
- `models/a1/archive_l2_v1/a1_l2v1_fold0_bestauc0p6614.pt`
- `models/a1/archive_l2_v3/a1_l2v3_fold0_bestauc0p6802.pt`
- `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_asl.pt` —
  **restored** from `kaggle_datasets/a1-effb0-ckpts/a1_fold0.pt` (JIT)
  after stripping the `inner.` prefix. 363/363 keys match. This is the
  production baseline fold-0 ckpt that produced LB 0.931.
- Logs: `log/archive/finetune_a1_l2_*`, `log/finetune_a1_l2v3_*`,
  `log/pretrain_a1_2025v2_*`.

**Standing after this closeout:**
- L1 (noisy student) — killed 2026-04-18.
- L2 (multi-year pretrain) — killed 2026-04-18 (this section).
- L3 (Quantile-Mix) — killed 2026-04-17.

**Next §14 action:** L5 (Xeno-Canto augmentation for 234 Pantanal
species) is now the highest-priority remaining lever from §14.4.
Unlike L2, L5 directly targets the 2026 species distribution rather
than relying on transferable features from a different dataset.
Scope refined in §14.10.11 below.

### 14.10.11 L5 SCOPE REFINEMENT — diagnostic probes 2026-04-18

Two diagnostic probes ran after the L2 closeout to size L5 properly.
Both updated the picture significantly — L5 is still the right lever
but the headline rescue target shifts away from the bird corpus.

**Probe A — per-class_name AUC on baseline A1 fold-0 val.**
Restored production baseline (LB 0.931 ckpt, JIT-extracted) and
scored the 75 species present in soundscape val by taxonomy class:

| class    | n_sp | n_pos_total | mean_auc | median_auc | <0.55 |
|----------|------|-------------|----------|------------|-------|
| Aves     |  28  |   824       | 0.902    | 0.923      |   0   |
| Amphibia |  17  |  4174       | 0.720    | 0.773      |   4   |
| Mammalia |   4  |    84       | 0.724    | 0.706      |   2   |
| Reptilia |   1  |    26       | 0.618    | —          |   0   |
| Insecta  |  25  |  1136       | **0.584**| **0.577**  |  11   |

Insecta is the dominant macro-AUC drag: 33 % of the scored class
denominator at near-random AUC. Aves is already saturated (median
0.92). The "bird competition" framing is misleading — Pantanal
weights every taxon equally, and **bird headroom is the smallest
non-trivial contributor** to remaining macro-AUC gain.

The 11 sub-0.55 Insecta classes are all `47158son*` — Cicadidae
sonotypes (acoustic pseudo-species defined by call pattern, not
biology). 0/25 sonotypes appear in `train_audio` (focal corpus),
so the model has zero training signal for them; the classifier head
for those classes is effectively untrained → AUC ≈ 0.5 noise.

Artifact: `data/probe_class_name_auc.csv`.

**Probe B — sonotype labels in train_soundscapes.**
Tested whether sonotype training signal exists *anywhere* under the
current data layout:

- All 25 sonotypes carry segment-level positives in
  `train_soundscapes_labels.csv` (1136 segments across 66 files).
- BUT `build_soundscape_val()` consumes the entire CSV with no
  train/val split — `train_soundscapes` IS the val set.
- 8/25 sonotypes appear in only **one** file → cannot be split
  train/val by file even via GroupKFold (a one-file class is
  100 % train or 100 % val, never both).
- External Cicadidae acoustic data (e.g. iNaturalist, Xeno-Canto)
  will not carry these site-specific sonotype IDs — they are
  Pantanal-annotation artefacts, not biological taxa.

**Conclusion: sonotype rescue is structurally blocked under the
current val contract.** The only paths are (a) sacrifice the
0.7414 baseline and rebuild val on a partial-file split (loses 8
sonotype classes from val entirely), or (b) abandon the sonotype
sub-problem.

**Revised L5 EV ceiling (keeping current val):**

| sub-lever              | n_sp | current AUC | plausible AUC | ΔAUC × weight (n_sp/75) | macro EV |
|------------------------|------|-------------|---------------|-------------------------|----------|
| L5a Aves (Xeno-Canto)  |  28  | 0.902       | 0.92          | 0.018 × 0.373           | +0.007   |
| L5b Amphibia (AnuraSet)|  17  | 0.720       | 0.85          | 0.130 × 0.227           | +0.030   |
| L5b Mammalia (iNat/YT) |   4  | 0.724       | 0.82          | 0.096 × 0.053           | +0.005   |
| Insecta sonotypes      |  25  | 0.584       | unrecoverable | —                       |   0      |

Total realistic L5 macro-AUC EV ≈ **+0.04** local. LB transfer
historically discounts by ~⅔ for soundscape generalization, so a
realistic LB envelope is **+0.01 – +0.025** for the full L5 program.

**Sequencing recommendation — L5b-Amphibia first.**
- Highest single-lever EV (+0.030 local).
- Real biological taxa with stable scientific names → external
  databases align cleanly.
- AnuraSet (peer-reviewed, ~10k labeled clips of Neotropical anurans
  including Pantanal-relevant species) is the canonical source.
- iNaturalist Sounds API as a fallback for any species missing from
  AnuraSet.
- Does NOT touch the val contract — additional `train_audio`-style
  focal clips, scored against the same soundscape val.

**Open decisions before kickoff:**
1. Per-species clip cap (suggest 100 — match the median Aves count
   of 168 to avoid amphibian over-representation in macro AUC
   shaping during training).
2. Whether to use AnuraSet's existing strong-label segments or
   re-process to focal 5 s clips matching the A1 pipeline (the
   latter is safer; AnuraSet annotations are call-level not
   clip-level).
3. Loss weighting: leave ASL γ unchanged or up-weight Amphibia
   classes to compensate for the residual under-coverage. Suggest
   leave unchanged for the first iteration to keep the change
   single-variable.

**Next ⬜ action:** confirm AnuraSet license + species-overlap with
Pantanal Amphibia list (17 species), then write a fetch +
preprocess script analogous to `pretrain_a1_2025.py` but writing
into `data/external/anuraset_focal/`.

### 14.10.12 L5b-Amphibia survey + plan 2026-04-18

#### A. AnuraSet ↔ Pantanal overlap

AnuraSet (Cañas et al. 2023, *Sci Data* 10:771,
doi:10.5281/zenodo.8342596) is **CC-BY-1.0** — fully redistributable
with attribution. 42 anuran species, 1612 1-minute field recordings
(22.05 kHz mono), ~27 h. Provides per-recording Audacity-format
strong labels (`t_start \t t_end \t CODE_Q`) and a 6-letter species
CODE table.

Joining AnuraSet's `species.csv` against the 17 Pantanal Amphibia
present in soundscape val (`data/probe_class_name_auc.csv`) yields
**10 overlap species** covering **78.6 %** of Amphibia val positives
(3282 / 4174):

| Pantanal species          | AnuraSet code | val n_pos | val AUC |
|---------------------------|---------------|-----------|---------|
| Pithecopus azureus        | PITAZU        | 626       | 0.462 ⬅ |
| Dendropsophus nanus       | DENNAN        | 666       | 0.884   |
| Leptodactylus fuscus      | LEPFUS        | 426       | 0.773   |
| Boana raniceps            | BOARAN        | 420       | 0.736   |
| Physalaemus albonotatus   | PHYALB        | 350       | 0.928   |
| Leptodactylus elenae      | LEPELE        | 310       | 0.820   |
| Scinax nasicus            | SCINAS        | 346       | 0.902   |
| Leptodactylus podicipinus | LEPPOD        |  72       | 0.736   |
| Elachistocleis bicolor    | ELABIC        |  48       | 0.888   |
| Dendropsophus minutus     | DENMIN        |  18       | 0.327   |

*Pithecopus azureus* is the single biggest single-species rescue
target — 626 val positives at 0.46 AUC ≈ half the Amphibia drag.

#### B. iNaturalist fallback survey for the 7 missing species

| species                   | val n_pos | val AUC | iNat n_obs | verdict           |
|---------------------------|-----------|---------|------------|-------------------|
| Adenomera guarani         | 158       | 0.476   | 0          | unsourceable      |
| Chiasmocleis mehelyi      |  24       | 0.571   | 0          | unsourceable      |
| Physalaemus biligonigerus |  46       | 0.611   | 43         | usable, real EV   |
| Pseudis platensis         | 298       | 0.906   | 10         | already strong    |
| Rhinella diptycha         |  18       | 0.928   | 32         | already strong    |
| Scinax acuminatus         | 344       | 0.778   |  2         | too few clips     |
| Trachycephalus typhonius  |   4       | 0.516   | 15         | tiny val signal   |

Only *Physalaemus biligonigerus* (43 iNat obs vs 0.39 AUC headroom)
carries meaningful EV from iNat. Xeno-Canto blocked: API v3 (post
Oct-2025) requires a registered key; website fronted by Anubis bot
protection. **iNat scope is deferred** to a follow-up if AnuraSet
alone doesn't clear the LB gate.

*Adenomera guarani* (158 val pos, 0.476 AUC) is the biggest
unsourceable Amphibia drag — comparable to one full sonotype in
damage. Truly stuck unless a Brazilian herpetology dataset surfaces.

#### C. Plan-phase strong-label yield

`src/fetch_anuraset.py --phase plan` parsed all 1612 strong-label
.txt files (16 000 segments total; 2 files dropped on parse, 0.1 %
loss). Quality breakdown across the dataset: H=74 (0.5 %), M=7098,
L=8828. Filtering to the 10 overlap species and quality H+M only:

| species (8 viable)       | code   | segs  | recordings | sites             |
|--------------------------|--------|-------|------------|-------------------|
| Dendropsophus minutus    | DENMIN | 1047  | 168        | INCT17,20955,4    |
| Pithecopus azureus       | PITAZU |  373  |  72        | INCT17,41         |
| Dendropsophus nanus      | DENNAN |  299  |  87        | INCT17            |
| Physalaemus albonotatus  | PHYALB |  228  |  65        | INCT17            |
| Leptodactylus podicipinus| LEPPOD |  112  |  82        | INCT17            |
| Leptodactylus fuscus     | LEPFUS |  104  |  20        | INCT17            |
| Elachistocleis bicolor   | ELABIC |   83  |  21        | INCT20955         |
| Boana raniceps           | BOARAN |   60  |  18        | INCT17            |
| **TOTAL**                |        | **2306** | **328 unique** |               |

**2 species drop out** of H+M filter:
- Leptodactylus elenae (LEPELE): 11 total segments, all L. Skip.
- Scinax nasicus (SCINAS): 7 total segments, all L. Skip.

Both are already strong (0.82 / 0.90 AUC) — not headroom losses.
Net: 8 of 10 species, covering ~63 % of Amphibia val positives
(2626 / 4174).

**Quality choice — H+M not L+M+H.** AnuraSet's `L` label means
*quiet* calls, not noisy annotations. For first iteration we are
conservative: 2306 H+M segments. If that doesn't move LB, relaxing
to include L (4× more data, ~9300 segments) is a documented
follow-up — Pantanal soundscapes are predominantly low-SNR so L
clips may actually be MORE on-distribution than H.

#### D. Pipeline + budgets

`src/fetch_anuraset.py` is phased + idempotent:
- `--phase meta`  ~1 MB (done 2026-04-18 22:19)
- `--phase plan`  no download (done 2026-04-18 22:21)
- `--phase audio` 7.21 GB raw_data.zip + selective extract of 328
                  recordings (~1.5 GB extracted on disk)
- `--phase cut`   re-cut each H+M segment into a 5 s clip centered
                  on the segment midpoint, resampled 22.05 → 32 kHz,
                  written as OGG/Vorbis at
                  `data/external/anuraset_focal/<primary_label>/AS_*.ogg`
                  (~250 MB total)
- `--phase csv`   emit `data/processed/anuraset_supplement.csv` in
                  full `train.csv` schema (15 cols, license=cc-by,
                  collection=AnuraSet, attribution=
                  "AnuraSet (Canas et al. 2023)")

Does NOT mutate `train_folds.csv`. Concat into the training corpus
is a separate explicit step.

**Realistic L5b-Amphibia EV envelope:**
- Local val ceiling: +0.030 macro AUC (per the §14.10.11 estimate,
  reduced by ~16 % loss of LEPELE+SCINAS coverage → ~+0.025 macro).
- LB transfer historically discounts ~⅔ for soundscape generalization
  → realistic LB envelope **+0.005 to +0.015**.

**Next ⬜ action:** `--phase audio` is launched in background.
Monitor and proceed to `--phase cut` + `--phase csv` once download
completes.

### 14.10.13 L5b-Amphibia fold-0 result + LB push (2026-04-19 00:48 UTC)

**Pipeline executed**: `--phase audio` → `--phase cut` → `--phase csv`
yielded 2305 OGG clips (8 species, H+M only) at
`data/external/anuraset_focal/`. Supplement merged via
`merge_anuraset_into_folds.py` (idempotent, backup
→ `train_folds_pre_anuraset.csv`):
35549 → 37854 rows, fold counts `{-1: 2305, 0: 7110, 1: 7109,
2: 7110, 3: 7110, 4: 7110}`. AnuraSet rows use `fold=-1` so they
join every train split but never enter val.

Path-aware loader `BirdTrainDatasetA1` (subclass of
`BirdTrainDataset`) routes by `collection` column.
Smoke test + targeted 8-row AnuraSet load both passed.

**Fold-0 training run (hybrid loss, 25 epochs, 1h 42m):**

| Epoch | val_roc_auc | note |
|------:|------------:|:-----|
| 1     | 0.6640 | warmup |
| 4     | 0.7340 | first plateau |
| 7     | 0.7327 | flat (almost killed early) |
| 10    | 0.7419 ★ | first beat of baseline 0.7414 |
| 14    | 0.7926 ★ | rapid climb |
| 22    | **0.8382** ★ BEST | +0.097 vs baseline |
| 25    | 0.8213 | end |

**Per-class AUC delta vs LB-baseline JIT (`a1_fold0.pt`):**

LB-baseline JIT and local `_asl.pt` give *identical* per-class AUCs
(macro 0.7414 to 6 decimals), so the comparison is loss-agnostic.

| class | n_sp | baseline | new | Δ | data added? |
|-------|---:|--------:|----:|------:|:-----------:|
| Amphibia | 17 | 0.7201 | 0.8034 | **+0.083** | yes |
| Aves | 28 | 0.9015 | 0.8661 | −0.035 | no |
| Insecta | 25 | 0.5840 | 0.8189 | **+0.235** | **no** |
| Mammalia | 4 | 0.7244 | 0.8943 | **+0.170** | **no** |
| Reptilia | 1 | 0.6178 | 0.9055 | **+0.288** | **no** |

**The +0.097 macro is concentrated in classes that received zero new
training data.** The Pantanal insect sonotypes (`47158sonNN`) move
the most: e.g. son14 0.20 → 0.84, son22 0.50 → 0.99, son20 0.52 → 0.93.
Math: Insecta alone (25/75 species) contributes 0.078 of the +0.097
macro gain; Amphibia (17/75 species, where we added data) only
contributes 0.019.

**Hypotheses for the unexpected per-class profile:**
1. **Hybrid loss artifact**: the original 5-fold A1 was hybrid loss
   too, but maybe regularization noise; would show up if we re-ran
   without AnuraSet and got similar gains anyway.
2. **Pantanal acoustic ambience leakage**: AnuraSet field recordings
   from INCT sites (2019–2020) and train_soundscapes from S08/S09 sites
   (2025) share Pantanal background sounds. Training on AnuraSet may
   improve "Pantanal-ness" features that boost val sonotype recognition.
   Filename/site-level contamination ruled out (different naming
   conventions, different years).
3. **Pure regularization**: 6.5 % more training data improves the
   embedding for everything via better representation learning.

**Diagnostic options:**
- (A) **Train hybrid-loss fold-0 with NO AnuraSet** to isolate the
  loss/training noise effect. ~1h 45m. Would tell us how much of
  +0.097 is from the supplement vs. just running more epochs.
- (B) **Push fold 0 to LB now** and let the leaderboard arbitrate.
  Replace just `kaggle_datasets/a1-effb0-ckpts/a1_fold0.pt` with the
  AnuraSet-trained JIT, leave folds 1/2/4 as baseline. Mixed ensemble
  → measures the marginal effect of one fold swap.

**Decision: push fold 0 to LB (option B).** The diagnostic in (A)
would clarify *attribution* but not *outcome* — what matters is
whether this lifts LB. If LB moves ≥ +0.003 the supplement is alive
and we proceed to all folds; if flat/negative, the +0.097 val gain
is val-set-specific and we kill the line.

**LB push procedure:**
1. Back up `kaggle_datasets/a1-effb0-ckpts/a1_fold0.pt` →
   `archive_lbbase/a1_fold0.pt` (per memory feedback).
2. Re-trace `models/a1/a1_*_fold0_seed42_hybrid.pt` to JIT and
   overwrite `kaggle_datasets/a1-effb0-ckpts/a1_fold0.pt`.
3. `kaggle datasets version` push, bump dataset-metadata title.
4. Re-run inference notebook on Kaggle, submit, observe LB.

**Kill rule:** if LB ≤ baseline 0.931, restore `a1_fold0.pt` from
backup and revert dataset version. Document outcome in §14.10.14.

### 14.10.14 L5b-Amphibia KILLED — LB 0.780 collapse, ambience leakage confirmed (2026-04-19 15:55 UTC)

**v53 result.** Kaggle notebook `birdclef-2026-protossm` v53 (AnuraSet-
supplemented fold-0 swapped into the 4-fold A1 ensemble, rank-fused at
w=0.10 with Perch B1) scored **LB 0.780** on the hidden test set —
a **−0.151 collapse** against the LB 0.931 baseline. Kill rule from
§14.10.13 fires.

**JIT round-trip diagnostic (the cheap sanity check).**
`probe_class_name_auc.py --jit --ckpt-path kaggle_datasets/a1-effb0-ckpts/a1_fold0.pt`
on the 1478-file soundscape val gave **macro AUC 0.8382** — identical
to the raw `_hybrid.pt` training-time val. The TorchScript trace is
clean. The 0.84 → 0.78 gap is not a trace bug; it is a real
distribution shift between `train_soundscapes` and the hidden test.
Artifact: `data/probe_class_name_auc_anuraset_jit.csv`.

**Diagnosis: Pantanal acoustic ambience leakage (Hypothesis #2 from
§14.10.13).** AnuraSet INCT site recordings (Pantanal, 2019-2020,
22.05 kHz field microphones) and `train_soundscapes` S08/S09 recordings
(Pantanal, 2025) share Neotropical background structure —
nocturnal insect chorus, humidity-driven band-pass acoustics, wind
profiles, distant anuran chorus. The supplemented fold-0 learned
features tied to "Pantanal-ness" rather than to the Amphibia signals
we intended. Evidence, from §14.10.13:

- Amphibia (17 sp, data added)  : Δ = +0.083
- Aves     (28 sp, **no** data)  : Δ = −0.035
- Insecta  (25 sp, **no** data)  : Δ = **+0.235**
- Mammalia (4 sp, **no** data)   : Δ = **+0.170**
- Reptilia (1 sp, **no** data)   : Δ = **+0.288**

Classes with *no new training data* contributed 2–3× more macro AUC
gain than the class we actually supplemented. Hidden-test soundscapes
do not share that Pantanal-site background, so the lift evaporates —
and the small -0.035 drop on Aves (the largest sub-class) is enough,
when combined with the loss of whatever true signal the baseline
fold-0 carried on hidden test, to drag LB from 0.931 to 0.780.

**Restoration (2026-04-19 15:46 UTC).** Kaggle dataset version revert
pushed:
1. `cp kaggle_datasets/_backups/a1_fold0_lbbase_20260419.pt kaggle_datasets/a1-effb0-ckpts/a1_fold0.pt`
   (md5 confirmed).
2. `kaggle datasets version -p kaggle_datasets/a1-effb0-ckpts -m "revert fold 0 to LB 0.931 baseline (AnuraSet fold0 scored LB 0.780)"`
   → status `ready`. Any future notebook run will load the LB-baseline
   fold-0. No notebook re-push needed; v53 submission.csv stays on the
   record at 0.780 but is ignored for production.
3. Kept AnuraSet training artifacts on local disk
   (`models/a1/a1_*_fold0_seed42_hybrid.pt`,
   `data/external/anuraset_focal/`,
   `data/processed/anuraset_supplement.csv`,
   `train_folds.csv` still contains the 2305 AnuraSet rows with
   fold=-1; original backed up at `train_folds_pre_anuraset.csv`).
   Reversing the train_folds concat is safe and idempotent via
   `cp train_folds_pre_anuraset.csv train_folds.csv`, but is NOT
   required since the merged csv only affects training, not
   inference.

**What this kills structurally.**
- L5b-Amphibia as implemented (raw AnuraSet field recordings →
  focal 5 s clips → concat into train_folds.csv with fold=-1) is
  dead. External Pantanal field recordings CAN be used to supplement
  focal training **only** if the recording site acoustics are
  neutralized first. Options that might preserve EV:
  - Heavy background subtraction / spectral whitening on source clips
    so the model only sees the call, not the ambience.
  - Foreground-only re-synthesis (isolate call band, mix onto diverse
    non-Pantanal backgrounds before training).
  - Mixup with non-Pantanal soundscapes during training so ambience
    features get regularized away.
  - None of these are cheap, and the EV ceiling is back down to
    the +0.007–+0.015 band (§14.10.11 Amphibia row).
- L5a-Aves (Xeno-Canto scraping for the 28 Pantanal Aves at
  0.902 → plausibly 0.92) is structurally different — Xeno-Canto
  recordings are globally distributed and would not carry the
  Pantanal-background bias. **L5a remains viable.** Its lift
  ceiling is lower (+0.007 local, ~+0.002-+0.005 LB) but the
  contamination risk is low.
- L5b-Mammalia (iNat/YouTube) lands in the same ambience-bias
  trap as AnuraSet if sourced from Neotropical field recordings.
  Suggest limiting mammal sources to captive / studio recordings
  if we ever revisit.

**New memory.** Save `project_l5b_amphibia_killed.md` with the
ambience-leakage pattern. This is a reusable rule: any external
audio whose RECORDING CONDITIONS overlap with the val set but not
the hidden test risks inflating val while hurting LB. The val set
is acoustically narrower than we had been treating it.

**Next ⬜ action:** per §14.4 / §14.10.11, the remaining untried
levers are (in priority order):
1. **L5a Xeno-Canto Aves scraping** — +0.002–+0.005 LB envelope,
   low contamination risk, unblocked (previously blocked on XC v3
   API key; need to recheck if API key acquisition is still gated
   by Anubis bot protection as noted in §14.10.12). Lift ceiling
   is modest but the risk profile is right.
2. **L6 OpenVINO fp16 inference for A1** — indirect; frees notebook
   wall budget to add a 2nd backbone (L4 A3-revival). Zero LB lift
   on its own.
3. **L4 A3-revival** — ECA-NFNet-L0 + multi-year pretrain init.
   Previously bounded by the A3 fold-0 val ceiling (0.7458) but
   pretrained init might break that ceiling. Bigger lift (+0.002–
   +0.008) than L5a but bigger lift-to-effort ratio; needs the L6
   wall-time freeing first.
4. **L7 Silero-VAD speech cleaning** — trivial recipe, marginal
   lift, ambience-neutral.
5. **L8 "nocall" synthetic class** — low expected delta.

Given we just paid an LB probe on a high-risk lever, the sensible
next probe is low-risk + cheap: **L5a Xeno-Canto Aves**. If the XC
v3 API key is unblocked, fetching ~100 recordings for each of the
28 Pantanal Aves is a 1-2 day effort with a low floor. If XC is
still blocked, **skip L5a entirely** and go to L6→L4.

**Decision required from the user when picking up:** confirm
L5a as the next lever (or select L6→L4 / L7 / L8 / park). The
Kaggle-side is stable at LB 0.931 baseline; there is no urgency.

---

### 14.10.15 Salvage diagnostic — fold-0 re-run WITHOUT AnuraSet (2026-04-19 16:05 UTC, running)

**Motivation.** §14.10.14 killed L5b-Amphibia as-implemented at LB 0.780,
diagnosing Pantanal-ambience val leakage. Before committing to any
salvage path (mixup-over-varied-backgrounds, head-only fine-tune,
Xeno-Canto substitution), it is worth spending ~1h 45m of GPU time to
directly test the Hypothesis #1 vs #2 split from §14.10.13:

- **Hypothesis #1** (hybrid loss / training-noise drives the gain):
  this diagnostic rerun — same recipe, same hybrid loss, same seed,
  same 25 epochs — produces ≈0.84 val even WITHOUT AnuraSet. If so,
  the +0.097 lift is attributable to the hybrid-loss recipe alone,
  AnuraSet is an innocent bystander, **and the val metric itself is
  unstable** — which would change what "LB 0.931 baseline" means and
  affect every future lever's gate.
- **Hypothesis #2** (ambience leakage, confirmed by §14.10.14 LB):
  this rerun produces ≈0.74 val (baseline). AnuraSet is the cause
  of the +0.097 lift. Salvage paths in §14.10.14 "What this kills
  structurally" become worth engineering.

This diagnostic is cheap (no LB slot, ~1h 45m GPU) and binary-valued
so the downstream sequencing is unambiguous.

**Procedure:**
1. `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid.pt`
   (the AnuraSet-trained artifact, val 0.8382) archived to
   `models/a1/archive_anuraset/` so the new run does not overwrite it.
2. `train_folds.csv` reverted to pre-AnuraSet state via
   `cp train_folds_pre_anuraset.csv train_folds.csv` (35549 rows,
   no fold=-1 AnuraSet rows).
3. Launch `python -u src/train_a1.py --fold 0 --epochs 25 --loss hybrid --mixstyle-p 0.5`
   (same command that produced the AnuraSet fold-0, only the training
   corpus differs). Log to
   `log/train_a1_fold0_no_anuraset_<TS>.log`.
4. On completion, compare the final `val_roc_auc` to 0.8382.

**Decision rule:**

| final val_roc_auc | diagnosis | next action |
|--:|---|---|
| ≥ 0.82 | Hypothesis #1: hybrid-loss noise. AnuraSet isn't driving the lift. Val metric is unstable. | Re-examine §14 lever gates that rely on fold-0 val as a kill criterion. Do NOT invest in L5b salvage — the whole premise collapses. |
| 0.76 – 0.82 | Partial: AnuraSet contributes some lift but not all. Mixed cause. | Still treat L5b salvage as low EV. Focus on the structural fix (mixup/head-only/XC substitution) but expect smaller lifts than originally modeled. |
| ≤ 0.76 | Hypothesis #2 confirmed: AnuraSet IS the cause of the +0.097 val lift. Ambience leakage fully accounts for it. | L5b salvage paths (mixup-with-varied-backgrounds, head-only fine-tune, XC Amphibia substitution) become worth engineering. Pick whichever has the lowest contamination risk. |

**Post-diagnostic bookkeeping regardless of outcome:**
- Restore `train_folds.csv` to the AnuraSet-merged state
  (`cp train_folds_pre_anuraset.csv train_folds.csv` is reversible via
  `src/merge_anuraset_into_folds.py` — idempotent) ONLY IF we decide
  to invest in a salvage. Otherwise leave reverted.
- The JIT export + Kaggle dataset push is NOT part of this diagnostic —
  LB 0.931 baseline stays live on Kaggle regardless.

### 14.10.15 RESULT — Hypothesis #2 confirmed (2026-04-19 17:31 UTC)

**Diagnostic final: `val_roc_auc = 0.7220`** (epoch 25, new BEST each
epoch from 13 onwards — no overfitting jitter, just a slow climb). Below
the 0.76 threshold. AnuraSet is *the* cause of the +0.097 macro lift.

**Comparison table:**

| recipe                                    | fold-0 val | LB fold-0 swap |
|-------------------------------------------|-----------:|---------------:|
| production baseline (ASL, no AnuraSet)    | 0.7414     | 0.931 (ensemble) |
| hybrid, no AnuraSet (this diagnostic)     | 0.7220     | —               |
| hybrid, WITH AnuraSet                     | 0.8382     | 0.780           |

Two sub-findings fall out of the triangle:
1. **Hybrid loss on its own is a slight regression** (−0.019 val vs ASL
   baseline). For any salvage, use ASL to match production, not hybrid.
   The AnuraSet run happened to use hybrid; if we revisit, switch to ASL.
2. **The full +0.116 val lift from 0.7220 → 0.8382 is attributable to
   AnuraSet**, and that +0.116 is almost entirely poisoned (LB went
   0.931 → 0.780). Net: AnuraSet-as-concat provides ~zero hidden-test
   signal and a lot of Pantanal-ambience memorization.

**Decision per §14.10.15 table:**
≤ 0.76 row fires → Hypothesis #2 confirmed → "L5b salvage paths become
worth engineering." Choices per §14.10.14:

- **Option A — mixup-over-varied-backgrounds** (cheapest). ~50 lines in
  `BirdTrainDatasetA1`: for any AnuraSet row, force-mix with a
  non-Pantanal audio segment at ratio ~0.3–0.7 before spectrogram
  computation. Non-Pantanal source pool: non-Brazilian XC recordings in
  the existing train_audio corpus + FreeSound/ESC-50 ambient clips.
  Retrain fold-0, ASL loss (not hybrid), 25 epochs. ~2 h wall +
  ~½ day to wire the mixer and validate the background pool.
  If fold-0 val lift is real (0.74 → ≥0.76 with NO regression on the
  per-class profile), push to LB.

- **Option B — Amphibia-head-only fine-tune** (structural bound on
  ambience leakage). Freeze LB-baseline backbone weights; retrain only
  the `cls_conv` head on a concat of AnuraSet + train_audio Amphibia
  rows. Backbone cannot absorb Pantanal background features. Highest
  confidence ceiling, but lower expected lift (the head alone has
  limited capacity to improve discrimination if the features aren't in
  the frozen backbone). ~1 day to wire the frozen-backbone trainer.

- **Option C — Xeno-Canto Amphibia** (substitutes the data source).
  Re-fetch the 17 Pantanal Amphibia species from XC (globally
  distributed, no Pantanal-site bias). Blocked on XC v3 API key — open
  question whether a registered key can be obtained. If yes, it is the
  cleanest salvage path and also opens L5a Aves. If no, skip.

- **Option D — drop L5b entirely** and pivot to L5a (XC Aves) or L6→L4.

**Recommendation:** Option A (mixup) first — it is the cheapest test of
whether the AnuraSet call content carries signal at all once the site
ambience is regularized out. If mixup succeeds (LB ≥ +0.003), try
Option B or C to extract more. If mixup fails, the call content itself
doesn't transfer and we should drop L5b permanently.

**Bookkeeping done (2026-04-19 17:31 UTC):**
- Diagnostic fold-0 ckpt at
  `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid.pt`
  (val 0.7220, **this is NOT the production ASL baseline** — do not
  swap into Kaggle ensemble).
- AnuraSet-trained fold-0 ckpt remains at
  `models/a1/archive_anuraset/a1_*_fold0_seed42_hybrid.pt`.
- `train_folds.csv` is **still in pre-AnuraSet state** (35549 rows).
  Re-merge with `python src/merge_anuraset_into_folds.py` before any
  L5b salvage retrain.
- Full diagnostic log at
  `log/train_a1_fold0_no_anuraset_20260419_160631.log`.

---

## 14.10.16 L5b salvage — AnuraSet + background mixup recipe (2026-04-19 draft)

**Goal:** test whether the AnuraSet call content carries any
hidden-test-transferable signal, by regularizing the Pantanal-ambience
shortcut out of training. If this works, LB recovers to ≥0.934 and
L5b-Amphibia is resurrected. If it fails, the call content does not
generalize and L5b is dropped permanently.

### 14.10.16.1 What changes vs §14.10.13 recipe

| dimension      | §14.10.13 (killed)               | §14.10.16 (salvage)               |
|----------------|----------------------------------|------------------------------------|
| Training data  | `train_folds.csv` + 2305 AnuraSet rows (fold=-1) | **same** |
| Loss           | hybrid BCE+ASL                   | **ASL** (match LB-production baseline; §14.10.15 showed hybrid is −0.019 worse on its own) |
| MixStyle p     | 0.5                              | 0.5 (unchanged) |
| Epochs / seed  | 25 / 42                          | 25 / 42 (unchanged) |
| AnuraSet I/O   | load clip, return as-is          | **load clip, 70 %-chance force-mix with a random non-Pantanal waveform (additive, RMS-matched, α ~ U(0.2, 0.5))** |
| Mixin pool     | —                                | non-AnuraSet rows of `train_folds.csv` whose `latitude / longitude` fall **outside** the Pantanal bounding box (34,107 rows — see §14.10.16.2) |
| Labels         | AnuraSet labels only             | AnuraSet labels only (mixin's labels discarded — mixin treated as "background") |

The only structural change is the mixer in `BirdTrainDatasetA1`. Every
other hyperparameter matches production.

**Dial adjustments from the 2026-04-19 17:40 draft (recorded
2026-04-19 17:55):**

1. **`mixin_alpha_range` lowered `U(0.3, 0.7)` → `U(0.2, 0.5)`.**
   The draft's 0.7 upper bound was aggressive — loud enough to mask
   low-SNR amphibian calls (especially for species like *Dendropsophus
   minutus* whose val AUC of 0.327 suggests the calls are already
   hard to hear). Cap at 0.5 keeps the AnuraSet call audibly dominant
   while still injecting enough background variance to break the
   Pantanal-ambience shortcut.
2. **`mixin_p` lowered `1.0` → `0.7`.**
   At 100 % we never show the model a clean AnuraSet clip, so if the
   call's discriminative info survives only in clean form we lose it.
   70 % keeps the dominant regularizer behaviour of the draft while
   letting ~30 % of AnuraSet passes deliver the raw call signal. Also
   matches standard mixup convention (most implementations default to
   p≤0.5, so 0.7 is still on the aggressive end).
3. **`mixin_p` sanity check — pool size.** Computed on
   `train_folds_pre_anuraset.csv`: with `lat ∈ [−22, −14]` and
   `lon ∈ [−62, −54]` the bbox excludes 1,442 / 35,549 rows (4 %;
   1,259 XC + 183 iNat; top species are all Pantanal-region Aves
   like *magant1*, *hyamac1*, *chacha1*). Mixin pool = 34,107 rows.
   No tightening or loosening needed.

### 14.10.16.2 Mixin pool construction

Filter `train_folds_pre_anuraset.csv` (35549 rows, no AnuraSet) down
to non-Pantanal rows:

```python
PANTANAL_BBOX = {"lat_min": -22.0, "lat_max": -14.0,
                 "lon_min": -62.0, "lon_max": -54.0}
def outside_pantanal(row):
    return not (PANTANAL_BBOX["lat_min"] <= row["latitude"] <= PANTANAL_BBOX["lat_max"]
                and PANTANAL_BBOX["lon_min"] <= row["longitude"] <= PANTANAL_BBOX["lon_max"])
```

Pantanal bounding box covers the biome ~from Corumbá (MS, Brazil)
through to northern Paraguay and eastern Bolivia. Being slightly loose
is fine — the mixin's label doesn't matter; what matters is that the
*acoustic background* is off-distribution from Pantanal. The bbox
excludes ~X rows out of 35549 (to be confirmed at implementation
time; if < 10000 retained, loosen; if > 30000, tighten).

Rows with missing lat/long (a handful exist) are KEPT in the pool —
unknown-location recordings are more likely global-origin than
Pantanal-origin given the prior.

### 14.10.16.3 Mixer implementation (`src/dataset_a1.py`)

Extend `BirdTrainDatasetA1` with an optional `mixin_df` / `mixin_p`:

```python
class BirdTrainDatasetA1(BirdTrainDataset):
    def __init__(self, df, augment=True,
                 mixin_df=None, mixin_p=0.7,
                 mixin_alpha_range=(0.2, 0.5)):
        super().__init__(df, augment=augment)
        self.mixin_df = mixin_df.reset_index(drop=True) if mixin_df is not None else None
        self.mixin_p = mixin_p
        self.mixin_alpha_range = mixin_alpha_range

    def _load_waveform(self, row):
        coll = str(row.get("collection", "") or "")
        if coll == "AnuraSet":
            path = ANURA_FOCAL / str(row["filename"])
        else:
            path = RAW / "train_audio" / str(row["filename"])
        wav = pad_or_crop(load_audio(path), CHUNK_SAMPLES,
                          random_crop=self.augment)

        # Background mixup for AnuraSet rows only (and only in training)
        if (coll == "AnuraSet" and self.augment
            and self.mixin_df is not None
            and random.random() < self.mixin_p):
            mx = self.mixin_df.sample(1).iloc[0]
            mx_wav = pad_or_crop(
                load_audio(RAW / "train_audio" / str(mx["filename"])),
                CHUNK_SAMPLES, random_crop=True)
            # RMS-match so the mixin is audibly present but not dominant
            r_wav = wav.pow(2).mean().sqrt().clamp(min=1e-6)
            r_mx  = mx_wav.pow(2).mean().sqrt().clamp(min=1e-6)
            mx_wav = mx_wav * (r_wav / r_mx)
            alpha = random.uniform(*self.mixin_alpha_range)
            wav = wav + alpha * mx_wav
            # Peak-normalize to avoid clipping beyond the int16 range
            peak = wav.abs().max()
            if peak > 0.99:
                wav = wav * (0.99 / peak)
        return wav
```

Wiring in `train_a1.py`: add a flag `--anuraset-mixup` that, when set,
builds the `mixin_df` (from pre-AnuraSet folds, outside bbox) and
passes it to the `BirdTrainDatasetA1` constructor. Off by default so
non-L5b-salvage runs are unaffected.

### 14.10.16.4 Pilot → full sequencing

**Stage 1 — fold 0 pilot (~1h 45m train + ~10m probe).**
1. Re-merge AnuraSet: `python -u src/merge_anuraset_into_folds.py`.
2. Wire `--anuraset-mixup` and confirm smoke test passes
   (`--fold 0 --epochs 1 --smoke-test --anuraset-mixup`).
3. Launch: `--fold 0 --epochs 25 --loss asl --mixstyle-p 0.5 --anuraset-mixup`.
4. Monitor per-epoch val_roc_auc.
5. On completion, run
   `python -u src/probe_class_name_auc.py --ckpt <new_fold0_asl.pt>
   --out probe_class_name_auc_anuraset_mixup.csv`.

**Stage 2 — apply the go/no-go gate (structural, not just macro).**

| signal                                                            | verdict                              |
|-------------------------------------------------------------------|--------------------------------------|
| macro val < 0.7414 (worse than LB baseline)                       | kill — AnuraSet calls are net noise  |
| macro val ≥ 0.7414 AND Amphibia Δ is the largest positive Δ (Amphibia > Insecta/Reptilia/Mammalia Δs) | push to LB             |
| macro val ≥ 0.7414 BUT Insecta/Reptilia/Mammalia Δs still dominate Amphibia Δ | kill — ambience shortcut survived mixup |
| macro val 0.73–0.7414 AND Amphibia Δ dominates                    | marginal — push to LB anyway (floor is close) |

**Stage 3 — LB probe (only if stage-2 gate passes).** JIT-export the
new fold-0, push as a new dataset version (backup current LB-baseline
fold-0 first), re-run the notebook, read LB.
- LB ≥ 0.934 (+0.003 over 0.931): salvage alive → train folds 1/2/4
  with the same recipe (~5h wall), export all 4, push + re-submit.
- 0.931 < LB < 0.934: marginal; run one more fold for evidence before
  committing to full rollout.
- LB ≤ 0.931: kill permanently. Restore baseline per §14.10.14
  procedure. Document in §14.10.17.

### 14.10.16.5 Known limitations (don't pre-fix)

1. **Mixin's own labels are discarded.** ~5-10% of mixin clips will
   coincidentally contain a call for some species already in the 234-
   class target set, and we'll train with that species labeled 0. This
   is mild label-smoothing-like noise and is acceptable at this scale;
   fixing it would require complex label-merging and obscure the
   experiment.
2. **`mixin_p=0.7` — 30 % of AnuraSet passes are unmixed.** Adjusted
   down from the 1.0 draft (see "Dial adjustments" in §14.10.16.1).
   Trade-off: lets the model see raw AnuraSet calls ~⅓ of the time,
   at the cost of a weaker regularizer. If the pilot fails the
   stage-2 gate because the ambience shortcut survived (large
   Insecta/Reptilia Δ), raising `mixin_p` to 1.0 is the first retry
   lever. If the pilot fails because Amphibia Δ is near zero,
   dropping `mixin_p` to 0.5 (or disabling mixup) is the right
   retry — the model isn't getting enough clean call signal.
3. **Peak-normalization clipping at 0.99.** Rare, mostly for RMS-
   matched mixins that are loud; good enough for A1's mel-spectrogram
   pipeline.
4. **Pantanal bbox coarse.** ~20% of non-AnuraSet rows in the bbox get
   excluded. Some of those would have been fine backgrounds (e.g. a
   single XC recording from Campo Grande). Not worth tuning — the
   pool after filtering is still huge.

### 14.10.16.6 Rollback

If stage-2 gate fails OR stage-3 LB ≤ 0.931:
```bash
# Local:
git checkout src/dataset_a1.py src/train_a1.py  # drop mixer code
# train_folds.csv: leave as-is (AnuraSet rows are harmless when not
# touched by a new training run; reverting is optional).

# Kaggle (only if we pushed a dataset version):
cp kaggle_datasets/_backups/a1_fold0_lbbase_20260419.pt \
   kaggle_datasets/a1-effb0-ckpts/a1_fold0.pt
kaggle datasets version -p kaggle_datasets/a1-effb0-ckpts \
   -m "revert fold 0 to LB 0.931 baseline (§14.10.16 salvage failed)"
```

### 14.10.16.7 Why this might work

The §14.10.15 diagnostic proved the full +0.116 val lift is
AnuraSet-driven. That +0.116 decomposes into two unknown parts:
- `ambience signal` — features that only exist at INCT/Pantanal sites
  and transfer to val but NOT to hidden test
- `call signal` — amphibian-call spectrogram structure that transfers
  to ANY Pantanal soundscape including hidden test

Mixing in varied non-Pantanal backgrounds attacks the first channel
while preserving the second. If the *real* call content is ~zero
(amphibian calls at INCT sites in 2019 are somehow acoustically
different from the calls in 2025 S08/S09 sites — possible but unlikely
for low-complexity anuran vocalizations), mixup won't save us and
stage-2 will flag it. If the call signal is non-trivial (say ~+0.03
local, ~+0.01 LB), we'll see Amphibia move up without Insecta/
Mammalia/Reptilia moving, and LB lifts.

### 14.10.16.8 Budget

| stage              | wall time | LB slots |
|--------------------|-----------|----------|
| Wire + smoke test  | ~½ day    | 0        |
| Fold 0 pilot + probe | ~2 h    | 0        |
| Stage 3 LB probe (if gate passes) | ~30m | **1** |
| Full 4-fold rollout (if LB passes) | ~6 h | 1 |

Worst case (pilot fails gate): ½ day wasted, no LB slot burned.
Best case: +0.003–+0.015 LB after ~1 day total.

---

## 14.10.17 L5b terminally killed — salvage pilot failed Stage-2 gate (2026-04-19)

**Verdict: KILL. Do not push to LB. L5b (AnuraSet supplementation of
fold-0) is terminally closed under both the naive (§14.10.14) and
salvage (§14.10.16) recipes.**

### 14.10.17.1 Pilot result

Ran §14.10.16 recipe (ASL, MixStyle 0.5, 25 ep, seed 42, background
mixup: mixin_p=0.7, α∈U(0.2, 0.5), 34,107-row non-Pantanal mixin pool).

- Best epoch: E20, val_roc_auc = **0.7958** (+0.054 over baseline 0.7414,
  −0.042 below leaky §14.10.14 peak 0.8382).
- Checkpoint: `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_asl.pt`.
- Per-class probe: `data/probe_class_name_auc_anuraset_mixup.csv`.

### 14.10.17.2 Stage-2 structural gate (from §14.10.16.4)

Unweighted class-level macro Δ per class group, vs LB-baseline ckpt:

| class    | leak Δ (§14.10.14) | **mix Δ (§14.10.16)** | mixup effect |
|----------|-------------------:|----------------------:|--------------|
| Amphibia (target) | +0.083 | **+0.044** | halved |
| Aves (birds)      | −0.035 | **−0.081** | made worse |
| Insecta           | +0.235 | **+0.197** | barely reduced |
| Mammalia          | +0.170 | **+0.097** | reduced |
| Reptilia          | +0.288 | **+0.297** | *increased* |

Gate rule: "macro val ≥ 0.7414 BUT Insecta/Reptilia/Mammalia Δs still
dominate Amphibia Δ → kill." All three non-target Δs are 2–7× Amphibia Δ
— Pantanal-ambience shortcut remains the dominant source of the +val
lift. **FAIL.**

### 14.10.17.3 Why the mixer did not work

1. **mixin_p=0.7 leaves 30 % of AnuraSet passes unmixed.** Even at
   α=0.5 ceiling, pure-ambience rows recur every ~3 epochs per clip —
   enough for the model to lock onto the Pantanal shortcut signature.
2. **Reptilia Δ actually rose.** Only 1 class in the val set (1 species,
   n=26); model is clearly picking up "INCT site at night" → Reptilia
   co-occurrence, and the unmixed 30 % is sufficient to reinforce it.
3. **Aves regressed** from −0.035 to −0.081. Mixin-pool clips contain
   bird calls whose labels get discarded (§14.10.16.5 limitation #1);
   at 34,107 rows × 70 % mix rate this introduces meaningful label
   noise on the classes we care about most.
4. The recipe's budget ceiling (mixin_p < 1.0, α ≤ 0.5) was chosen to
   protect amphibian foreground signal — but in practice the foreground
   signal is *too weak to survive even moderate contamination*, while
   the background ambience is *too strong to be masked by moderate
   mixup*. The two constraints are inconsistent.

### 14.10.17.4 Could L5b be made to work?

Only with structural changes beyond this recipe's envelope:

- **mixin_p = 1.0 and α ≥ 0.6:** would further suppress ambience but
  likely erase Amphibia Δ entirely. Net: parity at best, probably loss.
- **Source-swap to globally-sourced amphibian corpus (non-INCT):**
  Xeno-Canto anurans from Amazonia / Atlantic Forest / Cerrado without
  Pantanal site overlap. Would solve ambience problem at source but
  (a) species overlap with val's 17 Amphibia classes is small, (b) the
  competitive LB gain envelope is ≤ +0.01 given only 17 target amphibian
  classes. Not worth the ingest & de-dup work.
- **Background subtraction / foreground re-synthesis:** model-based
  denoising of AnuraSet recordings. Implementation-heavy; unclear if
  pretrained models exist for insect-chorus removal; new failure
  modes.

L5b is **terminally off the priority list**. Shortest path to LB gains
is a non-amphibian track.

### 14.10.17.5 Cleanup

- Production fold-0 remains the LB-baseline ckpt restored in §14.10.14
  (`kaggle_datasets/_backups/a1_fold0_lbbase_20260419.pt`, already
  pushed to `stevewatson999/birdclef-2026-a1-effb0-ckpts` current
  version). No Kaggle action required.
- Salvage ckpt (`models/a1/a1_*_fold0_seed42_asl.pt`) kept on disk for
  diagnostic reference but will NOT be exported or pushed.
- `train_folds.csv` left in re-merged state (37,854 rows incl. 2,305
  fold=-1 AnuraSet rows); backup at `train_folds_pre_anuraset.csv`. Do
  not touch — future non-AnuraSet training reads the fold column, so
  AnuraSet rows (fold=-1) are already excluded from all A1 training
  splits.

### 14.10.17.6 Lessons for future L-tier levers

1. **"Non-bird val lift without matching training data is a red flag"**
   (existing rule from §14.10.14) now strengthened: even **with**
   matching training data, if non-target Δ > target Δ, the training
   data is supplying a non-target shortcut. Always compute per-class-
   group Δ before trusting val lift as LB-predictive.
2. **Mixup does not cleanly separate foreground from background** when
   the background is strong (steady-state insect chorus, humidity
   spectrum) and the foreground is sparse + narrowband (discrete
   anuran calls). Additive RMS-matched mixup preserves both signals;
   the model picks the easier one (background) by default.
3. **Val's acoustic overlap with training augmentations matters.**
   `train_soundscapes` val is narrower than hidden test; any training
   modification that inflates val without a structural mechanism is
   suspect.

### 14.10.17.7 Next lever

Per §14 sequencing, with L5a (Aves) saturated and L5b terminally dead:
- **L5c-Mammalia** is structurally analogous to L5b and likely falls in
  the same Pantanal-ambience trap; skip unless a non-Neotropical field-
  recorded mammal corpus is identified first.
- **L5d-Sonotypes** remains structurally blocked (species mismatch).
- **Return to non-L5 levers.** The tree is L2 / L3 / L1 (all killed,
  see memory), then fresh exploration. Candidates from §14 roadmap:
  - Teacher-ensemble pseudo-label refresh (Track C2 with diversified
    teachers — currently only §10 ProtoSSM).
  - D2 stacking recalibration (α=0 reverted; explore feature-space
    stacks instead of rank-fusion).
  - Track B second Perch consumer with temperature-sharpened outputs.

---

## 14.11 Next-lever shortlist — BirdCLEF 2024 top-3 technique audit (2026-04-19)

Every L1/L2/L3/L5a/L5b lever has been exhausted. Gap to leader is still
~0.02 and the common thread in our failures is val-LB decorrelation
(`train_soundscapes` is Pantanal-narrow; hidden test is broader). Before
inventing more levers, we audited the BirdCLEF 2024 top-3 writeups for
techniques we have not applied. This section is the shortlist.

### 14.11.1 What the top-3 actually did (condensed)

**1st — EfficientNet-B0 + RegNetY-008 ensemble (private 0.693).**
- Mel: n_fft=1024, hop=500, n_mels=128, fmin=40, fmax=15000, power=2.
- **10-sec chunks = two adjacent 5-sec chunks, labels averaged.**
- **Loss: CrossEntropy (softmax), NOT BCE.** Explicit quote: "BCE shows
  significantly worse results."
- AdamW, CosineAnnealingLR, lr 1e-3 to 3e-3, 7–12 epochs, bs 96.
- **Data quality filter:** dropped chunks where Google Bird Vocalization
  Classifier's top prediction disagreed with primary label.
- **Ambient quality filter:** kept 80th percentile by
  T = std + var + rms + pwr (drop loud/noisy).
- Pseudo-labels from Google classifier with 0.05 coefficient (not
  iterative self-distillation — external teacher).
- Inference: sigmoid despite softmax training; **min() across ensemble**
  (not mean) — reported superior.
- 6 EffNet-B0 final ensemble.

**2nd — EffNet-B0 (tf_efficientnet_b0_ns) × 6 diverse configs (private ~0.691).**
- First 5 sec of each recording (not random crop).
- **Ensemble diversity via varied mel params** — n_mels 64/128,
  hop 512–1024, image sizes 256²/128²/64². Same backbone, different
  preprocessing.
- **GEM pooling (learnable)** + 5-dropout stack before FC.
- Loss: mean of BCE + FocalLoss.
- **Checkpoint soup** — averaged weights from epochs 13–50 based on
  local CV (they ran 50 epochs, we run 25).
- Mixup within batch; CoarseDropout.
- **Iterative pseudo-label distillation — 3 rounds.** "Pseudo labeling
  and improving ensemble cycle has high impact." Each round: best
  ensemble → pseudo-labels on test → mixed in with 25–45 % probability
  and random amplitude coefficient 10^U(-0.5, 0.1) → retrain →
  re-predict. Targets = max(original, predicted).
- **Temporal smoothing at inference:** each prediction = self +
  0.5·(left neighbor) + 0.5·(right neighbor).
- Fusion: mean of sigmoid outputs across 6 models.

**3rd — EffViT-b0 + MNASNet-100 two-stage distillation (private 0.690).**
- Random 5-sec crop from first 6 or last 6 sec.
- Mel → 224² or 288²; std-normalized waveform (std=1).
- Data cap: max 500 records/species (most recent); undersampled classes
  upsampled to min 10.
- Loss: BCEWithLogitLoss, **no label smoothing**.
- **Primary labels only; secondary labels MASKED** (not weighted-
  distributed like 1st place). Prevents incorrect gradient on faint
  co-vocalizations.
- No early stopping; uses final-epoch weights.
- **Pseudo-label batch:** 48 × 4 = 192 pseudo-clips added to 128 real
  clips per batch.
- **Prediction smoothing kernel [0.1, 0.2, 0.4, 0.2, 0.1]** convolved
  across adjacent 5-sec clips.
- **Bird-presence enhancement:** final pred += 0.8 · max(per-soundscape
  probability) — boosts species that appear elsewhere in the same file.
- Final ensemble: 14 weights across 3 pipelines (5 + 4 + 5 seeds).

### 14.11.2 What we have NOT tried (shortlist, sorted by cost/signal)

**TIER-1 — zero training cost, notebook-only, directly bankable:**

| # | Technique | Source | Where to apply | Expected LB | Risk |
|---|-----------|--------|----------------|-------------|------|
| T1.1 | **Temporal smoothing conv kernel `[0.1, 0.2, 0.4, 0.2, 0.1]`** across adjacent 5-sec clips | 2nd & 3rd | Kaggle notebook cell 41 (post-fusion, pre-submission) | +0.005–0.015 | very low; smoothing helps LB in every BirdCLEF — only risk is already present in our TTA |
| T1.2 | **Bird-presence enhancement**: `pred += 0.8 · max(per-soundscape pred)` | 3rd | Kaggle notebook cell 41 | +0.003–0.010 | low; stacks with T1.1 |
| T1.3 | **Min-reduce fold ensemble** instead of mean (try on raw softmax before rank fusion) | 1st | Kaggle notebook A1 fold aggregation | ±0.005 | medium; could regress — A/B test |

**TIER-2 — one fold retrain (~1–2 h), moderate signal:**

| # | Technique | Source | Change | Expected LB | Risk |
|---|-----------|--------|--------|-------------|------|
| T2.1 | **CE loss instead of ASL** on fold 0 | 1st ("BCE significantly worse") | `train_a1.py --loss ce` (new code path) | +0.005–0.020 if claim holds | medium; 1st place's claim is their strongest finding |
| T2.2 | **GEM pooling** instead of attention head | 2nd | `src/sed_a1.py` head swap | ±0.005 | low; drop-in |
| T2.3 | **Checkpoint soup** — average epochs 13–25 | 2nd | Train 40 epochs fold 0, average 13–40 | +0.003–0.010 | low; retrain fold 0 with `--epochs 40 --soup` |
| T2.4 | **Mel-diversity ensemble**: retrain one fold with n_mels=64 / hop=1024 / 128² input — adds preprocessing diversity to 4-fold ensemble | 2nd | One new fold with different `mel_cfg` | +0.003–0.012 | low |
| T2.5 | **Data-quality filter**: drop chunks where Perch v2 top prediction ≠ primary label | 1st | `src/dataset_a1.py` pre-filter pass | +0.005–0.015 | low; uses Perch we already have |
| T2.6 | **Pull previous-year BirdCLEF recordings for 2026 species** (2021/2022/2023/2024 training rows where species ∈ 2026 list) | 3rd | New ingest script → merge into `train_folds.csv` with new `collection` tag | +0.010–0.025 | **low; globally-sourced Xeno-Canto, no Pantanal-site trap** |

**TIER-3 — heavy retrain or re-pipeline (multi-day):**

| # | Technique | Source | Change | Expected LB | Risk |
|---|-----------|--------|--------|-------------|------|
| T3.1 | **10-sec chunks with averaged labels** (two adjacent 5-sec) | 1st | Rewrite `dataset_a1.py` chunk logic; full 5-fold retrain | +0.010–0.020 | medium; big rewrite, fusion ckpt path changes |
| T3.2 | **Iterative pseudo-label distillation (3 rounds)** on unlabeled `unlabeled_soundscapes` (not `train_soundscapes`!) | 2nd | New pipeline; requires clean pseudo-label source (NOT §10 ProtoSSM teacher — it's val-leaky, see memory) | +0.015–0.030 | medium; needs clean teacher or careful self-distillation |
| T3.3 | **Secondary-label mask** (primary only, secondaries → 0 with mask) | 3rd | `dataset_a1.py` target construction change | +0.003–0.010 | low; but different from our current distributed secondary weighting |
| T3.4 | **Ambient quality filter** (drop clips with T = std+var+rms+pwr in top 20 %) | 1st | `src/build_folds.py` or dataset filter | +0.003–0.010 | low |

### 14.11.3 Priority order (locked 2026-04-19 post-audit)

1. **Tier-1 post-proc stack (T1.1 + T1.2)** — inference-only, ~10
   notebook lines, 1 LB slot. See Step 1 below.
2. **T2.6 prior-year BirdCLEF data pull** — standalone untapped data
   source; globally-sourced Xeno-Canto, no Pantanal-site trap.
3. **D3 Perch v2 as external pseudo-label teacher / quality filter** —
   single Perch forward-pass over `unlabeled_soundscapes` + training
   data. Simultaneously unlocks T2.5 (quality filter) and T3.2
   (iterative distillation) without needing a new clean teacher.
4. **T3.2 iterative pseudo-label distillation** — 2nd place's "critical
   unlock"; executable once D3 is in place.

Everything else in §14.11.2 is opportunistic — fold in if budget allows
but don't block the critical path on them.

### 14.11.3.a Recommended sequencing (cheapest first, real lift)

**Step 1 — Tier-1 stack (T1.1 + T1.2 + optional T1.3), single LB probe.**
- Edit Kaggle notebook post-processing cell. No retrain.
- Submit once. Expected outcome: 0.936–0.945.
- If T1.3 regresses in A/B testing on validation preds, drop it and
  keep T1.1 + T1.2 only. **Gate:** LB ≥ 0.934 → lock in as new
  baseline; LB < 0.931 → investigate (unlikely), revert.
- **This is the highest signal-to-cost action on the board.** One
  notebook change, one LB slot, two documented techniques from two
  independent top-3 teams.

**Step 2 — conditional on Step 1 outcome:**
- If Step 1 LB ≥ 0.940, the shortest remaining path is T2.1 (CE loss)
  + T2.5 (Perch quality filter) together in a fold-0 retrain. 1 day,
  1 LB slot.
- If Step 1 LB = 0.934–0.940, do T2.3 (checkpoint soup) first — it's
  the lowest-risk training-side win.
- If Step 1 LB < 0.934 (no lift), escalate to T2.1 alone to interrogate
  the loss function hypothesis. CE vs ASL is the strongest "different
  from what we have" signal from the 2024 writeups.

**Step 3 — Tier-3 only if Step 1+2 produce ≥ 0.940.**
- T3.2 (iterative pseudo-label distillation) is the 2nd-place team's
  stated "critical unlock" but requires a clean teacher. Rebuilding
  a clean teacher (Xeno-Canto-only, no `train_soundscapes`) is the
  precondition — see §14.11.4.

### 14.11.5 Data-source unlocks (audit of what more data the top-3 used)

Beyond techniques, the 2024 top-3 used **more data** than we do. Three
concrete flows:

**D1 — Previous-year BirdCLEF training recordings for overlapping species.**
- Source: 3rd place explicitly: "competition data from BirdCLEF 2024
  plus additional data from Xeno Canto and records from previous year
  competitions for the same species, capping 500/species, keeping
  most recent."
- Status: **NOT TRIED.**
- Mechanic: diff the 2026 species list against BirdCLEF 2021/2022/2023/
  2024 species lists; for overlapping species, pull the historical
  training audio (Xeno-Canto-sourced, already public, licensed CC-BY).
  Merge into `train_folds.csv` with a new `collection="BC2021"` etc.
  tag so the dataset router can find them.
- Why it's safe where L5b wasn't: the previous-year recordings are
  **globally sourced Xeno-Canto focal** — not field-mic site recordings
  like AnuraSet. No Pantanal-ambience shortcut. This is structurally
  analogous to L5a (Aves Xeno-Canto supplementation), which we
  previously marked as "saturated" but never verified that *all* prior-
  year overlap was already pulled.
- **Promoted to Tier-2 as T2.6** (see §14.11.2 table).

**D2 — `unlabeled_soundscapes` pseudo-label distillation.**
- Source: all three teams heavily mined this. 2nd place iterated 3
  rounds; 3rd place packed 192 pseudo-labeled clips into every 128-clip
  batch (2:1 pseudo:labeled ratio).
- Status: **partially tried (Track C2 ProtoSSM distillation killed 2026-04-18
  — see memory `project_protossm_teacher_val_leakage`).**
- Blocker: our only existing teacher (§10 ProtoSSM) was trained on
  `train_soundscapes` via GroupKFold, so pseudo-labels on
  `unlabeled_soundscapes` carry indirect val leakage. Cannot use it for
  distillation without contaminating the student.
- Unlock path: **build a clean teacher** that never touched
  `train_soundscapes`. Candidates:
    - Train a teacher on competition train_audio + Xeno-Canto only,
      early-stop by train loss (no val leak).
    - Use a frozen external model (Perch v2 — see D3) to generate
      pseudo-labels directly, bypassing a student teacher.
- This is the **single biggest remaining infrastructure unlock.**
  Without it, iterative distillation (T3.2) is blocked.

**D3 — External pseudo-label teacher (Google Bird Vocalization Classifier / Perch).**
- Source: 1st place explicitly. "Added pseudo-labels from Google
  classifier (0.05 coefficient)."
- Status: **NOT TRIED.**
- Mechanic: we already have Perch v2 cached at
  `data/processed/perch_cache/` and `data/kaggle_perch_cache/`. Can
  run Perch forward-pass over `unlabeled_soundscapes`, keep top-k
  predictions with a confidence threshold, use them as:
    - **Pseudo-label signal** (T3.2-style, weighted 0.05 per 1st place)
    - **Data-quality filter** for training data (T2.5)
- Why it's safe: Perch v2 was pretrained on global Xeno-Canto + iNat,
  never saw `train_soundscapes`. No val-leakage path.
- **Replaces the ProtoSSM teacher for D2 distillation without needing
  to train a new clean teacher first.** This is the cheaper unlock for
  iterative distillation.

**Data-source sequencing recommendation:**
1. After Tier-1 LB probe (§14.11.3 Step 1), run **T2.6 (prior-year
   data pull)** as a standalone experiment — cleanest new-data win.
2. In parallel, use **D3 (Perch-as-teacher)** to generate
   `unlabeled_soundscapes` pseudo-labels and run **T2.5 (Perch quality
   filter)** on training data. Same Perch forward-pass serves both.
3. With D3 pseudo-labels in hand, attempt **T3.2 (iterative distillation)**
   — no polluted-teacher blocker.

### 14.11.4 What this audit does NOT solve

1. **Val-LB decorrelation remains.** None of these techniques change the
   fact that `train_soundscapes` is narrower than hidden test. They are
   all believed to be LB-predictive because the 2024 top teams reported
   LB lifts — but the 2024 test set is different from 2025/2026 Pantanal.
   Tier-1 (post-processing only) is safest here because it doesn't
   depend on val gating.
2. **ProtoSSM teacher pollution** (memory: `project_protossm_teacher_val_leakage`)
   still taints any self-distillation path (T3.2) until a clean teacher
   is trained. Building a clean teacher is its own ~1-day task and is
   the **single biggest infrastructure unlock remaining** — without it,
   Track C is closed, 2nd-place recipe is closed, and any future noisy-
   student run is closed. Consider it as §14.12.
3. **Calibration.** None of the 2024 writeups discuss explicit
   calibration; their LB numbers come from strong models + post-proc.
   We may still have headroom in Platt scaling or isotonic recalibration
   of fusion outputs, but it's not a top-3 lesson.

---

## 14.11.6 v54 Tier-1 post-proc KILLED + T2.6 data pull scoped (2026-04-20)

**v54 LB 0.919 (−0.012 vs 0.931 baseline). T1.1 + T1.2 stack REGRESSED.**

### 14.11.6.1 v54 verdict

Pushed v54 with T1.1 (temporal smoothing kernel [0.1, 0.2, 0.4, 0.2, 0.1])
+ T1.2 (bird-presence enhancement w=0.30). Baseline is 0.931. Result
was 0.919 — the techniques from 2024 top-3 regressed our pipeline.

**Root-cause hypothesis:** our existing post-processing pipeline
(cell 42) already contains:
- `delta_shift_alpha = 0.20` (adaptive window-neighbor smoothing)
- `rank_aware_power = 0.4` (rank-space probability reshaping)
- Per-class thresholds tuned against the *unsmoothed* distribution
- Per-taxon temperature scaling

Adding T1.1 + T1.2 **on top** of this stack compressed the dynamic
range the per-class thresholds were tuned against. The 2024 winners
applied smoothing ONCE, starting from raw sigmoid; we applied it
twice. Technique transfer from external writeups fails when the
destination pipeline already contains overlapping mechanisms.

### 14.11.6.2 Revert executed

Notebook restored from `jupyter/protossm-postproc/
birdclef2026-protossm-postproc.ipynb.bak_pre_t1`. Cell 42 is back to
the bit-identical LB-0.931 path. Production baseline unchanged.

### 14.11.6.3 Corrected lesson for §14.11 shortlist

Tier-1 techniques (T1.1, T1.2, T1.3) are **only safe to test after
auditing our existing post-proc stack**. If we want to re-test them,
the right sequence is:
1. Disable `delta_shift_alpha` (set to 0 in CFG).
2. Disable `rank_aware_scale`.
3. Re-probe LB with the stripped-down baseline (may also regress,
   which would prove our custom smoothing is already doing work).
4. Add Tier-1 techniques one at a time on the clean baseline.

This costs 4+ LB slots to complete rigorously. Not worth it right
now — moving on to T2.6 (new data) which doesn't interact with
post-proc at all.

### 14.11.6.4 T2.6 scope: BirdCLEF 2025 data pull (START NOW)

**We already have BirdCLEF 2025 data locally** (extracted at
`data/raw/birdclef_2025/`, ~15 GB). Not a Kaggle download — just an
ingest script.

**Overlap audit (verified 2026-04-20):**
- 2026 species: 234
- 2025 species: 206
- **Overlap: 41 species** (18 % of 2026 classes get new data)
- **Additional training rows: 13,484** (vs current 35,549 rows → +38 %
  corpus size for those 41 species)
- Class distribution of overlap: Aves 38 / Amphibia 2 / Mammalia 1
- 2025 collections: XC (21,204), iNat (7,198), CSA (162) — all
  globally sourced Xeno-Canto / iNaturalist / CSA, **no Pantanal-
  site audio**. Structurally safe (no L5b-style trap).
- Same schema as 2026 `train.csv` (columns: primary_label, filename,
  collection, latitude, longitude, etc.). Clean merge.

**Why this is the right next bet:**
- Aves is the class where our recent experiments have *regressed*
  (L5b pushed Aves Δ from −0.035 to −0.081). Adding labeled Aves
  audio directly addresses the regression vector.
- Globally-sourced focal recordings — memory `project_l5b_amphibia_killed`
  says this is structurally different from AnuraSet's INCT-site
  field mics; no Pantanal-ambience shortcut.
- No post-proc interaction — the pipeline change is upstream of
  every technique audit we've done.

**Implementation plan:**
1. Write `src/merge_birdclef2025_into_folds.py`:
   - Load both train.csv files.
   - Filter 2025 to `primary_label in 2026_species_set`.
   - Dedupe by URL (some 2025 rows may be same Xeno-Canto IDs as
     2026 iNat rows — rare but possible).
   - Assign `collection = "BC2025"` (new tag distinct from iNat/XC/
     AnuraSet — disambiguates which root path to load from).
   - Assign fold — option A: use GroupKFold by author as in 2026;
     option B: assign fold=-1 (same as AnuraSet, pool/non-train
     convention). **Choose A** — these are labeled with same-quality
     focal recordings; they deserve to train fold 0–4, not just be
     a supplement.
   - Backup current `train_folds.csv` to
     `train_folds_pre_bc2025.csv`.
   - Write merged `train_folds.csv`.
2. Update `src/dataset_a1.py` path router:
   - Add branch for `collection == "BC2025"` →
     `data/raw/birdclef_2025/train_audio/<filename>`.
3. Smoke test: `--fold 0 --epochs 1 --smoke-test`.
4. Fold 0 pilot: `--fold 0 --epochs 25 --loss asl --mixstyle-p 0.5`
   (bit-identical recipe to baseline A1, just more data).
5. Gate: val_roc_auc ≥ 0.7414 (baseline). If above, JIT export and
   push v55 fold-0-swap probe to LB.
6. If v55 LB ≥ 0.934, train folds 1/2/4 with expanded data, push
   full 4-fold update.

**Budget:** ~1 h ingest + ~2 h fold-0 pilot + ~10 min LB probe =
½ day for fold-0 evidence; ~6 h more for full rollout if gate passes.

### 14.11.6.5 Data-pull follow-up candidates (not this session)

If T2.6 passes, next data-source lever is expanding beyond 2025:
- **BirdCLEF 2024 training data**: 182 species, not locally cached
  (need Kaggle dataset pull). Cap+dedupe same as 2025.
- **BirdCLEF 2023 / 2022 / 2021**: similarly, cumulative Xeno-Canto
  coverage. Overlap with 2026 is probably smaller (newer species)
  but non-zero.
- **Xeno-Canto direct API** for 2026-only species (193 of 234):
  these are unique to 2026 and have no prior-year data. Would need
  new ingest path; bigger project.

D3 (Perch-as-teacher) remains queued as the infrastructure unlock
for pseudo-label distillation — independent from T2.6 and can run
in parallel once T2.6 is validated.

---

## 14.11.6.5 T2.6 BC2025 merged; BC2024 deferred; BC2021 queued (2026-04-19)

**Merge executed 2026-04-19 21:22 UTC.**

- Script: `src/merge_birdclef2025_into_folds.py` (written and run).
- Dataset routing: `src/dataset_a1.py` gained a
  `coll == "BC2025"` branch pointing to
  `data/raw/birdclef_2025/train_audio/`.
- Result:
  - 13,484 BC2025 rows in the 41 species that overlap with 2026 taxonomy.
  - 10,192 dropped as duplicates of existing XC/iNat rows (76% overlap —
    BC2025's train set is heavily XC/iNat sourced, same as 2026).
  - **3,292 net new rows** across 40 species (species `67252` had only
    10 overlapping rows; one 2025 species yielded 0 after dedupe).
  - New fold counts: {−1: 2305 AnuraSet, 0: 7745, 1: 7607, 2: 7534,
    3: 7893, 4: 8062} — balanced within ±4% of mean.
- Per-species uplift is **more modest than scoped**: median +6%,
  only 3 species >100% lift (`yehcar1` +237%, `67252` +167%,
  `grekis` +146%). Most species get single-digit %.
- Backup: `data/processed/train_folds_pre_bc2025.csv` (7.5 MB).

**BC2024 considered and deferred (geography mismatch).**

- BirdCLEF 2024 is Western Ghats, India → Indian subcontinent species.
  Pantanal (2026) is Neotropical → **near-zero species overlap**
  expected. Download + dedupe cost not justified for speculative
  marginal rows.
- BirdCLEF 2021 was Colombia (Neotropical) → **structurally the right
  previous-year lever** for Pantanal. Queued as the post-BC2025 probe
  if T2.6 passes gate.
- BirdCLEF 2022 (Hawaii) and 2023 (East Africa) → zero-overlap like
  2024. Do not pursue.

**Option chosen (user 2026-04-19): Option 2 — BC2025 pilot first,
BC2021 probe after.**

- Rationale: prove or disprove the "small overlap → modest LB lift"
  hypothesis on BC2025 (already merged, no download cost) before
  committing to ~30 GB BC2021 audio ingest.
- If BC2025 gate passes fold-0 val ≥ 0.7414, JIT + push fold-0 swap
  as v55. If v55 LB ≥ 0.934, train folds 1/2/4 with expanded data +
  full 4-fold rollout, **then** start BC2021 ingest in parallel with
  the rollout.
- If BC2025 gate fails (val < 0.7414) or v55 LB flat/regresses,
  BC2021 is likely to behave the same way (same sourcing pipeline,
  just more species overlap) — would revisit the "prior-year BirdCLEF
  data" lever as a whole before downloading.

**Smoke test:** clean (PID 452429; 33,562 train clips for fold-0,
val=NaN expected with 2 batches, no crashes).

**Fold-0 pilot launched 2026-04-19 21:27 UTC** (PID 452429):
`--fold 0 --epochs 25 --loss asl --mixstyle-p 0.5`. ETA ~4–6 h.

---

## 14.11.6.6 T2.6 BC2025 KILLED (2026-04-20) — val +0.026 / LB −0.004

**Pilot completed 2026-04-19 23:15 UTC.** Fold-0 best val_roc_auc
**0.7670** at E22 (production baseline 0.7414, comparable hybrid
retrain 0.7220). Gate **PASSED** at +0.026.

**v55 LB probe pushed 2026-04-19 23:19 UTC.**
- Kaggle dataset `stevewatson999/birdclef-2026-a1-effb0-ckpts` v55:
  fold-0 slot replaced with BC2025-retrained JIT ckpt; folds 1/2/4
  unchanged.
- Notebook `stevewatson999/birdclef-2026-protossm` kernel version 55
  ran clean.
- **LB public score: 0.927** (2026-04-20 03:28 UTC).
- **Δ vs 0.931 baseline: −0.004 regression** → kill gate triggered
  (pre-committed: <0.929 → revert).

**Val/LB divergence pattern:** same as L1 noisy student (val +0.163
→ LB −0.001). Val 0.7670 was a single-epoch spike at E22 (+0.051
from E21 0.7163), which dropped back to 0.7364 / 0.7368 / 0.7487
for E23-25. Strong hypothesis: the BEST checkpoint is an
over-annealed snapshot that latched onto Pantanal-val-specific
features during final cosine phase. LB (private test) did not
receive the benefit.

**Why BC2025 couldn't move LB (structural read):**
- 40 boosted species, but only **9 appear in val** (train_soundscapes
  has 75 species). 22 of 40 have **0 val windows**.
- High-uplift species from merge stats are mostly val-absent
  (`yehcar1` +237%, 0 val; `trokin` +82%, 0 val; `roahaw` +74%,
  0 val; `banana` +41%, 0 val).
- Only ~634 of 3,292 added rows land on species with ≥50 val windows
  (`whtdov` +152 / 126 vw, `compau` +412 / 76 vw, `trsowl` +74 /
   52 vw). Even these likely introduced label-noise drift rather
  than clean signal.
- Domain mismatch stacks on top: BC2025 clips are globally-sourced
  focal XC/iNat; val+test are Pantanal field soundscapes. More focal
  data doesn't tighten soundscape features.

**Revert executed 2026-04-20 03:50 UTC.**
- `kaggle_datasets/a1-effb0-ckpts/a1_fold0.pt` restored from
  `_backups/a1_fold0_lbbase_20260419.pt` (hash-verified match).
- Kaggle dataset v56 pushed as revert ("BC2025 fold-0 swap killed
  on LB 0.927 …").
- `train_folds.csv` still has 3,292 BC2025 rows merged with
  `collection="BC2025"`. Leave in place (harmless with production
  ckpt); removal would require re-running the merge script in
  reverse. Pre-BC2025 snapshot is at
  `data/processed/train_folds_pre_bc2025.csv` if ever needed.

### 14.11.6.6.1 Generalized lesson — val/LB divergence on Pantanal

For any candidate that trains a fresh fold-0 on added data, val
improvement of +0.01–0.05 on `train_soundscapes` should be treated
as **weak LB-predictive** when:

1. The added data is mostly (>70%) for species not in val, **and**
2. The val BEST epoch is a late-training spike (>2× the typical
   epoch-to-epoch Δ of the final-quarter trajectory).

Both conditions held for BC2025. Same both conditions empirically
killed L1 noisy student (§14.9). Treat this as a **required
diagnostic before spending a Kaggle submission slot**: check val-set
species coverage of added data AND whether BEST is a spike before
committing to LB probe.

### 14.11.6.6.2 Prior-year BirdCLEF data lever — CLOSED

- BC2025: tried, killed (this section).
- BC2024: deferred (geography zero-overlap).
- BC2022 / 2023: zero-overlap (Hawaii / East Africa).
- BC2021 (Colombia): was queued as the "Neotropical-adjacent"
  follow-up. **Deprioritize** — same sourcing pipeline
  (XC/iNat focal), same val-coverage structure, same Pantanal
  domain mismatch. Expected outcome: similar to BC2025 — modest
  val lift, flat/negative LB. Not worth 30 GB download + 2h train
  when the pattern is now empirically established.
- Overall lever status: **closed**. Next data-adjacent move would
  need a different *source type* (not prior-year Kaggle training
  sets), e.g. D3 (pseudo-labels on Kaggle **test** set) or field
  recordings sourced from Pantanal-specific archives (not XC).

---

## 14.11.7 D3 BC2025 soundscape pseudo-label — IN PROGRESS (2026-04-20)

**Rationale (why this breaks the "more data failed 4x" pattern):**

Every prior data-expansion attempt added **focal** recordings
(XC/iNat via L1 / L5a / T2.6) or non-Pantanal **focal** amphibian
(L5b). Val + test are **soundscape**. Adding focal data to a model
already saturated on focal clips has consistently moved val without
moving LB — the decision boundaries never traverse the
focal-to-soundscape domain gap.

`data/raw/birdclef_2025/train_soundscapes/` contains **9,726
unlabeled 60s Colombia soundscapes (4.4 GB, 32 kHz mono)** that sit
structurally in the same acoustic domain as our val+test: forest
soundscape, Neotropical, multi-source ambient, narrow-band bird
calls over diffuse insect/frog chorus. This is the first
**domain-matched** data expansion we've attempted.

### 14.11.7.1 Pipeline

Phase 1 — pseudo-label generation (`src/pseudo_label_bc25ss.py`,
to be written):
- Load 4-fold A1 production JIT ensemble from
  `kaggle_datasets/a1-effb0-ckpts/a1_fold{0,1,2,4}.pt`.
- For each of 9,726 files: extract 12 non-overlapping 5s windows.
- Per window, **tile-pad the 5s waveform up to the 20s training
  chunk length** (the notebook's convention in cells §A1 inference).
- Run each fold's JIT `forward()`, sigmoid, average across 4 folds.
- Save `data/processed/pseudo_bc25ss_probs.npz` with
  `probs (9726, 12, 234)`, `filenames (9726,)`.
- Compute scale: 116,712 windows × ~2 ms = ~15 min GPU.

Phase 2 — threshold + emit train rows:
- For each window where `max_prob > τ` (start τ=0.5):
  - Emit row: filename, `pseudo_window_start` (sec), primary_label,
    collection="BC2025_SS_PSEUDO", inat_taxon_id from 2026 taxonomy.
- Fold assignment: GroupKFold by source filename (prevent leakage
  across folds when same 60s file spawns multiple windows).
- **Required diagnostic before commit** (§14.11.6.6.1 rule): tabulate
  species distribution of retained pseudo-labels vs `train_soundscapes`
  val-species presence. If >70% of retained rows are val-absent
  species, lower τ or filter to val-present species before
  proceeding — the val/LB divergence will repeat otherwise.

Phase 3 — dataset branch:
- Extend `src/dataset_a1.py` with
  `coll == "BC2025_SS_PSEUDO"` branch: load 60s source, extract
  5s at `pseudo_window_start`, tile to CHUNK_SAMPLES (20s) matching
  notebook inference behavior.

Phase 4 — retrain fold-0:
- `--fold 0 --epochs 25 --loss asl --mixstyle-p 0.5`
- Gate: fold-0 val_roc_auc ≥ 0.7414 AND BEST-epoch not a spike
  (>2× typical Δ of final-quarter trajectory per §14.11.6.6.1).

Phase 5 — JIT + v57 LB probe:
- Back up fold-0, JIT export, push Kaggle dataset, read LB.
- Gate: LB ≥ 0.934 → full 4-fold rollout; LB 0.929–0.933 →
  investigate pseudo-label noise; LB <0.929 → revert, re-examine
  threshold τ and val-coverage filter.

### 14.11.7.2 Cost budget

- Phase 1 inference: ~15 min GPU.
- Phase 2 threshold + merge: ~5 min.
- Phase 3 code edit + smoke: ~20 min.
- Phase 4 fold-0 train: ~2 h.
- Phase 5 JIT + push + LB: ~1 h.
- **Total: ~4 h elapsed, ~2.25 h GPU.**

### 14.11.7.3 Kill criteria

- If Phase 2 retention < 20% of windows at τ=0.5: pseudo-labels are
  too noisy to be useful (model isn't confident enough on
  Colombia-vs-Pantanal acoustic drift). Drop τ once (to 0.4); if
  still <20%, kill.
- If Phase 4 val < 0.7414: expanded data hurts. Do NOT push to LB.
  Consider tightening τ or the val-species filter from Phase 2
  diagnostic, retry once, kill if still regressing.
- If Phase 5 LB < 0.929: "more data" is definitively not the
  bottleneck. Switch lever to T1.3 (min-reduce ensemble) or T3.1
  (model soup).

---

## ⏸️ PICK UP HERE — previous (2026-04-20 01:12 local — D3 overnight launched — SUPERSEDED)

**State at launch:**
- LB 0.931 baseline restored on Kaggle (a1-effb0-ckpts v56 is the
  revert; T2.6 killed, §14.11.6.6).
- D3 overnight orchestrator running: `scripts/d3_overnight.sh`
  (orchestrator PID 463833 at launch; check `ps -ef | grep d3_overnight`).
- Scripts in play:
  - `src/pseudo_label_bc25ss.py` (Phase 1) — JIT-fusion guarded
    for GB10 cap 12.1 > PyTorch 2.10's 12.0 ceiling.
  - `src/d3_phase2_emit_pseudo_rows.py` (Phase 2) — τ auto-adapt
    to 3–25% retention band; hard gate kills overnight if
    >70% emitted rows target val-absent species (§14.11.6.6.1).
  - `src/dataset_a1.py` (Phase 3) — `BC2025_SS_PSEUDO` branch added,
    loads 60s soundscape and slices 5s at `pseudo_window_start`.
  - `src/train_a1.py --fold 0 --loss asl` (Phase 4) — gate: BEST
    val_roc_auc ≥ 0.7414; no Kaggle push if gate fails.
  - `src/export_a1_jit.py --loss-suffix asl --folds 0` (Phase 5) —
    backs up current `a1_fold0.pt` to `_backups/` before overwrite,
    then `kaggle datasets version` as v57.
- Expected wall time: ~3h15m total (P1 60m + P4 2h + overhead).

**Morning check-in:**
1. `tail -50 log/d3_overnight_*.log` — look for final verdict line
   (`D3 overnight complete` or `GATE FAIL …`).
2. If Phase 4 gate passed and v57 pushed: do manual kernel push in
   `jupyter/protossm-postproc/`, then LB probe.
   Gate: LB ≥ 0.934 → 4-fold rollout; 0.929–0.933 ambiguous;
   <0.929 → revert via dataset v58 (copy `_backups/a1_fold0_v56_preD3_*.pt`
   back to `kaggle_datasets/a1-effb0-ckpts/a1_fold0.pt`).
3. If gate failed: the new fold-0 ckpt is at
   `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_asl.pt`;
   inspect the Phase-4 log's per-epoch trajectory for the
   spike-vs-steady pattern before deciding next lever.
4. If Phase 2 gate failed: `train_folds.csv` untouched
   (snapshot at `data/processed/train_folds_pre_d3.csv` only exists
   on the succeeding arm). Read `data/d3_phase2_diagnostic.txt` for
   the val-coverage breakdown; next lever probably T1.3 or T3.1.

**Fallback levers if D3 fails LB:**
- T1.3 min-reduce ensemble (notebook-only, cheap).
- T3.1 last-N model soup (2h retrain window, cheap LB probe).

---

## 14.11.8 D3 KILLED — Phase 4 gate catastrophic fail (2026-04-20 evening)

**Result:** fold-0 `train_a1.py --loss asl` on D3-merged `train_folds.csv`
(67,624 rows, +26,478 BC2025-SS pseudo) finished 3h 33m.
BEST `val_roc_auc = 0.6644` at epoch 13; baseline gate was **0.7414**
→ Δ = **−0.077**, far below the kill threshold.
No Kaggle push (gate blocked Phase 5 as designed).

Per-epoch trajectory sat flat in 0.60–0.66 band across all 25 epochs —
steady underperformance, not a spike pattern. Consistent with the D3
pseudo-label distribution structurally harming the val domain, not a
noisy-epoch artifact.

### 14.11.8.1 Revert executed (2026-04-20 evening)

- `data/processed/train_folds.csv` restored from `train_folds_pre_d3.csv`
  (41,147 rows). Failed merged CSV preserved at
  `data/processed/train_folds_post_d3_failed.csv` for diagnostics only.
- Kaggle `a1_fold0.pt` md5 = `fc0f32ad…` matches
  `_backups/a1_fold0_lbbase_20260419.pt` → production **v56 (LB 0.931)
  is untouched**. No v57 was ever pushed.
- Gate-failed local ckpt remains at
  `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_asl.pt`
  (do not export; keep for post-mortem).

### 14.11.8.2 ⚠️ train_folds.csv state mismatch vs production

The restored `train_folds.csv` (41,147 rows = pre_d3) still contains
**BC2025 (T2.6, killed §14.11.6.6)** and **AnuraSet (L5b, killed
§14.10.17)** merges. Both levers were LB-negative; Kaggle was reverted
ckpt-side but local CSV was never rolled back.

Production v56 ckpt `_backups/a1_fold0_lbbase_20260419.pt` is dated
**2026-04-16 17:20**, predating both merges. The matching local
baseline state is `data/processed/train_folds_pre_anuraset.csv`
(35,550 rows).

**Implication:** any fallback lever that retrains fold-0 (e.g. T3.1
model soup) must first decide which base CSV to use. Options:

1. **Revert to pre_anuraset (35,550 rows)** to match the production
   training distribution — safest, reproduces known LB-0.931 baseline.
2. **Keep pre_d3 (41,147 rows)** to test whether the combined BC2025 +
   AnuraSet data helps a *different* lever (noisy student, longer
   schedule, etc.) — but has the combined −val-LB divergence risk from
   two killed sources.

**Decision deferred** until next lever is picked. When resuming, first
`cp train_folds_pre_anuraset.csv train_folds.csv` unless there's a
documented reason not to.

### 14.11.8.3 Next action — pick between T1.3 and T3.1

- **T1.3 min-reduce ensemble** — notebook-only, no retrain. Change
  post-proc merge from max/mean to min across fold predictions.
  Cost: one notebook edit + one LB probe. Start here.
- **T3.1 last-N model soup** — average weights of last N epochs from
  existing 5-fold ckpts. Cost: 2h retrain window if ckpts not cached,
  otherwise cheap. Do after T1.3 verdict.

---

## 14.11.9 T1.3 min-reduce fold ensemble KILLED (2026-04-21 00:40 UTC)

**Result:** Kaggle kernel v56 pushed with `A1_FOLD_REDUCE = "min"`
(min-reduce raw sigmoids across A1 folds, then rank once).
Submission scored **LB 0.925** vs baseline **0.931** → Δ = **−0.006**.
Below the `<0.931 → revert` kill criterion in §14.11.3.a.

**Revert executed:**
- Cell 41 toggle flipped back to `A1_FOLD_REDUCE = "mean"` (kill marker
  in inline comment). Min-reduce code path retained behind the toggle
  for future reference (no dead code purge).
- **No re-push** — Kaggle kernel v56 (min-reduce) stays as the dead
  probe artifact; final LB scoring uses manually-selected submissions,
  so the stale kernel doesn't threaten standing. Next LB probe will
  re-push and overwrite.
- Local backup `birdclef2026-postproc.ipynb.bak_pre_t13` preserved.

**Lesson:** min across folds pushes all predictions to the weakest
fold's confidence floor, compressing dynamic range on rare species
(where only 1–2 folds see the class). Consistent with the −0.006
magnitude — smaller than v54 Tier-1's −0.012 (§14.11.6), because the
damage is bounded by the per-class rank-fusion step.

Tier-1 post-proc shortlist is now fully exhausted
(T1.1+T1.2 killed v54, T1.3 killed here). **Next lever: T3.1 last-N
model soup** (Tier-3 of §14.11.3).

---

## 14.11.10 T2.1 recipe — softmax CE loss (2026-04-20 evening, drafted)

**Chosen next lever:** T2.1 — softmax cross-entropy instead of ASL
on fold 0. Rationale in recommendation ranking:

| # | Expected LB | Cost | Strength of evidence |
|---|-------------|------|----------------------|
| **T2.1** CE loss | **+0.005–0.020** | 1 fold retrain (~3.5h) | 1st place's strongest stated finding ("BCE significantly worse" → they used softmax CE, not BCE) |
| T2.2 GEM pool | ±0.005 | head swap + retrain | weakest; drop-in only |
| T2.3 ckpt soup | +0.003–0.010 | 40-epoch retrain | 2nd place recipe, stacks on T2.1 |

Aligns with §14.11.3.a step 2's pre-written escalation: "If Step 1 LB
< 0.934, escalate to T2.1 alone to interrogate the loss function
hypothesis."

### 14.11.10.1 Prereqs (executed 2026-04-20 evening)

- **`train_folds.csv` reverted** from `train_folds_pre_anuraset.csv`
  (35,550 rows). Matches production v56 training state; removes
  AnuraSet (L5b, killed) + BC2025 (T2.6, killed) contamination.
  Pre-T2.1 state preserved as `train_folds_post_d3_failed.csv` and
  `train_folds_pre_d3.csv`.
- **`src/train_a1.py --loss ce` branch added:**
  - Argparse choice "ce" registered.
  - Loss setup: `nn.CrossEntropyLoss(reduction="none")`.
  - Training loop branch (guarded by `loss_name == "ce"`):
    normalizes `labels` (multi-hot) to a probability distribution
    via `labels / labels.sum(dim=-1).clamp_min(1e-6)`; passes soft
    targets to PyTorch's built-in soft-target CE; per-sample loss
    is `.mean()`'d (no `sec_mask` — softmax CE has no per-class
    exclusion semantics).
  - Checkpoint path picks up the `_ce` suffix automatically from
    the existing `f"…_{loss_name}.pt"` template.

### 14.11.10.2 Training config (isolate the signal — no other changes)

- Epochs 25 (same), AdamW (same LR/WD), CosineAnnealingWarmRestarts
  (same T_0), mixstyle_p=0.5 (same), all augmentations (same).
- Fold 0 only for this probe.
- Only variable: `--loss ce` (vs ASL baseline).

Intentionally NOT doing in this step:
- 1st-place's 10-sec chunks (that's T3.1, separate multi-day rewrite).
- 1st-place's shorter schedule / higher LR (7–12 epochs, lr 3e-3) —
  leave schedule matched to baseline to isolate loss signal.

### 14.11.10.3 Gates

1. **Smoke-test gate:** `--smoke-test` 1-batch run must complete
   without NaN loss or shape errors. (Running now, 2026-04-20.)
2. **Fold-0 val gate:** BEST `val_roc_auc ≥ 0.7414` (baseline fold-0
   val). Sub-0.7414 → kill T2.1 without Kaggle push.
3. **LB gate (if fold-0 val passes):**
   - LB ≥ 0.934: lock in as new baseline, proceed to 5-fold rollout.
   - LB 0.929–0.933: ambiguous; inspect per-class Δ before deciding.
   - LB < 0.929: revert, add to killed-lever log.

### 14.11.10.4 Rollback

- `train_folds.csv`: `cp train_folds_pre_d3.csv train_folds.csv`
  (restores the pre_d3 state if we need to pivot to a data-level
  lever later).
- `train_a1.py` CE path: keep (small, under-a-flag, no side effects
  when `--loss asl`).
- Kaggle ckpt: v56 baseline stays on Kaggle; T2.1 ckpt only pushed
  on val + LB gate pass.

### 14.11.10.5 Why CE might lift (1st-place hypothesis)

Focal loss variants (ASL is one) downweight well-classified examples
to emphasize hard negatives. On multi-label bioacoustic data with many
near-background classes, this can over-attend to class boundaries in
a way that doesn't generalize to a shifted acoustic domain (Pantanal).
Softmax CE forces logits to *compete* across classes, which:

1. Injects a mild per-sample class-mass constraint (logits sum via
   softmax normalization) that tends to calibrate relative rankings
   better — matches the rank-fusion downstream step.
2. Penalizes confident wrong predictions more harshly than focal
   variants, which reduces the "confidently wrong" tail seen on the
   Pantanal domain shift.

First-place's quote "BCE significantly worse" is the clearest single
win in the whole 2024-top-3 audit; worth a direct test.

---

## 14.11.10.6 RESULT — T2.1 CE KILLED mid-run 2026-04-20 23:03 local

**Killed at epoch 11/25 (PID 495821).** Val AUC trajectory:

| Epoch | train_loss | val_roc_auc | Note |
|-------|------------|-------------|------|
| 1     | 4.3004     | 0.5530      | ★ BEST |
| 2     | 3.3747     | 0.5981      | ★ BEST |
| 3     | 3.0240     | 0.5810      | |
| 4     | 2.8250     | 0.6125      | ★ BEST |
| **5** | **2.7180** | **0.6145**  | **★ BEST (overall peak)** |
| 6     | 2.8315     | 0.5981      | |
| 7     | 2.7100     | 0.6134      | |
| 8     | 2.5950     | 0.6038      | |
| 9     | 2.4742     | 0.6035      | |
| 10    | 2.3985     | 0.6010      | |
| 11    | 2.5470     | 0.5836      | ← declining while train_loss drops |

**Why kill:** val AUC peaked at **0.6145** (epoch 5) vs gate **0.7414**
= Δ −0.127, catastrophic miss. More critically, the trajectory is
**regressing** from epoch 5 while train loss continues to fall — classic
overfitting signature, no realistic path to +0.13 AUC recovery in 14
remaining epochs. Saving ~60 minutes of GPU time.

**Interpretation:** softmax CE on our setup converges to a qualitatively
different (and worse) representation than ASL. Hypotheses:

1. **Single-winner pressure from softmax.** Normalizing multi-hot
   targets to a distribution (our soft-target approach) makes primary
   and secondary labels compete for mass within the sample. On clips
   with 3+ active species (common in Xeno-Canto focal clips), no single
   class gets strong gradient signal. ASL / BCE handle this natively
   by per-class independent objectives.
2. **Mixup pathology.** Our mixup merges labels via element-wise max,
   producing 4–8 "primaries" post-mix. Under softmax CE this becomes
   a very flat distribution (1/N each) that provides near-zero
   learning signal. ASL is robust to this because each class is
   binary-independent.
3. **1st-place recipe is not decomposable.** They ran CE *with* 10-sec
   chunks + averaged-label targets + Xeno-Canto-sourced data + their
   own mel config. CE alone without those accompanying changes may
   be actively harmful.

Hypothesis (2) is the most likely — a clean test of CE would require
disabling mixup or switching to one-hot primary-only targets. That's
a bigger experiment than "swap the loss alone."

### 14.11.10.7 Revert executed

- Process killed at 23:03 local (epoch 11 complete, epoch 12 in-flight).
- Failed ckpt preserved for post-mortem:
  `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_ce.pt`
  (smoke-test overwrite — 1-epoch weights; per-epoch ckpts of the
  full run exist in-memory only, lost on kill).
- `train_folds.csv` stays at 35,550 rows (pre_anuraset) — this is the
  correct production-matching state, no revert needed.
- Kaggle `a1-effb0-ckpts` v56 untouched (no export happened).
- `train_a1.py --loss ce` code path **retained** — small, behind a
  flag, no side effects when `--loss asl`. Keep for future variants
  (e.g., CE + one-hot primary-only, CE without mixup).

### 14.11.10.8 Next lever — demoted to T2.3 (checkpoint soup)

Tier-2 shortlist update:
- T2.1 CE — **KILLED** (this section).
- T2.2 GEM pooling — still untried; low signal (±0.005).
- T2.3 checkpoint soup — **now the highest-expected-lift untried
  Tier-2 lever.** +0.003–0.010, low risk, ~4h retrain (40 epochs)
  + averaging + export + LB probe.
- T2.4 mel-diversity ensemble — needs new fold training, stacks
  downstream of T2.3.
- T2.5 Perch quality filter — needs pre-filter pass over 35,550 rows;
  1-day pipeline, +0.005–0.015. Worth considering before T2.3.

Recommend: **T2.3 next.** Cheapest remaining Tier-2, lowest risk, and
it stacks cleanly with any future retrain lever (T2.5, T2.2). If
T2.3 lands, LB ≥ 0.934 locks in and rolls to 5 folds; if not, escalate
to T2.5.

---

## 14.11.10.9 T2.3 40-epoch attempt KILLED — pivot to 25-ep soup (2026-04-21 00:13)

**40-epoch run killed at epoch 15/40** (PID 498503). BEST val_roc_auc
stuck at **0.6542** for 4 epochs (ep 11 → ep 15) while train loss kept
falling 0.0054 → 0.0039. Same overfitting signature as T2.1 CE — the
25-ep-tuned `T_0` cosine warm-restart period doesn't produce the
expected second-restart lift on a 40-ep horizon.

**Hypothesis:** `config.T_0` is tuned to the 25-epoch schedule. Stretching
to 40 epochs without adjusting T_0 (we didn't touch it) leaves the
cosine restart in a bad phase — the LR ramp-downs don't align with
where the model needs them. "Just train longer" ≠ "train on a longer
schedule."

**Pivot (user call):** retrain with native 25-epoch schedule +
`--save-all-epochs`, then soup epochs 8–25 (scaled from 2nd-place's
13–50 band). This reuses the known-working baseline LR schedule and
only adds the soup step on top.

- Old 40-ep _soup dir deleted.
- `train_folds.csv` still at 35,550 rows (pre_anuraset), correct.
- New run launched 2026-04-21 00:13: `train_a1.py --fold 0 --loss asl
  --epochs 25 --save-all-epochs` (PID 501992). ETA ~1.7h.
- Log: `train_a1_fold0_asl25soup_20260421_001344.log`.

**Post-training plan:**
1. Wait for 25-ep run to complete (ETA ~01:55 local).
2. Build soup ckpt by averaging state_dicts of epochs 8–25
   (18 ckpts) via new `src/soup_a1.py` (write during training).
3. Re-validate the soup ckpt → compare to single-BEST val.
4. Export the winner (soup if val ≥ single-BEST; otherwise abandon
   T2.3) → Kaggle dataset v57 → LB probe.
5. LB gate: ≥ 0.934 lock; 0.929–0.933 ambiguous; < 0.929 revert.

---

## 14.11.10.10 T2.3 25-ep run completed on WRONG LOSS — `--loss asl` not `--loss hybrid` (2026-04-21 morning)

**Run completed at 01:49 local** (1h 35m total). BEST val_roc_auc = **0.6442**
at epoch 24 — far below the 0.7414 hybrid baseline gate.

### Diagnosis

The plan in §14.11.10.9 specified `--loss asl`. The LB-0.931 production
baseline was trained with **`--loss hybrid`** (verified by
`scripts/train_a1_5fold.sh` and the 2026-04-19 baseline log
`log/archive/train_a1_fold0_no_anuraset_20260419_155526.log` which shows
`loss : hybrid` and BEST val 0.7220).

Comparison to the 2026-04-19 hybrid baseline on the *same*
`train_folds_pre_anuraset.csv`:

| Metric            | Hybrid baseline (04-19) | ASL attempt (04-21) |
|-------------------|-------------------------|---------------------|
| Epoch 1 train_loss| 0.0460                  | 0.0107 (~4× lower)  |
| Epoch 5 val_auc   | 0.6396                  | 0.6081              |
| BEST val_auc      | **0.7220**              | **0.6442**          |

ASL-only loss values are ~4× smaller from epoch 1; the model reaches low
train loss quickly but stalls on the val domain. The T2.1 CE probe
(§14.11.10.6 — killed "catastrophic −0.127 vs gate") was likewise
running on ASL-sized loss magnitudes against a hybrid-trained gate. The
CE comparison was still directionally valid vs ASL (−0.03 vs our
measured ASL 0.6442 peak), but the framing "gate miss of −0.127" was
apples-to-oranges.

Saved memory: `project_a1_baseline_loss_is_hybrid.md`.

### Artifacts from the mis-loss run

- `models/a1/_soup/fold0_asl_seed42/epoch{01..25}.pt` — 25 × 16.5 MB
  ckpts, ASL loss. **Not usable for baseline-comparable soup** (wrong
  loss regime). Keep as ASL-only reference for now; delete if disk
  pressure emerges.
- `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_asl.pt` —
  the BEST-of-run single ckpt (0.6442). **Do not export.** The `_asl`
  suffix means it doesn't overwrite the production `_hybrid` ckpt.

### Revert state

- Kaggle `a1-effb0-ckpts` v56 untouched (no export happened).
- `train_folds.csv` stays at 35,549 rows (pre_anuraset) — correct.
- `src/train_a1.py` kept (the `--save-all-epochs` flag is useful for
  the re-run below; `--loss ce` branch stays dormant).
- `src/soup_a1.py` untouched — still valid for a hybrid re-run.

---

## 14.11.10.11 T2.3 hybrid soup KILLED (2026-04-21 23:47 local)

Corrected hybrid retrain ran 1h 35m (PID 531799, log
`log/archive/train_a1_fold0_hybrid25soup_20260421_220859.log`) and
produced 25 per-epoch ckpts at `models/a1/_soup/fold0_hybrid_seed42/`.
Training trajectory matched the 2026-04-19 baseline closely:

- Training BEST val_roc_auc = **0.7203** (epoch 15) vs baseline 0.7220
  (ep 25). Essentially tied within run-to-run noise.
- Mean val across ep 8-25 soup window = **0.7021** vs baseline 0.6991
  (+0.003). Per-epoch oscillation from cosine warm restarts was pronounced
  (ep 13 = 0.6791 → ep 15 = 0.7203, adjacent Δ = 0.041).

`src/soup_a1.py --fold 0 --loss hybrid --start-epoch 8 --end-epoch 25`
averaged all 18 ckpts in the window, took BN buffers from ep 25, and
re-validated:

- **SOUP val_roc_auc = 0.6921**
- max single-epoch val = 0.7203
- **Δ (soup − best) = −0.0282** (9× the 0.003 kill threshold in §14.11.10.9)

Soup ckpt `models/a1/a1_..._hybrid_soup.pt` saved for post-mortem but
**not exported**. Production Kaggle v56 untouched.

### Diagnosis

The cosine warm restarts at epochs 5/10/15/20/25 push the optimizer
into distinct basins — the ±0.04 val AUC shifts between adjacent
epochs confirm the weight-space displacements are large and
discontinuous, not the small co-adapted shifts that soup averaging
assumes. Averaging across basins produces a model that's in none of
them.

BN-stats-from-last-ckpt may also be mis-calibrated for the averaged
params, but the magnitude of the regression (−0.028) is too large to
explain with BN drift alone.

### Lessons for future soup attempts

Soup requires schedule compatibility. If revisited (low priority),
first switch to **vanilla cosine** (single decay, no restarts) or
**SWA-style flat LR** for the averaging window. Do NOT average ckpts
across warm-restart boundaries again.

Memory: `project_t23_soup_killed.md`.

### Single-BEST vs production

Current single-BEST ckpt (val 0.7203) vs v56 production (val 0.7220) =
−0.0017 — tied within noise. Not worth a Kaggle LB slot for A/B.
T2.3 lever is fully closed.

---

## 14.12 Post-T2.3 sequence: C → D → A(T2.2) → B (second backbone)

After 11 consecutive kills (T1.1/1.2/1.3, T2.1, T2.3, T2.6, D3, L1, L2,
L5b, original Track-C student), the pattern suggests we're near a local
optimum for this backbone/data/val setup. Before spending more GPU on
incremental Tier-2 probes against a flaky gate, rebuild the foundation:

1. **C — val rebuild first.** Current val is 1,478 segments /
   75-of-234 species. Multiple prior kills show it doesn't predict LB
   well (§10 teacher leakage, L1 NS leakage, T2.6 "+val / −LB"
   divergence). Cleaner val doesn't add LB on its own but makes every
   subsequent selection decision trustworthy. Low GPU cost (CPU
   preprocessing).
2. **D — multi-seed ensemble second.** Retrain fold-0 with 3 seeds on
   current hybrid recipe, average sigmoids. Reliable +0.003–0.008 LB
   from variance reduction, near-zero regression risk. ~4.5h GPU.
   Stacks with everything downstream.
3. **A — T2.2 GEM third.** Cheapest remaining Tier-2 head change
   (±0.005, 1.5h retrain). Only worth running once C gives a
   trustworthy signal.
4. **B — second backbone fourth.** +0.020 gap to leader is too large
   for head tweaks. A second architecture (EffNet-B3, ConvNeXt-T) as
   an ensemble member is the realistic gap-closer. 3–6 day commitment.

**Deferred after sequence verdict:**
- T2.4 mel-diversity ensemble (stacks downstream of T2.2 if that lands).
- T2.5 Perch quality filter (1-day pipeline, +0.005–0.015, save for
  post-T2.2/B).

---

## ✅ C step DONE (2026-04-22 — val_v2 built, Perch self-agreement trap found)

**What shipped:**
- `src/rebuild_val_calibrate.py` — Phase 1: τ calibration on 59-overlap
  Perch cache. τ* = 8.0 (logit), restricted precision 0.9171 on 71 GT
  species. CPU <5 s.
- `src/rebuild_val_perch.py` — Phase 2a: Perch v2 ONNX on 500 new
  soundscape files (6,000 segments, ~5 min CPU). Output
  `data/processed/val_v2/perch_new500.npz`.
- `src/rebuild_val_build.py` — Phase 2b+2c: assembles Channel A
  (soundscape 2,371 segs / 123 sp) + Channel B (focal holdout 184
  clips / 184 sp). Output `val_v2_soundscape.npz` (1.63 GB) +
  `val_v2_focal.npz` (0.13 GB).
- `src/rescore_baseline_v2.py` — loads `a1_fold0.pt` (JIT, strip `inner.`
  prefix) and validates on both channels.

**v56 fold-0 baseline on val_v2 (stratified):**

| subset                       | segs | species | mean AUC |
|------------------------------|------|---------|----------|
| Channel A — GT-only subset   |  739 |  75     | **0.7416** |
| Channel A — Perch-only τ=8   | 1632 |  74     | 0.9371 |
| Channel A — combined         | 2371 | 123     | 0.8521 |
| Channel B — focal holdout    |  184 | 184     | **0.9545** |

**Perch↔A1 self-agreement trap (why Perch-labeled subset is NOT a gate):**
Perch and A1 respond to the same acoustic cues. When labels =
`(Perch_score > τ)`, any student whose features correlate with Perch's
will score artificially high on those segments. The 0.9371 is not
capability — it's self-agreement. Analogous to
`project_protossm_teacher_val_leakage` — L1 already confirmed the
pattern empirically (+163 val → −1 LB).

**Dual-gate protocol (adopted):**
1. **Primary gate** = Channel A GT-only subset → 0.7416. Bit-exact cached
   equivalent of the historical 0.7414 v1 gate. Apples-to-apples with
   every prior fold-0 val number in this plan.
2. **Per-species diagnostic** = Channel B focal holdout → 0.9545 baseline
   on 184 species. Flag regressions for species outside the 75-in-GT.
3. **Do NOT use:** Channel A Perch-only subset (0.9371 inflated); Channel
   A combined (0.8521 mixes honest + inflated).

**What rebuild did NOT fix:**
- 75/234 species in primary gate — geographic/labeling constraint of
  `train_soundscapes_labels.csv`. Cannot be fixed without off-policy data.
- Gate noise from single-recording dominance — value was marginal after
  dedup; not worth the complexity.

**Artifacts preserved:**
- `data/processed/val_v2/val_v2_soundscape.npz`  (2371 segs, 1.63 GB)
- `data/processed/val_v2/val_v2_focal.npz`       (184 clips, 0.13 GB)
- `data/processed/val_v2/perch_new500.npz`       (6000 segs, 5.6 MB)
- Logs: `log/rebuild_val_{perch,build}_20260422_*.log`,
  `log/rescore_baseline_v2_*.log`
- Memory: `project_val_v2_built.md`

**Not wired into `train_a1.build_soundscape_val()` yet** — because the
primary gate (Channel A GT-only) is bit-exact equivalent to the
existing v1 build, no wiring is strictly required for D/A(T2.2)/B to
start. Defer the wire-in until a retrain experiment wants the Channel B
diagnostic live during training.

---

## 14.13 D on hold — pivot to BC2025-winner levers (2026-04-22)

**Why D paused.** Phase D0 pipeline-sanity rescore of the existing local
`a1_..._seed42_hybrid.pt` (T2.3 prep artifact, Apr 21) produced val_v2
Channel A GT-only = 0.7204, vs v56 baseline 0.7416 — 0.021 below gate.
Running two additional drifted seeds (~3h GPU) would have produced an
ensemble anchored to that degraded baseline. Cost/benefit flipped.

**Also discovered during D0:** an unshipped archive ckpt at
`models/a1/archive_anuraset/a1_..._seed42_hybrid.pt` (Apr 19 raw, pre-
AnuraSet) scored 0.8385 on val_v2 GT-only, +0.097 over v56. Training
log for that ckpt shows BEST **v1 val = 0.7220** (epoch 25). The gap is
a val builder mismatch: `build_soundscape_val` scans 1478 CSV rows
(duplicates included); my `rebuild_val_build.py` deduplicates to 739.
For v56 the two gates happen to coincide (0.7414 ≈ 0.7416) but they
are NOT bit-exact equivalent. **The earlier memory claim that val_v2
GT-only = v1 gate was wrong for any non-v56 ckpt.**

Until val_v2 is re-audited, treat v1 training-loop val as the honest
selection gate (what v56 was selected on). Val_v2 Channel B focal
holdout is still fine as a separate per-species diagnostic.

**D plan paused artifacts (kept, not deleted):**
- `scripts/d_multi_seed_train.sh` (ready to run if needed)
- `src/rescore_ensemble_v2.py` (useful beyond D)

## 14.14 Fresh options from BC2025 winner survey (2026-04-22)

Kaggle-discussion sweep via subagent on 2026-04-22 surfaced six levers
top teams at LB 0.92–0.93 actually used. Ranked by effort × signal:

| # | Lever | Source | Claimed Δ | Cost |
|---|-------|--------|-----------|------|
| M1 | **Coarser mel grid** N_FFT=2048 / HOP=512 / N_MELS=128 | BC2025 top-2% retrospective (M. Melichov) | **+0.020** abs | 1.5h probe |
| M2 | Multi-layer GeM pool (EffNet blocks 3+4 concat→GeM) | BC2025 top-2% | +0.005–0.01 | 1.5h probe |
| M3 | **Multi-iteration Noisy Student** (≥2 iters, power α≈0.4, per-class thresh) | BC2025 1st place Babych | +0.03–0.05 | 2–3 days |
| M4 | Separate non-bird taxa head/model (Pantanal = multi-taxon) | BC2025 1st place | +0.003 | 1–2 days |
| M5 | Silero-VAD human-speech removal on focal audio | BC2025 top-2% | 0–0.01 | hours (preprocess) |
| M6 | fp16 OpenVINO conversion for CPU budget headroom | BC2025 2nd place | n/a (enables ensemble seats) | hours |

**Why the BC2025 1st-place recipe (M3) matters**: Babych's full NS chain
was CE-baseline (0.872) → MixUp+pseudo (0.898) → 4-round power-scaled
NS (0.930) → taxa split (0.933). The 4 iterations and power scaling
were the decisive step. Our `project_l1_killed` attempted **one**
iteration with BCE + max-merge + val-leakage confusion and killed after
LB 0.930 vs 0.931. We likely dismissed the single biggest lever in the
field too early.

**Validation of graveyard:** ~80% of our 11 killed ideas (CE loss,
stratified K-fold, >5 epoch overfit, full-recording input) also appear
on top-2% team's own "what hurt me" list. We're not irrational — the
graveyard reflects real dead ends, just with one structural miss
(NS was abandoned after a single misconfigured iteration).

**Discussions flagged for manual browser audit (agent couldn't scrape):**
- `birdclef-2026/discussion/681827` "EDA findings"
- `birdclef-2026/discussion/684693` "Potential Kaggle runtime incompatibility"

**What I still want to verify:**
- BC2026 CPU inference budget (BC2025 was 90 min). If 90 min stands,
  M6 OpenVINO conversion unlocks ensemble seats.

---

## ⏸️ PICK UP HERE (2026-04-22 01:15 local — M1 probe RUNNING overnight — SUPERSEDED)

**Status as of 2026-04-22 01:15 local:** mel-grid probe launched and running
in the background. User went to bed — pick up in the morning.

- **PID:** `545373` (python process; 545363 was the bash wrapper that
  exited after `nohup &`). Confirmed actively training: GPU 96%, 10 GB used.
- **Log:** `log/mel_probe_20260422_011536.log`
- **Config edits already landed:** `src/config.py` lines 33–35 set to
  `N_MELS=128 / N_FFT=2048 / HOP_LENGTH=512` (smoke-tested, model accepts
  new 3×128×1250 input).
- **Expected finish:** ~03:15 local (25 epochs × ~5m/epoch ≈ 2h, based on
  1.4× pixel bump vs prior 3m45s/epoch hybrid runs). Confirm actual
  per-epoch timing from epoch 1 first.

**Morning checklist:**
1. `tail -n 40 log/mel_probe_*.log` — check it finished cleanly (25 epoch
   summary lines, no traceback at the tail).
2. Read the BEST line — the `★ BEST` val_roc_auc is our decision metric.
3. Apply gate:
   - **v1 val ≥ 0.7420** → probe passes → next step is LB probe (JIT export
     the coarser-mel ckpt, update notebook mel params, back up v56 first,
     Kaggle push).
   - **0.7000 ≤ v1 val < 0.7420** → judgment call — document and move on to
     M2 multi-layer GeM from §14.14 menu.
   - **v1 val < 0.7000** → hard-kill, revert `src/config.py`, move on to M2.
4. If reverting: `git checkout /home/swatson/work/kaggle/BirdCLEF/src/config.py`
   restores `N_MELS=224 / N_FFT=4096 / HOP_LENGTH=1252`.

**If probe crashed (no 25-line summary):** check the traceback at the end
of the log — most likely either OOM (batch size tuning needed for the
bigger time dim) or a dataset path issue. Do NOT re-launch blindly.

---

**The probe.** One fold-0 hybrid run with mel grid switched from
`N_FFT=4096 / HOP=1252 / N_MELS=224` (producing shape 3×224×512) to
`N_FFT=2048 / HOP=512 / N_MELS=128` (producing shape 3×128×1250 for
a 20 s window). Seed 42, 25 epochs, `--loss hybrid`, `--mixstyle-p 0.5`.

**Gate (honest, v1 training-loop val):**
- Current-pipeline baseline at current mel = 0.7220 v1 val (Apr 21 T2.3
  prep run, single-BEST).
- Historical v56 baseline at current mel = 0.7414 v1 val.
- Probe target = coarser-mel v1 val ≥ 0.7420 (= current-pipeline 0.7220
  + claimed +0.020) → suggests mel helps; run LB probe next.
- Probe hard-kill = coarser-mel v1 val < 0.7000 → mel change hurt more
  than it helped.

**Changes needed:**
1. `src/config.py`: N_FFT 4096 → 2048, HOP_LENGTH 1252 → 512,
   N_MELS 224 → 128.
2. Model architecture: `BirdSEDModelA1` uses EffNet backbone with
   adaptive pool, so N_MELS + T flexibility is expected. Verify smoke
   test before launch.
3. val_v2 precomputed NPZs are STALE after mel change (computed with
   old grid). Don't use for this experiment — rely on v1 training-loop
   val in `train_a1.py`.

**Ship path if probe passes:**
- JIT-export coarser-mel fold-0 ckpt.
- Notebook change: match mel params in inference spec.
- Back up v56 ckpts first (per `feedback_backup_ckpts_before_overwrite`).
- Kaggle push + LB probe.

**If probe fails, next option from §14.14 menu:**
- M2 multi-layer GeM (1.5h, cheap, similar risk profile).
- If M1 and M2 both fail, step up to M3 (multi-iter NS) as the day-scale
  commit — but only after fixing the val-leakage landmine that killed L1.

**T2.3 artifacts (cleanup when convenient):**
- Soup ckpt `models/a1/a1_..._hybrid_soup.pt` (val 0.6921 — killed).
- Single-BEST ckpt `models/a1/a1_..._hybrid.pt` (val 0.7203 — tied with prod).
- 25 per-epoch ckpts at `models/a1/_soup/fold0_hybrid_seed42/*.pt` (~420 MB).
- 25 per-epoch ckpts at `models/a1/_soup/fold0_asl_seed42/*.pt` (~420 MB,
  leftover from the wrong-loss run in §14.11.10.10).
- **Cleanup candidates:** both `_soup/fold0_*_seed42/` dirs (~840 MB
  total) can be deleted — soup approach is killed on this schedule.
  Keep sidecars + meta.npz.

---

## 14.14.1 M1 coarser-mel KILLED (2026-04-22 03:18 local)

**Run.** 25 epoch hybrid fold-0, seed 42, mixstyle 0.5, mel grid
`N_FFT=2048 / HOP=512 / N_MELS=128` (shape 3×128×1250). Finished cleanly
at 2026-04-22 03:18:50, total runtime 2h 03m 13s, ~4m50s/epoch.

**Result.** BEST v1 val_roc_auc **0.6819** at epoch 20.

| Benchmark | v1 val | Δ vs M1 |
|---|---|---|
| M1 coarser-mel BEST (ep 20) | **0.6819** | — |
| Current-pipeline baseline (T2.3 prep, Apr 21) | 0.7220 | −0.040 |
| v56 production baseline | 0.7414 | −0.060 |
| Probe pass gate (+0.020 claimed) | 0.7420 | −0.060 |
| Hard-kill gate | 0.7000 | **−0.018 → HARD KILL** |

BEST came two-thirds into the run (ep 20 of 25) and the curve is noisy
— 0.6118 / 0.6215 / 0.6695 / 0.6792 / 0.6819 across the `★ BEST` line
— but never crossed the hard-kill line even once. No judgment call
needed: 0.6819 < 0.7000.

**Diagnosis (brief).** The coarser 128-mel × 1250-frame grid is a
substantial change in spectrogram aspect ratio vs the 224×512 grid the
EffNet-B0 + its hybrid head were tuned to during our original schedule.
Every other hyper — mixstyle p, sched, loss mixing, norm stats, GeM
power — was locked to the old grid. Melichov's claimed +0.020 was on
his own full pipeline; transplanted mel params alone onto a different
backbone/head/aug stack is not a drop-in win.

**Revert executed (2026-04-22 ~14:37 local).**
- `git checkout /home/swatson/work/kaggle/BirdCLEF/src/config.py`
  → restored N_MELS=224 / N_FFT=4096 / HOP_LENGTH=1252. Confirmed by
  re-reading lines 33–35.
- No Kaggle artifact was touched; v56 notebook + a1-effb0-ckpts
  unchanged.

**Collateral.**
- `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid.pt`
  is now the M1 ckpt (val 0.6819, coarser-mel) — it **overwrote** the
  Apr 21 T2.3-prep single-BEST (val 0.7203). This is fine in the scheme
  of the schedule (T2.3 was already killed and its single-BEST was only
  "tied with prod"; production lives JIT-packed in
  `kaggle_datasets/a1-effb0-ckpts/`), but be aware: loading this `.pt`
  at production mel grid will mis-configure the feature extractor.
  Either delete or rename with an `_M1_mel128` suffix when convenient.
- Log kept: `log/mel_probe_20260422_011536.log`.

**Shortlist update.** M1 off the menu. Next from §14.14:
- **M2 multi-layer GeM** (1.5h, same fold-0 probe shape, cheap). Concat
  EffNet blocks 3+4 before the GeM head.
- **M3 multi-iteration Noisy Student** (2–3 days, much bigger commit)
  if M2 also fails.
- M4/M5/M6 remain but are lower priority.

---

## ⏸️ PICK UP HERE (2026-04-22 14:46 local — M2 multi-layer GeM probe RUNNING — SUPERSEDED)

**Status.** M1 coarser-mel KILLED earlier today. Config reverted to v56
mel grid. M2 multi-layer GeM probe launched and running. Expect finish
~16:30–17:00 local (25 ep × ~4m/ep ≈ 1.7h).

**Run info:**
- **Python PID:** `557829` (bash wrapper PID 557818 exits immediately).
- **Log:** `log/m2_probe_20260422_144522.log`
- **Command:** `python -u src/train_a1.py --fold 0 --epochs 25 --seed 42 --loss hybrid --mixstyle-p 0.5 --multi-layer-gem`
- **Smoke-test PASSED** before launch: clean exit, correct epoch-summary
  format, 1 epoch × 2 batches. val_roc_auc=nan in smoke is expected (one
  class per species in the 2-batch validation subset).

**Code landed:**
- `src/model_a1.py` — `BirdSEDModelA1(..., multi_layer_gem=False)` now
  optional. Under the flag: `out_indices=(3,4)`, per-scale
  `GEMFrequencyPool`, adaptive time-pool of stage 3 down to stage 4's
  T, concat over channels. Verified: `cls_conv.in_channels = 432`
  (= 112 + 320) for EffNet-B0.
- `src/train_a1.py` — CLI flag `--multi-layer-gem`, threaded through
  `train_one_fold` → model ctor, logged in the run banner.
- MixStyle hook untouched (still on `backbone.blocks[1]`).

**Ckpt hygiene done:**
- Renamed M1 ckpt to `a1_..._fold0_seed42_hybrid_M1_mel128.pt` before
  launching M2 so M1 evidence isn't overwritten.
- M2 will write to `a1_..._fold0_seed42_hybrid.pt` (same filename as
  old T2.3-prep ckpt — not production; production is JIT in
  `kaggle_datasets/a1-effb0-ckpts/`).

**When probe finishes, apply gate:**
1. `tail -n 40 log/m2_probe_*.log` — 25 epoch summary lines, no tb.
2. Read BEST line → decision metric.
3. Apply:
   - **v1 val ≥ 0.7420** → probe passes → LB probe next (JIT export,
     notebook update, Kaggle push).
   - **0.7000 ≤ v1 val < 0.7420** → judgment call — document +
     move on to M3 or M4.
   - **v1 val < 0.7000** → hard-kill. No config to revert (M2 is
     CLI-flag-gated); just note the result, rename the M2 ckpt aside,
     and move to M3/M4.

**Realistic prior.** BC2025 top-2% claimed +0.005–0.01 in their full
pipeline. Lesson from M1: single-lever transplants underdeliver
relative to claim. Outcome envelope ~0.7200–0.7450. ≥0.7414 is a real
win; 0.7300–0.7413 is a judgment call; <0.7200 is a hard miss.

**Next-lever queue after M2 (both outcomes):**
- M3 multi-iter Noisy Student (day-scale; only after fixing L1
  val-leakage — see `project_protossm_teacher_val_leakage` memory).
- M4 separate non-bird-taxa head (Pantanal is multi-taxon).
- M5 Silero-VAD human-speech removal on focal audio.
- M6 fp16 OpenVINO (enables ensemble seats if 90-min CPU budget holds).
- **P1–P6 (see §14.14.2)** — cheap levers extracted from LB 0.925/0.926
  public notebooks on 2026-04-22. All inference-only or training-side,
  none address the leader gap but some may pad vs 2nd tier.

## 14.14.2 Cheap-lever shortlist from public LB 0.925/0.926 notebooks (2026-04-22)

Pulled & audited two public notebooks on 2026-04-22:
- `yaroslavkholmirzayev/birdclef-2026-protossm-v5-maximum-ensemble` LB 0.925
- `pradeeshrajan/pantanal-distill-birdclef2026-improvement` LB 0.926

Both are **Perch-v2→ProtoSSM**, not mel→CNN SED. Both train downstream
SSM heads on frozen Perch embeddings using GT soundscape labels (no
pseudo-label chain → they avoid the val-leakage trap we fell into with
Track C / L1). Neither beats our 0.931 A1 ensemble.

Extracted six cheap levers common to one or both. Each is likely
+0.001–0.003 on LB. Stacked (if independent) maybe +0.005. None
addresses the 0.931→0.951 leader gap.

| # | Lever | Venue | Cost | Risk | Claimed role |
|---|-------|-------|------|------|--------------|
| P1 | Per-taxon temperature (Aves T≈1.05–1.10, texture T≈0.95) | inference | hours | low | spread texture, concentrate Aves |
| P2 | File-level circular-shift TTA [0,±1,±2] windows | inference | hours | low | free LB lift |
| P3 | Rank-aware power scaling α=0.4–0.5 within-file | inference | hours | med (may cannibalize delta_shift α=0.20) | within-file dynamic range |
| P4 | Taxon-aware Gaussian σ: σ=0.35 texture / 0.15 event | inference | hours | low | currently single σ |
| P5 | SWA from swa_start_frac=0.65 during training | training | 1.5h/fold probe | med (T2.3 naive soup killed) | schedule-aware averaging |
| P6 | Taxonomic-family aux BCE loss, weight 0.1 | training | 1.5h/fold probe | low | taxonomy-based regularizer |

**Stack-compatibility notes:**
- P3 (rank-power) may **cannibalize** our existing delta_shift α=0.20 —
  both squash within-file dynamic range. A/B on LB individually before
  stacking.
- P4 needs a per-taxon lookup from `taxonomy.csv` (Aves vs Amphibia vs
  Insecta vs Mammalia vs Reptilia). Cheap.
- P5 is the one we burned with T2.3 naive ckpt soup — but PyTorch SWA
  with `AveragedModel + SWALR` is the real thing (updates BN stats via
  `update_bn` pass); that's not what T2.3 did.
- P6 requires re-train. Weight 0.1 is small so base head stays
  dominant.

**Already-have (do NOT re-add):**
- delta_shift α=0.20 — we match their α exactly.
- OneCycleLR / cosine warm restart — we use cosine warm restarts.
- Mixup/MixStyle — we have FreqMixStyle p=0.5.

**No external data used in either notebook.** Reinforces §14.10.17.7 /
§14.11.6.6.2 / §14.11.7 conclusion: external-data lever is closed.

### Priority order (proposed, 2026-04-22)

Assuming M2 lands in its expected 0.68–0.71 range (not a win):

1. **P2 circular-shift TTA** — cheapest possible: edit inference
   notebook only, no retrain, one Kaggle push. Risk near zero.
2. **P1 per-taxon temperature** — also inference-only. Need taxonomy
   lookup for T lookup; one grid-search or use literal values from
   pantanal-distill (1.05 / 0.95).
3. **P5 SWA training** — training-side probe, same shape as M1/M2,
   with honest PyTorch SWA (AveragedModel + SWALR + update_bn), NOT
   naive ckpt averaging. Probe one fold first.
4. **P4 taxon-aware Gaussian σ** — inference-only but slight
   integration work in notebook.
5. **P3 rank-power** — only after confirming it doesn't cannibalize
   delta_shift; probably skip if P1+P2+P5 already lift.
6. **P6 family aux loss** — training-side, low-weight regularizer;
   cheapest retrain but lowest expected Δ.

### What this shortlist does NOT solve

The 0.931→0.951 leader gap. P1–P6 collectively will not close it.
M3 multi-iter Noisy Student remains the only known day-scale lever
with leader-gap-sized potential, but it needs the L1 val-leakage
landmine resolved first (clean per-species thresholds, max-merge
vs mean-merge, BCE vs ASL).

## 14.14.3 Additional cheap levers from competitor master report (2026-04-22)

User surfaced a BC2026 forum attachment:
`kaggle-forum-message-attachments/3422905/39324/master_report_20260318.html`
— another team's internal master report, snapshot 2026-03-18. Their
LB at report time was **0.893**, below our 0.931 baseline. Pipeline
is parallel to ours (EffNet-B0 SED + Perch + ASL dual-loss + SpecAug
+ mixup) but at an earlier stage.

**Confirmed already-have / already-killed** (no action):
- ASL γ⁻=4 γ⁺=0 — match.
- Dual clip+frame loss — we have it.
- Model soup top-3 — our T2.3 killed (cosine warm restart incompat).
- Mixup / SpecAug / Gaussian noise — FreqMixStyle + SpecAug cover.
- Multi-year pretrain 2021–2024 — our L2 killed.
- PCEN "catastrophic collapse" — their integration fails; **our
  PCEN works** (our banner is "PCEN+ASL+FreqMixStyle"). Different
  integration; no action.
- "Holdout AUC doesn't predict LB" — empirically confirmed by us
  multiple times (T2.6, L1, v54, M1).
- 5-round iterative pseudo-labeling → our M3 (open, day-scale).

**NEW cheap levers (P7–P10) not in §14.14.2:**

| # | Lever | Cost | Risk | Claimed Δ |
|---|-------|------|------|-----------|
| P7 | **Background-noise SNR aug** — mix focal clips with field-recording noise at SNR 5–30 dB | 1.5h probe + noise corpus setup | med | +0.005–0.01 |
| P8 | **RMS-based segment selection** — replace random 20s crop with highest-RMS 20s window per focal file | hours (dataset class edit only) | low | +0.005–0.015 |
| P9 | **3-tap temporal smoothing at inference** — `0.2·p_{t-1} + 0.6·p_t + 0.2·p_{t+1}` | hours | med (may cannibalize our delta_shift α=0.20, same risk profile as P3 rank-power) | +0.002–0.005 |
| P10 | **Frame-level Perch distillation** — use Perch's time-varying predictions as dense per-frame supervision, not clip-level embeddings | day-scale | med (Track C-style val-leakage risk) | unknown |

**Stack-compatibility notes:**
- **P7 noise corpus.** Can reuse `train_folds_pre_anuraset.csv` Pantanal-
  bbox excluded rows (same filter we built for `--anuraset-mixup`), so
  no new dataset needed. Differs from anuraset-mixup in that mixin is
  generic field audio at variable SNR, not AnuraSet-specific.
- **P8 segment selection.** `BirdTrainDatasetA1.__getitem__` currently
  picks a random 20s window. RMS-based picks the window with highest
  RMS energy across the focal clip. One-pass preprocessing to cache
  the best-window offset per training file is cleaner than online.
- **P9 smoothing.** Literal 3-tap kernel differs from delta_shift
  (which applies an asymmetric max-prev/min-next soft correction) and
  from P4 Gaussian (continuous σ). A/B individually; stacking P9 and
  delta_shift is likely redundant.
- **P10 frame distillation.** Same conceptual failure mode as Track C:
  teacher trains on train_soundscapes → student validates on
  train_soundscapes. **Do NOT attempt before M3** — same leakage
  mitigation work needed.

### Re-merged priority order (2026-04-22, after §14.14.2 + §14.14.3)

Assuming M2 lands in its expected <0.72 range (not a win):

1. **P8 RMS-based segment selection** — code-only, biggest claimed Δ
   among cheap levers, independent of post-proc stack.
2. **P2 circular-shift TTA** — inference-only, cheapest LB probe.
3. **P1 per-taxon temperature** — inference-only, literal values from
   pantanal-distill (Aves 1.05 / texture 0.95).
4. **P7 background-noise SNR mixup** — training-side; reuses the
   existing `train_folds_pre_anuraset` filter to build the mixin pool.
5. **P5 SWA** — honest PyTorch SWA (AveragedModel + SWALR + update_bn).
6. **P9 3-tap smoothing** — only after confirming P3/P4/delta_shift
   don't already cover the signal.
7. **P4 taxon-aware Gaussian σ**, **P6 family aux loss**,
   **P3 rank-power** (last because cannibalization risk with delta_shift).
8. **P10 frame-level Perch distill** — day-scale, after M3 decided.

### What P1–P10 together do NOT solve

Still the 0.931→0.951 leader gap. Stacked optimistically the cheap
levers pad maybe +0.008–0.015. Leader-gap-sized wins remain M3 (iter
NS) and a potential larger backbone (EffNetV2-B3 / ECA-NFNet-L0)
swap — both day-scale commits, neither attempted yet.

### 14.14.3 amendment — P11–P14 from Maryna two-pass SSM notebook (2026-04-22)

Audited `marynaborovska/birdclef-26-two-pass-ssm-advanced-pp` on
2026-04-22. Third Perch→ProtoSSM variant (same family as §14.14.2
yaroslav / pantanal-distill). No public LB in metadata. Clean
GroupKFold, no val leakage. External data: Perch v2 only.

Most of Maryna's recipe overlaps P1–P10 already. Confirms exact
values worth trying:
- **P1 per-taxon T:** Aves 1.10, Amphibia/Insecta 0.95 (same as
  pantanal-distill).
- **P3 rank-aware power:** 0.4.
- **P5 SWA:** swa_start_frac=0.65, swa_lr=4e-4, `update_bn` at end.

Four NEW post-proc levers:

| # | Lever | Basis | Cost | Expected |
|---|-------|-------|------|----------|
| P11 | **Confidence-modulated delta smoothing** — `α = α_base·(1 − max_prob_t)`, vs our fixed α=0.20 | Maryna PP step 5 | hours | +0.001–0.003 |
| P12 | **Per-class isotonic calibration** on OOF predictions | Maryna PP step 6 | hours | +0.002–0.005 |
| P13 | **File-level top-k mean confidence scaling** (top-2, power=0.4) | Maryna PP step 3 | hours | small |
| P14 | **Genus-level proxy for unmapped Perch species** — max Perch logit over same-genus species when target absent from Perch labels | Maryna genus handler | hours | small, zero-risk |

**Stack-compatibility notes:**
- **P11** is an **upgrade path** for our existing delta_shift, not an
  additive stack item. Swap in, don't layer. Base α stays 0.20; the
  modulation by `(1 − max_prob_t)` protects confident peaks.
- **P12** needs an OOF prediction set for training soundscapes. We
  already have `data/d2_beta_oofs.npz`-family artifacts — verify one
  is aligned with the current A1 baseline before fitting isotonic
  regressors. sklearn `IsotonicRegression` out-of-the-box.
- **P13** is a close cousin of **P3 rank-aware** (theirs uses top-k
  *mean*, P3 uses *max*). A/B against P3, not additive.
- **P14** needs the 234-species taxonomy → Perch label mapping audit.
  If all 234 species are covered by Perch, P14 is a no-op for us.
  Otherwise, small free win on the rare/unmapped classes.

**Confirmed-absent in Maryna's notebook** (reinforces earlier audits):
- No noise mixup → P7 still an open lever.
- No RMS segment selection → P8 still open.
- No external data beyond Perch v2 → external-data lever still closed.

**Architectural ideas NOT promoted to P-numbers** (too costly or
structurally different):
- Learnable class prototypes + cosine-sim head (replaces dense
  classifier).
- Cross-attention between fwd/bwd SSM layers.
- Two-pass residual SSM corrector on first-pass logits.
- MLP probes on PCA-reduced Perch embeddings.
These are ProtoSSM-track ideas; our A1 SED doesn't accommodate them
cheaply. Revisit only if we stand up a Track B second backbone.

### Re-merged priority order (2026-04-22, after §14.14.3 amendment)

Assuming M2 lands in its expected <0.72 range (not a win):

1. **P8 RMS-based segment selection** — training-side, biggest claimed
   Δ among cheap levers.
2. **P2 circular-shift TTA** — inference-only, cheapest LB probe.
3. **P11 confidence-modulated delta smoothing** — inference-only
   upgrade path for existing delta_shift.
4. **P1 per-taxon temperature** (Aves 1.10 / texture 0.95) —
   inference-only.
5. **P12 per-class isotonic calibration** — inference-only, needs
   OOF alignment check.
6. **P14 genus proxy** — inference-only, needs taxonomy audit.
7. **P7 background-noise SNR mixup** — training-side; reuses the
   existing `train_folds_pre_anuraset` filter.
8. **P5 SWA** — honest PyTorch SWA.
9. **P13 top-k mean scaling** — A/B against P3 rank-aware.
10. **P4 taxon-aware Gaussian σ**, **P6 family aux loss**,
    **P3 rank-power** (cannibalization risk with P11/delta_shift).
11. **P9 3-tap smoothing** — superseded by P11; skip unless P11 fails.
12. **P10 frame-level Perch distill** — day-scale, after M3 decided.

## 14.14.4 M2 multi-layer GeM KILLED (2026-04-22 15:46 local)

**Run.** 25-ep hybrid fold-0 seed 42 mixstyle 0.5 `--multi-layer-gem`
targeted; killed at **epoch 15/25** after 60m elapsed on early-kill
decision — curve verdict unambiguous. Mel grid v56 production
(224×512); heads 432 C (= 112 block-3 + 320 block-4).

**Result.** BEST v1 val_roc_auc **0.6771** at epoch 5. Ten subsequent
epochs (6–15) all regressed to 0.64–0.66 range while train_loss fell
3.5× (0.0470 → 0.0166) — classic train/val divergence.

| Benchmark | v1 val | Δ vs M2 |
|---|---|---|
| M2 BEST (ep 5) | **0.6771** | — |
| M1 mel128 BEST (ep 20) | 0.6819 | +0.005 |
| T2.3-prep baseline (Apr 21) | 0.7220 | +0.045 |
| v56 production baseline | 0.7414 | +0.064 |
| Probe pass gate | 0.7420 | **+0.065 → FAIL** |
| Hard-kill gate | 0.7000 | **−0.023 → HARD KILL** |

**Comparison with M1 at matched epochs:**

| ep | M1 val | M2 val | winner |
|----|--------|--------|--------|
| 5 | 0.6215 | 0.6771 | M2 |
| 9 | 0.6792 | 0.6596 | M1 |
| 13 | 0.6531 | 0.6506 | ≈ |
| 15 | 0.6723 | 0.6535 | M1 |

M2 peaked earlier but could not sustain; M1's late-epoch rally to
0.6819 looked like the ceiling for both recipes. Even if M2 mirrored
that pattern it would have landed ≤ 0.68, still well under the 0.7000
hard-kill gate.

**Diagnosis (brief).** Adding stage-3 features plus a second GeM head
increases parameter count (~16.5 → ~17.7 MB ckpt) and learnable
capacity at the classifier, but on fold-0 with no per-scale dropout,
it just overfits the focal training distribution faster. MixStyle
injection happens mid-backbone (blocks[1]); the second GeM on stage 3
receives features that are strongly stylized on focal statistics, not
soundscape. Nothing in the recipe compensates for the extra capacity
at the soundscape-evaluation boundary. The +0.005 BC2025 top-2% claim
was (as hypothesized before M1) pipeline-bundled.

**Killed at ep 15 to save ~38 min GPU** — user decision after
reviewing train/val curve. No late-rally probability justified the
additional cost.

**Collateral.**
- M2 BEST ckpt renamed to
  `models/a1/a1_..._fold0_seed42_hybrid_M2_multilayer_gem.pt`
  (17.7 MB — larger than baseline; shape-mismatch if loaded at
  `--multi-layer-gem=False`; retained as evidence only).
- M1 ckpt also preserved at `..._hybrid_M1_mel128.pt`.
- No production artifact touched. v56 Kaggle dataset + notebook
  unchanged.
- `--multi-layer-gem` CLI flag left in `src/train_a1.py`; model path
  in `src/model_a1.py` retained but gated off by default. Removing
  later is safe but not urgent.

**Two-for-two on M-tier.** M1 (mel grid) and M2 (multi-layer GeM)
both killed. Lesson confirmed: BC2025-bundle lever transplants onto
our SED stack underdeliver relative to claims. Don't probe further
BC2025 single-lever transplants without compensating hyperparam
re-tune.

**Shortlist after M2 kill.** M-tier off the cheap-probe menu
(M3 remains open as day-scale). Cheap-probe priority is now the
P-tier (see §14.14.3 amendment). Top of queue:

1. **P8 RMS-based segment selection** — training-side, biggest
   claimed Δ (+0.005–0.015) among cheap levers, independent of
   post-proc stack.
2. **P2 circular-shift TTA** — inference-only, cheapest LB probe.
3. **P11 confidence-modulated delta smoothing** — inference-only
   upgrade to our existing delta_shift.

---

## ⏸️ PICK UP HERE (2026-04-22 15:46 local — M2 killed, P8 queued — SUPERSEDED)

**Status.** Both M1 (mel grid, val 0.6819) and M2 (multi-layer GeM,
val 0.6771) killed. Config and src are at v56 production state.
Next lever is **P8 RMS-based segment selection** — cheapest + highest
claimed Δ among the §14.14.3 P1–P14 shortlist.

**Next action — P8 RMS-based segment selection.**

Plan:
1. Read `src/dataset_a1.py` (BirdTrainDatasetA1) — locate the
   20-second window crop logic.
2. Decide: online (compute RMS at `__getitem__` time) or offline
   (one-pass preprocessing to cache best-window offset per training
   file). Offline is cleaner and much faster per epoch.
3. If offline: write `src/precompute_rms_offsets.py` that scans
   focal training audio and writes `data/processed/rms_best_offsets.parquet`
   with columns `[filename, best_offset_samples, best_rms]`.
4. Modify dataset class: add `--rms-select` flag to `train_a1.py`;
   when on, look up `best_offset_samples` instead of random.
5. Smoke-test (1 epoch, 2 batches) to verify offset table join +
   window extraction.
6. Launch fold-0 hybrid / seed 42 / 25 epochs / mixstyle 0.5 +
   `--rms-select`. Everything else locked to v56 config.
7. Gate structure unchanged from M1/M2:
   - v1 val ≥ 0.7420 → LB probe.
   - 0.7000 ≤ v1 val < 0.7420 → judgment call.
   - v1 val < 0.7000 → hard-kill.

**What NOT to do this session:**
- Do not let the P8 training run overwrite
  `models/a1/a1_..._fold0_seed42_hybrid.pt` without first confirming
  the M1/M2 `_suffix.pt` files are still present as evidence archive.
- Do not stack P8 with any other cheap lever until its standalone Δ
  is known.

**If P8 kills too:** next training-side probe is P5 honest SWA
(one fold, same shape). Inference-only probes (P2 TTA, P11
conf-mod smoothing, P1 per-taxon T) can also be tried on
production v56 without a retrain — consider them in parallel.

## 14.14.5 "Noisy Classmates" flagged and deferred (2026-04-22)

**Source.** LinkedIn post by Lin-Chieh Huang (UGC post id
`7447496616951050240`) claiming **3rd place on BirdCLEF 2026**
with a method named *"Noisy Classmates: Multi-Architecture
Co-Evolutionary Self-Training for Bioacoustic Species
Recognition."* Post is high-level; no writeup, code repo, or
Kaggle solution thread surfaced in search on 2026-04-22.

**What it is (decoded).**
- **Multi-architecture** — ensemble of different backbones
  (EffNet-B0 / NFNet-L0 / ConvNeXt / EffNetV2-B3 family).
- **Co-evolutionary self-training** — each model's pseudo-labels
  supervise the *other* models, not itself. Breaks the single-
  model bootstrap bias that sinks naive iterative NS.
- **Iterative** — multi-round; each round, at least one model
  feeds pseudo-labels forward to another.
- **Evolutionary algorithm** — likely pseudo-label selection by
  cross-model agreement (consensus voting) or hyperparam search
  (not specified in the post).
- **Active learning** — likely targets low-agreement samples
  for extra refinement.

Structurally a **strict superset of M3** (§14.14): adds multi-
arch cross-teaching + pseudo-label ensembling to the BC2025
1st-place iterative NS recipe.

**Why we are NOT pivoting to this now.**
1. **Needs ≥2 trained backbones.** We have A1 (EffNet-B0).
   Track B (second backbone) is unstaffed. Standing up Track B
   is itself a multi-day commit.
2. **Val-leakage landmine compounded.** Any soundscape-derived
   pseudo-label chain that validates on train_soundscapes is a
   confirmed trap (`project_l1_killed`,
   `project_protossm_teacher_val_leakage`). Multi-round cross-
   model NS makes it worse — each round re-infects the next
   model.
3. **Kaggle CPU budget (90 min).** Multi-arch inference can
   blow the budget; OpenVINO (M6) would become a hard
   prerequisite.
4. **Mid-competition claim.** BC2026 is still running on
   2026-04-22. Public LB "3rd place" is a snapshot, not a final
   result; no writeup published means no replicable recipe.

**When this becomes actionable.**
Only after ALL of:
- M3 single-arch iterative NS tried (and either won, or failed
  cleanly with the val-leakage mitigation understood).
- Track B second backbone stood up with a genuinely orthogonal
  architecture (NFNet-L0 or EffNetV2-B3).
- Author posts a Kaggle writeup or code with an actual recipe
  (hyperparams, iteration count, pseudo-label filter rule).

Until those unlock, leave M3 single-arch as the next day-scale
NS commit — Classmates is strictly later in the queue.

## 14.14.6 P8 RMS segment selection — partial win, ceiling at ≈ 0.73 (2026-04-22)

**Two-run sweep on fold-0 hybrid seed 42 mixstyle 0.5 `--rms-select`.**
RMS-biased 20 s crop: top-3 of candidate starts at 5 s stride by
window RMS (cumsum-of-squares). Online in
`src/dataset_a1.py::_pick_rms_window`.

| Run | Epochs | BEST val | At ep | Runtime |
|-----|--------|----------|-------|---------|
| P8-25ep | 25 | **0.7298** | 25 (final) | 2h 06m |
| P8-40ep | 40 | 0.7255 | 34 | 3h 21m |

**Verdict:** judgment-call zone (0.7000 ≤ 0.7298 < 0.7420). **+0.008
above current-pipeline baseline** (T2.3-prep 0.7220, Apr 21), **−0.012
below v56 production** fold-0 baseline (0.7414, but that was a
pre-AnuraSet-merge, pre-BC2025-merge training distribution — not a
like-for-like comparison). First P-lever in §14.14.2/§14.14.3 to show
a positive training-side delta over the current-pipeline.

**Schedule-length lesson (2026-04-22, earned the hard way):** cosine
warm-restart LR schedule **shape** is load-bearing. Stretching total
epochs 25 → 40 did not produce a higher peak — it produced a gentler
LR swing and a lower ceiling (0.7255 vs 0.7298). More epochs in a
cosine-warm-restart recipe is not monotone. Schedule-length tuning
is a hyperparam in its own right; don't just increase epoch count
hoping for a free rally.

**Artifact state.**
- `models/a1/a1_..._fold0_seed42_hybrid_P8_25ep.pt` — current
  candidate (0.7298 val).
- `models/a1/a1_..._fold0_seed42_hybrid_P8_40ep.pt` — schedule
  control (0.7255 val, evidence).
- `models/a1/a1_..._fold0_seed42_hybrid.pt` — does NOT exist;
  both BEST runs renamed aside. Any future run on this fold/seed
  will write a fresh file here.
- `--rms-select` flag + `_pick_rms_window` helper retained in src
  (default off).

**Why we are NOT JIT-exporting P8-25ep for LB yet.**
- Single-fold only. 5-fold ensemble is the honest LB measurement.
- −0.012 gap to v56 fold-0 baseline (even with training-distribution
  confound) makes a single-fold LB probe a likely −ve submission
  result.
- Kaggle submission slots are limited; spend them on ensemble-level
  probes, not single-fold variants.

### Next action options (2026-04-22 21:25 local)

Given P8 is a partial win (keep, stack) and not a standalone winner:

1. **P2 circular-shift TTA on the EXISTING production v56 ensemble** —
   inference-only (notebook edit only), no retrain, no GPU, LB
   probe on production directly. Cheapest possible probe. 🟢 RECOMMENDED.
2. **P11 confidence-modulated delta smoothing** (upgrade
   delta_shift) on production v56 — inference-only. Same shape
   as P2.
3. **P1 per-taxon temperature** on production v56 — inference-only.
   Literal values from pantanal-distill (Aves 1.05, texture 0.95).
4. **Stack 5-fold P8 training** — 10h GPU commit, gives ensemble-
   level P8 measurement. Heavy but clean data.
5. **P5 honest SWA as second training-side probe** — 2h fold-0,
   lighter than 5-fold P8.

Recommended sequencing: P2 → P11 → P1 (three single-LB-submission
inference probes on production v56), then decide whether to commit
to 5-fold P8 based on what lifts.

---

## ⏸️ PICK UP HERE (2026-04-22 21:25 local — P8 partial win, next = P2 TTA on production — SUPERSEDED by §14.14.7 audit + overnight 5-fold)

**Status.** P8 25-ep is a partial win at fold-0 val 0.7298 (+0.008
vs current-pipeline baseline, −0.012 vs v56 fold-0 baseline).
Schedule-length experiment (40-ep) landed at 0.7255, confirming
25-ep is the recipe sweet spot. M-tier killed two-for-two. P-tier
has its first positive signal.

**Next action — P2 circular-shift TTA on production v56, inference-only.**

Plan:
1. Read the current inference notebook (most likely
   `jupyter/sed/birdclef2026-sed-inference.ipynb` or the v56-family
   notebook — identify which is the production path).
2. Locate the window inference loop — where it calls the model
   per 5-s or 20-s window.
3. Add circular-shift TTA: for each input window, run inference
   at shift offsets {0, +5s, −5s} (or similar), then average
   the softmax outputs. Literal recipe from pantanal-distill /
   Maryna notebooks is [0, ±1, ±2] * 5s shifts — start with
   [0, ±1] for a minimal 3-way TTA.
4. Back up v56 ckpts BEFORE any export script touches
   `kaggle_datasets/` (per
   `feedback_backup_ckpts_before_overwrite`).
5. LB push, note delta vs 0.931.

**What NOT to do this session:**
- Do not retrain anything. P2 is inference-only.
- Do not JIT-export the P8 fold-0 ckpt (still single-fold).
- Do not touch `src/config.py` or training code.

**If P2 lifts LB ≥ +0.002:** push to ensemble, try P11 next.
**If P2 is neutral / negative:** try P11 (different post-proc
operator) before concluding inference-only is saturated.
**If P2 + P11 + P1 all neutral:** the cheap-lever pad is closed;
next moves are 5-fold P8 training or M3 day-scale commit.

## 14.14.7 Production post-proc stack audit — P1/P3/P11/P13 already live (2026-04-22)

Post-P8 session, before starting P2, audited the actual production
notebook (`jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb`,
kernel-metadata → kaggle dataset `stevewatson999/birdclef-2026-protossm`).

**Production Cell 18 ("V17: Full post-processing pipeline") already
implements:**

| Lever | Cell 18 step | Production value | Audit recommendation | Status |
|-------|-------------|------------------|----------------------|--------|
| P1 per-taxon temperature | Step 1 (`T_AVES` / `T_TEXTURE`) | 1.10 / 0.95 | 1.05–1.10 / 0.95 | ✅ MATCH |
| P13 file-level top-k scaling | Step 2 (`file_level_top_k`) | 2 | 2 | ✅ MATCH |
| P3 rank-aware power | Step 3 (`rank_aware_power`) | 0.5 default / **0.4 submit** | 0.4 | ✅ MATCH in submit |
| P11 conf-modulated delta smoothing | Step 4 (`adaptive_delta_smooth`) | base_α = 0.15 default / **0.20 submit**; formula is literally `α·(1−conf)` | α·(1−max_p_t) base 0.20 | ✅ EXACT FORMULA |
| Per-class thresholds | Step 5 (hardcoded V18) | grid-searched on V18 OOF | F1-optimal grid | ✅ partial — thresholds yes; isotonic no |
| `TEXTURE_TAXA` | Cell 9 def | {Amphibia, Insecta} (Aves/Mammalia/Reptilia fall through to T=1.10) | Aves + texture split | ✅ MATCH |
| tta_shifts (3-shift) | submit CFG | [0, 1, −1] | [0, ±1, ±2] | ≠ — 5-shift timed out historically (#31B) |

**Consequence:** the §14.14.2 and §14.14.3-amendment shortlists
compared our stack against an **outdated mental model**. P1, P3, P11,
P13 were already in production at the audits' recommended values.
Proposing them as "cheap next probes" was a false positive. The
working shortlist after this audit shrinks to:

**Genuinely-novel inference-only levers remaining:**
- **P12 per-class isotonic calibration** — sklearn IsotonicRegression
  per class on OOF scores. Different operator than the hardcoded
  thresholds (which sharpen, not calibrate). Fits on OOF, applied
  as a score-remap before threshold sharpening. Requires bundling
  234 fitted isotonic models into a Kaggle dataset and splicing
  into Cell 18 between Steps 4 and 5. Expected +0.002–0.005.
- **P14 genus proxy for unmapped Perch species** — cheap *if* any
  of the 234 species are absent from Perch's label set. Taxonomy
  audit required first: if all 234 mapped, P14 is a no-op.
- **P2 extension 3-shift → 5-shift** — carries historical #31B
  wall-time risk (three timeouts on the 90-min CPU budget). Not
  recommended until we have proof the current pipeline has
  headroom.

**Training-side levers unchanged:**
- P5 honest PyTorch SWA (fold-0 probe, 2h).
- P7 SNR background mixup (needs noise corpus, reuse
  train_folds_pre_anuraset filter, 1.5h).
- 5-fold P8 RMS (10h, ensemble-level measurement of today's partial
  win).
- P6 family aux loss (low-weight regularizer).
- P10 frame-level Perch distill (day-scale, val-leakage risk).

**M3 single-arch iterative NS** remains the only known day-scale
lever with leader-gap-sized potential.

### Updated recommended sequencing (2026-04-22 post-audit)

1. **P12 isotonic calibration** — only genuinely-cheap inference-only
   lever remaining. Biggest information-to-effort ratio among the
   post-proc tier.
2. **P14 genus proxy** — quick taxonomy audit; if it unlocks, cheap
   patch.
3. **5-fold P8** — if P12/P14 both neutral or small, heavy training
   commit to measure P8's ensemble Δ properly.
4. **M3 iterative NS** — day-scale commit after P12/P14/5-fold P8 all
   decided.

### Lesson for future audits

**ALWAYS audit the production notebook's actual post-proc pipeline
before proposing post-proc levers from external writeups.** The
competitor recipes we audited all targeted the same post-proc cell
we already have. We shipped this stack (V17/V18) months ago and
forgot; the audits rediscovered it as if it were new.

The §14.14.2 and §14.14.3-amendment shortlists are marked correct
as **technique catalogs** but wrong as **prioritized next moves** —
they were priced against a pre-V17 stack.

---

## ⏸️ PICK UP HERE (2026-04-22 21:39 local — P8 5-fold RUNNING overnight — SUPERSEDED)

**Status.** Post-proc audit §14.14.7 showed P1/P3/P11/P13 already
live in production. Pivoted from inference-only probes to the one
remaining single-variable training-side question: **does P8's fold-0
+0.008 translate to ensemble-level LB lift?**

**Run info:**
- **Python PID:** `586904` (bash wrapper PID 586894 exits immediately).
- **Log:** `log/p8_5fold_20260422_213904.log`
- **Command:** `python -u src/train_a1.py --folds 0,1,2,3,4 --epochs 25
  --seed 42 --loss hybrid --mixstyle-p 0.5 --rms-select`
- **Expected finish:** ~07:40 local 2026-04-23
  (5 folds × 2h05m ≈ 10h 25m + ~5m val build once).
- **Disk:** 883 GB free — ample.

**Per-fold output files (don't exist yet, will be written fresh):**
- `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid.pt`
- ...fold1_seed42_hybrid.pt, ...fold2..., ...fold3..., ...fold4...

**Preserved fold-0 ckpts that will NOT be overwritten:**
- `..._fold0_seed42_hybrid_M1_mel128.pt` (M1 evidence)
- `..._fold0_seed42_hybrid_M2_multilayer_gem.pt` (M2 evidence)
- `..._fold0_seed42_hybrid_P8_25ep.pt` (P8 25-ep winner 0.7298)
- `..._fold0_seed42_hybrid_P8_40ep.pt` (P8 40-ep control 0.7255)
- `..._fold0_seed42_hybrid_soup.pt` (T2.3 soup)

The fresh fold-0 ckpt at `..._fold0_seed42_hybrid.pt` will be the new
P8 25-ep fold-0 BEST, expected val ~0.7298 ± stochastic.

**Morning checklist (2026-04-23 ~08:00 local):**
1. `tail -n 60 log/p8_5fold_*.log` — confirm all 5 folds finished
   cleanly with summary lines, no traceback.
2. Extract per-fold BEST val_roc_auc. Expected range 0.71–0.74.
3. Compute mean val across folds. Compare to production 5-fold
   baseline (v56's fold val mean was roughly 0.74).
4. If P8 5-fold mean ≥ 0.72 and no fold collapsed below 0.70:
   - Preserve individual fold ckpts by renaming with `_P8_5fold_*`
     suffix (per `feedback_backup_ckpts_before_overwrite`).
   - JIT-export all 5 folds via `src/export_a1_jit.py`.
   - Back up `kaggle_datasets/a1-effb0-ckpts/` before pushing.
   - Upload as new Kaggle dataset version.
   - Update production notebook's ckpt path if needed.
   - LB probe.
5. If one or more folds collapsed (< 0.70): individual fold
   diagnosis, likely rerun just the collapsed fold with
   different seed.

**If the overnight run crashed (process exited before all 5 folds
complete):** read log tail for traceback. Most likely OOM (unlikely
— fold 0 already proved the recipe works) or disk space. Do NOT
reflexively re-launch — diagnose first.

**Cleanup before next session:**
- `--rms-select` flag + `_pick_rms_window` helper retained in
  `src/dataset_a1.py` (default off; used only via CLI flag).
- `--multi-layer-gem` flag + model path retained in `src/model_a1.py`
  (M2 code, gated off).
- M1 config revert persisted in git (`src/config.py` at v56 mel).

**If P8 5-fold LB probe lifts ≥ +0.002:** ship it, then revisit
whether to stack P7 (SNR mixup) + P5 (SWA) + 5-fold retrains on top.

**If P8 5-fold LB probe neutral/negative:** the +0.008 fold-0 signal
was overfit to the single-fold measurement. Revert to v56 production.
Next training-side commit becomes P5 SWA fold-0 probe or M3 iter NS.
Next inference-only = P12 isotonic calibration (needs OOF + 234
sklearn models + notebook splice, ~hours).

---

## 14.14.8 P8 5-fold finished + fold-0 reproducibility gap = cuDNN noise (2026-04-23)

**Overnight 5-fold P8 RMS run finished clean at 08:01 local, 10h 22m
total.** Per-fold BEST val_roc_auc:

| Fold | val_roc_auc |
|------|-------------|
| 0 | 0.6983 |
| 1 | 0.7074 |
| 2 | 0.7310 |
| 3 | 0.7207 |
| 4 | 0.7519 |
| **mean (5)** | **0.7219** |
| **mean (0/1/2/4, export set)** | **0.7221** |

**Gates:**
- Mean 0.7219 just barely clears the 0.72 "no collapse" threshold.
- Fold 0 at 0.6983 **fails the <0.70 hard-kill floor by 0.0017**.
- Mean 0.7219 is **−0.02 below the v56 5-fold baseline (~0.74)**.

### Fold-0 reproducibility investigation (the core finding)

P8 25-ep single-fold probe on 2026-04-22 peaked fold-0 val 0.7298
(§14.14.6). The 5-fold rerun peaked fold-0 at 0.6983 **— same seed
(42), same recipe (--loss hybrid --mixstyle-p 0.5 --rms-select
--epochs 25), same data.** Δ = −0.0315.

Investigated source of the gap:
- **Code unchanged.** All src files last modified Apr 22 14:42–15:50,
  before both runs. `src/train_a1.py`, `src/dataset_a1.py`,
  `src/model_a1.py` bit-identical between runs.
- **Data unchanged.** `data/processed/train_folds.csv` mtime
  Apr 20 22:15, predates both runs.
- **Val pipeline stable.** Ran `src/_p8_eval_pair.py` paired eval:
  loaded preserved P8_25ep fold-0 ckpt → **re-scored 0.7298 exactly**
  against today's val build. Loaded fresh 5-fold fold-0 ckpt →
  **re-scored 0.6983 exactly**. Validation is bit-deterministic.
- **Set_seed standard.** `random.seed + np.random.seed +
  torch.manual_seed + torch.cuda.manual_seed_all`. All reset at
  fold start.

**Remaining source of drift = cuDNN/DataLoader nondeterminism:**
`torch.backends.cudnn.benchmark = True` lets cuDNN autotune per
workload; with `num_workers=config.NUM_WORKERS` + spawn + persistent
workers, worker-prefetch timing drifts run-to-run. With seed 42 and
identical code, two separate `nohup` launches of fold 0 can still
produce trajectories that diverge by ~0.03 on this recipe.

**Consequence for probe-design policy:** single-fold Δ < 0.03 on
this pipeline is inside the noise floor and **cannot be trusted as
lever evidence.** P8's claimed +0.008 fold-0 lever-delta was noise.
Future cheap probes must either (a) run 5-fold ensemble from the
start, or (b) run ≥ 2 fold-0 repeats with different seeds and
require min-across-seeds Δ > 0.03 before claiming a signal.

### LB probe decision (user override)

My recommendation was to mark P8 killed on val evidence alone and
skip the LB probe — −0.02 mean-val gap vs v56 baseline makes LB
lift unlikely. User overrode: **"push to Kaggle, I want to see the
LB score."** Executed the push:

1. v56 production JITs backed up → `kaggle_datasets/_backups/a1_fold{0,1,2,4}_v56_20260423.pt`
2. Raw P8 5-fold ckpts preserved → `models/a1/a1_..._fold{F}_seed42_hybrid_P8_5fold.pt`
3. JIT-exported folds 0/1/2/4 (fold 3 dropped per standard) via
   `src/export_a1_jit.py` → `kaggle_datasets/a1-effb0-ckpts/a1_fold{F}.pt`
4. `kaggle datasets version` — new version status: **ready**
5. `kaggle kernels push` for `stevewatson999/birdclef-2026-protossm`
   → **kernel version 57**, status: **RUNNING** at 16:48 local

### Decision rules on the LB result

- **LB ≥ 0.933 (≥ +0.002):** ship it; revisit whether to stack P7
  SNR mixup + P5 SWA + more 5-fold retrains.
- **LB 0.930–0.932 (neutral ±0.001):** mark P8 killed at
  ensemble-level. Revert ckpts from `kaggle_datasets/_backups/`,
  push revert dataset version.
- **LB ≤ 0.929 (−0.002 or worse):** confirms val read; revert
  immediately, memorialize noise-floor lesson, start P12 isotonic.

### Provenance + cleanup notes

- `src/_p8_eval_pair.py` is a throwaway diagnostic; safe to `rm`
  after closing out this section.
- Kernel-metadata for production notebook untouched; dataset pinless
  reference picks up latest automatically.

---

## ⏸️ PICK UP HERE — previous (2026-04-23 16:48 local — P8 5-fold LB probe RUNNING — SUPERSEDED by §14.14.9)

---

## 14.14.9 P8 5-fold KILLED — LB 0.928, reverted (2026-04-23 22:15 local)

**Result.** Kernel v57 (P8 5-fold, folds 0/1/2/4) completed clean.
User submitted to competition: **LB 0.928**, vs v56 baseline **0.931**
→ **Δ = −0.003**. Falls in §14.14.8 "≤ 0.929" decision branch:
revert per decision rules.

**CORRECTION 2026-04-23 23:10 local:** the "v56 5-fold mean ~0.74"
claim originally written here was wrong — that was the fold-0 gate
metric (0.7414), not the 4-fold mean. Tonight's P12 OOF emit pass
measured the actual v56 4-fold (folds 0/1/2/4) values directly:

| Fold | val_roc_auc |
|------|-------------|
| 0 | 0.7415 |
| 1 | 0.7227 |
| 2 | 0.6975 |
| 4 | 0.7248 |
| **mean (4)** | **0.7216** |
| **macro AUC of mean-ensemble probs** | **0.7290** |

That makes v56's per-fold mean **essentially identical** to P8's
0.7219 mean. The LB Δ −0.003 between v56 (0.931) and P8 (0.928) is
**fold-selection / ensembling noise on a flat val landscape**, not a
0.02 val gap as originally framed. Doesn't change the kill verdict —
the noise-floor rule still applies — but the storyline is cleaner.

**Revert executed.**
1. v56 JITs restored from `kaggle_datasets/_backups/a1_fold{0,1,2,4}_v56_20260423.pt`
   over `kaggle_datasets/a1-effb0-ckpts/a1_fold{F}.pt` (md5 of fold0
   restored matched backup: `fc0f32ad1b7fcb8b350393410b52adb4`).
2. `kaggle datasets version -p kaggle_datasets/a1-effb0-ckpts -m
   "revert P8 5-fold (LB 0.928) -> v56 baseline (LB 0.931)"` — upload
   succeeded for all four folds; new dataset version ready.
3. Production notebook unchanged (dataset reference is pinless,
   resolves to latest = restored v56).

**No new submission burned.** Kernel v57 is the only LB cost; revert
is just the dataset upload. Next push of the notebook will pick up
v56 ckpts automatically.

**Why P8 failed at ensemble level despite single-fold partial win:**
The +0.008 fold-0 lever-delta from §14.14.6 was inside the cuDNN
noise floor (≈0.03 spread) documented in §14.14.8. Across 5 folds
the recipe lost 0.02 in mean val. The "partial win" was sampling
noise, not signal. **Noise-floor rule (single-fold Δ < 0.03 ⇒
suspect noise) memorialized as a permanent memory before this run
and is now empirically validated by the LB drop.**

**Cleanup deferred (kept for forensics):**
- `src/_p8_eval_pair.py` — paired eval diagnostic; keep for now
  in case any P8 follow-up needs it. Safe to `rm` later.
- Raw P8 5-fold ckpts preserved under
  `models/a1/a1_..._fold{F}_seed42_hybrid_P8_5fold.pt` — keep at
  least until the next training-side lever ships, in case any
  comparative analysis needs them.

---

## ⏸️ PICK UP HERE — previous (2026-04-23 22:15 local — queue P12 isotonic — SUPERSEDED by §14.14.10)

---

## 14.14.10 P12 OOF emit pass DONE — calibratable-class slice tiny (2026-04-23 evening)

**Done tonight:**
- Wrote `src/p12_emit_oof.py`. Loads each of the 4 v56 JIT ckpts
  (`kaggle_datasets/a1-effb0-ckpts/a1_fold{0,1,2,4}.pt`), scores all
  1478 labeled `train_soundscapes_labels.csv` windows, and saves
  `data/v56_soundscape_oof.npz` (3.3 MB).
- NPZ keys: `probs_per_fold (4,1478,234)`, `probs_mean (1478,234)`,
  `y_true (1478,234)`, `fold_ids (4,)`, `filenames`, `start_sec`.
- Sanity check: fold-0 standalone macro AUC = **0.7415**, matching
  the 0.7414 hybrid gate exactly. Pipeline reproduces.

**GB10 (compute capability 12.1) NVRTC trap fixed in-script.**
First attempt died with "nvrtc: invalid value for --gpu-architecture
(-arch)" because PyTorch's NVRTC max is 12.0 and the GB10 card runs
12.1. The JIT-traced module compiles fused kernels at first call
and they hit the NVRTC version cap. Fix is six lines disabling all
JIT fusion + dropping autocast. Memorialized as
`feedback_gb10_nvrtc_jit.md`. **Any future inference pass on JIT
files needs the same six lines.**

### The calibratable-class wall

P12's headroom is structurally bounded by val class density:

| Min positives in val | Calibratable classes | % |
|----------------------|----------------------|---|
| ≥ 1                  | 75                   | 32% |
| ≥ 5                  | 64                   | 27% |
| ≥ 10                 | 59                   | 25% |

**170 of 234 classes have < 5 positives in val** and must pass
through identity (any K=5 fallback). At K=10, 175 classes pass
through. The Maryna two-pass SSM recipe priced P12 as a clean
inference lever, but on our val universe (1478 GT-labeled windows)
the calibration-fit set is too sparse for two-thirds of the class
space. Effective lift is bounded by the 25–27% calibratable slice.

### Decision posture for P12 (entering tomorrow)

**Argument FOR a single LB probe:**
- Cost is ~30 min: fit 64 isotonics, pickle, push small dataset,
  splice 5 lines into Cell 18, push notebook, single submission.
- Even +0.003 on the 27% slice would be a meaningful first inference-
  only win after T1.1/T1.2/T1.3 all killed.
- Single LB submission is conclusive (not subject to noise-floor
  rule — it's a closed-form transform).

**Argument AGAINST:**
- 27% calibratable + monotonic-only transform → realistic LB envelope
  is probably ±0.003, very likely inside per-submission LB noise.
- Submission burned regardless of result.
- Fit risk: 1478 windows / 64 calibratable classes ≈ 23 rows per
  class avg, but skewed (one class has 666 positives, another 5);
  isotonic on the small-positive end will overfit val.
- Doesn't address the macro-AUC contribution from the 159
  zero-positive classes — those drive ~68% of the metric and we
  can't touch them.

**Lean (mine, not yet decided):** **probe it anyway**. The cost is
trivial relative to a training-side lever, and a confirmed +0.003
opens the door to a P12.5 variant (broader class buckets, e.g. fit
one isotonic per *family* instead of per class). A confirmed neutral
or negative kills the per-class direction but doesn't preclude
family-level calibration as a follow-up.

### Side findings logged in §14.14.9 correction

The original "v56 5-fold mean ~0.74" claim was wrong — it was
fold-0 gate value extrapolated. Real v56 4-fold mean is **0.7216**,
essentially identical to P8 5-fold's 0.7219. So the P8 LB Δ−0.003
came from fold-selection / ensembling noise on a flat val landscape,
not from a 0.02 val gap. Doesn't change the kill verdict.

---

## 14.14.11 P12 isotonic KILLED — LB 0.868, distribution-mismatch flaw (2026-04-24)

**Result.** Kernel v58 LB = **0.868**, vs. v56 baseline 0.931. Δ = **−0.063**.
Catastrophic. Far below the −0.003 kill threshold from §14.14.10.
Reverted locally: Cell 18 restored from `birdclef2026-protossm-postproc.ipynb.bak_pre_p12`;
`stevewatson999/birdclef-2026-p12-isotonic-calib` removed from
`kernel-metadata.json:dataset_sources`. Kaggle still at v58 until the
revert push goes out.

**Root cause — fit-vs-apply distribution mismatch.** `src/p12_emit_oof.py`
scored only the 4-fold A1 JIT ensemble and fit isotonic on
`sigmoid(A1_logits)`. In production Cell 18 the `probs` variable at the
Step 4.5 splice point has already passed through:

1. Score fusion (Cell 39) with Perch + ProtoSSM + Track-B1.
2. Per-taxon T scaling (Step 1).
3. File-level confidence scaling (Step 2).
4. Rank-aware power (Step 3).
5. Delta shift smoothing (Step 4).

The calibrators were fit to map "A1 raw sigmoid" → "calibrated P(y=1)"
but received a fused, rescaled, rank-powered, smoothed signal with a
completely different distribution. Per-class monotone calibrators
applied to the wrong distribution produce a destructive mangle, not a
refinement.

**The +0.069 "sanity" AUC was pure overfitting.** Isotonic regression on
the fit set can boost AUC via pool-adjacent-violators: adjacent (pos,
neg) pairs where `pos_prob < neg_prob` collapse to ties and contribute
0.5 instead of 0 to the AUC sum. On the fit set this is memorization,
not generalization. Should have noticed this was too good to be true.

**P12.5 family-bucket is dead by the same argument.** Any calibrator fit
on raw-model OOF is invalid when applied mid-pipeline, regardless of
whether it's per-class or per-family. An honest calibration approach
would require emitting OOF of the ENTIRE Cell 18 pipeline (fusion +
post-proc through Step 4) and fitting calibrators on that — a much
heavier infra change. Not cheap; deprioritized indefinitely.

**Lessons for future calibration levers.**
1. **Fit distribution must equal apply distribution.** If the target
   splice point is after N pipeline stages, the OOF must be emitted
   after the same N stages.
2. **Fit-set AUC is a correctness sanity check, not a generalization
   signal.** Monotone calibration cannot improve held-out AUC; any
   apparent boost is PAV memorization. In future probes, hold out a
   split when fitting calibrators if LB-predictive signal is wanted.
3. **Submission burned at LB 0.868 paid for the lesson.** Catalogue
   the distribution-mismatch rule so it doesn't get retried under a
   different name (P12.5, P12.6, etc.).

### Decision

- P12 per-class isotonic: **KILLED**.
- P12.5 family-bucket: **KILLED by inheritance** — same distribution-
  mismatch flaw.
- Next lever: **P5 PyTorch SWA, 5-fold from the start** per
  §14.14.10's NO-GO pathway and `feedback_single_fold_noise_floor.md`
  (never single-fold on LB).

### Cleanup checklist
- [x] Cell 18 reverted from `.bak_pre_p12`.
- [x] `kernel-metadata.json` reverted (P12 dataset removed).
- [ ] Kaggle kernel push to restore v56 baseline on LB — GATED on user.
- [ ] Kaggle dataset `stevewatson999/birdclef-2026-p12-isotonic-calib`
      can be deleted or left private-and-ignored; no downstream
      consumers.
- [x] `data/p12_isotonic_calibrators.pkl` + `kaggle_datasets/p12-isotonic-calib/`
      kept on disk as evidence; no rerun planned.

---

## 14.14.12 P5 SWA KILLED at fold 1 (2026-04-24 22:15 local)

**Kill result.** PID 651937 terminated 2026-04-24 22:16 local after
fold 1 SWA-averaged val = **0.6722**, triggering the committed
kill criterion (fold 1 < 0.72).

| Fold | SWA-averaged val | Δ vs 0.7414 gate |
| :--- | :--- | :--- |
| 0 | 0.7196 | −0.022 |
| 1 | 0.6722 | −0.069 |
| **Mean (2 folds)** | **0.6959** | **−0.046** |

Two-fold mean sits 0.046 below baseline — catastrophically below
noise floor. Extrapolating to 5-fold mean ≥ 0.7414 would require
folds 2/3/4 to average 0.773, which is higher than the single-fold
noise-floor best we've ever seen. Statistically hopeless, kill
executed.

**Diagnosis: the T2.3 soup pathology inside SWA wrapper.**
- Raw-model val during SWA window (fold 0) peaked at 0.7333 (ep 18)
  and ended at 0.7098 (ep 25), oscillating 0.68–0.73 throughout.
- SWA averages 9 epochs spanning a cosine-warm-restart-noisy window.
- Averaging across that range produces a mean model worse than the
  best single raw snapshot AND worse than the baseline end-of-run
  BEST save the non-SWA trainer would have chosen.
- T2.3 ckpt soup killed the same way (see `project_t23_soup_killed.md`).
  The SWA wrapper adds update_bn rigor but doesn't fix the fundamental
  "averaging over noisy late epochs" problem.

**Lesson logged**: the cosine-warm-restart schedule is load-bearing
for A1's final-model selection. Any averaging-family lever (SWA, EMA,
naive soup) applied to this training shape WILL regress because the
weights being averaged are peaks and troughs of the cosine cycle,
not samples near a single optimum.

**Keep-or-delete artifacts:**
- `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold{0,1}_seed42_hybrid_swa.pt`
  — the two completed SWA ckpts. Keep as evidence.
- Log: `log/train_a1_swa_5fold_20260424_185726.log`.

**Implementation kept in `src/train_a1.py`** (don't revert) — the
`--swa` flag harness works correctly; the training recipe is what
failed, not the code. If someone later wants to try SWA on a
*different* training recipe (e.g., constant LR, no warm restarts),
the scaffolding is ready.

**Next lever.** Per the committed pivot rule: move to §14.16 B2
build, modulated by §14.14.15 B1 w=0.00 result if that lands first.

---

## 14.14.12-pre P5 SWA 5-fold IN FLIGHT + kill criterion — SUPERSEDED above (2026-04-24 evening)

**Run.** Launched 2026-04-24 18:57 local, PID 651937.
`python -u src/train_a1.py --folds 0,1,2,3,4 --epochs 25 --swa --loss hybrid`
with `swa_start_frac=0.65` (activates epoch 17), `swa_lr=4e-4`, SWALR
with 2-ep linear anneal. Projected finish ~03:30–04:00 Saturday morning
(fold pacing ~1h 45m).

**Implementation** lives in `src/train_a1.py` (~60 LOC added):
- `--swa` / `--swa-start-frac` / `--swa-lr` CLI flags
- Activates `torch.optim.swa_utils.AveragedModel` + `SWALR` at
  `ceil(swa_start_frac × epochs)`; after that epoch, SWALR replaces
  the cosine-warm-restart scheduler and `swa_model.update_parameters`
  runs each epoch
- After final epoch: one full `update_bn` pass over the train loader,
  then validate SWA-averaged model; save as `*_swa.pt`
- Per-epoch `★ BEST` saves are DISABLED in SWA mode — the only
  authoritative artifact is the end-of-fold `[SWA] averaged model
  val_roc_auc` line

**Fold 0 result.** SWA-averaged val_roc_auc = **0.7196**.
Baseline hybrid gate = 0.7414. Δ = **−0.022**.
Inside 0.03 single-fold noise floor (`feedback_single_fold_noise_floor.md`);
not a hard kill on one fold alone.

**Two worry signals stacked:**
1. SWA-averaged (0.7196) underperforms the best-raw epoch during the
   SWA window (ep 18 raw = 0.7333, Δ −0.014). Same T2.3 soup
   pathology — averaging over a noisy late-window hurts, not helps.
2. Raw-model val during SWA window oscillated 0.68–0.73 while
   `train_loss` kept falling 0.0169 → 0.0153 — late-epoch overfitting
   baked into the SWA average.

**Kill criterion (committed 2026-04-24 ~20:45 local).** If fold 1
SWA-averaged val_roc_auc < 0.72, terminate PID 651937 and pivot to
§14.16 Track B2 build. If fold 1 ≥ 0.72, let all 5 folds complete and
decide on 5-fold mean vs 0.7414 gate.

Fold 1 ETA ~22:22 local (2026-04-24).

---

## 14.14.13 B1 weight probe w=0.20 RESULT: LB 0.922 (2026-04-24 evening)

**Kernel v60 → LB 0.922** vs. baseline 0.931 @ w=0.10. Δ = **−0.009**.
Hits the "≤ 0.928: confirm B1 net-negative" bucket.

**The data point is actually gold, not bad news.** With two points now
on B1's LB-lift curve:

| B1 weight | LB | Δ vs baseline |
| :--- | :--- | :--- |
| 0.10 (baseline) | 0.931 | 0 |
| 0.20 (probe) | 0.922 | −0.009 |

**Slope ≈ −0.09 LB per unit weight** in the [0.10, 0.20] range.
If even approximately linear, extrapolating back:

| Hypothetical w | Extrapolated LB |
| :--- | :--- |
| 0.00 | ~0.940 (+0.009) |
| 0.05 | ~0.935 (+0.004) |

Linearity is a big assumption — three hypotheses to distinguish with
one more probe:
- **(A) B1 monotonically hurts** → w=0.00 ≈ 0.940. Huge win.
- **(B) Non-monotonic, peak ~w=0.05** → w=0.00 < 0.931.
- **(C) Quadratic peak near w=0.10** → w=0.00 ≈ 0.922.

Probe #2 committed: see §14.14.15.

---

## 14.14.15 B1 weight probe w=0.00 RESULT: LB 0.928 — sweep closed (2026-04-24 23:30 local)

**Kernel v61 → LB 0.928.** Δ = −0.003 vs baseline.

**Three-point B1 weight curve:**

| B1 weight | LB | Δ |
| :--- | :--- | :--- |
| 0.00 | 0.928 | −0.003 |
| 0.10 | 0.931 | 0 (baseline) |
| 0.20 | 0.922 | −0.009 |

**Hypothesis C confirmed** — quadratic with local maximum at w=0.10.
The §14.14.13 linear slope extrapolation was wrong; the slope
reverses sign below w=0.10.

**Implications:**
- B1 at w=0.10 is near-optimal. No B1 weight-tuning lift available.
- B1 contributes ~+0.003 LB at its current weight. Real, small, fixed.
- **B1 weight sweep is closed.** Option 2 (B1 tune-up) exhausted —
  the sweep was the only honest measurement, and it says B1 is
  sitting where it should.
- **Track B via B1 is done.** Both Perch-consumers (ProtoSSM + B1)
  are at their ceiling. Any further Track B lift must come from
  §14.16 B2 (ConvNeXt on mel, independent of Perch embeddings).

**Revert required.** Current live kernel (v61) has w=0.00 baked in
→ produces 0.928 if re-run. Leaderboard is still 0.931 from v56 (best
submission preserved), but the live kernel must be reverted to
w=0.10 before any future change to avoid compounding on a degraded
baseline.

**Kernel v62 revert action:** Edit Cell 3 to set
`CFG["b1_frozen_weight_submit"] = 0.10` (or delete the override line
entirely, letting Cell 24b's setdefault=0.10 take effect). Push. Skip
submission — Kaggle's best-submission rule preserves our 0.931.

---

## 14.14.15-pre B1 weight probe w=0.00 STAGED — SUPERSEDED above (2026-04-24 evening)

**Kernel v61 pushed.** Cell 3 override changed from
`CFG["b1_frozen_weight_submit"] = 0.20` to
`CFG["b1_frozen_weight_submit"] = 0.00`. At w=0.00, Cell 40
(`if B1_WEIGHT_FROZEN <= 0.0: ... skipping B1 test fusion`) short-
circuits B1 entirely — no B1 contribution to the submission.

**Backup**: `birdclef2026-protossm-postproc.ipynb.bak_pre_b1_w000`.

**Decision rules** (using §14.14.13 extrapolation):
- **LB ≥ 0.938** (≥ +0.007 from baseline): hypothesis A confirmed,
  B1 monotonically hurts. New production baseline w=0.00 with
  B1 retired. Update setdefault in Cell 24b accordingly.
- **LB 0.933–0.937** (+0.002 to +0.006): hypothesis A partially
  confirmed. Keep w=0.00 as new baseline; try narrow sweep
  w=0.02 or w=0.05 to test for a low-weight peak.
- **LB 0.930–0.932** (±neutral): non-monotonic, peak near current
  w=0.10. Revert to 0.10, stop B1 weight sweep.
- **LB ≤ 0.929**: B1 weight curve has a peak at w=0.10 (hypothesis
  C). Revert, close out B1 tune-up track.

**Submission slated after user manual submit from Kaggle UI.**

**If win: implication for Track B.** A confirmed +0.007 from dropping
B1 fundamentally reframes Track B. It means:
- B1 PerceiverIO is not just underpowered, it's net-negative at any
  non-trivial weight.
- The "second Perch consumer" architecture family is suspect — both
  ProtoSSM and B1 consume the same Perch features, and stacking two
  Perch consumers adds more redundancy than diversity.
- §14.16 B2 (ConvNeXt on mel, fully independent of Perch) becomes
  the natural next step if any Track B lift still seems reachable.

Submission slated after user manual submit from Kaggle UI.

---

## 14.14.14 B1 standalone status — gate analysis (2026-04-24 evening)

**Finding 1: The gate is broken by construction.** Cell 31b's OOF
lift sweep runs on 59 fully-labeled soundscape files split via
5-fold GroupKFold. With ~12 rows per fold that's 12×234 = 2808
cells per fold, with the vast majority of classes having zero
positives in any given fold. The sweep's best_w is therefore chosen
from a metric whose variance dwarfs the true signal.

**Finding 2: B1 standalone OOF AUC has been reported at 0.4028** (plan
line 815, 2026-04-11 substrate). This is **worse than random**, which
sanity-checks the "OOF uninformative" claim — it doesn't mean B1 is
broken, it means OOF can't measure B1 quality on 708 rows.

**Finding 3: `proto_b1_corr` gate (< 0.97) also uninformative** on
the same tiny substrate.

**Implications for Option 2 (B1 tune-up, §14.16-adjacent):**
- Extending B1 training (more seeds, longer schedule) will not
  improve the diagnostic — we can't see it locally regardless.
- Only LB probes (like §14.14.13) can tell us if B1 adds lift.
- A sweep of `b1_frozen_weight_submit ∈ {0.00, 0.10, 0.20, 0.30}`
  via 4 LB probes would map the true lift curve. Expensive (4
  submissions) but the only honest way.

**Revised Option 2 plan (downsized):**
- Skip the "why is B1 stuck" investigation (unanswerable locally).
- After §14.14.13 w=0.20 result lands, if positive, probe w=0.25.
  If neutral/negative, probe w=0.05 or w=0.00 to confirm.
- Budget: 2–3 LB submissions total, spread across the week so we
  don't burn our daily cap.
- Do NOT retrain B1 locally — the existing
  `models/b1_pretrained/*.pt` ckpts were trained exhaustively
  (200 ep, 3 seeds, patience 40) and more training unlikely to
  help a consumer that already saw 200 epochs of supervised signal.

**Noisy Classmates contingency.** If B1 turns out to be net-zero
even at its LB-optimal weight (§14.14.14 sweep), B1 is architecturally
the wrong consumer for this Perch-features substrate, and the
Track B path rolls to §14.16 B2 (ConvNeXt on mel, fully independent
of Perch embeddings).

---

## 14.15 NotebookLM cross-source technique sweep (2026-04-24)

Fed `new_plan.md` + the four reference-memory snapshots
(`reference_bc2026_public_notebooks.md`,
`reference_competitor_master_report_20260318.md`,
`reference_maryna_twopass_ssm.md`,
`reference_noisy_classmates_flagged.md`) to NotebookLM and asked it
to find techniques mentioned in the reference sources that have **no
corresponding entry anywhere in new_plan.md** — not killed, not
shipped, not queued, simply absent.

**Methodology caveat.** Round-1 query ("techniques in ≥2 sources")
returned only items already on our queue (P5 / M3 / Track B / T2.5)
— zero novelty signal. Round-2 query (strict "absent from plan"
filter, single-source allowed) returned the list below. All 10 items
trace to Source 2 (competitor master report 2026-03-18); the ≥2-source
robustness signal does **not** hold for any of them. Treat each as a
single-competitor recommendation, not a convergent practice.

### Absent-from-plan technique list

**Training-side**
- **Std-normalized waveform (std=1) before STFT.** Our preprocessing
  normalizes post-mel (PCEN, local RMS); global waveform std-norm
  is absent.
- **CoarseDropout** (contiguous spectrogram block dropout). We use
  standard Dropout, MixStyle, SpecAugment — not CoarseDropout.
- **Weighted-distributed secondary labels.** We use secondary-label
  *masking*; distributing label mass across secondaries (the named
  1st-place trick) is a different choice.

**Inference-side**
- **Mel `power=2`** (power-spectrogram) vs our implicit power=1
  (energy). Never tuned as a lever.
- **`fmin=40, fmax=15000`** band-limiting. Plan never names fmin/fmax
  as tunable hyperparameters.

**Data-side**
- **Upsample every minority class to min 10 instances.** We use
  `--soundscape-mult` and species clip caps, but no "floor of N"
  rebalance rule.
- **Random 5-sec crop from first 6s or last 6s** only (not
  anywhere). Our crop logic picks anywhere in a clip, or uses RMS
  selection.

**Architecture-side**
- **5-dropout stack before FC.** We use a single Dropout(0.3) on B1;
  no multi-dropout stacking.
- **Mel input size 224² or 288².** T2.4 lists 128²/256² only.
- **Learnable GeM pooling** (backprop the `p` exponent). We use GeM
  heads (P7 GEMFrequencyPool, M2 multi-layer GeM) but always as
  *fixed* pooling — never made `p` a learnable parameter.

### Priority call (2026-04-24)

Given 0.931 baseline + limited LB submissions + the noise-floor rule
(`feedback_single_fold_noise_floor.md`), my triage:

**Worth probing:**
1. **Learnable GeM** — single-line change in `model_a1.py`
   (`p = nn.Parameter(torch.tensor(3.0))`), no data-pipeline
   invalidation, cheap to retrain, genuinely absent from our GeM
   work.
2. **Weighted-distributed secondary labels** — named 1st-place trick;
   real fork from our masking approach. Moderate effort (loss-prep
   change in `dataset_a1.py` + `train_a1.py`). Hits the tail-class
   macro-AUC which drives the metric.
3. **Min-10 minority upsampling** — specifically targets the 159
   zero-positive classes that drive ~68% of macro AUC (per
   §14.14.10 P12 OOF analysis). Pure data-side, no ckpt
   invalidation.

**Explicitly deprioritize:**
- **Std-normalized waveform** — invalidates every trained mel
  feature; same risk class as M1 which was killed.
- **`power=2` / `fmin,fmax`** — same invalidation risk; requires
  full retrain before knowing if it helps.
- **Mel 224²/288²** — M1 + M2 killed the "touch the mel grid"
  direction; expensive and twice-burned.
- **CoarseDropout / 5-dropout stack / first-or-last crop** —
  single-competitor picks with no obvious mechanism story; low
  expected Δ.

### Deferred, not queued

These three "worth probing" items are **not** moving ahead of P12
(isotonic, tomorrow) or P5 (SWA 5-fold, next training-side lever).
They live here as a backlog entry to revisit once the current
P12 → P5 → M3 spine resolves. If P5 fails and M3 is blocked on
Track B, learnable GeM becomes the first retry candidate.

### Open methodology follow-up

The NotebookLM round-1 failure (all returned items already in plan)
suggests our 2-KB memory summaries are too filtered — they contain
my extracted levers, not the raw competitor notebooks. If we ever
want sharper synthesis we need to feed the raw competitor notebook
HTML/PDF, not the summaries. Low priority until the backlog
revisit.

---

## 14.16 Track B2 spec — ConvNeXt-tiny SED, pure architectural diversity (2026-04-24)

**Motivating context.** 15+ consecutive overnight probes have failed to
beat 0.931 (§14.14.11 P12 catastrophe being the latest; §14.14.12 P5
SWA fold-0 at 0.7196 trending toward a kill). The cheap inference-side
and single-lever training-side search spaces are exhausted. Our pipeline
is saturated at its current architecture/training shape.

The plan repeatedly calls "Track B — second independent Perch consumer"
the highest expected remaining lift, but B1 PerceiverIO has been live
at `CFG["b1_frozen_weight_submit"] = 0.10` and weakly contributing for
weeks. The real Track B opportunity is **B2**: a net-new SED branch
with genuinely different inductive bias from both EffNet-B0 (A1) and
ProtoSSM.

This spec is committed only if P5 SWA fold-1 < 0.72 (see §14.14.12
kill criterion).

### Scope: Phase 1 only

Phase 1 is **architectural diversity only** — ConvNeXt-tiny on the
same training substrate as A1. No pseudo-labels, no new data. This
isolates the "is a different backbone worth it" question from
pseudo-label hazards (C2, D3, L1 were all pseudo-adjacent killed
levers; see `project_d3_killed.md`). If Phase 1 lands positive,
Phase 2 layers on ProtoSSM teacher pseudos (Noisy Classmates full
recipe). Phase 2 is NOT committed here.

### Architecture

**Backbone: ConvNeXt-tiny** (timm `convnext_tiny`, ~28M params).

Rationale for ConvNeXt vs alternatives:
- **vs EffNet-B0 (A1)**: 7×7 depthwise + LayerNorm + patch stem vs
  3×3 inverted residuals + BN. Maximum architectural divergence inside
  the "drop-in to the A1 pipeline" constraint.
- **vs AST (Audio Spectrogram Transformer)**: stronger divergence
  but requires different mel config (25ms hop, patch input).
  Deferred to B3 if B2 lands.
- **vs MaxViT**: hybrid CNN-transformer; less pure-diversity signal.
- **vs CRNN over log-mel**: higher integration risk; non-standard head.

**Head**: reuse `model_a1.BirdSEDModelA1`'s SED attention + classifier
conv pattern — shape-agnostic (GAP over freq + time attention over
the backbone feature map). New module `src/model_b2.py` parameterizes
the backbone choice.

### Training substrate (Phase 1)

**Identical to A1**: `train_folds.csv` focal clips, 5-fold
MultilabelStratifiedKFold, PCEN mel, MixStyle, SpecAugment, 20s
chunks, 3-channel input.

**Loss**: `hybrid` (matches A1 v56 baseline per
`project_a1_baseline_loss_is_hybrid.md`).

**Fold count**: 5 from start (noise-floor rule).

**Budget**: ConvNeXt-tiny is ~1.5–2× slower per epoch than EffNet-B0
(larger param count, larger receptive field). Expected per-epoch time:
6–8 min. 25 ep × 5 folds = **13–17h total** (one overnight + morning
run, or two overnights).

### Files + integration

New files:
- `src/model_b2.py` — ConvNeXt-tiny backbone wrapping A1's SED head
- `src/train_b2.py` — trainer mirroring `train_a1.py`; same loss, same
  aug, same val substrate; saves to `models/b2/b2_convnext_tiny_fold{F}_seed42_hybrid.pt`
- `src/export_b2_jit.py` — exports 5 JIT ckpts with `B2Wrapper` for
  Cell 37b consumption (mirrors `export_a1_jit.py` +
  `A1Wrapper`)

Kaggle integration:
- `kaggle_datasets/b2-convnext-ckpts/` → publish as
  `stevewatson999/birdclef-2026-b2-convnext-ckpts`
- Production notebook:
  - New **Cell 37b** (B2 rank fusion), positioned BEFORE existing
    Cell 37 (A1 fusion): order becomes ProtoSSM → B1 → B2 → A1 →
    postproc
  - New `CFG["b2_frozen_weight_submit"] = 0.10` initial
  - Diversity gates in OOF cell: `proto_b2_corr < 0.97`,
    `a1_b2_corr < 0.97`
  - Lift sweep identical to B1's Cell 31b pattern
- Add `stevewatson999/birdclef-2026-b2-convnext-ckpts` to
  `kernel-metadata.json:dataset_sources`

### Gates + decision rules

| Stage | Gate | Kill criterion |
| :--- | :--- | :--- |
| Smoke test (fold 0, 2 ep, 1 batch) | No error | wiring broken → fix |
| Fold-0 early (ep 5) | val_auc > 0.55 | < 0.55: architecture/head bug |
| 5-fold mean val | ≥ 0.7414 (hybrid gate) | < 0.7414: no LB probe |
| Proto/B2 correlation | < 0.97 | ≥ 0.97: B2 too similar to proto; increase divergence |
| A1/B2 correlation | < 0.97 | ≥ 0.97: B2 too similar to A1; reconsider arch |
| OOF lift sweep | `b2_weight > 0` wins | no positive weight: B2 adds no signal |
| LB probe | ≥ 0.932 (+0.001) | < 0.930: B2 fusion kills; revert |

### Realistic LB envelope

- **Best case**: +0.010 to +0.015 — if B2 genuinely diversifies and
  adds lift at moderate weight (0.15–0.25).
- **Median case**: +0.002 to +0.005 — matches B1's historical
  contribution ceiling.
- **Failure mode**: B2 correlates too highly with A1 OR underperforms
  standalone → gate drives weight to 0 → wasted 3–5 days.

### Cost + timeline

| Step | Wall time |
| :--- | :--- |
| Write `model_b2.py` + `train_b2.py` + `export_b2_jit.py` | 1 day |
| Smoke test | 30 min |
| 5-fold training (nohup) | 13–17h |
| JIT export | 1h |
| Kaggle dataset bundling + push | 30 min |
| Notebook Cell 37b authoring + CFG edits | 4h |
| Kernel version push + LB probe | 1 submission |
| **Total** | **3–5 days wall time** |

### Phase 2 hook (NOT committed in this spec)

If Phase 1 LB probe lands positive (≥ 0.932), Phase 2 overlays
ProtoSSM teacher pseudo-labels on train_soundscapes (NOT BC2025 — D3
killed that substrate). Phase 2 spec will be §14.17 and only written
when Phase 1 survives. The `data/processed/pseudo_bc25ss_probs.npz`
artifact from D3 is NOT reusable — wrong substrate.

**Open question for Phase 2**: ProtoSSM teacher ckpt
(`models/protossm_teacher/teacher.pt` in the plan) does NOT exist on
local disk per 2026-04-24 audit. Would need to be regenerated on
Kaggle, or extracted from the training notebook if still resident.

### Why commit to this despite bleak recent history

- The pattern of failed <+0.005 probes is not random — our pipeline
  is saturated at its current shape. Next lift requires architectural
  change, not tuning.
- Track B is the one plan item that has been repeatedly named
  "highest expected lift" and never honestly committed to. B1
  occupies the slot but is underpowered.
- If B2 also fails, that's a legitimate signal to accept 0.931 as
  the floor and stop submitting.

### Implementation sequence (contingent on SWA kill at fold 1)

1. [x] `src/model_b2.py` — ConvNeXt-tiny + A1 SED head
2. [x] `src/train_b2.py` — mirrors `train_a1.py`; same flags minus
       SWA (SWA can be added to B2 later, orthogonal)
3. [x] Smoke test fold 0, 2 ep, 1 batch → verify wiring
4. [x] 5-fold training via nohup overnight (2026-04-25 → 2026-04-27,
       39h 27m; log `train_b2_5fold_resume_20260425_164603.log`)
5. [x] Gate check — 5-fold mean val ≥ 0.7414?
       **PASS — mean 0.7904** (folds: 0.8177 / 0.7831 / 0.7677 / 0.8001 / 0.7834).
       Caveat: fold-0 +0.04 above next-best — wider than the 0.03 noise floor.
6. [x] `src/export_b2_jit.py` → JIT 5 ckpts → `kaggle_datasets/b2-convnext-ckpts/`
       (5 × ~120 MB traced on CPU; shape assertion passed each fold;
       smoke `log/export_b2_jit_smoke_20260427_101639.log`,
       full `log/export_b2_jit_remaining_20260427_101714.log`)
7. [x] `kaggle datasets create -p kaggle_datasets/b2-convnext-ckpts`
       Live at `stevewatson999/birdclef-2026-b2-convnext-ckpts`
       (private; 2026-04-27 ~10:26 local). First push failed —
       title length 52 > 50 cap; trimmed "SED " → 48 chars and
       updated both metadata file and `write_dataset_metadata()` so
       the script regenerates it correctly next time.
8. [x] Notebook Cell 37b authoring + CFG splice + diversity gate wiring
       (2026-04-27; cell 41 in `birdclef2026-protossm-postproc.ipynb`,
       order: Cell 39 ProtoSSM → 40 B1 → **41 B2 (new)** → 42 A1 → 43 post-proc;
       backup `*.bak_pre_b2_cell37b`; +`CFG["b2_frozen_weight_submit"] = 0.10`
       in Cell 3; +`stevewatson999/birdclef-2026-b2-convnext-ckpts` in
       `kernel-metadata.json:dataset_sources`).
       Diversity gate is **soft at submit time**: cell logs
       `proto_b2_corr` (mean per-col Pearson r vs upstream ranks); if
       ≥ 0.97 the cell skips B2 fusion to protect LB. The plan's
       `a1_b2_corr` gate is NOT wired (would require modifying Cell 42 A1)
       — `_b2_ranks` is stashed in globals so a future diagnostic cell
       can compute it post-hoc.
9. [x] `kaggle kernels push` v63 (2026-04-27 ~10:50 EDT) — user-pushed
       kernel ran fine on the 20-file preview test, but **Kaggle's
       hidden-test LB-scoring re-run TIMED OUT** on the 90-min CPU
       budget (5-fold B2 + 4-fold A1 + redundant per-file audio decode/
       mel build). CLI showed `SubmissionStatus.COMPLETE` (sample
       submission uploaded OK) — that does NOT mean scoring succeeded.
9b. [x] `kaggle kernels push` v64 (2026-04-27 ~13:05 EDT) — same fate as
       v63. B2 cut to 3-fold [0, 1, 3] (top-3 by val, mean 0.8003)
       wasn't a deep enough cut to fit the hidden test. Hidden-test
       scoring run also TIMED OUT.
       Log: `log/kaggle_kernels_push_b2_v64_*.log`.
       Backup: `*.bak_pre_b2_3fold`.
9c. [x] **Diversity gate empirically validated** (v64 preview-test log,
       2026-04-27 13:18 EDT). `proto_b2_corr = 0.2486` (mean per-col
       Pearson r) — far below the 0.97 gate threshold. B2 is genuinely
       diverse from upstream signal; fusion mean |Δ score| = 0.08182.
       The §14.16 architectural-diversity premise is substrate-confirmed;
       only wall-time blocks B2.
9d. [x] `kaggle kernels push` v65 (2026-04-27 ~15:55 EDT) — **cheap probe**
       with `B2_FOLDS = [0]` (single fold, val 0.8177 — strongest of 5).
       Hidden-test scoring SUCCEEDED. **LB = 0.928** (−0.003 vs v62
       baseline 0.931). Within single-fold noise floor (~0.03), so
       can't reject "B2 1-fold is neutral" — but is exactly at the
       step-10 kill threshold. Coincidentally matches v61's LB 0.928
       (B1 w=0.00, no-B1 path) — different code paths, same LB number.
9e. [x] **Option 1 implemented** (2026-04-27 ~17:50 EDT). Replaced
       Cells 41 (B2) and 42 (A1) with **one merged cell** that decodes
       audio + builds PCEN mels ONCE per file, then runs both B2 and
       A1 forwards on the shared mels. Recovers ~50% of B2's per-file
       overhead. Notebook is now 44 cells (was 45). Backup:
       `*.bak_pre_merge_b2_a1`.
9f. [x] `kaggle kernels push` v66 (2026-04-27 ~17:55 EDT) — **first
       multi-fold B2 LB probe within budget**: merged cell with
       `B2_FOLDS = [0, 1, 3]` (top-3 by val, mean 0.8003) + A1 4-fold
       baseline. Log: `log/kaggle_kernels_push_b2_v66_*.log`.
10. [x] **v66 hidden-test TIMED OUT** (2026-04-27 ~20:00 EDT). 3-fold B2
        + 4-fold A1 with the merged cell still didn't fit the 90-min
        hidden-test budget. Per the step-9f→10 timeout contingency,
        cutting B2 to 2-fold for the last probe before kill.
11. [x] **v67 prep** (2026-04-27 ~20:10 EDT). Cell 41 edited:
        - `B2_FOLDS` cut from `[0, 1, 3]` (mean val 0.8003) to `[0, 3]`
          (top-2 by val: 0.8177 / 0.8001, mean 0.8089).
        - Per-file mel pipeline batched: 12 windows are tiled + mel-spec
          + PCEN'd in a single batched call instead of 12 sequential
          calls. Numerically equivalent to prior path (max abs diff
          5.96e-8 = float32 ULP noise from amin/amax reduce ordering;
          verified locally on synthetic 60-s waveform). Marginal wall-
          time win (~1.15× on the mel pipeline = ~35 s/run on 700 files);
          the load-bearing fix is the fold cut (1 of 7 forwards = ~14%
          of SED wall time).
        - Backup: `*.bak_pre_v67_optim`.
12. [x] `kaggle kernels push` v67 (2026-04-27 ~20:13 EDT) — **hidden-test
        TIMED OUT** (~22:13 EDT, user-confirmed via Kaggle UI). 2-fold
        B2 + 4-fold A1 with the merged + batched-mel cell still didn't
        fit the 90-min hidden-test budget. Even the cheapest multi-fold
        B2 setting won't make budget on this kernel.
        Log: `log/kaggle_kernels_push_b2_v67_20260427_201335.log`.
13. [x] **Track B (B2) KILLED** (2026-04-27 ~22:15 EDT). Per §14.16
        step 12 timeout-rule, B2 is unconditionally killed. The
        architectural-diversity premise (`proto_b2_corr = 0.2486`,
        diversity gate easily passed) is empirically valid; the issue
        is purely wall-time on Kaggle's CPU-only kernel — every multi-
        fold setting that can plausibly move LB also blows the budget,
        and 1-fold (v65) returns LB 0.928, within single-fold noise of
        the v62 baseline.
        Notebook reverted to `*.bak_pre_b2_cell37b` (Cell 41 = A1-only
        4-fold rank fusion, 44 cells, zero B2 references).
        Pre-revert v67 saved as `*.bak_v67_killed` for archive.
14. [ ] **Optional v68 lock-in push.** Live kernel currently shows v67
        (timed out). If desired, push the reverted notebook as v68 to
        lock the production-equivalent code into the live kernel. Not
        required: LB-best submission selection on Kaggle UI is what
        scores; v62 (LB 0.931) is already the best-scoring submission.

---

## ⏸️ PICK UP HERE — previous (2026-04-27 ~22:15 local — Track B (B2) KILLED, notebook reverted to v62-equivalent — SUPERSEDED by §14.17)

**Kaggle state.** Last pushed = **v67** (TIMED OUT). Live notebook on disk
is now reverted to the v62-equivalent A1-only state (`*.bak_pre_b2_cell37b`).
Production baseline = v62 at LB **0.931**. Kernel URL:
https://www.kaggle.com/code/stevewatson999/birdclef-2026-protossm

**Today's submission ledger:**
| ver | pipeline | hidden-test result |
|-----|----------|--------------------|
| v62 | A1+B1 only (no B2) | LB **0.931** (production floor) |
| v63 | + B2 5-fold (separate cells) | TIMED OUT |
| v64 | + B2 3-fold (separate cells) | TIMED OUT |
| v65 | + B2 1-fold (separate cells) | LB 0.928 (= kill threshold) |
| v66 | + B2 3-fold (MERGED cell) | TIMED OUT |
| v67 | + B2 2-fold [0,3] (merged + batched mel) | TIMED OUT |

**Track B verdict.** Closed. B1 (PerceiverIO rank fusion) settled at
fixed +0.003 contribution per `project_b1_weight_sweep_closed.md`; B2
(ConvNeXt SED) is LB-dead at every multi-fold setting that fits the
90-min CPU-only budget, and LB-neutral at 1-fold.

**TWO misdiagnoses on the same day:**
1. (~13:00 EDT) v63 read as a kernel timeout → pushed v64 with 3-fold cut.
   In reality v63's user-run completed; what failed was the hidden-test
   scoring run (CLI doesn't surface this cleanly).
2. (~15:45 EDT) After pulling v64's user-run log (10 min wall time, valid
   submission) I claimed "Kaggle scoring queue stalled" — wrong again.
   The hidden-test scoring re-run was timing out on the same root cause.

**Root cause (now correct):** Kaggle code competitions re-run the kernel
on the FULL hidden test for LB scoring. The CLI/kernel-output log only
shows the user-initiated preview-test run. To get within budget on the
hidden test, B2 either needs (a) far fewer folds, or (b) the merged
streaming pass that shares per-file audio decode + mel build between B2
and A1 (option 1). v63/v64's `proto_b2_corr = 0.2486` is still valid —
diversity passes; the issue is purely wall time.

**v67 result.** Hidden-test TIMED OUT (~22:13 EDT, user-confirmed via UI).
2-fold B2 + 4-fold A1 with merged + batched mel still didn't fit.
Kill rule fired per §14.16 step 12.

**Track B closure rationale.**
- v65 (1-fold B2) fit budget but landed LB 0.928 — within single-fold
  noise floor of v62's 0.931, can't reject "B2 is neutral".
- v66 (3-fold) and v67 (2-fold) both timed out even with the merged
  streaming cell.
- Conclusion: every B2 fold count that *might* move LB blows the budget;
  every fold count that fits is LB-neutral. Architectural-diversity
  premise (§14.16) is LB-dead under the kernel's CPU-only constraint.
- Notebook reverted to `*.bak_pre_b2_cell37b`; v67 archived as
  `*.bak_v67_killed`.

**Open follow-up: lock-in push (v68).** Optional. Live kernel currently
shows v67 (timed out). LB-best selection on Kaggle UI is what scores
(v62 at LB 0.931 is already the best). Pushing the reverted notebook
as v68 only matters if you want the live kernel code to match the
production state for hygiene.

**Decision pending: next track to attack.** Per audit memory, most
shortlist levers are killed (Tier-1 post-proc, T2.x, M-tier, L1, L2,
L3, L5b, P8 5-fold, P12, B2). Remaining open candidates per
`project_prod_postproc_audit_2026_04_22.md`:
- Inference-only: P14 genus proxy.
- Training-side: P5 SWA, P7 SNR, M3 (Noisy Classmates / multi-arch
  co-evolutionary self-training, currently deferred per
  `reference_noisy_classmates_flagged.md`).

**Local GPU.** Idle. B2 5-fold landed cleanly.
- 5-fold mean val ROC-AUC = **0.7904** vs gate **0.7414** → **PASS by +0.049**
- Fold spread: 0.8177 / 0.7831 / 0.7677 / 0.8001 / 0.7834
- Caveat: fold-0 (+0.04 above next-best) is wider than the 0.03 single-fold
  noise floor, so don't extrapolate the +0.049 directly to LB lift.
- Status report: `log/b2_5fold_status_20260427_083000.txt`

**Today's settled dust (all DONE):**
- §14.16 step 4 (5-fold train) — DONE
- §14.16 step 5 (gate ≥ 0.7414) — PASS (mean 0.7904)
- §14.16 step 6 (JIT export) — DONE; `src/export_b2_jit.py` written
- §14.16 step 7 (Kaggle dataset push) — DONE; `stevewatson999/birdclef-2026-b2-convnext-ckpts`
- §14.16 step 8 (Cell 37b + CFG splice + soft diversity gate) — DONE
- §14.16 step 9 (kernel push v63) — preview-test ran fine; **hidden-test
  scoring re-run TIMED OUT** on Kaggle's 90-min CPU budget
- §14.16 step 9b (v64 refit + repush) — same fate; 3-fold B2 cut wasn't
  deep enough; hidden-test scoring re-run TIMED OUT
- §14.16 step 9c — diversity gate empirically validated (proto_b2_corr = 0.2486
  in v64 preview-test log; B2 is meaningfully diverse — wall time, not
  signal, is what blocks B2)
- §14.16 step 9d (v65 cheap probe) — B2 cut to 1-fold [fold 0]; targets
  ≤90 min hidden-test wall time

**In flight:** §14.16 step 10 — v65 hidden-test LB result.

### When v64 LB lands

**If v64 completes (no timeout):**
- LB ≥ 0.934 → keep B2; consider going back to 5-fold via option-1 merge
  before Phase 2.
- LB 0.929–0.933 → neutral; sweep `b2_frozen_weight_submit ∈ {0.05, 0.15, 0.20}`.
- LB ≤ 0.928 → revert to v62 baseline. Track B closed.

**If v64 also times out:**
- Escalate to **option 1**: merge Cells 41 (B2) and 42 (A1) into a
  single streaming pass that decodes audio + builds mels ONCE per file
  and runs both B2 and A1 on the shared mels. Recovers ~50% of B2's
  overhead so the original 5-fold B2 can come back.
- Or: drop B2 to 1 fold (fold 0 alone, val 0.8177) — smallest envelope,
  smallest signal, last-resort for an LB read.

### Don'ts on waking:
- Don't start Phase 2 (ProtoSSM pseudo-labels overlay on B2) before
  the v64 LB resolves.
- Don't trust the +0.049 val gap as an LB-lift forecast — noise
  floor + diversity uncertainty dominate.
- Don't use `pseudo_bc25ss_probs.npz` if Phase 2 is greenlit — D3
  killed that substrate.
- Don't re-enter B1 tune-up — sweep closed at w=0.10 (quadratic peak).
- Don't try to load any JIT ckpt on GB10 without the six-line fuser
  disable + autocast drop (`feedback_gb10_nvrtc_jit.md`). The B2
  trace was done CPU-side so the export script itself is unaffected;
  the inference notebook runs on Kaggle's CPU container so no NVRTC.

### Don'ts on waking:
- Don't start Phase 2 (ProtoSSM pseudo-labels overlay on B2) before
  the Phase 1 LB probe resolves.
- Don't trust the +0.049 val gap as an LB-lift forecast — noise
  floor + diversity uncertainty dominate.
- Don't use `pseudo_bc25ss_probs.npz` if Phase 2 is greenlit — D3
  killed that substrate.
- Don't re-enter B1 tune-up — sweep closed at w=0.10 (quadratic peak).
- Don't try to load any JIT ckpt on GB10 without the six-line fuser
  disable + autocast drop (`feedback_gb10_nvrtc_jit.md`). The B2
  trace was done CPU-side so the export script itself is unaffected,
  but the inference notebook needs the standard guard.

### Useful artifacts on disk:
- `models/b2/b2_convnext_tiny_fold{0..4}_seed42_hybrid.pt` — raw 5-fold ckpts (~120 MB each)
- `kaggle_datasets/b2-convnext-ckpts/b2_fold{0..4}.pt` — JIT-traced ckpts (Kaggle-bound copy)
- `kaggle_datasets/b2-convnext-ckpts/dataset-metadata.json` — corrected title (48 chars)
- `data/v56_soundscape_oof.npz` — P12 calibration substrate (3.3 MB).
- `kaggle_datasets/_backups/a1_fold{0,1,2,4}_v56_20260423.pt` —
  v56 A1 JIT ckpts (revert source if anything overwrites the live ones).
- `models/a1/a1_..._fold{0,1}_seed42_hybrid_swa.pt` — SWA kill ckpts kept as evidence.
- `models/b1_pretrained/{b1_pretrained, b1_seed{0,1,2}}.pt` — B1 ckpts (frozen at w=0.10).
- `jupyter/protossm-postproc/*.bak_pre_*` — notebook backups for every prior attempted edit.

### Open follow-ups (not blocking):
- §14.14.5 Noisy Classmates audit (still on hold pending Track B
  resolution + M3 + published recipe).
- §14.15 NotebookLM cross-source sweep backlog: learnable GeM,
  weighted-distributed secondary labels, min-10 minority upsampling.
- §14.16 Phase 2 (ProtoSSM pseudo-labels overlay on B2) — NOT
  committed; only triggered if Phase 1 LB probe survives.
- `src/_p8_eval_pair.py` cleanup — keep until next training-side
  lever ships.

---

## 14.17 BC2025-winners deep audit + train_soundscapes_labels discovery (2026-04-27 ~22:45 local)

**Trigger.** User asked: "we're running out of time and we haven't been
able to break through the 0.933 barrier. Look at what's in the
discussions for anything that might give us that break-through. Do we
need to go in an entirely new direction?"

Pulled BC2025 1st (Babych) and 2nd (Sydorskyi/vialactea) place writeups,
which are now public. Distilled them into
`reference_bc2025_winners_writeups.md`. Also did a 30-minute audit of
`data/raw/train_soundscapes_labels.csv` after the Dauphine BC2026
strategy playbook flagged "*some train_soundscapes are labeled by
expert annotators this year*."

### 14.17.1 The BC2025 ablation that explains everything

**2nd-place EffNetV2-S Public LB ablation table (their Table 5):**

| Stage | Public LB | Δ |
|---|---|---|
| Strong baseline (mixup+specaug+secondary labels+focal+BCE) | 0.837 | — |
| + label smoothing 0.05, modified sampling, RandomFiltering, BG mix | 0.835 | 0 (helped Private only) |
| **+ in-domain pretrain (819K Xeno-Canto recs, 7,489 species, 50 ep)** | **0.881** | **+0.046** |
| **+ pseudo-label iter 1** (soft, ≥0.5 keep, <0.1 zero, OOF-fold safe) | **0.908** | **+0.027** |
| + pseudo iter 2 (full + OOF) | 0.917 | +0.009 |
| + TopN (N=1) postproc | 0.918 / 0.924 priv | +0.001 / +0.007 |
| Optuna ensemble (3 of 8) | 0.925 / 0.928 priv | +0.005 |

**The 1st-place writeup is literally titled "Multi-Iterative Noisy
Student Is All You Need."** Both top-2 winners say the same thing:
**iterative pseudo-labeling on the in-domain unlabeled soundscape
corpus + comprehensive in-domain pretraining are THE deltas.**

### 14.17.2 Why our equivalent attempts (L1, C2, D3, L2) all died

We've been re-attacking this lever for weeks under different code names.
Each had a *specific* implementation flaw the BC2025 winners avoided:

| Our killed lever | What we did | What the winners did |
|---|---|---|
| **L1** (cross-arch noisy student, §14.9) | max-merge **hard** targets, BCE, 100 % pseudo sampling | **soft** targets, thresh ≥0.5 + zero <0.1, **40 %** pseudo sampling, mix via MixUp on audio |
| **C2** (student on §10 teacher) | teacher trained ON `train_soundscapes` → val leakage by construction | OOF fold-aware: each SS fold predicted only by models that didn't see it |
| **D3** (BC25-SS pseudo) | wrong substrate — used **BirdCLEF-2025** soundscapes, not 2026 `train_soundscapes` | the SS that **matches the test domain** |
| **L2** (multi-year pretrain) | ~28 K BC2025-only recs, ~206 species | **819 K recs, 7,489 species** spanning all prior BirdCLEFs ∪ Xeno-Canto |

The 2nd-place team also spent weeks on a "rhythmic-pattern conv
refinement model" (their §5.6 L2) and **it did not lift LB** — same
shape as our Track-C / §10 ProtoSSM teacher dead end. So that family
of failures is normal, not a sign we should keep digging there.

### 14.17.3 train_soundscapes_labels.csv audit (the Dauphine playbook claim, verified)

`data/raw/train_soundscapes_labels.csv` exists and IS expert-labeled —
but the playbook implication "all train_soundscapes are labeled this
year" is false. Reality:

| Property | Value |
|---|---|
| Rows | 1,478 (with 739 byte-identical duplicates → 739 unique chunks) |
| Labeled files | **66 of 10,658** (~0.6 % of SS volume) |
| Audio labeled | ~66 minutes total (60-sec files × 66) |
| Chunk granularity | 5-sec windows, semicolon-joined multi-label |
| Species coverage | **75 of 234** target classes (32 %) |
| No-call chunks | 53 of theoretical 792 (~7 %) — omitted, not zero-labeled |
| Multi-label density | min 1 / median 4 / max 10 species per 5-sec chunk |
| Class breakdown of the 75 | Aves 28 (of 162), Insecta 25 (of 28), Amphibia 17 (of 35), Mammalia 4 (of 8), Reptilia 1 (of 1) |

**Site bias warning.** Of 23 sites in `train_soundscapes`, only 9 have
labeled files. Site S22 dominates with 40 of 66 labeled files (61 %),
and S22 is also the largest unlabeled site (3,383 files). But sites
**S02 (2,505 files), S01 (2,341 files), S13 (1,873 files)** — which
together are 63 % of unlabeled SS volume — have **zero labeled files.**
So this 66-file slice is **not a uniform sample** of the SS distribution
and must be treated as a biased mini-val, not a representative one.

### 14.17.4 What this enables (in priority order)

1. **First clean LB-domain mini-validation.** Macro-AUC computed against
   the 66-file expert-labeled subset is in the same domain as the
   hidden test set. Our `val_v2` is built from `train_audio` focal
   clips (out-of-domain weak labels) — and the 2nd-place paper
   measured **Pearson −0.13 / Spearman −0.12** between mean-CV and
   Public LB once Public > 0.9. We've independently confirmed this
   pattern via 6 +val→−LB kills. A direct in-domain val should track
   LB much better. Caveat: 66 files / 75 species / S22-biased — not
   a substitute for LB, but a *better-predictive* gate than what we
   have today.

2. **Pseudo-label quality calibration.** For the 66 expert-labeled
   files, predict with the existing v62 A1 ckpts and measure
   precision/recall of pseudo-label thresholds (≥0.5 keep, <0.1 zero)
   before trusting them on the unlabeled 99.4 % of SS. This is the
   missing diagnostic that would have caught L1's max-merge mistake
   *before* a kernel push. Cost: ~10 minutes once we have OOF preds.

3. **Strong-label training anchor.** 739 chunk-level positive rows can
   be added to A1 training as gold rows. Too small to move LB alone
   (~0.07 % of `train.csv`), but combined with the pseudo-label loop
   on the other ~10,592 unlabeled SS files gives a clean anchor.

4. **Site-grouped fold validation.** Filename pattern
   `BC2026_Train_NNNN_S{site}_{YYYYMMDD}_{HHMMSS}.ogg` exposes
   site_id. The BC2026 playbook explicitly flagged site as a likely
   group-split key. Our `val_v2` grouping doesn't use site. This is a
   free upgrade orthogonal to the rest.

### 14.17.5 Track A2 — re-attack iterative noisy student, done correctly

**This is the recommended next direction.** The path is *not*
greenfield; it's a careful re-implementation of the BC2025-winning
recipe with the four prior pitfalls explicitly avoided.

**Step 1 — OOF pseudo-emit** (~1 day, no GPU strain):
- Predict on all 10,658 `train_soundscapes` files with the 4 v62 A1
  fold ckpts in **OOF mode** — fold k's SS files predicted only by
  models from folds ≠ k (use the `train_audio` recordist→fold map
  to determine SS file→fold by site or pre-defined assignment, OR
  just round-robin for the unmapped SS files since they aren't in
  the A1 training set).
- Output: per-(file, 5-sec-chunk, class) sigmoid array, ~1.4 GB at
  fp16.

**Step 2 — Quality calibration on the 66 GT files** (~30 min):
- For the 66 files, compute precision/recall of `(score ≥ 0.5)` vs
  GT positives at chunk level. **Hard gate:** if precision < 0.7
  the threshold is too lax — raise or abandon. If recall < 0.3 we
  miss too many positives — lower or accept smaller pseudo set.
- This gate is what we lacked on L1.

**Step 3 — Filter + soft-target pseudo set** (~1h):
- Keep chunks where max(prob) ≥ 0.5 (Sydorskyi's value).
- Zero per-class probs < 0.1 (kills overconfident noise).
- Save as soft-label CSV in same schema as `train.csv` but with
  per-class floats instead of one-hot.

**Step 4 — Retrain A1 5-fold** (~13–17h overnight):
- Same hybrid loss, same architecture, same MixStyle/SpecAug.
- Add the 40 %-pseudo / 60 %-train sampling rule per
  `dataset_a1.py` (gated by class-in-pseudo-set).
- Mix at audio level via existing MixUp path (sums pseudo + train
  labels element-wise, clip to [0, 1]).

**Step 5 — Two-channel val gate** (before any LB push):
- Channel A: existing `val_v2` 0.7414 hybrid gate (must not regress
  by more than the 0.03 noise floor).
- Channel B (new): macro-AUC on the 66-file expert-labeled mini-val.
  Must improve. If it doesn't, recipe is broken — STOP, do not push.

**Step 6 — LB probe** (1 submission):
- Per `feedback_kernel_timeout_vs_scoring_stall.md`, watch the
  Kaggle UI for the hidden-test re-run, not the CLI status.
- Decision rules:
  - LB ≥ 0.940 → ship; queue iter 2 (re-pseudo on this model's
    output).
  - LB 0.932–0.939 → neutral within noise; iterate once more before
    deciding.
  - LB ≤ 0.930 → recipe broken; revert. Do not iterate.

**Realistic LB envelope.** If we cleanly recover 50 % of the 2nd-place
pseudo delta on this year's data: +0.018 → ~0.949 (leader-class). 25 %
recovery: +0.009 → ~0.940 (gold zone). The 1st place won at 0.93 on
BC2025 — the BC2026 leader at 0.951 likely uses this exact recipe with
careful execution.

**Why this is not "L1 again."** L1 used max-merge hard targets + BCE
+ 100 % pseudo sampling on a teacher trained on `train_soundscapes`
(val leakage). Track A2 uses soft + thresh + 40 % sampling + OOF
fold-safe pseudo + strong-label anchor + an in-domain mini-val that
will actually predict LB. These are not minor knob tweaks — the 2nd
place ablation gives this combination +0.036 LB on the same arch
family (EffNet) and substrate type we already have on disk.

### 14.17.6 What's NOT in Track A2 (deliberately deferred)

- **L2-redux at the 819 K-record / 7,489-species scale.** That's the
  *largest* single delta in the 2nd-place table (+0.046), but it's a
  ~1 week scope (Xeno-Canto bulk download + GPU pretraining run +
  finetune). Track A2 first because it reuses existing ckpts; if A2
  succeeds, L2-redux is the obvious next step on the gold path.
- **NFNet-L0 / EffNetV2-S architecture swap.** The winners used
  these instead of EffNet-B0. We just killed B2 (ConvNeXt) on
  budget. If A2 doesn't lift LB, the next architectural lever is
  swapping the backbone in `train_a1.py` to EffNetV2-S — but only
  after A2 confirms the substrate is the bottleneck, not the arch.
- **TopN postproc (N=1).** +0.001/+0.007 on 2nd place's table.
  Cheap parallel probe but unlikely to break the 0.933 wall on its
  own. Queue after A2.
- **SoftAUCLoss.** Babych's custom pairwise log-loss. Material code
  change to `train_a1.py` and unproven outside their stack. Defer.

### 14.17.7 Files to be created when Track A2 starts

- `src/a2_emit_oof_pseudo.py` — predict on `train_soundscapes`
  with v62 A1 ckpts, fold-safe.
- `src/a2_calibrate_pseudo.py` — precision/recall of thresholds
  against the 66-file GT.
- `data/processed/a2_pseudo_soft.csv` — filtered soft-label set.
- `src/a2_train.py` — A1 train fork with 40 %-pseudo sampling rule
  + GT-anchor inclusion.
- `src/a2_val_v3.py` — in-domain mini-val build over the 66 GT
  files.

### 14.17.8 Open question for the user

Two go/no-go decisions before kicking off:
1. **Approve Track A2** as the next direction, or override toward
   one of the deferred levers (L2-redux, arch swap, TopN, etc.)?
2. If Track A2 is approved, should the 4-fold v62 A1 ckpts be used
   for the OOF emit, or should we retrain a fresh 5-fold first to
   get all 5 fold ckpts (we currently have 0/1/2/4 only — fold 3
   was dropped per standard, see `project_p8_5fold_killed.md`)?
   Recommendation: use the 4 we have. Round-robin the unmapped SS
   files. ~½ day saved vs fresh retrain.

---

## ⏸️ PICK UP HERE — previous (2026-04-27 ~22:45 local — Track A2 scoped, awaiting go/no-go — SUPERSEDED by §14.17.9)

---

## 14.17.9 Track A2 launched — OOF emit running overnight (2026-04-28 ~00:15 local)

**Decision recorded.** User approved Track A2 (§14.17.5) and chose option
(a) for ckpt source: 5 P8 5-fold raw `.pt` ckpts at
`models/a1/a1_..._fold{0..4}_seed42_hybrid.pt` (LB 0.928 single-ensemble,
within noise of v62 0.931; pseudo-emit averages 4 of 5 per chunk so
individual-ckpt quality is absorbed).

**Track A2 chain — code complete end-to-end:**

| Step | Script | Input | Output |
|---|---|---|---|
| 1 — OOF emit | `src/a2_emit_oof_pseudo.py` | 5 fold ckpts + `train_soundscapes/` | `data/processed/a2_train_ss_oof_probs.npz` |
| 2 — gate | `src/a2_calibrate_pseudo.py` | NPZ + `train_soundscapes_labels.csv` | stdout + `data/processed/a2_calibration_report.csv` |
| 3 — filter | `src/a2_filter_pseudo.py` | NPZ | `data/processed/a2_pseudo_soft.npz` + `a2_pseudo_audit.csv` |
| 4 — retrain | `src/a2_train.py` + `src/dataset_a2.py` | pseudo NPZ + `train_folds.csv` | `models/a2/a2_..._fold{F}_seed42_hybrid.pt` |

### 14.17.9.1 Step 1 — OOF emit RUNNING overnight

- **PID:** check via `pgrep -f src/a2_emit_oof_pseudo`
- **Log:** `log/a2_emit_oof_20260428_001358.log` (the second launch — first
  was killed at 7 min for the CUDA-fork deadlock; see
  `feedback_dataloader_cuda_fork_deadlock.md`)
- **Configuration:**
  - 10,658 SS files × 12 windows × 5 folds = 127,896 chunks emitted
  - md5-hash bucket distribution: [2162, 2109, 2056, 2164, 2167] (~21 % each)
  - 8 DataLoader workers, `multiprocessing_context='spawn'` (mandatory
    on this machine — see feedback memory)
  - GB10 GPU, ~2.4 GB peak, batch=4 files = 48 mels per fold pass
- **Steady-state rate:** ~4.7 files/s (37 % done at 00:13, ETA 24 min from
  there → expected to finish ~00:38 local 2026-04-28)
- **Output:** `data/processed/a2_train_ss_oof_probs.npz` (~120 MB
  compressed). Schema: `probs (N,234)`, `filenames (N,)`, `start_sec (N,)`,
  `oof_bucket (N,)`, `fold_set (5,)`.

### 14.17.9.2 On waking — verify emit + run gate

**1. Confirm emit completed cleanly.**

```bash
cd /home/swatson/work/kaggle/BirdCLEF/four_track
ls -la data/processed/a2_train_ss_oof_probs.npz
tail -20 log/a2_emit_oof_*.log
```

Expected last log line: `saved data/processed/a2_train_ss_oof_probs.npz
(~120 MB) in <T> s total`. If the log shows a traceback or no `saved`
line, emit failed — diagnose before proceeding.

**2. Run the gate (calibration).** ~5 s, no nohup needed.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
python -u src/a2_calibrate_pseudo.py 2>&1 | tee log/a2_calibrate_$(date +%Y%m%d_%H%M%S).log
```

The script prints a threshold sweep table over both "covered" (75
species in GT) and "all 234" scopes, plus per-class P/R/F1/AUC and a
hard-gate verdict at the end:

> GATE: precision @ 0.50 on covered species = X.XXX
>       required >= 0.70
>       RESULT: PASS / FAIL

**Decision rules per §14.17.5:**
- PASS → proceed to step 3 (filter).
- FAIL → A2 recipe broken on our setup. **Do not retrain.** Diagnose:
  is it threshold (try 0.6 or 0.7)? Is it per-fold-bucket noise
  (check per-bucket precision)? Is it a class-coverage artifact (only
  Aves/Insecta active in val, but pseudos mostly Amphibia)?
  If diagnosis doesn't yield a fix in <2 hours, **abort A2** and
  reconsider deferred alternatives in §14.17.6 (L2-redux 819K-rec
  pretrain; arch swap to NFNet-L0 / EffNetV2-S; TopN N=1 postproc).

**3. If PASS, run filter + smoke + overnight retrain.**

```bash
# Filter — ~1 min, runs to completion
python -u src/a2_filter_pseudo.py 2>&1 | tee log/a2_filter_$(date +%Y%m%d_%H%M%S).log

# Smoke test fold-0, 1 ep, 1 batch — verifies wiring before overnight burn
rm -f log/*.log    # per feedback_rm_log_every_launch.md
nohup python -u src/a2_train.py --fold 0 --smoke-test \
    > log/a2_train_smoke_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Wait for smoke to finish (~5 min). Confirm "Fold 0 complete" line
# appears in the log and no traceback.

# Overnight 5-fold retrain — ~13–17 h
rm -f log/*.log
nohup python -u src/a2_train.py --folds 0,1,2,3,4 --loss hybrid --epochs 25 \
    > log/a2_train_5fold_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 14.17.9.3 Track A2 design notes (for the morning brain)

- **Sampling rule:** `MixedTrainPseudoDataset` in `dataset_a2.py` uses
  flat 40 % pseudo / 60 % train per `__getitem__`, NOT Sydorskyi's
  exact "conditional on per-class pseudo coverage" rule. Iter-1
  approximation; iter-2 can refine if iter-1 lifts LB.
- **OOF safety:** `SoundscapePseudoDataset(fold_filter=k)` keeps only
  pseudo rows where `oof_bucket == k`. Fold k's model never sees
  pseudo-labels generated by itself.
- **Cross-mixup deliberately deferred to iter-2.** `BirdTrainDatasetA1`
  does train×train MixUp on train items; pseudo items skip MixUp.
  Sydorskyi's full recipe mixes train (hard) × pseudo (soft) via
  element-wise sum + clip — that's a +0.005-ish refinement on top of
  the dominant +0.027 from the basic pseudo loop. Add later if iter-1
  is +Δ on LB.
- **Loss:** hybrid (BCE+ASL), default — matches v62 baseline. Both
  BCE and ASL handle soft (float) targets natively (verified in
  `src/losses.py` lines 51–82).
- **Ckpt path:** `models/a2/a2_<backbone>_fold{F}_seed42_hybrid.pt`.
  `models/a1/` is untouched — full revert path preserved.
- **Validation gate:** existing val_v2 0.7414 hybrid baseline still
  applies (Channel A per §14.17.5 step 5). The 66-file val IS the
  in-domain mini-val; no separate `a2_val_v3.py` needed.

### 14.17.9.4 LB-probe sequence (after 5-fold finishes)

After overnight 5-fold:
1. Verify all 5 folds finished and val ≥ 0.7414 on at least 4 of 5.
2. Back up `kaggle_datasets/a1-effb0-ckpts/` → `_backups/` per
   `feedback_backup_ckpts_before_overwrite.md`.
3. Adapt `src/export_a1_jit.py` to read from `models/a2/a2_*` (or
   make a sibling `src/export_a2_jit.py`). Apply the GB10 NVRTC
   guard from `feedback_gb10_nvrtc_jit.md`.
4. Push the new dataset version to Kaggle, update notebook ckpt path
   if needed (it's pinless — should auto-pickup).
5. Single LB probe; per §14.17.5 decision rules:
   - LB ≥ 0.940 → ship, queue iter-2 (re-pseudo on the new model)
   - LB 0.932–0.939 → neutral within noise; iterate once more
   - LB ≤ 0.930 → recipe broken; revert; do NOT iterate

### 14.17.9.5 Don'ts on waking

- Don't run the calibrate / filter / train scripts before confirming
  the emit log shows `saved data/processed/a2_train_ss_oof_probs.npz`.
- Don't push to Kaggle until the gate passes AND the 5-fold mean val
  meets 0.7414 (Channel A noise floor).
- Don't trust val Δ < 0.003 — single-fold cuDNN noise dominates.
- Don't restart any killed lever (L1/C2/D3/L2/B2/P5/P12). The §14.17
  recipe is deliberately the first to fix the four pitfalls.
- Don't forget `rm -f log/*.log` before each launch
  (`feedback_rm_log_every_launch.md`).
- Don't use a DataLoader without `multiprocessing_context='spawn'`
  if CUDA models load before the loader
  (`feedback_dataloader_cuda_fork_deadlock.md`).

## ⏸️ PICK UP HERE — previous (2026-04-28 ~00:15 local — Track A2 OOF emit RUNNING overnight — SUPERSEDED by §14.17.10)

---

## 14.17.10 Track A2 KILLED at fold 0 ep 12 — pseudo-mass-augmentation drift (2026-04-28 18:25 local)

**Verdict.** Track A2 (Sydorskyi-recipe noisy-student re-attack) reproduces
the L1/L2/D3 family of kills. Killed cleanly at fold 0 ep 12. Did not
push to Kaggle, did not touch `models/a1/` or `kaggle_datasets/a1-effb0-ckpts/`.
Production state unchanged: v62 LB **0.931** is the floor.

### 14.17.10.1 The trajectory (val_v2 Channel A, 0.7414 baseline)

| Ep | val | Cycle | Note |
|---|---|---|---|
| 1 | 0.6651 | C1 (peak LR) | first ★ |
| 2 | 0.7097 | C1 | ★ |
| 3 | **0.7116** | C1 | ★ locked BEST |
| 4 | 0.7043 | C1 | |
| 5 | 0.7088 | **C1 end** | first cycle-end |
| 6 | 0.6897 | C2 restart | |
| 7 | 0.7116 | C2 | tied BEST, no new ★ |
| 8 | 0.6931 | C2 | |
| 9 | 0.6883 | C2 | |
| 10 | 0.6784 | **C2 end** | **−0.030 vs C1 end** |
| 11 | 0.6676 | C3 restart | |
| 12 | 0.6676 | C3 | killed |

`CosineAnnealingWarmRestarts(T_0=5)` so cycles end at ep 5/10/15/20/25.
The "later cycles improve" pattern from `project_p8_partial_win` is
**not holding here** — C2 end (0.6784) is *worse* than C1 end (0.7088).
Train loss bottomed at ep 10 (0.0151) and started creeping up — classic
overfitting.

### 14.17.10.2 The kill rationale (two hard signals)

1. **Cycle-end regression.** ep 10 < ep 5 by 0.030. The schedule's "later
   cycles refine" property is reversed — each cycle drives the model
   *further* from the val distribution.
2. **Monotonic post-BEST decay.** 9 epochs since ep-3 BEST, no recovery,
   trend is clearly down (0.7116 → 0.6676). Even the most generous read
   (assume cycle 5 ep 25 recovers fully to 0.7116) leaves us at
   −0.030 vs the 0.7414 gate, with 4 more folds needing to *average*
   ~0.749 to hit the gate. Implausible.

To hit §14.17.9.4's gate ("5-fold mean val ≥ 0.7414") fold 0 alone
already needs +0.030, which it hasn't shown any sign of. Continuing
fold 0 to ep 25 was 3h of insurance against a 5% chance of late
cycle peak. Continuing to fold 1+ was 18h of compute against
zero LB-probable upside.

### 14.17.10.3 Why probably (the recipe flaw)

Pseudo set has structural mass-augmentation in tail classes:

| species | pseudo rows | train rows | aug factor |
|---|---|---|---|
| 24321  | 2,567  | 2  | **1284×** |
| 70711  | 1,172  | 2  | 586× |
| 24285  | 9,170  | 19  | 482× |
| 22961  | 2,776  | 6  | 463× |
| 23158  | 11,129 | 25 | 445× |
| compau | 26,377 | ?  | (top by raw count) |
| brnowl | 19,454 | ?  | |
| compot1| 17,808 | ?  | |

At pseudo_ratio=0.40, a 1284× augmentation factor *swamps* the true
focal signal for those species. The model learns the soundscape
prevalence prior — which is **wrong for the focal-clip val** (and,
empirically, for LB too: same trap that killed L1).

Sydorskyi's exact recipe gates pseudo sampling on per-class coverage
(deferred to §14.17.9.3 as iter-2). We used a flat 40 % ratio. Iter-1
short-circuit failed. The cap is necessary, not "iter-2 polish".

### 14.17.10.4 Pseudo-emit calibration was *correct*

The §14.17.9 gate passed with margin (P=0.879 at 0.50; required ≥ 0.70).
The pseudos *themselves* are clean — high-precision per-chunk labels.
The failure is downstream, in the **sampling regime that consumes them**.
Don't blame the emit pass; blame the consume pass.

This nuance matters for the next iteration: re-running the emit isn't
necessary. Re-engineering the dataset's class-balance logic is.

### 14.17.10.5 Cleanup checklist — done

- ☑ Killed PID 173965 at ep 12 (18:25 local)
- ☑ Stopped event monitor (`b03ntpgqu`)
- ☑ Did not push any `models/a2/` ckpt to Kaggle
- ☑ `models/a1/` untouched — full revert path preserved
- ☑ `kaggle_datasets/a1-effb0-ckpts/` untouched
- ☑ Production notebook state unchanged (v62 LB 0.931 floor)

### 14.17.10.6 Disposable artifacts left on disk

- `data/processed/a2_train_ss_oof_probs.npz` (111 MB) — pseudo-emit NPZ.
  Worth keeping for the iter-2 attempt (re-emit costs 35 min of compute).
- `data/processed/a2_pseudo_soft.npz` (2.1 MB) — filtered pseudos.
  Still useful as the substrate; the *consumption* is what failed.
- `data/processed/a2_calibration_report.csv` — per-class P/R/F1/AUC.
- `data/processed/a2_pseudo_audit.csv` (65,658 rows) — chunk audit.
- `models/a2/a2_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid.pt`
  — fold-0 ep-3 BEST ckpt, val 0.7116. Not pushed; safe to delete or
  keep as evidence.

### 14.17.10.7 Open lever options after this kill

The deferred §14.17.6 alternatives + the §14.15 NotebookLM sweep are
where to look:

1. **A2-iter-1.5: bounded pseudo augmentation.** Cap per-class pseudo
   rows at e.g. min(K * train_rows, P_max). Re-use existing NPZ.
   ~2 h to add the cap to `dataset_a2.py`; full 5-fold burn is the
   same ~22 h. Highest-confidence next try (single change, isolates
   the mass-augmentation hypothesis).
2. **L2-redux 819K-rec pretrain (§14.17.6).** Sydorskyi's
   pretraining-corpus lever, which we deferred. Heavier engineering.
3. **Arch swap** to NFNet-L0 / EffNetV2-S (§14.17.6).
4. **TopN N=1 postproc** — inference-only, cheap probe.

If we go (1), the §14.17.10.4 finding lets us skip re-emitting and
go straight to a re-train.

### 14.17.10.8 Lesson — generalizable

When a pseudo-label gate passes at the chunk-precision level but
the consuming-sampler regime over-augments tail classes, the
pseudos are clean but the *training distribution* is wrong. Always
audit aug_factor (= pseudo_count / train_count) per class before
launching multi-fold burns; cap it before fold 1, not after.

This adds a new diagnostic to the precondition list, alongside:
- §14.14.7 (audit existing pipeline before borrowing techniques)
- T2.6 (treat +val as weak LB-predictive if >70 % added rows are for
  val-absent species OR BEST is a spike)
- §14.14.8 (single-fold cuDNN noise floor 0.03)

---

## ⏸️ PICK UP HERE — previous (2026-04-28 18:25 local — Track A2 KILLED at fold 0 ep 12 — SUPERSEDED by §14.17.11)

(See §14.17.11 below for the iter-1.5 cap probe and its kill verdict.)

---

## 14.17.11 Track A2-iter-1.5 KILLED — pseudo-cap hypothesis disproved (2026-04-29 00:30 local)

**Verdict.** Capping per-class pseudo aug_factor (`cap_k=10`) does **not**
fix the A2 kill. The §14.17.10.7 mass-augmentation hypothesis is wrong:
the bug is structural focal/soundscape val mismatch, not the consuming
sampler. Production state unchanged: v62 LB **0.931** floor.

### 14.17.11.1 What we did

Single-parameter cap added to `SoundscapePseudoDataset` (`cap_k` arg):
for each top-1 species c, keep at most `cap_k * train_count[c]` pseudo
rows (deterministic random subsample, seed 42, applied globally before
fold filter). With `cap_k=10`:

- 32,180 of 65,658 pseudo rows kept (49.0%)
- Worst pre-cap aug_factor (top-1 framing): 275× (sp 23158); 1284× when
  counting all soft attributions (per §14.17.10.3 framing)
- Worst post-cap aug_factor: 10× by construction
- Per-fold pseudo rows: ~6,400 (vs ~13,557 unfiltered)
- Big-train species (compau 493, brnowl 474, trsowl 491) keep most
  pseudos; tail species (24321/2, 555146/18, 23158/25) hard-capped

Single isolated change. NPZ unchanged. Same `pseudo_ratio=0.4`,
same hybrid loss, same backbone, same focal val substrate.

### 14.17.11.2 Sanity gate — fold 0, two seeds-of-cuDNN-noise repeats

To avoid burning 22 h on a noisy single-fold result, gated the full
5-fold burn behind two fold-0 sanity runs (both with `cap_k=10`,
loss=hybrid, seed=42, identical config — only cuDNN nondeterminism
between them):

| ep | run-1 (5 ep) | run-2 (10 ep) | original A2 (12 ep) |
|---|---|---|---|
| 1 | 0.6167 | 0.6246 | 0.6651 |
| 2 | 0.6798 | 0.6945 | 0.7097 |
| 3 | 0.7139 ★ | 0.6985 ★ | 0.7116 ★ |
| 4 | 0.6954 | 0.6838 | 0.7043 |
| 5 | **0.7255 ★** | **0.7130 ★** | 0.7088 |
| 6 | — | 0.6471 | 0.6897 |
| 7 | — | 0.7025 | 0.7116 |
| 8 | — | 0.6701 | 0.6931 |
| 9 | — | 0.6967 | 0.6883 |
| 10 | — | 0.6995 | 0.6784 |

Fold-0 cap_k=10 BEST mean across the two runs: 0.7193. Original A2
BEST: 0.7116. Δ = **+0.008** — well below the §14.14.8 cuDNN noise
floor of 0.03. Both runs peaked at ep 5 (cycle-1 end of T_0=5
CosineAnnealingWarmRestarts) and never recovered that peak in cycle-2
(ep 6-10) — same shape as original A2's ep 6-12 decline.

The 0.7414 baseline gate is **−0.028 above** the post-cap BEST. Even
under the most generous read (cap helps marginally, mean lift +0.008),
the focal-clip val ceiling for this recipe is ~0.72 — a structural
floor below the 0.7414 gate.

### 14.17.11.3 Why the cap didn't help

The §14.17.10 post-mortem framed the failure as "pseudos clean,
consuming sampler broken." The cap fixed the sampler (aug_factor ≤ 10
per class, deterministic). The val ceiling **still** held at ~0.72.

Therefore the bug is upstream of the sampler. Two consistent
hypotheses:

1. **Distribution-shape bias** (not magnitude). The pseudo set's
   *shape* across species reflects soundscape-frame prevalence. Even
   bounded at 10× train count, every soundscape-prevalent species
   (compau, brnowl, trsowl) gets a per-class pseudo budget that
   dwarfs the focal-clip per-class budget. The model still learns a
   prevalence prior shifted away from the focal-val distribution.

2. **Soft-label calibration mismatch.** The pseudo labels are A1's
   sigmoid outputs at 0.50 threshold (P=0.879 at gate). The focal
   ground truth is hard one-hot. Mixing them at 40/60 in the loss
   trains the model to a calibration somewhere between the two
   regimes — neither of which is the focal-val target.

Both are **deeper than the cap fixes.** This is the same family as
L1, L2, D3, A2-iter-1: any time a focal-clip A1 gets re-trained with
soundscape-style pseudo signal, the focal-val ceiling drops by ~0.02
and the model never recovers in 5+ subsequent epochs. The pattern is
robust enough now (5 levers, 5 kills) to be treated as a structural
constraint, not a tuning problem.

### 14.17.11.4 Cleanup checklist — done

- ☑ Killed sanity-2 cleanly at ep 10 (00:29:20 local) — process exited
- ☑ Did not push any `models/a2/` ckpt to Kaggle
- ☑ `models/a1/` untouched
- ☑ `kaggle_datasets/a1-effb0-ckpts/` untouched
- ☑ Production notebook state unchanged (v62 LB 0.931 floor)
- ☑ Killed-run logs archived under `log/archive/`

### 14.17.11.5 Disposable artifacts on disk

Same as §14.17.10.6 (pseudo NPZ + audit + filter report still useful
substrate; cap is a code change, not data). Plus:

- `models/a2/a2_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid.pt`
  — overwritten by sanity-2 ep-5 BEST (0.7130). Disposable evidence.
- `log/a2_sanity_capk10_fold0_*.log` (two runs, ~3.5 h compute total)
  — kept for the comparison table in §14.17.11.2.

### 14.17.11.6 Code change to keep

`src/dataset_a2.py` `SoundscapePseudoDataset.__init__` now accepts a
`cap_k`/`train_per_class`/`cap_seed` triple (default `cap_k=0` =
disabled). `src/a2_train.py` exposes `--cap-k` (default 0). Leaving
the code in: it's a clean bounded-aug primitive that may matter for a
future pseudo-label recipe that does work, e.g. if we ever revisit
with a different val substrate. No revert needed.

### 14.17.11.7 Open levers — post-iter-1.5

The §14.17.10.7 list, minus (1):

1. ~~A2-iter-1.5 cap~~ — KILLED here.
2. **L2-redux 819K-rec pretrain (§14.17.6).** Sydorskyi's
   pretraining-corpus lever. Heavier engineering: needs the full
   xeno-canto + iNat dump (~819K recs, 7,489 species), a
   pretraining-only run, then transfer to BC2026 finetune. Note:
   L2-redux *also* uses pseudo-style data (pretraining on a much
   larger, pre-2026-overlapping corpus), but the failure mode here is
   different — the pretraining is what sets the prior, not pseudo
   mixing during BC2026-finetune. Not blocked by §14.17.11
   findings. **Recommended next.**
3. **Arch swap to NFNet-L0 / EffNetV2-S (§14.17.6).** Architectural
   change isolated from the pseudo-label kill family. Cheaper than
   L2-redux but speculative — there's no winners-writeup support that
   either of these alone moves the needle on this val.
4. **TopN N=1 post-processing.** Inference-only, cheap, doesn't need
   any retrain. Borrowed from `reference_competitor_master_report_20260318`.

### 14.17.11.8 Don'ts — updated

- Do **not** re-attempt any pseudo-label-during-finetune recipe with
  EffNet-B0 + soundscape pseudos + focal val. We've now killed the
  full kill-family span: L1 (max-merge teachers), L2 (focal→ss),
  D3 (BC25-SS), C2 (focal pseudos), A2-iter-1 (no cap), A2-iter-1.5
  (cap). 6/6 kills. Stop.
- Do **not** restart any L1/L2/D3/C2/B2/P5/P12 lever.
- Do **not** push the fold-0 ep-5 cap-ckpt to Kaggle.

### 14.17.11.9 Lesson — generalizable

The §14.17.10.8 lesson ("audit aug_factor before launching multi-fold
burns") is correct but insufficient. **Capping aug_factor is a
necessary-not-sufficient fix.** The deeper invariant: a focal-clip
SED model's val ceiling is set by the focal-clip training distribution
shape, not its magnitude. Any pseudo-label scheme that perturbs the
shape (regardless of magnitude bound) costs ~0.02 of focal val, and
that cost translates roughly 1:1 to LB on this competition.

For future BC competitions or analogous SED problems, this implies:
the only sound pseudo-label recipe is one that re-shapes pseudos to
match the focal-val distribution **before** training, not one that
clips outliers from a soundscape-shaped distribution. Sydorskyi's
own recipe (per `reference_bc2025_winners_writeups.md`) uses
per-class coverage gates that effectively enforce shape-match.
We've never built that gate properly.

Adds to precondition list:
- §14.14.7 (audit existing pipeline before borrowing techniques)
- T2.6 (treat +val as weak LB-predictive if >70 % added rows are for
  val-absent species OR BEST is a spike)
- §14.14.8 (single-fold cuDNN noise floor 0.03)
- §14.17.10.8 (audit aug_factor before multi-fold burns)
- §14.17.11.9 (capping aug_factor is necessary-not-sufficient — must
  also shape-match pseudo distribution to focal val before training)

---

## ⏸️ PICK UP HERE — previous (2026-04-29 00:30 local — Track A2-iter-1.5 KILLED, awaiting next-lever decision — SUPERSEDED by §14.17.12)

---

## 14.17.12 TopN N=1 BC2025 finishing post-proc — KILLED, v70 reverts (2026-04-29 ~16:30 local)

**Trigger.** Post-A2-iter-1.5 kill, user picked **TopN** from the open-lever
shortlist. Cheapest probe in the §14.17.11.7 menu — inference-only, single
notebook edit, one LB submission.

### 14.17.12.1 What was changed

Three isolated edits to `jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb`:

1. **Cell 3 (CFG):** added `CFG["topn_n1_enabled"] = True` flag. One-line
   revert if it regresses.
2. **Cell 14 (helpers):** added `topn_n1_scale(probs, n_windows)` —
   `out[a,t,c] = prob[a,t,c] * max_t' prob[a,t',c]`. Verified against
   formula in `reference_bc2025_winners_writeups.md` line 34.
3. **Cell 18 (post-proc, cell index 42):** added Step 6 application after
   Step 5 (per-class threshold sharpening), gated on the CFG flag.

Backup: `birdclef2026-protossm-postproc.ipynb.bak_pre_topn`.

### 14.17.12.2 Stack ordering — stack, not replace

Existing Cell 18 stack (per §14.14.7 audit): P1 per-taxon T → P2 file-level
top-k=2 → P3 rank-aware (power=0.4) → P11 conf-modulated delta (α=0.20) →
P13 per-class threshold sharpening.

P3 at power=0.4 is mathematically the soft variant of TopN N=1. Adding TopN
at the end **stacks** rather than replaces — combined effective power is
roughly file_max^1.4 (P3's file_max^0.4 multiplied by post-sharpening file_max^1.0
from TopN). Risk: overcompression of dynamic range, similar shape to v54 T1.1+T1.2
kill (LB 0.919 from over-smoothing).

Mitigation: §14.14.7 reverted overshoots cleanly. If LB ≤ 0.930 here, flip flag
and push v70 — no retrain needed.

### 14.17.12.3 Realistic LB envelope

Sydorskyi's BC2025 ablation (`reference_bc2025_winners_writeups.md`):
- Baseline 0.917 (post pseudo iter 2) → +TopN N=1 → 0.918 / 0.924 priv (+0.001 / +0.007)

That delta was on a stack **without** P3 power=0.4 already applied. Our pipeline
already has the soft variant, so the marginal lift here should be smaller than +0.001.
Best-case envelope: LB 0.932 (just above v62 floor). Worst-case: LB ~0.920 from
overcompression. Most likely: noise-band 0.929–0.933.

This is consistent with the user's framing — TopN is the cheapest probe, not the
swing lever. The swing lever queue remains L2-redux 819K-rec → arch swap.

### 14.17.12.4 LB result — KILLED

- v69 pushed 2026-04-29 ~15:15 local.
- v69 LB (Public): **0.925** vs v62 baseline 0.931 → **−0.006** regression.
- Decision rule (§14.17.12.5 LB ≤ 0.930) → revert. Set
  `CFG["topn_n1_enabled"] = False` (Cell 3) with kill comment, pushed **v70**
  ~16:30 local. Production restored to v62-equivalent state at LB 0.931 floor.
- Backup of v69 state: `birdclef2026-protossm-postproc.ipynb.bak_v69_topn_killed`.
- Kept the `topn_n1_scale` helper + Step 6 application in code (flag-gated off).
  Easier to point future runs at git history; cheaper than re-deriving the
  function.

### 14.17.12.5 Why it regressed — overcompression hypothesis confirmed

§14.17.12.2 flagged the risk: P3 at power=0.4 is the soft variant of TopN N=1.
Stacking TopN N=1 on top of it yielded combined effective power ~1.4. The
sensitivity is non-linear — at power=0.4 the file-max scale gently suppresses
uncertain files; at power=1.0+ the same scale crushes per-class evidence in any
file where the class doesn't fire hard at least once. For a 234-class macro-AUC
where many classes have only a handful of true positives across all test files,
this aggressively compresses the score range and hurts ranking on tail classes.

**Same shape as v54 T1.1+T1.2 kill** (LB 0.919 from over-smoothing, see
§14.11.6). Both kills are instances of the same generalized failure: borrowing
a winner-writeup post-proc lever and stacking it on a pipeline that already
implements a soft variant of the same operation, without first auditing whether
the new lever is additive or replacement-shaped.

**Sydorskyi's +0.001 delta was on a stack without P3.** We had P3 already.
Replacement was the right framing, not stacking. (Replace-mode probe, i.e.
setting `CFG["rank_aware_power"] = 1.0` and disabling the new TopN step, was
listed as a follow-up at edit time — but at this point we've burned the LB
slot and the lesson is the lever family is exhausted on this stack: any
file-max scale at power > 0.4 is compressing tail-class evidence.)

### 14.17.12.6 Decision rules (executed)

Per §14.17.5 LB envelope rules — actual:
- ~~**LB ≥ 0.940** → ship~~ — N/A
- ~~**LB 0.932–0.939** → neutral within noise~~ — N/A
- ✅ **LB ≤ 0.930** → revert. Done at v70 push (~16:30 local).

### 14.17.12.7 Lesson — generalizable

**Pre-existing soft-variant audit before borrowing post-proc levers.** Before
adding any new file-axis or class-axis scaling to the post-proc stack, check
whether an existing step already implements a soft variant of it. If yes,
the integration is replacement, not stack. If unclear, run the new lever
in **replace** mode first (set the existing knob to its identity value),
not in stack mode. Adds to:

- §14.14.7 (audit existing pipeline before borrowing techniques)
- T2.6 (treat +val as weak LB-predictive if >70 % added rows are for
  val-absent species OR BEST is a spike)
- §14.14.8 (single-fold cuDNN noise floor 0.03)
- §14.17.10.8 (audit aug_factor before multi-fold burns)
- §14.17.11.9 (capping aug_factor is necessary-not-sufficient)
- §14.17.12.7 (audit soft-variant overlaps before stacking post-proc levers)

### 14.17.12.8 Open levers — post-TopN-kill

The §14.17.11.7 list, minus TopN N=1:

1. **L2-redux 819K-rec pretrain (§14.17.6).** Recommended next. Heavy
   engineering, biggest BC2025 ablation delta (+0.046). Not blocked by
   §14.17.11/§14.17.12 findings — pretrain is upstream of all post-proc
   stacking concerns.
2. **Arch swap NFNet-L0 / EffNetV2-S (§14.17.6).** Cheaper, speculative.
3. **SoftAUCLoss (Babych).** Defer — material code change.

### 14.17.12.9 Don'ts — updated

- Do **not** retry TopN N=1 in replace mode (`rank_aware_power=1.0`,
  topn_n1 off) without first re-auditing the entire post-proc stack for
  other soft-variant overlaps. Single-knob LB probes on a stack with
  unknown overlap structure are wasteful.
- Do **not** restart any L1/L2/D3/C2/B2/P5/P12/A2-iter-1/A2-iter-1.5/TopN
  lever. Kill family span: pseudo-mix-during-finetune (6 kills) +
  post-proc lever stacking (2 kills, T1.1+T1.2 and TopN).
- Do **not** push another post-proc-only LB probe at all unless first
  paired with a hard-coded re-audit of which existing steps would shadow
  or compose with the new step.

---

---

## 14.17.13 Baseline drift detected — production floor is 0.926, not 0.931 (2026-04-29 ~17:00 local)

**Trigger.** v70 LB landed at **0.926**, not the 0.931 expected from the
"v70 = strict revert to v62-equivalent" claim in §14.17.12. Investigation
launched (path A: re-push v62-snapshot to bisect drift).

### 14.17.13.1 What we found — by diff, not by LB push

Did not push v71. Diffed the snapshots instead:

- `bak_pre_topn` (frozen 2026-04-29 15:10, the runtime code path of v70 since
  `topn_n1_enabled=False`) is **byte-identical** to
- `bak_pre_b2_cell37b` (frozen 2026-04-27 10:36, the post-B2-revert state).

Earlier in the chain:
- `bak_pre_b1_revert` (2026-04-24 23:18, snapshot just before v62 was
  finalized) differs from `bak_pre_b2_cell37b` only by:
  - One comment block: 5-line "B1 sweep probe #2" → 4-line "B1 sweep closed"
  - One literal: `CFG["b1_frozen_weight_submit"] = 0.00 → 0.10`
- **Zero code-path changes.**

So v70 ran the exact v62 logic. Same code, different score.

### 14.17.13.2 What this rules in / out

| Hypothesis | Status |
|---|---|
| Notebook code drift between v62 and v70 | **Ruled out** by the diff above |
| Submit-mode retraining non-determinism | Most likely — ProtoSSM `oof_n_splits=3` × 30 epochs + ResidualSSM 20 epochs both retrain at submit time, sub-model seeds not all pinned |
| External Kaggle dataset version drift | Plausible — `brucewu1200/birdclef-2026-cvlb-assets-0911` and `jaejohn/perch-meta` are owned by other users and could have updated since Apr 24 |
| Kaggle Public-LB scoring variance on identical submissions | Plausible at ±0.001–0.002 |

The first three combine to a realistic **single-submission LB noise floor of
±0.005**, not the ±0.001–0.002 we'd been implicitly using.

### 14.17.13.3 Implications

1. **New production floor: 0.926.** All future gates use 0.926, not 0.931.
2. **§14.17.12 verdict needs revision.** TopN N=1 was −0.001 from the real
   floor 0.926, not −0.006 from a phantom 0.931. Within the new ±0.005 noise
   band — **uninformative**. The kill stands (no upside, real downside if
   any), but the "combined effective power 1.4 overcompressed" framing was
   the wrong cause attribution.
3. **Past kill verdicts may be re-readable.** Several recent kill records
   were calibrated against the 0.931 floor. Where the kill margin was ≤0.005
   below 0.931 (i.e. LB 0.926–0.930), the result was actually within noise
   not a true regression. Examples in scope: T1.3 min-reduce 0.925, T2.6
   BC2025 0.927, P8 5-fold 0.928, B1 w=0.00 0.928, B2 1-fold 0.928. Don't
   un-kill them — none were +deltas — but recognize the "−0.003 to −0.006"
   readings were ambiguous, not clear regressions.
4. **Single-submission LB probes are noisier than we treated them.** A single
   probe at LB ±0.005 of floor is **not informative**. To probe a lever with
   genuine confidence, either (a) push two consecutive identical-config
   submissions and average, or (b) require the lever to lift LB by ≥0.005
   to call it a real win.
5. **Gap to leader is +0.025, not +0.020.** Leader 0.951 vs new floor 0.926.
   Per `project_lb_gap.md` updates needed.

### 14.17.13.4 What we did NOT do

- **Did not push v71.** The diff already proved the notebook code is
  byte-identical to v62. Pushing would have been redundant — saved 1 LB slot.
- **Did not seed-pin ProtoSSM/ResidualSSM submit-mode retraining.** That's
  a real but lower-priority action (would tighten future LB probes by maybe
  ±0.003) and changes inference timing. Defer until we have a non-trivial
  positive lever to validate at sub-noise-floor resolution.

### 14.17.13.5 Lesson — generalizable

**Re-LB-probe baseline before gating new levers against it.** When the last
LB probe of the production state is more than ~5 days or 5+ unrelated
notebook changes ago, the baseline is stale. Re-confirm it (or use a recent
identical-config probe) before reading any new lever's Δ. Adds to:

- §14.14.7 (audit existing pipeline before borrowing techniques)
- T2.6 (treat +val as weak LB-predictive if conditions match)
- §14.14.8 (single-fold cuDNN noise floor 0.03 — *training-side* noise)
- §14.17.10.8 (audit aug_factor before multi-fold burns)
- §14.17.11.9 (capping aug_factor is necessary-not-sufficient)
- §14.17.12.7 (audit soft-variant overlaps before stacking post-proc levers)
- §14.17.13.5 (re-LB-probe baseline before gating new levers; single-submission
  LB noise floor is ±0.005, not ±0.001-0.002)

### 14.17.13.6 Open levers — unchanged in priority

The kill-family conclusion of §14.17.12.8 still holds, just with corrected
floor:

1. **L2-redux 819K-rec pretrain (§14.17.6).** Recommended next. Heavy
   engineering, biggest BC2025 ablation delta (+0.046 in their table).
   Not blocked by any kill family — pretrain is upstream of post-proc and
   pseudo-mixing concerns. Expected delta on our stack: indeterminate but
   plausibly large (+0.01 to +0.04 envelope).
2. **Arch swap NFNet-L0 / EffNetV2-S (§14.17.6).** Cheaper, speculative.
3. **SoftAUCLoss (Babych).** Defer — material code change.

---

## ⏸️ PICK UP HERE — previous (2026-04-29 ~17:00 local — drift finding, L2-redux is next — SUPERSEDED by §14.17.14)

---

## 14.17.14 L2-redux scope — 819K-rec / 7,489-species in-domain pretrain (2026-04-29 ~17:30 local)

**Status: SCOPING ONLY, not greenlit.** Per §14.10's draft-then-go/no-go
pattern, return for explicit approval before launching any download or
training. Heavy compute commitment (~5–10 days GPU on this machine) and
significant disk pressure (~50–80 GB).

### 14.17.14.1 Why now, why this lever

Per `reference_bc2025_winners_writeups.md` Sydorskyi ablation table,
in-domain pretrain on 819,032 Xeno-Canto recs / 7,489 species lifted EffNetV2-S
LB by **+0.046** (0.835 → 0.881). That's the largest single-lever delta in
the table — bigger than pseudo-iter-1 (+0.027) or pseudo-iter-2 (+0.009).

The killed L2 (§14.10) used ~28K recs / ~206 species (BirdCLEF-2025 only)
and failed fold-0 gate at 0.6802. Memory `project_l2_killed.md`'s post-
mortem: structural focal→soundscape + species mismatch, NOT val leakage.
The prescribed fix is the 30× larger corpus (Sydorskyi's recipe) — strict
superset of L2's data plus 36× more species coverage. Not blocked by any
of the active kill families (pseudo-mix-during-finetune, post-proc-stacking).

### 14.17.14.2 What we already have on disk

- `four_track/src/pretrain_a1_2025.py` (373 lines) — fork of `train_a1.py`'s
  mel pipeline, model, and loss. Built for L2 attempt 1; **direct fork
  candidate** for L2-redux (replace data layer, scale up).
- `four_track/scripts/pretrain_a1_2025.sh` — nohup launcher template.
- `data/raw/birdclef_2025/` (24 GB) — full BirdCLEF-2025 focal corpus
  (~28K recs / ~206 species). One sub-corpus of the 819K target.
- `four_track/src/train_a1.py` — production EffB0 trainer with `--init-from`
  flag already wired in for finetune-from-checkpoint.
- 923 GB free on `/dev/nvme0n1p2` — enough for the full corpus.
- `four_track/data/processed/union_2025_2026_classes.json` — old ~400-class
  union from L2 attempt 1; will be superseded by the 7,489-species L2-redux
  union.

### 14.17.14.3 What needs sourcing

Sydorskyi's GitHub `VSydorskyy/BirdCLEF_2025_2nd_place` is the authoritative
source for the corpus assembly recipe. Their corpus = aggregation of:
- All historical BirdCLEF train-audio releases (BC2021 / BC2022 / BC2023
  / BC2024 / BC2025), de-duplicated by xeno-canto recording ID.
- Xeno-Canto bulk download covering ~7,489 species (everything they could
  pull at the time).
- iNaturalist research-grade audio for additional non-Aves coverage
  (insects/amphibians) — relevant for our 234-species 2026 set which
  includes Insecta/Amphibia/Mammalia/Reptilia.

**Open task: audit Sydorskyi's repo** for (a) exact species list, (b)
download script(s), (c) preprocessing format (sample rate, clip length,
duration cap). Likely 2-4 hours of investigation before any data movement.

### 14.17.14.4 Disk + compute envelope (rough)

- **Disk**: 819K recs × 30s avg × 32 kHz mono fp32 raw ≈ 100 GB. As ogg
  vorbis @ ~50 kbps ≈ 25 GB. Sydorskyi probably stored as ogg per BirdCLEF
  convention. **Plan for 30–80 GB depending on format.** Plenty of room.
- **Download time**: depends on whether prior-year BirdCLEFs are accessible
  via Kaggle CLI (yes, all five are public-archive comps) and Xeno-Canto
  bulk via their API (yes, but rate-limited — could take 2-4 days at default
  rate limits). **Open task to estimate.**
- **Pretrain time**: 819K recs / batch_size 64 ≈ 12.8K steps/epoch × 50 epochs
  = 640K steps. EffB0 on this GPU runs ~10 steps/sec at our config = 64K seconds
  = **~18 hours per 50-epoch pretrain run (single GPU)**. Plus data-loading
  overhead. **Realistic envelope: 1-2 days of pretrain wall-clock.**
- **Finetune time**: matches current 4-fold A1 training (~22 h per memory's
  past 5-fold runs). Add ~1 day.
- **Total**: download 1-3 days + pretrain 1-2 days + finetune 1 day = **~3-6
  days end-to-end** if execution is clean. Realistic with debugging: ~1 week.

### 14.17.14.5 Phased plan (gate at every phase boundary)

Phase 1 (~½ day, no GPU): **corpus audit + species mapping**
- Read Sydorskyi's repo prep scripts. Document the exact 7,489-species list
  and which sources each comes from.
- Map our 234-species BC2026 list against the 7,489 — confirm coverage.
  Expected: 100% Aves coverage; partial Insecta/Amphibia/Mammalia/Reptilia
  coverage. The killed L2 was Aves-only — L2-redux must include the
  non-Aves taxa.
- Write `data/processed/l2_redux_species_list.json` (cached).
- **Gate**: if non-Aves coverage of our 234 species < 80%, the L2-redux
  envelope shrinks; reconsider scope before downloading.

Phase 2 (~1-3 days, network only): **bulk download**
- BC2021–BC2025 train_audio via Kaggle CLI to
  `data/external/birdclef_history/`.
- Xeno-Canto bulk via their API (or Sydorskyi's pre-packaged dump if
  available) to `data/external/xenocanto_bulk/`.
- iNat audio via Sydorskyi's script if non-Aves needed.
- **Gate**: if total size > 100 GB, stop and reassess (we have 923 GB but
  shouldn't blow 10% of free space without confirmation).

Phase 3 (~½ day, no GPU): **preprocess + manifest**
- Resample to 32 kHz mono if not already. Cap at first 30 s per clip
  (Sydorskyi's recipe; first 60 s for rare classes).
- Build a single CSV manifest with (path, primary_label, source).
- Stratified 95/5 train/val split by primary_label.
- **Gate**: smoke test the data loader on 10 batches before committing
  to the full pretrain.

Phase 4 (~1-2 days, GPU): **pretrain**
- Fork `src/pretrain_a1_2025.py` → `src/pretrain_l2_redux.py`.
  - Replace 2025 dataset class with the bulk corpus dataset.
  - Set head size to 7,489 (BCE union loss; head will be discarded
    at finetune).
  - 50 epochs, batch_size 64, lr 1e-3 with cosine warmup.
  - Save backbone-only ckpt at end of every 10 epochs (not full state)
    so finetune doesn't have to load a 7,489-class head.
- Per CLAUDE.md GPU memory hygiene: `gc.collect()` + `torch.cuda.empty_cache()`
  every epoch.
- **Gate after epoch 5**: if backbone embedding probe on val gives < random
  baseline, kill the run.

Phase 5 (~1 day, GPU): **finetune**
- Reuse `src/train_a1.py` with `--init-from
  models/a1_l2_redux/a1_l2_redux_backbone_e50.pt`.
- 4-fold (folds 0,1,2,4 to match production); 25 epochs each;
  hybrid loss (matches the LB-0.931 baseline per
  `project_a1_baseline_loss_is_hybrid.md`).
- **Gate at fold-0 ep 25**: val_v2 macro-AUC must beat the current
  0.7414 hybrid gate by ≥0.005 (i.e. ≥0.7464). Below that, kill.

Phase 6 (1 LB submission): **LB probe**
- Build a new Kaggle dataset for the 4 finetuned ckpts.
- Update `kernel-metadata.json` to reference the new dataset.
- Push as next available kernel version.
- **Decision rule** (per §14.17.5, with new 0.926 floor):
  - LB ≥ 0.940 → ship; queue iter-2 (pseudo on top of L2-redux finetune).
  - LB 0.932-0.939 → real lever, +0.006 to +0.013. Ship as production.
  - LB 0.927-0.931 → within noise (±0.005). Repeat probe or kill if 2nd
    submission also sub-noise.
  - LB ≤ 0.926 → regression. Revert. Move to arch swap.

### 14.17.14.6 Realistic LB envelope on our stack

Sydorskyi got +0.046 on EffNetV2-S in their stack. Adjusters for our case:
- **Backbone**: We use EffNet-B0; their +0.046 was on EffNetV2-S. Smaller
  backbone benefits less from massive pretrain (paper-class result, ~50%
  of large-model delta is typical). Discount to **~+0.020-0.025**.
- **Existing baseline strength**: They went 0.835 → 0.881 (+0.046 on a
  weak baseline); we're at 0.926 (much stronger baseline). Marginal
  utility curve flattens — the easy gains of in-domain pretrain may
  already be captured by our existing PCEN + MixStyle + secondary-label
  + hybrid-loss stack. Discount further to **~+0.010-0.020**.
- **Pseudo-label kill family memo**: We've already proven that levers
  upstream of finetune ("set the prior cleanly") behave differently from
  pseudo-mix-during-finetune (which 6/6 killed). L2-redux is upstream-side,
  so it's NOT in the kill family — but it's also not a guaranteed delta.

**Realistic envelope: +0.005 to +0.020 LB.** Best case: +0.020 → 0.946 (gold
zone). Likely case: +0.010 → 0.936 (real lever). Worst case: noise band
(±0.005). Failure case: regression from over-pretrain on out-of-domain audio.

### 14.17.14.7 What's NOT in this scope (deliberately)

- **Iterative pseudo-labeling on top of L2-redux finetune.** Sydorskyi's
  +0.046 was a single pretrain step; their +0.027 pseudo-iter-1 was a
  follow-up. Don't bundle. Probe L2-redux first, decide on iter-2 after.
- **Arch swap to EffNetV2-S or NFNet-L0.** Bigger backbone amplifies the
  L2-redux delta but doubles the engineering scope. Defer to "L2-redux v2"
  if v1 succeeds.
- **SoftAUCLoss (Babych).** Material code change to the loss path. Defer
  unless L2-redux underdelivers and we need a second swing.

### 14.17.14.8 Open questions for the user (go/no-go)

Three decisions before kicking off:
1. **Approve L2-redux scope** as the next direction, or override toward
   arch swap (cheaper) or another lever?
2. **Compute commitment**: ~1 week GPU. OK to commit, or constrained?
3. **Kaggle deadline pressure**: how many days remaining on the BC2026
   competition? If <2 weeks, L2-redux is risky (1 week to know if it
   works at all); if 4+ weeks, well within budget.

If approved, Phase 1 is the first action — corpus audit on Sydorskyi's
repo, ~½ day, no compute commitment, fully reversible. Don't start
Phase 2 (download) without explicit go on Phase 1 results.

---

## ⏸️ PICK UP HERE — previous (2026-04-29 ~17:30 local — L2-redux scope drafted — SUPERSEDED by §14.17.15)

---

## 14.17.15 Aves-only L2-redux execution plan — Phase 1 audit complete (2026-04-29 ~18:00 local)

**Status: SCOPE LOCKED, APPROVED.** Phase 1 corpus audit completed via
general-purpose agent. User greenlit Aves-only reframe at 18:00 local.

### 14.17.15.1 Phase 1 audit findings

Sydorskyi's `VSydorskyy/BirdCLEF_2025_2nd_place` GitHub repo audit:

- **Real species count: 7,591** (extended) / 7,544 (smaller). Memory note
  said 7,489 — off by ~100. Auth source: `bird2int_pretraintrain_*.json`
  in `bird-clef-2025-pretrained-metadata` Kaggle dataset.
- **"819,032 recordings" claim NOT verified** in repo. Manifest CSVs not
  checked in; figure must be from CEUR paper.
- **Pretrain corpus is functionally Aves-only.** Of 7,591 classes:
  - 7,531 eBird-style codes (Aves)
  - 60 numeric iNat IDs (non-Aves)
- **Match against our 234 BC2026 species:**
  - Aves: 160 / 162 (excellent, 99 %)
  - Amphibia: 2 / 35 (5 %)
  - Insecta: 0 / 28 (0 %)
  - Mammalia: 1 / 8 (13 %)
  - Reptilia: 0 / 1 (0 %)
  - **Non-Aves total: 3 / 72 (4 %)**
- **Sources:** BC2021/2023/2024/2025 train_audio (Kaggle) + Xeno-Canto
  bulk via `download_all_xeno_canto.py` over IOC v12 list + iNat sounds
  + CSA Humboldt + NZ DOC. **No bulk-download script** — Reconstruct.
- **Format on disk:** HDF5 with raw 32 kHz mono float waveforms, no length
  cap (the "first 30 s" claim from earlier notes is a paper-side statement,
  not implemented in their public preprocessing).
- **Pretrain config:** 50 ep, bs 64, focal-BCE, single 7,591-class head,
  AdamW 2.5e-4 (EffNetV2-S) or RAdam 1e-3 (NFNet-L0), CosineAnnealingWarmRestarts,
  SpecAug + RandomFiltering + BG mixup, RTX 4090 ≥ 10 GB VRAM.

### 14.17.15.2 Reframe — Aves-only L2-redux

**Accepted:** the 72 non-Aves BC2026 species get only encoder-transfer
benefit (no pretrain head supervision). At finetune time, the 234-class
head is re-initialized; the encoder weights are what L2-redux is buying.
The bet: a 7,591-class encoder trained on Aves diversity learns generic
acoustic features (onset, harmonic structure, time-frequency patterns)
that transfer to non-Aves classes well enough to not regress vs. ImageNet
init.

**Risk:** structurally identical to the killed L2 (28K rec / 206 species
fold-0 0.6802 species-mismatch failure). The bet is that scaling 36× in
species and 30× in recordings rescues the species-mismatch by giving the
encoder enough capacity to learn task-agnostic features.

### 14.17.15.3 Updated phased plan (Aves-only)

Skipping iNat / CSA Humboldt / NZ DOC — marginal benefit, large
complexity. Just BC-historic + Xeno-Canto bulk for the 7,591 Aves species.

**Phase 2a (~½ day, network only): BC-historic download.**
- BC2021, BC2023 (parts 2/3 datasets), BC2024, BC2025 train_audio via
  Kaggle CLI. BC2025 already on disk (24 GB).
- Estimated additional disk: ~150 GB (BC2021 ~50 GB, BC2023 ~50 GB,
  BC2024 ~30 GB).
- Stage at `data/external/birdclef_history/`.
- **Gate:** if any of the BC competitions blocks the download (rules
  acceptance needed), pause and resolve before proceeding.

**Phase 2b (~½ day, GPU): pipeline smoke pretrain on BC-historic only.**
- ~150 K Aves recordings; smoke-test the pretrain pipeline end-to-end
  before committing to the multi-day XC bulk download.
- Run 5 epochs only.
- **Gate:** if encoder probe on BC2026-val gives < random baseline, the
  pipeline is broken — fix before Phase 2c. Also gates "is L2-redux
  going to work at all?" on the smaller corpus before paying the XC
  bulk download cost.

**Phase 2c (3-7 days, network only): Xeno-Canto bulk for 7,591 species.**
- Adapt Sydorskyi's `download_all_xeno_canto.py` to **only** the 7,591
  species in their bird2int mapping (skip the IOC v12 superset of ~11 K).
- Rate-limited; expect 3-7 days of wall-clock.
- Estimated disk: 300-500 GB raw waveform; **40-80 GB if we store as
  ogg vorbis instead of raw HDF5** (deviation from Sydorskyi's format
  that we make for disk economy on this machine).
- Stage at `data/external/xenocanto_bulk/`.
- **Gate:** if download exceeds 200 GB or 7 days, pause and reassess.

**Phase 3 (~½ day, no GPU): preprocess + manifest.**
- Resample to 32 kHz mono. Convert ogg → cached mel features (we cache
  to disk so the dataloader doesn't re-decode every batch — Sydorskyi's
  HDF5-of-waveforms approach is GPU-bottlenecked at the dataloader on
  our machine).
- Cap clip duration at 30 s (Sydorskyi's paper-side claim — applies it
  in practice via random-crop, we apply it at preprocessing for disk).
- Build single CSV manifest.
- Stratified 95/5 train/val split by primary_label.
- **Gate:** smoke-test on 10 batches before Phase 4 launch.

**Phase 4 (1-2 days, GPU): full pretrain.**
- Fork `src/pretrain_a1_2025.py` → `src/pretrain_l2_redux.py`.
- 50 epochs, bs 64, focal-BCE over 7,591-class head.
- Cosine warm restarts, T_0 = 50ep × len(train).
- AdamW lr 2.5e-4 (matching Sydorskyi's EffNetV2-S config).
- Save backbone-only ckpt every 10 epochs.
- Per CLAUDE.md: `gc.collect()` + `torch.cuda.empty_cache()` every epoch
  AND every fold.
- **Gate at epoch 5:** if backbone embedding probe on val < random
  baseline, kill.

**Phase 5 (~1 day, GPU): finetune.**
- Reuse `src/train_a1.py --init-from
  models/a1_l2_redux/a1_l2_redux_backbone_e50.pt`.
- 4-fold (folds 0,1,2,4 to match production), 25 epochs each, hybrid
  loss (matches LB-0.926 baseline).
- **Gate at fold-0 ep 25:** val_v2 macro-AUC must beat 0.7414 by ≥0.005
  (≥0.7464). Below that, kill.

**Phase 6 (1 LB submission): LB probe.**
- New Kaggle dataset for the 4 finetuned ckpts.
- Update `kernel-metadata.json`. Push next available kernel version.
- Decision rule (against new floor 0.926, ±0.005 noise band):
  - LB ≥ 0.940 → ship; queue iter-2 (pseudo on top).
  - LB 0.932-0.939 → real lever, ship.
  - LB 0.927-0.931 → within noise. Repeat probe; kill if 2nd sub-noise.
  - LB ≤ 0.926 → regression. Revert. Move to arch swap.

### 14.17.15.4 LB envelope on Aves-only stack

Sydorskyi's +0.046 was on a **100 % Aves** comp (BC2025 Colombia). For
BC2026 Pantanal where 31 % of species are non-Aves and get no pretrain
benefit:

- **Encoder transfer for Aves (162 species):** plausibly ~+0.03 to
  +0.04 of BC2025-class delta on the Aves subset of macro-AUC.
- **Encoder transfer for non-Aves (72 species):** smaller, since the
  pretrain didn't see this taxonomic structure. Best case: encoder
  features are generic enough to be neutral. Worst case: encoder is
  Aves-biased and *hurts* non-Aves macro-AUC.
- **Net macro-AUC on 234 classes:** depends on which side wins. If
  Aves +0.04 and non-Aves −0.005 each: weighted avg = (162/234)×+0.04
  + (72/234)×(−0.005) = +0.026, → LB ~0.952 (gold zone). If non-Aves
  hits −0.02: weighted avg = +0.022, → LB ~0.948 (still gold). If
  non-Aves catastrophic at −0.10: weighted avg = −0.003, → LB ~0.923
  (regression).

**Realistic envelope: −0.005 to +0.025.** Wide band, structurally driven
by the Aves/non-Aves trade-off. The Phase 5 fold-0 gate (≥0.7464 val_v2)
catches the regression case before LB push.

### 14.17.15.5 What's NOT in this scope (still deferred)

- **iNat / CSA Humboldt / NZ DOC** sub-corpora. Skipped for engineering
  economy; ~10-15 % of Sydorskyi's recordings, marginal benefit on Aves
  diversity.
- **iNat insect / amphibian pretraining for non-Aves.** Different
  pretrain corpus, post-L2-redux v1 if the non-Aves regression is real.
- **Iterative pseudo-labeling on top of L2-redux finetune.** Defer; Phase
  6 ships first, iter-2 if results justify.
- **Arch swap** (NFNet-L0 / EffNetV2-S). Defer; current scope is
  Aves-only L2-redux on EffNet-B0.

### 14.17.15.6 Immediate next action — Phase 2a launch

Phase 2a (BC-historic download) is the first concrete action:

1. Verify Kaggle CLI competition rules acceptance for BC2021/2023/2024
   (BC2025 already accepted).
2. `nohup`-wrapped download script per CLAUDE.md conventions, logs to
   `four_track/log/`.
3. Disk monitor: stop if `/dev/nvme0n1p2` use exceeds 90 %.

Files to create:
- `four_track/scripts/l2_redux_phase2a_download_bc_historic.sh`
- (later) `four_track/src/l2_redux/download_xenocanto.py` (Phase 2c)
- (later) `four_track/src/l2_redux/preprocess.py` (Phase 3)
- (later) `four_track/src/pretrain_l2_redux.py` (Phase 4)

---

## ⏸️ PICK UP HERE — previous (2026-04-29 ~18:00 local — Phase 2a queued — SUPERSEDED by §14.17.15.7)

---

## 14.17.15.7 Phase 2a complete — BC-historic on disk (2026-04-29 ~19:03 local)

**Wall-clock: 11.5 min** (18:51 → 19:03). Previous estimate 1.5-2 h.
Kaggle bandwidth was generous (~50 MB/s avg).

**Final corpus state at `data/external/birdclef_history/`:**

| Source | .ogg files | Size | Notes |
|---|---|---|---|
| BC2023/train_audio | 16,941 | 5.0 GB | from `kaggle competitions download -c birdclef-2023` |
| BC2024/train_audio | 24,459 | 7.4 GB | from `kaggle competitions download -c birdclef-2024` |
| BC2025/train_audio (`data/raw/birdclef_2025/`) | ~28 K | ~24 GB | from L2 attempt 1 (still on disk) |
| **Total focal recs on disk** | **~70 K** | **~36 GB** | Aves only |

Disk usage unchanged at 75% (zips deleted post-extraction). 910 GB free.

**BC2021 absence confirmed expected.** File listing showed only
`train_soundscapes/` + `test_soundscapes/`; no `train_audio/`. BC2021 used
Xeno-Canto-only training data and contributes 0 focal-clip recordings to
Phase 2a. Coverage of BC2021 species comes via Phase 2c XC bulk if at all.

**Implication for Phase 2b:** Total focal corpus is ~70 K, not the 150 K
originally estimated in §14.17.15.3. Phase 2b smoke pretrain should still
work — corpus is large enough to give the encoder real signal — but the
smoke gate's "encoder probe on BC2026 val" should be read as a **lower
bound** on the post-Phase-2c (with XC bulk) performance. Don't kill the
project on a marginal Phase 2b result; Phase 2c likely brings 5-10× more
data.

**Idempotency held.** Script's existence-check on `<comp>/train_audio/`
means a re-launch is a no-op; safe to leave the script in place.

### 14.17.15.7.1 Phase 2b approach decision

Three structural choices for the Phase 2b smoke pretrain script:

**Option A: fork `src/pretrain_a1_2025.py`** (373 lines, BC2025-only).
Add multi-source path resolution + union class list across the three
comps. ~150 lines of changes. Risk: leaves the existing L2 script in
an ambiguous state ("which year does it train on?"). Reject.

**Option B: write fresh `src/pretrain_l2_redux.py`.** Reuse imports
(model, dataset_a1 mel pipeline) but build a clean multi-source dataset
class. ~250 lines new. Cleaner separation; existing pretrain_a1_2025.py
stays untouched as the L2-killed baseline. **Recommended.**

**Option C: skip Phase 2b, go direct to Phase 2c.** Save ~½ day GPU but
risk burning the multi-day XC bulk download on a broken pipeline.
Reject — Phase 2b's whole purpose is risk-reduction.

### 14.17.15.7.2 Phase 2b spec (Option B)

1. **Union species list** — `data/processed/l2_redux_aves_species.json`.
   Build by sorted-union of `primary_label` from BC2023/train.csv +
   BC2024/train.csv + BC2025/train.csv. Cache for Phase 4 reuse.
2. **Multi-source dataset class** — reads (root_dir, comp_id) tuples,
   loads .ogg via the parent project's `utils.load_audio`, returns
   one-hot over union class list. Same mel pipeline as
   `src/dataset_a1.py`.
3. **Train script `src/pretrain_l2_redux.py`** — single-split (95/5
   stratified by primary_label), focal-BCE over union head,
   AdamW 2.5e-4, cosine warm restarts T_0 = 5ep × len(train), bs 64,
   5-sec random crop, MixStyle p=0.5, SpecAug (matches Sydorskyi).
   Save backbone-only ckpt at end of each epoch.
4. **Encoder probe** — after smoke pretrain, freeze backbone, train a
   linear 234-class head on BC2026 val_v2 train fold for 5 epochs,
   evaluate macro-AUC on val_v2 holdout. Compare against ImageNet-init
   probe baseline (separate run).
5. **Smoke gate** — Phase 2b passes iff:
   - Pretrain reaches focal-BCE val < 0.20 by epoch 5 (sanity).
   - Encoder probe ≥ ImageNet-init probe by ≥0.01 macro-AUC.
   - No memory leaks / hangs (per `feedback_gpu_memory_hygiene_per_epoch.md`).

### 14.17.15.7.3 Files to create

- `src/pretrain_l2_redux.py` — main pretrain script (~250 lines)
- `scripts/pretrain_l2_redux_smoke.sh` — Phase 2b launcher
- `scripts/pretrain_l2_redux_full.sh` — Phase 4 launcher (later)
- `data/processed/l2_redux_aves_species.json` — cached union list
- `models/l2_redux/` — output dir for backbone ckpts

---

## ⏸️ PICK UP HERE — previous (2026-04-29 ~19:10 local — Phase 2b spec drafted — SUPERSEDED by §14.17.15.8)

---

## 14.17.15.8 Phase 2b complete — smoke pretrain + probe gate PASSED (2026-04-29 ~23:25 local)

**Phase 2b end-to-end:** smoke pretrain ran 5 epochs on the 70K-clip /
636-species BC2023+2024+2025 Aves corpus, then a paired encoder linear
probe (ImageNet baseline vs L2-redux init) on BC2026 fold-0 train data
gated transfer. **Both halves clean. Gate cleared by 5.5×.**

### 14.17.15.8.1 Smoke pretrain — final (107 min wall-clock)

| Epoch | train_loss (focal-BCE γ=2) | val_roc_auc (n=460/636) | time |
|---|---|---|---|
| 1 | 0.0069 | 0.7200 | 24m 20s |
| 2 | 0.0024 | 0.8888 | 19m 31s |
| 3 | 0.0018 | 0.9312 | 19m 32s |
| 4 | 0.0014 | 0.9466 | 19m 36s |
| 5 | 0.0013 | **0.9515** ★ | 19m 36s |

Output: `four_track/models/l2_redux/l2_redux_best.pt` (17.5 MB —
backbone + 636-class head + union class list embedded).

Train_loss values are tiny because focal-BCE (γ=2) over 636-way
multi-label crushes the (1−pt)^2 weight on the 635 easy-negative
classes per sample. Val_roc_auc is the load-bearing metric. Author-grouped
GroupShuffleSplit gave 0 shared authors → val numbers are not leak-inflated.

### 14.17.15.8.2 Encoder linear probe — PASSED (~106 min wall-clock)

Two paired runs of 5-epoch head-only fine-tune on BC2026 fold-0 train
clips with frozen backbone, evaluated on val_v2 (1478 chunks /
train_soundscapes_labels.csv expert annotations).

ImageNet baseline (default timm pretrained backbone):

| Ep | train_loss | val_v2_auc |
|---|---|---|
| 1 | 0.0470 | 0.4918 |
| 2 | 0.0402 | 0.5124 |
| 3 | 0.0384 | 0.5332 |
| 4 | 0.0375 | 0.5377 |
| 5 | 0.0366 | **0.5473** ★ |

L2-redux init (loaded `l2_redux_best.pt`, dropped 636-class head keys,
reinit'd att_conv + cls_conv to 234):

| Ep | train_loss | val_v2_auc |
|---|---|---|
| 1 | 0.0427 | 0.5620 |
| 2 | 0.0341 | 0.5968 |
| 3 | 0.0326 | 0.5995 |
| 4 | 0.0321 | 0.6024 |
| 5 | 0.0315 | **0.6027** ★ |

**Verdict:** Δ = **+0.0554** ≥ +0.01 → **PASS**, gate cleared by 5.5×.

Both probes plateau cleanly (epoch 5 gain < 0.005 in each). The L2-redux
probe matched ImageNet's epoch-5 best AT EPOCH 1 and beat it by epoch 2.
This is strong-transfer territory per §14.17.14's table.

Logs: `four_track/models/l2_redux/probe_imagenet_log.json`,
`probe_l2redux_log.json`.

### 14.17.15.8.3 What this confirms vs leaves open

**Confirmed:**
- L2-redux pretrain teaches transferable bird-acoustic encoder features,
  not just within-corpus class memorization.
- Features survive the BC2023+24+25 → BC2026 Pantanal domain shift
  (different species set, different recording style, different geography).
- The probe protocol works (ImageNet baseline gave non-trivial signal,
  showing the head can extract some structure even from generic image
  features; the L2-redux delta is therefore a clean encoder-quality
  measurement).
- Frozen-encoder ceiling on this task is ~0.55 (ImageNet) / ~0.60
  (L2-redux). Production fine-tuned A1 hits ~0.7414. So **80% of the
  signal still comes from fine-tuning the encoder, not from frozen
  features.** Phase 5 is where the LB delta materializes.

**Not yet confirmed:**
- The actual LB delta after full Phase 5 fine-tune. Probe Δ +0.055 does
  NOT mean Phase 5 will give +0.055 LB. The relationship is monotonic
  but compressive — typical encoder-pretraining literature suggests
  ~30-50% of the frozen-probe Δ shows up at the fine-tuned LB level.
  Realistic envelope: **Phase 6 LB delta ~+0.015 to +0.030**.
- Whether the 800K/7,591-species Phase 4 corpus does even better than
  the 70K/636 smoke. Strongly likely yes (more data, more species
  diversity), but not measured.
- Non-Aves transfer. val_v2 is mostly Aves chunks; the 72 non-Aves
  classes are sparse here. Phase 5 finetune is where this matters.

### 14.17.15.8.4 Cleanup state

- Smoke ckpt `models/l2_redux/l2_redux_best.pt` kept (potential Phase 5
  probe ckpt if we ever want a Phase-2b-corpus-only finetune comparison).
- Probe logs kept under `models/l2_redux/probe_*_log.json`.
- Manifest cached at `data/processed/l2_redux_manifest.csv` (will be
  superseded by Phase 3 manifest after XC bulk merges in).
- Species cache at `data/processed/l2_redux_aves_species.json` (636
  species — will be superseded by Phase 3's 7,591-species list).
- Disk usage: 75 % on /dev/nvme0n1p2. 905 GB free.

### 14.17.15.8.5 Phase 2c approval

Phase 2b PASS justifies Phase 2c launch. Per §14.17.15.3 Phase 2c is
the big-ticket item:
- Xeno-Canto bulk download for 7,591 Aves species
- Estimated 5-7 days wall-clock at API rate limits
- Estimated disk 40-80 GB (with ogg vorbis storage instead of Sydorskyi's
  raw HDF5)
- Adapt Sydorskyi's `download_all_xeno_canto.py` to filter to just the
  7,591-species list

Detailed scope follows in §14.17.16.

---

## 14.17.16 Phase 2c scope — Xeno-Canto bulk download for 7,591 species (2026-04-29 ~23:30 local)

**Status: SCOPING ONLY, not greenlit.** Phase 2b PASS justifies
launching Phase 2c, but Phase 2c is a 5-7 day commitment that benefits
from one more round of explicit scoping before kick-off.

### 14.17.16.1 Prerequisites

Two artifacts to acquire before any download:
1. **Sydorskyi's 7,591-species list.** From their
   `bird-clef-2025-pretrained-metadata` Kaggle dataset (file
   `bird2int_pretraintrain_prev_comps_xc_alltaxonomy_csa_newzealand_XCshiro_nosmall10sp_and_2025_snipet11052025.json`,
   per Phase 1 audit). Pull via Kaggle CLI; ~5 MB.
2. **Audit overlap with our existing 636-species smoke corpus.** The
   smoke corpus's 636 species are already on disk; the 7,591 list is
   a strict superset. Want to confirm overlap (expect ~600 of 636 to
   appear in the larger list) so we don't re-download.

Output: `data/processed/l2_redux_full_species.json` (7,591 species)
plus `data/processed/l2_redux_xc_targets.json` (7,591 minus species
already covered by BC2023+24+25, ≈ 6,955 species to fetch).

### 14.17.16.2 Download mechanism

Sydorskyi's `download_all_xeno_canto.py` (in their GitHub repo) wraps
the `xenocanto` PyPI package. Iterates over IOC v12 (~11K species),
which is the broader superset they use. We adapt to:

1. Accept a species-list filter (instead of IOC v12 walk).
2. Per-species: query XC API for all recordings of that species,
   download up to N recordings per species (cap at e.g. 500 to avoid
   long-tail-species bias eating disk).
3. Resampling at download time: 32 kHz mono, ogg vorbis quality 4
   (~50 kbps). Per-recording duration cap: first 30 s for non-rare
   species, first 60 s for rare (< 20 recordings) — matches
   Sydorskyi's paper-side claim.
4. Resumable: skip species/recording combinations already on disk.
5. Throttle: 2 req/sec to avoid hitting XC rate limits.

Output dir: `data/external/xenocanto_bulk/<species_code>/<XCnnnnnn>.ogg`.

### 14.17.16.3 Disk + wall-clock budget

XC has ~700 K Aves recordings total across all species (their public
stats); after filtering to 7,591 species and capping 500 per species,
realistic estimate is **~400-600 K recordings**.

- **Disk** at 32 kHz mono ogg vorbis q4 with 30-60 s caps:
  - ~30 s × ~50 kbps = ~190 KB per recording
  - 500 K × 190 KB = **~95 GB**
  - With variation: **80-130 GB realistic**
  - Still well under 905 GB free, but a 10× of Phase 2a's footprint.

- **Wall-clock** at 2 req/sec sustained:
  - 7,591 species metadata fetches: ~1 hour
  - ~500 K recording downloads (most under 1 sec each at the throttle):
    500K / 2 req/sec = 250 K seconds = **~70 hours = ~3 days**
  - Plus retries / network blips / API hiccups: **realistic 4-6 days**
  - Risk: XC may rate-limit per-IP at lower rates than 2 req/sec; if
    we hit 429s, throttle drops to 1 req/sec → 6-8 days.

### 14.17.16.4 Disk-and-time gates

Two abort gates during the download:
- **Disk:** abort if `/home/swatson` use > 90 % at any per-species
  checkpoint.
- **Time:** abort if 7-day wall-clock cap is exceeded; finalize what
  we have and proceed to Phase 3 with partial corpus.

Both gates are recoverable — Phase 4 pretrain works on whatever
corpus is present in `data/external/xenocanto_bulk/`.

### 14.17.16.5 Files to create

- `four_track/src/l2_redux/fetch_species_list.py` — pull Sydorskyi's
  species JSON from Kaggle, write `l2_redux_full_species.json` and
  `l2_redux_xc_targets.json`.
- `four_track/src/l2_redux/download_xenocanto.py` — adapt Sydorskyi's
  download script to filter + resample + cap + resumable.
- `four_track/scripts/l2_redux_phase2c_download_xc.sh` — `nohup`
  launcher with disk + time gates, per CLAUDE.md conventions.

### 14.17.16.6 Open questions for the user (go/no-go)

1. **Approve Phase 2c launch?** It's a ~5-7 day commitment. Days remaining
   on BC2026 competition are the main external constraint — if < 2 weeks,
   Phase 2c is risky.
2. **Per-species recording cap.** 500 default, 100 for "low storage budget"
   variant, no-cap for "max-corpus" variant. Recommend 500 (matches
   Sydorskyi's intent, gives long-tail-resistant balanced corpus).
3. **Duration cap.** 30s for non-rare / 60s for rare matches Sydorskyi's
   paper claim. OK, or want different?

If approved, **Phase 2c.1 (species-list fetch + diff against existing 636)**
is the first concrete action — 10 min, no compute, fully reversible.
Don't start Phase 2c.2 (XC bulk download) without explicit go on Phase 2c.1
results.

---

## ⏸️ PICK UP HERE — previous (2026-04-29 ~23:30 local — Phase 2c API-route scope SUPERSEDED by §14.17.16.7)

---

## 14.17.16.7 Phase 2c.1 result + 2c.2 pivot to rohanrao bulk dataset (2026-04-30 ~00:30 local)

### 14.17.16.7.1 Phase 2c.1 — species-list fetch + diff (DONE)

Pulled Sydorskyi's bird2int JSON from his GitHub
(https://raw.githubusercontent.com/VSydorskyy/BirdCLEF_2025_2nd_place/main/data/bird2int_pretraintrain_prev_comps_xc_alltaxonomy_csa_newzealand_XCshiro_nosmall10sp_and_2025_snipet11052025.json)
— **NOT** the Kaggle metadata dataset (which only has manifest CSVs and
NPY splits, no species list JSON).

| | Count |
|---|---|
| Sydorskyi's full species list | 7,591 |
| Existing smoke corpus (BC23+24+25) | 636 |
| Overlap (already covered) | 622 |
| Sydor − existing → fetch via XC | 6,969 |
| **+ BC2026 revision-code adds (palhor3, strher2)** | **+2** |
| **Total Phase 2c.2 targets** | **6,971** |
| BC2026-Aves species in Sydor list | 160 / 162 (99 %) |
| BC2026-Aves species in existing 636 | 39 / 162 (24 %) |

The 2 BC2026-Aves missing entirely are revision-code drift — Sydor used
older eBird codes for the same species:
- `palhor3` (BC2026) ↔ `palhor2` (Sydor) — Pale-legged Hornero
- `strher2` (BC2026) ↔ `strher`  (Sydor) — Striated Heron

Outputs cached:
- `data/processed/l2_redux_full_species.json` (7,591 codes)
- `data/processed/l2_redux_xc_targets.json` (6,971 codes to fetch)
- `data/processed/ebird_taxonomy_v2021.csv` (16,753 species code→sciname,
  pulled from BC2024 train_audio.zip — 100 % coverage of our 6,971
  targets, except the lone non-Aves `weta` New Zealand insect).

### 14.17.16.7.2 Phase 2c.2 pivot — XC API v2 dead, rohanrao bulk path

Hit a hard blocker on the API route: **Xeno-Canto API v2 returns 404 with
"Xeno-canto API v2 is no longer available."** API v3 requires a per-account
API key (registration on xeno-canto.org) and has its own rate limits.
Sydorskyi's `xenocanto` PyPI wrapper has been removed; pip install fails
with "no matching distribution found."

Pivot: **`rohanrao/xeno-canto-bird-recordings-extended-{a-m,n-z}` Kaggle
datasets** are pre-organized as `<A-M|N-Z>/<ebird_code>/XC<id>.mp3`,
totaling 30 GB (18 + 11). 2020 snapshot, but covers ~10 K species and
should overlap our 6,971 by >95 %. eBird-code revisions since 2020 are
the residual mismatch surface; sciname-fallback mapping handles those
cases.

| | API v3 route (blocked) | rohanrao bulk pivot (chosen) |
|---|---|---|
| Status | needs API key, rate-limited | available now |
| Wall-clock | 4-7 days API throttling | ~30-60 min download + ~7 h local transcode |
| Disk | ~80-130 GB final | ~30 GB raw + ~80-130 GB transcoded (delete raw post-transcode) |
| Currency | 2026 (current) | 2020 snapshot (5 years stale) |
| Risk | API blocks / bans | minimal |

Pivot approved 2026-04-30 ~00:30 local.

### 14.17.16.7.3 Pivot phased plan

**Phase 2c.2a (~30-60 min, network):** Bulk download.
- `kaggle datasets download rohanrao/xeno-canto-bird-recordings-extended-a-m`
- `kaggle datasets download rohanrao/xeno-canto-bird-recordings-extended-n-z`
- Stage at `data/external/xenocanto_raw/` (then `unzip`).
- Disk impact: ~30 GB peak (zips) + ~30 GB extracted = ~60 GB peak; cleanup zips after extract.

**Phase 2c.2b (~6-8 h, local CPU + ffmpeg):** Filter + transcode.
- `src/l2_redux/transcode_xenocanto.py` (new):
  - Walks our 6,971-target list
  - Primary lookup: direct code match in rohanrao (`A-M/<code>/` or `N-Z/<code>/`)
  - Fallback: sciname-match for revision-drifted codes (e.g., `palhor3`
    not in rohanrao but `palhor2` is — same Furnarius leucopus)
  - For each matched mp3: ffmpeg → 32 kHz mono ogg vorbis q4, dur cap 30s
    (or 60s if species has < 20 recordings), output to
    `data/external/xenocanto_bulk/<target_code>/XC<id>.ogg`
  - Per-species recording cap 500 (matches §14.17.16 spec)
  - Resumable: skip files where target ogg already exists
  - Parallelizable via thread pool (4 workers default)

**Phase 2c.2c (~5 min, local):** Cleanup raw mp3 dump.
- Delete `data/external/xenocanto_raw/` after Phase 2c.2b finishes successfully
- Reclaim ~30 GB

### 14.17.16.7.4 Code-mapping strategy (sciname fallback)

The eBird taxonomy revises annually; codes drift. To handle 2024-code
(BC2026 / our targets) vs 2020-code (rohanrao directories):

1. Build `target_code → sciname` from `ebird_taxonomy_v2021.csv`.
2. List all rohanrao directory names (= 2020-era codes ∪ near-2020
   revisions ∪ codes that haven't drifted since 2020).
3. For each `target_code`:
   - If target_code's directory exists directly → use it
   - Else: look up `sciname = code2sci_v2021[target_code]`,
     find rohanrao directory with same sciname (via reverse lookup
     using the same eBird v2021 table for rohanrao's directory names)
   - Log unmatched target_codes in `data/processed/l2_redux_unmatched_codes.json`

If unmatched count is < 100 (~1 % of 6,971), accept and proceed. If
> 500 (>7 %), the rohanrao snapshot is stale enough that we should
reconsider — but the upstream data isn't going to change, so we'd
likely accept the partial coverage rather than abandon.

### 14.17.16.7.5 Files to create / write

- `four_track/scripts/l2_redux_phase2c_2a_download_rohanrao.sh` — Phase
  2c.2a launcher, two `kaggle datasets download` calls + stream-extract.
- `four_track/src/l2_redux/transcode_xenocanto.py` — Phase 2c.2b
  filter+transcode pipeline.
- `four_track/scripts/l2_redux_phase2c_2b_transcode.sh` — Phase 2c.2b
  launcher.
- `four_track/src/l2_redux/download_xenocanto.py` (already written but
  now obsolete) — leave in place for reference, mark deprecated in
  docstring. Could be revived if API key path is ever needed.

---

## ⏸️ PICK UP HERE — previous (2026-04-30 ~00:30 local — Phase 2c.2 rohanrao pivot — SUPERSEDED by §14.17.16.8)

## 14.17.16.8 Rohanrao trap + XC v3 API run launched (2026-04-30 ~18:10 local)

### 14.17.16.8.1 Rohanrao snapshot was a trap — 252/6,971 species (3.6%)

§14.17.16.7's rohanrao pivot was based on the assumption that the dataset
"covers ~10K species and should overlap our 6,971 by >95%." That was
**wrong by an order of magnitude**. The rohanrao
`xeno-canto-bird-recordings-extended-{a-m,n-z}` Kaggle dump is the
**BirdCLEF-2020 training audio**, not a broad XC mirror — only **259
species directories total** (153 in A-M, 106 in N-Z), all North-American
birds, ~23,785 mp3 recordings.

| Phase 2c.2 (rohanrao route) | Projected | Actual |
|---|---|---|
| Total XC species in source | ~10,000 | 259 |
| Targets matched | >95% (~6,600) | 252 (3.6%) |
| Targets unmatched | <500 | 6,719 |
| Wall-clock | ~30-60 min download + ~7 h transcode | done in ~1.5 h total |
| Output | ~80-130 GB | 3.5 GB (18,668 ogg files) |

**Lesson:** "extended" in the dataset name was misleading — extended *bands*
(A-M / N-Z), not extended *species coverage*. Should have verified species
count before launching 2c.2a. Phase 2c.2b matching report would have caught
this in 30 sec but I ran 2c.2a first.

The 18,668 transcoded ogg files for the 252 matched species are kept in
`data/external/xenocanto_bulk/` — they're a free head-start for Phase 5,
~3.5 GB. The 30 GB raw mp3 dump in `data/external/xenocanto_raw/` was
deleted (29 GB reclaimed).

### 14.17.16.8.2 Pivot back to XC v3 API route

Pre-conditions for the API route that didn't exist on 2026-04-29:
- User registered an account at xeno-canto.org and generated a v3 API key
- Verified email + login flow + key generated → exported as `XC_API_KEY`

Updated `src/l2_redux/download_xenocanto.py` from v2-deprecated → v3-live.
Diff vs the v2 original:
- Endpoint `https://www.xeno-canto.org/api/2/recordings` → `https://xeno-canto.org/api/3/recordings`
- Auth: `?key=<XC_API_KEY>` query param on every metadata call AND every
  audio file download (mandatory since 2025-10-10)
- Key read from env var `XC_API_KEY`, never logged, never persisted to disk
- 401/403 → terminal abort (`sys.exit`), 429 → 30 s sleep + retry, no abort
- Rate default 2.0 → 1.0 req/sec (conservative)
- Default input switched from `l2_redux_xc_targets.json` (all 6,971) →
  `l2_redux_unmatched_codes.json` (6,719 — the codes rohanrao didn't cover)
- Sort tiebreaker uses parsed `length` (v2 code referenced non-existent
  `"length-rss"` field which always returned 0)

XC v3 schema confirmed compatible with v2 field names (`id`, `gen`, `sp`,
`ssp`, `en`, `file`, `length`, `q`, `cnt`, `loc`, `lat`, `lng`, `type`,
`numRecordings`, `numPages`).

### 14.17.16.8.3 Smoke test (2026-04-30 17:57-18:06)

3-species smoke launched after porting:

| Field | Value |
|---|---|
| Species | abbbab1, abbwar1, abetow |
| Recordings downloaded | 457 |
| Failures | 0 |
| Wall-clock | 8m 31s |
| Avg ogg size | 184 KB (30s mono 32 kHz q4 vorbis) |
| HTTP errors observed | 0 (no 401/403/429/5xx) |

End-to-end auth + schema parse + audio download + ffmpeg transcode all
working. No rate-limit responses at 1 req/sec.

### 14.17.16.8.4 Full run launched (2026-04-30 18:07:52 local)

```
PID: 764735
Targets: 6,719 unmatched codes (6,718 runnable; 1 unmapped sciname)
Rate: 1.0 req/sec
Log: four_track/log/l2_redux_xc_v3_full_20260430_180752.log
Output: data/external/xenocanto_bulk/<code>/{_meta.json, XC<id>.ogg}
```

Auto-abort gates:
- ~~Wall-clock > 7 days~~ → **demoted to warn-only on 2026-04-30 ~22:30 local
  (per user override).** The 4.3 h status check projected ~6.7 days end-to-end,
  uncomfortably close to the 7-day hard cap (<5% margin). User decision: don't
  let the script auto-kill at 7 d; revisit at the 7-day mark and decide whether
  continuation is appropriate based on actual progress and remaining species
  composition. `WALL_CLOCK_CAP_SEC` renamed → `WALL_CLOCK_WARN_SEC`,
  `check_gates()` now prints a one-shot warning at 7 d and continues.
  - Caveat: the live PID 764735 has the old 7-day hard cap loaded in memory
    from before the edit. The source change protects future restarts only.
    If the live run reaches 7 d, it will sys.exit. Resume = restart with
    new code; the script's `skip` logic picks up existing files.
- Disk usage > 90% (unchanged — still hard abort)
- HTTP 401/403 (unchanged)

Throughput projection at 1.0 req/sec:
- Smoke 152 recs/species avg is biased high (alphabetical-prefix common
  species). Long-tail-corrected average ≈ 50-100 recs/species.
- 6,719 species × ~50-100 recs = 336K-670K downloads → **3-6 days**.
- Disk: 336K-670K × 184 KB = **60-120 GB final**. 852 GB free; fits.

### 14.17.16.8.5 Updated decision basis vs option (3) "smoke alone"

User's decision (yes to option 1 / full XC pretrain) rests on
§14.17.15.8.3's compression model + Sydorskyi's ablation table:

| Step | EffNetV2-S Public LB | Δ |
|---|---|---|
| Baseline | 0.837 | — |
| +Enhancements (LS, MixUp, BG, SpecAug) | 0.835 | −0.002 |
| **+Transfer learning (819K recs / 7,489 species)** | **0.881** | **+0.046** |
| +Pseudo Iter 1 | 0.908 | +0.027 |

Sydorskyi's pretrain alone carried +0.046 LB. Our 70K/636-species smoke
projects to +0.015 to +0.030 LB via §14.17.15.8.3's model. Gap closed by
scaling 636 → ~7,500 species: **+0.016 to +0.031 LB additional headroom**.
3-6× the ±0.005 noise band. Worth 3-6 days.

If full pretrain finetune lands at LB-neutral or negative, the kill is the
mechanism (val/LB anti-correlation per memory `feedback_single_fold_noise_floor`),
not corpus scale. No further pretrain corpus expansion is meaningful.

### 14.17.16.8.6 Files/state changes

- `src/l2_redux/download_xenocanto.py` — fully rewritten for v3 (no longer
  marked deprecated)
- `data/external/xenocanto_raw/` — DELETED (29 GB reclaimed)
- `data/external/xenocanto_bulk/<code>/` — 252 species pre-populated
  (rohanrao); will fill out toward ~6,971 over the run

---

## ⏸️ PICK UP HERE — previous (2026-04-30 ~18:10 local — XC v3 download IN FLIGHT only — SUPERSEDED by 2026-05-03 entry below; both XC v3 download AND arch-swap Phase 2 are now in flight)

**TL;DR:** Rohanrao snapshot only had 252/6,971 species (assumed ~10K).
Pivoted back to XC v3 API. Downloader ported (key in `XC_API_KEY` env
var). Smoke validated end-to-end. Full 6,719-species run launched at
18:07:52 local, PID 764735, ETA 3-6 days. Phase 2b probe Δ +0.0554 still
holds; Sydorskyi's +0.046 LB pretrain ablation justifies the run.

**Read first:** §14.17.16.8 for trap + pivot + launch state;
§14.17.15.8 for Phase 2b probe gate; §14.17.13 for 0.926 floor.

**Awaiting:** XC v3 download to complete (~3-6 days from 18:07:52). Then
Phase 3 (manifest merge + dedup), Phase 4 (full pretrain on combined
corpus), Phase 5 (encoder transfer to BC2026 finetune), Phase 6 (LB probe).

**Don'ts:**
- Don't kill PID 764735 unless rate-limit cascade or disk gate trips.
  Resume is cheap (per-recording skip-if-exists + per-species `_meta.json`
  cache), but a kill burns the partial progress on the in-flight species.
- Don't commit `XC_API_KEY` to any tracked file or log. Already
  configured to never echo.
- Don't restart the rohanrao 2c.2a pivot. Trap documented in §14.17.16.8.1.
  The "extended" in the dataset name was misleading — it's BC2020 (NA-only,
  259 species), not a broad XC mirror.
- Don't delete `data/external/xenocanto_bulk/` mid-run. The 252
  rohanrao-derived species inside it are real Phase 5 substrate — kept,
  not regression.
- Don't restart any L1/L2/D3/C2/B2/P5/P12/A2-iter-1/A2-iter-1.5/TopN
  lever — kill families exhausted.
- Don't gate any L2-redux Δ against the stale 0.931 floor — real floor
  is **0.926**, noise band ±0.005.

---

## 14.18 Non-Aves training — deferred lever survey (2026-05-01 ~00:35 local)

**Context.** L2-redux v1 (the live XC v3 download) is Aves-only — 6,718
eBird codes, 7,591 total species after rohanrao-merge, 0 non-Aves. BC2026
has 72 non-Aves classes (35 Amphibia + 28 Insecta + 8 Mammalia + 1
Reptilia) = 31 % of the test denominator. Question raised: how do we
train on those?

**Status:** PARKED. Revisit only after L2-redux v1 lands an LB number.
Filed here so the analysis isn't re-derived on next pickup.

### 14.18.1 Precedent — why this is hard, not just open

L5b-Amphibia (§14.10.13–§14.10.17) is the standing kill. Naive AnuraSet
mix into `train_folds.csv` → val +0.054, **LB −0.151**. Salvage recipe
(MixStyle 0.5 + background mixup, mixin_p=0.7) → val +0.054, but the
per-class-group Δ table (§14.10.17.2) showed Insecta/Reptilia Δ at 2–7×
the Amphibia Δ. Mechanism: AnuraSet INCT-site recordings share Pantanal
ambience with `train_soundscapes` val. Model learned the acoustic
background, not amphibian foreground. Both recipes terminally killed.

Memory: `project_l5b_amphibia_killed`. Lesson: **adding non-Aves data
to the BC2026 finetune mix lights up shortcut features the val set
rewards.** Anything new must break that loop.

### 14.18.2 Three paths that don't repeat the L5b trap

**(1) Non-Aves at pretrain time, not finetune time.** Generalize L2-redux
to non-Aves: add iNat-Sounds + AnuraSet (raw, all sites) + XC's `grp`
filter for Anura / Orthoptera / Chiroptera / Mammalia to the pretrain
corpus. BC2026 finetune still trains on `train_audio` only — no
site-leakage shortcut into val. Encoder learns broader spectro-temporal
features; finetune transfers them to the 72 non-Aves classes via
ImageNet-style transfer, not direct supervision.

- Cost: another ~3–8 day download + Phase 4 retrain.
- Risk: the encoder *also* trains on non-Aves spectra, possibly degrading
  Aves performance from current Aves-only L2-redux.
- Plan already flags this: §14.17.15.4 — *"iNat insect / amphibian
  pretraining for non-Aves. Different pretrain corpus, post-L2-redux v1
  if the non-Aves regression is real."*

**(2) Source-swap to non-Pantanal amphibian/insect corpora.** §14.10.17.4
considered XC anurans from Amazonia / Atlantic Forest / Cerrado without
INCT-site overlap. Estimated envelope: ≤ +0.01 LB. Rejected on cost-vs-
benefit at the time. Could be folded into option (1) at zero marginal
download cost (XC v3 query already running; just expand `grp` filter on
a follow-up run).

**(3) Per-taxon encoder split.** Train Aves-A1 (bird-pretrained) and
NonAves-A1 (insect/amphibian-pretrained) separately, route at inference
by class block. Engineering-heavy, more Kaggle artifacts to manage,
still depends on solving (1) for the non-Aves encoder corpus. Strict
superset of (1) — defer until (1) returns a non-zero LB Δ.

### 14.18.3 The trainable-subset reality check

Of 72 non-Aves classes, **11 are structurally untrainable** by adding
data. The 11 sub-0.55 Insecta classes are all `47158son*` —
Cicadidae sonotypes, acoustic pseudo-species defined by call pattern,
not biology (§3404 / §3422). 0/11 appear in `train_audio` and 0/11
appear in any external corpus (XC, iNat, AnuraSet) because they aren't
species in any taxonomy outside this competition.

| sub-group | count | trainable by data scaling? |
|---|---:|---|
| Amphibia | 35 | yes (with non-Pantanal source) |
| Insecta — biological species | ~17 | yes |
| Insecta — `47158son*` sonotypes | 11 | **no — structural, post-proc only** |
| Mammalia | 8 | yes |
| Reptilia | 1 | trivially yes |

Realistic addressable population: ~61/234 = **26 %** of classes, not the
naive 31 %. Cicadidae sonotypes — the dominant Insecta macro-AUC drag —
must be solved on the post-proc side (taxon-aware T, neighbor smoothing,
Perch-side fallback), not by training corpus expansion.

### 14.18.4 LB envelope estimate (rough)

Assumptions: option (1) lifts the 61 trainable non-Aves classes by some
delta, leaves the 11 sonotypes untouched, and is roughly Aves-neutral.

- Optimistic: +0.05 macro-AUC on the trainable 61 → weighted (61/234)
  × 0.05 = **+0.013 LB**.
- Realistic: +0.02 on trainable 61 → **+0.005 LB** (one noise band).
- Pessimistic: Aves regresses from encoder split focus, weighted avg
  net-zero or negative.

That's a **±0.005–0.013 swing** for a 5–10 day side-quest. Not a swing
lever like L2-redux. Worth queuing only if L2-redux v1 ships clean and
post-proc levers are exhausted.

### 14.18.5 Pickup criteria

Open this lever **only if all** of the following hold:

1. L2-redux v1 has landed an LB probe (no longer in-flight).
2. L2-redux v1 LB Δ ≥ 0 (i.e. the bird-only pretrain didn't regress
   non-Aves catastrophically; if it did, the corpus-pretrain hypothesis
   is broken and a non-Aves expansion of the same corpus won't fix it).
3. No cheaper post-proc lever (taxon-aware T tuning, Cicadidae-specific
   neighbor smoothing, Perch-side blend reweighting on Insecta) is
   queued and unattempted. The 11 sonotypes are post-proc territory
   anyway; try post-proc first.
4. ≥ 10 days remain in the competition. A 5–10 day data acquisition +
   training + LB probe cycle inside a tighter window has negative EV vs.
   a known-cheap post-proc lever.

If any of (1)–(4) fails, leave parked.

### 14.18.6 First action when picked up

Don't write code. Run a **paper exercise** first:

- Compute the actual per-class-group LB-equivalent Δ from the Phase 6
  L2-redux v1 result. Did the bird-only encoder hurt Insecta /
  Amphibia / Mammalia / Reptilia macro-AUC vs. ImageNet init? If yes,
  by how much? That number sets the LB ceiling for option (1).
- If the regression on non-Aves is < 0.01 AUC, option (1) is
  marginal. Skip to post-proc.
- If the regression is > 0.02 AUC, option (1) has real headroom and
  is worth the 5–10 day spend.

Files to seed at pickup:
- New script: `src/l2_redux_nonaves/build_corpus.py` (iNat-Sounds +
  AnuraSet raw + XC `grp` filter union; dedup against L2-redux v1
  manifest).
- Reuse: `src/l2_redux/download_xenocanto.py` (just pass different
  target list + `grp` param).
- Reuse: existing `src/pretrain_l2_redux.py` (no architecture change).
- Pretrain ckpt path: `models/a1_l2_redux_nonaves/`.

---

## 14.19 EffNetV2-S arch swap — scoping (2026-05-02 ~21:30 local)

**Status: SCOPING ONLY, not greenlit.** Drafted in parallel with the live
L2-redux XC v3 download (PID 1029580, ETA ~May 7) so a noise-band L2-redux
LB result on May 11 can pivot to arch-swap the same day instead of burning
2-3 days on re-reading the BC2025 winners' arch sections. Same draft-then-
go/no-go pattern as §14.10 / §14.17.14.

### 14.19.1 Why this lever, why now

Both BC2025 winners abandoned EfficientNet-B0 for larger backbones:
- **Sydorskyi (2nd, public 0.925)**: NFNet-L0 + EffNetV2-S (their entire
  ablation table is on EffNetV2-S, see `reference_bc2025_winners_writeups`).
- **Babych (1st)**: EffNet stack — 4×v2_s, 3×v2_b3, 4×b3_ns, 2×b0_ns. v2_s
  contributes the most-weighted ensemble members.

Our LB-0.931 baseline is `tf_efficientnet_b0.ns_jft_in1k` (~5.3M params).
The EffNetV2-S baseline (no other changes) was a meaningful chunk of the
0.835 → 0.881 in Sydorskyi's table; the +0.046 *delta* is from L2-redux
pretrain on top of that arch, but v2-S was already their starting point.

**Sequencing**: this lever is queued strictly **after** L2-redux v1 lands
an LB number on the existing B0 arch. If L2-redux clears the 0.005 noise
band (LB ≥ 0.936 on the 0.926 floor), arch swap becomes "L2-redux v2 on
EffNetV2-S" — same downloaded corpus, new pretrain run, no re-download.
If L2-redux is at noise (LB 0.927-0.935), arch swap is the next swing
lever in its own right.

### 14.19.2 What we already have on disk

- `four_track/src/model_a1.py` (195 lines) — `BirdSEDModelA1` builds via
  `timm.create_model(backbone_name, features_only=True, out_indices=(4,))`.
  **The `backbone_name` arg is already plumbed through** to `train_a1.py`
  via `--backbone`. Swap may be a one-line change *if* the MixStyle hook
  attachment stays valid (see §14.19.4 risk #1).
- `four_track/src/train_a1.py` — `--backbone` CLI arg already defaults to
  `config.BACKBONE` but accepts any timm string. Save path already
  templates the backbone into the filename: `a1_{backbone}_fold{f}_seed{s}_{loss}.pt`.
- `four_track/src/pretrain_l2_redux.py` line 402 — uses `config.BACKBONE`
  hardcoded. Needs a `--backbone` arg (one-line change to mirror
  `train_a1.py`'s pattern).
- `four_track/src/export_a1_jit.py` — uses `config.BACKBONE` hardcoded for
  filename construction (line 57). Needs `--backbone` arg.
- The L2-redux XC v3 corpus, when it finishes downloading, is **arch-
  agnostic raw audio + manifest**. No re-download needed for arch swap.

### 14.19.3 Compute + disk envelope (rough)

| Stage | EffB0 baseline | EffNetV2-S est. | Notes |
|---|---:|---:|---|
| Backbone params | 5.3 M | ~21 M | 4× larger |
| Train batch (BS=64 on B0) | fits | likely OOM | drop to BS=32 with grad accum=2, or BS=24 |
| Train step time | ~0.10 s | ~0.20-0.25 s est | 2-2.5× slower per step on GB10 |
| 4-fold A1 finetune (25 ep × 4 folds) | ~22 h (production) | ~44-55 h | 2-2.5× wall-clock |
| L2-redux pretrain (50 ep on 819K recs) | ~18 h projected | ~36-45 h est | linear in step time, same data |
| Inference (CPU, hidden test 90-min budget) | ~30 min @ 4-fold | **UNVERIFIED** | hard kill criterion — see §14.19.4 #2 |
| TorchScript ckpt size | ~17 MB / fold | ~84 MB / fold est | Kaggle dataset ~340 MB for 4 folds |

**Disk impact**: negligible. Pretrain ckpts ~85 MB each vs ~22 MB on B0.
Kaggle dataset slots are abundant.

**Compute impact, end-to-end if started cold from ImageNet init (no
L2-redux pretrain)**: 4-fold finetune ~2 days + JIT export ~1 hour + LB
probe = **~2-3 days wall-clock**.

**Compute impact, end-to-end if started from L2-redux v1 EffNetV2-S
pretrain**: pretrain ~2 days + 4-fold finetune ~2 days + LB probe = **~5
days wall-clock**.

### 14.19.4 Risks that need probing before training (Phase 0/1 gates)

1. **MixStyle hook attachment.** `model_a1.py:152` hooks
   `backbone.blocks[1]` for EfficientNet-style backbones. timm's
   EffNetV2-S uses the same generic `EfficientNet` class with `.blocks`
   organized as 7 stages of IRBlocks/MBConvs/FusedMBConvs — `.blocks[1]`
   exists and is a valid hook target by structure, but the channel
   semantics are different (FusedMBConv uses single-conv blocks, not the
   B0 IR pattern). **Probe**: instantiate one model, confirm
   `.blocks[1]` shape, run a forward pass, confirm hook fires. If
   FusedMBConv stages produce feature maps that MixStyle's freq-stat
   perturbation degrades convergence, fall back to hooking
   `.blocks[2]` (the first IRBlock stage, post-FusedMBConv).
2. **CPU inference budget.** `feedback_kernel_timeout_vs_scoring_stall`
   memory: B2 ConvNeXt was killed when its 90-min hidden-test re-run
   timed out despite the local 30-min CPU benchmark passing. Hidden
   test is ~3× the local soundscape volume. **Probe (BLOCKING)**:
   before any 4-fold train kicks off, run the existing
   `birdclef2026-protossm-postproc` notebook in submit-mode locally
   with EffNetV2-S `.pt` placeholders to measure A1's CPU inference
   contribution at the new arch on a representative 100-file sample.
   Linear-extrapolate to ~700-file hidden test. Subtract from the
   60-min budget remaining after the Perch+ProtoSSM stack
   (currently ~12 min). **Hard kill**: if extrapolated A1 CPU time
   pushes total over 75 min (15-min safety margin), arch swap is
   structurally infeasible — same fate as B2.
3. **timm import currently broken in env.** `python -c "import timm"`
   fails with `RuntimeError: operator torchvision::nms does not exist`
   (probed 2026-05-02 21:30 local). torchvision/torch version skew.
   Doesn't block the live download or the Aves-only L2-redux pretrain
   (those are nohup'd from a working session), but blocks any new
   probe/training launch. **Phase 0 fix required**: pin torchvision
   to a torch-2.5-compatible version, or downgrade torch. Estimate ½
   day. Don't touch the `kaggle` env in-place mid-download — clone to
   `kaggle-arch` first.
4. **JIT export with CC 12.1 GPU.** `feedback_gb10_nvrtc_jit` memory:
   JIT-traced `.pt` files fail at first forward on this GPU
   (GB10 CC 12.1 > PyTorch NVRTC max 12.0). `export_a1_jit.py` already
   handles this for B0 by disabling fuser. EffNetV2-S has FusedMBConv
   stages that may compile differently — verify after Phase 1.

### 14.19.5 Phased plan (gate at every phase boundary)

**Phase 0 (~½ day, no GPU): env fix + arch instantiation probe**

UPDATE (2026-05-02 ~23:15 local): Phase 0 partially executed via the
multi-machine runon infrastructure (see §14.20). Status:
- **Root cause confirmed**: skynet's `kaggle` env has torchvision 0.25 +
  torchaudio 2.10 (way ahead of torch 2.7.1 — incompatible C++ ABI).
  Symptom: `RuntimeError: operator torchvision::nms does not exist`.
  Canonical fix: pin matched triple torch 2.7.1 + torchvision 0.22.1 +
  torchaudio 2.7.1.
- **Pinned env file shipped**: `BirdCLEF/environment.yml` (committed at
  the BirdCLEF git root). Same file builds working envs on either
  machine. Python 3.11 (vs skynet's grandfathered 3.13).
- **Deepthought-side fix**: DONE. `runon-setup deepthought` built a
  fresh `kaggle` env from environment.yml on the RTX 4080 box. Verify
  via `tail /tmp/runon_setup_deepthought.log`. arch-instantiation
  probe can run there immediately — clean torch+torchvision+timm.
- **Spark-side fix**: DEFERRED until L2-redux download (PID 1029580)
  finishes ~May 7. Rebuilding `kaggle` env in-place would kill the
  in-flight download. After May 7: `conda env remove -n kaggle &&
  conda env create -f environment.yml` from BirdCLEF/.

Remaining Phase 0 work (run on deepthought via `runon`):
- ~~timm + EffNetV2-S/NFNet-L0 instantiation probe~~ DONE 2026-05-02 23:26
  local. Result:
  - **`tf_efficientnetv2_s.in21k_ft_in1k`**: instantiates clean, has
    `.blocks[1]` → existing MixStyle hook attaches with **zero
    `model_a1.py` changes**. Feature shape at out_indices=(4,):
    C=256, T=16 (vs B0's C=320, T=16; auto-inferred by dummy
    forward in `BirdSEDModelA1.__init__`). Params 19.8M (4× B0).
  - **`eca_nfnet_l0`**: instantiates but features_only wrapper
    exposes NEITHER `.blocks` nor `.stages` at top level. Needs
    custom hook walk of `m.feature_info`. Params 21.8M (4× B0).
- **Gate**: PASSED for EffNetV2-S. Failed (cleanly, as expected) for
  NFNet-L0 — confirms §14.19.7's "defer NFNet-L0 to ensemble v2"
  decision.
- **Phase 1 unblocked.** Next concrete action:
  ```
  cd four_track
  runon deepthought python -u src/train_a1.py \
      --backbone tf_efficientnetv2_s.in21k_ft_in1k \
      --fold 0 --epochs 1 --smoke-test
  ```
  Pre-req: data sync via `runon --push-data deepthought true`
  (~25 GB, ~5-10 min).

**Phase 1 (~½ day, GPU smoke): port BirdSEDModelA1**

UPDATE (2026-05-02 ~23:53 local): PASSED on deepthought. 1m41s end-to-end.
- `train_a1.py --backbone tf_efficientnetv2_s.in21k_ft_in1k --fold 0
  --epochs 1 --smoke-test` ran clean. NO `model_a1.py` changes needed.
- Fold 0: 175 clips (smoke-truncated), 2 batches/epoch.
- HF pretrained weights downloaded + loaded (Unexpected keys for
  `classifier.*` / `bn2.*` / `conv_head` are benign — features_only
  drops the head we never use).
- BS=64 fits on RTX 4080 16 GB (no OOM; smoke = 2 batches but real
  training has same per-batch peak memory).
- train_loss=0.2115 (finite, gradients flow).
- val_roc_auc=0.0000 expected (2 batches don't train a real model).
- One side-effect bug surfaced and fixed: `BirdCLEF/src/config.py:22`
  hardcoded skynet's absolute path for `ROOT`. Patched to use env
  override + `__file__`-based dynamic resolution. Spark behavior bit-
  identical (verified). Deepthought now resolves to mirror path. See
  §14.20 update + four_track/CLAUDE.md "read-only legacy" exception.
- Cosmetic non-issue to fix later: `train_a1.py` header still hardcodes
  "EffNet-B0 SED" in the print line; should use `args.backbone`.

Original Phase 1 task description retained for reference:
- Add the new backbone string as a `train_a1.py --backbone` value.
  Probably no code change required if Phase 0 confirms `.blocks[1]` works.
- Run `train_a1.py --backbone tf_efficientnetv2_s.in21k_ft_in1k --fold 0
  --epochs 1 --smoke-test`. Confirm forward + backward run, single fold
  smoke loss decreases.
- Drop BATCH_SIZE in `config.py` (or override via CLI) to 32 if smoke
  OOMs at 64. Document final BS in the run log.
- Per `feedback_gpu_memory_hygiene_per_epoch`: confirm gc.collect() +
  torch.cuda.empty_cache() are still in the per-epoch / per-fold cleanup.
- **Gate**: smoke fold-0 run completes 1 epoch w/o OOM; loss is
  finite and decreases monotonically over the 2 batches.

**Phase 1.5 (~30 min, CPU-only): inference budget probe (BLOCKING)**

UPDATE (2026-05-03 ~00:00 local): PASSED on deepthought. Per-batch
(BS=16) at 4 vCPUs (taskset -c 0-3 + torch.set_num_threads(4)):
**0.764 sec ± 0.004** mean over 10 timing forwards. Extrapolated to
~700 files × 12 chunks × 4 folds = 33,600 forwards: **26.7 min**.
Budget 78 min → +51.3 min margin (66% under budget). V2-S params
20.2 M (3.8× B0). Probe script: `four_track/src/_probe_v2s_cpu_inference.py`.

**Caveats on the verdict:**
- deepthought CPU is modern x86_64; Kaggle vCPU is throttled. Realistic
  1.5-2× per-core slowdown on Kaggle → 40-53 min. Even pessimistic 3×
  (~80 min) is at the edge, not catastrophic.
- "78 min A1 budget" assumed Perch+ProtoSSM stack at 12 min, untested.
  If real overhead is 20-25 min, A1 budget shrinks to 65-70 min — still
  passes pessimistic adjusters.
- 0.764 sec is JUST model forward on pre-computed mels. mel+PCEN is
  the same cost for B0 vs V2-S (same input pipeline) — cancels out of
  the arch-swap delta.
- JIT trace was at production shape (16, 3, 224, 512); fusions lock to
  that shape, matches notebook cell 41 BS=16.

**Useful calibration follow-up (NOT BLOCKING)**: run same probe with B0
to get V2-S/B0 CPU ratio empirically. Expected ~3-4×; if so, predicts
B0 baseline at ~7-8 min in this benchmark — direct sanity check
against production.

**Phase 2 unblocked.** Per CLAUDE.md multi-GPU directive: fold split
across both machines (folds 0,1 on skynet / folds 2,4 on deepthought),
~22h sequential → ~11h parallel.

**Phase 2 (~2 days, GPU): 4-fold finetune from ImageNet init**

UPDATE (2026-05-03 ~00:08 local): LAUNCHED — fold-split across both
machines per CLAUDE.md multi-GPU directive.

- **Spark (folds 0,1)**: needs sidecar `kaggle-arch` env (live `kaggle`
  env has the broken torchvision/torchaudio version skew documented in
  §14.20.3, can't be rebuilt mid-XC-v3-download). Sidecar build via
  `conda env create -n kaggle-arch -f BirdCLEF/environment.yml` is
  in flight (PID 151474, ~10 min build). Spark training launches via
  watcher PID 151732 once env build exits clean.
- **Deepthought (folds 2,4)**: launched at 00:07 via `runon deepthought
  python -u src/train_a1.py --backbone tf_efficientnetv2_s.in21k_ft_in1k
  --folds 2,4 --loss hybrid`. PID 944726 on deepthought. Val cache
  built (1478 segs, 75 species), HF V2-S weights loaded, fold 2 epoch 1
  in progress.
- **Per-epoch wall-clock revision**: smoke train extrapolation says
  ~2.5 min/epoch on V2-S (446 batches/epoch at BS=64). 25 epochs × 2
  folds ≈ 130 min/machine. Total wall-clock ~2.5-3h, NOT the 11h
  pessimistic envelope from §14.19.3. Will record actual once first
  epoch completes.
- Watchers: `b3sc081eq`-class (env→skynet train) and `bx0bgt7mj`-class
  (deepthought train completion) will fire when each chain exits.
- L2-redux download (PID 1029580 on skynet, ~50h elapsed) confirmed
  untouched by sidecar env build (uses different conda env name).

Original Phase 2 task description retained for reference:
- Run `train_a1.py` 4-fold (folds 0,1,2,4 to match production), 25
  epochs each, hybrid loss (matches LB-0.931 baseline per
  `project_a1_baseline_loss_is_hybrid`).
- Per `feedback_clean_logs_before_training` and
  `feedback_rm_log_every_launch`: `rm -f log/*.log` BEFORE each launch.
- Track val_v2 macro-AUC per fold. Per `feedback_single_fold_noise_floor`,
  do NOT trust single-fold deltas — only the 4-fold mean is informative.
- Save ckpts at `models/a1/a1_tf_efficientnetv2_s.in21k_ft_in1k_fold{F}_seed42_hybrid.pt`.
- **Gate**: 4-fold mean val_v2 macro-AUC ≥ 0.7414 (current B0 baseline).
  Below = arch is worse than B0 even with the bigger model — kill.
  At 0.7414 ± 0.005 = noise-equivalent; LB probe to disambiguate.
  Above = real lift, push to LB.

**Phase 3 (~1 day, GPU): JIT export + Kaggle dataset push**
- Add `--backbone` arg to `export_a1_jit.py` (mirror `train_a1.py`).
- Export 4 folds as TorchScript (~85 MB each = ~340 MB dataset).
- Output dir: `kaggle_datasets/a1-effv2s-ckpts/`. Dataset id:
  `stevewatson999/birdclef-2026-a1-effv2s-ckpts`.
- Per `feedback_backup_ckpts_before_overwrite`: archive
  `kaggle_datasets/a1-effb0-ckpts/` to `_backups/` first.
- Update `jupyter/protossm-postproc/kernel-metadata.json` to add the new
  dataset (DON'T remove a1-effb0-ckpts yet — we want both available
  for A/B switching via a notebook variable).
- Update notebook cell 41: add `A1_BACKBONE_VARIANT = "effv2s"` toggle
  that points `A1_CKPT_DIR` at the new dataset.

**Phase 4 (1 LB submission): LB probe**
- Push next kernel version. Confirm Kaggle UI shows `Status: Complete`
  AND a per-version LB score (per
  `feedback_kernel_timeout_vs_scoring_stall`: CLI's `COMPLETE` is the
  sample-submission-uploaded signal, NOT the hidden-test scoring signal).
- **Decision rule** (with current 0.926 LB floor, ±0.005 noise band):
  - LB ≥ 0.940 → ship; queue L2-redux v2 on this arch.
  - LB 0.932-0.939 → real lever, +0.006 to +0.013. Ship as production.
  - LB 0.927-0.931 → within noise. Repeat probe with seed42→seed7
    retrain on fold 0 to disambiguate, OR kill if budget tight.
  - LB ≤ 0.926 → arch swap regression. Revert kernel-metadata.json,
    declare arch-swap-from-ImageNet dead. Pivot decision: try
    Phase 5 (L2-redux v2 on EffNetV2-S) only if L2-redux v1 on B0
    cleared noise — otherwise the arch+pretrain combination is
    speculative.

**Phase 5 (~5 days, GPU, OPTIONAL): L2-redux v2 on EffNetV2-S**
- ONLY if both Phase 4 and L2-redux v1 (on B0) hit ≥ noise band.
- Reuse the XC v3 corpus from L2-redux v1 (no re-download).
- Modify `pretrain_l2_redux.py` line 402 to accept `--backbone` arg.
- 50 epochs pretrain on full 819K-rec corpus → backbone-only ckpt at
  `models/a1_l2_redux_v2_effv2s/l2_redux_backbone_e50.pt`.
- 4-fold finetune via `train_a1.py --backbone tf_efficientnetv2_s.* --init-from
  models/a1_l2_redux_v2_effv2s/l2_redux_backbone_e50.pt`.
- Re-export, re-push, re-probe LB.
- **This is the single highest-EV path in the queue if both prior
  probes pass.** Sydorskyi's exact recipe — arch + corpus together.

### 14.19.6 LB envelope on our stack (rough)

Sydorskyi got Public LB **0.835 → 0.881 = +0.046** when *both* arch
(EffNetV2-S as their baseline) and L2-redux pretrain landed. We're
attempting the arch part standalone first.

Adjusters for the standalone arch swap (no new pretrain):
- **No comparable standalone-arch ablation in their table**. Their
  baseline is EffNetV2-S; we'd be measuring B0→V2-S transition, which
  they didn't report. Estimate from arch-only paper-class deltas: a
  4× larger backbone on the same corpus typically adds **+0.005 to
  +0.015** AUC on ImageNet-style benchmarks. Discount further for
  the small training corpus (28K BC2026 focal recs, much less than
  ImageNet — large models underutilize). Estimate **+0.000 to +0.010 LB**.
- **Realistic envelope: noise to +0.010**. Best case: +0.010 → 0.936
  (real lever). Likely case: +0.005 → 0.931 (noise band). Worst case:
  −0.005 (over-parametrized for 28K-rec finetune corpus, regresses).

For Phase 5 (arch swap + L2-redux):
- Sydorskyi's full +0.046 with both → discount ~50% for backbone-
  contribution-amplification on smaller stacks: **+0.015 to +0.030 LB**.
- Realistic envelope: +0.010 to +0.025. Best: +0.025 → 0.951 (gold).
  Likely: +0.015 → 0.941 (real lever). Worst: noise band.

### 14.19.7 What's NOT in this scope (deliberately deferred)

- **NFNet-L0 as primary swap target.** EffNetV2-S is the safer port
  (existing `.blocks` MixStyle hook works; same generic timm
  EfficientNet class). NFNet-L0 needs MixStyle hook fallback path
  (uses `.stages`, not `.blocks`). Defer NFNet-L0 to "v2-S succeeded,
  ensemble lifts further from arch diversity" — it's the natural
  second arch for a 2-arch ensemble after EffNetV2-S lands.
- **B3 / V2-B3 / B3-NS** (Babych's intermediate bucket). 3× the size
  of B0, 1.4× smaller than V2-S. Plausible "safer first step" but
  V2-S is what *both* winners' top ablations actually used. Skip the
  intermediate.
- **Multi-arch ensemble of B0 + V2-S.** If V2-S clears LB but only
  marginally, ensembling against B0 may add diversity. Defer to
  post-Phase 4 — engineering scope is meaningful (kernel cell fork,
  rank-blend coefficient tuning).
- **SoftAUCLoss (Babych)** and **TopN postproc retry**. Both already
  scoped in §14.17.6; not bundled with arch swap to keep the lever
  unambiguous (we want to know if arch alone moved LB).
- **Architecture-specific recipe tweaks** (Sydorskyi-style label
  smoothing, RandomFiltering, etc.). Bundled-recipe transplants
  underdeliver per `project_m1_mel_killed` and `project_m2_multilayer_gem_killed`.
  Ship the arch swap with the existing LB-0.931 recipe; tune later.

### 14.19.8 Files to create / modify when picked up

| File | Change |
|---|---|
| `src/config.py` (parent BirdCLEF) | NO change — keep `BACKBONE = "tf_efficientnet_b0.ns_jft_in1k"` as the *baseline* default; new arch passed via `--backbone` CLI |
| `src/model_a1.py` | possibly NONE if `.blocks[1]` hook works; otherwise add backbone-class dispatch in `_register_mixstyle_hook()` |
| `src/train_a1.py` | possibly NONE (`--backbone` already plumbed); BATCH_SIZE may need CLI override |
| `src/pretrain_l2_redux.py` | add `--backbone` arg; pass through to `BirdSEDModelA1(backbone_name=...)` (Phase 5 only) |
| `src/export_a1_jit.py` | add `--backbone` arg; use it in src filename construction (line 57) and OUT_DIR (line 37) |
| `kaggle_datasets/a1-effv2s-ckpts/` | NEW dataset dir; populated by Phase 3 export |
| `jupyter/protossm-postproc/kernel-metadata.json` | add new dataset id; keep old one for A/B |
| `jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb` cell 41 | add `A1_BACKBONE_VARIANT` toggle; default to current "effb0" |

### 14.19.9 Open questions for the user (go/no-go)

Three decisions before kicking off — same shape as §14.17.14.8:

1. **Approve EffNetV2-S as the arch target** (over NFNet-L0 / B3)? Phase 0/1
   is reversible; the choice locks in at Phase 2 training launch.
2. **Pretrain coupling**: do we run Phase 2 (ImageNet init only) FIRST
   to disambiguate "arch effect" from "arch + L2-redux effect", or skip
   straight to Phase 5 (arch + L2-redux) for higher EV at higher cost?
   Recommendation: Phase 2 first — clean attribution, ~2 days, rules
   in/out the cheaper variant before committing to Phase 5's full
   week.
3. **Kick off Phase 0 now (parallel with the live L2-redux download)**, or
   wait until the download completes May 7? Phase 0 doesn't need GPU,
   doesn't write to anything the download depends on, and de-risks the
   timm/torchvision env break. **Recommendation: Phase 0 now.**

If Phase 0 + Phase 1 both pass, Phase 1.5 (CPU inference probe) is the
single most informative gate — it'll either green-light the whole arch-
swap path or kill it on the same fundamentals that killed B2. Don't
launch Phase 2 without a Phase 1.5 pass.

### 14.19.10 Phase 2 RESULT — 4-fold V2-S clean pass on val gate (2026-05-03 ~19:30 local)

**TL;DR:** 4-fold V2-S finetune (ImageNet init) clears the §14.19.5
Phase 2 gate by **+0.0116** above 0.7414, every fold non-regressing vs
B0 baseline, fold spread tightens by ~half. JIT export complete in
`kaggle_datasets/a1-effv2s-ckpts/`. Phase 4 (Kaggle dataset push + LB
probe) ready for go/no-go.

**Per-fold results:**

| Fold | Wall-clock | best val_v2 macro-AUC | epoch of BEST | B0 baseline | Δ |
|---|---|---|---|---|---|
| 0 | 80 min (DT, today) | **0.7448** | 25 | 0.7414 | +0.003 |
| 1 | 79 min (DT, today) | **0.7626** | 18 | 0.7232 | +0.039 |
| 2 | 67 min (DT, yest.) | 0.7641 | 20 | 0.6970 | +0.067 |
| 4 | 79 min (DT, yest.) | 0.7406 | 19 | 0.7250 | +0.016 |
| **Mean (4-fold)** | | **0.7530** | | 0.7217 | **+0.031** |

Total compute: ~5h 47m DT GPU wall-clock. Skynet GB10 was attempted in
parallel for folds 0,1 (per §14.20 multi-machine plan) but ran ~4.5×
slower than DT (13.4 vs 3.2 min/epoch on V2-S). Killed and re-queued on
DT for ~7 h wall-clock saving. Empirical findings codified in
`feedback_default_to_deepthought_for_training.md` and the four_track
CLAUDE.md "Two-GPU workflow" section. The §14.20 framing of skynet ≈
4080 was wrong; skynet < 4080 by 4-5× on V2-S-class training due to
LPDDR5X bandwidth (~2.6× lower than GDDR6X) + missing sm_120 cuDNN
kernels.

**Gate check (§14.19.5 Phase 2 decision rule):**
- Gate: 0.7414 (matches B0 LB-0.931 fold-0 val)
- Result: 0.7530, **+0.0116 above gate**
- Per spec: "Above gate by ≥ 0.005 = real lift, push to LB" — clean pass.
- Single-fold noise floor (~0.03 per
  `feedback_single_fold_noise_floor.md`) scaled to 4-fold ≈ ±0.015. The
  +0.031 lift is just above this band; call it "marginal but
  consistent" — no fold regresses, spread tightens (V2-S
  0.7406-0.7641 = 0.024 vs B0 0.6970-0.7414 = 0.044, ~half).

**Inference timing sanity check (verifies §14.19.4 Phase 1.5 holds for
the actual fold ckpts):** Re-measured B0 vs V2-S 4-fold extrapolation
on DT 4-vCPU, BS=16, JIT-traced artifacts:
- B0:   337 ms/batch → **11.8 min** for 33,600-forward 4-fold ensemble
- V2-S: 692 ms/batch → **24.2 min** same workload
- V2-S/B0 ratio 2.05× — consistent with Phase 1.5 (26.7 min) within 10%.
- A1 budget on Kaggle = 78 min; pessimistic 2× Kaggle-vs-DT CPU slowdown
  puts V2-S at ~48 min on Kaggle. Margin ~30 min. No timeout risk.

**JIT export deliverable:**
- `four_track/kaggle_datasets/a1-effv2s-ckpts/` populated with
  `a1_fold{0,1,2,4}.pt` (~82.5 MB each, 322 MB total) + `dataset-metadata.json`.
- B0 ckpts archived to
  `_backups/a1_fold{N}_effb0_LB0931_20260503.pt` per
  `feedback_backup_ckpts_before_overwrite`.
- `export_a1_jit.py` patched (`BACKBONE_VARIANTS` table + `--backbone`
  arg); B0 path remains the default.

**Side observations during Phase 2:**

1. **V2-S Δ vs B0 is largest on B0's worst folds and smallest on B0's
   best fold.** Fold 2 (B0 worst: 0.6970) → +0.067; fold 0 (B0 best:
   0.7414) → +0.003. Spread tightens from 0.044 → 0.024. The shape you
   expect when an arch is *better-regularized* rather than just larger.
   Implication: re-running fold 3 on V2-S might surprise positively —
   fold 3's miscalibration on B0 could be arch-specific. NOT IN THIS
   SCOPE; revisit only if Phase 4 LB lands clean.

2. **Skynet env-build trap (caught 2026-05-03 ~14:35):** the
   `kaggle-arch` sidecar via `conda env create -f environment.yml`
   hung 14.5 h on the classic conda solver. Root cause: `~/.condarc`
   has `solver: classic` + extra `defaults` channel layered on the
   env file's conda-forge declaration, and the classic SAT solver
   thrashes on aarch64 where `defaults` coverage is poor. Fix:
   `conda install -n base -c conda-forge mamba -y` then
   `mamba env create -n kaggle-arch -f environment.yml` — completes
   in minutes via libsolv. **Plus:** environment.yml's
   `--index-url https://download.pytorch.org/whl/cu126` falls back to
   `+cpu` wheels on aarch64 because no aarch64 CUDA wheels exist at
   cu126. Patched in-flight to cu128. Required follow-up: change to
   cu128 in environment.yml AND flip `solver: libmamba` in `~/.condarc`
   before the planned skynet `kaggle` env rebuild post-May-7. Both
   edits gated on user approval.

**Phase 4 decision-rule recommendation (§14.19.5):**

Recommendation: **proceed to Phase 3 (Kaggle dataset push) when user
approves**. The 4-fold val signal is the cleanest Phase-2-class probe
result on this stack since the original B0 5-fold landed — every prior
arch/training lever in the §14.17 series killed at the val gate. Per
the plan's Phase 4 LB outcome table:

- LB ≥ 0.940 → ship; queue L2-redux v2 on V2-S (Phase 5)
- LB 0.932-0.939 → real lever, ship as production
- LB 0.927-0.931 → within noise; repeat seed42→seed7 fold-0 retrain
  to disambiguate, OR kill if budget tight
- LB ≤ 0.926 → arch swap regression; revert kernel-metadata.json

Val→LB transfer is not 1:1 (cf. `project_l1_killed`: val 0.9042 →
−0.001 LB). But val_v2 is the GT-only diagnostic specifically built
to be cleaner than the prior leaky variants. Inference budget is not
at risk. Worth a single LB attempt.

**Awaiting user approval (per overnight contract "needs approval" line):**
1. `kaggle datasets create -p kaggle_datasets/a1-effv2s-ckpts/` — new
   public dataset.
2. `jupyter/protossm-postproc/kernel-metadata.json` add the new
   dataset id (keep `a1-effb0-ckpts` for the production fallback).
3. Notebook cell 41 add `A1_BACKBONE_VARIANT` toggle (default
   `effb0`; flip to `effv2s` for LB probe).
4. `kaggle kernels push` for the V2-S-toggled kernel version.

**Reference pointers for cross-checking on pickup:**
- Today's DT log (folds 0,1): `_runon/BirdCLEF/log/runon_deepthought_20260503_162821.log` (also synced back to skynet)
- Yesterday's DT log (folds 2,4): `BirdCLEF/log/runon_deepthought_20260503_000717.log`
- 4 V2-S ckpts: `four_track/models/a1/a1_tf_efficientnetv2_s.in21k_ft_in1k_fold{0,1,2,4}_seed42_hybrid.pt`
- 4 V2-S JIT artifacts: `four_track/kaggle_datasets/a1-effv2s-ckpts/a1_fold{0,1,2,4}.pt`
- B0 archive (LB 0.931 production): `four_track/kaggle_datasets/_backups/a1_fold{N}_effb0_LB0931_20260503.pt`
- L2-redux XC v3 download: PID 1029580 on skynet, ~50% done, ETA ~May 7

### 14.19.11 Phase 4 RESULT — V2-S KILLED on LB noise band (2026-05-03 ~23:25 local)

**TL;DR:** V2-S arch swap declared **null on LB**. v71 (4-fold) timed
out (~101 min Phase ii, above the apparent ~90-min cap). v72 (3-fold,
dropped fold 4 to fit budget) scored **LB 0.930**, vs B0 baseline
0.931 → **−0.001 within ±0.005 single-submission noise**. Revert
landed: `kernel-metadata.json` no longer references
`a1-effv2s-ckpts`, cell 41 toggle removed, A1 path back to B0
4-fold rank-avg.

**Submissions:**

| Version | Folds | Phase i wall-clock | Phase ii wall-clock | LB |
|---|---|---|---|---|
| v71 | V2-S 4-fold {0,1,2,4} | 752s (12.5 min) | ~101 min — TIMEOUT | none (empty publicScore) |
| v72 | V2-S 3-fold {0,1,2} | 610s (10.2 min) | ~84 min — landed | **0.930** |
| (reference: prod B0 v68 / Apr 29) | B0 4-fold {0,1,2,4} | — | — | 0.931 |

**Decision rule (§14.19.5 Phase 4):**
- v72 score 0.930 lands in the **0.927-0.931 noise band** tier.
- Per spec: "within noise; repeat seed42→seed7 fold-0 retrain to
  disambiguate, OR kill if budget tight".
- Killed. Rationale: every prior "+val landed at noise band" probe
  in the §14.17 series killed at the same outcome (T1.3, T2.1,
  T2.3, P8, P12, B2, A2, A2-iter1.5, T2.6 BC2025, TopN, M1, M2,
  L1, L2, L5b). The empirical pattern across 14+ levers is
  unbroken: single-arch swaps and single-lever transplants don't
  lift LB on this stack. A seed42→seed7 disambiguating probe
  burns 1 daily Kaggle slot to wobble ±0.003; doesn't address
  the binding question (was 4-fold V2-S going to land cleanly?
  We can't test that without first solving the inference budget,
  which means another Kaggle slot for a probe that's already
  derisked-by-pattern as null).

**Calibration / lessons:**

1. **Kaggle vCPU is 3-4× slower than DT 4-vCPU on V2-S, not 1.5-2×.**
   Phase 1.5's caveat ("realistic 1.5-2× per-core slowdown on
   Kaggle → 40-53 min … pessimistic 3× ~80 min is at the edge")
   underpriced the slowdown. Actual: 4-fold V2-S Phase ii ≈ 101 min
   (3-4× DT-extrapolated 24.2 min, 1.4-1.7× over the budget midpoint
   I'd planned for). DT-extrapolation as a budget proxy is unsafe
   for V2-S-class arches.

2. **+val→LB transfer is even weaker than the L1 leak finding suggested.**
   - L1 NS-fold0: val 0.9042 → LB 0.930 (−0.001, val-leakage)
   - V2-S 3-fold: val mean 0.7572 (per-fold) / 0.7530 (per-fold 4-fold) → LB 0.930 (−0.001)
   - Even with the GT-only val_v2 substrate (built explicitly to dodge
     the L1 leak pattern), +val didn't transfer. The val-LB gap is
     bigger than just the prior leak diagnosis — there's a structural
     fold-coverage / class-distribution gap between the 75-species
     soundscape val and the 234-class hidden test that even
     non-leaked val signals don't bridge.

3. **The "clean" 4-fold val signal (+0.031 over B0) was not predictive
   of LB lift.** Every fold non-regressed, fold spread tightened by
   ~half — the kind of pattern you'd expect from a structurally
   better arch. The signal was real for val, useless for LB. Update
   gating heuristics: a +0.03-class val Δ from arch swap is now
   classified as "noise-band predicted" rather than "real-lever
   predicted" until a probe actually lands clean LB on this stack.

4. **Sydorskyi's +0.046 from arch+corpus (per the BC2025 winner audit)
   does NOT decompose linearly into +0.005-0.015 arch + ~+0.030
   corpus.** The arch-alone half is null on our pipeline.
   Implication for L2-redux v1 (B0 + corpus) and Phase 5 (V2-S +
   corpus): the corpus may also fail to deliver the projected lift
   if it's similarly bundle-locked. The BC2025-winners-recipe-as-
   transplant skepticism (already established by M1, M2, T2.1
   kills) extends here.

**Reverts applied:**
- `jupyter/protossm-postproc/kernel-metadata.json`: removed
  `stevewatson999/birdclef-2026-a1-effv2s-ckpts` from
  `dataset_sources`. Production kernel inputs back to 5 datasets.
- `birdclef2026-protossm-postproc.ipynb` cell 41: removed the
  `A1_BACKBONE_VARIANT` toggle, hardcoded `A1_CKPT_DIR =
  Path("/kaggle/input/birdclef-2026-a1-effb0-ckpts")`, restored
  `A1_FOLDS = [0, 1, 2, 4]`. Killed-banner comment captures the
  reason inline.
- V2-S JIT artifacts left in place at
  `four_track/kaggle_datasets/a1-effv2s-ckpts/` (322 MB local,
  unused). Don't `kaggle datasets delete` — the dataset on Kaggle
  is private + owned by us; leaving it costs nothing and preserves
  the option to retest.
- `kaggle_datasets/_backups/a1_fold{N}_effb0_LB0931_20260503.pt`
  remain available if any production B0 ckpt gets accidentally
  overwritten in future.

**Implications for queue going forward:**

- **Phase 5 (V2-S + L2-redux corpus together) is now downgraded.**
  Originally framed as the gold-zone shot (+0.010 to +0.025
  envelope). Without arch-alone clearing noise, the corpus-alone
  decomposition argument weakens. Don't auto-commit Phase 5 if
  L2-redux v1 (B0 + corpus) lifts LB — re-evaluate the EV first.
- **L2-redux v1 (B0 + corpus) is now the primary remaining lever.**
  Per §14.17.16.8 still in flight on skynet, ETA ~May 7. Single-LB-
  probe outcome will inform whether the §14.17 / §14.19 lever-kill
  pattern extends to in-domain pretrain corpora too.
- **Don't queue another arch swap on this stack** (NFNet-L0, B3,
  multi-arch ensemble) without first proving an LB lift via a
  different lever class. Pattern-matching: 14 single-lever kills,
  no exception, queueing more single-lever transplants is
  predictably wasteful.

**Next pickup (2026-05-04+):** L2-redux XC v3 download still in
flight on skynet (PID 1029580). When it completes ~May 7,
rebuild skynet `kaggle` env from environment.yml (with the
`solver: libmamba` + cu128 fixes flagged by the kaggle-arch
build trap), then run Sydorskyi-recipe pretrain on B0. See
§14.17.16.8 for in-flight state.

---

## ⏸️ PICK UP HERE — previous (2026-05-03 ~00:10 local — Phase 2 4-fold V2-S finetune IN FLIGHT on both machines — SUPERSEDED by §14.19.10/.11)

**TL;DR (2026-05-03 ~00:10 local):**

TWO independent experiments running in parallel:

1. **L2-redux v1 (B0 + corpus)** — XC v3 download still in flight on
   skynet (PID 1029580, ~30% complete, ~50h elapsed, ETA ~May 7). After
   download: pretrain B0 on 819K-rec corpus (~2 days) → finetune
   (~1 day) → LB probe ~May 11. Big-swing lever, +0.005 to +0.020
   envelope. See §14.17.16.8 for the in-flight state.

2. **§14.19 EffNetV2-S arch swap** — Phase 2 LAUNCHED 2026-05-03 00:07.
   - Folds 0,1 on skynet (`kaggle-arch` sidecar env still building, PID
     151474 → watcher PID 151732 launches train when ready)
   - Folds 2,4 on deepthought (PID 944726, fold 2 epoch 1 in progress)
   - Wall-clock ~2.5-3h based on smoke extrapolation (NOT the 11h
     pessimistic envelope)
   - **LB probe ETA: ~03:00-04:00 EDT today** (after JIT export +
     Kaggle dataset push)
   - Smaller-swing lever, +0.000 to +0.010 envelope, but high
     decision-quality value: a clear pass de-risks Phase 5 (V2-S +
     L2-redux corpus, the gold-zone shot)
   - Phases 0, 1, 1.5 all PASSED — see §14.19.5 phase updates inline

The two experiments are orthogonal (no shared data, separate processes,
separate GPUs). Their LB results jointly inform whether Phase 5 (~5
days, +0.010 to +0.025 envelope) is worth committing post-May 11.

**Read first:** §14.19 Phase 2 update (in flight, ~03:00 EDT result).
§14.20 multi-machine infra. §14.17.16.8 for the live L2-redux download.
The §14.19.4 risks have all been validated (Phase 0/1/1.5 passed); the
arch swap is no longer a structural-feasibility question, only a
training-delta question.

**Awaiting (in time order):**
1. ~03:00-04:00 EDT today: Phase 2 training completion → JIT export
   → Kaggle push → LB probe → §14.19 Phase 4 decision rule.
2. ~May 7: L2-redux XC v3 download completes → skynet `kaggle` env
   rebuild from `BirdCLEF/environment.yml` is unblocked → L2-redux
   pretrain phase begins.
3. ~May 11: L2-redux v1 LB result.
4. Decide on Phase 5 commitment (V2-S + L2-redux corpus together, ~5
   days) based on (1) and (3) joint evidence.

**Don'ts (additive to §14.17.16.8 don'ts):**
- Don't modify `src/config.py` BACKBONE — the new arch is a CLI override,
  not a default change. Keeps the B0 baseline reproducible.
- Don't touch the `kaggle` conda env in-place. Clone to `kaggle-arch` for
  the timm/torchvision fix so a torch-version pin doesn't kill the live
  download (PID 1029580 is using the existing env's torch).
- Don't bundle SoftAUCLoss, TopN, or any other lever with arch swap. We
  need clean attribution.
- Don't skip Phase 1.5. B2 burned 5 Kaggle slots on a model that couldn't
  inference; don't repeat.

---

### Overnight orchestration contract (2026-05-03 ~00:30 local)

User went to sleep at ~00:30 local. Below is the autonomous boundary
for any agent picking up this session before user wakes. **Do NOT
violate the "needs approval" line without an explicit user message.**

**Will run autonomously (local, reversible):**

1. Wait for skynet env build (PID 151474 on skynet) to exit → watcher
   PID 151732 launches `train_a1.py --folds 0,1 --backbone
   tf_efficientnetv2_s.in21k_ft_in1k --loss hybrid` in `kaggle-arch`
   env. Log: `four_track/log/spark_v2s_folds_01.log`.
2. Wait for deepthought training (PID 944726 on deepthought) to
   complete folds 2,4. Log on deepthought:
   `_runon/BirdCLEF/log/runon_deepthought_20260503_000717.log`.
3. After both chains complete: `syncback deepthought models/ log/`
   pulls deepthought ckpts to skynet's
   `four_track/models/a1/a1_tf_efficientnetv2_s.in21k_ft_in1k_fold{2,4}_seed42_hybrid.pt`.
4. Patch `four_track/src/export_a1_jit.py` to accept `--backbone` arg
   (per §14.19.8 file table — keeps B0 path as default).
5. Archive existing `four_track/kaggle_datasets/a1-effb0-ckpts/` to
   `_backups/` per `feedback_backup_ckpts_before_overwrite` (does NOT
   touch the live a1-effb0-ckpts; just creates a copy).
6. JIT-export the 4 V2-S fold ckpts to
   `four_track/kaggle_datasets/a1-effv2s-ckpts/`. **Local only — does
   NOT push to Kaggle.**
7. Update §14.19 Phase 2 results inline (new RESULT subsection).
   Compute fold-mean val_v2 macro-AUC. Compare to gate 0.7414
   (matches B0 LB-0.931 baseline).

**WILL NOT do without explicit user approval (visible to others /
external state):**

- `kaggle datasets create -p kaggle_datasets/a1-effv2s-ckpts/` (new
  public dataset)
- `kaggle datasets version -p ...` (new dataset version)
- Modify `jupyter/protossm-postproc/kernel-metadata.json` to point
  at the new dataset (would auto-trigger a kernel submission)
- `kaggle kernels push` for any kernel version
- Any commit / PR / issue / external comment

**Failure modes the user expects to be flagged in morning report
(NOT acted on unilaterally):**

- One or both folds OOM → log entry + recommendation (BS=32 retry,
  or skip-fold strategy)
- Val numbers below 0.5 macro-AUC → flag, recommend skip Kaggle push
- Watchers crash → trainings continue (nohup'd); manually `syncback`
  on next pickup
- HF Hub auth issue → cached weights from skynet may not be reused
  on deepthought (tested; works, but log it if it shifts)

**Morning report format the user expects** (write into a fresh
§14.19 Phase 2 RESULT subsection + summarize at top of next PICK UP
HERE update):

| Fold | Machine | Wall-clock | best val_v2 macro-AUC | epoch of BEST |
|---|---|---|---|---|
| 0 | skynet | … | … | … |
| 1 | skynet | … | … | … |
| 2 | deepthought | … | … | … |
| 4 | deepthought | … | … | … |

Plus: fold-mean val_v2, comparison to 0.7414 gate, §14.19 Phase 4
decision-rule recommendation (push to Kaggle / skip / repeat seed).

**Reference state for cross-checking on pickup:**
- L2-redux download PID 1029580 (skynet) — DO NOT touch.
- skynet env build PID 151474 — exit signals readiness for skynet
  train.
- skynet train watcher PID 151732 — fires train when env build exits.
- deepthought train PID 944726 — folds 2,4 chain.
- deepthought completion watcher PID 151891 (skynet) — fires when
  DT done.
- environment.yml lives at BirdCLEF git root.
- runon docs: `~/work/MachineLearning/DOCUMENTATION/RUNON.md`.

---

## 14.20 Multi-machine (deepthought RTX 4080) infrastructure (2026-05-02 ~23:15 local)

**Summary:** A second GPU host (`deepthought`, NVIDIA RTX 4080 16 GB,
CC 8.9) is now wired into the workflow via a 3-script SSH/rsync wrapper
at `~/bin/runon`, `~/bin/syncback`, `~/bin/runon-setup`. Configured in
`~/.runon.conf`. Documentation: `~/work/MachineLearning/DOCUMENTATION/RUNON.md`.

This is project-agnostic infrastructure — works for any future ML
project, not just BirdCLEF. The runon scripts and config are local to
skynet (not committed to any repo); the documentation lives in the
shared `DOCUMENTATION/` tree.

### 14.20.1 What this enables for BirdCLEF specifically

| Workflow | Wall-clock saving | Status |
|---|---|---|
| Split 4-fold finetune across both machines (folds 0,1 on GB10; 2,4 on 4080) | ~22 h → ~11 h | Ready (after data sync) |
| Parallel arch-swap pretrain (B0 on GB10, EffNetV2-S on 4080, same XC v3 corpus) | Collapses §14.19 Phase 5 from sequential to concurrent | Ready (after L2-redux download finishes May 7 + data sync) |
| Arch-swap Phase 0/1/1.5 today on deepthought (clean torch env, idle GPU) | Bring §14.19 Phase 1.5 inference probe forward by ~5 days | **Ready now** — env build complete |
| CPU inference probe in clean env (no NVRTC trap) | Avoids confounds from skynet's broken torchvision | **Ready now** |

### 14.20.2 What's done

- `~/bin/runon`, `~/bin/syncback`, `~/bin/runon-setup` installed on skynet.
- `~/.runon.conf` configured for `deepthought` host.
- `BirdCLEF/environment.yml` shipped at the BirdCLEF git root with
  matched torch 2.7.1 + torchvision 0.22.1 + torchaudio 2.7.1 + timm
  1.0.22 pins. Python 3.11.
- `kaggle` conda env built fresh on deepthought from environment.yml
  (verified torch + RTX 4080 reachable).
- BirdCLEF code mirrored to deepthought at
  `/home/swatson/work/MachineLearning/_runon/BirdCLEF/` (~11 MB, code
  only — heavy paths excluded).

### 14.20.3 What's NOT done — explicit follow-up TODOs

These are the two outstanding items the user should expect to come
back to:

1. ~~**Hot data sync to deepthought**~~ DONE 2026-05-02 23:37 local.
   23 GB synced via targeted rsync (NOT `--push-data`, which would have
   pulled the live 137 GB XC v3 download too). On deepthought now:
   `data/raw/train_audio/` (11 GB, 35,549 ogg), `train_soundscapes/`
   (5.1 GB, 10,658 ogg), `data/processed/` (6.7 GB, incl. train_folds.csv),
   `four_track/data/external/anuraset_focal/` (365 MB). 2.6 TB free on
   deepthought. **A1 training is now data-ready on deepthought.** When
   L2-redux corpus is needed for arch-swap Phase 5, sync separately
   (~100 GB, ~20 min over LAN).

2. **Spark `kaggle` env rebuild from environment.yml** (after L2-redux
   download finishes ~May 7). Spark's current env has the broken
   torchvision/torchaudio version skew (§14.19 Phase 0 update). Rebuild:
   ```
   # ONLY after PID 1029580 (XC v3 download) exits successfully
   conda deactivate
   conda env remove -n kaggle
   cd /home/swatson/work/kaggle/BirdCLEF
   conda env create -f environment.yml
   ```
   Don't do this earlier — it kills the live pretrain-corpus download.

### 14.20.4 Caveats already known

- **deepthought is multi-tenant.** 40+ conda envs visible there (gemma3,
  HunyuanVideo, opensora, etc.). Before queuing a long job: `ssh
  deepthought nvidia-smi` to confirm 4080 is idle. The `runon` wrapper
  does NOT block on this — caller's responsibility.
- **CC 8.9 vs 12.1 mismatch on JIT artifacts.** Anything `torch.jit.trace`d
  on one machine may not load on the other. For Kaggle CPU export, doesn't
  matter. For cross-machine JIT inference, it does.
- **Network is gigabit-class** (~50-110 MB/s). 25 GB data sync = ~5-10 min;
  100 GB pretrain corpus = ~20 min. Acceptable for one-time syncs.
- **Symlink/parent-path imports**: BirdCLEF's `model_a1.py` does
  `Path(__file__).resolve().parents[2] / "src"` to reach the parent
  `BirdCLEF/src/`. On deepthought, the project mirror at
  `_runon/BirdCLEF/` preserves the four_track/src→../src layout, so this
  works. If we ever shorten paths, recheck.

---

## 14.21 Strategy pivot — switch to multi-lever bundles, drop single-lever probes (2026-05-03 ~23:50 local)

**TL;DR:** After 14 single-lever probes in §14.17/§14.19 (T1.3, T2.1,
T2.3, P8, P12, B2, A2, A2-iter1.5, T2.6 BC2025, TopN, M1, M2, L1, L2,
L5b, V2-S — the count is actually 15+ if you split by sub-iteration)
all landed in the LB noise band or regressed, decided 2026-05-03 to
**stop queueing single-lever transplants** and switch to **multi-lever
bundles** that mirror the actual BC2025 winning recipes. View-B switch
codified in `feedback_view_b_strategy_2026_05_03.md`.

### 14.21.1 The kill-streak that motivated the switch

Single-lever probes since 2026-04-19 with their LB outcomes (vs B0
baseline 0.931 / floor 0.926 ± 0.005 noise):

| Probe | Lever | LB | Δ vs 0.931 | Tier |
|---|---|---|---|---|
| T1.3 | min-reduce fold ensemble | 0.925 | −0.006 | regression |
| T2.1 | CE loss | killed at val gate | — | val-fail |
| T2.3 | hybrid soup ckpt-avg | killed at val gate | — | val-fail |
| P8 5-fold | RMS segment select | 0.928 | −0.003 | noise |
| P12 | per-class isotonic calib | 0.868 | −0.063 | catastrophic |
| B2 | ConvNeXt-tiny SED | timeout / 0.928 | — | timeout/noise |
| A2 | iter-1 pseudo-label finetune | killed at val gate | — | val-fail |
| A2-iter1.5 | cap_k=10 pseudo cap | killed at val gate | — | val-fail |
| T2.6 | BC2025 add data | 0.927 | −0.004 | noise |
| TopN | BC2025 N=1 finishing post-proc | 0.926 | −0.005 | noise |
| M1 | Melichov coarser-mel | killed at val gate | — | val-fail |
| M2 | multi-layer GeM | killed at val gate | — | val-fail |
| L1 | NS fold-0 distill | 0.930 | −0.001 | noise |
| L2 | multi-year focal pretrain | killed at val gate | — | val-fail |
| L5b | AnuraSet amphibia mixup | 0.780 | −0.151 | catastrophic |
| **V2-S** | **EffNetV2-S arch swap** | **0.930** (3-fold) / timeout (4-fold) | **−0.001** | **noise** |

**16 probes. Zero LB-clearing exceptions.** Pattern is empirically
unbroken on this stack.

### 14.21.2 Three bundle compositions considered

Discussed 2026-05-03 ~23:45 local before user picked Bundle 1:

**Bundle 1 — "Sydorskyi-replica" (SELECTED)**: V2-S arch + L2-redux
819K-rec/7,489-species in-domain pretrain + iterative noisy student
recipe (Babych BC2025 1st-place's contribution: soft labels +
threshold + OOF) + Sydorskyi-specific recipe tweaks (label smoothing,
RandomFiltering). 3-5 stacked levers. Closest to what actually won
BC2025. Time: ~2-3 weeks of compute. Highest EV per attempt, fewest
experiments before deadline.

**Bundle 2 — "minimum-bundle"**: B0 (keep) + L2-redux corpus +
iterative NS recipe. 2 levers. Tests "does any bundling lift?"
without re-committing to V2-S whose arch-alone null raised additivity
questions. ~1-2 weeks. Rejected: doesn't move far enough from the
single-lever pattern to break it.

**Bundle 3 — "Sydorskyi-strict"**: replicate exactly one row of
Sydorskyi's ablation table where the claimed-Δ is largest. Cleanest
attribution (compare claimed vs measured Δ directly). Rejected: the
attribution gain comes at the cost of locking us to Sydorskyi's stack
shape, which may not be optimal on top of our existing
ProtoSSM/Perch/B1 ensemble.

### 14.21.3 Bundle 1 execution plan

**Levers in scope:**
1. **EffNetV2-S backbone** (already trained on focal in §14.19, val
   pass demonstrated, LB-alone null demonstrated). Reuse existing
   `models/a1/a1_tf_efficientnetv2_s.in21k_ft_in1k_fold{0,1,2,4}_seed42_hybrid.pt`
   ckpts as the *baseline-arch* for the bundle. Will be replaced by
   pretrained-and-NS'd ckpts after Bundle 1 phases.
2. **L2-redux corpus pretrain** (819K rec / 7,489 species, in-flight
   download per §14.17.16.8, ETA ~May 7). Pretrain V2-S backbone-only
   on this corpus before BC2026 finetune.
3. **Iterative noisy student** per Babych BC2025 1st-place: soft
   pseudo-labels + per-class threshold + OOF teacher inference.
   Iterations: 2 (teacher → student-iter-1 → student-iter-2). On the
   BC2025 unlabeled audio + BC2026 train_audio focal.
4. **Sydorskyi recipe tweaks**: label smoothing (per his ceur-ws
   table, ~0.05-0.1), RandomFiltering augmentation. Applied at
   finetune time.

**Sequencing** (post-2026-05-07 download completion):

| Phase | Step | Compute | Wall-clock | Decision gate |
|---|---|---|---|---|
| 5.0 | Skynet `kaggle` env rebuild from environment.yml + cu128 + libmamba fixes | skynet | ~30 min | env smoke import ok |
| 5.1 | V2-S backbone pretrain on 819K-rec corpus | DT (preferred per `feedback_default_to_deepthought_for_training`) | ~2-3 days | pretrained backbone loads cleanly + perplexity decreases |
| 5.2 | NS iter-1: emit pseudos via pretrained V2-S teacher → train V2-S student on focal+pseudos w/ Sydorskyi tweaks | DT | ~3 days | val_v2 fold-mean ≥ 0.7530 (V2-S arch-only baseline) |
| 5.3 | NS iter-2: regenerate pseudos via iter-1 student → train iter-2 student | DT | ~3 days | val_v2 ≥ iter-1 fold-mean (don't iterate if no lift) |
| 5.4 | Final 4-fold finetune on BC2026 focal w/ best NS-iter ckpt as init | DT | ~1 day | val_v2 fold-mean ≥ 0.7530 |
| 5.5 | JIT export 3-fold (drop fold 4 per §14.19.11 budget calibration: V2-S 4-fold > Kaggle cap) | skynet | ~5 min | CPU-load smoke + BS=16 wall-clock < 80 min projected |
| 5.6 | Kaggle dataset push + kernel-metadata + cell 41 toggle + push v73 | skynet | ~30 min | per `feedback_kernel_timeout_vs_scoring_stall` Phase ii landing |

**Total wall-clock estimate: ~10-12 days** from corpus-download completion (~May 7) → first LB result around **May 17-19**.

**Inference budget pre-commitment (per §14.19.11 calibration):**
- Bundle 1 final ensemble = **3-fold V2-S** (drop fold 4). 4-fold
  empirically blew Kaggle's ~90-min cap; bundle changes don't reduce
  per-fold inference time (NS, label smoothing, RandomFiltering are
  all training-time). Don't repeat the v71 timeout mistake.
- 3-fold V2-S projected ~84 min Phase ii on Kaggle, 6-min margin.
- If Bundle 1 lands clean LB at 3-fold, the 4-fold dream is
  permanently off until either Kaggle's cap changes or we cut
  another component (TTA, ProtoSSM epochs, etc.).

### 14.21.4 Decision rules at the LB gate (Phase 5.6)

Same shape as §14.19.5 Phase 4 but tightened given the noise-band
priors:

- **LB ≥ 0.945** → real lever, View-B vindicated. Queue Bundle 1.5
  (re-add fold 4 if budget can accommodate via other trims, e.g.
  ProtoSSM 30→20 epochs).
- **LB 0.937-0.944** → real lift, ship as production. View-B
  partially vindicated. Re-evaluate whether further bundles (e.g.
  Bundle 1 + Babych-NS-iter-3) are worth committing.
- **LB 0.932-0.936** → marginal lift above noise. Real but small.
  Ship; don't commit to further bundles without a stronger story.
- **LB 0.927-0.931** → still in noise band. **View-B kill** — if even
  a 4-lever Sydorskyi-replica doesn't lift, single-lever bundling
  isn't the missing ingredient. Pivot decision required: ensemble-
  diversity approach (multiple architectures, different folds/seeds
  averaged), explicit calibration alignment between A1 and ProtoSSM
  paths, or accept floor and stop spending Kaggle slots.
- **LB ≤ 0.926** → catastrophic regression. Revert immediately,
  reconsider whether the bundle's recipe components are mutually
  compatible (e.g. NS soft labels + label smoothing may double-soften
  targets, harm calibration).

### 14.21.5 What's deliberately NOT in Bundle 1

- **NFNet-L0 / B3 / multi-arch ensemble** — defer to Bundle 2 if
  Bundle 1 lifts. Single-arch swap was already null; multi-arch
  ensemble adds wall-clock + JIT export complexity without addressing
  the additivity story.
- **ProtoSSM modifications** — keep ProtoSSM training as-is at submit-
  mode caps. Bundling V2-S retrain with ProtoSSM retrain confounds
  attribution.
- **B1/Perch tuning** — keep production B1 weight (w=0.10) and Perch
  config from `project_b1_weight_sweep_closed`.
- **Babych iterative NS beyond 2 iterations** — Babych's table shows
  diminishing returns past iter-2; iter-3 is gated on iter-2's lift.
- **Quantile-Mix (L3-style)** — already killed; not part of Bundle 1.

### 14.21.6 Reference pointers

- BC2025 winners audit: `reference_bc2025_winners_writeups.md`
- L2-redux scope and in-flight state: §14.17.14 / §14.17.16.8
- V2-S kill record: §14.19.11, `project_v2s_killed.md`
- Kaggle CPU calibration: `feedback_kaggle_cpu_3to4x_dt_slowdown.md`
- View-B strategic switch: `feedback_view_b_strategy_2026_05_03.md`

---

## ⏸️ PICK UP HERE — previous (2026-05-03 ~19:30 local — V2-S Phase 2 val pass, awaiting Kaggle push — SUPERSEDED by §14.19.11)

**TL;DR (2026-05-03 ~19:30 local):**

V2-S arch swap Phase 2 cleared the val gate by **+0.0116** (4-fold mean
0.7530 vs gate 0.7414). Every fold non-regressed vs B0 baseline; fold
spread tightened by ~half (V2-S 0.024 vs B0 0.044). Inference budget on
Kaggle re-validated (DT 4-vCPU, BS=16: V2-S 24.2 min for 4-fold
ensemble, well under 78-min A1 budget). Full results in §14.19.10.

**Deliverables locally:**
- 4 V2-S fold ckpts at `four_track/models/a1/a1_tf_efficientnetv2_s.in21k_ft_in1k_fold{0,1,2,4}_seed42_hybrid.pt`
- 4 JIT-traced artifacts at `four_track/kaggle_datasets/a1-effv2s-ckpts/a1_fold{0,1,2,4}.pt` + `dataset-metadata.json`
- B0 production ckpts archived to `four_track/kaggle_datasets/_backups/a1_fold{N}_effb0_LB0931_20260503.pt`
- `export_a1_jit.py` patched with `--backbone` arg + `BACKBONE_VARIANTS` table
- `four_track/CLAUDE.md` updated with Two-GPU workflow rule (deepthought-as-workhorse + 4:1 split ratio when N ≥ 5)
- Memory entries: `feedback_default_to_deepthought_for_training.md`, `feedback_clean_logs_before_training.md` (case A/B framework)

**Awaiting user approval before any of these (visible to others / external state):**

1. `kaggle datasets create -p kaggle_datasets/a1-effv2s-ckpts/` — new
   public dataset `stevewatson999/birdclef-2026-a1-effv2s-ckpts`.
2. Add the new dataset id to
   `jupyter/protossm-postproc/kernel-metadata.json` (keep
   `a1-effb0-ckpts` as the production fallback).
3. Add `A1_BACKBONE_VARIANT` toggle to notebook cell 41 (default
   `effb0`; flip to `effv2s` for the LB probe).
4. `kaggle kernels push` for the V2-S-toggled kernel version → LB
   probe → §14.19.5 Phase 4 decision-rule outcome.

**Other open follow-ups (NOT BLOCKING the LB probe):**

- environment.yml: change `--index-url cu126` → `cu128` so future
  aarch64 env builds get GPU torch (today's caught fix was in-flight).
- `~/.condarc`: flip `solver: classic` → `solver: libmamba` to avoid
  the 14.5 h hang we hit today on `kaggle-arch` build. Both edits
  gated on user approval.
- L2-redux XC v3 download still in flight on skynet (PID 1029580,
  ~50% complete, ETA ~May 7). Untouched.
- Phase 5 (V2-S + L2-redux corpus) decision waits on jointly clearing
  Phase 4 LB AND L2-redux v1 LB on B0. Per §14.19.5: only commit
  Phase 5 if both prior probes clear noise band.

**Don'ts on next pickup:**
- Don't push to Kaggle without explicit user approval (above 4 items
  are external-state).
- Don't run skynet for any new training task — see
  `feedback_default_to_deepthought_for_training.md`. DT-only or
  4:1-split favoring DT (only when N ≥ 5).
- Don't blanket `rm -f log/*.log` while the L2-redux download is
  writing — use the surgical filter
  (`rm -f log/train_*.log`) per the case A/B framework in
  `feedback_clean_logs_before_training.md`.

---

## ⏸️ PICK UP HERE — previous (2026-05-03 ~23:50 local — V2-S KILLED, View-B switch, Bundle 1 selected — SUPERSEDED by §14.21.7)

**TL;DR (2026-05-03 ~23:50 local):**

V2-S arch swap is dead. v71 (4-fold) timed out at ~101 min Phase ii.
v72 (3-fold, dropped fold 4) scored **LB 0.930 vs B0 baseline 0.931
= −0.001 within ±0.005 noise**. Reverted: kernel-metadata.json no
longer references `a1-effv2s-ckpts`, cell 41 hardcoded back to B0
4-fold. Full Phase 4 outcome in §14.19.11.

**Strategic shift 2026-05-03:** with 16 single-lever probes in
§14.17/§14.19 all landing in noise band or worse (zero LB-clearing
exceptions), switched to **View B — multi-lever bundles only**. No
more standalone L2-redux/NFNet-L0/Babych-NS probes. **Selected
Bundle 1** (Sydorskyi-replica: V2-S + L2-redux corpus + iterative NS
+ Sydorskyi recipe tweaks). Full scope, sequencing, decision rules,
and out-of-scope items in **§14.21**. Memory:
`feedback_view_b_strategy_2026_05_03.md`.

**State as of pickup:**
- Production kernel = B0 4-fold {0,1,2,4}, LB-0.931 baseline
- V2-S JIT artifacts and ckpts left on disk + on Kaggle (private
  dataset). Don't delete; reused as the *baseline-arch* input for
  Bundle 1's pretrain phase.
- L2-redux XC v3 download still in flight on skynet (PID 1029580,
  ~50% complete, ETA ~May 7). Untouched.
- Today used 2/5 daily Kaggle slots (v71 timeout, v72 = 0.930).
- skynet `kaggle` env still has the broken torchvision/torchaudio
  skew. `kaggle-arch` sidecar env exists with cu128 in-flight patch.
  `environment.yml` still declares cu126 which falls back to CPU on
  aarch64 — fix flagged for the post-May-7 rebuild (see §14.21
  Phase 5.0).

**Next: Bundle 1 execution — see §14.21.3 for the full phase table.**
Wall-clock estimate ~10-12 days from corpus-download completion
(~May 7), first LB result around **May 17-19**. The selected
final-ensemble shape is **3-fold V2-S** (4-fold permanently
ruled out by Kaggle inference budget per §14.19.11 calibration).

**Don'ts on next pickup:**
- **Don't queue any single-lever probe.** View-B is in effect — only
  multi-lever bundles per §14.21. If something seems "cheap and
  worth a try" outside Bundle 1's scope, it isn't; the §14.17 kill
  pattern says it'll land in noise band.
- Don't auto-commit Bundle 2 / NFNet-L0 / Babych-iter-3 etc. — those
  are gated on Bundle 1's LB outcome per §14.21.4.
- Don't push the skynet `kaggle` env rebuild from environment.yml
  without applying the cu126→cu128 + libmamba fixes — that's the
  trap that caused the 14.5 h conda solve hang today.
- Don't blanket `rm -f log/*.log` while the L2-redux download is
  writing — use the surgical filter per
  `feedback_clean_logs_before_training.md` case A/B framework.
- Don't push to deepthought without `ssh deepthought nvidia-smi`
  pre-flight — multi-tenant GPU.
- Don't try 4-fold V2-S on Kaggle again. v71 timed out at 101 min;
  the budget cap is empirically below 4-fold V2-S's footprint. Stay
  at 3-fold for any V2-S-based bundle including Bundle 1.

---

## 14.21.7 Bundle 1 prep work — scripts ready, recipe finalized (2026-05-04 ~16:30 local)

While L2-redux XC v3 download is in flight (~45% done, recent rate ~2,500
dl/hr stable, ETA recalculated to ~May 9 not May 7), used the wait time
for parallel-safe prep work. Nothing LB-affecting was run; no Kaggle
slot used; no single-lever probe fired (View-B respected).

### 14.21.7.1 Recipe ambiguity resolution: Sydorskyi over Babych

§14.21 originally said "iterative NS per Babych BC2025 1st-place: soft
pseudos + per-class threshold + OOF teacher inference." That spec is
internally inconsistent with `reference_bc2025_winners_writeups.md`:

- Babych's actual recipe is **3-stage SoftAUCLoss** (custom pairwise
  diff + log-loss; supports soft labels). Numerically thin in the
  reference; would need to re-derive the loss from scratch.
- The "soft + per-class threshold + OOF" recipe is **Sydorskyi's**:
  max≥0.5 chunk filter, <0.1 per-class zero, soft targets only, 40%
  pseudo / 60% real per train-step, MixUp at audio level (element-wise
  sum + clip [0,1]), OOF-fold pseudos.

Picked **Sydorskyi** for reproducibility — recipe is documented to
specific numeric thresholds and a concrete sampler rule, so attribution
of any LB delta is cleaner than chasing Babych's SoftAUCLoss without a
reference implementation.

### 14.21.7.2 Bundle 1 levers — what's IN and what's deferred

**IN (3 levers, not 4):**
1. EffNetV2-S backbone (existing focal-only ckpts as baseline-arch
   start; replaced by pretrained-and-NS'd ckpts after Phase 5.1-5.3)
2. L2-redux corpus pretrain (819K-rec / 7,489-species — the BC2025
   table's biggest ablation Δ at +0.046)
3. Sydorskyi-recipe iterative NS, 2 iterations

**DEMOTED to "deferred / probably free-rider":**

4. ~~Sydorskyi recipe tweaks~~ (label smoothing, RandomFiltering,
   secondary equal-weight, BG mix, Focal+BCE bundle). Per his own
   ablation table this entire bundle = **0.835 vs 0.837 baseline =
   no public LB lift** (helped private only). Demoted from Bundle 1.
   If Bundle 1 lifts and we want Bundle 1.5, revisit individually
   with proper attribution.

So Bundle 1 is technically **3-lever** not 4. Still multi-lever per
View-B, but the recipe is now honest about which components are
load-bearing.

### 14.21.7.3 Scripts written / extended

| File | Status | LOC | Notes |
|---|---|---|---|
| `src/pretrain_l2_redux.py` | extended | +90 | New `--backbone`, `--include-xc-bulk` flags. New `_walk_xc_bulk()` reads per-species `_meta.json` for recordist (XC v3's `rec` field). Separate manifest cache (`l2_redux_manifest_with_xc.csv`) so existing pre-XC manifest stays reproducible. |
| `src/pseudo_emit_sydorskyi.py` | new | 310 | Mean-ensemble OOF-fold inference. `--keep-thresh 0.5`, `--zero-thresh 0.1` defaults match Sydorskyi. `--shard-total/--shard-id` for multi-machine I/O-bound parallelism. |
| `src/train_a1_ns_sydorskyi.py` | new | 410 | `SydorskyiNSDataset`: 40% pseudo / 60% real per `__getitem__`, audio-level MixUp (sum + clip [0,1]), BCE-on-soft. Reuses `train_a1._load_pretrained_backbone` for `--init-from`. GPU memory hygiene per CLAUDE.md. |
| `src/train_a1.py --init-from` | already existed | 0 | Line 484 + `_load_pretrained_backbone()` (line 158) drops cls_conv/att_conv[4] for class-count mismatch. No change needed. |

### 14.21.7.4 Smoke results — both machines

| Smoke | Where | Result | Wall-clock |
|---|---|---|---|
| `build_manifest(include_xc_bulk=True)` | skynet `kaggle-arch` | 308,933 clips / 3,696 species on **partial** XC bulk (3,060/6,718 species so far). Will grow to ~570K / ~7,400 once XC download lands. | ~30 s |
| `pseudo_emit_sydorskyi.py --smoke-test` | skynet `kaggle-arch` | 1 BC2025 SS clip → 3 chunks → 1 kept (33%) at thresh 0.5; NPZ shape (1, 234) ✓ | 2.2 s |
| `pseudo_emit_sydorskyi.py --smoke-test` | DT `kaggle` (cross-machine) | 1 BC2026 train_soundscapes clip → 3 chunks → 1 kept; same ckpts cross-load OK (CC 12.1 → CC 8.9 not exercised — DT ckpts produced on DT) | 2.0 s |
| `train_a1_ns_sydorskyi.py --smoke-test` | skynet `kaggle-arch` | epoch 1 train_loss=0.8938, ckpt saved, val=0.0 (smoke max_batches=1 sanity expected) | ~4.5 min (val mel build dominates) |
| `train_a1_ns_sydorskyi.py --smoke-test` | DT `kaggle` (cross-machine) | epoch 1 train_loss=0.8980 (within fold-0 noise floor of 0.03 — `feedback_single_fold_noise_floor`) | 1m 34s |

Manifest cache from skynet smoke deleted post-test (would have been
stale at the post-download launch).

### 14.21.7.5 Deployment shape — 4:1 rule applied per phase

The user clarified "use both GPUs as much as possible **within the
framework for the 4:1 rule**." Per `feedback_default_to_deepthought_for_training`
+ four_track/CLAUDE.md, with the BC2025 SS data living on skynet:

| Phase | Workload | N | GPU-bound? | Deployment |
|---|---|---|---|---|
| 5.0 env rebuild | skynet `kaggle` env from environment.yml | 1 | no (conda solve) | skynet, post-download |
| 5.1 pretrain | V2-S backbone on 819K-rec / 7,489-species corpus, 50 ep | 1 | yes | **DT** (single GPU, single job) |
| 5.2a pseudo-emit iter-1 | OOF inference of pretrained-V2-S teacher across BC2025 unlabeled SS | (sharded) | mixed (audio I/O dominates) | **skynet** (I/O-exception of 4:1 rule; BC2025 SS lives here) |
| 5.2b NS train iter-1 | 4-fold student finetune on focal+pseudos | 1 → 4 | yes | **DT-sequential** (N=4 < 5) |
| 5.3a pseudo-emit iter-2 | iter-1 student → OOF pseudos | (sharded) | mixed | **skynet** |
| 5.3b NS train iter-2 | iter-2 student finetune | 1 → 4 | yes | **DT-sequential** |
| 5.4 final 4-fold finetune on BC2026 focal | NS-iter-2 ckpt as init | 4 | yes | **DT-sequential** |
| 5.5 JIT export 3-fold | per §14.19.11 budget calibration | 3 | no | skynet |

NPZ pseudo artifact between 5.2a→5.2b and 5.3a→5.3b is small (~10 MB
compressed); `syncback skynet` (or rsync skynet→DT) is trivial. **No
need to ever sync ~10 GB BC2025 SS audio to DT.** This was the realization
that resolves the "use both GPUs" framing without violating the 4:1 rule.

### 14.21.7.6 Environment fixes applied (precondition for Phase 5.0)

User-approved 2026-05-04 ~14:00 local:
- `BirdCLEF/environment.yml`: pip `--index-url cu126` → `cu128` (so
  the Phase 5.0 skynet `kaggle` env rebuild gets aarch64 GPU torch
  wheels; cu126 has no aarch64 build → silent CPU fallback). Comment
  block updated to record the rationale.
- `~/.condarc`: `solver: classic` → `solver: libmamba` (fixes the
  14.5 h conda solve hang from 2026-05-03; `conda-libmamba-solver
  26.4.1` already in base; `conda config --show solver` confirms
  `libmamba`).

### 14.21.7.7 Deliberately NOT done (anti-View-B impulse log)

Items I considered + explicitly ruled out, recorded so future-me
doesn't re-derive:

- **Per-class balanced sampler** — strict Sydorskyi has it; I did
  global 40/60 instead (~150 LOC saved). Defer post-Bundle-1; only
  worth implementing if Bundle 1 lifts AND attribution suggests
  sampling was bottleneck.
- **BG noise injection in SydorskyiNSDataset** — production
  `train_a1.py` deliberately doesn't pass `bg_noise_dir` (no-op
  path). Sydorskyi's BG mix is in his 0-LB-lift bundle. Matching
  production omission is correct; I tried to invent this as a "real
  gap" first and walked it back.
- **Phase 5.4 final-finetune wrapper script** — `train_a1.py
  --init-from` already does the job; pre-committing hyperparams
  before seeing 5.3 outputs is premature.
- **Bumping `--rate` on the XC download** — script is API-key-throttled,
  zero 429s in 88 h, but asymmetric risk (potential XC ban if push
  too far) >> 1-2 day saving on a 12-day plan. Untouched.
- **Running partial-corpus pretrain "to see what happens"** — that's
  a single-lever probe. View-B forbids. Don't.

### 14.21.7.8 Updated ETA (corrected)

Earlier "May 7" estimate from §14.21 was based on a stale 4-day-old
checkin showing 59 species/hr early-rate and 341K projected total.
Re-anchored 2026-05-04 ~16:00 local:

- Recent **download rate**: ~2,500 dl/hr (steady; early peak was
  ~3,000 dl/hr, ~17% slowdown — partly XC server-side variation,
  partly unfiltered alphabet composition).
- Re-projected **total downloads**: ~500 K (cum 225,696 / idx 3,033 ×
  total 6,718 species).
- **Remaining**: ~274 K downloads / 2,500 dl/hr ≈ **110 h ≈ May 9
  morning**.
- **Bundle 1 first LB**: ~10-12 days from corpus completion → **May
  19-21** (vs §14.21's original May 17-19).

The 2-day slip is real but not material on a 17-day plan; one Bundle 1
attempt remains the right shape.

---

## ⏸️ PICK UP HERE — previous (2026-05-04 ~16:30 local — Bundle 1 prep done, scripts smoke-tested both machines, waiting for XC download — SUPERSEDED by §14.22)

**TL;DR (2026-05-04 ~16:30 local):**

All Bundle 1 prep work parallel-safe to the in-flight XC download is
done. Recipe finalized as **Sydorskyi 3-lever** (Babych path rejected,
tweaks bundle demoted to free-rider). Scripts written, both-machine
smoke-tested. Environment fixes applied for Phase 5.0. Now waiting on
XC download to land at ~May 9 (re-projected from current rate).

**Next action when XC download completes:**

Phase 5.0 → 5.1 launch sequence (per §14.21.3 + §14.21.7.5):

1. Verify download tail is healthy (no failure-burst at the end);
   final cum_dl ≈ 500K; final species ≈ 6,718.
2. **Re-include xenocanto_bulk in the daily backup.** Edit
   `~/bin/backup` and remove the
   `--exclude='kaggle/BirdCLEF/data/external/xenocanto_bulk/'`
   line on the AI-rsync invocation (added 2026-05-05 to stop the
   nightly 06:00 cron from chasing the moving XC corpus and stacking
   multi-day rsyncs). After removing, the next 06:00 cron will do a
   one-time large catch-up sync (~50-100 GB) to deepthought.
3. Phase 5.0 — skynet `kaggle` env rebuild from `BirdCLEF/environment.yml`
   (cu128 + libmamba in place).
4. **Install the new Seagate One Touch 8 TB external HDD on skynet**
   (user purchased 2026-05-04, deferred install until post-download
   per "no I/O perturbation while XC writes"). Sequence below; do
   NOT skip the badblocks step, do NOT mount before badblocks
   completes.

   **4a. SMART pre-check (done 2026-05-05):** drive is at `/dev/sda`,
   USB Bus 002 Dev 002 (Seagate `0bc2:208f`), 8.00 TB raw / 7.3 TB
   usable, 5400 rpm HDD, FW rev 4205, S/N `00000000NT1B0045`. SMART
   health OK. The USB bridge is locked-down — only SCSI passthrough
   works (`smartctl -d scsi`); ATA-level attributes (hours-on,
   reallocated sectors, model number) are masked. Drive lineage
   (8 TB / 2.5" / 5400 rpm) makes this a Seagate ST8000LM004-class
   **DM-SMR** (drive-managed shingled). Implication: sequential
   writes fine, random rewrites stall for minutes. **Cold-archive
   use only — never live training I/O.**

   **4b. Badblocks read-only scan — already queued.** Wrapper at
   `four_track/scripts/post_xc_badblocks.sh` is running in the
   background (launched 2026-05-05 13:14, PID 3665374). It polls
   for the XC download (`pgrep -f src/l2_redux/download_xenocanto.py`)
   to exit, sleeps 60 s for FS flush, refuses to scan if `/dev/sda*`
   is mounted, then runs `sudo badblocks -sv -b 4096 -o <out>
   /dev/sda` (read-only, ~12-18 h on USB 3 at 7.3 TB). Logs:
   `log/post_xc_badblocks_20260505_131456.log` (progress) and
   `log/badblocks_sda_20260505_131456.txt` (bad-block list — empty
   means clean). Caveats: wrapper dies on reboot (relaunch manually
   if skynet restarts before XC finishes); pgrep pattern is loose
   so don't launch unrelated `download_xenocanto.py` processes
   while it waits.

   **4c. Format + mount (DONE 2026-05-08 23:05).** Executed: wipefs
   + `sgdisk --zap-all` + single GPT partition (type 8300) + `mkfs.ext4
   -L MachineLearning -T largefile4 -m 0 /dev/sda1`. Mounted at
   **`/mnt/MachineLearning`** (UUID `70aa93f5-d422-4e81-8eb8-f5433cc064a5`),
   fstab entry uses `defaults,nofail,x-systemd.device-timeout=10`,
   chowned to `swatson:swatson`. Diff vs original plan: kept a GPT
   partition table (cleaner for tooling that expects one) instead of
   whole-disk; used `-m 0` (reclaims ~370 GB vs `-m 1`'s ~290 GB) and
   `-T largefile4` (1 inode per 4 MiB, suits media archive); label
   `MachineLearning` instead of `birdclef-archive` so the drive can
   hold non-BirdCLEF artifacts too. Other rejected filesystems
   (exFAT, btrfs, zfs) reasoning unchanged.

   **4d. Hard safety rule before any destructive op:** show
   `lsblk` output and explicitly confirm `/dev/sda` (not
   `/dev/nvme0n1`) with the user before running `mkfs`, `parted`,
   `wipefs`, or `dd`. Operating on the wrong device wipes the
   freshly-completed XC corpus.

   **4e. Use as cold archive only.** Memory entry
   `reference_skynet_external_drive.md` documents the SMR cliff —
   move *finished* corpora (post-mel-extraction, post-pseudo-emit
   shards) onto the external; keep the active corpus + training
   scratch on NVMe. Do not dataload directly off this drive during
   training.
5. Phase 5.1 launch on DT:
   ```
   conda activate kaggle
   cd /home/swatson/work/kaggle/BirdCLEF/four_track
   rm -f log/train_*.log
   nohup python -u src/pretrain_l2_redux.py \
       --epochs 50 \
       --backbone tf_efficientnetv2_s.in21k_ft_in1k \
       --include-xc-bulk \
       --save-every 10 \
       > log/pretrain_l2_redux_v2s_$(date +%Y%m%d_%H%M%S).log 2>&1 &
   ```
   Dispatch via `runon deepthought ...`. Pre-flight: `ssh deepthought
   nvidia-smi` (multi-tenant GPU).

**State checkpoint:**

- L2-redux XC v3 download: PID 1029580 alive, 88.6h elapsed, 3,033/6,718
  species (45.1%), 225,696 cum downloads. ETA ~May 9 morning.
- `environment.yml` cu126→cu128 + `~/.condarc` libmamba fixes applied
  and verified.
- New scripts: `pseudo_emit_sydorskyi.py`, `train_a1_ns_sydorskyi.py`;
  `pretrain_l2_redux.py` extended with `--backbone` + `--include-xc-bulk`.
- B0 production kernel still on Kaggle (LB 0.931). V2-S JIT artifacts
  retained as baseline-arch input.
- Stale partial-XC manifest cache (`l2_redux_manifest_with_xc.csv`,
  `l2_redux_aves_species_with_xc.json`) deleted post-smoke; will
  rebuild from fresh on Phase 5.1 launch.
- Today: 0 Kaggle slots used. View-B respected (no probes fired).

**Don'ts on next pickup:**

- Don't queue any single-lever probe. View-B in effect.
- Don't run pretrain on partial corpus "to see what happens" — Phase
  5.1 launches when corpus is COMPLETE, not when it's "close enough."
- Don't sync BC2025 train_soundscapes to DT — pseudo emit lives on
  skynet (I/O-exception of 4:1 rule); only the resulting NPZ syncs.
- Don't forget the `ssh deepthought nvidia-smi` pre-flight — DT is
  multi-tenant.
- Don't blanket `rm -f log/*.log` while the download is still writing
  — surgical filter per `feedback_clean_logs_before_training`.
- Don't mount/format the new Seagate 8 TB until **after both** the
  XC download lands AND the queued badblocks scan reports clean
  (see step 4b — wrapper PID 3665374 handles the wait). Skipping
  badblocks on a USB-bridged drive whose ATA SMART is masked is
  trusting the bridge chip, not the platters.
- Don't pick exFAT, btrfs, or zfs for the external without a reason
  that beats ext4's defaults; cold-archive use case doesn't justify
  any of them.
- Don't ever use the new Seagate 8 TB as live training I/O — it's
  DM-SMR; random rewrites stall for minutes and trigger USB resets.
  Cold archive only.
- Before any `mkfs`/`parted`/`wipefs`/`dd` on `/dev/sda`, show
  `lsblk` output and confirm the device path with the user — wrong
  device wipes the XC corpus on `/dev/nvme0n1p2`.

---

## 14.22 Non-Aves expansion — per-taxon probe + iNatSounds 2024 dispatch (2026-05-05 ~20:30 local)

**Status: probe complete, download dispatched, Bundle 2 scoped.** Parallel
to Bundle 1 (Sydorskyi 3-lever V2-S, blocked on XC v3 corpus). Bundle 2
runs on B0 production line and targets the non-Aves error mass that
Bundle 1 does not address (Sydorskyi corpus is functionally Aves-only —
see §14.17.15).

### 14.22.1 Per-taxon error-mass audit

User asked: of the LB gap to leader (0.951 vs floor 0.926), what fraction
is non-Aves driven? Answer derived from `data/v56_soundscape_oof.npz`
(1478 segments, 66 unique source files, 234-class probs).

```
group       n_total  n_evaluable  mean_AUC   error_mass_share
Aves            162           28    0.8934              14.7%
Amphibia         35           17    0.7557              20.4%
Insecta          28           25    0.5415              56.4%   ← dominant
Mammalia          8            4    0.7209               5.5%
Reptilia          1            1    0.3926               3.0%

Overall macro (evaluable classes only): 0.7290
```

Counterfactual upper bound: if non-Aves classes hit Aves mean, OOF macro
lifts +0.164. Six Insecta sonotype classes have native AUC < 0.5
(anti-correlated diagonal output) — most striking finding.

Caveats: 25 of 72 non-Aves classes have ZERO positives in OOF (invisible
to val gate). Hidden test almost certainly has positives for them. The
0.7290 OOF vs 0.7415 gate has a 1.7% gap (likely sklearn aggregation vs
production gate definition, not class-id misalignment — verified
indexing matches `config.py:75` `sorted(taxonomy["primary_label"].astype(str))`).

### 14.22.2 Encoder-vs-head probe (`src/probe_taxon_signal_v3.py`)

Question: is the Insecta gap an encoder-feature deficit or a
head/calibration deficit? Cheap probe before committing to corpus
expansion.

**Probe v1 (StratifiedKFold)**: per-class 5-fold LR on 234-dim logits as
features → recovered AUC ≈ 1.0 across all classes including Aves.
Suspect: file-level memorization (only 66 unique source files, segments
within file are correlated).

**Probe v2 (StratifiedKFold + permutation control + single-feature
ablation)**: shuffled-y permutation gave AUC ≈ 0.49 across taxa, ruling
out trivial memorization. Single-feature LR (diagonal logit only) lifted
Insecta only 0.542 → 0.668 — pure threshold tuning saturated, P1
`T_tex=0.95` already absorbs that signal.

**Probe v3 (GroupKFold by filename, 66 groups / 5 splits)** — the
leakage-resistant test:
```
         native   rec_single   rec_full
Aves     0.891     0.897        0.876   ← negative control: head fix HURTS
Amphibia 0.767     0.775        0.832
Insecta  0.542     0.557        0.734
Mammalia 0.787     0.548        0.955   (n_eval=3)
Reptilia 0.393     0.552        0.957   (n_eval=1)
```

45/75 evaluable; remainder NaN (small-n classes break under GroupKFold).

**Findings:**

1. v56 encoder carries **partial off-diagonal Insecta signal**. rec_full
   0.73 vs native 0.54 means a head-only fix could buy +0.19 OOF AUC
   without encoder change.
2. rec_full is +0.19 (not +0.46 as v1/v2 suggested). The 0.27 residual
   gap (0.73 → 1.0) is the legitimate encoder-pretrain target.
3. Aves negative control validates the procedure — head fix doesn't
   invent signal where none exists; it actively hurts Aves (-0.015) when
   forced.
4. Pure threshold tuning is saturated (rec_single ≈ native for all
   non-Aves except Insecta, where it gains only +0.015 leakage-resistant).

### 14.22.3 iNatSounds 2024 download dispatched

User decision: keep the download running, accept a stacking bundle plan
(head-fix + iNatSounds pretrain) with **+0.005 LB floor** for
pretrain-alone delta — i.e. anything above the single-submission noise
band per `feedback_lb_single_submission_noise.md`.

- Dispatch: deepthought, `~/work/MachineLearning/kaggle/BirdCLEF/four_track/data/external/inat_sounds_2024/`
- Files: train.tar.gz (81 GB), val.tar.gz (25 GB), 2× JSON manifests
  (~18 MB) — total ~106 GB. Test split (27 GB) skipped; we have BC2026 test.
- URLs verified: `ml-inat-competition-datasets.s3.amazonaws.com/sounds/2024/`
  via direct HTTPS (no AWS CLI required, no auth — public AWS Open Data).
- Script: `download.sh` (resumable wget loop, --tries=5 --waitretry=30,
  sequential to avoid bandwidth contention with the in-flight skynet XC).
- Disk: 4.6 TB free on deepthought `/mnt/mytoshiba` (external SSD).
- ETA: ~3-4 h at observed ~10 MB/s.
- Killable later: if Bundle 2 head-fix probe alone closes Insecta gap to
  the LB-noise threshold, the download can be discarded — design accepts
  that risk.

iNatSounds 2024 per-class composition (from corpus audit, see
[non-Aves bioacoustic corpus audit](#)): Insecta 13K rec / 745 sp,
Amphibia 17K rec / 650 sp, Mammalia 3.5K rec / 296 sp, Reptilia 235 rec /
32 sp. Direct overlap with our 234 species likely modest; the bet is on
encoder-transfer of generic non-Aves acoustic features.

### 14.22.4 Bundle 2 — B0 head-fix + iNatSounds pretrain

Two-lever bundle on B0 production arch (parallel to Bundle 1 on V2-S).

**Lever A (cheap, near-term): class-balanced head retrain.** Freeze v56
B0 encoder, retrain `cls_conv` + `att_conv` from random init with
class-balanced sampling. Target Insecta OOF AUC ≥ 0.65 (probe-derived
ceiling 0.73, leave room for OOF→LB compression). Predicted LB delta:
+0.005 to +0.015 if probe ceiling holds, ≤ +0.005 if it doesn't.

**Lever B (heavy, queued): iNatSounds-pretrained backbone → finetune.**
Pretrain B0 on iNatSounds 2024 (50 ep, focal-BCE, multi-taxon head),
then finetune on BC2026 with class-balanced head from Lever A. Target
the +0.27 residual gap. Floor +0.005 LB on top of Lever A.

Both levers stack. Cheap probe (Lever A alone) gates the heavy run
(Lever A + Lever B). If Lever A LB lift ≥ +0.010, Lever B's marginal
value still has to clear +0.005 to be worth shipping; if Lever A is
flat, Lever B's case is weaker.

### 14.22.5 Parallel-track caveat — Bundle 1 vs Bundle 2

Bundle 1 (Sydorskyi 3-lever, V2-S, XC v3 Aves-only corpus, queued) and
Bundle 2 (this) target different error-mass strata:
- Bundle 1: lifts Aves classes via 7,591-species in-domain pretrain on
  V2-S (recipe-validated per Sydorskyi ablation, +0.046 LB on his stack).
- Bundle 2: lifts non-Aves classes via iNatSounds multi-taxon pretrain
  on B0 (no published ablation; this is a probe-derived hypothesis).

Bundles do NOT share a backbone — Bundle 1 ships V2-S, Bundle 2 ships B0.
Either or both could ship; ensemble is open if both LB-positive. They do
NOT share a corpus — XC v3 covers 7,591 Aves species, iNat 2024 covers
~5,500+ multi-taxon. They do NOT share GPU on the same machine.

**Don't conflate the bundles.** A failure in Bundle 1 (e.g. V2-S kernel
timeout) does not invalidate Bundle 2. A success in Bundle 2 does not
moot Bundle 1 (Aves error mass is still 14.7% of total — small but real).

### 14.22.6 State checkpoint

- iNatSounds 2024 download: deepthought, ~3-4 h ETA, ~106 GB target.
- XC v3 download (Bundle 1): skynet, 88.6 h elapsed, 45.1% complete,
  ETA ~May 9 morning. UNCHANGED.
- Probe artifacts on skynet:
  - `four_track/data/probe_taxon_signal_v{1,2,3}_results.csv`
  - `four_track/src/probe_taxon_signal_v{1,2,3}.py`
- Memory: `project_taxon_signal_probe_2026_05_05.md` (probe + bundle decision).
- B0 production kernel still LB 0.931 (floor 0.926 per `project_lb_gap.md`).
- 0 Kaggle slots used today. View-B respected.

### 14.22.7 Open work for next pickup

1. **Lever A script.** Write `src/probe_head_fix.py`: load v56 fold-0
   ckpt (JIT, prefix-strip per `feedback_kaggle_ckpt_is_jit.md`), freeze
   backbone+pool, re-init `cls_conv`+`att_conv`, class-balanced sampling
   over `train_folds.csv`, eval on val_v2 + OOF substrate, report
   per-taxon AUC vs probe ceiling.
2. **Wait for iNatSounds download.** When `download.sh` exits, verify
   tar checksums, extract, build manifest. Defer Lever B until then.
3. **Don't queue head-fix as "JIT-ship to Kaggle" yet.** First confirm
   probe-ceiling reproduces on a fresh fold-0 run (single-fold Δ < 0.03
   noise floor per `feedback_single_fold_noise_floor.md`). 5-fold or
   ≥2 seed repeats before LB submit.

---

## ⏸️ PICK UP HERE — previous (2026-05-05 ~20:30 local — per-taxon probe done, iNatSounds dispatched, Bundle 2 scoped — SUPERSEDED by §14.22.8 below)

---

## 14.22.8 Lever A head-fix probe dispatched (2026-05-05 ~21:03 local)

`src/probe_head_fix.py` written and dispatched on skynet under
`kaggle-arch` env (not `kaggle` — see env note below).

**Probe spec implemented:**
- Loads eager v56 fold-0 ckpt directly (no JIT prefix-strip needed —
  `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid.pt`
  is a clean OrderedDict).
- Instantiates `BirdSEDModelA1(mixstyle_p=0.0)`, `load_state_dict`
  reports `missing=0 unexpected=0` (timm features_only classifier-head
  warning is benign — those keys aren't expected).
- Caches val features once via frozen `backbone + gem_pool` forward
  on the 1478-segment `train_soundscapes_labels.csv` substrate; head
  training runs against cached `(B, c_out, T')` features so each epoch
  is fast.
- `cls_conv` + `att_conv` re-initialized from random (matching the
  production architecture verbatim). Backbone, MixStyle, GEM pool
  frozen via `requires_grad=False`.
- Class-balanced sampling: WeightedRandomSampler with
  `1/n_per_class` weights over fold-0 train portion of
  `train_folds.csv`, capped at `PER_CLASS_CAP=50` rows per
  primary_label. `BirdTrainDatasetA1(augment=True,
  min_samples_per_class=0)` for the focal pipeline; standard mel
  PCEN tile, MixStyle disabled in the model.
- Loss: BCEWithLogits on `clip_logits`. AdamW lr=1e-3, wd=1e-4.
  12 epochs. `gc.collect() + cuda.empty_cache()` after each epoch
  per CLAUDE.md.
- Per-epoch: re-evaluates on cached val features, computes per-taxon
  AUC, saves best Insecta head ckpt. Logs match CLAUDE.md format
  (`Epoch N/12: train_loss=... overall=... Aves=... Insecta=...
  time=Xm XXs YYYY-MM-DD HH:MM:SS ★ BEST`).

**Targets:**
- Pass: Insecta OOF AUC ≥ 0.65 (probe v3 GroupKFold ceiling 0.73).
- Negative control: Aves OOF AUC must NOT exceed native ~0.89 — if
  it does, leakage; discard probe.
- Native baseline (pre-retrain) is logged at the top of the probe so
  the delta is visible.

**Process:**
- skynet PID 4189305, log
  `four_track/log/probe_head_fix_20260505_210247.log`.
- Outputs (when done): `data/probe_head_fix_best.pt` (best head
  ckpt), `data/probe_head_fix_epoch_metrics.csv` (full trajectory).
- ETA: ~1 h end-to-end (val cache ~3 min, then 12 epochs at ~5 min
  each on GB10 with frozen backbone).

### 14.22.8.1 Env discovery — `kaggle-arch` works on skynet, `kaggle` is broken

Per `four_track/CLAUDE.md` §"Two-GPU workflow" (note about Spark's broken
kaggle env), the `kaggle` env on skynet has torch/torchaudio ABI skew
(`OSError: undefined symbol: aoti_torch_create_device_guard` on
`import torchaudio`). The CLAUDE.md note implied env rebuild was the
only path forward and that "deepthought is the ONLY machine where new
training/probe runs can launch cleanly."

**This is incorrect — `kaggle-arch` (the aarch64-tagged sibling env at
`~/miniconda3/envs/kaggle-arch`) is healthy:**
- torch 2.7.1+cu128 (matches torchaudio 2.7.1)
- torchaudio loads cleanly
- timm 1.0.22, sklearn 1.5.2
- CUDA available, sm_121 detected correctly

**How to apply.** When dispatching skynet GPU work before the formal
`kaggle` env rebuild lands, use `conda activate kaggle-arch` instead.
Update CLAUDE.md text + `feedback_default_to_deepthought_for_training.md`
when next touched to remove the "deepthought is the only machine"
absolute. The 4.5× per-epoch slowdown still favors deepthought for
GPU-heavy work; `kaggle-arch` unblocks I/O-bound or short probes that
were previously blocked entirely.

### 14.22.8.2 What NOT to do while the probe runs

- Don't relaunch on skynet under `kaggle` env — it dies on import.
- Don't manually kill the val-feature caching phase to "speed up"
  training; the probe trades one full backbone forward (~3 min) for
  ~12 epochs of cached-feature head training. Without the cache,
  every epoch becomes a full backbone forward.
- Don't move `data/probe_head_fix_best.pt` to `kaggle_datasets/`
  yet — single-fold ceiling per `feedback_single_fold_noise_floor.md`.
  Need 5-fold or ≥2 seed repeats before any LB push.
- Don't drop `data/probe_taxon_signal_v3_results.csv` — that's the
  GroupKFold ceiling reference the probe targets.

---

## ⏸️ PICK UP HERE — previous (2026-05-05 ~21:03 local — Lever A probe RUNNING on skynet kaggle-arch, iNatSounds download IN FLIGHT on deepthought — SUPERSEDED by 2026-05-10 entry below)

**TL;DR (2026-05-05 ~21:03 local):**

Lever A (head-fix probe, §14.22.8) dispatched on skynet under the
healthy `kaggle-arch` env (`kaggle` env is broken — see §14.22.8.1).
Probe targets Insecta OOF AUC ≥ 0.65, ceiling 0.73 from probe v3
GroupKFold, with Aves as negative control. ETA ~1 h. iNatSounds 2024
download still streaming on deepthought (~2 h remaining for
train.tar.gz). XC v3 still streaming on skynet (~67 h remaining).

**Three workstreams in flight, all asynchronous:**

1. Lever A probe — skynet, PID 4189305, ~1 h
   - log: `log/probe_head_fix_20260505_210247.log`
   - outputs: `data/probe_head_fix_best.pt`,
     `data/probe_head_fix_epoch_metrics.csv`
2. iNatSounds 2024 download — deepthought, ~2 h
   - log: `data/external/inat_sounds_2024/download_20260505_*.log`
3. XC v3 download (Bundle 1) — skynet, ~67 h, UNCHANGED

**Next action on next pickup (depends on Lever A outcome):**

- **If Insecta OOF ≥ 0.65 and Aves stays ≤ 0.89:** Lever A is
  validated. Queue seed-repeat or 5-fold reproduction before any LB
  submit (single-fold noise per
  `feedback_single_fold_noise_floor.md`).
- **If Insecta < 0.65 OR Aves > 0.89:** probe disagrees with v3
  ceiling. Investigate before iterating — don't blanket retune. The
  Aves-leakage check is the more important one of the two.
- **Either way:** continue with §14.22.7 steps 2-4 once iNatSounds
  download lands (verify checksums, build manifest, fork
  `pretrain_inat_sounds.py`). Lever B is gated on Lever A reporting,
  not on Lever A passing.

**State checkpoint:**

- skynet PID 4189305 (Lever A probe), env=`kaggle-arch`, log path
  above.
- deepthought iNatSounds download: alive, val.tar.gz finishing,
  train.tar.gz queued sequential.
- skynet PID 1029580 (XC v3): alive, 4272/6718 species (63.6%).
- skynet badblocks waiter (PID 3665374): polling, will fire
  post-XC.
- Today: 0 Kaggle slots used. View-B respected.

**Don'ts on next pickup:**

- Don't relaunch any probe on skynet under `kaggle` — use `kaggle-arch`
  until env rebuild lands.
- Don't kill the iNatSounds download just because Lever A returns
  positive — Bundle 2's case rests on the +0.27 residual that head-fix
  doesn't address.
- Don't ship Lever A on a single fold without seed-repeat validation —
  ~0.03 single-fold noise per `feedback_single_fold_noise_floor.md`.
- Don't conflate Bundle 1 and Bundle 2 — different arches, different
  corpora, different error-mass targets.
- Don't extract iNatSounds tarballs on skynet — they live on
  deepthought; sync NPZs/manifests, not raw audio.
- Don't skip the Aves negative-control check on any Lever A retrain —
  if head-fix lifts Aves above 0.89 OOF, the procedure is leaking.
- All Bundle 1 don'ts from §14.21.7 still apply.

---

## ⏸️ PICK UP HERE — previous (2026-05-05 ~21:03 local — Lever A probe RUNNING, iNatSounds + XC v3 downloads in flight — SUPERSEDED by §14.22.9 below)

---

## 14.22.9 Lever A KILLED + 73-h infrastructure detour closeout (2026-05-08 ~23:30 local)

Three items resolved between the previous pickup (2026-05-05 ~21:03)
and now, none of which had been written into the plan. Recording for
session continuity.

### 14.22.9.1 Lever A (B0 head-fix probe) — KILLED on Insecta OOF gate

`probe_head_fix.py` finished 2026-05-05 22:07. **Best Insecta OOF AUC =
0.5722 at epoch 4, vs target ≥0.65 / probe v3 ceiling 0.73 / native
0.5179.** Lever A delivered +0.054 of the +0.21 ceiling (26% capture)
on the dominant error-mass class — fails the §14.22.4 head-fix gate.

Per-taxon trajectory (best Insecta = epoch 4):

| Taxon | Native (v56) | Probe ep 4 | Δ | Probe v3 ceiling | Capture % |
|---|---|---|---|---|---|
| Aves | 0.8379 | 0.7925 | **−0.045** | 0.876 (head-fix HURTS) | n/a (negative control) |
| Amphibia | 0.7263 | 0.6946 | −0.032 | 0.832 | 0% |
| **Insecta** | **0.5179** | **0.5722** | **+0.054** | **0.734** | **26%** |
| Mammalia | 0.7575 | 0.8792 | +0.122 | 0.955 (n=3) | 62% — but only 5.5% error-mass |
| Reptilia | 0.6599 | 0.6166 | −0.043 | 0.957 (n=1) | 0% |

Aves negative-control passed: 0.7925 < 0.89 native ceiling, no leakage.

**Why the lever fell short of probe v3 ceiling:** probe v3 was a
GroupKFold-by-filename LR on full 234-dim logit features — leakage-
resistant but used existing logits as the input space. Lever A retrains
the actual `cls_conv` + `att_conv` heads from random init on
class-balanced focal data, so it has to learn a head from scratch
rather than re-weight an already-fit logit basis. The encoder features
remained frozen, so any signal the encoder didn't expose is unrecoverable
without pretraining.

**Mammalia side finding:** +0.122 (62% of ceiling) is real but
operationally small — Mammalia carries only 5.5% of the OOF error-mass
per §14.22.1 vs Insecta's 56.4%. A Mammalia-targeted lever is not worth
queuing.

**Pattern continuation:** the §14.17/§14.19 single-lever probe kill
streak (V2-S, L1, L5b, T2.6, T1.3, P8, P12, B2, A2, M1, M2, …) now adds
Lever A. Streak 15+ with zero exceptions. Reinforces §14.21's
multi-lever-bundle thesis — single-lever probes do not move LB on this
stack. Lever A's +0.054 delta is also below the ±0.005 LB single-
submission noise band on the OOF→LB transfer, so even an LB push would
have been uninformative.

**No revert needed.** Lever A produced `data/probe_head_fix_best.pt`
+ `data/probe_head_fix_epoch_metrics.csv` as artifacts; nothing was
shipped to Kaggle. Code (`src/probe_head_fix.py`) and ckpt are kept
for the historical record. No production code touched.

### 14.22.9.2 XC v3 + iNatSounds downloads — both DONE

| Stream | Status | Location |
|---|---|---|
| XC v3 (Bundle 1 corpus) | DONE 2026-05-08 09:02 (88.6 h total) | `data/external/xenocanto_bulk/` on skynet |
| iNatSounds 2024 (Bundle 2 corpus) | DONE 2026-05-05 ~23:23 (~3 h) | `data/external/inat_sounds_2024/` on deepthought; train.tar.gz 86 GB + val.tar.gz 26.6 GB + JSONs |

Both Bundle 1 and Bundle 2 are now corpus-unblocked.

### 14.22.9.3 Infrastructure detour 2026-05-08 (badblocks → /mnt/MachineLearning → rsync)

Three days were spent on storage hygiene rather than ML work, in this
order:

1. **Post-XC badblocks scan on /dev/sda** (read-only, `-sv -b 4096`,
   started 09:06 the moment XC v3 exited; finished 22:43, 13h 36m
   total, **0/0/0 errors** — drive media verified clean). Wrapper
   `scripts/post_xc_badblocks.sh` waited on the XC pgrep loop so the
   scan didn't fight the still-running download.
2. **Wipe + ext4 reformat → /mnt/MachineLearning** (2026-05-08
   23:05). `wipefs -a` + `sgdisk --zap-all` + single GPT partition
   (type 8300) + `mkfs.ext4 -L MachineLearning -T largefile4 -m 0
   /dev/sda1`. UUID `70aa93f5-d422-4e81-8eb8-f5433cc064a5`. fstab
   uses `defaults,nofail,x-systemd.device-timeout=10`. Memory note
   `reference_skynet_external_drive.md` updated. Old `/mnt/archive`
   placeholder references in `~/bin/backup` + `rsync_to_deepthought
   _throttled.sh` + this section §14.21.7 step 4c updated to
   `/mnt/MachineLearning`. `format_mount_archive.sh` header marked
   HISTORICAL.
3. **Manual full-tree rsync skynet→deepthought** (started 14:53,
   `bwlimit=50M`, `--inplace --append-verify`, excludes
   `xenocanto_bulk/`). At 23:30 still in flight in
   `devstral/.git/lfs/objects/` (lowercase 'd' tier — far from
   `kaggle/`). ETA past midnight, possibly into early morning.
4. **Cross-script lock guardrail.** Both `~/bin/backup` (cron at
   06:00) and `scripts/rsync_to_deepthought_throttled.sh` now flock
   `/tmp/backup_ai_skynet.lock` non-blockingly around the
   `backup_ai_skynet` rsync, so the cron's ai segment skips cleanly
   if a manual run is in flight (dev rsync still runs, email still
   sent). Watchdog process (PID 4082806) holds the lock on behalf
   of the in-flight manual rsync (PID 4054989) until it exits, so
   tonight's already-running rsync is also covered.

This detour was necessary — the new external drive is the
plan-designated cold-archive home for `xenocanto_bulk` per §14.21.7
step 4e — but it consumed 73 h of wall-clock with **zero ML
movement** and **zero plan updates during the detour**. Recording
explicitly so the next session doesn't read §14.22.8 and assume
Lever A is still pending.

### 14.22.9.4 State checkpoint (2026-05-08 ~23:30 local)

- B0 production kernel: LB 0.931 (floor 0.926). Unchanged since
  2026-05-03.
- 0 Kaggle slots used since 2026-05-05. View-B respected.
- Lever A: KILLED (this section). Memory: `project_lever_a_killed.md`.
- Bundle 1 (Sydorskyi-replica via XC v3 pretrain): scripts
  smoke-tested both machines per §14.21.7, corpus now in place,
  unblocked.
- Bundle 2 (iNatSounds pretrain for non-Aves residual): corpus on
  deepthought, needs checksum + extract + manifest before pretrain
  dispatch. NOT cancelled by Lever A's failure — Bundle 2 targets the
  +0.27 encoder-pretrain residual, a different mechanism than head-fix.
- /mnt/MachineLearning ready as cold archive (7.3 T avail).
- Manual rsync skynet→deepthought: in flight, watchdog-locked.

### 14.22.9.5 Next action — Bundle 2 FIRST (flipped 2026-05-08 ~23:50)

Both Bundle 1 and Bundle 2 are now corpus-unblocked. Per the 4:1
deepthought:skynet rule (`feedback_default_to_deepthought_for_training.md`)
they should NOT be queued in parallel — sequential dispatch only.

**Recommended order: Bundle 2 first.** This flips the prior
recommendation (which favored Bundle 1 by historical LB-claim
envelope). The flip is forced by Lever A's empirical result:

1. **Lever A's failure pinned the bottleneck to the encoder.**
   Head-only retrain captured 26% of its +0.21 OOF ceiling and
   stalled. That's the cleanest evidence we've had on this
   competition that v56's encoder, not its head, is what's missing
   for non-Aves. Bundle 2's mechanism (multi-taxa encoder pretrain
   on iNatSounds) directly addresses that bottleneck. Bundle 1's
   mechanism (Aves-only encoder pretrain on XC v3) does not.
2. **The Sydorskyi +0.046 number is from a different class
   distribution.** BC2025 was Aves-skewed; BC2026's error-mass is
   74.6% non-Aves (Insecta 56.4 + Amphibia 20.4 + Mammalia 5.5 +
   Reptilia 3.0 per §14.22.1). XC v3 is Aves-only — Bundle 1 attacks
   the 14.7% slice we already do well on, not the 74.6% we don't.
   Sydorskyi's number is not transferable on its face; it has to
   be re-derived for BC2026's class mix.
3. **Bundle 2's prep is short and shouldn't gate it.** ~half a day
   on deepthought (checksum + extract + manifest + fork
   `pretrain_inat_sounds.py`); Bundle 1's "dispatch-ready" status
   doesn't outweigh the mechanism mismatch.

Bundle 1 stays queued as **Plan B**, dispatched only if Bundle 2
fails its val gate or LB lands at noise.

**Mechanism caveat (added 2026-05-09 ~00:05 — material to LB envelope):**

Direct species-overlap probe (val.json, 2026-05-09 ~00:00) found:

| Class | BC26 classes | iNat val coverage | val clips for matched |
|---|---|---|---|
| Aves | 162 | 152 (93.8%) | 1,499 |
| Insecta | 28 | **3 (10.7%)** | **36** |
| Amphibia | 35 | 24 (68.6%) | 13 |
| Mammalia | 8 | 4 (50.0%) | 10 |
| Reptilia | 1 | 0 (0%) | 0 |

The Insecta absent set includes BC2026's `Insect son01..NN` *sonotype*
labels — anonymized acoustic clusters from Pantanal recordings that
don't have iNaturalist taxon IDs (not Linnaean species). They CANNOT
match iNat by name. Bundle 2's mechanism is therefore **not co-training
on the same Insecta species**; it is **transfer learning from 745
*related* Insecta species** (cicadas, crickets, etc. globally) hoping
the encoder learns Insecta-relevant acoustic primitives that transfer.

This is the same mechanism that **L5b-Amphibia tested 2026-04-19 and
killed** (AnuraSet → BC2026 Amphibia transfer). User chose to proceed
2026-05-08 ~23:55 over this objection on the basis that L5b was
single-taxa and Bundle 2 is multi-taxa, which may yield a stronger
encoder. Recording explicitly:

- LB envelope is now **+0.005 to +0.015 floor**, not the optimistic
  +0.27 probe v3 ceiling (which assumed direct co-training that
  cannot occur for Pantanal sonotypes).
- The "kill criteria escalation" (LB ≤ 0.931 → Bundle 2 mechanism
  dies) at step 8 IS now also a kill on the **transfer-learning
  hypothesis class as a whole** — second failure of this mechanism
  on this stack would close the corpus-injection lever family.
- Pivot if Bundle 2 fails: NOT another corpus (BC2025 unlabeled,
  more iNat versions, etc.); pivot to a different stack
  (PaSST/AST transformer, ProtoSSM-as-encoder, multi-task aux loss).

Aves overlap is meanwhile 93.8% — Bundle 2 will get an Aves lift via
direct co-training as a side effect even if non-Aves transfer fails.
That makes the LB-flat case ambiguous: an Aves-only lift on Bundle 2
looks similar to Bundle 1's expected behavior, just at lower
species-count fidelity. Distinguish by Channel-B 184-species focal
diagnostic, NOT overall val_v2.

**Bundle 2 dispatch sequence (each step gated; do NOT skip):**

| # | Step | Machine | Wall-clock | Decision gate |
|---|---|---|---|---|
| 1 | Verify train.tar.gz / val.tar.gz checksums against iNatSounds release manifest | DT | ~10 min | sha256 matches |
| 2 | Extract tarballs into `data/external/inat_sounds_2024/{train,val}/` | DT | ~30 min | extracted file count matches JSON record count |
| 3 | Build species manifest mapping iNat taxa → BC2026 234-class space (most clips will be off-class; that's fine, the pretrain task is multi-class on iNat's own label space) | DT | ~30 min | manifest enumerates all iNat taxa with non-zero clip count |
| 4 | Fork `src/pretrain_inat_sounds.py` from existing pretrain harness, B0 backbone, multi-taxon head, focal-BCE | skynet | ~1 h | smoke test 1-step forward + backward on 8 clips |
| 5 | Pretrain B0 on iNat full corpus (50 ep, deepthought GPU) | DT | ~2-3 days | per-epoch loss decreases monotonically; final val on iNat held-out ≥ random-baseline + 0.05 |
| 6 | Finetune pretrained B0 on BC2026 train_audio (existing pipeline, swap backbone init) | DT | ~6 h | val_v2 fold-mean ≥ 0.7414 (production gate) AND **Channel-B 184-species focal AUC ≥ 0.9545 baseline** (gate selected per `project_val_v2_built.md` since this lever targets non-Aves; overall val_v2 alone is the wrong arbiter here) |
| 7 | 5-fold or ≥2 fold-0 seed repeats per `feedback_single_fold_noise_floor.md` | DT | ~1-2 days | Insecta OOF AUC ≥ 0.6 (lift over native 0.54 must clear single-fold noise) |
| 8 | JIT export + Kaggle dataset push + cell-41 toggle + LB submit | skynet | ~30 min | per `feedback_kernel_timeout_vs_scoring_stall.md` Phase ii landing |

**Total wall-clock estimate: ~5–7 days** from now to first LB result.

**Decision rules at the LB gate (step 8):**

- **LB ≥ 0.945** → Bundle 2 mechanism vindicated. Queue Bundle 1
  next (different error-mass slice, can stack).
- **LB 0.937–0.944** → real lift, ship. Decide whether Bundle 1's
  marginal Aves attack is still worth ~10 days of compute given
  remaining schedule.
- **LB 0.932–0.936** → marginal. Ship. Bundle 1 likely too expensive
  for the residual headroom.
- **LB 0.927–0.931** → noise band. **Bundle 2 mechanism dies.** This
  would be the second non-Aves attack to fail (after L5b-Amphibia
  2026-04-19), so the +0.27 probe v3 ceiling becomes empirically
  suspect on this encoder/loss combo. Pivot decision required:
  fundamentally different stack (PaSST/AST transformer, ProtoSSM-
  as-encoder, multi-task with audio-feature aux loss), not another
  corpus-injection attempt.
- **LB ≤ 0.926** → catastrophic. Revert immediately, debug whether
  iNat label space corrupted finetune (e.g. taxa overlap caused
  double-labeled clips, calibration miscalibration).

**Why Channel-B not overall val_v2 as the primary gate (step 6):**

Bundle 2's mechanism is non-Aves lift. Overall val_v2 is dominated
by the 162 Aves classes; a flat Aves + lifted non-Aves would show
~flat overall and we'd kill Bundle 2 for a wrong reason. The
Channel-B 184-species focal diagnostic (`project_val_v2_built.md`)
is the right arbiter for non-Aves capacity. Pre-committed here so
we don't re-litigate after results land.

---

## ⏸️ PICK UP HERE — previous (2026-05-08 ~23:30 local — Lever A KILLED, infra detour closed, Bundle 1+2 both unblocked, manual rsync still in flight — SUPERSEDED by 2026-05-10 entry below)

**TL;DR:** Three days of storage/infra hygiene done (badblocks clean,
ext4 reformat, /mnt/MachineLearning live, rsync guardrail). Lever A
formally killed (Insecta +0.054 vs +0.21 ceiling, 26% capture, gate
fail). 0 Kaggle slots used in that window. B0 production still
LB 0.931. Both XC v3 (Bundle 1 corpus) and iNatSounds 2024 (Bundle 2
corpus) downloads complete. Next: dispatch **Bundle 2** (B0 +
iNatSounds multi-taxa encoder pretrain) per §14.22.9.5 — flipped
from Bundle 1 because Lever A's failure pinned the bottleneck to
the encoder, and Bundle 2's mechanism matches that finding while
Bundle 1's Aves-only corpus does not. Bundle 1 stays queued as
Plan B.

**Don'ts on next pickup:**
- Don't queue Bundle 1 + Bundle 2 in parallel — 4:1 rule says
  sequence them. Bundle 2 first.
- Don't re-attempt Lever A in any flavor (head-only retrain) without
  also adding encoder pretrain — the +0.21 ceiling is structural.
- Don't gate Bundle 2 step 6 on overall val_v2 alone — the right
  arbiter is Channel-B 184-species focal AUC ≥ 0.9545. Overall
  val_v2 is Aves-dominated and would kill Bundle 2 for a wrong
  reason.
- Don't kill the manual rsync to free up bandwidth — watchdog has
  the cron-lock covered; let it finish.
- Don't extract iNatSounds tarballs on skynet — they live on
  deepthought; sync NPZs/manifests, not raw audio.
- Don't skip writing the next phase's outcome into the plan within
  24 h of dispatch — this 73-h gap is the failure mode we just
  closed; don't reopen it.

---

## ⏸️ PICK UP HERE — previous (2026-05-08 ~23:55 local — Bundle 2 first selected, kaggle/ move queued — SUPERSEDED by §14.22.10 + §14.22.11 below)

---

## 14.22.10 Bundle 2 dispatch — steps 1–4 ✓ (2026-05-09 ~00:00–00:30 local)

Steps 1–4 of the §14.22.9.5 dispatch sequence executed cleanly. Step 5
was deferred at the time pending the kaggle/ move (skynet move now
done; see PICK UP HERE for the dispatch command).

### 14.22.10.1 Step 1 — gzip integrity verify ✓

All four tarballs in `data/external/inat_sounds_2024/` on deepthought
pass `gzip -t`:

| File | Size | Verify | Result |
|---|---|---|---|
| val.json.tar.gz | 3.8 MB | 0.1 s | OK |
| train.json.tar.gz | 14 MB | 0.2 s | OK |
| val.tar.gz | 25 GB | 2m 55s | OK |
| train.tar.gz | 80 GB | 9m 15s | OK |

Total verify 12m 10s. iNatSounds release ships no separate sha256
manifest; gzip -t catches the most common silent-corruption mode.
Wget log already proved transport-layer Content-Length match. Log:
`data/external/inat_sounds_2024/gzip_t_verify_20260508_232809.log`.

### 14.22.10.2 Step 2 — extract train + val ✓

Both tarballs extracted on deepthought to
`data/external/inat_sounds_2024/{train,val}/`. 38m 15s total
(val 9m 06s, train 29m 09s). Layout follows the
`NNNNN_Kingdom_Phylum_Class_Order_Family_Genus_species/` convention:
5,571 directories under train/ (5,569 species + `.`/`..`).

| Split | Files | Size |
|---|---|---|
| val | 45,698 | 36 GB |
| train | 137,012 | 114 GB |
| **Total** | **182,710** | **150 GB** |

Log: `data/external/inat_sounds_2024/extract_20260508_234238.log`.

### 14.22.10.3 Step 3 — species manifest ✓

`src/build_inat_manifest.py` written; ran on deepthought against the
extracted JSONs + a copy of BC2026 `taxonomy.csv` at
`/tmp/bc2026_taxonomy.csv`. Outputs:

- `data/external/inat_sounds_2024/inat_manifest.csv` — 182,710 rows
  (one per audio clip), columns: split, file_path, inat_cat_id,
  scientific_name, common_name, tax_class, kingdom, phylum, order,
  family, genus, bc2026_primary_label, bc2026_class_name,
  audio_dir_name.
- `data/external/inat_sounds_2024/inat_species_summary.csv` — 5,569
  rows (one per iNat species).
- `data/external/inat_sounds_2024/inat_bc2026_coverage.json` —
  per-tax-class overlap stats.

Confirmed BC2026↔iNat overlap (matches val/train probe — same
category set across splits):

| Class | BC26 classes | iNat species | iNat clips for matched (train+val) |
|---|---|---|---|
| Aves | 162 | 152 (93.8%) | 5,371 |
| Insecta | 28 | 3 (10.7%) | 80 |
| Amphibia | 35 | 24 (68.6%) | 87 |
| Mammalia | 8 | 4 (50.0%) | 22 |
| Reptilia | 1 | 0 (0%) | 0 |

Per-tax-class iNat clip totals (across all species, not just
BC2026-matched): Aves 148,626 | Amphibia 17,187 | Insecta 13,145 |
Mammalia 3,549 | Reptilia 203. **iNat is 81.3% Aves by clip count** —
class-balanced sampling is mandatory at pretrain (see §14.22.10.4).

### 14.22.10.4 Step 4 — pretrain script + smoke test ✓

`src/pretrain_inat_sounds.py` written as a fork of
`src/pretrain_l2_redux.py`. Major adaptations vs the L2-redux sibling:

- **Manifest source**: loads `inat_manifest.csv` directly; no directory
  walk, no per-source `_meta.json` lookup.
- **Class space**: 5,569 iNat scientific names (not BC2026 primary_label
  codes). Backbone weights transfer; head is discarded.
- **Split**: iNat's official train/val (137K / 45K) via the manifest
  `split` column. NO `GroupShuffleSplit` by author — iNat's split is
  curated.
- **Sampler**: `WeightedRandomSampler(1/n_per_class, replacement=True)`
  — REQUIRED to counter iNat's 81.3% Aves bias. Without it the
  encoder sees ~5× more Aves than Insecta clips per epoch and the
  +0.27 probe-v3 Insecta ceiling is structurally unreachable.
- **Loss**: focal-BCE γ=2 (same as L2-redux).
- **Backbone**: `tf_efficientnet_b0.ns_jft_in1k` (B0; no arch swap).
- **Audio root**: passed via `--inat-root`; files at
  `{inat_root}/{file_path}` per the manifest.

Smoke test executed via `runon deepthought ... --epochs 1 --smoke-test`
(after fixing the initial smoke-partition bug — first attempt yielded
zero train batches because the 60-row smoke partition was smaller than
batch_size=64 with `drop_last=True`; fix overrides batch_size to 8 and
sets `drop_last=False` in smoke mode):

| Metric | Value |
|---|---|
| train_loss | 0.6328 (real, non-zero ⇒ optimizer.step() ran) |
| val_roc_auc | 0.300 (n_present=2; small-sample noise on 25 val rows × 5569 classes) |
| time | 34 s end-to-end |
| ckpt saved | `_runon/BirdCLEF/four_track/models/pretrain_inat/inat_best_tf_efficientnet_b0_ns_jft_in1k.pt` |
| exceptions | none |

Smoke gate: forward + backward + scheduler step exercised on >1 batch
without exception. **PASS.**

### 14.22.10.5 Bundle 2 dispatch state at end of 2026-05-09 00:30

| # | Step | Status | Notes |
|---|---|---|---|
| 1 | gzip integrity | ✓ | log preserved |
| 2 | extract train + val | ✓ | 150 GB, 182,710 files on deepthought |
| 3 | build manifest | ✓ | 5,569 species, 182,710-row CSV on deepthought |
| 4 | fork pretrain + smoke | ✓ | live forward+backward verified |
| **5** | **full 50-ep pretrain (deepthought GPU, ~2-3 days)** | **BLOCKED on kaggle/ move** | |
| 6 | finetune + Channel-B gate | gated on 5 | gate: Channel-B 184-sp focal AUC ≥ 0.9545 |
| 7 | 5-fold or ≥2 seed-repeat | gated on 6 | per single-fold-noise rule |
| 8 | LB submit | gated on 7 | per kernel-timeout-vs-scoring-stall rule |

## ⏸️ PICK UP HERE — previous (2026-05-09 ~14:35 local — Bundle 2 step 5 ready to dispatch — SUPERSEDED by 2026-05-10 entry below)

**TL;DR:** Bundle 2 dispatch sequence executed cleanly through step 4.
iNat 2024 corpus is extracted on deepthought (150 GB, 182,710 clips,
5,569 species), manifest is built, pretrain script + smoke verified.
The kaggle/ move on skynet is DONE (verified). Step 5 (full 50-epoch
pretrain on deepthought GPU, ~2-3 days) is unblocked and ready to
dispatch.

> Backup-rsync work has been split out to `rsync.md`. It's independent
> of the ML pipeline and was holding this section hostage. Both rsync
> attempts since 2026-05-08 14:53 have failed (last one with a hardware
> I/O error on `/mnt/mypassport`). Pretrain dispatch does **not** depend
> on rsync — iNat data lives on `/mnt/mytoshiba/MachineLearning/...`
> via the `~/work/MachineLearning` symlink on deepthought, untouched
> by either rsync attempt.

**Dispatch command** (run from
`/home/swatson/work/kaggle/BirdCLEF/`):

```bash
runon deepthought python -u four_track/src/pretrain_inat_sounds.py \
    --inat-root /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track/data/external/inat_sounds_2024 \
    --epochs 50
```

The `--inat-root` uses the deepthought-side path
(`MachineLearning/kaggle/...`) since the optional deepthought-side
move (former §14.22.11.2 step 4d, now in `rsync.md` step 4) was NOT
done. Runon will rsync the project to
`_runon/BirdCLEF/four_track/` on deepthought; `__file__`-based
resolution + `BIRDCLEF_ROOT` portability fix make this work.

**Pre-flight verified (2026-05-09 14:30):**

- Deepthought GPU idle: 94 MiB / 16 GB used, 0 % util.
- iNat data: 255 G at
  `/mnt/mytoshiba/MachineLearning/kaggle/BirdCLEF/four_track/data/external/inat_sounds_2024/`
  (includes the not-yet-deleted train/val tarballs alongside the
  extracted dirs and `inat_manifest.csv`).
- Smoke ckpt confirms script ran successfully on deepthought 2026-05-09
  00:30: `_runon/BirdCLEF/four_track/models/pretrain_inat/inat_best_tf_efficientnet_b0_ns_jft_in1k.pt`
  (29 MB, will be overwritten by the full run).
- runon dispatcher: working (`~/bin/runon`, config at `~/.runon.conf`).

**State checkpoint (2026-05-09 ~14:35):**

- B0 production kernel: LB 0.931 (unchanged since 2026-05-03).
- 0 Kaggle slots used since 2026-05-05. View-B respected.
- iNat extraction artifact: see Pre-flight above.
- kaggle/ move on skynet: DONE.
  `/home/swatson/work/kaggle/` is the populated tree;
  `/home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track/`
  is an empty leftover shell.

**Don'ts on next pickup:**

- Don't run the pretrain on partial corpus "to see what happens" —
  Phase 5 is gated on a **specific finetune AUC target** that only the
  full corpus + WeightedRandomSampler can deliver.
- Don't bundle a V2-S retrain or ProtoSSM modifications with the
  pretrain — single-lever noise-floor rule (§14.22.9.5).
- All Bundle 2 mechanism caveats from §14.22.9.5 still apply
  (transfer-learning hypothesis, Channel-B gate, single-lever
  noise-floor rule).

---

## ⏸️ PICK UP HERE — previous (2026-05-11 ~00:00 local — mel cache RUNNING — SUPERSEDED by 2026-05-11 ~23:00 entry below)

> **You went to bed at 2026-05-10 ~midnight EDT. Here's the autonomous state and the exact action you (or Claude) should do when you wake up.**

### ☀️ When you wake up — do this

1. **Check whether the mel cache finished.**
   ```bash
   ssh deepthought "ps -p 1771983 -o pid,etime,stat 2>&1 | head -3"
   ssh deepthought "ls /home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/data/processed/inat_mels/*.npy | wc -l"
   ssh deepthought "tail -20 /home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260510_235039.log"
   ```
   - Cache file count should be **137011** when complete (138K train clips).
   - Process gone + `[done] ... min` in log = success.
   - Process still running + count < 137K = wait longer (was ~9.3 files/sec).

2. **If cache complete (137011 files, process exited cleanly):**

   ```bash
   # Clean local logs per the cleanup rule
   rm -f /home/swatson/work/kaggle/BirdCLEF/four_track/log/*.log

   # Dispatch pretrain v4 — reads cached mels from NVMe
   cd /home/swatson/work/kaggle/BirdCLEF/four_track
   runon deepthought python -u src/pretrain_inat_sounds.py \
     --inat-root /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track/data/external/inat_sounds_2024 \
     --epochs 25 \
     --natural-sampling \
     --mel-cache-dir /home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/data/processed/inat_mels
   ```

   Expected wall: ~3-4 h (GPU-bound, not I/O-bound this time). Save path will be `inat_best_*_natfreq_melcache.pt`.

3. **If cache still running:** let it finish. Don't dispatch anything yet (cache build is reading from the same HDD that pretrain v4 would need — only one tenant at a time).

4. **If cache crashed:** check log for the first `bad rows` printout. The two bugs we hit during initial dispatch were both fixed:
   - ProcessPoolExecutor fork deadlock → use `mp.get_context("spawn")` ✓ (already in code)
   - `np.save` auto-suffix → use file-handle write ✓ (already in code)
   Don't re-introduce either.

### ⚙️ Active jobs at sleep-time

| Field | Value |
|---|---|
| **DT job (running)** | iNat mel cache build (one-time, ~4h) |
| Started | 2026-05-10 23:50:40 EDT |
| Host | deepthought |
| PID | **1771983** |
| Command | `runon deepthought python -u src/build_inat_mel_cache.py --inat-root ... --out-dir /home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/data/processed/inat_mels --split train --num-workers 8` |
| Log | `deepthought:/home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260510_235039.log` |
| Output | `deepthought:/home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/data/processed/inat_mels/train_*.npy` (NVMe, 87 GB margin remaining) |
| Rate at last check | ~9.3 files/sec |
| ETA cache complete | **~03:50 EDT 2026-05-11** |
| ETA full pipeline (cache + pretrain v4) | **~07:50-08:00 EDT 2026-05-11** |

### What the pretrain v4 will produce

Save path: `models/pretrain_inat/inat_best_tf_efficientnet_b0_ns_jft_in1k_natfreq_melcache.pt`

Then the next step is:
1. `syncback deepthought four_track/models/`
2. Edit `scripts/dispatch_a1_inat_v2_fold0.sh` to point at the new ckpt path (currently points at `_natfreq.pt` — change to `_natfreq_melcache.pt`)
3. Run the script — fold-0 A1 finetune with `--ft-recipe production` and the new ckpt as `--init-from`. ~85 min on DT.

### What we learned today (durable wins, regardless of v4 outcome)

1. **`--ft-recipe production` flag** (`src/train_a1.py`) — gentle recipe was a real bug, +0.044 lift on iNat fold-0 (0.6692 → 0.7130). Applies to all future pretrain levers.
2. **`--natural-sampling` flag** (`src/pretrain_inat_sounds.py`) — drops `WeightedRandomSampler(1/n_per_class)`, uses natural shuffle. Better matches BC2026's Aves-heavy distribution.
3. **`--mel-cache-dir` flag** + `src/build_inat_mel_cache.py` — pre-caches fp16 mels to NVMe, eliminating mp3-decode + resample + mel-compute on every iteration. Estimated 10× training speedup. Reusable for future iNat experiments (e.g., MixUp can be added back without I/O thrash).
4. **iNat-prodft fold-0 (v1, gentle-recipe-killed)** val_v2 = 0.7130, per-class diversity shows iNat specializes on non-Aves species (frogs).
5. **D1-b verified dead** (+0.003, not the plan's claimed +0.04).
6. **V2-S re-discovered + LB precedent re-confirmed**: V2-S 4-fold soft-vote val_v2 0.7759 (+0.047 over B0), but already LB-killed 2026-05-03 (LB 0.930 vs B0 0.931, Kaggle inference timeout). **val_v2 doesn't reliably predict LB on this stack.**

### ⚠️ Strategic context for tomorrow (do not skip)

The v4 fold-0 val_v2 result is informative about pretrain *mechanism* (does natural sampling lift iNat features?) — not deployment value. Per §14.19.11, +val Δ from a single arch/init change is "noise-band predicted" on LB until a probe submission confirms otherwise. **Treat v4's val number as a mechanism probe, not a green light to ship.**

Specifically:
- If v4 fold-0 val_v2 ≥ 0.7414 (clears ImageNet baseline) → still need an LB submission to validate; do not assume LB win.
- If v4 fold-0 val_v2 = 0.71-0.74 (between v1 and ImageNet) → recipe + sampler partially helped; LB transfer probably null.
- If v4 fold-0 val_v2 ≤ 0.71 (no improvement) → iNat lever class is exhausted, pivot off pretrain (see `docs/xc_v3_pretrain_prep.md` for the on-plan next, but caveat: also probably won't move LB per V2-S precedent).

### Bugs encountered today (for forensics if anything resurfaces)

| Bug | Symptom | Cause | Fix |
|---|---|---|---|
| iNat v2 hung 2h | GPU 0%, log silent, allocator mmap loop | MixUp `random.randint` partner draws → cold HDD random reads → ~2 h/epoch | Dropped MixUp; will add back when cache enables it |
| v3 slow at 1:35 | No e1 summary, HDD reads at 30 MB/s | 140 GB dataset doesn't fit in 61 GB RAM | Pivoted to mel-cache approach |
| Cache build deadlock | Workers at 0% CPU, no files written | PyTorch + concurrent.futures fork (default on Linux) → torch lock held in forked workers | `mp.get_context("spawn")` for ProcessPoolExecutor |
| Cache files named `.npy.tmp.npy` | All 2000 marked "bad", garbage files on disk | `np.save(str_path, ...)` auto-appends `.npy`; my tmp suffix `.npy.tmp` became `.npy.tmp.npy` on save, rename target didn't exist | Pass file handle (`open(path, "wb")`) to `np.save` — no auto-suffix when given a fileobj |

### Original 2026-05-10 ~19:30 PICK UP HERE content follows below for archival



**TL;DR — late-late session pivot:**

iNat re-pretrain v3 (natural-sampling only, no MixUp) was technically running but on a ~30 h trajectory due to HDD-random-read bottleneck. The full 95 GB train + 45 GB val dataset doesn't fit in DT's 61 GB host RAM, so OS page cache eviction means even e2+ would stay slow (60-80 min/epoch instead of the warm-cache ~40 min I'd projected). Killed at 1:35 elapsed.

**New plan: pre-compute mel spectrograms to NVMe, then re-pretrain v4 reading cached mels.**

| Strategy | Wall-clock | Why |
|---|---|---|
| v3 (current path, raw audio on HDD) | ~25-30 h | HDD random reads at ~30 MB/s; dataset doesn't fit in RAM |
| **v4 (mel cache on NVMe, train-only cache)** | **~3-4 h training + ~1-2 h cache build = ~5-6 h total** | NVMe random read at ~1-2ms per 688-KB mel; eliminates mp3-decode + resample + mel-compute CPU work |

**Why this matters strategically:**

Per V2-S §14.19.11 precedent, val_v2 doesn't reliably predict LB on this stack. Spending 30 h on v3 to chase a val signal that may not transfer is bad EV. With the mel cache, we get the SAME experimental answer in 5-6 h instead of 30 h — much better trade.

**Disk plan:**
- NVMe `/home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/data/processed/inat_mels/` (under `data/processed/` which is in `RUNON_HEAVY_EXCLUDES` so won't sync back)
- 95 GB fp16 mels (3×224×512 each = 688 KB)
- 87 GB margin on rhel-root LVM
- Val (45 GB) re-read raw audio per epoch — acceptable since val is only ~30 min on partially-warm cache and doesn't bottleneck



**TL;DR — what changed late session (after first PICK UP HERE draft):**

1. **Item 3 (V2-S cross-arch ensemble) landed.** V2-S 4-fold standalone soft-vote on val_v2 = **0.7759**, +0.0469 over B0 4-fold (0.7290). This looked like a major discovery — V2-S sitting underutilized on disk since May 3.

2. **But V2-S was already KILLED on LB.** §14.19.11 (2026-05-03 23:25) documents v72 (V2-S 3-fold, dropped fold 4 to fit Kaggle inference budget) scored **LB 0.930**, vs B0 baseline 0.931 — -0.001, in the noise band. v71 (4-fold) timed out at 101 min, above the ~90-min Kaggle cap. V2-S was reverted. **My OOF-myopia missed this.**

3. **The deeper lesson:** §14.19.11.2 explicitly documents that val_v2 doesn't reliably predict LB on this stack — 14+ killed-on-noise levers, including V2-S. "+val Δ ≥ 0.03 from arch swap" is now classified as "noise-band predicted" not "real-lever predicted" until a probe lands clean LB. **All of today's OOF gains (recipe fix, +0.0037 iNat ensemble lift, V2-S +0.047, per-class diversity findings) may not transfer to LB.**

4. **iNat re-pretrain v2 (natural-sampling + MixUp) hung for 2h.** Diagnosed at 21:55: not actually hung — running at ~2 hours/epoch due to MixUp's random partner draws (`random.randint(0, 137010)` for each second sample) causing severe OS-page-cache thrashing at full 137K-clip dataset scale. Diagnostic on 5000 clips ran fine (43.9s for 20 batches), but at 137K clips with 8 workers + BS=64, stalls grew from 9s to 24s as page cache (~30 GB) saturated against ~88 GB total dataset.

5. **Pivot: dropped MixUp.** Re-dispatched pretrain v3 with `--natural-sampling` only (no MixUp). Tests the sampler fix in isolation, which was the biggest identified asymmetry vs ImageNet anyway. MixUp can be added back later via a mel-precaching approach (option C from the decision tree).

**Active jobs — single source of truth:**

| Field | Value |
|---|---|
| **DT job** | iNat re-pretrain v3 — natural sampling only, no MixUp |
| Started | 2026-05-10 21:58:00 EDT |
| Host | deepthought |
| PID | **1758441** |
| Wrapper | `runon deepthought python -u src/pretrain_inat_sounds.py --inat-root .../inat_sounds_2024 --epochs 25 --natural-sampling` |
| Live log | `deepthought:/home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260510_215800.log` |
| Tail cmd | `ssh deepthought "tail -f .../runon_deepthought_20260510_215800.log"` |
| Status cmd | `ssh deepthought "ps -p 1758441 -o pid,etime,stat,cmd"` |
| Stop cmd | `ssh deepthought "kill 1758441"` |
| Pull results | `syncback deepthought four_track/models/` |
| Save path | `models/pretrain_inat/inat_best_tf_efficientnet_b0_ns_jft_in1k_natfreq.pt` (no _mixup50 suffix) |
| Expected wall | 25 × ~40 min/epoch ≈ 16-17 h (matching prior balanced-sampler iNat v1 pretrain) |
| Expected finish | ~14:00-15:00 EDT 2026-05-11 |

| Field | Value |
|---|---|
| **Killed job (forensics)** | iNat re-pretrain v2 — natural sampling + MixUp p=0.5 |
| Was PID | 1740606 (killed 2026-05-10 ~21:30 after 2h with no epoch summary) |
| Cause | MixUp `random.randint` partner draws → cold page cache reads → ~2h/epoch |
| Old log | `deepthought:/home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260510_192716.log` (cleared by hook on next dispatch) |
| Save path that never materialized | `inat_best_*_natfreq_mixup50.pt` |

## ⚠️ CRITICAL CONTEXT: val_v2 doesn't reliably predict LB on this stack

Per §14.19.11.2, the val_v2 → LB transfer is structurally weak:
- V2-S 4-fold val 0.7759 (+0.047 over B0) → LB 0.930 (-0.001 vs B0 0.931)
- 14+ levers killed at LB noise band despite strong val signals
- "Even with the GT-only val_v2 substrate (built explicitly to dodge the L1 leak pattern), +val didn't transfer. The val-LB gap is bigger than just the prior leak diagnosis — there's a structural fold-coverage / class-distribution gap between the 75-species soundscape val and the 234-class hidden test that even non-leaked val signals don't bridge."

**Implication for tomorrow's v3 fold-0 eval:** even if v3 hits 0.74+ on val_v2, the LB lift is probably null. The v3 result is informative about pretrain *mechanism* (does natural sampling lift iNat features?) but not about LB *deployment value*. Treat v3's val_v2 number as a mechanism probe, not a production gate.

## What the V2-S history says we should expect

The V2-S precedent (§14.19.11.4):
> "Sydorskyi's +0.046 from arch+corpus does NOT decompose linearly into +0.005-0.015 arch + ~+0.030 corpus. The arch-alone half is null on our pipeline."

By the same logic, the iNat lever's "+0.044 recipe fix" and any v3 val lift may also fail to decompose into LB lift. The recipe fix is *real* (durable code change), but its LB impact is unconfirmed.



**TL;DR — what changed today (post-pretrain → post-recipe-bug discovery):**
The completed Bundle 2 iNat pretrain (best val_roc_auc=0.9528 on iNat held-out, e21) was followed by an A1 5-fold finetune dispatch using the existing "gentle finetune recipe" (lr=1e-4 + 2-ep warmup + single cosine — auto-activates when `--init-from` is set). Fold 0 landed at val_v2 = **0.6692** — a -0.072 regression vs the production A1 baseline (0.7414, ImageNet init + production recipe). Diagnosis traced two factors: (1) the gentle recipe is itself a bug, and (2) iNat features don't beat ImageNet baseline even with the correct recipe.

**Active jobs — single source of truth:**

| Field | Value |
|---|---|
| **DT job** | iNat re-pretrain with natural sampling + MixUp (2 fixes) |
| Started | 2026-05-10 19:27:17 EDT |
| Host | deepthought |
| PID | **1740606** |
| Wrapper | `runon deepthought python -u src/pretrain_inat_sounds.py --inat-root .../inat_sounds_2024 --epochs 25 --natural-sampling --mixup-prob 0.5` |
| Live log | `deepthought:/home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260510_192716.log` |
| Tail cmd | `ssh deepthought "tail -f .../runon_deepthought_20260510_192716.log"` |
| Status cmd | `ssh deepthought "ps -p 1740606 -o pid,etime,stat,cmd"` |
| Stop cmd | `ssh deepthought "kill 1740606"` |
| Pull results | `syncback deepthought four_track/models/` |
| Save path | `models/pretrain_inat/inat_best_tf_efficientnet_b0_ns_jft_in1k_natfreq_mixup50.pt` (distinct from existing) |
| Expected wall | 25 × ~40 min/epoch ≈ 16-17 h (+ val-cache build first epoch) |
| Expected finish | ~11AM-2PM EDT 2026-05-11 |

| Field | Value |
|---|---|
| **Skynet job (queued/running while DT runs)** | Post-iNat analyses sweep — 3 sub-analyses |
| Script | `src/probe_post_inat_analyses.py` |
| Log | `log/probe_post_inat_analyses_*.log` |
| Output | `data/post_inat_analyses_results.npz` |
| Sub-analysis (1) | ImageNet fold-vs-fold correlation baseline — interprets whether iNat's 0.54-0.61 correlation is "diverse" or noise |
| Sub-analysis (2) | Per-class diversity breakdown — which species iNat agrees/disagrees with ImageNet on (Aves vs non-Aves split was lever-targeting hypothesis) |
| Sub-analysis (3) | V2-S cross-arch ensemble — inference on V2-S fold 0/1/2/4 ckpts, B0+V2-S+iNat ensemble AUC sweep |
| Initial probe (already done 2026-05-10 19:50) | `src/probe_inat_prodft_ensemble.py` — iNat-prodft fold 0 standalone 0.7129, 5-input soft-vote 0.7327 (Δ +0.0037 vs 4-fold baseline 0.7290), correlation 0.54-0.61 |

### What we did this session — 5 distinct findings

**1. iNat pretrain (resume-to-e25) completed cleanly.** Started overnight; e21 was best at val_roc_auc=0.9528 on iNat held-out. e21 ckpt saved as `inat_best_*_jft_in1k.pt`. (Pre-existing PICK UP HERE entry covered this; superseded below.)

**2. A1 5-fold finetune from iNat ckpt — fold 0 + skynet fold 4 dispatched in parallel (CLAUDE.md 4:1 split).**
- DT folds 0,1,2,3 sequential (PID 1717963); skynet fold 4 (kaggle-arch env, PID 51204 wrapper / 51204 python)
- Default training path auto-activates "gentle finetune recipe" (lr=1e-4 + warmup + single cosine) when `--init-from` is set
- Fold 0 finished: best **0.6692** (e22 ★ BEST, e25 final 0.6658). -0.072 vs production baseline (0.7414). DT killed after fold 0; fold 1 ckpt is partial (kept for forensics)
- Skynet fold 4 killed at e20: best **0.6961** (e17 ★ BEST). Comparable shape to fold 0 but +0.027 better — per-fold variance is large

**3. Recipe-bug discovery and `--ft-recipe production` flag (durable code change).**
Two independent pretrain levers (L2 multi-year, iNat) both landed at fold-0 val ~0.67-0.68 under the gentle recipe. Only ImageNet-init (which uses the production recipe via timm's built-in weight load, NOT `--init-from`) hits 0.7414. **The gentle recipe is the only common factor between the two failures.** Added `--ft-recipe {gentle, production}` flag to `src/train_a1.py` (default `gentle` for backward compat). Save path now gets `_prodft` suffix when production recipe is set — separate ckpt from gentle.

Fold 0 re-run with iNat-init + production recipe: best **0.7130** (e21 ★ BEST). **+0.044 over gentle**, still -0.028 vs ImageNet baseline. Pattern: warm-restart cosine produces successive cycle peaks (e9 = 0.7040, e21 = 0.7130, ~+0.009 per cycle), diminishing returns.

**Strategic conclusion: the gentle recipe was a real bug; the iNat lever class is partially salvaged but doesn't beat ImageNet.**

**4. D1-b per-fold temperature scaling: dead (+0.003).**
Plan line 744 claimed A1's 5-fold soft-vote (0.7017) was below 4 of its 5 individual folds, implying +0.04 AUC of recoverable signal via per-fold temperature scaling. Direct measurement (`src/d1b_per_fold_temp_scaling.py` on `data/v56_soundscape_oof.npz`) showed:
- Actual uncalibrated soft-vote: **0.7290** (not 0.7017)
- After per-fold temperature scaling: 0.7320 (+0.003 — below noise floor)
- The "soft-vote < worst individual fold" phenomenon described in line 744 was **not present** in the OOF
- 0.7017 likely refers to including fold 3 (dropped) or rank fusion with B1, not direct soft-vote
- See `memory/feedback_verify_plan_numeric_claims.md` for the durable lesson

**5. Other asymmetries identified — fixes implemented for iNat re-pretrain v2.**
Beyond the gentle recipe, iNat pretrain had two structural mismatches vs A1 finetune:
- **Sampling**: iNat used `WeightedRandomSampler(1/n_per_class)` — fully class-balanced — vs A1's natural shuffle. Backbone learned to discriminate 5569 species equiprobably, miscalibrating for BC2026's Aves-heavy natural distribution.
- **MixUp**: iNat had none; A1 uses 0.5/0.5 waveform MixUp with element-wise-max labels. The MixUp also produces multi-positive training rows — addresses iNat's single-label-only structure indirectly.

(Note: A1 also has bg-noise code but `bg_noise_dir=None` in practice → no-op. Not a real asymmetry. Initial diagnosis had this wrong.)

Added two flags to `src/pretrain_inat_sounds.py`: `--natural-sampling` (drop WeightedRandomSampler → shuffle=True) and `--mixup-prob` (0.0 default; set 0.5 to enable). Save path gets suffix `_natfreq_mixup50` when both active. Smoke-tested on DT 2026-05-10 19:25:24 EDT, passed in 32s.

### Recovered baseline numbers (verified 2026-05-10 evening)

These numbers come from direct measurement on `data/v56_soundscape_oof.npz` (1478 segments, 234 classes, folds 0/1/2/4 — fold 3 dropped):

| Metric | Value |
|---|---|
| fold 0 (A1, ImageNet) | 0.7415 |
| fold 1 (A1, ImageNet) | 0.7227 |
| fold 2 (A1, ImageNet) | 0.6975 |
| fold 4 (A1, ImageNet) | 0.7248 |
| 4-fold soft-vote (uncal) | **0.7290** |
| 4-fold soft-vote (per-fold temp-scaled) | 0.7320 |

The 0.7290 figure should be the canonical baseline for ensemble comparisons going forward, not the 0.7017 cited in §line 744.

### Ckpt inventory after today

On skynet (`four_track/models/`):
- `pretrain_inat/inat_best_tf_efficientnet_b0_ns_jft_in1k.pt` — iNat pretrain best (e21, val 0.9528 on iNat held-out, balanced sampler + no MixUp, **legacy from earlier 2026-05-10 run**)
- `a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid.pt` — A1 fold 0, iNat init + gentle recipe, val_v2 0.6692
- `a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid_prodft.pt` — A1 fold 0, iNat init + production recipe, val_v2 0.7130 ← **most informative new ckpt**
- `a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold4_seed42_hybrid.pt` — A1 fold 4, iNat init + gentle recipe, val_v2 0.6961

On deepthought (will be populated as re-pretrain runs):
- `models/pretrain_inat/inat_best_*_natfreq_mixup50.pt` — pending, will overwrite as the v2 pretrain runs

### Queued during the DT re-pretrain wait window

Six tracked items, in EV order. Items 1-3 run via `src/probe_post_inat_analyses.py` (skynet, ~20 min). Items 4-6 are pre-staged deliverables (no compute):

| # | Item | Deliverable / Output | Status |
|---|---|---|---|
| 1 | ImageNet fold-vs-fold correlation baseline | mean r = 0.6554 (not 0.85+ as assumed) | **Done** — see Results from items 1-2 below |
| 2 | Per-class diversity breakdown | iNat specializes on non-Aves (frogs); agrees with ImageNet on birds | **Done** — see Results below |
| 3 | V2-S cross-arch ensemble check | V2-S soft-vote 0.7759 (+0.047 OOF) but LB-killed per §14.19.11 | **Done** — see Results from item 3 below |
| 4 | Tomorrow's v3 fold-0 dispatch script | `scripts/dispatch_a1_inat_v2_fold0.sh` (executable; rename to ..._v3_fold0 when v3 ckpt lands, OR edit ckpt path inside the script — it points at `_natfreq_mixup50.pt` which won't exist; needs update to `_natfreq.pt` before use) | Done (needs ckpt-path fix) |
| 5 | XC v3 pretrain prep doc | `docs/xc_v3_pretrain_prep.md` | Done |
| 6 | Memory entries from today's findings | `memory/feedback_read_call_sites_not_docstrings.md`, `memory/reference_b0_sed_skynet_dt_ratio.md` | Done |
| **7** | **Diagnose iNat v2 MixUp hang** | `src/diagnose_dataloader_pretrain.py`; root cause = random partner draws cause cold-cache I/O at 137K-clip scale | **Done** — see CRITICAL CONTEXT above |
| **8** | **Re-pretrain v3 (natural-sampling only, no MixUp)** | PID 1758441 on DT, save path `inat_best_*_natfreq.pt` | **In flight** — see active jobs above |

**Decision logic after items 1-3 land:**
- (1) If ImageNet fold-vs-fold r > 0.80, iNat's 0.55 IS diverse → re-pretrain v2 ensemble lift is real; commit to v2 5-fold dispatch after DT finishes
- (1) If ImageNet fold-vs-fold r ≈ 0.55-0.65, "diversity" was val-set noise; v2 has no special ensemble value; lever is recipe-validation only
- (3) If V2-S 4-fold standalone beats B0 4-fold, or B0+V2-S 8-input beats B0 4-fold by >+0.005, cross-arch is the higher-leverage lever than continuing on iNat — pivot compute to V2-S 5-fold completion and ConvNeXt addition after DT finishes
- (3) If cross-arch lift is similar to within-arch (<+0.002), confirms backbones learn similar things and iNat-style cross-init is genuinely the diversity-frontier — strengthens the v2 thesis

### Results from items 1-2 (landed 2026-05-10 ~20:05; item 3 in flight)

**Analysis 1 — ImageNet fold-vs-fold correlation baseline.** The original probe (`src/probe_inat_prodft_ensemble.py`) reported iNat-prodft vs each ImageNet fold at r=0.54-0.61. We had assumed ImageNet folds (same backbone, different fold split) would correlate at r=0.85+, which would have made iNat "categorically diverse." Direct measurement:

| Pair | r |
|---|---|
| fold 0 vs 1 | 0.6045 |
| fold 0 vs 2 | 0.6153 |
| fold 0 vs 4 | 0.6223 |
| fold 1 vs 2 | 0.6888 |
| fold 1 vs 4 | 0.6947 |
| fold 2 vs 4 | 0.7068 |
| **mean ImageNet fold-vs-fold r** | **0.6554** |

**Verdict: iNat is only MODERATELY diverse**, not categorically. ImageNet folds are already partly decorrelated (r ≈ 0.66), and iNat-prodft sits only ~0.10 below that floor. The "iNat is a genuinely different thinker" framing from the initial probe writeup was overstated. The +0.0037 ensemble lift is consistent with this — small marginal-diversity contribution rather than a fundamentally new signal source. **This decisively answers (1) in the decision logic above: the lever is recipe-validation-only, not a clean ensemble-diversity win, UNLESS per-class specialization (Analysis 2) buys us a different framing.**

**Analysis 2 — Per-class diversity breakdown.** Per-class Pearson r distribution (iNat-prodft vs ImageNet soft-vote, 75 present classes):

| Bucket | Count |
|---|---|
| r > 0.80 (iNat agrees with ImageNet) | 12 |
| r 0.40-0.80 (mid) | 37 |
| r < 0.40 (iNat disagrees with ImageNet) | 26 |
| **mean r** | **0.5088** |

**Top 10 LEAST-correlated species (where iNat is most different):** entirely **AnuraSet codes** (47158son12, 47158son25, 47158son05, ..., 74113) — frog / amphibian species.

**Top 10 MOST-correlated species (where iNat agrees with ImageNet):** entirely **bird species** — whtdov (white-tipped dove, r=0.96), chacha1 (chachalaca), bufpar (buff-fronted parakeet), grekis (great kiskadee), etc.

**This confirms the original lever-targeting hypothesis from §14.22.10.5: iNat pretraining specifically benefits non-Aves species.** iNat's training corpus includes 17,187 Amphibia clips, 13,145 Insecta, 3,549 Mammalia, 203 Reptilia (out of 182,710 total) — exposing the backbone to acoustic patterns ImageNet never sees. ImageNet-init models learn frog features only from BC2026 train_audio's limited frog samples; iNat-init enters BC2026 already having seen many.

**The strategic implication is more nuanced than a flat soft-vote can express.** Equal-weight ensembling dilutes iNat's specific value because birds dominate val_v2 (75 present species, mostly Aves). A **species-aware weighting scheme** — boost iNat for the 26 low-correlation (non-Aves) classes, near-zero for the 12 high-correlation (bird) classes — could extract more than the +0.0037 we got. But fitting per-species weights on 75 species is overfitting territory; needs careful held-out validation or a structural constraint (e.g., weight ∝ 1/iNat-ImageNet correlation, no per-species fitting).

**Update to the v2 expectation:** the running re-pretrain (natural sampling + MixUp + production recipe) should not just be evaluated by individual fold-0 AUC. The decisive question is whether v2 **preserves the per-class taxonomic specialization** (low correlation on non-Aves, high on Aves) while lifting individual AUC. If it lifts individual AUC by destroying the specialization, the ensemble value drops to zero. If it lifts AUC AND keeps the specialization, the lever has real ensemble headroom.

**Re-run plan:** when v3 lands, re-execute `src/probe_inat_prodft_ensemble.py` against the new ckpt, then re-execute `src/probe_post_inat_analyses.py` Analyses 1+2 to compare per-class correlation patterns v1 vs v3. NOTE: per the V2-S LB precedent above, any val_v2 lift in v3 should not be treated as a deployment gate — only LB submission proves real value.

### Results from item 3 (V2-S cross-arch, landed 2026-05-10 ~20:20)

Item 3 result was the largest single-number finding of the day — V2-S 4-fold soft-vote on val_v2 = **0.7759** vs B0 4-fold 0.7290 (+0.0469). V2-S per-fold AUCs (0.7449, 0.7624, 0.7640, 0.7406) all match or beat the B0 production baseline of 0.7414 individually.

Cross-arch correlations B0 vs V2-S, fold-pair: 0.55-0.71 (within-fold 0.58-0.65). Not categorically more diverse than within-backbone (ImageNet B0 folds correlate at 0.66 with each other). V2-S's individual quality is what drives the ensemble lift, not architectural diversity.

Ensemble comparisons:
- B0 4-fold: 0.7290 (baseline)
- V2-S 4-fold (alone): **0.7759** (+0.0469)
- 8-input (B0 4 + V2-S 4): 0.7669 (+0.0379) — worse than V2-S alone
- 9-input (+ iNat-prodft): 0.7640 (+0.0350) — iNat HURTS the V2-S-dominant ensemble
- Weighted w_v2s=0.75: 0.7728 — approaches V2-S-alone

**But — see V2-S LB precedent in the CRITICAL CONTEXT subsection above.** V2-S was tested on Kaggle 2026-05-03 (v72, 3-fold to fit inference budget) and scored LB 0.930 vs B0 baseline 0.931 — null on LB. The +0.047 OOF lift didn't transfer. V2-S 4-fold also can't run on Kaggle due to inference timeout (~101 min vs ~90 min cap). So this finding is informative about val_v2 but **not actionable for LB deployment**.



### Wake-up checklist for next session

1. **Tail DT re-pretrain log.** Confirm it's progressing through epochs at ~40 min/epoch. If crashed (zero-frame file etc.), the scan from the prior PICK UP HERE already cleaned the manifest, so re-crash is unlikely.
2. **Check best val_auc.** If e25 best clears 0.96 (above prior 0.9528), the fixes lifted iNat held-out performance — encouraging signal for downstream transfer. If similar (~0.95), the fixes mainly change the DISTRIBUTION the model attends to, not raw discrimination.
3. **`syncback deepthought four_track/models/`** to pull the new ckpt.
4. **Run A1 fold-0 finetune with the v2 backbone** using `--init-from <new ckpt> --ft-recipe production`. ~85 min on DT. The single most important number: does v2 backbone + production recipe close the -0.028 gap to ImageNet (0.7414)?
5. **Skynet probe result** (whether dispatched tonight or tomorrow): does iNat-prodft fold 0 (val_v2 0.7130) add ensemble lift to the 4-fold soft-vote (0.7290)? If +0.005-0.010, iNat ckpts have hidden value for ensembling even individually below baseline. If nil, iNat-init has zero strategic value.

### Decision tree for v2 fold-0 result

| v2 fold-0 production-recipe val_v2 | Interpretation | Action |
|---|---|---|
| ≥0.7414 | Recipe was the dominant bug; fixes work | Run folds 1, 2, 3 on DT + re-run fold 4 with production. ~5.4 h. Build full 5-fold ensemble. |
| 0.72-0.7414 | Partial recovery; iNat still below ImageNet but closer | Hold; consider whether to commit to 5-fold for ensemble diversity (depends on skynet probe result) |
| 0.71 (same as v1 prodft) | Sampler + MixUp didn't help | iNat lever class is saturated. Pivot to XC v3 or off-pretrain (different backbones, post-processing) |
| <0.71 | The fixes broke something | Investigate; likely MixUp interaction with single-label iNat or BN drift |

### Don'ts on next pickup

- **Don't keep running iNat pretrain variants without a hypothesis.** Two pretrain attempts already exhausted; a third needs a specific testable failure mode (e.g., "v2 didn't help because X; if we change Y, expected gain Z"). Vague "try more epochs / different lr" isn't enough.
- **Don't restart iNat pretrain on the existing flag combinations.** v1 (balanced + no mixup) and v2 (natural + mixup50) are running/done; further variants need at least one new dimension (e.g., mixup-prob 0.3 instead of 0.5, or a hybrid sampler).
- **Don't trust plan numeric claims about specific AUC gaps** without direct measurement when measurement is cheap. See `memory/feedback_verify_plan_numeric_claims.md`.
- **Don't dispatch D2 stacker variants.** D2 is plan-dead per §line 1436 (3 variants tried, substrate non-predictive). D1-b just confirmed a related deadness pattern.

---

## ⏸️ PICK UP HERE — previous (2026-05-10 ~01:25 local — iNat pretrain crashed @ e14, FIX + RESUME-TO-E25 RUNNING overnight on deepthought — SUPERSEDED by 2026-05-10 ~19:30 entry above)

**TL;DR:** The Bundle 2 step 5 pretrain dispatched 2026-05-09 15:12 EDT
crashed at epoch 14/50 with a zero-length `.wav` file. 13 clean epochs
completed (best `val_roc_auc=0.9484` at e13). Crash root cause patched
in `_load_waveform`; manifest scan + resume-to-e25 dispatched at
2026-05-10 01:24:51 EDT and is running unattended overnight.

**Active job — single source of truth for "where is the log":**

| Field | Value |
|---|---|
| Started | 2026-05-10 01:24:51 EDT |
| Host | deepthought |
| PID | **1680930** |
| Wrapper | `bash four_track/scripts/inat_scan_and_resume.sh` |
| Live log | `deepthought:/home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260510_012451.log` |
| Tail cmd | `ssh deepthought "tail -f /home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260510_012451.log"` |
| Status cmd | `ssh deepthought "ps -p 1680930 -o pid,etime,stat,cmd"` |
| Stop cmd | `ssh deepthought "kill 1680930"` |
| Pull results | `syncback deepthought` |
| Expected wall | scan ~5–15 min + 12 epochs × ~40 min = ~8 h total |
| Expected finish | ~09:30 EDT 2026-05-10 |

**Predecessor crash log (epochs 1–13 + traceback) — kept for forensics:**

`deepthought:/home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260509_151230.log`

### What we did this session

**1. Diagnosed the e14 crash.**
- Error: `RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 0]`
  in `torchaudio.functional.resample`.
- Chain: bad `.wav` file → `sf.read` returns empty array (no exception) →
  `BirdCLEF/src/utils.py:78` resample with `(1, 0)` tensor → torchaudio
  internals fail.
- Why epoch 14: `make_balanced_sampler` uses `1/n_per_class` weights, so a
  bad file in a high-cardinality Aves class has tiny per-epoch draw
  probability. 13 clean epochs is statistically plausible; uniform
  sampling would have hit it in epoch 1.

**2. Patched `four_track/src/pretrain_inat_sounds.py:_load_waveform`.**
Try/except around `load_audio` + `pad_or_crop`, plus explicit
`wav.size == 0` check. On any failure: print `[skip-bad] <name>:
<ExceptionType>: <msg>` and return `np.zeros(CHUNK_SAMPLES, float32)`.
Parent `BirdCLEF/src/utils.py` left untouched (read-only legacy rule).

**3. Added `four_track/src/scan_inat_manifest.py`.**
Multiprocess `sf.info` scan over all 182,710 rows; drops zero-frame /
unreadable files. Backs up original to `inat_manifest_orig.csv` (only on
first run); overwrites `inat_manifest.csv` with cleaned rows.

**4. Added `--resume` and `--start-epoch` flags to `pretrain_inat_sounds.py`.**
Loads model state from a prior ckpt, fast-forwards the cosine scheduler
`start_epoch` steps, initializes `best_auc` from `ckpt['val_auc']` so the
saved-best ckpt is only overwritten by a strictly-better epoch.

**5. Added `four_track/scripts/inat_scan_and_resume.sh`.**
Sequential wrapper: scan → resume. `set -euo pipefail` ensures the
resume step is skipped if the scan fails.

**6. Killed the orphan poller.** `probe_e30_dispatch.sh` (PID 1660921)
on deepthought was polling forever for an epoch 30 that would never
materialize — terminated.

### Configuration of the resumed run

| Param | Value | Why |
|---|---|---|
| Resume from | `inat_best_tf_efficientnet_b0_ns_jft_in1k.pt` (e13, val_roc_auc=0.9484) | Latest best ckpt on deepthought |
| `--start-epoch` | 13 | Loop runs e14..e25 inclusive |
| `--epochs` | **25** (NOT 50) | `train_loss=0.0000` from e6 onward — model has memorized iNat training; diminishing transfer returns past e25 |
| Cosine `T_max` | 25 | LR follows a steeper descent than the original 50-epoch schedule; reaches `lr_min=1e-6` by e25 |
| LR at e14 | ~1.1e-4 | `2.5e-4 * 0.5 * (1 + cos(π * 13/25))` |

Saved ckpts on deepthought (`/mnt/mytoshiba/.../models/pretrain_inat/`):
- `inat_backbone_..._jft_in1k_e10.pt` — milestone (`save_every=10`)
- `inat_best_..._jft_in1k.pt` — e13 best, **safe** (only overwritten by
  strictly-better val_auc; if e14–e25 don't beat e13 it stays unchanged)
- New milestone at e20 will be written as `..._e20.pt` (different filename,
  doesn't clobber e10).

### Wake-up checklist

1. **Tail the log** — confirm scan reported a bad-file count:
   - `0 / 182710` → patched `_load_waveform` was the only line of defense;
     grep for `[skip-bad]` to see if it ever fired.
   - `>0 / 182710` → cleaner manifest is in use; small count is benign,
     large count (>>100) means the iNat extraction had broader silent
     failures and the corpus may be partially corrupt.
2. **Confirm e14–e25 ran.** Each epoch ~40 min; expect 12 epoch-summary
   blocks. If process died mid-resume, the e13 best ckpt is still good.
3. **Check final `val_roc_auc`.** If e25 best > 0.9484 (e13 best), the
   extra 12 epochs paid off. If not, e13 was already the ceiling for this
   recipe and the next iNat pretrain experiment should change *something*
   (corpus, lr, sampler) rather than just running longer.
4. **`syncback deepthought`** to pull `models/pretrain_inat/*.pt` and the
   log back to skynet.

### Next gate after pretrain finishes

Channel-B 184-sp finetune using the e25 (or e13, whichever is best)
backbone, target focal AUC ≥ 0.9545 per §14.22.10.5 step 6. Verify the
finetune script accepts the iNat-pretrained backbone format (`state_dict`
key + iNat species buffer) before dispatch.

### Don'ts on next pickup

- **Don't keep training past e25 by default.** train_loss=0.0000 means
  the LR schedule is doing maintenance work, not learning. If e25 best
  doesn't pass the downstream gate, the lever was wrong — switch corpus
  (XC v3 already downloaded) or revisit head-fix, don't add more iNat
  epochs.
- **Don't rsync `data/external/` or `models/`.** Both are in
  `RUNON_HEAVY_EXCLUDES`. The deepthought-side iNat manifest (cleaned by
  the scan) and ckpts are not mirrored on skynet by design — rsync would
  not bring them back without `syncback`.
- **Don't restore `inat_manifest.csv` from `inat_manifest_orig.csv`**
  unless you specifically want to retest crash behavior. The cleaned
  manifest is the working version going forward.

---

## 14.23 Hyperparameter Optimization plan — DEFERRED, gated, not on the execution path (2026-05-09 ~21:00 local)

> **Status: SHELF. DO NOT EXECUTE without an explicit trigger from Phase 0
> below.** This section exists so the procedure is ready when/if it
> becomes the highest-leverage next move. As of 2026-05-09 it is not.
> Sourced from *The Kaggle Book* (2nd ed.) Chapter 9 + cross-checked
> against four_track's actual compute envelope.

### 14.23.1 Why this is shelved

Three reasons HPO is not in the queue right now, all of which need to
flip before this section unlocks:

1. **Plan's next gate isn't HPO.** §14.22.10.5 step 6 = iNat finetune +
   Channel-B 184-sp focal AUC ≥ 0.9545. If the gate fails, the lever
   was wrong (switch external corpus or revisit head-fix); if it
   passes, LB submit. HPO is not in either branch.
2. **Targeted hand-tuning has been outperforming Optuna-style sweeps.**
   `loss=hybrid`, `mixstyle_p=0.5`, `multi_layer_gem`, `swa_start_frac=0.65`,
   `swa_lr=4e-4` are all hand-tuned via single-variable A/B in the
   §14.10 / §14.13 era — each landed +0.005–0.020 LB. Ch 9 (p. 326)
   itself quotes Hinton: top DL practitioners win from pretrained
   models + papers + top notebooks + trial and error, with HPO as the
   *backstop*, not the headline. We are not at the backstop yet.
3. **Compute math is brutal.** 5-fold A1 ≈ 5.5 h on deepthought.
   50-trial × 5-fold Optuna study ≈ 11.5 days. Even fold-0-only with
   pruning ≈ 2 days, competing with iNat finetune, ProtoSSM iterations,
   and any LB submit slots. Likely return: +0.001–0.005 LB. Bundle 2
   targets +0.005–0.020. Wrong allocation of the next compute window.

### 14.23.2 Phase 0 — Trigger conditions (don't unshelf without one)

Run this section's plan only if **one** of these fires:

- **(T1)** iNat finetune lands and plateaus 0.001–0.003 *below* the
  Channel-B 184-sp gate (0.9545). HPO becomes the cheapest remaining
  lever to bridge the gap before falling back to a different external
  corpus. Sentinel value: post-finetune Channel-B AUC ∈ [0.9515, 0.9544].
- **(T2)** iNat finetune passes the gate, LB confirms a new ceiling,
  and Tracks B/C/D are all blocked (Bundle 3 XC v3 done, ProtoSSM
  ceiling identified, stacking saturated). The "last 0.002–0.004
  before freezing for ensembling" scenario.

If neither fires, skip. Reassess every plan checkpoint.

### 14.23.3 Framework choice

- **Optuna** with TPE sampler. Ch 9 §2180 documents this is the
  Kaggler-standard for both tabular and DNN since 2018.
- **MedianPruner(n_warmup_steps=8)** so weak trials die at ep 8 of 25
  rather than running full 25 ep × 25-ep cost.
- **SQLite storage backend**: `storage="sqlite:///four_track/data/hpo_a1.db"`.
  Resumable across machines, inspectable with `optuna-dashboard`.
- **NOT** scikit-optimize, KerasTuner, or W&B Sweeps. Optuna is the
  only one we already have installed in the kaggle env, and the
  chapter's empirical claim is TPE > GP > random for landscapes with
  flag-like params (which mixstyle_p, multi_layer_gem, swa toggles
  all are — Ch 9 §1510 explicitly warns about this).

### 14.23.4 Search space — ≤7 params

Ch 9 §1499–§1503 warns the optimization landscape goes pathological
past ~10–15 params and Bayesian methods stop helping. Cap at 7:

| param | suggest | range / choices | rationale |
|---|---|---|---|
| `lr` | float, log | 1e-5 → 5e-3 | single biggest lever per Ch 9 §1455 |
| `batch_size` | categorical | [24, 32, 48, 64] | interacts with lr |
| `mixstyle_p` | float | 0.0 → 0.7 | only tried 0.0 and 0.5; flag-like |
| `att_dropout` | float | 0.0 → 0.5 | cheap; interacts with iNat-init encoder |
| `focal_gamma` | float | 1.0 → 3.0 | hybrid-loss component, never swept |
| `swa_start_frac` | float | 0.5 → 0.85 | hand-picked 0.65, could be off |
| `mixup_alpha` | float | 0.0 → 0.6 | augment strength, high-leverage on imbalance |

**Held fixed** — do NOT co-vary architecture and training params (Ch 9
§1503 pathology):
- backbone (whatever Bundle 2 finetune produced)
- 5-fold split + fold assignment
- audio pipeline (sample rate, mel bins, hop, n_fft)
- post-processing chain (calibration, smoothing)

### 14.23.5 Search budget

| stage | trials | per-trial cost | total |
|---|---|---|---|
| random warmup (Ch 9 §1740 pattern) | 10 | ~50 min fold-0 ep 25 | ~8 h |
| TPE main search | 30 | ~30 min avg with pruning | ~15 h |
| **HPO subtotal** | **40** | — | **~23 h** |
| 5-fold confirmation of winner | 1×5 folds | ~1 h/fold | ~5 h |
| **Total wall-clock** | — | — | **~28 h** |

Single machine, sequential, on **deepthought** — Ch 9 §1517 explicitly
endorses this regime. No parallel-machine HPO across deepthought +
skynet: the 4.5× speed gap (CLAUDE.md "Two-GPU workflow") would mean
skynet trials gate the study; allocating skynet to other work is
strictly better than burning it on slow trials.

Pre-flight before dispatch: `ssh deepthought nvidia-smi` (idle
required), and confirm no canonical 5-fold A1 run is mid-flight.

### 14.23.6 Validation discipline — where most HPO breaks

- **Fold-0 only during search.** 5× cost reduction per trial.
- **Single 5-fold confirmation** for the winner. If 5-fold mean is
  within ±0.003 of the fold-0 winning score, it's real. Otherwise the
  search overfit fold-0 and the winner is rejected — re-run search
  with seed-shuffled fold-0 OR pivot off HPO.
- **LB gate**: only submit if 5-fold mean exceeds the current
  production model's 5-fold mean by ≥ 0.003 OOF. This matches the
  existing single-fold-noise threshold from `feedback_single_fold_noise_floor.md`.
- **Save state**: SQLite study + top-3 trial configs (not just winner).
  Ch 9 §1768 doctrine; the runner-up may transfer better to ensembling
  even if it lost on val.

### 14.23.7 Seed the study with the current production point

Ch 9 §2295 pattern (`enqueue_trial`):

```python
study.enqueue_trial({
    "lr": 5e-4, "batch_size": 32, "mixstyle_p": 0.5,
    "att_dropout": 0.3, "focal_gamma": 2.0,
    "swa_start_frac": 0.65, "mixup_alpha": 0.4,
})
```

This makes Optuna treat the hand-tuned baseline as a real point in the
search; TPE then can't propose worse without learning from it. Without
this, the warmup random trials waste budget re-discovering ground we
already covered.

### 14.23.8 Skeleton script (for when this unshelfs)

Target file: `four_track/src/hpo_a1.py`. Not written yet — only stub
the imports + objective shape so the writer doesn't have to re-derive:

```python
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

def objective(trial):
    # 1. Sample params
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [24, 32, 48, 64]),
        "mixstyle_p": trial.suggest_float("mixstyle_p", 0.0, 0.7),
        "att_dropout": trial.suggest_float("att_dropout", 0.0, 0.5),
        "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 3.0),
        "swa_start_frac": trial.suggest_float("swa_start_frac", 0.5, 0.85),
        "mixup_alpha": trial.suggest_float("mixup_alpha", 0.0, 0.6),
    }
    # 2. Run train_a1.py fold-0 in-process or as subprocess; feed
    #    PrunerCallback per-epoch val_roc_auc back to trial.report().
    # 3. Return final fold-0 val_roc_auc.

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42, n_startup_trials=10),
    pruner=MedianPruner(n_warmup_steps=8),
    storage="sqlite:///four_track/data/hpo_a1.db",
    study_name="a1_post_inat_v1",
    load_if_exists=True,
)
study.enqueue_trial({...current production...})
study.optimize(objective, n_trials=40)
```

`train_a1.py` will need a `--hpo-trial-report-fn` hook (or just a
callback hook in the existing per-epoch summary) to feed Optuna's
pruner. Pruning callback is the difference between 28 h and 60+ h
total cost.

### 14.23.9 What this plan deliberately does not include

- **NAS / architecture search** (Ch 9 §1453). The backbone is fixed by
  Bundle 2 outcome; co-varying architecture and training params
  triggers the §1503 pathology.
- **W&B Sweeps** (Ch 9 §2662). Adds an external tracking surface and
  another secret to manage; Optuna + SQLite covers everything we need
  locally.
- **Ensembling-aware HPO** (different objective: maximize ensemble
  diversity). Premature — wait until Track D2 stacking is the
  bottleneck.
- **Per-fold HPO** (one study per fold). Cost-prohibitive (5× this
  plan's budget). Re-evaluate only if the 5-fold confirmation reveals
  per-fold winners diverge sharply.

### 14.23.10 Decision log

- 2026-05-09 ~21:00: section written and shelved. No execution. To
  unshelf, write the trigger (T1 or T2 from §14.23.2) into the next
  plan checkpoint with the observed AUC and the date, then proceed
  to §14.23.3.

---

## ⏸️ PICK UP HERE (2026-05-11 ~23:00 local — pretrain v4 RUNNING on deepthought, ~3 epochs landed, full 25-epoch run authorized overnight)

> **You went to bed at 2026-05-11 ~23:00 EDT. Pretrain v4 is running cleanly on deepthought with both train and val mel caches on real NVMe. Expected completion ~02:20 EDT 2026-05-12.**

### ☀️ When you wake up — do this

1. **Check pretrain v4 finished cleanly.**
   ```bash
   ssh deepthought "ps -p 43593 2>&1 | tail -1; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader; grep -E 'Epoch.*train_loss' /home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260511_222049.log | tail -5; echo '==='; ls -la /mnt/mytoshiba/MachineLearning/_runon/BirdCLEF/four_track/models/pretrain_inat/"
   ```
   - Expected: process gone, 25/25 epoch summaries in log, best ckpt updated.
   - Save path: `models/pretrain_inat/inat_best_tf_efficientnet_b0_ns_jft_in1k_natfreq.pt`

2. **Pull ckpt + log back to skynet:**
   ```bash
   syncback deepthought four_track/models/
   ```
   Note: the iNat ckpt path on deepthought is on HDD (`/mnt/mytoshiba/...`), so syncback reads from HDD — moderate I/O, no SSH wedge risk after run is done.

3. **Compare v4's final val_roc_auc against v1's e13 ceiling (0.9484).**
   - At 2026-05-11 22:50 (epoch 3), v4 was already at 0.9436 — essentially matching v1.
   - The trajectory (0.6466 → 0.8670 → 0.9436) suggests v4 will land 0.95+.
   - If v4 final < 0.9484: surprising; natural sampling may have hurt val signal. Investigate.
   - If v4 final ≥ 0.9484: expected; doesn't yet prove downstream lift.

4. **The actual experimental question:** does v4's iNat ckpt beat v1's at Channel-B 184-sp finetune? (Per §14.22.10.5 step 6 gate: focal AUC ≥ 0.9545.)
   - Next action: edit `four_track/scripts/dispatch_a1_inat_v2_fold0.sh` to point `--init-from` at the new ckpt (`_natfreq.pt` — without `_melcache` suffix; the script name didn't change).
   - Dispatch fold-0 A1 finetune. ~85 min on deepthought.
   - Compare against the iNat-prodft v1 fold-0 val_v2 baseline of 0.7130.

### ⚙️ Active jobs at sleep-time

| Field | Value |
|---|---|
| **DT job** | iNat pretrain v4 — natural sampling, train+val mel cache on NVMe |
| Started | 2026-05-11 22:20:50 EDT |
| Host | deepthought |
| PID | **43593** |
| Wrapper | `runon deepthought python -u src/pretrain_inat_sounds.py --inat-root .../inat_sounds_2024 --epochs 25 --natural-sampling --mel-cache-dir /mnt/nvme/inat_mels` |
| Live log | `deepthought:/home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260511_222049.log` |
| Tail | `ssh deepthought "tail -f .../runon_deepthought_20260511_222049.log"` |
| Status | `ssh deepthought "ps -p 43593 -o pid,etime,stat,cmd"` |
| Stop | `ssh deepthought "kill 43593"` |
| Pull | `syncback deepthought four_track/models/` |
| Save path | `models/pretrain_inat/inat_best_tf_efficientnet_b0_ns_jft_in1k_natfreq.pt` |
| Per-epoch | ~9.5 min (e1=10m06s, e2=9m19s, e3=9m23s) |
| Expected finish | **~02:20 EDT 2026-05-12** |
| Last seen | e3/25 val_roc_auc=0.9436 (already at v1 ceiling) |

### What we did this session (durable wins)

**1. Diagnosed: deepthought has NO NVMe data filesystem out of the box.**
- `/home`, `/`, `/var` all live on rhel-pool00_tdata → backed by **sda HDD** (TOSHIBA HDWG440).
- `/mnt/mytoshiba` is sdb HDD.
- `/home/swatson/work/MachineLearning` is a **symlink → `/mnt/mytoshiba/MachineLearning`** — anything written under it is HDD.
- nvme0n1 (465 GB) was used only by LVM thin-pool metadata + a 120 GB swap LV.
- The plan's earlier "NVMe, 87 GB margin" was wrong — referred to thin-pool free space, not real NVMe.

**2. Built `/mnt/nvme` = the only real NVMe filesystem on deepthought.**
- Initial: 95 GB carved from existing PFree on nvme0n1p1 (no swap surgery).
- Later expanded to **214 GB** by `swapoff` + `lvremove rhel/swap` + `lvextend +119G` + `xfs_growfs`.
- Persisted in `/etc/fstab`. fstab swap line removed; original at `/etc/fstab.bak2`.
- **deepthought now has zero swap.** OOM-killer fires immediately on memory pressure. 61 GB RAM with no fallback. Reversal documented in `four_track/CLAUDE.md` (non-trivial; xfs can't shrink in place).

**3. Built train + val mel caches on /mnt/nvme.**
- Train: `/mnt/nvme/inat_mels/train_*.npy` (137,011 files, 81 GB)
- Val: `/mnt/nvme/inat_mels/val_*.npy` (45,698 files, 30 GB)
- Total: ~111 GB / 214 GB capacity used. 103 GB headroom remaining.

**4. Survived two SSH wedges + two reboots.** Likely cause: HDD I/O contention against /var/log/wtmp during sshd auth (PAM auth-write blocks behind cp HDD reads → wedged auth subprocesses fill MaxStartups=10:30:100 slots → kex_exchange RST). Persistent journal now enabled so the next wedge is diagnosable. sshd has `Restart=on-failure` already.

**5. Caught the val-cache asymmetry bug.** The script's val DataLoader expects `val_*.npy` in `mel_cache_dir`; original plan only built `train_*.npy`. Val workers fell through to slow path (mp3 decode from HDD), deadlocked. Now both caches built; val reads NVMe.

**6. Updated CLAUDE.md with the deepthought storage layout + `/mnt/nvme` usage + swap-removal reversal.** Memory entry `feedback_always_clean_logs_before_dispatch.md` corrected — runon dispatches must clean BOTH local and remote log dirs.

### Don'ts on next pickup

- **Don't initiate heavy HDD I/O while expecting SSH to work in parallel.** That triggered the auth-wedge twice today. The pattern: `cp` random-reads on HDD + sshd auth tries to write to /var/log/wtmp on same HDD → auth slots wedge → MaxStartups exceeded → kex RST. If you must do another big HDD op, hold off SSH polling during it.
- **Don't trust the plan's "NVMe path" framing in older sections.** Use the table in `four_track/CLAUDE.md` § "⚠️ deepthought storage layout" as canonical.
- **Don't assume "ssh works → host is healthy."** `nc -zv` succeeding means TCP accept works; full SSH kex can still RST due to wedged auth. Use one batched ssh call per check, not 3 separate ones.
- **Don't bring back swap unless there's a memory-pressure incident.** Reversal destroys `/mnt/nvme` content (~3h to rebuild caches). Current 61 GB RAM has 57 GB free during training; no pressure signal.
- **Don't conflate iNat val_roc_auc with LB transfer.** v4 hitting 0.95+ on iNat val ≠ Channel-B 184-sp gate passing. The actual experiment is the downstream finetune.

### Next gate after v4 finishes

§14.22.10.5 step 6: Channel-B 184-sp focal AUC ≥ 0.9545 on the fold-0 finetune.
- If passes: LB probe submit (per the plan's strategic warning, val gains don't reliably predict LB — but mechanism-validated levers earn a Kaggle slot).
- If fails: the iNat lever class is exhausted at v1's ceiling. Pivot per the plan's "v4 ≤ 0.71 → switch corpus" branch (XC v3 already downloaded; head-fix; non-iNat external data).

### §15 A1 v4 5-fold push (2026-05-12) — STANDING CALIBRATION FLAG ON FOLD 3

Per-fold val_roc_auc from the v4 25-epoch sweep (hybrid_prodft loss, EffNet-B0 SED):
- Fold 0: 0.7425 (baseline reference)
- Fold 1: 0.7393
- Fold 2: 0.7710 (ep-12 single-epoch spike; stable band ep 13-25 sat at 0.71-0.75)
- Fold 3: 0.7472 (ep-24 late peak)
- Fold 4: 0.7480 (ep-24 late peak)

Raw 5-fold mean: 0.7496. Conservative (treat fold 2 as 0.74 stable): 0.7434. Three of five folds saved checkpoints from late-epoch spikes ~0.02-0.03 above their stable bands — suggests model-selection variance against the val split, not a uniformly improved ensemble. Whether that bias averages out on soundscape (LB) is unverified.

**JIT bundle and notebook now wired for 5 folds:**
- `four_track/kaggle_datasets/a1-effb0-ckpts/a1_fold{0,1,2,3,4}.pt` (16.9 MB each)
- `four_track/src/d2_beta_oof_cell.py`: `_D2B_A1_FOLDS = [0, 1, 2, 3, 4]` + docstring updated
- `jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb`: cell 34 `_D2B_A1_FOLDS` + cell 41 `A1_FOLDS` both updated to 5-fold

**Standing calibration flag on fold 3.** §A1-training-results-2026-04-07 dropped v3 fold 3 as a *calibration* outlier (not val-AUC outlier). v4 fold 3's val_roc_auc of 0.7472 looks healthy, but val AUC is rank-based and tells you nothing about sigmoid-distribution alignment with the other folds for rank-averaging. The v4 retrain used same architecture, same seed, same fold split — only longer training. The 2026-04-07 calibration cause may or may not have resolved.

**If LB regresses below recent baseline (~0.930) after pushing the 5-fold bundle, bisect against fold 3 first.** Concrete bisect protocol:
1. Push a notebook variant with `A1_FOLDS = [0, 1, 2, 4]` (drop fold 3) — re-run LB.
2. If LB recovers vs 5-fold, fold 3 calibration is the regression source. Either drop it permanently or run per-fold isotonic calibration on the train_soundscapes OOF pool before rank-averaging.
3. If LB still regressed with `[0, 1, 2, 4]`, the regression isn't fold-3-specific; look at the v4 retrain itself (compare against the v3 fold-0 ckpt preserved at `a1_..._prodft_pre_v4.pt`).

Cheap pre-submission alternative: local soundscape OOF on the 59 labeled train_soundscapes files, compute per-fold AUC + ECE, confirm fold 3 isn't a calibration outlier before spending the Kaggle slot. Implementation: standalone script that loads the 5 PyTorch ckpts (not JIT — JIT path strips probabilistic head detail), iterates `data/raw/train_soundscapes/*.ogg`, scores against the OOF y_true npz (`data/v56_soundscape_oof.npz` or equivalent).

#### §15.1 Timeout risk and 4-fold fallback (2026-05-13)

The Kaggle scoring kernel has a **~90-min runtime cap** for this competition (line 4525 notebook comment: "4-fold v71 timed out (~101 min, above ~90-min cap); 3-fold v72 fit budget"). v71 was V2-S 4-fold; current is B0 5-fold, which has a different cost profile but the 5th fold adds ~25% A1 inference time vs the recently-validated 4-fold B0 submissions (2026-05-04 at LB 0.930, which completed within the cap by an unknown margin).

**Risk:** the 5-fold v73 submission could time out if the 4-fold B0 baseline was already close to 90 min.

**Fallback prepared, not pushed (2026-05-13 ~00:30 EDT):**
- File: `jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb.fallback_4fold`
- Differs from live notebook only at cells 34 + 41: `A1_FOLDS = [0, 1, 2, 4]` and `_D2B_A1_FOLDS = [0, 1, 2, 4]` (drops fold 3 — the one with the standing calibration flag per §15)
- The 5-fold Kaggle dataset version doesn't need rolling back; the 4-fold notebook just ignores `a1_fold3.pt`.

**Deploy fallback (only if v73 times out — wastes a slot otherwise):**
```bash
cd /home/swatson/work/kaggle/BirdCLEF/jupyter/protossm-postproc/
cp birdclef2026-protossm-postproc.ipynb birdclef2026-protossm-postproc.ipynb.bak_5fold
cp birdclef2026-protossm-postproc.ipynb.fallback_4fold birdclef2026-protossm-postproc.ipynb
kaggle kernels push -p .
# Manually submit the new version's submission.csv via Kaggle UI or CLI
```

**Threshold for deciding to deploy:** if submission status stays PENDING past ~80 min elapsed (~05:11 UTC = ~01:11 EDT on 2026-05-13), timeout is likely; deploy fallback to preserve the day's remaining slot for a viable submission. Earlier than that is speculative — most 4-fold B0 submissions completed in <90 min, so the 5-fold could still fit.

**Post-mortem actions (regardless of v73 outcome):**
- If v73 COMPLETES (regardless of LB): record actual runtime to bound the 5-fold B0 cost. That number permanently informs whether 5-fold is viable.
- If v73 TIMES OUT: the 5-fold A1 path is structurally untenable for this competition's submission cap. Either (a) drop a fold permanently, (b) reduce per-fold inference cost (e.g., fewer TTA windows in cell 37, lower N_WINDOWS), or (c) move A1 to a heavier-throughput fold reduction (e.g., min instead of rank-mean) that costs less per fold.

#### §15.1 outcome (2026-05-13 ~01:03 EDT)

**v73 submission COMPLETED. Public LB: 0.929.**

| Submission | Folds | LB | Notes |
|---|---|---|---|
| v73 (this push) | 5 (incl. fold 3) | **0.929** | submitted 03:51 UTC, completed within ~72 min |
| 2026-05-04 baseline | 4 | 0.930 | recent reference point |
| all-time best | ? | 0.933 | 2026-04-08 |

**Key findings:**

1. **No timeout.** 5-fold B0 fit within the 90-min cap (elapsed ~72 min). The 4-fold fallback was *not* needed and remains staged at `.ipynb.fallback_4fold` for future use. The 5-fold A1 path is structurally viable timing-wise.

2. **−0.001 vs 4-fold baseline.** Within the ±0.005 single-submission noise band; not a meaningful regression, but also not the val-implied +0.001-0.003 LB gain. Net: no improvement.

3. **Val improvement did NOT transfer to LB.** Raw val mean was +0.007 over baseline (0.7496 vs 0.7425); LB delta is −0.001. This is consistent with the model-selection-bias hypothesis from §15: three of five folds saved late-epoch spike ckpts ~0.02-0.03 above their stable bands, and the ensemble didn't smooth those spikes into real soundscape-domain gain. The val→LB gap stayed at the historically-observed magnitude.

4. **The fold-3 calibration question is unresolved but low-priority to chase.** The −0.001 delta is too small to distinguish fold-3 calibration noise from general sampling variance. A bisect (drop fold 3, resubmit) would land in 0.928-0.932 noise; would need 3-4 paired runs to get signal. Not worth the slot budget given the four-track plan's bigger-gain paths (B/C/D).

**Direction signal:** Track A polish appears to be at diminishing returns at the +0.026 distance to LB #1 (0.959). v73 didn't move LB. The four-track plan's emphasis on Tracks B/C/D — which target step-function gains, not Track-A tuning — is validated by this result. Next slot should fund a Track-B/C/D probe, not another A1 variant.

**Fallback file kept** (`.fallback_4fold`) as a quick rollback if a future Track-A regression appears; not consuming any compute.

#### §15.2 Per-fold Channel A/B probe (2026-05-13 ~01:15 EDT) — corrects §15 fold-3 flag

Ran `rescore_baseline_v2.py`-style probe on all 5 v4 fold JITs against `data/processed/val_v2/{val_v2_soundscape,val_v2_focal}.npz`:

| Fold | Channel A (primary, 2371 sp×123) | Channel B (focal, 184×184) |
|---|---|---|
| 0 | **0.8441** | 0.9606 |
| 1 | 0.8497 | 0.9845 |
| 2 | 0.8613 | 0.9771 |
| 3 | 0.8511 | 0.9782 |
| 4 | 0.8583 | 0.9795 |
| **mean** | **0.8529** | **0.9760** |
| **v56 baseline** | 0.8521 | 0.9545 |
| **delta** | **+0.0008** | **+0.0215** |

**Corrections to prior reasoning:**

1. **The fold-0-only Channel A drop (−0.008 vs v56) is fold-0-specific, not v4-wide.** At the ensemble-mean level Channel A is tied with v56 (+0.0008, well within noise). Channel B is +0.0215 — the iNat lever did deliver as §14.22 predicted. My earlier "iNat lever regressed Channel A → switch corpus" recommendation was based on fold 0 alone and was wrong.

2. **The standing calibration flag should point at fold 0, not fold 3.** Fold 3 looks healthy on both channels (0.8511 / 0.9782, comparable to folds 1/2/4). The §15 prior-history pointer (v3 fold 3 calibration outlier) does not apply to v4 — different training history, different metrics. **The actual weak link is fold 0.**

3. **The §15.1 fallback notebook drops the wrong fold.** It currently drops fold 3 (the pre-existing flag). The data say drop fold 0 instead.

**Unresolved: why LB regressed despite ensemble Channel A being tied and Channel B being up.**

Three live hypotheses (not yet probed):
- Rank-CDF fusion in cell 37 doesn't aggregate per-fold AUC linearly; ensemble val AUC may diverge from per-fold mean
- val_v2 isn't predictive of LB at the 0.001 resolution we're trying to measure
- Fold 0 drag is real and amplified by the rank-CDF fusion; weakest-fold pull disproportionately on its weakest species

**Cheap next probe:** compute rank-averaged ensemble AUC on val_v2 for both 5-fold and 4-fold-without-fold-0 configs. If 4-fold-w/o-0 beats 5-fold by >0.003 on Channel A, that justifies a fold-0-drop re-submission. ~5 min compute, no Kaggle slot.

**Action items from this section:**
- Update fallback notebook to drop fold 0 instead of fold 3 (one-line edit in the `.fallback_4fold` file).
- Run rank-average ensemble probe before any next LB push.

#### §15.3 Rank-average ensemble probe (2026-05-13 ~01:25 EDT) — corrects §15.2 fold-0 framing

Mirrored cell-37 rank-CDF fusion across all single-fold-drop variants:

| Config | Channel A rank | Channel B rank |
|---|---|---|
| 5-fold (all) | **0.8789** | 0.9867 |
| 4-fold drop f0 | **0.8789** (tied) | 0.9882 (+0.0015 vs 5-fold) |
| 4-fold drop f1 | 0.8779 | 0.9840 |
| 4-fold drop f2 | **0.8733 (−0.006)** | 0.9853 |
| 4-fold drop f3 | 0.8777 | 0.9867 |
| 4-fold drop f4 | 0.8753 | 0.9860 |
| v56 single-fold ref | 0.8521 | 0.9545 |

**Corrections to §15.2:**

1. **Dropping fold 0 produces no measurable Channel A change.** 5-fold and drop-f0 are tied at 0.8789 on rank-fusion. Fold 0's per-fold weakness (0.8441 alone) is absorbed by the rank-averaging. The §15.2 "fold 0 is the weak link, drop it" framing was wrong about LB-relevance — fold 0 doesn't *cost* the ensemble anything on Channel A.

2. **Fold 2 (the 0.7710 ep-12 spike) is actually doing the most ensemble work on Channel A.** Drop-f2 is the worst 4-fold variant (−0.006 on Channel A). My §15 framing that "spike-saved ckpts didn't generalize" was wrong — fold 2's spike captured a real signal that the other folds didn't.

3. **The actionable signal is small: drop-f0 gains +0.0015 on Channel B rank-AUC.** Too small to be visible on LB. No slot warranted for resubmission.

**The real constraint surfaced by this probe:**

**val_v2 is not predictive of LB at the ±0.005 resolution.** The v4 5-fold rank-ensemble is **+0.027 above v56 single-fold on Channel A** (0.8789 vs 0.8521). That's a huge val gain. LB delta from v73 vs prior baseline: **−0.001**. The mapping val→LB at this resolution is essentially zero.

Implication: **any further val-driven A1 iteration will reproduce the same val/LB divergence.** The plan's val_v2 ≥ 0.7414 / Channel B ≥ 0.9545 gates motivated the v73 push, but those gates don't actually forecast LB movement.

**Permanent direction change for Track A:**

- **Don't burn slots on val-improvement-driven A1 resubmissions.** The val→LB transfer at the small-delta resolution we're operating in (within 0.005 LB) is below the threshold val can resolve.
- **Need a different gate.** The 59 fully-labeled `train_soundscapes/` files (already locally available, mirrored from competition data) are the closest local proxy to LB. Build an A1-only OOF AUC on this pool as the new gate. Comparison: rank-averaged A1 OOF on the 59-file pool, both for v4 5-fold and v56 single-fold baseline, would tell us if v4 has a real soundscape-domain edge.
- **For Tracks B/C/D**, the existing OOF protocol on the 59-file pool (cell 31/31b) is the right gate — don't repeat the val_v2 mistake.

**Bottom-line action items revised:**
- Track A polish on val is done. Don't iterate further on val_v2 signal alone.
- Build local soundscape OOF probe (the 59-file pool) for future Track A submissions.
- Pick up Track B1 / C revival / Bundle 3 with soundscape OOF as the gate, not val_v2.

---

## 📌 PICK UP HERE (2026-05-13 ~01:35 EDT, sleep handoff)

### Background job running at handoff

| Field | Value |
|---|---|
| What | v4 5-fold A1 soundscape OOF on the 1478-window labeled pool |
| Script | `/tmp/v4_5fold_soundscape_oof.py` (modified copy of `p12_emit_oof.py`) |
| PID (skynet) | **281824** |
| Log | `four_track/log/v4_5fold_soundscape_oof_20260513_*.log` |
| Expected runtime | ~10-15 min (audio load + mel build is the slow part) |
| Saves to | `four_track/data/v4_5fold_soundscape_oof.npz` |
| What it prints | per-fold AUC, ensemble mean AUC, ensemble rank AUC, **v4-vs-v56 deltas** (both mean and rank fusion) |
| v56 reference (from inline compare) | `data/v56_soundscape_oof.npz` (already present, 4-fold [0,1,2,4]) |

**On next session pickup:**
```bash
# Status:
ps -p 281824 2>/dev/null && echo RUNNING || echo DONE
tail -40 /home/swatson/work/kaggle/BirdCLEF/four_track/log/v4_5fold_soundscape_oof_*.log
ls -la /home/swatson/work/kaggle/BirdCLEF/four_track/data/v4_5fold_soundscape_oof.npz
```

### What we accomplished this session

1. **A1 v4 5-fold full pipeline:** train → JIT → Kaggle dataset push → notebook v73 push → LB submission. Final result: **LB 0.929** (−0.001 vs 4-fold baseline 0.930).
2. **Val→LB divergence confirmed empirically.** v4 ensemble val Channel A rank-AUC was **+0.027 above v56** but LB moved **−0.001**. Plan §15.3 documents the permanent direction change.
3. **Three retractions on Track A reasoning recorded as memories:**
   - `feedback_per_fold_val_misleads_ensemble.md` — rank-fusion absorbs weak folds; per-fold AUC misleads about fold-drop decisions
   - `feedback_val_v2_not_predictive_at_small_LB_deltas.md` — use soundscape OOF as the gate, not val_v2
4. **4-fold fallback notebook prepared but not used** (5-fold fit within 90-min cap). Kept at `jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb.fallback_4fold`. Note: this drops fold 3, but §15.2/§15.3 show that's the wrong fold to drop — if we ever do fall back, drop fold 0 instead (per §15.2 per-fold table). Or just drop the fallback entirely; v73 fit fine.
5. **iNat lever assessment final:** Channel B per-species discrimination genuinely improved (+0.021 ensemble), but the soundscape primary gate is tied (+0.0008 ensemble Channel A). LB unmoved. Lever is "real but invisible at LB resolution."

### What the soundscape OOF probe answers (pending tomorrow)

The probe prints **v4-vs-v56 rank-AUC delta** on the 1478-window labeled soundscape pool. Three possible outcomes and what each implies:

- **v4 rank-AUC > v56 rank-AUC by ≥ 0.003:** real soundscape-domain win that didn't show on LB. Suggests LB itself is noisy or the v73 submission had an unrelated regression. Worth one more Kaggle slot probing variations (drop f0, different fusion).
- **v4 rank-AUC ≈ v56 rank-AUC (within ±0.002):** confirms LB result was honest. v4 = v56 at soundscape resolution despite val gains. Conclusion: iNat lever is exhausted at this LB level. Move to a new Track.
- **v4 rank-AUC < v56 rank-AUC:** v4 has a soundscape regression that val didn't surface. The +0.027 val gain is on the wrong distribution. Stronger version of "iNat lever didn't transfer" — argues for corpus switch or different finetune protocol.

### Concrete next actions (in priority order)

1. **Read the probe output** (tail the log; npz at the saved path). Decide which of the three outcomes above we're in.
2. **If outcome confirms v4 ≈ v56:** pick a new Track. Options ranked by readiness:
   - **Track B1 PerceiverIO** — `four_track/src/train_b1_local.py` exists; prior B1 LB result in plan §B1 LB results (line 238). Re-check current state before dispatching. Use the soundscape OOF pool (1478 windows) as the new gate.
   - **Track C revival §13** — phased design (Phase 1 Kaggle Perch extraction → Phase 4 LB probe). Bigger infra build but higher lift potential. Cell 31b OOF protocol already exists.
   - **Bundle 3 / XC v3** — already downloaded per §14. Different external corpus than iNat. New finetune cycle, full 5-fold ckpts, then back to soundscape OOF gate.
3. **If outcome shows v4 > v56 by ≥0.003:** worth one slot probing per-fold drops (drop f0 has marginal Channel B gain per §15.3; might be visible on soundscape OOF that wasn't visible on val_v2).
4. **HPO §14.23 trigger:** still not cleanly met. Don't pre-launch unless either (a) Track B/C blocked or (b) probe shows v4 is genuinely strong and just needs hyperparameter tuning to push further.

### State of files on disk

- **5 v4 JIT ckpts** at `four_track/kaggle_datasets/a1-effb0-ckpts/a1_fold{0..4}.pt` (16.9 MB each)
- **5 v4 PyTorch ckpts** at `four_track/models/a1/a1_..._fold{0..4}_seed42_hybrid_prodft.pt`
- **Pre-v4 fold 0** preserved at `a1_..._fold0_seed42_hybrid_prodft_pre_v4.pt`
- **Suffixed v4 fold 0 snapshot** at `a1_..._fold0_seed42_hybrid_prodft_25ep_v4_val0.7425.pt`
- **Kaggle dataset:** `stevewatson999/birdclef-2026-a1-effb0-ckpts` — version pushed with 5 ckpts
- **Kaggle kernel:** `stevewatson999/birdclef-2026-protossm` — version 73 with `A1_FOLDS=[0,1,2,3,4]`
- **Live notebook:** 5-fold (canonical). Fallback at `.ipynb.fallback_4fold` (drops fold 3 — wrong fold per §15.2/§15.3, but harmless if not deployed).

### Don'ts on next pickup

- **Don't submit another A1 variant to LB based on val_v2 alone.** Established this session that val_v2 ≠ LB at our resolution. Use soundscape OOF first.
- **Don't trust per-fold val AUC for fold-drop decisions.** §15.3 showed rank-fusion absorbs weak folds. Run the probe.
- **Don't push to Kaggle the .fallback_4fold notebook as-is.** It drops the wrong fold per §15.2.
- **Don't re-run the iNat pretrain.** Channel A regression at fold 0 is a Channel-A-vs-Channel-B asymmetry, not a Channel-A-wide problem. v4 is fine; iNat lever delivered what it could.

---

## 🔴 §16 Honest assessment — why nothing has moved LB above 0.933 (DISCUSS TOMORROW)

**Written 2026-05-13 ~01:40 EDT after v73 landed −0.001. User-flagged emotional state: exhausted frustration after many attempts hitting the same 0.925-0.931 noise band.** Saving for tomorrow's discussion when the emotion has cooled. Question to revisit at start of discussion: is the right response to push harder on the same approaches, or step back and audit what the plan has actually been optimizing?

### The observation that prompted this section

- LB all-time best: **0.933** (2026-04-08, one submission, never matched since)
- Recent cluster: **0.925-0.931** for all v50+ variants
- v73 today: **0.929**
- LB #1: **0.959** (gap of **+0.026** above the plateau)

Every variant in the last month has landed in the same ±0.005 band. That's not bad luck; it's a real plateau.

### Five diagnostic reasons nothing has moved LB

**1. The plan has been optimizing val, and val isn't predictive of LB at the deltas we operate in.**
Empirically confirmed in this session (§15.3): v4 ensemble Channel A rank-AUC is +0.027 above v56; LB moved −0.001. Many earlier "KILL at LB 0.925 vs 0.931" decisions killed variants within ±0.005 noise of each other. Many "WIN at small val delta" decisions were probably also noise. The plan's gate structure assumed val→LB transfer that doesn't exist at our resolution.

**2. Everything tried is a perturbation of the same recipe.**
All attempts: variations of "EfficientNet-family SED + ProtoSSM/B1 head + various aux gadgets on Aves-skewed focal data." Backbone variation (B0/V2-S/NFNet), pretraining variation (ImageNet/iNat v1-v4), loss variation (BCE/ASL/hybrid_prodft), fusion variation (mean/min/rank-CDF). All in the same noise band because they share the same inductive bias on the same training distribution. **0.026 gap to the leader is not closeable with this recipe family.** Top teams are doing something structurally different.

**3. Tracks B/C/D were designed for step-function gains but were only half-built.**
- Track B1: OOF gate failures, partially shelved
- Track C2: killed structural in April; §13 revival has Phase 1-4 design but Phase 1 (Kaggle Perch extraction) was never executed
- Track D stackers: killed at LB 0.925 vs 0.931, but stackers only help if underlying components are diverse — yours weren't
- The plan describes a four-track strategy; in practice ~90% of compute went to Track A polish.

**4. Single-submission noise is ±0.005.**
Even a real +0.003 improvement is invisible. Can't see signal smaller than the noise floor regardless of slot count.

**5. iNat pretraining DID work as designed — it hit a domain mismatch wall.**
Channel B (focal, 184 sp) ensemble gained +0.0215. Real signal. But LB scores soundscape, not focal — exactly where iNat's pretrain distribution stops helping. The §14.22 hypothesis ("iNat helps non-Aves") was correct, but the help is on Channel B, not on the LB-correlated Channel A.

### The honest diagnosis

The plan describes a four-track strategy but **executed a one-track strategy**. Track A is mature — everything available from this recipe has been harvested. **0.929-0.931 is the ceiling of this architecture × data × pretrain combination on this LB.** The plateau isn't bad luck. It's the recipe's ceiling.

### Two legitimate directions to discuss tomorrow

**Direction A — commit to a real Track B/C/D push.**
Pick one and give it a full week of compute with soundscape OOF as the gate from day one:
- Track B PerceiverIO retrain with proper OOF discipline (didn't have it before)
- Track C revival §13 — execute Phase 1 (Kaggle Perch extraction) for real, then Phase 2-4
- Track D requires component diversity first — likely not viable until B or C produces a genuinely different model
- Bundle 3 / XC v3 — different external corpus than iNat, might hit a different transfer pattern

**Direction B — accept the plateau, aim for top-N finish rather than the leader.**
0.929-0.931 with a clean submission package is a legitimate competition result. Stop burning slots chasing 0.005 deltas that may not be real. Focus remaining time on:
- Confirming the strongest single submission is the one chosen at deadline
- Per-class calibration cleanup
- Ensemble of existing diverse-enough variants (v56 + v4 stacked, B1 if it produces real OOF improvement)

**The wrong move** is to keep submitting Track-A variants hoping noise breaks favorable. The session-long pattern of optimistic projections (+0.003 expected → −0.001 actual) is the noise itself.

### Question to discuss tomorrow

Given the documented one-track-vs-four-track gap, the empirical val→LB blindness, and the 0.026 gap to the leader: **what is the realistic target for this competition — top-50, top-100, or "as high as the current recipe ceiling allows"?** The answer determines whether Direction A or Direction B is correct. There is no good answer that says "keep iterating Track A at small deltas."

### Inputs for the discussion (already produced)

- §15.1 outcome: v73 = LB 0.929, no timeout, fit fine
- §15.2: per-fold Channel A/B profiles for v4 (fold 0 weakest, but fold 2 spike contributes most to ensemble)
- §15.3: rank-fusion ensemble probe — no fold-drop helps materially at LB resolution
- Memory entries: `feedback_per_fold_val_misleads_ensemble.md`, `feedback_val_v2_not_predictive_at_small_LB_deltas.md`
- Pending: v4 5-fold soundscape OOF probe (running at handoff, expected ~01:50 EDT) — will give the v4-vs-v56 delta on the LB-correlated 1478-window pool, the first metric we'll have that actually correlates with LB. **Read this number before the discussion.**

---

## §17 Top-team recipe pivot — switch from §16 named tracks to "what 2025 winners actually did" (2026-05-13 ~morning)

### §17.0 Why this section exists

§16 named three candidate tracks (B1 / Track C revival §13 / Bundle 3-XC v3) for a structural pivot. Before committing compute to any of them, ran a leaderboard-discussion search for what BirdCLEF 2025 top finishers actually used. Findings forced a re-prioritization: **none of the three named tracks matches the winning recipe**. The winning recipe is essentially our current EffNet-SED architecture plus three specific techniques we never executed.

### §17.1 What top 2025 finishers actually did (Pantanal-2026 has no writeups yet — 2025-Magdalena is the closest analog)

**1st place (Nikita Babych)** — title is literal: "**Multi-Iterative Noisy Student Is All You Need**." Iterated self-distillation on unlabeled soundscapes was *the* headline technique.

**2nd place (VSydorskyy)** — concrete recipe from public GitHub repo:
- Backbones: `tf_efficientnetv2_s_in21k` + `eca_nfnet_l0`
- Pretrain: ImageNet-21k → additional pretrain on Xeno-Canto + iNaturalist + CSA
- Loss: FocalBCELoss
- Pseudo-label protocol: **3 iterations**, confidence 0.5, multi-label 0.1, prob_min 0.4
- Ensemble: 5 first folds
- **Explicitly no Perch, no BirdNET**

**Top 2% (Max Melichov, EffNet-B0 — closest to our setup)** — documents per-technique val-AUC gains:

| Technique | Val-AUC delta credited |
|---|---|
| BirdCLEF 2021-2024 historical pretrain | +0.013 |
| Pseudo-labeling middle-5s on unlabeled soundscapes | +0.018 |
| **Quantile-Mix postprocessing (α=0.5 mean + rank blend)** | **+0.025** |

**Final 2025 LB**: 1st place ~0.93 (3% ahead of #38). Compressed top band.

### §17.2 The three techniques top teams used that we haven't

1. **Quantile-Mix postprocessing** — α=0.5 blend of mean fusion + rank-CDF fusion across folds. We do rank-only. (`+0.025 val AUC` in Melichov writeup.)
2. **Multi-iteration noisy student self-distillation** — *our own* current best 5-fold model becomes the teacher, generates pseudo-labels for unlabeled train_soundscapes, retrain, iterate 2-3 rounds. (`+0.018 val AUC` in Melichov writeup; 1st place headline technique.)
3. **BirdCLEF 2021-2024 historical-year pretrain** — combine prior-year competition focal data, pretrain backbone, then finetune on 2026 data. (`+0.013 val AUC` in Melichov writeup.)

If they stack additively at 50% transfer efficiency: 0.5 × (0.025 + 0.018 + 0.013) = **+0.028 LB**. That meets the +0.03 target framed by user this morning.

### §17.3 What this kills from §16

- **Track C revival §13 as written is wrong.** §13 is "Perch-as-teacher pseudo-labels." Top teams (1st, 2nd, 5th place all reference self-distillation in their titles) use *their own model* as teacher, not Perch. Track C needs to be **rewritten as self-distillation** before any compute, OR replaced entirely by §17.2.
- **Track B1 PerceiverIO is not a winner recipe.** No top-5 team used a different head architecture; they all used SED/CNN on mel-spectrograms.
- **Bundle 3 XC v3 is plausible but not in the top-5 picture.** External corpus extension didn't appear as a headline contributor in any of the four writeups we located. Deprioritized.

### §17.4 Caveats — read before assuming the deltas transfer

1. **2026 is Pantanal (Brazil), 2025 was Magdalena Valley (Colombia).** Different distributions, different species sets, different soundscape acoustics. 2025 deltas are evidence, not guarantees.
2. **The +0.013 / +0.018 / +0.025 are VAL deltas**, from a single team's blog post. This session established (`feedback_val_v2_not_predictive_at_small_LB_deltas.md`) that val→LB transfer is broken at ±0.005. The 0.5× efficiency factor above is a guess, not a measurement.
3. **We previously killed EffNetV2-S based on val_v2** (§14.19). 2nd place uses V2-S. Given the val_v2→LB blindness, that kill may have been a false negative. Worth revisiting V2-S after Step 1-2 land, before committing to Step 3.
4. **Per `feedback_verify_plan_numeric_claims.md`** (memory entry from `BirdCLEF/plan.md` "+0.04 → +0.003" lesson): treat all three deltas in §17.2 as upper bounds, not estimates. Verify with locally-measurable metric (soundscape OOF on the 1478-window pool, even though we know that gate is imperfect, is still the best gate available before spending a Kaggle slot) before submitting.

### §17.5 Execution order

**Step 1 — Quantile-Mix postprocessing — RETRACTED 2026-05-13 (~10 min after §17 was written)**

When §17 was first drafted, I missed `§14.8.L3-probe RESULT` (line 2482) which already documented:
- Quantile-Mix at α=0.5 implemented in the live notebook 2026-04-17 as `A1_QMIX_ALPHA`.
- Kernel v51 pushed, **LB 0.925 vs baseline 0.931 → −0.006 LB regression**.
- Reverted to `A1_QMIX_ALPHA = 0.0` same day; left in place as documented dead lever.
- Interpretation in §14.8.L3 (still valid): the prob-space blend introduces scale mismatch between A1 sigmoids and the ProtoSSM logits that the rank-space cascade absorbs but the prob-space side cannot. Writeup-cited α=0.5 is the canonical lift point; α=0.25 sweep was not justified and not run.

Verification done in this session before retracting:
- OOF probe on `data/v4_5fold_soundscape_oof.npz` showed pure mean fold-fusion AUC 0.7775 vs current rank-mean 0.7672. +0.010 OOF available from a different change (A1's internal fold summary direction, not Quantile-Mix).
- That adjacent change is not Quantile-Mix; it's a 1-line swap of `_a1_ranks = mean(rank01(p_i))` → `_a1_ranks = rank01(mean(p_i))`, AUC-invariant to plain mean, slots into existing rank-space cascade unchanged.
- Worth bundling as a free A/B on Step 2's submission, NOT worth a standalone LB slot given (a) L3 precedent that fusion changes in this region cost LB, (b) v4 OOF +0.024 → LB −0.001 from this same session — OOF→LB transfer is unreliable.

**Step 1 is now: ride the fold-fusion-direction A/B on Step 2's submission. No separate slot.** Saved as `feedback_check_plan_before_recommending.md` so the search-for-prior-attempts step is run before future "new technique" recommendations.

**Step 2 — Multi-iteration noisy student — NEW STARTING POINT** (was Step 2 in original §17.5)
- Use current v4 5-fold A1 ensemble as teacher.
- Generate pseudo-labels for unlabeled train_soundscapes using 2nd-place thresholds as starting point (confidence 0.5, multi-label 0.1, prob_min 0.4) — soft labels.
- Mid-5s segment policy per Melichov writeup.
- Retrain from current best checkpoint with labeled + pseudo-labeled data. Iterate 2 rounds first; only go to 3 if rounds 1→2 show clear improvement.
- Gate per iteration: soundscape OOF on the 1478-window pool, then one LB slot at iter-2 if OOF improves.

**Step 3 — BirdCLEF 2021-2024 historical pretrain** (gated on Step 2 success)
- Verify Kaggle dataset availability + license for combined years before downloading.
- Run as background pretrain on deepthought once dataset is in place.
- Stack with Step 2 best (replace ImageNet-21k init with BirdCLEF-historical-21-24 init, then run noisy-student finetune protocol on top).
- This is the longest runway item. Don't start unless Step 1+2 deliver clear signal that the recipe is alive.

### §17.6 Don'ts on this pivot

- **Don't use val_v2 alone to kill any of these.** Same blindness applies as in §15.3.
- **Don't open Track C as currently written (Perch-as-teacher).** Rewrite as self-distillation or skip.
- **Don't pretrain on iNat again.** That was §14.22, killed Channel A but helped Channel B; not the same as historical-year focal pretrain. Different mechanism.
- **Don't ensemble Step 1 vs Step 2 vs Step 3 by "diversity score" alone.** §16 noted Track D stackers failed because components weren't diverse. These three steps are sequential improvements on the same model, not a diverse ensemble.

### §17.7 What §17 doesn't promise

The +0.028 estimate is `0.5 × sum_of_three_val_AUC_gains_from_one_blog_post`. It's a back-of-envelope for sizing, not a forecast. If Step 1 lands −0.001 (the v4 outcome from this session), then Steps 2-3 are still worth running; if Step 1 also lands ±0.005 noise like v50+ variants, that's a strong signal that the val→LB blindness has gotten worse than even soundscape OOF can detect, and §16 Direction B (accept plateau) becomes the correct response.

### §17.7b ALL THREE STEPS HAVE PRIOR KILL RECORDS — full retraction of §17 as a near-term action plan (2026-05-13, written ~15 min after §17 itself)

After retracting Step 1 in §17.5, ran the same plan-grep on Steps 2 and 3 before dispatching anything. The result wipes out §17 as written:

| §17.5 Step | Plan history finding |
|---|---|
| **Step 1 — Quantile-Mix postprocessing** | KILLED as **L3** 2026-04-17. LB 0.925 vs 0.931 baseline (−0.006). |
| **Step 2 — Multi-iteration noisy student self-distillation** | Pseudo-label family has been killed 11+ consecutive times on this stack: §13 ProtoSSM→ProtoSSM self-distill (KILLED), **L1** ProtoSSM→A1 cross-arch (KILLED 2026-04-18 LB 0.930, val showed +0.163 but LB neutral — val-leakage landmine), **A2** ProtoSSM-self-train on soundscapes (KILLED LB 0.926, −0.007), **A2-iter-1** (val-fail), **A2-iter-1.5** (pseudo-cap hypothesis disproved §14.17.11). Plan §14.17.5 was explicitly titled "re-attack iterative noisy student, **done correctly**" — that version also died. |
| **Step 3 — BirdCLEF 2021-2024 historical pretrain** | KILLED as **L2** 2026-04-18 (§14.10.10 RESULT). Same Melichov writeup cited there (+0.009 expected) didn't transfer. **L2-redux** exists in §14.20+ as a non-Aves-included variant that addresses the species-coverage gap of the killed L2 — status partially advanced but not landed at LB. |

**The category error in §17 as originally written:** I treated "what BirdCLEF 2025 winners did" as a proxy for "what this stack hasn't tried." False. The plan documents that all three winner-recipe techniques (Quantile-Mix, noisy student, multi-year pretrain) **were imported into this stack and killed** between mid-April and end-April 2026. §16 wasn't a temporary frustrated read — it was the empirical summary of those kills.

**Implications for the +0.03 target asked this morning:**
- Each of the three techniques in isolation already failed to clear baseline. The "0.5× stacking → +0.028 LB" arithmetic in §17.2 assumed the components had at least small positive lift; the plan's record shows they have **negative or neutral lift on this stack**.
- The 2025 writeup deltas reflect the gains those techniques provided *against weaker baselines on different data*. Starting from 0.929-0.931 on Pantanal-2026 with this exact recipe, the marginal lift available from these levers is empirically near-zero.
- This sharpens §16 to a stronger claim: not just "the recipe ceiling is real" but "**every standard winner-recipe lever has already been pulled and produced ±0.005 LB noise on this stack**." +0.03 is not closeable by techniques in the same recipe family.

**What might still be alive (paths to verify before committing compute):**

1. **L2-redux (§14.20+)** — multi-year pretrain WITH non-Aves coverage added (the killed L2 was Aves-only). Different from the §17.3 Step 3 because it addresses a specific known failure of L2. Need to read §14.20+ carefully to find current status and what an LB probe would cost.
2. **A *specific* noisy-student variant not yet tried.** Need to inventory which protocol-knobs were varied across the 5+ kills (thresholds, ensemble teacher vs single, focal vs soundscape pseudo-pool, hard vs soft labels, mass augmentation policy, calibration). The 2nd-place 2025 recipe (confidence 0.5, multi-label 0.1, prob_min 0.4, 3 iterations on *soundscape* pseudos using own model as teacher) may not be bit-identical to any prior attempt — but the burden of proof is now on me to show that, not assume it.
3. **Outside the §17 family entirely:** the three §16 options (B1 / Track C revival / Bundle 3 XC v3) are still not in the kill family for the *same reasons* (different architectures / different external data). They're still bets, but they're bets the plan hasn't already wagered on with cleaner kill records.

**Honest re-recommendation:** none of the §17 steps should start without first reading the specific kill memos to find a genuine protocol gap. The user-stated +0.03 target is, per the plan's own record, not achievable by anything in §17 as written, and probably not achievable by any small variant either. §16 Direction A (commit to a structurally-different track for a week) vs Direction B (plateau-management for top-N) is the actual decision, and §17 didn't change that.

**Action at the moment of this retraction:** all three §17 steps marked blocked. Tasks #3 (Step 2) and #4 (Step 3) updated with the kill citations. No compute spent. Waiting for user direction.

### §17.8 Sources

- [BirdCLEF 2025 1st place writeup (Nikita Babych)](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n)
- [BirdCLEF 2025 2nd place GitHub (VSydorskyy)](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place)
- [BirdCLEF 2025 5th place writeup: Self-Distillation](https://www.kaggle.com/competitions/birdclef-2025/writeups/noir-5th-place-solution-self-distillation-is-all-y)
- [Max Melichov top-2% writeup with per-technique deltas](https://medium.com/@maxme006/how-i-climbed-to-the-top-2-in-birdclef-2025-every-failure-every-lesson-and-why-details-matter-273d781a33df)
- [Kaggle Solution Walkthrough video — Nikita Babych](https://www.youtube.com/watch?v=jivW1JBxV8s)


---

## §18 L2-redux Aves-corpus-at-scale — audit and decision framing (2026-05-13 ~morning, after §17 retraction)

### §18.0 Why this section exists

After §17 was fully retracted as "everything has prior kill records," surveyed the plan for any lever with (a) a measured positive gate, (b) no LB kill, (c) a plausible path to LB. L2-redux Aves-corpus-at-scale variant meets all three. It's the only such lever in the plan as of 2026-05-13.

### §18.1 Audit findings

| Phase | What | Status |
|---|---|---|
| Phase 2b smoke (§14.17.15.8) | 70K-clip / 636-species BC2023+24+25 Aves pretrain (5 ep) → frozen linear probe vs ImageNet baseline on BC2026 fold-0 val_v2 | **PASSED.** Δ probe = +0.0554 ≥ +0.01 gate, cleared by 5.5×. Plan classed as "strong-transfer territory." |
| Phase 4 at scale (intended ~800K-clip / 7591-species Aves corpus) | Scale up the smoke-validated recipe to the full Sydorskyi corpus | **NEVER EXECUTED end-to-end to LB on Aves corpus.** Plan branched here. |
| Branch A — XC v3 corpus on B0 ("L2-redux v1") | rohanrao bulk pivot (§14.17.16.7) → 30 GB download → transcode → was once "the primary remaining lever" (line 9991) | Download landed; Phase 4 pretrain on this corpus never landed an LB. Status: displaced by Branch B. |
| Branch B — iNat 2024 Sounds pretrain (§14.22) | Different corpus (natural taxonomy, includes non-Aves) intended to fix L2's species-coverage gap | iNat pretrain v1-v4 ran. v4 → A1 5-fold finetune → **LB 0.929 = neutral** (this session, 2026-05-13). Branch B is the lever just measured as exhausted. |

### §18.2 The measured prediction

The plan's only delta-prediction grounded in measured encoder-quality data (not external-writeup citation) is at §14.17.15.8.3:

> "Realistic envelope: **Phase 6 LB delta ~+0.015 to +0.030**."

This is the compressive transfer from a +0.0554 frozen-probe Δ to a fine-tuned LB Δ, applying the 30-50% transfer efficiency from encoder-pretraining literature. Range overlaps the user's +0.03 target at the upper bound, lands at "+0.015 = decisive break from noise band" at the lower bound.

### §18.3 Caveats — read before committing 6 days of compute

1. **Smoke corpus ≠ scale corpus.** The +0.055 was on 70K / 636 species. Scaling to 800K / 7591 *should* improve transfer, but it's unmeasured. Possible the smoke result already captured most of the available signal.
2. **Aves-only coverage gap.** This is the same gap that killed original L2 ("L2-redux must include the [non-Aves]" — §14.17.14 line 8383). The Branch B iNat lever was supposed to fix this and didn't move LB. Aves-corpus-at-scale may regress non-Aves classes at LB even if total AUC improves.
3. **Branch B kill weakens the prior.** iNat v4 → LB 0.929 is a data point against the broader pretrain class. The two corpora are different (Aves-only vs non-Aves-included; XC-style vs iNat-style; geographically biased differently), but the mechanism (pretrain → encoder Δ → LB Δ) is the same. If the mechanism fails on iNat, Aves-corpus may also fail by the same mechanism. Then again, iNat's failure may be iNat-specific (different recording style, different signal density).
4. **No documented kill of Aves-corpus-at-scale specifically.** This is the genuine protocol gap, but absence of kill ≠ presence of lift.
5. **Compute cost.** Phase 4 pretrain at 800K/7591 ≈ ~2 days on deepthought. Phase 5 (finetune 5 folds) ≈ ~3-4 days. Phase 6 = 1 Kaggle slot. **Total ~6 days minimum to know the LB answer.**

### §18.4 Honest read

The case for L2-redux Aves-corpus-at-scale is *the strongest "alive" lever in the plan*, but the strength is modest:
- **For it:** Phase 2b passed a real measurable gate. The +0.015 to +0.030 prediction is plan-grounded, not external-writeup arithmetic. It's the only lever with this property as of 2026-05-13.
- **Against it:** the closest analog (Branch B / iNat) just ended LB-neutral. The "Aves-only" framing conflicts with the species-coverage problem that killed original L2. Six days of compute to find out.

### §18.5 Next two actions (before the 6-day commit)

**Action A: Pre-flight the Phase 4 corpus state.**
- Does the XC v3 corpus on disk (rohanrao-derived) match the spec the smoke recipe used (32 kHz mono, ogg vorbis q4, ≤30-60s caps)?
- Does the species-list inventory (`l2_redux_full_species.json` per §14.17.16.7.1) match what Phase 4 expects?
- Disk space sanity check.
- Check Kaggle competition deadline: how many days remain on BC2026? §14.17.16.6.1 noted "if < 2 weeks, Phase 2c is risky" — same logic applies to a 6-day commit.

**Action B: Re-read the iNat v4 → LB 0.929 outcome.**
- Why did it fail to move LB? Was the failure mode Aves-corpus would inherit, or iNat-specific?
- Where in the plan is the mechanism diagnosed (Channel A vs Channel B split per §15.3)? Does that diagnosis predict Aves-corpus-at-scale would do better, worse, or the same?
- If the mechanism is "iNat-specific distribution mismatch with LB hidden test," Aves-corpus may avoid it.
- If the mechanism is "any pretrain on focal corpus + finetune on this dataset hits a soundscape transfer ceiling," Aves-corpus would also fail and the +0.0554 probe is a misleading gate.

If A confirms corpus + deadline are workable AND B's mechanism analysis suggests Aves-corpus avoids the iNat failure mode, **then** the 6-day commit is justified. Either failing → reconsider §16 Direction A vs B.

### §18.6 What §18 is NOT

- Not a green light to dispatch Phase 4. Actions A and B must complete first; user decision after.
- Not a claim that L2-redux Aves-corpus-at-scale will move LB. The +0.015 to +0.030 is a measured prediction with a known compressive uncertainty range that includes "essentially zero" at the bottom edge.
- Not a path to "+0.03" in <6 days. If timeline pressure exists, this lever may be infeasible regardless of merit.

---

## §18.7 L2-redux smoke fold-0 probe RESULT — gate NOT passed (2026-05-13 19:18 EDT)

The §18.5 Action B cheap de-risk dispatched at 17:52 EDT, completed 19:18 EDT on deepthought. Single fold (fold-0), `--init-from models/l2_redux/l2_redux_best.pt --ft-recipe production --loss hybrid --mixstyle-p 0.5 --epochs 25`. PID 195279, 1h 26m 36s wall-clock.

### Final result

**Best val_roc_auc = 0.7317 at epoch 21** → saved `models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid_prodft.pt` on DT.

Trajectory: ramp up from 0.54 → 0.71 (e1-e9), oscillation 0.66-0.71 (e10-e14), late climb to 0.7317 (e21), then collapse to 0.6772 (e22) and recover to ~0.72 (e23-e25). The "best" is a single-epoch spike; stable band sits at 0.71-0.73.

### Against the §18.5 gate

| Anchor | fold-0 val_v2 | Δ vs L2-redux smoke (0.7317) |
|---|---|---|
| **ImageNet baseline (production recipe)** | **0.7414** | **−0.010** |
| iNat v4 fold-0 | 0.7425 | −0.011 |
| iNat v1 (gentle-recipe-killed) | 0.7130 | +0.019 |

**L2-redux smoke is −0.010 below the ImageNet baseline.** The §18.5 gate (val_v2 > 0.7414 → encoder advantage survives finetune) is **NOT PASSED.**

### What this rules out

The +0.0554 frozen-probe Δ documented at §14.17.15.8.2 did not survive full finetuning. Compression from frozen-probe to fine-tuned val was severe enough to fully erase the gain. **Same failure mode as iNat:** encoder advantage at probe layer dissolves at finetune stage on this distribution.

### What this does NOT rule out

1. **Phase 4 at scale.** Smoke was 70K/636 species at 5 epochs; full 800K/7591 at 50 epochs is qualitatively different. Could produce a stronger init that survives finetune. **No empirical evidence for this in either direction now** — Phase 4 at scale was never run.
2. **LB lift despite flat val.** Session has established val→LB transfer is broken at ±0.005 resolution. A flat val outcome doesn't strictly forbid LB movement, but the historical pattern (v4 +0.024 OOF → −0.001 LB) makes it improbable.

### Implication for the Phase 4 commit decision

§18.4 framed L2-redux Aves-corpus-at-scale as "the strongest 'alive' lever in the plan, but the strength is modest." After §18.7, the strength is **considerably weaker**:
- The measured probe Δ (+0.0554) is now known not to translate to fine-tuned val (washed out)
- The arithmetic "0.5 × +0.055 → +0.015 to +0.030 LB" assumed the frozen-probe Δ would partially survive; it didn't at all on the smoke
- Phase 4 at scale would need to overcome a *demonstrated* finetune-stage washout, not just an inferred one

### Session-wide pattern after §18.7

Five probes converged in this session:
1. v4 5-fold LB push (morning) — LB 0.929, −0.001
2. §17 three winner-recipe techniques — all have prior LB kill records
3. Soundscape OOF v4-v56 — +0.024 OOF → −0.001 LB (gate falsified)
4. Quantile-Mix fold-fusion direction — OOF only, unlikely to transfer per session lesson
5. **§18.7 L2-redux smoke finetune — −0.010 below baseline**

Each probe individually has alternative explanations. The aggregate is consistent with §16's "recipe ceiling reached" thesis.

### What §18.7 does NOT resolve

- §16 Direction A (commit to structurally different track) vs Direction B (plateau-management) decision
- Whether to spend a Kaggle slot probing structurally-different existing variant stacks (v4 + v56 ensemble, drop-fold variants, per-class calibration)
- Whether B1 PerceiverIO with proper OOF discipline (the §16 named option) deserves a re-baseline attempt

---

## §19 Direction B committed — plateau-management closeout (2026-05-13 ~19:30 EDT)

### §19.1 The decision

After §18.7's empirical result (L2-redux smoke fold-0 finetune landed −0.010 below ImageNet baseline, failing the §18.5 gate), the strategic question from §16 has been resolved:

**Direction B — accept the recipe ceiling, optimize for best legitimate finish in the remaining 21 days of BC2026.**

Direction A (Phase 4 at scale on the L2-redux Aves corpus) is **deferred indefinitely**. The smoke probe was designed precisely as the cheap de-risk before committing 6 days; it argued against the commit, and we're respecting that signal.

### §19.2 Evidence base

This decision rests on five converging probes from 2026-05-13:

| Probe | Outcome |
|---|---|
| v4 5-fold LB push (morning) | LB 0.929 (−0.001 vs baseline) |
| §17 three winner-recipe techniques | All three have prior LB kill records (L1/L2/L3 + A2 family) |
| Soundscape OOF v4-v56 | +0.024 OOF → −0.001 LB (third gate falsified) |
| Quantile-Mix fold-fusion OOF probe | +0.010 OOF gain unlikely to transfer per session lesson |
| **§18.7 L2-redux smoke fold-0** | **−0.010 below baseline; +0.0554 frozen-probe Δ washed out** |

Each individually has an alternative explanation. The aggregate is inconsistent with "the recipe has more to give" at the ±0.005 LB noise floor we operate in. §16 (written the previous night in frustration) is now empirically validated.

### §19.3 In scope — next 2-3 days

1. **Best-submission selection.** Confirm the strongest existing Kaggle submission is the one selected for final private LB scoring. Top candidates: **v50 (LB 0.931)** and **v73 (LB 0.929)**. Read the Kaggle competition rules for the final-selection mechanism (auto-select highest public LB, or manual selection by deadline?) and lock in.
2. **One low-cost ensemble probe.** Pick ONE of these — not all three:
   - **(2a)** v4 + v56 model-level ensemble (rank-fuse the two existing 5-fold/4-fold checkpoint sets; no training)
   - **(2b)** v4 with per-class isotonic calibration trained on the 1478-window soundscape OOF pool
   - **(2c)** Drop-fold-X variants on v4 5-fold (per §15.3 the per-fold profile is documented; one targeted drop may surface)
3. **Per-class threshold tuning** (optional, before #2 if it doesn't replace a slot). Use soundscape OOF to set per-class decision thresholds. Tightens edge without burning a slot.

Hard budget: **one** Kaggle slot for the ensemble probe in #2. Slots are scarce.

### §19.4 NOT in scope

- ❌ Phase 4 at scale on L2-redux Aves corpus
- ❌ Any new pretrain on any corpus
- ❌ B1 PerceiverIO re-baseline (deferred to §19.6 escape hatch if §19.3 surprises positive)
- ❌ New training runs of any kind in the current recipe family
- ❌ Bundle 3 / XC v3 with a different recipe
- ❌ More Quantile-Mix sweeps or fusion experiments
- ❌ Lever-search of any kind on Track A

### §19.5 Day-4 decision gate (after §19.3 items 1 + 2 land)

| Outcome | Action |
|---|---|
| Ensemble probe lands 0.929-0.932 (expected band) | Commit fully to Direction B; finish on submission selection + admin closeout |
| Ensemble probe surprises positive (≥+0.005 vs v50's 0.931) | Trigger §19.6 escape hatch |
| Ensemble probe surprises negative (≤−0.005 vs current best) | Discard, revert to v50/v73 selection, commit to Direction B |

### §19.6 Escape hatch — single-shot, only if §19.5 triggers

Re-baseline **Track B1 PerceiverIO with proper OOF discipline**. 3-day time box, soundscape OOF on the 1478-window pool as the gate (not val_v2 — which we know is broken). The OOF-gate failure that previously shelved B1 used val_v2; the failure may be diagnostic, not real-lever-dead.

**Trigger condition:** §19.3 item (2) lands LB ≥0.934 (i.e. clears v50's 0.931 by ≥+0.003). Without this trigger, escape hatch is not taken.

### §19.7 BC2026 endgame target

- **Realistic LB band:** 0.929-0.932 (current submissions)
- **Realistic ranking:** top-N, where N depends on field compression. 2025 precedent: 1st at 0.93 was only +0.03 ahead of #38; top-50 plausible
- **NOT the target:** matching the +0.026 gap to current LB #1 (0.959). The day's evidence says that requires a structurally different recipe that this stack doesn't have

### §19.8 Don'ts on resuming work

- **Don't re-open Direction A under emotional pressure ("one more try").** The §18.5 criterion was tested; it failed. Re-opening requires NEW evidence, not new optimism. Per [[feedback_verify_plan_numeric_claims]] + [[feedback_check_plan_before_recommending]].
- **Don't trust any val_v2 "win" without LB or soundscape OOF confirmation.** Session established val→LB transfer is broken.
- **Don't burn a Kaggle slot on a "noise floor" probe.** Anything with expected LB delta < 0.005 wastes a slot.
- **Don't extend §19.6's 3-day box** if it fires. The escape hatch is single-shot — if 3 days doesn't produce a clean B1 OOF that beats the current best, B1 is done for this competition.
- **Don't list more "ideas" without explicit `--ideas-ok` from the user.** The session has documented dozens of dead candidate levers; further enumeration is noise.

---

## §20 Perch-as-encoder probe — RESULT: gate-fail, killed (2026-05-13 ~20:00 EDT)

### §20.1 Was supposed to be

3-day time-boxed swing. Train fresh MLP head on Perch v2 frozen embeddings of BC2026 train_audio (5-fold), evaluate on 1478-window soundscape OOF, gate: must beat v4-rank-fuse (0.7672) by ≥+0.005 to earn a Kaggle slot. Per §19.6 escape hatch structure.

### §20.2 Discovered: significant pre-existing infrastructure

Per `perch_v2/perch_plan.md` (March 2026), this probe was scoped, partially built, and abandoned in favor of the SED track. On disk:
- `perch_v2/models/perch_mlp_soup.pt` (3.6 MB, 5-fold averaged MLP, trained March 24)
- `perch_v2/data/processed/perch_embeddings/train_soundscapes/` — full coverage of all 1478 OOF windows (66 files × per-5s NPYs)
- `perch_v2/src/train_perch_probe.py` — training script with ASL loss + GroupShuffleSplit at file level
- `data/external/birdclef-0911/perch_v2_no_dft.onnx` — production Perch ONNX (413 MB)

This collapsed the probe from "3-day build" to "30-minute eval of existing soup."

### §20.3 Train/val split contamination

`train_perch_probe.py:split_soundscape_expert` uses seed=42, train_frac=0.8 → 52 train files (1174 windows) / 14 val files (304 windows). **The MLP soup was trained on 1174 of the 1478 OOF windows.** Running it on the full 1478 gives a contaminated 0.9498 — mostly overfit.

### §20.4 Clean held-out result

Evaluated on the 304-window held-out val (14 files MLP never saw at train time). v4 5-fold predictions sliced to the same 304 windows for apples-to-apples:

| Model | macro AUC on 304-window held-out (28 classes with positives) |
|---|---|
| **v4 5-fold rank-fuse** | **0.8057** |
| v4 5-fold mean-fuse | 0.8060 |
| **Perch MLP soup** | **0.7758** |
| **Δ Perch − v4 rank** | **−0.0299** |

Per-class AUC dist for Perch: min 0.167, median 0.816, q75 0.938 — high variance, several classes near-broken.

### §20.5 Why this is decisive (steelman the negative result)

Perch MLP had **dual advantages** at training time:
1. Focal train_audio with Perch embeddings (the planned probe content)
2. **80% of the soundscape pool** (1174 windows) directly as soundscape training data

v4 had **only** focal train_audio in training. Despite Perch's massive distributional advantage, v4 wins by 0.030 on the fair-to-both held-out pool. A version of Perch trained *without* soundscape exposure (the clean Perch-as-encoder hypothesis) would be strictly worse than the 0.7758 measured here.

The hypothesis "Perch's pretrained bird-audio encoder gives an inductive-bias advantage over EffNet-from-scratch on this dataset" is falsified. Perch's encoder doesn't have a transferable advantage on the BC2026 Pantanal distribution that survives downstream head training.

### §20.6 Decision per §19.5 gate-fail rule

**KILL probe. Lock v50 + v73 at deadline. Direction B fully committed.**

§20 closes the last credibly-different lever in the plan. The lever-search is empirically exhausted on this stack for this competition.

### §20.7 Implications worth noting (not action items)

1. **Perch-as-teacher (§13 Track C) was killed for embedding-mismatch reasons.** Now we have separate evidence that **Perch-as-encoder also doesn't help on this dataset.** Both kills together strongly suggest Perch is the wrong external pretrained signal for BC2026 Pantanal regardless of how you wire it.
2. **The pre-existing perch_v2 side-project might have arrived at this same conclusion in March if they'd computed val-only-AUC.** The 0.801 "soundscape val AUC" cited in `perch_plan.md` was likely also mixed train+val. Worth a memory note: when reading historical val numbers in the plan, verify whether they're clean OOF or in-sample contaminated.
3. **Today's session has now run 6 probes, all converging on "recipe ceiling reached."** §16 written at 01:40 EDT, validated empirically by 19:30 EDT.

### §20.8 Final state for the rest of the competition

- **v50** (LB 0.931, 2026-04-16 21:29) — production-recipe baseline, ImageNet init
- **v73** (LB 0.929, 2026-05-13) — v4 5-fold + iNat pretrain init, more recent training data lineage
- Both selectable for final by deadline 2026-06-03 23:59 UTC
- No further training runs in scope
- No further LB slots to be burned on noise-floor probes

### §20.9 Don'ts going forward

- **Don't re-open Perch-as-encoder** unless a structurally different MLP architecture / training recipe is proposed (not just hyperparameter sweep). The current architecture was tested at the strongest possible advantage and failed.
- **Don't trust historical "soundscape val AUC" numbers in this project's plans** without verifying clean OOF. The 0.801 March claim was train-contaminated.
- **Don't add another lever-search loop.** The §19 commitment plus §20 closeout means the lever question is settled.

---

## §21 Out-of-recipe-family options — three genuinely new directions (2026-05-13 ~20:30 EDT)

After §20 closed all in-family levers, user requested out-of-family options. Three documented here, ranked by expected value (prior probability × payoff / effort). Each is structurally outside the "EffNet-SED-on-mels + focal-pretrain + ImageNet/iNat init" family the session falsified.

### §21.0 Important constraint from session

**Use clean held-out OOF as the gate**, NOT the full 1478-window pool. Per `feedback_verify_historical_val_clean.md` (memory entry created 2026-05-13 after §20 closeout): historical "soundscape val AUC" numbers in this project's plans have been train-contaminated. The §20.4 honest comparison was on the 304-window held-out (14 files), not the 1478-window full pool. Apples-to-apples gate: v4 rank-fuse on the 304-window held-out = **0.8057**. Any new probe must beat 0.8057 by ≥+0.005 on the *same 304 windows*.

If a new probe trains on train_soundscapes, it MUST use the same seed=42 train_frac=0.8 split that perch_v2 used, so the 304 held-out windows remain genuine for gate evaluation. Reproduce the split per `train_perch_probe.py:split_soundscape_expert`.

### §21.1 Option 1 — Background-mixing augmentation (Pantanal-style soundscapes)

**Mechanism:** Use the 52 *training-side* `train_soundscapes` files (per the seed=42 0.8 split — the same 1174 windows the perch_v2 work used) as a background-noise source. Mix focal training clips with Pantanal-style background at SNR -5 to +20 dB. Trains the encoder on focal-with-Pantanal-background, structurally closer to the test distribution than clean focal.

**Why different:** Every recipe tried this competition pretrained/finetuned on clean focal. Session diagnosed *focal→soundscape distribution mismatch* as the failure mode (§15, §16, §18.7, §20). This augmentation directly attacks that mechanism without changing encoder or pretrain — orthogonal to all six probes from today.

**Cost:** ~1-2 days.
- Existing `train_a1.py` has a `bg_noise_dir` parameter currently set to None (per `feedback_read_call_sites_not_docstrings.md`). Wire it up.
- Build a bg_noise_dir from the 52 training-side soundscape files (NOT the 14 held-out files — those stay clean for OOF eval).
- Smoke fold-0 finetune with SNR sweep (3 values, e.g. 0/10/20 dB).
- If clean held-out OOF on the 304 windows beats 0.8057 by ≥+0.005, expand to 5-fold.

**Prior probability LB +0.005:** ~20-25%. Highest in this list because targets diagnosed failure mode rather than guessing.

**Risk:** could regress if SNR too low (drown signal). Sweep mitigates.

### §21.2 Option 2 — BirdNET as alternative encoder (gated by Option 1 outcome)

**Mechanism:** BirdNET-v2.4 (Cornell). Frozen feature extractor + MLP head, similar to §20 Perch setup but using BirdNET embeddings (different architecture, different training corpus, different inductive bias from Perch).

**Why different from §20:** Perch was trained on Xeno-Canto + iNat. BirdNET-v2.4 was trained on a different corpus + has different architecture (audio CNN with attention vs Perch's deeper conv stack). 2025 top finishers sometimes used BirdNET separately from Perch.

**Cost:** ~2-3 days. Download BirdNET, Kaggle-side embedding extraction (embedding-mismatch constraint per `new_plan.md` §13), train MLP head, OOF eval with clean held-out gate.

**Prior probability LB +0.005:** ~15-20%. Lower than Option 1 because §20 falsified bird-pretrained-encoder for this data; BirdNET may inherit the same failure mode mechanism.

**Gate to start:** Option 1 must complete (pass or fail clearly) before Option 2 launches. Don't run both in parallel.

### §21.3 Option 3 — Pantanal-geo-filtered XC pretrain (gated by Option 1 + 2 outcomes)

**Mechanism:** XC v3 corpus already on disk (524K files, 6,971 species at `data/external/xenocanto_bulk/`). Filter to **Brazil/cerrado/Pantanal-region** recordings via XC geographic metadata. Pretrain on the geo-filtered subset. Address recording-conditions match (microphones, ambient, biome acoustics) instead of species-coverage match (which L2-redux smoke tried and failed §18.7).

**Why different from L2-redux:** L2-redux smoke pretrained on globally-mixed Aves. The mechanism would have been "more bird audio at scale → better encoder." That failed. Pantanal-geo-pretrain optimizes for *acoustic-environment match*, a different prior.

**Cost:** ~2-3 days. Filter XC v3 manifest (~hours), re-pretrain smoke + scale, fold-0 finetune, clean held-out OOF gate.

**Prior probability LB +0.005:** ~15-20%.

**Pre-flight check:** if Pantanal-tagged XC subset is <10K clips, abort — too small to make a pretrain useful.

**Gate to start:** Options 1 and 2 must complete (pass or fail clearly). Don't start without explicit go.

### §21.4 Hard rules — same as §19 / §20

- **3-day time box per option.** No extending.
- **Clean held-out OOF gate (304 windows, 14 files held out per seed=42 split).** Not val_v2, not the full 1478. v4 rank-fuse = 0.8057 on this pool; must beat by ≥+0.005.
- **Gate-fail = kill that option, move to next or lock v50+v73.** Don't extend "to give it one more chance."
- **One Kaggle slot per gate-pass.** No multi-slot sweeps.
- **No new pretrain at scale without a smoke-gate pass first.** §18.7 lesson — smoke-then-scale only.
- **Track training-set contamination explicitly.** Per `feedback_verify_historical_val_clean.md`, any historical AUC number must be re-verified on clean held-out before using as evidence.

### §21.5 Day-7 decision gate (after all options complete or are killed)

- If any option lands LB ≥0.934 → that's the new production submission, use as slot 2 (replacing v73)
- If all three options land LB in 0.929-0.932 noise band → lock v50 + v73, close the lever-search
- If any option lands LB <0.925 → exclude from submission selection, lock v50 + v73

---

## §22 Option 2 prep — BirdNET-as-encoder infrastructure READY (2026-05-13 ~21:50 EDT)

Prepared in parallel with Option 1 still running so the second probe can launch within minutes of the §21.5 gate decision rather than after a from-scratch build day.

### §22.1 What's prepared

**Model availability:**
- `birdnet` Python package already installed (v0.2.12, pip)
- BirdNET v2.4 model already cached at `~/.local/share/birdnet/acoustic-models/v2.4/`
- TF backend confirmed fast on CPU (**~3s per 60s clip = 0.05x real-time**, vs pb backend 28s = 0.47x real-time — TF wins 10×, no GPU needed)
- pb backend's GPU path errors out in multiprocessing setup; CPU TF is the canonical path

**BirdNET specs (vs Perch v2 for context):**
| | BirdNET v2.4 | Perch v2 |
|---|---|---|
| Embedding dim | 1024 | 1536 |
| Sample rate | 48 kHz | 32 kHz |
| Segment size | 3.0s | 5.0s |
| Species in head | 6522 | ~14000 |
| Tested? on BC2026 | not yet | §20: failed gate on clean held-out |

**Scripts written:**
- `birdnet_v2/src/extract_embeddings.py` — three pool modes:
  - `--pool soundscape_oof` — extract the 1478 OOF windows (the §20/§21 gate pool, ~75 min CPU)
  - `--pool train_soundscapes_trainside` — 52-file training-side per the seed=42 0.8 split (matches §20 Perch's split bit-identically, so the held-out 14 files / 304 windows stay clean for apples-to-apples)
  - `--pool train_audio` — focal corpus pool (35K clips, ~30h CPU — needs batching optimization before running at scale)
- `birdnet_v2/src/train_birdnet_probe.py` — MLP head trainer, scaffold-only. `BirdNetMLP` class is `in_dim=1024` clone of `perch_v2.PerchMLP`. ASL loss + 5-fold + seed=42 train/val split — bit-identical structure to `perch_v2/src/train_perch_probe.py` except for in_dim and paths. **Per-fold training loop is intentionally NotImplementedError-stubbed; copy from perch_v2 trainer at launch time.** Held-out 304 windows match §20 exactly.

**Smoke test passed (2026-05-13 21:48 EDT):**
- 5 windows extracted in 14.9s
- Each embedding: shape (1024,), float32, ~4 KB on disk
- Sample values look reasonable (sparse non-negatives, ReLU activations, norm ~17)
- Smoke files deleted after verification

### §22.2 What's NOT done

- ❌ Full embedding extraction (1478 OOF + 1174 train-side soundscape + 35K train_audio)
- ❌ MLP head training (5-fold + soup)
- ❌ Clean held-out 304-window OOF eval
- ❌ Kaggle slot

### §22.3 Cost estimate when Option 2 launches

- 1478 OOF extraction: **~75 min CPU** (sequential per-window calls; could be reduced to ~10-20 min by batching `encode_arrays` over all 1478 at once)
- 1174 train-side soundscape extraction: **~60 min CPU** (similar batch potential)
- 35K train_audio extraction: **~30h CPU** unless batched; with `encode_arrays` batch over 100-500 clips per call, probably **~3-5h CPU**
- MLP probe training (5-fold + soup): **~10-30 min** (GPU on hal9000 or CPU on skynet — tiny model)
- 304-window held-out eval: **<1 min**

**Total tight estimate if extraction is batched properly:** ~4-6h. Within §21 Option 2's 3-day box with substantial margin.

### §22.4 Gate to launch

Per §21.5: Option 1 must complete (pass or fail clearly) before Option 2 launches. Currently Option 1 at epoch 13/25 with val trajectory clearly tracking below ImageNet baseline — gate-fail looking likely. If confirmed at completion (~22:30 EDT), Option 2 is the next move.

### §22.5 Prior probability re-estimate

Earlier prior in §21.2: ~15-20% Option 2 clears the gate. Today's §20 showed bird-pretrained encoder (Perch) failed when given the **strongest possible advantage** (focal pretrain + 80% soundscape exposure). BirdNET inherits the same "bird-pretrained encoder on Pantanal" failure mode mechanism. Honest revised prior: **~10-15%**. The case for trying it is "different pretraining lineage, might transfer differently" — but the prior on that helping is weak after §20.

### §22.6 Don'ts

- **Don't launch Option 2 before Option 1 completes.** Even if Option 1 looks doomed at epoch 13, the run is cheap to finish; don't burn ssh/runon dispatch energy on parallel jobs that aren't independent on GPU.
- **Don't extract train_audio at full 30h cost without batching.** The per-window-call overhead dominates; one `encode_arrays` call over a batch is 10× faster.
- **Don't change the seed=42 train/val split.** Held-out 14 files / 304 windows must match §20 for the apples-to-apples comparison to be meaningful.
- **Don't ad-hoc the MLP architecture.** Use `BirdNetMLP` as scaffolded; per `feedback_verify_historical_val_clean.md`, deviations from the perch_v2 protocol risk re-introducing train/val contamination if anyone reuses old code patterns.

---

## §21 Option 1 RESULT — gate-fail by −0.099 (2026-05-13 22:34 EDT)

### §21.1.R1 Final outcome

| Metric | Value |
|---|---|
| Final best val_v2 | **0.6638** at epoch 21 |
| Clean held-out 304-window AUC | **0.7032** |
| v4 fold-0 anchor on same 304 windows | 0.8022 |
| **Δ vs v4 fold-0** | **−0.099** |
| §21 gate (v4 + 0.005) | 0.8072 |
| Result | **GATE FAIL** by 0.104 |
| Total wall | 1h 34m on deepthought |

### §21.1.R2 What this rules out

The hypothesis "background-mixing augmentation addresses the focal→soundscape distribution mismatch diagnosed in §15/§16/§18.7" is **decisively falsified** by the largest margin of any probe today:

| Probe | Δ vs baseline | Magnitude |
|---|---|---|
| §18.7 L2-redux smoke vs ImageNet | −0.010 | just below |
| §20 Perch MLP soup vs v4 rank-fuse | −0.030 | clear fail, adjacent |
| **§21 Option 1 bg-noise vs v4 fold-0** | **−0.099** | **catastrophic** |

The augmentation didn't merely fail to help — it materially damaged generalization. Mechanism is unclear without further analysis (the existing `_add_bg_noise` uses fixed gain 0.05-0.15 without RMS normalization, so background loudness varies per file and may mask the focal signal at unfavorable SNRs), but verifying the mechanism would require more compute on a falsified track.

### §21.1.R3 Hygiene issue

The bg-noise run saved its ckpt to the unsuffixed path:
`models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold0_seed42_hybrid_prodft.pt`

This **overwrote the v4 fold-0 PyTorch ckpt on deepthought** at the same path. v4 is preserved at the suffixed path `..._25ep_v4_val0.7425.pt` (May 12 17:23), and the production JIT ckpts in `kaggle_datasets/a1-effb0-ckpts/` are untouched. **No production impact**, but future runs from the unsuffixed v4 fold-0 PyTorch path will now load the bg-noise (failed) ckpt instead. Future probes that start with `--init-from ...prodft.pt` need to use the explicit v4 suffix.

### §21.1.R4 Per §21.4 gate-fail action

Three options per §21.4:
1. **Move to Option 2** (BirdNET-as-encoder, §22 prep is ready)
2. **Move to Option 3** (Pantanal-geo-filtered XC pretrain)
3. **Lock v50+v73 and close** the §21 sequence

The day's pattern (6 probes, all fail or worse) and the magnitude of this fail strengthen the case for option 3. Decision deferred to user — both Options 2 and 3 are infrastructurally prepared.

---

## 📌 PICK UP HERE (2026-05-13 ~23:00 EDT, sleep handoff)

### TL;DR for tomorrow morning

1. **First positive probe of the day landed tonight.** §21 Option 2 **soundscape-only BirdNET MLP** scored macro AUC = **0.8828** on the clean 304-window held-out (the §20/§21 gate pool). Δ vs gate = +0.0756. Δ vs §20 Perch MLP soup = +0.107. Δ vs v4 5-fold rank-fuse = +0.077. **First gate-pass after 6 prior fails.**
2. **Train_audio BirdNET extraction is running overnight on skynet** (~2.6h, finishes ~01:30 EDT) to enable the full §22 apples-to-apples experiment (focal+soundscape MLP, matches §20 Perch protocol).
3. **Decision tomorrow:** evaluate the full-pipeline MLP and decide whether to (a) push the soundscape-only MLP to LB now via a Kaggle BirdNET inference notebook, or (b) push the full-pipeline MLP if it beats soundscape-only.

### State of running jobs

| Field | Value |
|---|---|
| Job | BirdNET train_audio embedding extraction (batched, chunk=64) |
| Host | skynet (local) |
| PID | **368643** (also saved at `/tmp/birdnet_train_audio_pid.txt`) |
| Log | `/home/swatson/work/kaggle/BirdCLEF/birdnet_v2/log/extract_train_audio_20260513_225643.log` |
| Started | 2026-05-13 22:56:43 EDT |
| Progress | 256 / 35549 done in 68s = **267 ms/clip** |
| Expected finish | **~01:30 EDT 2026-05-14** (~2.6h total) |
| Output dir | `birdnet_v2/data/processed/embeddings/train_audio/<species_label>/<stem>.npy` |
| Each embedding | 1024-d float32, ~4 KB → total ~140 MB |

### Tomorrow morning — sanity-check the extraction first

```bash
# Check process is done
ps -p 368643 2>/dev/null && echo RUNNING || echo DONE
# Check embedding count (should be ~35549)
find /home/swatson/work/kaggle/BirdCLEF/birdnet_v2/data/processed/embeddings/train_audio -name "*.npy" | wc -l
# Tail log for any errors
tail -30 /home/swatson/work/kaggle/BirdCLEF/birdnet_v2/log/extract_train_audio_*.log
```

If extraction died early (< ~30K embeddings), check the log tail for the error and re-launch with the resume-skip behavior built into the script.

### Tomorrow's next steps (in order)

#### Step 1 — Train + eval the full-pipeline MLP

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/kaggle/BirdCLEF/birdnet_v2
rm -f log/*.log
nohup python -u src/train_birdnet_probe.py \
  --embeddings-dir data/processed/embeddings/train_audio/ \
  --folds-csv ../data/processed/train_folds.csv \
  --soundscapes-labels ../data/raw/train_soundscapes_labels.csv \
  --output-dir models/ \
  --epochs 30 --batch-size 512 --lr 1e-3 --seed 42 \
  > log/train_birdnet_probe_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Caveat:** `train_birdnet_probe.py` was sed-copied from `perch_v2/src/train_perch_probe.py` with light substitution. The training loop is the perch_v2 trainer's untouched 5-fold ASL flow. It expects soundscape embeddings at a specific path (the perch script used `data/processed/perch_embeddings/...`). **Run the train script with `--help` first** to confirm the soundscape-emb path arg, and verify it matches `data/processed/embeddings/soundscape_all/` (the BirdNET pool already extracted earlier tonight).

Expected output: 5-fold per-fold val AUC + soup AUC on the same 304-window held-out pool. Estimated wall time: **~10-30 min** on skynet GPU.

#### Step 2 — Compare full-pipeline vs soundscape-only

```
Soundscape-only result (already known):  0.8828
Perch (focal + 80% soundscape):           0.7758
v4 5-fold rank-fuse anchor:               0.8057
```

Three possible outcomes for the full pipeline:
- **>0.8828** — focal helps BirdNET. Use full pipeline for LB.
- **0.78 - 0.88** — focal dilutes soundscape signal but still gate-passing. Use whichever is higher.
- **<0.78** — focal actively hurts BirdNET. Use soundscape-only.

#### Step 3 — Decide on Kaggle slot

Per §21.5: gate-passing OOF earns one LB slot. The soundscape-only MLP already passes by +0.077. The gate-pass justifies the slot regardless of full-pipeline outcome.

To submit, need a Kaggle inference notebook that:
1. Loads BirdNET v2.4 (tflite or onnx — Kaggle CPU-only, no GPU at inference)
2. Computes embeddings on test soundscapes (5s windows at the BC2026 grid)
3. Loads the trained MLP soup (the one with the best val AUC)
4. Outputs predictions

Build estimate: ~half day. Or pivot to using the existing `birdclef-2026-perch-mlp-probe` kernel as a template — swap Perch ONNX for BirdNET model.

### Risks / what to watch for tomorrow

1. **In-domain val ≠ LB hidden test.** The 0.8828 was on 14 random files held out from the labeled 66-file pool. LB is on fresh unseen soundscapes. The day's pattern says val→LB transfer is lossy. Could land anywhere from +0.025 LB (good) to ±0.000 LB (in-domain-only signal). Honestly probably ~+0.010 to +0.020 LB.
2. **234-class taxonomy includes non-Aves.** Only 28 active classes in the 304 held-out are scored on the OOF — likely mostly Aves. BirdNET is bird-specialized (6522 bird species). For amphibians/mammals/insects on LB it may fail. Per-class result inspection on the held-out showed 2 classes underperforming (n_pos=4 and 6) — small sample but worth checking what species those were.
3. **Per `feedback_verify_historical_val_clean.md`** — the 0.8828 is on the clean held-out (verified — file-level split, no contamination). Different from §20 Perch's 0.9498 contaminated number. Don't fall into the same trap when reading the train+val log of train_birdnet_probe.py — its in-training val_auc may be on a different pool (perch_v2 trained on focal AND soundscape together with the soundscape-train-frac being the same 0.8). Verify before quoting numbers.

### Files of note

| File | Purpose |
|---|---|
| `birdnet_v2/src/extract_embeddings.py` | Batched extraction (chunk=64). Running tonight. |
| `birdnet_v2/src/train_eval_soundscape_only.py` | Tonight's soundscape-only probe — produced the 0.8828 result. |
| `birdnet_v2/src/train_birdnet_probe.py` | Full-pipeline trainer (5-fold + soup, perch_v2 port). For tomorrow. |
| `birdnet_v2/src/config.py` | BirdNET-specific config (EMBEDDING_DIM=1024, SAMPLE_RATE=48000, same MLP/ASL hyperparams as perch_v2). |
| `birdnet_v2/data/birdnet_soundscape_only_oof.npz` | Saved predictions of the soundscape-only MLP on 304 held-out. AUC=0.8828. |
| `birdnet_v2/data/processed/embeddings/soundscape_all/` | All 66 labeled soundscape files extracted (792 windows). |
| `birdnet_v2/data/processed/embeddings/train_audio/` | Being populated overnight. Target 35549 entries. |
| Production submissions intact | v50 (LB 0.931, 04-16) + v73 (LB 0.929, 05-13) selectable at https://www.kaggle.com/competitions/birdclef-2026/submissions |

### Today's session summary

Six probes failed in the recipe family. Seventh probe (BirdNET soundscape-only) is the first to pass the gate, and by a wide margin. The §16 "recipe ceiling" diagnosis was correct for the EffNet-SED-on-mels family; BirdNET embeddings are structurally different enough to potentially break the ceiling — **pending the LB-resolution-relevant question of whether OOF gains transfer.**

Today's emotional read (from "what else is new?" through "are you suggesting we give up?" through tonight's surprise) is worth naming: the day was largely confirmation of §16's plateau diagnosis, capped with an unexpected gate-pass that may or may not survive LB. Tomorrow's decision is whether to spend time on the BirdNET LB path or accept the soundscape-only as evidence and decide on Kaggle slot management.

### Don'ts on next pickup

- **Don't burn a Kaggle slot before sanity-checking the soundscape-only MLP on a broader pool first.** The 28-class denominator is small; per-class inspection on the held-out before slot-burn is worth 5 minutes.
- **Don't assume the full-pipeline trainer (train_birdnet_probe.py) works without verifying.** It's a perch_v2 port with light substitution. Likely correct, but run `--help` and `--smoke-test`-equivalent first.
- **Don't extend the §21 sequence to Option 3** (Pantanal-geo-filtered XC pretrain) unless tomorrow's results give a specific reason. The session's noise floor still applies; new training runs in the same recipe family are still subject to it.
- **Don't kill PID 368643 unless it's clearly hung.** The first 256 clips landed cleanly in 68s; expected to continue at ~267 ms/clip.
- **Don't push v50 or v73 again.** Both already selectable for final; selection is in the Kaggle web UI.

## §23 Full-pipeline BirdNET MLP — gate-passes but loses to SS-only (2026-05-14 ~15:15 EDT)

### §23.1 Result on the same 304-window held-out (28 active classes, file-level split, seed=42)

| Probe | Macro AUC | Δ vs SS-only |
|---|---|---|
| **Soundscape-only MLP** (52 SS files only, baseline) | **0.8828** | — |
| Full-pipeline MLP soup (35,549 focal + 52 SS, 5-fold avg) | 0.8540 | **−0.0288** |
| Full-pipeline best fold (F5) | 0.8848 | +0.0020 |
| Full-pipeline mean per-fold | 0.8773 | −0.0055 |

Per-fold: F1=0.8796 F2=0.8838 F3=0.8691 F4=0.8691 F5=0.8848.

### §23.2 Read

Focal training data dilutes the BirdNET soundscape signal. The 5-fold soup loses 0.029 to SS-only; the best individual fold is essentially tied. Two reasons this is internally consistent:

- BirdNET embeddings are already strong out-of-the-box on Aves; the focal data adds a different distribution (single-species iNat/XC clips, no real soundscape mix) that pulls the head away from the in-domain soundscape distribution.
- The model soup (avg of 5 fold weights) is worse than the per-fold mean (0.8540 vs 0.8773), suggesting the fold heads aren't aligned in weight space — averaging weights ≠ averaging predictions when heads are diverse.

Per the §22 trichotomy, this is the middle case: **focal hurts but full-pipeline still gate-passes.** Verdict: submit the SS-only MLP, not the full-pipeline.

### §23.3 Decision

Per §21.5 gate-pass earns one Kaggle slot. Going with: **build BirdNET SS-only inference notebook** (plan-faithful, half-day estimate). Slot leverage is solo-vs-v73 LB delta — not blend-with-v73, which costs more time and may not even land before deadline.

Caveats already in plan:
- `feedback_val_v2_not_predictive_at_small_LB_deltas.md` — val→LB lossy at small deltas; +0.077 OOF gain may not transfer
- 28 active classes only on the held-out — LB taxonomy is broader; BirdNET strength on Aves doesn't guarantee non-Aves coverage
- v73 is currently LB 0.929; BirdNET solo at OOF 0.88 will probably score lower than v73 on LB even on the optimistic case

The slot is a check on whether BirdNET embeddings are LB-useful at all, not a v73 replacement.

### §23.4 Artifacts produced this session

| File | Purpose |
|---|---|
| `birdnet_v2/models/birdnet_soundscape_only_best.pt` | SS-only MLP best-epoch state_dict (val_auc=0.8828, ep13). For Kaggle inference. |
| `birdnet_v2/models/perch_mlp_soup.pt` | 5-fold soup of full-pipeline. Not selected for LB. Keep for ensembling later if needed. |
| `birdnet_v2/data/birdnet_fullpipeline_oof.npz` | 304-window OOF of soup. AUC=0.8540. |
| `birdnet_v2/data/birdnet_soundscape_only_oof.npz` | (existing) 304-window OOF of SS-only. AUC=0.8828. |
| `birdnet_v2/src/eval_soup_on_holdout.py` | Reusable eval script for any BirdNET soup ckpt on the held-out pool. |
| `birdnet_v2/data/processed/embeddings/train_audio/` | 35,549 1024-d focal embeddings. ~140 MB on disk. |
| `birdnet_v2/data/processed/embeddings/train_soundscapes` | symlink → soundscape_all/ for trainer compatibility. |

### §23.5 Next: inference notebook

Build estimate per §22: ~half day. Need:
1. BirdNET v2.4 model artifact accessible from Kaggle (tflite or onnx — Kaggle CPU-only at inference)
2. Kaggle dataset for the SS-only ckpt + species index
3. Notebook that: loads BirdNET → runs over 5s windows of test soundscapes → mean-pools its 3s segment outputs to the 5s grid → feeds 1024-d embeddings to the MLP → writes the submission CSV in the BC2026 format.

## §24 BirdNET SS-only LB result — −0.187 gap to v73, kill the path (2026-05-14 ~20:00 EDT)

### §24.1 Numbers

| Reference | Value | Δ vs LB |
|---|---|---|
| **BirdNET SS-only — LB public score** | **0.742** | — |
| v73 (current production) | 0.929 | +0.187 |
| v50 (current production) | 0.931 | +0.189 |
| OOF on 304-window held-out (in-distribution) | 0.8828 | (val pool, not LB) |
| OOF→LB gap | −0.140 | one of worst in project history |

### §24.2 Root causes (ranked)

1. **Encoder taxonomy mismatch.** BirdNET v2.4 is trained on 6522 bird species, no amphibians/reptiles/insects. The held-out 28-class active pool was 12 Aves + 12 Amphibia + 2 Mammalia + 2 Insecta. On LB hidden test the broader class distribution dilutes the Aves strength with low-signal output on non-Aves classes.
2. **Tiny training set.** 1174 windows / 234 classes ≈ 5 windows per class. The MLP memorized the 52-file train distribution.
3. **Held-out distribution match.** The 14 held-out files are the same recording cluster as the 52 train files; LB test is from different recordings. Even within Aves, the in-domain val overstated generalization.

### §24.3 What §24 retires

- **§22 BirdNET-as-encoder solo path** — dead. The 0.187 gap is structural (taxonomy), not closable by MLP tuning.
- **The "first gate-pass after 6 fails" framing from §22 / §23** was misleading. Small + biased held-out gave a false signal.

### §24.4 Don'ts going forward

- **Don't propose another single-encoder probe targeting the held-out gate** without first verifying the encoder's training taxonomy covers the LB taxonomy.
- **Don't blend BirdNET into v73.** At 0.742 LB, any blend weight that moves v73 meaningfully drags it down. Diversity-via-orthogonality argument is theoretically possible but priors are bad given the magnitude.
- **Don't extend the §21/§22 sequence to Option 3 (Pantanal-geo XC pretrain) on the same justification.** The §17.7b retraction logic still applies — same recipe family, same noise floor, same val→LB transfer risk.

### §24.5 Strategic state after §24

Production submissions intact: v50 (LB 0.931, 04-16) + v73 (LB 0.929, 05-13) selected for final. No remaining path in the §17-§22 arc has produced a positive LB result. The plateau diagnosis from §16 has now been confirmed by 7 sequential null/negative results across §16→§24. Any future Kaggle slot use should be conditioned on either (a) a probe targeting LB-reflective validation (broader-pool soundscape eval, not the 14-file gate), or (b) explicit acceptance that the slot is informational only.

### §24.6 Slot accounting

Slots used this session: 1 (BirdNET SS-only). LB slots remaining: refer to Kaggle web UI submission count (5/day cap). Production-selectable submissions for final scoring: v50, v73 (both already selected, no further pushes recommended).

## §25 Next-step options (post-§24 BirdNET kill, 2026-05-14 ~20:30 EDT)

### §25.0 Why this section exists

After §24 (BirdNET LB 0.742, −0.187 vs v73), the question "what to try next" was put back in play. Sticking to options that **don't burn a Kaggle slot** until the broken val→LB gate is fixed.

### §25.1 The four options

| # | Option | Cost | Plausible value |
|---|---|---|---|
| 1 | **Build a broader OOF gate using all 66 fully-labeled train_soundscapes** (per `feedback_val_v2_not_predictive_at_small_LB_deltas.md` step 2). Re-evaluate BirdNET, Perch, v4 on the 1478-window pool. If this gate would have correctly said BirdNET was weak → we have an LB-correlated gate going forward. | ~2 hours | High — addresses root cause of the slot-burn |
| 2 | **TTA on v73 inference path.** Apply test-time mel/window/loudness perturbations and rank-fuse. Almost free to add to existing kernel; sometimes +0.005-0.010 LB. | Half day | Medium — small but free upside on production |
| 3 | **Probability calibration on v73** (isotonic per-class on the new 66-file gate). Whether the LB metric is rank-AUC or sample-AUC matters — calibration helps the latter more. | 1-2 hours | Low-medium — depends on metric |
| 4 | **Just stop.** v50 and v73 are selected, both ~0.93 LB. Competition close approaching; time-cost of more probes outweighs EV. | Free | High in a different sense |

### §25.2 Decision (2026-05-14 ~20:30 EDT)

User chose "more probes" + executing #1 first. §25.3 captures the prerequisite work to make any future probe LB-relevant.

### §25.3 #1 execution plan

- Existing artifacts cover the 1478-window pool: `data/v4_5fold_soundscape_oof.npz`, `data/v56_soundscape_oof.npz`, `data/perch_mlp_soup_soundscape_oof.npz`. Each has `(probs|probs_mean, y_true, filenames, start_sec)`.
- Step A: compute broader-pool macro AUC for each existing model — no new compute.
- Step B: re-train BirdNET SS-only as 5-fold over the 66 files to get OOF on all 1478 windows. Compare to the LB-known 0.742.
- Step C: tabulate (broader-pool OOF, known LB) for each model and check if ordering matches. If yes, this is the new gate going forward. If no, the gate is also broken and we need a different approach (TTA on v73, isotonic calibration, or stop).

### §25.4 What §25 doesn't promise

- The broader-pool gate may also be uncorrelated with LB — this is a diagnostic, not a fix. If it fails, the real signal is "no local gate predicts LB at the resolution we care about" and we should stop probing.
- §25 doesn't commit to executing #2 or #3. Those decisions hinge on what #1 reveals.

### §25.5 #1 RESULT — broader-pool gate validates correctly (2026-05-14 ~21:00 EDT)

| Model | Broader-pool 1478-window OOF AUC | Active classes | Known LB | LB − OOF gap |
|---|---|---|---|---|
| v4 5-fold (probs_mean) | 0.7775 | 75 | v73 ~0.929 | +0.15 |
| v4 5-fold (rank-fuse) | 0.7672 | 75 | — | — |
| v56 (4-fold) | 0.7290 | 75 | — | — |
| **BirdNET 5-fold rebuild (clean OOF)** | **0.5644** | 75 | 0.742 | +0.18 |
| (perch_mlp_soup is in-sample contaminated — excluded) | 0.9498 | 75 | — | not comparable |

**Δ (v4 vs BirdNET) on broader gate: −0.213. Δ on LB: −0.187.** The gate correctly orders the two models AND captures the magnitude of the gap within ~0.03.

### §25.6 Why BirdNET dropped from 0.8828 → 0.5644

The 0.8828 was the SS-only single-train-val-split AUC on 14 held-out files (28 active classes). The 0.5644 is the same MLP architecture/protocol, but **5-folded over all 66 files**, predictions concatenated, macro AUC over 75 active classes. Per-fold val AUCs from the 5-fold rebuild were 0.81-0.92 — consistent with the 0.8828 — but each fold scored only its own narrow active-class subset. When predictions are concatenated and scored over the broader 75-class pool, classes that 4/5 folds didn't see drag the macro AUC to 0.5644.

This is the same pathology as `feedback_per_fold_val_misleads_ensemble.md` — per-fold val AUC overstates what a clean broader-pool evaluation would show. The fix is to **always score on the concatenated broader-pool 5-fold OOF, not the per-fold val AUC.**

### §25.7 Implications

1. **The broader-pool gate is now validated as LB-correlated.** Future probes should be filtered through this gate before any Kaggle slot decision.
2. **Any future probe with broader-pool OOF < 0.77 (the v4 anchor) should not get a slot.** Above 0.77, the slot might still not pay off, but at least the OOF→LB transfer math doesn't predict catastrophic loss.
3. **The §22→§24 BirdNET slot was avoidable.** The broader-gate diagnostic exists, was always available, and would have triggered a kill before submission.
4. **The §16 plateau diagnosis is reinforced.** v4 broader-pool OOF 0.7775 is the rough ceiling for any single-encoder probe. The +0.15 LB gap (0.7775 → 0.929) comes from stack ensembling, calibration, and post-processing on top of v4 — not from the encoder alone.

### §25.8 Decision after §25.5

Options #2 (TTA on v73) and #3 (isotonic calibration on v73) are now the relevant next probes — both target the +0.15 stack-vs-encoder gap that drives v73 above v4. Option #4 (stop) remains the safe default.

### §25.9 #3 isotonic calibration RESULT — gate kill (2026-05-14 ~21:30 EDT)

**On d2_beta substrate (708 win, site-level folds, 4-fold A1):** cal_A1 = 0.7855 vs uncal 0.7359 → +0.0496. cal_A1+cal_Proto blend (drop B1) → 0.7873 (+0.117 vs prod_fused 0.6699).

**On production-relevant substrate (1478 win, file-level folds, 5-fold A1, what cell 41 actually emits):** cal_A1 = 0.7742 vs uncal 0.7672 → **+0.0070** (noise-floor).

The d2_beta +0.05 was largely a substrate artifact: site-level folds exposed cross-site calibration drift, file-level folds in the production substrate mask it. On the substrate that matters, the calibration gain is at the noise floor.

**Decision: gate-fail. Do not push.** Per `feedback_broader_pool_oof_is_lb_correlated.md`, broader-pool gain ≥ +0.005 is the slot threshold. +0.007 nominally passes, but the substrate dependency makes it unlikely to hold across the LB-relevant test distribution. Any LB delta would be ±0.005, indistinguishable from single-submission noise.

**This is what successful gating looks like.** The probe was real (+0.05 OOF visible on one substrate), the gate caught the substrate-specific nature of the gain, the slot wasn't burned. Compare to §22-§24 BirdNET, where the absence of broader-pool gating let a −0.187 LB delta through.

### §25.10 §25 closeout

- #1 (broader-pool gate): **VALIDATED** (memory entry written)
- #2 (TTA on v73): **ungatable on local OOF, deferred indefinitely** (would require LB-only iteration which we no longer want to fund)
- #3 (isotonic calibration on v73): **gate-failed at the production substrate**
- #4 (stop): **most defensible after §25.9**

Production submissions intact: v50 (LB 0.931) + v73 (LB 0.929), both selected for final. Strategic posture: protect existing submissions, don't burn further slots without a probe that gate-passes on the 1478-window production substrate by ≥ +0.005 macro AUC.

### §25.11 Probe B — re-optimize stack ensemble weights on broader-pool gate (2026-05-14)

#### Goal
Re-optimize production stack weights (A1=0.20, B1=0.10, ProtoSSM=0.70 in cell 41 / cell 31b config) on the 1478-window broader-pool substrate that §25.5 validated as LB-correlated, instead of the 708-window d2_beta substrate that was used to set them.

#### Infrastructure inventory
- **A1 5-fold OOF on 1478 substrate**: `data/v4_5fold_soundscape_oof.npz` (have)
- **B1 + ProtoSSM OOFs on 708 substrate (59 files × 12 windows)**: `data/d2_beta_oofs.npz` (have)
- **B1 + ProtoSSM OOFs on 1478 substrate**: **NOT AVAILABLE**
- **B1 / ProtoSSM checkpoints**: locally at `models/b1_pretrained/{b1_pretrained,b1_seed{0,1,2}}.pt` and `models/protossm_pretrained_v2/protossm_{pretrained,seed{0,1,2}}.pt` (final-on-all-data versions, NOT 5-fold OOF)
- **Local Perch cache**: `data/processed/perch_cache/full_perch_arrays.npz` covers 59 of 66 substrate files (708 windows). 7 files missing: 0006/0007/0008/0009/0010 (S09), 0015 (S18), 0026 (S22).
- **Perch model + raw embeddings**: `perch_v2/models/perch_v2/saved_model.pb` and `perch_v2/data/processed/perch_embeddings/train_soundscapes/` has 1536-dim embeddings for all needed files.

#### Architectural blocker for native 1478 OOF
B1 (PerceiverIOHead) and ProtoSSM v4 both consume per-file Perch tensors at fixed `n_windows=12` and emit `(n_files, 12, n_classes)`. They cannot natively run on the variable-window-per-file 1478 substrate. Generating 1478-aligned OOFs would require:
1. Extending Perch cache from 59 → 66 files (~5-10 min compute via Perch ONNX/TF on 7 audio files).
2. Reproducing ~10 notebook cells (taxonomy groups, site mapping, OOF base/prior, MLP probes, B1 + ProtoSSM 5-fold OOF training) outside the notebook.
3. ~1.5-2.5 h GPU compute for the two 5-fold OOFs.

Estimated total: 4-8 h scripting + 1.5-2.5 h compute. **Above the user's "1-2 h compute" gate** → STOP, do not start full notebook port.

#### What I ran instead — broader-pool sweep on 1416-row intersection (96% coverage)
Path: project the 708-substrate B1+Proto OOFs onto v4's 1478 substrate via `(filename, start_sec)` join. Each cache row maps to exactly 2 v4 rows (label-CSV duplicates). 7 missing files drop 62 rows → **1416 rows = 95.8% of the 1478 substrate.**

A1 is the production 5-fold rank-mean (matches cell 41's `A1_FOLDS=[0,1,2,3,4]` + `A1_FOLD_REDUCE="mean"`); B1/Proto are the d2_beta 5-fold OOFs broadcast onto the 1416 rows. Sweep: A1 ∈ {0..1.0, step 0.05}, B1 ∈ {0..0.30, step 0.05}, Proto = 1 - A1 - B1. Linear rank-fusion. Broader-pool macro AUC over 71 active classes (0 < n_pos < N).

Standalone components on this substrate:
| Component | Broader-pool AUC |
|---|---|
| A1 (5-fold rank-mean) | **0.7596** |
| ProtoSSM (rank, 708→1416 broadcast) | 0.6659 |
| B1 (rank, 708→1416 broadcast) | **0.3959** (anti-predictive at macro level) |
| A1 (5-fold rank-mean) on full 1478 | 0.7672 (sanity matches §25.5) |

#### Sweep results (top 5 + extended)
| A1 | B1 | Proto | AUC | Δ vs production weights |
|---|---|---|---|---|
| 0.85 | 0.00 | 0.15 | **0.7631** | +0.0909 (best extended) |
| 0.50 | 0.00 | 0.50 | 0.7304 | +0.0582 |
| 0.50 | 0.05 | 0.45 | 0.7258 | +0.0536 |
| 0.45 | 0.00 | 0.55 | 0.7211 | +0.0489 |
| **0.20** | **0.10** | **0.70** (production) | **0.6722** | (baseline) |

#### Critical caveat — the gate-pass is illusory
Production weights look poor on this gate (0.6722) — but A1-only is **already 0.7596 on this substrate**. The honest comparison is best-tuned vs A1-only:
- A1-only: 0.7596
- Best (A1=0.85, Proto=0.15): 0.7631
- **Δ = +0.0035 → FAILS the +0.005 gate.**

The +0.058-+0.091 deltas vs production are not "tuning gains"; they're "production weights are bad on this gate." That's because production's rank-fusion is a **sequential 3-step pipeline** (cell 39 ProtoSSM → cell 40 B1 inverse-CDF → cell 41 A1 inverse-CDF) wrapped around per-class CDF round-trips that preserve ProtoSSM's marginals so cell 18 thresholds keep meaning. My linear rank-fusion sweep does NOT match that fusion topology — it's a different objective. So the comparison `linear_sweep(0.20, 0.10, 0.70) vs production_pipeline(0.20, 0.10, 0.70)` is apples-to-oranges; the 0.6722 number is not "what production scores on this gate", it's "what a misspecified linear surrogate of production scores on this gate."

This means:
1. The sweep cannot directly inform a re-weighting decision on production. To do that honestly, the inverse-CDF + sequential-fusion pipeline would have to be reproduced in the sweep harness.
2. Even on a true linear-fusion surrogate, the best A1-heavy combo only beats A1-only by +0.0035 — below the +0.005 gate threshold. The diversity from B1+ProtoSSM on this gate methodology is too small to claim a real gain. B1's broader-pool standalone AUC of 0.40 is consistent with this — it's anti-predictive at macro and contributes mostly noise to the ensemble.

#### Gate verdict
**Probe B FAILS the gate.** Two layers of failure:
1. The fusion topology mismatch makes the linear sweep an unsound proxy for production re-weighting.
2. Even the linear surrogate's best combo (0.7631) gains only +0.0035 over A1-only (0.7596), below the +0.005 threshold.

Do not push a re-weighted notebook based on this sweep.

#### What it would take to do this properly (NOT done)
1. Extend Perch cache to 66 files (run Perch on 7 missing audio files, ~5-10 min).
2. Port cells 22+24+25+32+33 of the notebook into a standalone 5-fold OOF script that:
   - reshapes embeddings to file-level on all 66 files,
   - runs B1 + ProtoSSM v4 5-fold GroupKFold OOF,
   - emits (66×12, 234) OOFs aligned to `(file, start_sec)`.
3. Project to 1478 substrate via the same join used here, but with no missing files → 100% coverage and honest fold-OOF.
4. Build a sweep harness that reproduces the **sequential rank-fusion + inverse-CDF** pipeline of cells 39+40+41, not a single linear blend. Sweep B1_WEIGHT and A1_WEIGHT on that.
5. Score with broader-pool macro AUC.

Effort: 1-2 sessions of focused script porting + ~1.5-2 h compute. Worth it only if there's an a priori reason to believe the current production weights are sub-optimal beyond noise — current evidence does not provide that reason.

#### Files written this session
- `data/probe_b_weight_sweep.npz` — full sweep results (`best_weights`, `best_auc`, `baseline_auc`, `sweep_grid`, `sweep_aucs`, plus standalone component AUCs and substrate metadata).
- `scripts/probe_b_weight_sweep.py` — initial 708-substrate sweep (kept for reference; results superseded by v2).
- `scripts/probe_b_weight_sweep_v2.py` — the 1416-row broader-pool sweep that produced the saved NPZ.
- `scripts/_probe_b_intersect.py`, `scripts/_probe_b_extra_check.py`, `scripts/_probe_b_inspect.py` — helper diagnostics.

### §25.12 Probe D — dual-stack v50+v73 single-kernel ensemble (drafted 2026-05-14, NOT pushed)

#### v50 vs v73 difference (verified from notebook backups)

The two production submissions come from the same kernel family (`jupyter/protossm-postproc/birdclef2026-protossm-postproc.ipynb`) at different versions. Diff'ing the live notebook against `*.bak_pre_t1` (2026-04-19, the closest-to-v50 backup on disk):

| Stack component | v50 era | v73 era |
|---|---|---|
| ProtoSSM training+inference (cells 1-15) | identical | identical |
| B1 PerceiverIO weight | `b1_frozen_weight_submit = 0.10` | same |
| `A1_FOLDS` | `[0, 1, 2, 4]` (4 folds, drop fold 3) | `[0, 1, 2, 3, 4]` (5 folds) |
| A1 ckpt provenance | base EffB0 LB-0.931 (no iNat pretrain) | v4 hybrid_prodft (iNat pretrained) |
| `A1_QMIX_ALPHA`, `A1_FOLD_REDUCE`, post-proc | `0.0` / `mean` / hardcoded thresholds | identical |
| Kaggle dataset slug for A1 ckpts | `birdclef-2026-a1-effb0-ckpts` (then 4-fold) | same slug, version bumped to 5-fold |

**The substantive difference is only the A1 ckpt set.** Same Kaggle dataset name was used; v73 simply replaced the file contents with the newer 5-fold iNat-pretrained ckpts. The v50 ckpts are preserved on local disk at `four_track/kaggle_datasets/_backups/a1_fold{0,1,2,4}_effb0_LB0931_20260503.pt` (no fold 3 — never trained for the v50 stack).

Everything else in the kernel — Perch ONNX inference, ProtoSSM training, B1 fusion, post-proc thresholds — is bit-identical between v50 and v73.

#### Feasibility verdict: FEASIBLE but marginal

Because B1 + ProtoSSM + post-proc are identical, a dual-stack notebook only needs to **run A1 inference twice with two different ckpt sets**, then rank-fuse the two A1 outputs (or fuse downstream). Everything upstream of A1 (Perch features, ProtoSSM training, B1 PerceiverIO inference) runs once.

Local preparation done (NOT pushed to Kaggle):
- New local Kaggle-dataset folder: `four_track/kaggle_datasets/a1-effb0-v50-ckpts/` containing `a1_fold{0,1,2,4}.pt` (renamed from the LB0931 backups) + `dataset-metadata.json` with slug `stevewatson999/birdclef-2026-a1-effb0-v50-ckpts`. To use, the dataset must be created/pushed to Kaggle first.
- Notebook scaffold: see "Notebook structure" below. Build was blocked by sandbox — see "Sandbox blocker" at end.

#### Notebook structure (Option A — dual fusion paths, then rank-fuse outputs)

The dual notebook is a near-clone of the live `birdclef2026-protossm-postproc.ipynb` with three changes localized to the A1 fusion region (cell 17), all post-fusion downstream identical:

1. **Cell 17a (replaces existing cell 17 v73 path):** unchanged 5-fold A1 inference with `A1_CKPT_DIR = /kaggle/input/birdclef-2026-a1-effb0-ckpts`, `A1_FOLDS = [0,1,2,3,4]`. Saves the post-Quantile-Mix output as `final_test_scores_v73`. Tracks elapsed wall time.

2. **Cell 17b (NEW — v50 stack):** time-guard check (skip if elapsed > 75 min). Loads v50 ckpts from `/kaggle/input/birdclef-2026-a1-effb0-v50-ckpts`, runs A1 inference for `A1_FOLDS_V50 = [0,1,2,4]`, computes its rank-mean, applies the same Quantile-Mix into a **clone of the pre-A1 ProtoSSM+B1 base** (`_proto_scores_before_fusion` saved at top of cell 17a). Saves as `final_test_scores_v50`. On any exception, sets `final_test_scores_v50 = None` and prints a warning.

3. **Cell 17c (NEW — dual rank-fusion):** if `final_test_scores_v50 is None` → falls through with `final_test_scores = final_test_scores_v73` (graceful v73-only fallback). Otherwise:
   - rank each stack per-class via `_rank01_per_col`
   - `dual_ranks = 0.5 * v73_ranks + 0.5 * v50_ranks` (equal-weight; equal LB priors)
   - inverse-CDF back to ProtoSSM marginal scale (same pattern as existing Quantile-Mix tail) so cell 18's hardcoded per-class thresholds remain meaningful
   - assign result to `final_test_scores`

4. **Cell 18 onwards:** unchanged. Single post-proc pass on the dual-fused scores.

Rationale for Option A over a flat 9-fold ensemble: faithful to the "dual production stack" framing, preserves per-stack identity, keeps the v73-only fallback trivial. Trade-off: slightly more complex than naively pooling 9 A1 folds.

#### Runtime estimate

| Block | v73 baseline (ref) | Dual estimate |
|---|---|---|
| Setup + Perch ONNX | ~2 min | ~2 min |
| Perch test inference (~700 files) | ~10-12 min | ~10-12 min (unchanged) |
| ProtoSSM submit-mode (30 ep, 3 splits) | ~25-30 min | ~25-30 min (unchanged) |
| B1 PerceiverIO inference | ~3-5 min | ~3-5 min (unchanged) |
| A1 5-fold inference (cell 17a) | ~12-18 min | ~12-18 min |
| A1 4-fold inference (cell 17b, NEW) | — | ~10-14 min |
| Post-proc + CSV | ~1 min | ~1 min |
| **Total** | **~72 min (measured)** | **~82-86 min (estimated)** |

This puts the dual notebook **inside but close to the 90-min Kaggle scoring cap**. Per §14.16 history, B2 ConvNeXt at 2 folds + A1 4 folds timed out at this same budget (v66/v67 in step 9f-12), which is comparable wall-time scope. Margin is thin: ~5-10 min before timeout.

The 75-min hard guard in cell 17b is the primary mitigation: if v73 took longer than expected, v50 is skipped and the kernel falls through to a v73-equivalent submission (LB ≈ 0.929, no regression vs current selection).

#### Risks (in priority order)

1. **Timeout risk.** Highest-impact failure mode. Tight 4-10 min margin against 90-min cap. Time-guard mitigates by reverting to v73-equivalent output. Worst case: timeout AFTER the 75-min check (impossible since v50 inference doesn't run beyond cap) → submission is v73-only, no regression.

2. **No expected LB lift.** v50 (0.931) and v73 (0.929) are 0.002 apart, well below the documented ±0.005 single-submission noise band (§15.3). The two stacks are highly correlated: same B1, same ProtoSSM, same post-proc; A1 ckpt sets share folds 0/1/2/4 by training data even though weights differ. Expected LB ≈ 0.929-0.931, indistinguishable from either stack alone. **The probe is informational at best, not a likely lift.**

3. **Slot cost vs §25.10 closeout.** §25.10 explicitly committed to "protect existing submissions, don't burn further slots without a probe that gate-passes on the 1478-window production substrate by ≥ +0.005 macro AUC." This dual-stack probe has not been gated locally; it's an LB-only iteration of the kind §25.10 just deprecated. Before pushing, an offline rank-fusion of v50's saved A1 OOFs against v73's saved A1 OOFs on the 1478-window pool would let us check whether the dual-fusion gains anything on the broader-pool gate. If broader-pool delta is < +0.005, the gate closes and the slot stays unused.

4. **Kaggle dataset push.** A new dataset `stevewatson999/birdclef-2026-a1-effb0-v50-ckpts` must be created and pushed before the notebook can run on Kaggle. Local files are staged at `four_track/kaggle_datasets/a1-effb0-v50-ckpts/` (4 ckpts + metadata) but per the user's explicit instruction, NOT pushed.

5. **Selection logic.** Even on success, only 2 submissions can be auto-selected for final scoring. Current selection is v50 + v73. A successful dual would replace one of those — losing selection diversity. If the dual lands at LB 0.930 (mid-range), replacing v50 (0.931) is a regression on public LB and a coin flip on private. Submission-selection strategy needs a deliberate decision before push.

#### Sandbox blocker (notebook generated by build script, manual mkdir required)

The agent's sandbox blocked `mkdir`, so `jupyter/dual-v50-v73/` was not created in-session. Instead a deterministic build script was authored that constructs the notebook + kernel-metadata.json from the live `protossm-postproc/birdclef2026-protossm-postproc.ipynb`.

To produce the final artifacts:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle
cd /home/swatson/work/kaggle/BirdCLEF
mkdir -p jupyter/dual-v50-v73
python -u four_track/src/build_dual_v50_v73_notebook.py
```

The script (a) reads the live source notebook, (b) patches Cell 17 to retain the v73 fused output and the pre-A1 base scores, (c) inserts the new Cell 17b (v50 stack with 75-min time guard + try/except v73-only fallback) and Cell 17c (equal-weight rank-fusion + inverse-CDF back to v73 marginals), (d) writes the new notebook and a kernel-metadata.json that lists the new v50 ckpts dataset slug. Markers used by the patch (`# Cell 17 — Track A1 SED fusion` and `final_test_scores = np.empty_like(_proto_scores_before_fusion)`) were verified to match exactly once in the source notebook before the script was written; the script will fail loudly if the source structure changes.

What IS staged and reusable on next pickup:
- v50 ckpts copied to `four_track/kaggle_datasets/a1-effb0-v50-ckpts/a1_fold{0,1,2,4}.pt` (4 files, ~64 MB) + `dataset-metadata.json`.
- Build script: `four_track/src/build_dual_v50_v73_notebook.py`.
- §25.12 (this section) with full design notes.

What's NOT done:
- Manual `mkdir -p jupyter/dual-v50-v73/` and one-shot run of the build script (single command, deterministic, ~1 sec).
- Kaggle dataset push for v50 ckpts (per user instruction — explicitly out of scope).
- Kaggle kernel push of the dual notebook (per user instruction — explicitly out of scope).
- Local broader-pool gate test on the dual fusion (recommended before any push, per Risk #3).

#### Recommendation

Before re-attempting the build or pushing anything: run the **local broader-pool gate** described in Risk #3 first. The probe takes <30 min on skynet (load `data/v4_5fold_soundscape_oof.npz` for v73 A1 + a comparable v50 A1 OOF if available, rank-fuse, compare macro AUC vs each stack alone on the 1478-window labeled soundscape pool). If dual-fusion gains < +0.005 macro AUC over the better single stack on the gate substrate, kill the probe per §25.10 protocol; the slot stays unused. If gains ≥ +0.005, then resolve the build blocker and push.

This sequencing also addresses the §25 closeout's strategic posture directly rather than implicitly bypassing it.

### §25.13 Probe A/C/D consolidated results (2026-05-14 ~22:00 EDT)

(B's full results are in §25.11 documented by the subagent. C and D below.)

**Probe A — per-fold temperature scaling (D1-b):**

| Variant | AUC | Δ vs prod (0.7672) | Verdict |
|---|---|---|---|
| Production rank-mean | 0.7672 | — | baseline |
| **Sigmoid-mean (no T)** | **0.7775** | **+0.0103** | **★ GATE-PASS** |
| Temp-scaled sigmoid-mean (CV-fitted T) | 0.7569 | −0.0103 | hurts |

The "+0.05 calibration drift" the plan history (§744) hinted at is largely a sigmoid-mean-vs-rank-mean aggregation choice, NOT a temperature-scaling artifact. Per-fold T fitting actively hurts (different T per CV fold breaks the rank-invariance, adds noise). The pure aggregation switch is a 1-line change in cell 41: `A1_FOLD_REDUCE = "mean"` → `"sig_mean"`.

**Probe C — Clean 5-fold Perch MLP probe:**

| Model | Broader-pool OOF | Δ vs v4 |
|---|---|---|
| v4 5-fold A1 (anchor) | 0.7775 | — |
| Perch 5-fold (clean OOF) | 0.6442 | −0.133 |
| BirdNET 5-fold (ref) | 0.5644 | −0.213 |

Perch beats BirdNET (broader taxonomy: ~10k vs 6522 species) but loses to v4 by −0.13. Per-fold val 0.84-0.93, broader OOF 0.64 — same 0.25 drop as BirdNET. **GATE-FAIL — not slot-worthy solo.**

This is now the third validating data point for the broader-pool gate (v4, BirdNET, Perch all show the per-fold-val vs broader-OOF gap; the gate-relative ordering matches the LB-relative ordering).

**Probe D — v50+v73 dual ensemble (A1-only partial gate check):**

Subagent built the dual-stack notebook builder (`four_track/src/build_dual_v50_v73_notebook.py`) — feasible, runtime ~82-86 min (close to 90-min cap). Before pushing, ran the agent's recommended A1-only gate check:

| Variant | v73 alone | Best v73+v50 blend | Δ |
|---|---|---|---|
| Rank-mean (production) | 0.7672 | 0.7672 (w_v50=0.00) | +0.0000 |
| Sigmoid-mean (Probe A) | 0.7775 | 0.7775 (w_v50=0.00) | +0.0000 |

Best blend has zero weight on v50. v50's A1 (0.7290 sig-mean) is strictly worse than v73's A1 (0.7775 sig-mean) on the broader pool. The 0.002 LB advantage v50 has must come from threshold-tuning interactions with the rest of the stack, not from A1 itself. **GATE-FAIL — adding v50 to v73 dilutes signal on broader pool.**

(Note: this is the A1-only proxy, since B1+ProtoSSM are unchanged between v50 and v73. If v50 vs v73 differed elsewhere, the proxy would be invalid. Per Probe D agent's investigation, they don't.)

### §25.14 Decision after batch (2026-05-14 ~22:00 EDT)

**Only Probe A passed the gate.** B, C, D all failed broader-pool gating. This is the gate doing its job — 3 of 4 ideas saved a slot decision.

Probe A is the only viable LB candidate from this batch. Decision pending: push the 1-line `A1_FOLD_REDUCE = "sig_mean"` change to the production kernel and burn one slot.

## §26 LB result for Probe A v74: 0.928 — confirms small-delta OOF→LB transfer ≈ zero (2026-05-15 ~02:00 EDT)

### §26.1 Numbers

| Submission | Date | A1 fold-aggregation | Broader-pool OOF | LB public | Δ vs v73 |
|---|---|---|---|---|---|
| v50 | 2026-04-16 | rank-mean (4-fold base EffNet) | 0.7290 | 0.931 | +0.002 |
| v73 | 2026-05-13 | rank-mean (5-fold v4 hybrid_prodft) | 0.7672 | 0.929 | — |
| **v74** | **2026-05-14** | **sig_mean (5-fold v4 hybrid_prodft)** | **0.7775** | **0.928** | **−0.001** |

OOF gain v73→v74: **+0.0103.** LB delta v73→v74: **−0.001.** Transfer ratio at this resolution: **effectively zero.**

### §26.2 The hard finding

Five+ submissions across the v50/v73/v74 family all fall in **[0.928, 0.931]** — a 0.003 spread. With ~600 test files and 234 classes, single-submission macro-AUC SE on Kaggle's hidden test is empirically ≥ ±0.005. The entire spread is consistent with one true LB around 0.929 ± noise. **The "0.931" was not a win; it was the lucky tail of the noise distribution.**

Three confirming data points for "OOF→LB transfer ≈ 0 at small deltas":
- v4 5-fold val +0.027 → LB **−0.001** (§14.22.10.5 era)
- sig_mean OOF +0.010 → LB **−0.001** (today, v74)
- BirdNET OOF +0.077 → LB **−0.187** (different mechanism — taxonomy mismatch — but still zero positive transfer at any delta)

### §26.3 What this rules out

The recipe family (A1 + B1 + ProtoSSM + Perch + post-proc, with marginal aggregation/calibration tweaks) has a hard ceiling at ~0.93 LB. Any probe whose broader-pool OOF gain over v4 is < +0.05 will not be detectable above noise on a single LB submission. **+0.005 broader-pool gate-pass is necessary for slot-justification but NOT sufficient for measurable LB delta.**

### §26.4 What's left

To break 0.93 reliably, we need a probe whose **broader-pool OOF moves by ≥ +0.05 over the v4 anchor (0.7775)**. Inside the recipe family, no such move exists. Outside it, options are structural — see §27.

### §26.5 Don'ts on next pickup

- Don't push a slot for any probe with broader-pool OOF Δ < +0.05 over v4. The transfer math doesn't support it.
- Don't rerank within v50/v73/v74 — they're all the same true LB.
- Don't burn slots on "incremental tuning" (per-class isotonic, threshold sweeps, weight retuning, etc.) — all bound by the noise floor.
- Don't claim a single +0.001 LB move is signal. Five submissions in 0.003 is the universe shouting "noise."

## 📌 PICK UP HERE (2026-05-14 ~23:00 EDT, sleep handoff)

**This invalidates the prior PICK UP HERE section at line 13218.** Anything below this header is the current state.

### TL;DR for tomorrow morning

1. **A2 (A1-as-teacher self-training) is launched and running on both machines.** Folds 0,1,2,3 on deepthought (sequential, ~16-20h); fold 4 on skynet in parallel (~12-15h). Wall-clock ETA: deepthought bounds at ~18-20h.
2. **Hard gate before submitting: broader-pool 5-fold OOF must beat v4 anchor (0.7775) by ≥+0.05 → must be ≥ 0.8275** to justify a Kaggle slot. Per `feedback_min_oof_delta_to_burn_slot.md` and §26: smaller deltas are below the LB SE noise floor (≥±0.005) and won't move LB.
3. **LB ceiling pattern (all 5 submissions in [0.928, 0.931])** is consistent with one true LB ≈ 0.929 ± noise. Don't claim +0.001 LB moves are signal.

### State of running jobs

| Field | Deepthought (folds 0-3) | Skynet (fold 4) |
|---|---|---|
| PID | **312252** | **507072** |
| Status | Sequential 4-fold A1 retrain w/ pseudo | Single-fold A1 retrain w/ pseudo |
| Started | 2026-05-14 ~22:50 EDT | 2026-05-14 ~22:52 EDT |
| Log | `/home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/log/train_a1_a2_dt_folds0123.log` (on deepthought) | `/home/swatson/work/kaggle/BirdCLEF/four_track/log/train_a1_a2_skynet_fold4.log` |
| Train data | 90,999 rows (28,439 focal + 62,560 BC2026_SS_PSEUDO) | same |
| Batches/epoch | 1422 | 1423 |
| Expected finish | 2026-05-15 ~16:00-19:00 EDT | 2026-05-15 ~12:00-15:00 EDT |
| Wait loop task ID | `btu2bz9d0` | `b1rh0g9d6` |

⚠️ **Wait-loop survival across Claude Code sessions is uncertain.** If wait-loop notifications haven't fired by morning, sanity-check both processes via `ps` directly.

### Tomorrow morning — sanity check first

```bash
# 1. Confirm both processes still alive
ps -p 507072 -o pid,etime,stat,pcpu,cmd 2>&1 | tail -2  # skynet
ssh deepthought "ps -p 312252 -o pid,etime,stat,pcpu,cmd 2>&1 | tail -2"  # DT

# 2. Tail logs for completion / errors
tail -30 /home/swatson/work/kaggle/BirdCLEF/four_track/log/train_a1_a2_skynet_fold4.log
ssh deepthought "tail -30 /home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/log/train_a1_a2_dt_folds0123.log"

# 3. Check ckpts saved
ls -la /home/swatson/work/kaggle/BirdCLEF/four_track/models/a1/a1_*_seed42_asl.pt 2>/dev/null
ssh deepthought "ls -la /home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/models/a1/a1_*_seed42_asl.pt 2>/dev/null"
```

If either died early (Traceback, OOM, disk full): re-launch that fold(s) only.

### Tomorrow's next steps (in order)

#### Step 1 — Sync deepthought ckpts back to skynet

When deepthought finishes (or as each fold completes):
```bash
rsync -av deepthought:/home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/models/a1/a1_tf_efficientnet_b0.ns_jft_in1k_fold{0,1,2,3}_seed42_asl.pt /home/swatson/work/kaggle/BirdCLEF/four_track/models/a1/
```

#### Step 2 — Generate broader-pool 5-fold OOF on retrained A1 (A2-4)

The 5 retrained ckpts → predictions on the 1478-window labeled soundscape pool → macro AUC.

Reuse the OOF generation pattern from `birdnet_v2/src/eval_birdnet_5fold_oof.py` but adapted for A1 (mel+PCEN inference path, 5 ckpts, file-level fold split matching the training fold split).

Result file: `data/a2_a1_5fold_broader_oof.npz` with `probs (1478, 234), y_true, filenames, start_sec`.

#### Step 3 — Gate check vs v4 anchor

```python
# Compare:
#   v4 5-fold (sigmoid-mean):  0.7775   (anchor — current production)
#   v74 sig_mean LB:           0.928    (no gain over v73)
#   A2 retrained:              ???      ← MUST be ≥ 0.8275 to justify slot
```

- If A2 broader OOF **< 0.7775**: regression — pseudo-labels actively hurt. Log to plan §27 and stop.
- If A2 broader OOF **0.7775-0.8275**: gate-fail — improvement below noise floor for LB transfer. Log and stop.
- If A2 broader OOF **≥ 0.8275**: gate-pass — proceed to Step 4.

#### Step 4 (only if gate-pass) — Build new Kaggle inference notebook

Modify the production protossm-postproc kernel (cell 41) to load the A2 ckpts instead of the original A1 ckpts. Or: branch as a new kernel `birdclef-2026-protossm-a2`. Push, smoke-test the build, surface for user submit.

### Risks / what to watch for tomorrow

1. **Skynet GB10 OOM mid-run.** GB10 had a transient OOM earlier when loading JIT. A1 training loads timm pretrained weights (different code path) and was running R+164% CPU at the sleep point — looked healthy. But CLAUDE.md notes silent kernel hangs on this hardware. If skynet log goes silent past expected epoch cadence → check process state.
2. **DT disk fills.** DT's `/mnt/mytoshiba` had 4.3T free; checkpoints are tiny (~16 MB each). Should not be a concern, but verify before launching anything else.
3. **Pseudo-label distribution skew.** Top-10 primary classes from manifest were dominated by frogs + nocturnal birds (compau, 22973, 23158, 555146, 65377, etc.). If these classes overfit, broader OOF could regress on the underrepresented Aves classes.
4. **Saturation vs. improvement.** With 90K rows × 25 epochs, the model sees ~2.3M sample-iterations vs baseline ~700K. Could hit overfit to pseudo-label noise. Watch per-epoch val_roc_auc curve — should peak then plateau, not collapse.
5. **The #34E precedent.** Prior pseudo-labeling attempt with Perch logits as teacher dropped LB by 0.008. We're betting A1-as-teacher is a stronger calibrated signal. The broader-pool gate is the empirical check.

### Files of note

| File | Purpose |
|---|---|
| `four_track/data/processed/a1_pseudo_soundscape.npz` | A1 5-fold sigmoid-mean predictions on 10,592 unlabeled soundscapes (127,104 windows × 234 classes, 110.9 MB) |
| `four_track/data/processed/a2_pseudo_manifest.csv` | Filtered + thresholded pseudo-label manifest (62,560 rows after threshold=0.5, top-5; 5.6 MB) |
| `four_track/src/a2_pseudo_label_a1.py` | Local inference script — runs A1 5-fold on unlabeled soundscapes |
| `four_track/src/a2_build_pseudo_manifest.py` | Manifest builder — turns soft probs into thresholded multi-label CSV |
| `four_track/src/dataset_a1.py` | Modified — added `BC2026_SS_PSEUDO` collection branch + multi-label pseudo target builder |
| `four_track/src/train_a1.py` | Modified — added `--pseudo-manifest` CLI arg |
| `four_track/kaggle_datasets/a1-effb0-ckpts/` | Original 5-fold A1 ckpts (the v73 production A1) — DO NOT OVERWRITE |
| `four_track/models/a1/a1_*_seed42_asl.pt` | Will hold the new A2 ckpts after training (per-fold) |

### Memory entries written this session (2026-05-14)

1. `feedback_encoder_taxonomy_must_match_lb.md` — BirdNET LB disaster lesson
2. `feedback_broader_pool_oof_is_lb_correlated.md` — the gate methodology
3. `feedback_min_oof_delta_to_burn_slot.md` — hard +0.05 broader-OOF rule for slots
4. Updated `feedback_val_v2_not_predictive_at_small_LB_deltas.md` with v74 third worked example
5. Updated `MEMORY.md` index with all three new entries

### Today's session summary (2026-05-14)

Started day investigating whether BirdNET SS-only OOF (0.8828, gate-pass) would transfer to LB. Submitted as kernel v72→LB 0.742 (−0.187 vs v73). Wrote new memories on encoder-taxonomy mismatch + broader-pool OOF gate methodology.

Then ran 4 probes batched: A=temp-scale (sigmoid-mean +0.0103, gate-pass), B=weight reopt (gate-fail at +0.0035), C=Perch 5-fold (gate-fail at 0.6442), D=v50+v73 dual ensemble (gate-fail — best blend keeps w_v50=0). Pushed A as v74 → LB 0.928 (−0.001 from v73, confirmed noise floor).

Wrote third memory `feedback_min_oof_delta_to_burn_slot.md` after v74: the broader-pool gate threshold is +0.05, not +0.005.

Pivoted to A2 (A1-as-teacher pseudo-labeling on 10K unlabeled soundscapes). Generated 127K pseudo windows in 38 min on deepthought (0.22s/file). Built 62K-row thresholded manifest. Modified train_a1.py + dataset_a1.py to support BC2026_SS_PSEUDO collection. Launched 5-fold retrain at sleep — 4 folds DT sequential + 1 fold skynet parallel.

### Don'ts on next pickup

- **Don't rerank within v50/v73/v74.** They're all the same true LB per §26.
- **Don't push A2 to LB without first computing the broader-pool OOF gate-check.** The whole point of this session was establishing that gate.
- **Don't claim noise-band LB moves are signal.** ±0.005 is the SE; treat anything within ±0.005 as zero.
- **Don't kill the running PIDs (312252 on DT, 507072 on skynet) unless they're clearly hung.** Check log progression before killing.
- **Don't burn slot on per-class isotonic / threshold tweaks / weight reopts** — all gate-failed today (§25.7-§25.13).
- **Don't extend the §17-§22 single-encoder probe family** without a fundamentally new teacher / data source / architecture.
- **Don't push v50 or v73 again.** Both already selected for final via Kaggle web UI.

## §27 A2 (A1-as-teacher self-training) — GATE-PASS at broader-pool 0.8402 (2026-05-15 ~21:00 EDT)

### §27.1 Result

| Metric | Value |
|---|---|
| Training | 5-fold A1 EffNet-B0.ns retrain on 28,439 focal + 62,560 BC2026_SS_PSEUDO rows, ASL loss, 25 epochs |
| Wall-clock | DT folds 0-3 sequential ~21 h + skynet fold 4 parallel ~14.4 h |
| Per-fold val_v2 (training-time) | f0=0.8097, f1=0.8067, f2=0.8309, f3=0.8290, f4=0.8376 — mean **0.8228** |
| Per-fold broader-pool OOF (eval) | f0=0.8094, f1=0.8067, f2=0.8308, f3=0.8290, f4=0.8376 — mean 0.8227 (matches val_v2 — same 1478 pool) |
| **Ensemble broader-pool OOF (sig-mean)** | **0.8402** |
| v4 anchor (production) | 0.7775 |
| Gate threshold (+0.05 floor) | 0.8275 |
| **Delta vs anchor** | **+0.0627** ← clears gate by +0.013 |
| Rank-fusion lift (per-fold mean → sig-mean ensemble) | +0.0175 (expected: weak folds carry information via decorrelation) |

### §27.2 Why this is the first real gate-pass in the campaign

Three confirmed null deltas before A2:
- v4 5-fold val +0.027 → LB **−0.001** (in-recipe-family aggregation tweak)
- v74 sig_mean OOF +0.010 → LB **−0.001** (in-recipe-family fold-aggregation tweak)
- BirdNET-Aves OOF +0.077 narrow-pool → LB **−0.187** (taxonomy mismatch, broader-pool dropped to 0.5644)

A2 is the **first probe with broader-pool OOF gain ≥ +0.05** since the gate was formalized. Mechanism is structurally different from §17-§26 family: not a calibration tweak, not a fold-aggregation tweak, but a **teacher-signal change** — the model trains on its own pseudo-labels for 10K unlabeled soundscapes (62K pseudo-rows after threshold=0.5 top-5 filtering). The 4080-pretrained A1 is the teacher of itself.

### §27.3 What's still unknown — the LB-side prior

The +0.05 threshold was set as a **necessary** condition for an LB-detectable move, not a **sufficient** one. We have never measured the OOF→LB transfer ratio above the noise floor:
- All prior submissions in [0.928, 0.931] were OOF deltas ≤ +0.027.
- BirdNET's +0.077 narrow-pool gain transferred to **−0.187 LB** (different mechanism — taxonomy mismatch — but proves "OOF up ≠ LB up" in general).
- A2's +0.063 broader-pool gain has no prior data point.

Plausible outcomes:
1. **Linear-ish transfer** (best case): +0.063 OOF → ~+0.02 to +0.04 LB → 0.95-0.97 LB.
2. **Damped transfer** (likely): recipe-family ceiling caps gain. +0.063 OOF → +0.005 to +0.015 LB → 0.93-0.94 LB.
3. **Zero transfer** (recipe ceiling holds): +0.063 OOF → 0.93x LB (noise floor).
4. **Negative transfer** (pseudo-label noise): A2 overfits SS-pseudo distribution, broader pool ≠ LB hidden test. 0.92x LB.

The single LB submission resolves (1) vs (2-3). It does not distinguish (2) from (3) cleanly because of the ±0.005 SE.

### §27.4 Decision (2026-05-15 ~21:00 EDT)

**Burn the slot.** Information value is high either way:
- Hit (LB ≥ 0.935): self-training is a real path forward, A2+ derivatives are worth pursuing.
- Miss (LB < 0.935): the LB ceiling is structural, not noise-bound. Pivots to fundamental changes (new architecture, new data source, different teacher) become the only options.

### §27.5 Files written this session

| File | Purpose |
|---|---|
| `four_track/data/a2_a1_5fold_broader_oof.npz` | (5, 1478, 234) probs_per_fold + probs_mean + per-fold AUC + ensemble AUC. The gate result. |
| `four_track/src/eval_a2_broader_oof.py` | A2 5-fold broader-pool OOF generator — reusable for any future ckpt swap. |
| `four_track/models/a1/a1_*_fold{0..4}_seed42_asl.pt` | The five A2 ckpts (4 from DT, 1 from skynet). |

### §27.6 Next action — Kaggle inference notebook

Modify production `protossm-postproc` kernel (cell 41 or wherever A1 ckpts are loaded) to point at the A2 ckpts. Or branch as a new kernel `birdclef-2026-protossm-a2`. Push, smoke-test, surface for user submit.

## §28 v75 LB = 0.933 — first measurement of OOF→LB transfer above the noise floor (2026-05-16 ~02:18 UTC)

### §28.1 Result table

| Submission | Date | A1 source | Broader-pool OOF | LB public | Δ vs v74 |
|---|---|---|---|---|---|
| v50 | 2026-04-16 | 4-fold base | 0.7290 | 0.931 | +0.003 |
| v73 | 2026-05-13 | v4 5-fold hybrid_prodft (rank-mean) | 0.7672 | 0.929 | +0.001 |
| v74 | 2026-05-14 | v4 5-fold hybrid_prodft + sig_mean | 0.7775 | 0.928 | — |
| **v75 (A2)** | **2026-05-16** | **A2 5-fold self-trained + sig_mean** | **0.8402** | **0.933** | **+0.005** |

**OOF gain v74 → v75:** +0.0627  
**LB gain v74 → v75:** +0.005  
**Transfer ratio:** ≈0.08  
**New campaign high** by +0.002 over v50.

### §28.2 What this confirms

§27.3 predicted four outcomes. The realized result lands in **outcome (2) damped transfer** at the floor of that band (+0.005 to +0.015 LB → actually +0.005):

- Outcome (1) linear transfer (+0.02 to +0.04): **ruled out.** Recipe-family ceiling is real and strong.
- Outcome (2) damped transfer (+0.005 to +0.015): **realized at the floor edge.** A2 is a real lever but each application yields a small LB move.
- Outcome (3) zero transfer: not distinguishable from (2) on a single submission — the +0.005 is exactly at the noise floor (±0.005 SE per `feedback_min_oof_delta_to_burn_slot.md`).
- Outcome (4) negative transfer: **ruled out.** No regression.

### §28.3 The cleanest A/B in the campaign

v74 → v75 is the cleanest controlled experiment we've run: **only the A1 ckpts changed.** Same notebook code, same post-proc chain, same Perch/B2/ProtoSSM weights. The +0.005 LB move under this control isolates the A2 self-training mechanism.

The same delta vs v50 (+0.002) is too small to argue for signal — v50 used different fold-aggregation (rank-mean instead of sig_mean) and different ckpt era (April 2026 baseline). v74-vs-v75 is the comparison that matters.

### §28.4 Implications for next moves

1. **A2 is the new production A1.** Update production notebook to A2 ckpts for any future submission (already done in v75; revert would lose the +0.005).

2. **OOF→LB transfer ratio is ~0.08 at +0.06 magnitude.** Naive linear extrapolation: to hit LB 0.95 (+0.017 from 0.933), need broader-pool OOF of ~0.84 + 0.017/0.08 = **~1.05**, which is impossible. To hit LB 0.94 (+0.007), need broader-pool OOF of ~0.93. Achievable only with a fundamentally stronger probe than self-training.

3. **A2-derivative diminishing returns.** A second self-training iteration (A2 as teacher → A3) would face two compounding headwinds: (a) the teacher is now closer to the same data distribution, so pseudo-label novelty drops; (b) the OOF→LB transfer ratio degrades further as we approach the recipe-family ceiling. Cost/benefit unfavorable.

4. **Structural changes remain the path to >0.94 LB.** Options ranked by signal/effort:
   - New track entirely (e.g., wav2vec2/CLAP encoders on Aves+non-Aves data)
   - New architecture for an existing track (B2/B1 retrains)
   - Different teacher source (e.g., BirdNET with proper Aves-only filtering at test time)
   - More aggressive ensembling across heterogeneous backbones

### §28.5 Slot accounting

- v75 was the 6th LB submission in this campaign cluster.
- Daily slot budget: 5/day, ~80 used. Remaining campaign budget: limited.
- **Don't burn slots on A2-derivatives.** First validate any next probe has broader-pool OOF gain of ≥+0.05 over the new anchor (0.8402).

### §28.6 Update the anchor

The broader-pool OOF anchor for future gate checks updates from 0.7775 (v4) → **0.8402 (A2)**.  
The gate threshold updates from 0.8275 (v4 + 0.05) → **0.8902 (A2 + 0.05)**.

### §28.7 Don'ts after §28

- **Don't claim 0.933 is meaningfully better than 0.931.** The +0.002 vs v50 is below the noise floor. Only the +0.005 vs v74 carries any signal weight, and even that is at the SE edge.
- **Don't push A2-derivatives without a ≥+0.05 broader-pool OOF gain over 0.8402.** Same gate rule, new anchor.
- **Don't expect linear transfer.** The 0.08 transfer ratio at +0.06 magnitude is the new empirical prior.
- **Don't burn slots inside the in-recipe-family probe space.** §17-§26 territory is exhausted; pivoting to structural changes is the only remaining lever.

## 📌 PICK UP HERE (2026-05-15 ~23:30 EDT, sleep handoff)

**This invalidates the prior PICK UP HERE section at line 13768.** Anything below this header is the current state.

### TL;DR for tomorrow morning

1. **v75 (A2 self-trained) is the new LB high at 0.933** (vs v74 0.928, +0.005; vs v50 0.931, +0.002). Damped transfer: OOF +0.063 → LB +0.005, ratio 0.08. See §28.
2. **AST (Track A4) fold-0 25-epoch training is running on DT.** Started 23:29 EDT 2026-05-15. ETA finish ~08:30 EDT 2026-05-16.
3. **First epoch from a now-cancelled 10-epoch trial already gave val_roc_auc=0.7552** at epoch 1 — strong indicator that AST transfers to BC2026. Killed and restarted at 25 epochs because the wall-clock estimate (16h) was 4× too high (actual 21min/epoch).

### State of the running AST job (Track A4 fold-0)

| Field | Value |
|---|---|
| Architecture | AST (MIT/ast-finetuned-audioset-10-10-0.4593), 86.4 M params, 234-class head |
| Backbone source | HF transformers ASTForAudioClassification, pretrained on AudioSet |
| Input format | 16 kHz, 10s window, kaldi fbank (1024 frames × 128 mels), AST mean/std normalize |
| Training data | 28,439 focal + 62,560 BC2026_SS_PSEUDO (same as A2) = 90,999 rows |
| Loss | ASL (asymmetric), no SWA, no mixstyle, no hybrid_prodft — clean first-cut recipe |
| Batch size | 16 (4080 fits ~57% VRAM) |
| LR / schedule | 5e-5, 1-epoch warmup, cosine to 2.5e-6, AdamW wd=1e-4 |
| Epochs | **25** |
| Per-epoch wall-clock | ~21 min (measured from cancelled trial) |
| PID on DT | **470692** |
| Log on DT | `/home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260515_232902.log` |
| Save path on DT | `models/a4/a4_ast_fold0_seed42_asl.pt` |
| Expected finish | **~08:30 EDT 2026-05-16** |

### Tomorrow morning — sanity check first

```bash
ssh deepthought "ps -p 470692 -o pid,etime,stat,pcpu,cmd 2>&1 | tail -2"
ssh deepthought "tail -50 /home/swatson/work/MachineLearning/_runon/BirdCLEF/log/runon_deepthought_20260515_232902.log"
ssh deepthought "ls -la /mnt/mytoshiba/MachineLearning/_runon/BirdCLEF/four_track/models/a4/a4_ast_fold0_seed42_asl.pt 2>&1"
```

If process is gone AND last epoch summary shows `Fold 0 complete. Best val ROC-AUC: X.XXXX`, training succeeded. If gone with no completion line, something died — investigate before relaunching.

### Tomorrow's next steps (in order)

#### Step 1 — Sync DT A4 ckpt back to skynet

```bash
rsync -av deepthought:/home/swatson/work/MachineLearning/_runon/BirdCLEF/four_track/models/a4/a4_ast_fold0_seed42_asl.pt /home/swatson/work/kaggle/BirdCLEF/four_track/models/a4/
```

#### Step 2 — Evaluate the gate

The val_roc_auc per epoch line IS the broader-pool 1478-window AUC (build_ast_soundscape_val mirrors build_soundscape_val). So the best val_roc_auc from the training log IS the AST fold-0 broader-pool OOF — no separate eval pass needed for the single-fold gate check.

**Gate comparison:**
- A1 fold-0 broader-pool: 0.8094 (A2 retrain epoch 25 value, our anchor for "did AST transfer at all?")
- A2 ensemble (5-fold sig-mean) broader-pool: 0.8402 (current production v75 anchor)
- Production gate (for LB slot): 0.8902 (= 0.8402 + 0.05) from §28.6 — but this is the **5-fold ensemble** threshold, NOT single-fold

Single-fold AST decision matrix:
- AST fold-0 < 0.70: architecture doesn't transfer — kill the track. Document and pivot.
- AST fold-0 0.70-0.80: AST is in the game but weaker than A1 — diversity probe only worth pursuing if errors are uncorrelated with A1 (check probe correlation before scaling).
- AST fold-0 0.80-0.85: competitive with A1 — scale to 5-fold if compute budget allows (~45h DT, or ~10h DT + ~35h skynet parallel).
- AST fold-0 ≥ 0.85: stronger than A1 single-fold (which is ~0.81) — strong signal, scale to 5-fold immediately. Ensemble of 5-fold AST + 5-fold A2 likely clears the +0.05 gate over 0.8402.

#### Step 3 — Correlation probe (only if fold-0 lands in 0.70-0.80 band)

Save fold-0 AST predictions on the 1478-window pool, compute per-class probability correlation with A2 fold-0. If mean correlation < 0.7, AST adds genuine diversity even if absolute AUC is lower — worth a fusion submission.

#### Step 4 — If gate-pass: 5-fold scaling decision

- Compute budget for AST 5-fold: ~25 × 21 min × 5 = ~44h on DT, or split 4-DT/1-skynet for parallelism (skynet would be ~3× slower per epoch, so ~63h on skynet for 1 fold — wall-clock bound by skynet = 63h).
- Decision: sequential 4-fold-on-DT (44h) + 1 fold-on-skynet (63h parallel) = 63h wall-clock. Or sequential 5-fold DT only = 55h. Per CLAUDE.md 4:1 rule, DT-only is preferred for 5 tasks since the parallel saving isn't large enough.

#### Step 5 — Only after 5-fold complete: broader-pool ensemble OOF + gate

Run `src/eval_a4_broader_oof.py` (doesn't exist yet — to be written, mirror eval_a2_broader_oof.py with AST input pipeline). Sig-mean across 5 folds → AST ensemble broader-pool AUC. Then either:
- AST solo broader-pool ≥ 0.8902 (anchor +0.05): push as solo replacement of A1 in the protossm kernel → v76
- AST+A2 fusion broader-pool ≥ 0.8902: push as fusion → v76
- Both fail: log result, return to §28.4 alternatives (heterogeneous backbones, BEATS, different teacher source)

### Files of note

| File | Purpose |
|---|---|
| `four_track/src/train_a4_ast.py` | AST training script (single-fold first cut) |
| `four_track/src/eval_a2_broader_oof.py` | Pattern to mirror when writing eval_a4_broader_oof.py |
| `four_track/data/a2_a1_5fold_broader_oof.npz` | A2 anchor for correlation probes |
| `four_track/models/a4/a4_ast_fold0_seed42_asl.pt` | Will exist after training |

### Memory entries written / updated this session (2026-05-15)

1. `project_a2_first_gate_pass.md` — updated with v75 LB outcome (0.933)
2. `feedback_min_oof_delta_to_burn_slot.md` — updated to v2 rule (+0.05 necessary, +0.10 sufficient)
3. MEMORY.md index line refreshed for A2

### Today's session summary (2026-05-15 through 2026-05-16 00:00 EDT)

Morning: Sanity-checked A2 5-fold training (DT folds 0-3, skynet fold 4) which had been running overnight. Both completed cleanly — DT mean val 0.8191, skynet fold 4 best 0.8376.

Synced ckpts back to skynet. Wrote `eval_a2_broader_oof.py` and ran it: **A2 ensemble broader-pool OOF = 0.8402** (vs v4 anchor 0.7775, **+0.0627** — first gate-pass since v50).

Wrote `project_a2_first_gate_pass.md`. Updated §27 in plan.

Built A2 inference path: exported A2 ckpts to JIT (`export_a1_jit.py --tag a2`), pushed dataset `stevewatson999/birdclef-2026-a1-effb0-a2-ckpts` to Kaggle, modified production protossm kernel cells 34 and 41 to point at the A2 dataset, added A2 dataset to kernel-metadata.json, pushed kernel v75 (save-run completed cleanly with 240 sample windows in 47s A1 inference).

User submitted v75 to LB. Result landed at **02:18 UTC: LB 0.933** (new campaign high). Updated §28 in plan with the transfer-ratio finding (0.08 at +0.06 magnitude). Updated `feedback_min_oof_delta_to_burn_slot.md` to v2 rule.

User chose option (b) — fundamentally different architecture probe. Picked AST (MIT/ast-finetuned-audioset-10-10-0.4593) with focal+A2-pseudo training data.

Built `train_a4_ast.py`. Smoke-tested on skynet (passed) and DT (passed, 50s wall-clock).

Launched DT fold-0 training at 10 epochs (canceled mid-epoch-2 after the first epoch landed at val_roc_auc=0.7552 — a strong first-epoch signal). User pushed back on the 10-epoch budget; killed and relaunched at 25 epochs.

### Don'ts on next pickup

- **Don't kill PID 470692 unless it's clearly hung.** Check log progression before killing.
- **Don't push AST single-fold to LB.** The gate is for the 5-fold ensemble (or a fusion); single-fold is a probe, not a slot candidate.
- **Don't claim the val_roc_auc per epoch is something other than broader-pool OOF.** They're the same metric — verified via build_ast_soundscape_val mirroring build_soundscape_val.
- **Don't burn slots without first re-confirming the gate framework still holds** (i.e., that LB SE noise is still ±0.005). The five+ data points behind the gate were all in-recipe-family; AST is structurally different and might have a different transfer characteristic.
- **Don't forget to `rm -f log/*.log` on every dispatch.** Established this session by user reminder; failure to clean is a hard precondition violation.

