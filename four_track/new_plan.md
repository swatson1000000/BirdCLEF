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
cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF/four_track
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
| `d0fc7119db` | P10 | 0.916 | n/a | no | **fail** (−0.015) — reverted |

