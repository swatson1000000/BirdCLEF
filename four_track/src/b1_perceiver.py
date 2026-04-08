"""Track B1 — PerceiverIO Perch consumer for the LB notebook.

This file is the canonical source for the three notebook cells that implement
Track B1 (a second Perch-features consumer trained on Kaggle alongside
ProtoSSM). Per `four_track/new_plan.md` §"B1 — concrete design", B1 exists
to add architectural diversity to the rank-fusion stack: it sees the same
inputs as ProtoSSM but uses them with a different inductive bias
(latent-bank cross-attention instead of bidirectional SSM scan + prototype
cosine), so its OOF errors should decorrelate from ProtoSSM's enough to
yield +0.002–0.004 LB.

Cell layout (all anchor-matched, no hardcoded indices):
  - CELL_24B → injected after cell 24 (ProtoSSM training defs).
                Defines `PerceiverIOHead`, `train_b1_perceiver_single`,
                `run_b1_perceiver_oof`. Mirrors cell 24's structure.
  - CELL_31B → injected after cell 31 (ProtoSSM instantiate + train).
                Runs B1 OOF on the same `file_groups` ProtoSSM used,
                logs `LOGS["oof_auc_b1"]` and the proto/B1 OOF correlation,
                then retrains a single `b1_model` on all soundscapes.
  - CELL_36B → injected after cell 36 (Score Fusion) and *before* cell 37
                (the A1 cell). Runs B1 inference on test, rank-fuses into
                `final_test_scores` via the same per-class CDF round-trip
                pattern as the A1 cell (preserves ProtoSSM's per-class
                marginals so the post-proc thresholds in cell 38 still
                mean what they meant before).

Runtime contract — what must already exist in the kernel before each cell:

  Cell 24b: torch, nn, F, time, CFG, N_CLASSES, N_WINDOWS, DEVICE, GroupKFold,
            macro_auc_skip_empty, mixup_files, focal_bce_with_logits.
            (All defined in cells 0–24.) No new dataset is mounted.

  Cell 31b: emb_files, logits_files, labels_files, site_ids_all, hours_all,
            file_groups, MODE, LOGS, oof_proto_flat (from cell 31), and
            everything cell 24b defined.

  Cell 36b: meta_test, emb_test, scores_test_raw, test_site_tensor,
            test_hour_tensor, final_test_scores, b1_model (from cell 31b),
            N_CLASSES.

Gates (enforced in cell 31b — see new_plan.md §B1):
  - oof_auc_b1_fused ≥ oof_auc_proto + 0.002 after sweeping B1_WEIGHT.
  - corrcoef(oof_proto_flat, oof_b1_flat) < 0.97 (else B1 is not adding
    diverse signal — kill).
  - Wall time stays under the 35-min ResidualSSM gate.
"""

# ---------------------------------------------------------------------------
# CELL 24b — PerceiverIO arch + training/OOF functions
# ---------------------------------------------------------------------------
CELL_24B = r'''
# Cell 24b — Track B1 PerceiverIO training (def + OOF)
#
# B1 = a second Perch-features consumer with a totally different inductive
# bias from ProtoSSM. Same inputs (emb, perch_logits, site_ids, hours) but
# routed through a learned latent bank (PerceiverIO) instead of a sequential
# SSM scan + prototype cosine, and Perch logits go in as a front-end feature
# instead of being mixed at the output via a per-class alpha.
#
# See four_track/new_plan.md §"B1 — concrete design" for the rationale and
# the OOF/correlation gates this branch must clear before any LB submission.

# --- Default config (added once; idempotent if cell is re-run) ---
CFG.setdefault("b1_perceiver", {
    "d_latent":       256,
    "n_latents":      16,
    "n_cross_layers": 2,
    "n_self_layers":  4,
    "n_heads":        8,
    "dropout":        0.3,
    "meta_dim":       16,
    "n_sites":        CFG["proto_ssm"]["n_sites"],
})
CFG.setdefault("b1_perceiver_train", {
    "n_epochs":        30,
    "lr":              3e-4,
    "weight_decay":    1e-2,
    "patience":        8,
    "pos_weight_cap":  CFG["proto_ssm_train"]["pos_weight_cap"],
    "focal_gamma":     CFG["proto_ssm_train"].get("focal_gamma", 2.0),
    "label_smoothing": 0.0,
    "mixup_alpha":     0.3,
    "swa_start_frac":  0.7,
    "distill_weight":  0.05,
    "oof_n_splits":    CFG["proto_ssm_train"].get("oof_n_splits", 5),
})


# --- PerceiverIO building blocks ---

class _PreNormCrossBlock(nn.Module):
    """Pre-norm cross-attention + FFN. q attends into kv."""
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        q_n  = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.attn(q_n, kv_n, kv_n)
        q = q + attn_out
        q = q + self.ffn(self.norm_ff(q))
        return q


class _PreNormSelfBlock(nn.Module):
    """Pre-norm self-attention + FFN."""
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm_a = nn.LayerNorm(d_model)
        self.attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_f = nn.LayerNorm(d_model)
        self.ffn    = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_n = self.norm_a(x)
        attn_out, _ = self.attn(x_n, x_n, x_n)
        x = x + attn_out
        x = x + self.ffn(self.norm_f(x))
        return x


class PerceiverIOHead(nn.Module):
    """B1 model: PerceiverIO over 12 Perch windows.

    Inputs (identical to ProtoSSM):
      emb           : (B, T=12, d_input=1536)   — Perch v2 embeddings
      perch_logits  : (B, T,    n_classes=234) — raw Perch class logits
      site_ids      : (B,)  long
      hours         : (B,)  long

    Output:
      species_logits : (B, T, n_classes) — per-window per-class logits
                       (matches ProtoSSM output shape so the rank-fusion
                       cell 36b can drop in alongside cell 36/37).
    """
    def __init__(self, d_input=1536, d_logits=234, d_latent=256, n_latents=16,
                 n_cross_layers=2, n_self_layers=4, n_heads=8,
                 n_classes=234, n_windows=12,
                 n_sites=20, meta_dim=16, dropout=0.3):
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.d_latent  = d_latent

        # 1. Input encoder — projects each window's (emb, perch_logits, site, hour)
        #    into the latent dim. Note we use Perch logits as a *front-end*
        #    feature, not an output gate (that's ProtoSSM's job).
        self.emb_proj    = nn.Linear(d_input,  d_latent)
        self.logits_proj = nn.Linear(d_logits, d_latent)
        self.site_emb    = nn.Embedding(n_sites, meta_dim)
        self.hour_emb    = nn.Embedding(24,      meta_dim)
        self.meta_proj   = nn.Linear(2 * meta_dim, d_latent)

        self.window_pos = nn.Parameter(torch.randn(1, n_windows, d_latent) * 0.02)
        self.input_norm = nn.LayerNorm(d_latent)
        self.input_drop = nn.Dropout(dropout)

        # 2. Learned latent bank
        self.latents = nn.Parameter(torch.randn(n_latents, d_latent) * 0.02)

        # 3. Cross-attn blocks: latents query into the 12 windows
        self.cross_blocks = nn.ModuleList([
            _PreNormCrossBlock(d_latent, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        # 4. Self-attn blocks among the latents (PerceiverIO "process" stack)
        self.self_blocks = nn.ModuleList([
            _PreNormSelfBlock(d_latent, n_heads, dropout)
            for _ in range(n_self_layers)
        ])

        # 5. Decoder: T learned query tokens cross-attend back into latents
        self.query_pos     = nn.Parameter(torch.randn(1, n_windows, d_latent) * 0.02)
        self.decoder_cross = _PreNormCrossBlock(d_latent, n_heads, dropout)
        self.decoder_norm  = nn.LayerNorm(d_latent)

        # 6. Per-window per-class classifier
        self.classifier = nn.Linear(d_latent, n_classes)

    def forward(self, emb, perch_logits, site_ids=None, hours=None):
        B, T, _ = emb.shape

        e = self.emb_proj(emb)
        l = self.logits_proj(perch_logits)
        x = e + l

        if site_ids is not None and hours is not None:
            s = self.site_emb(site_ids)            # (B, meta_dim)
            h = self.hour_emb(hours)               # (B, meta_dim)
            m = self.meta_proj(torch.cat([s, h], dim=-1))   # (B, d_latent)
            x = x + m[:, None, :]

        x = x + self.window_pos[:, :T, :]
        x = self.input_drop(self.input_norm(x))

        # Broadcast latent bank to batch
        latents = self.latents[None, :, :].expand(B, -1, -1).contiguous()

        for blk in self.cross_blocks:
            latents = blk(latents, x)
        for blk in self.self_blocks:
            latents = blk(latents)

        queries = self.query_pos[:, :T, :].expand(B, -1, -1).contiguous()
        out = self.decoder_cross(queries, latents)
        out = self.decoder_norm(out)

        species_logits = self.classifier(out)      # (B, T, n_classes)
        return species_logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_b1_perceiver_single(model, emb_train, logits_train, labels_train,
                              site_ids_train=None, hours_train=None,
                              emb_val=None, logits_val=None, labels_val=None,
                              site_ids_val=None, hours_val=None,
                              cfg=None, verbose=True):
    """Train one B1 PerceiverIOHead. Mirrors train_proto_ssm_single closely
    so the loss landscape is comparable; only the model body differs."""
    if cfg is None:
        cfg = CFG["b1_perceiver_train"]

    label_smoothing = cfg.get("label_smoothing", 0.0)
    mixup_alpha     = cfg.get("mixup_alpha", 0.0)
    focal_gamma     = cfg.get("focal_gamma", 0.0)
    swa_start_frac  = cfg.get("swa_start_frac", 1.0)
    n_epochs        = cfg["n_epochs"]
    swa_start_epoch = int(n_epochs * swa_start_frac)

    labels_np = labels_train.copy()
    if label_smoothing > 0:
        labels_np = labels_np * (1.0 - label_smoothing) + label_smoothing / 2.0

    has_val = emb_val is not None
    if has_val:
        emb_v    = torch.tensor(emb_val,    dtype=torch.float32)
        logits_v = torch.tensor(logits_val, dtype=torch.float32)
        labels_v = torch.tensor(labels_val, dtype=torch.float32)
        site_v   = torch.tensor(site_ids_val, dtype=torch.long) if site_ids_val is not None else None
        hour_v   = torch.tensor(hours_val,    dtype=torch.long) if hours_val    is not None else None

    # pos_weight from un-smoothed labels (matches ProtoSSM convention)
    labels_tr_t = torch.tensor(labels_np, dtype=torch.float32)
    pos_counts  = labels_tr_t.sum(dim=(0, 1))
    total       = labels_tr_t.shape[0] * labels_tr_t.shape[1]
    pos_weight  = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos",
    )

    best_val_loss = float("inf")
    best_state    = None
    wait          = 0
    history       = {"train_loss": [], "val_loss": [], "val_auc": []}

    swa_state = None
    swa_count = 0

    for epoch in range(n_epochs):
        # Mixup (after a 5-epoch warmup, matching ProtoSSM)
        if mixup_alpha > 0 and epoch > 5:
            emb_mix, logits_mix, labels_mix, _, _, _ = mixup_files(
                emb_train, logits_train, labels_np,
                site_ids_train, hours_train, None,
                alpha=mixup_alpha,
            )
        else:
            emb_mix, logits_mix, labels_mix = emb_train, logits_train, labels_np

        emb_tr    = torch.tensor(emb_mix,    dtype=torch.float32)
        logits_tr = torch.tensor(logits_mix, dtype=torch.float32)
        labels_tr = torch.tensor(labels_mix, dtype=torch.float32)
        site_tr   = torch.tensor(site_ids_train, dtype=torch.long) if site_ids_train is not None else None
        hour_tr   = torch.tensor(hours_train,    dtype=torch.long) if hours_train    is not None else None

        model.train()
        species_out = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)

        if focal_gamma > 0:
            loss_main = focal_bce_with_logits(
                species_out, labels_tr,
                gamma=focal_gamma,
                pos_weight=pos_weight[None, None, :],
            )
        else:
            loss_main = F.binary_cross_entropy_with_logits(
                species_out, labels_tr,
                pos_weight=pos_weight[None, None, :],
            )

        loss_distill = F.mse_loss(species_out, logits_tr)
        loss = loss_main + cfg["distill_weight"] * loss_distill

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch >= swa_start_epoch:
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_state:
                    swa_state[k] += model.state_dict()[k]
                swa_count += 1

        model.eval()
        with torch.no_grad():
            if has_val:
                val_out = model(emb_v, logits_v, site_ids=site_v, hours=hour_v)
                val_loss = F.binary_cross_entropy_with_logits(
                    val_out, labels_v,
                    pos_weight=pos_weight[None, None, :],
                )
                val_pred = val_out.reshape(-1, val_out.shape[-1]).numpy()
                val_true = labels_v.reshape(-1, labels_v.shape[-1]).numpy()
                try:
                    val_auc = macro_auc_skip_empty(val_true, val_pred)
                except Exception:
                    val_auc = 0.0
            else:
                val_loss = loss
                val_auc  = 0.0

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_auc"].append(val_auc)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            swa_info = f" swa={swa_count}" if swa_count > 0 else ""
            print(f"  B1 Epoch {epoch+1:3d}: train={loss.item():.4f} val={val_loss.item():.4f} "
                  f"auc={val_auc:.4f} lr={lr_now:.6f} wait={wait}{swa_info}", flush=True)

        if wait >= cfg["patience"]:
            if verbose:
                print(f"  B1 early stop at epoch {epoch+1} (best val_loss={best_val_loss:.4f})", flush=True)
            break

    if swa_state is not None and swa_count >= 3:
        if verbose:
            print(f"  B1 applying SWA (averaged {swa_count} checkpoints)", flush=True)
        avg_state = {k: v / swa_count for k, v in swa_state.items()}
        model.load_state_dict(avg_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"  B1 training complete. Best val_loss={best_val_loss:.4f}", flush=True)

    return model, history


def run_b1_perceiver_oof(emb_files, logits_files, labels_files,
                         site_ids_all, hours_all, file_groups,
                         cfg=None, verbose=True):
    """GroupKFold OOF for B1. Reuses ProtoSSM's `file_groups` so the splits
    are 1:1 identical and the proto/B1 OOF correlation is honest."""
    if cfg is None:
        cfg = CFG["b1_perceiver_train"]

    n_splits  = cfg.get("oof_n_splits", 5)
    n_files   = len(emb_files)
    b1_cfg    = CFG["b1_perceiver"]

    oof_preds      = np.zeros((n_files, N_WINDOWS, N_CLASSES), dtype=np.float32)
    fold_histories = []

    n_unique = len(set(file_groups))
    if n_unique < n_splits:
        print(f"  WARNING: only {n_unique} groups, reducing n_splits {n_splits}→{n_unique}", flush=True)
        n_splits = n_unique
    gkf = GroupKFold(n_splits=n_splits)
    dummy_y = np.zeros(n_files)

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(dummy_y, dummy_y, file_groups)):
        if verbose:
            print(f"\n--- B1 Fold {fold_i+1}/{n_splits} (train={len(train_idx)}, val={len(val_idx)}) ---", flush=True)

        fold_model = PerceiverIOHead(
            d_input=emb_files.shape[2],
            d_logits=N_CLASSES,
            d_latent=b1_cfg["d_latent"],
            n_latents=b1_cfg["n_latents"],
            n_cross_layers=b1_cfg["n_cross_layers"],
            n_self_layers=b1_cfg["n_self_layers"],
            n_heads=b1_cfg["n_heads"],
            n_classes=N_CLASSES,
            n_windows=N_WINDOWS,
            n_sites=b1_cfg["n_sites"],
            meta_dim=b1_cfg["meta_dim"],
            dropout=b1_cfg["dropout"],
        ).to(DEVICE)

        fold_model, fold_hist = train_b1_perceiver_single(
            fold_model,
            emb_files[train_idx], logits_files[train_idx],
            labels_files[train_idx].astype(np.float32),
            site_ids_train=site_ids_all[train_idx], hours_train=hours_all[train_idx],
            emb_val=emb_files[val_idx], logits_val=logits_files[val_idx],
            labels_val=labels_files[val_idx].astype(np.float32),
            site_ids_val=site_ids_all[val_idx], hours_val=hours_all[val_idx],
            cfg=cfg, verbose=verbose,
        )

        fold_model.eval()
        with torch.no_grad():
            val_emb    = torch.tensor(emb_files[val_idx],    dtype=torch.float32)
            val_logits = torch.tensor(logits_files[val_idx], dtype=torch.float32)
            val_sites  = torch.tensor(site_ids_all[val_idx], dtype=torch.long)
            val_hours  = torch.tensor(hours_all[val_idx],    dtype=torch.long)
            val_out    = fold_model(val_emb, val_logits, site_ids=val_sites, hours=val_hours)
            oof_preds[val_idx] = val_out.numpy()

        fold_histories.append(fold_hist)

    return oof_preds, fold_histories


print("B1 PerceiverIO functions defined (PerceiverIOHead, train, OOF).", flush=True)
_b1_test_model = PerceiverIOHead(
    d_input=1536, d_logits=N_CLASSES,
    **{k: v for k, v in CFG["b1_perceiver"].items() if k != "n_sites"},
    n_classes=N_CLASSES, n_windows=N_WINDOWS,
    n_sites=CFG["b1_perceiver"]["n_sites"],
)
print(f"B1 parameter count: {_b1_test_model.count_parameters():,}", flush=True)
del _b1_test_model
'''


# ---------------------------------------------------------------------------
# CELL 31b — instantiate, run OOF, log gates, retrain on full data
# ---------------------------------------------------------------------------
CELL_31B = r'''
# Cell 31b — Track B1 PerceiverIO instantiate + OOF + retrain on full
#
# Reuses everything cell 31 already built: emb_files, logits_files,
# labels_files, site_ids_all, hours_all, file_groups, oof_proto_flat.

_B1_T0 = time.time()

if MODE == "train":
    print("\n=== Track B1 — PerceiverIO OOF ===", flush=True)
    oof_b1_preds, b1_fold_histories = run_b1_perceiver_oof(
        emb_files, logits_files, labels_files,
        site_ids_all, hours_all, file_groups,
        cfg=CFG["b1_perceiver_train"],
        verbose=CFG["verbose"],
    )
    oof_b1_flat = oof_b1_preds.reshape(-1, N_CLASSES)
    y_flat      = labels_files.reshape(-1, N_CLASSES).astype(np.float32)

    overall_oof_auc_b1 = macro_auc_skip_empty(y_flat, oof_b1_flat)
    print(f"\nB1 OOF macro AUC (standalone): {overall_oof_auc_b1:.4f}", flush=True)
    print(f"ProtoSSM OOF macro AUC (recall): {LOGS.get('oof_auc_proto', float('nan')):.4f}", flush=True)

    # --- Diversity gate: proto/B1 OOF correlation ---
    _po = oof_proto_flat.flatten().astype(np.float64)
    _bo = oof_b1_flat.flatten().astype(np.float64)
    _po -= _po.mean(); _bo -= _bo.mean()
    _denom = (np.sqrt((_po * _po).sum()) * np.sqrt((_bo * _bo).sum())) + 1e-12
    proto_b1_corr = float((_po * _bo).sum() / _denom)
    print(f"Proto/B1 OOF Pearson correlation: {proto_b1_corr:.4f}  (gate: <0.97)", flush=True)

    # --- Sweep B1_WEIGHT on OOF (rank-space, mirrors A1 cell 37) ---
    def _rank01_per_col(mat):
        n = mat.shape[0]
        order = np.argsort(mat, axis=0, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float32)
        rows = np.arange(n, dtype=np.float32)
        for c in range(mat.shape[1]):
            ranks[order[:, c], c] = rows
        if n > 1:
            ranks /= (n - 1)
        return ranks

    _proto_ranks_oof = _rank01_per_col(oof_proto_flat.astype(np.float32))
    _b1_ranks_oof    = _rank01_per_col(oof_b1_flat.astype(np.float32))

    sweep = []
    for w in [0.00, 0.10, 0.15, 0.20, 0.25, 0.30]:
        fused_ranks = (1.0 - w) * _proto_ranks_oof + w * _b1_ranks_oof
        try:
            auc_w = macro_auc_skip_empty(y_flat, fused_ranks)
        except Exception:
            auc_w = 0.0
        sweep.append((w, auc_w))
        print(f"  B1_WEIGHT={w:.2f}  proto+b1 OOF AUC={auc_w:.4f}", flush=True)

    best_w, best_auc = max(sweep, key=lambda x: x[1])
    proto_only_auc = sweep[0][1]
    delta = best_auc - proto_only_auc
    print(f"\nBest B1_WEIGHT={best_w:.2f}  fused OOF AUC={best_auc:.4f}  "
          f"Δ vs proto-only={delta:+.4f}  (gate: ≥+0.0020)", flush=True)

    LOGS["b1"] = {
        "oof_auc_b1_standalone": float(overall_oof_auc_b1),
        "oof_auc_proto_only_rank":   float(proto_only_auc),
        "oof_auc_proto_b1_fused":    float(best_auc),
        "best_b1_weight":            float(best_w),
        "delta_vs_proto_only":       float(delta),
        "proto_b1_corr":             float(proto_b1_corr),
        "sweep":                     [(float(w), float(a)) for w, a in sweep],
        "wall_time_seconds":         float(time.time() - _B1_T0),
    }

    # --- Decide whether to retrain on full data ---
    _gate_diversity = proto_b1_corr < 0.97
    _gate_lift      = delta >= 0.002
    if _gate_diversity and _gate_lift:
        print("B1 OOF gates PASSED. Retraining on all soundscapes for test inference.", flush=True)
        B1_WEIGHT_FROZEN = best_w
    else:
        print("B1 OOF gates FAILED. Skipping test inference; rank fusion will be a no-op.", flush=True)
        print(f"  diversity gate (<0.97): {'PASS' if _gate_diversity else 'FAIL'}", flush=True)
        print(f"  lift gate     (≥0.002): {'PASS' if _gate_lift      else 'FAIL'}", flush=True)
        B1_WEIGHT_FROZEN = 0.0
    LOGS["b1"]["frozen_weight"] = float(B1_WEIGHT_FROZEN)
else:
    print("Submit mode: skipping B1 OOF (using last train-mode frozen weight).", flush=True)
    B1_WEIGHT_FROZEN = float(CFG.get("b1_frozen_weight_submit", 0.20))

# --- Final B1 model on ALL labeled soundscapes ---
print("\n--- Training final B1 model on all soundscapes ---", flush=True)
b1_cfg = CFG["b1_perceiver"]
b1_model = PerceiverIOHead(
    d_input=emb_files.shape[2],
    d_logits=N_CLASSES,
    d_latent=b1_cfg["d_latent"],
    n_latents=b1_cfg["n_latents"],
    n_cross_layers=b1_cfg["n_cross_layers"],
    n_self_layers=b1_cfg["n_self_layers"],
    n_heads=b1_cfg["n_heads"],
    n_classes=N_CLASSES,
    n_windows=N_WINDOWS,
    n_sites=b1_cfg["n_sites"],
    meta_dim=b1_cfg["meta_dim"],
    dropout=b1_cfg["dropout"],
).to(DEVICE)

b1_model, _b1_full_history = train_b1_perceiver_single(
    b1_model,
    emb_files, logits_files, labels_files.astype(np.float32),
    site_ids_train=site_ids_all, hours_train=hours_all,
    emb_val=None, logits_val=None, labels_val=None,
    cfg=CFG["b1_perceiver_train"], verbose=CFG["verbose"],
)
print(f"B1 cell 31b wall time: {time.time() - _B1_T0:.1f}s", flush=True)
'''


# ---------------------------------------------------------------------------
# CELL 36b — B1 test inference + rank fusion into final_test_scores
# ---------------------------------------------------------------------------
CELL_36B = r'''
# Cell 36b — Track B1 inference + rank fusion
#
# Runs the final b1_model on test, rank-fuses into final_test_scores via
# the same per-class CDF round-trip pattern as the A1 cell so ProtoSSM's
# per-class marginals are preserved (cell 38 thresholds keep their meaning).
#
# Order in the notebook:
#   cell 36  → ProtoSSM/MLP/prior fusion writes final_test_scores
#   cell 36b → B1 rank fusion (this cell) overwrites final_test_scores
#   cell 37  → A1 rank fusion overwrites final_test_scores
#   cell 38  → post-processing reads final_test_scores

_B1_INF_T0 = time.time()

if B1_WEIGHT_FROZEN <= 0.0:
    print("B1_WEIGHT_FROZEN=0 → skipping B1 test fusion (gate failed in cell 31b).", flush=True)
else:
    print(f"B1 test inference + rank fusion at w={B1_WEIGHT_FROZEN:.2f} …", flush=True)

    b1_model.eval()
    with torch.no_grad():
        _b1_test_out = b1_model(
            emb_test_tensor, logits_test_tensor,
            site_ids=test_site_tensor, hours=test_hour_tensor,
        )
    _b1_test_flat = _b1_test_out.reshape(-1, N_CLASSES).cpu().numpy().astype(np.float32)

    def _rank01_per_col(mat):
        n = mat.shape[0]
        order = np.argsort(mat, axis=0, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float32)
        rows = np.arange(n, dtype=np.float32)
        for c in range(mat.shape[1]):
            ranks[order[:, c], c] = rows
        if n > 1:
            ranks /= (n - 1)
        return ranks

    _proto_scores_before = final_test_scores.astype(np.float32).copy()
    _proto_ranks         = _rank01_per_col(_proto_scores_before)
    _b1_ranks            = _rank01_per_col(_b1_test_flat)
    _fused_ranks         = (1.0 - B1_WEIGHT_FROZEN) * _proto_ranks + B1_WEIGHT_FROZEN * _b1_ranks

    # Per-class inverse CDF — preserve ProtoSSM marginals
    _n_rows = _proto_scores_before.shape[0]
    _sorted_proto = np.sort(_proto_scores_before, axis=0)
    _idx = np.clip(
        (_fused_ranks * (_n_rows - 1)).round().astype(np.int64),
        0, _n_rows - 1,
    )
    final_test_scores = np.empty_like(_proto_scores_before)
    for _c in range(_proto_scores_before.shape[1]):
        final_test_scores[:, _c] = _sorted_proto[_idx[:, _c], _c]

    _abs_delta = float(np.abs(final_test_scores - _proto_scores_before).mean())
    print(f"  B1 fusion mean |Δ score|: {_abs_delta:.5f}", flush=True)
    print(f"  B1 cell 36b wall time   : {time.time() - _B1_INF_T0:.1f}s", flush=True)

    LOGS.setdefault("b1_fusion", {})
    LOGS["b1_fusion"].update({
        "weight": float(B1_WEIGHT_FROZEN),
        "mean_abs_delta": _abs_delta,
        "wall_time_seconds": float(time.time() - _B1_INF_T0),
        "score_range_after": [float(final_test_scores.min()), float(final_test_scores.max())],
    })
'''
