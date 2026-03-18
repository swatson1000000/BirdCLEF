---
description: "Use when writing or editing training scripts, self-training, or pseudo-labeling. Covers BF16 setup, hyperparameters, epoch logging, validation strategy, and secondary label masking."
applyTo: "src/train*.py, src/self_train.py, src/pseudo_label.py"
---

# Training Guidelines

## BF16 Boilerplate (include in every training script)

```python
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True   # fixed input shapes

device = torch.device("cuda")
model = model.to(device)
model = torch.compile(model)            # fused kernels for GB10

autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
# BF16 does NOT need GradScaler
```

> `torch.compile` is for **training only**. Export un-compiled weights for ONNX. `torch.compile` + `torch.onnx.export` are incompatible in the same session.

## Key Hyperparameters (Stage 1)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, weight_decay=1e-4 |
| LR | 5e-4 → 1e-6, CosineAnnealingWarmRestarts (T_0=5 epochs) |
| Epochs | 15 (Stage 1), 25–35 (self-training) |
| Batch size | 64 |
| Folds | 5-fold stratified (MultilabelStratifiedKFold) |
| Seeds | 42, 123, 777 (multi-seed ensemble) |
| Precision | BF16 |

## Secondary Labels: Mask Loss to Zero

Set secondary label loss weight to **0** (not 0.5): species location within the clip is unknown — forcing a label at a specific 5s window adds noise. This gave +0.01 LB in BirdCLEF 2024.

```python
# secondary_mask: 0 for secondary label positions, 1 for primary
loss = (bce_loss * secondary_mask).mean()
```

## Epoch Logging Format (required)

Every epoch must output this summary line (implement via callback):

```
========================================
Epoch  N/15: train_loss=X.XXXX val_roc_auc=X.XXXX time=Xm XXs  YYYY-MM-DD HH:MM:SS ★ BEST
========================================
```

- Time as `Xm XXs` (not raw seconds) — use `divmod(elapsed, 60)`
- Append `★ BEST` when best validation ROC-AUC is achieved
- Always include `time.strftime('%Y-%m-%d %H:%M:%S')` so finish times can be estimated

## Validation

- Validate against **`data/raw/train_soundscapes_labels.csv`** (expert-labeled 5-second segments)
- This matches the competition metric domain (field recordings, not focal clips)
- Per-segment macro ROC-AUC = competition metric

## Pseudo-Labeling & Self-Training

- Pseudo-label unlabeled `train_soundscapes/` with Stage 1 ensemble
- Self-train with Noisy Student: mix focal clips + pseudo-labeled soundscape chunks at 0.5/0.5
- Apply power transform after each iteration: `pseudo_prob ** power` (e.g., 1.5 for sharpening)
- See [plan.md](../../plan.md) Phase 2–3 for full details
