#!/usr/bin/env python3
"""
Train ProtoSSM locally using the same code as the Kaggle notebook.
Extracts cells from the notebook, patches paths for local execution,
and saves all artifacts to models/protossm_pretrained/.

Usage:
    conda activate kaggle
    cd /home/swatson/work/MachineLearning/kaggle/BirdCLEF
    python -u src/train_protossm_local.py
"""
import json
import sys
import os
import time
import re
from pathlib import Path

# Prevent TF from allocating GPU memory (causes segfault with PyTorch)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "jupyter" / "protossm" / "birdclef2026-protossm.ipynb"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "protossm_pretrained"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# --- Extract code from notebook ---
print(f"Extracting code from {NOTEBOOK_PATH}")
with open(NOTEBOOK_PATH) as f:
    nb = json.load(f)

cells = []
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cells.append("".join(cell["source"]))

print(f"Extracted {len(cells)} code cells")

# --- Build the training script from notebook cells ---
# We need cells 0-34 (training) but NOT 35-38 (test inference)
# Cell indices from the notebook: 0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,20,22,24,26,28,29,30,31,33,34

# Map cell indices to their position in the cells list
# (some cell indices are skipped in the notebook: 11, 19, 21, 23, 25, 27, 32)
# The cells list is sequential, so cell[0]=Cell0, cell[1]=Cell1, etc.

code_parts = []

for i, cell_code in enumerate(cells):
    # Skip cells 35-38 (test inference, submission, logging)
    # These are the last 4 code cells
    # Cell 35 starts with "test_paths = sorted"
    # Cell 36 starts with "Score Fusion"
    # Cell 37 starts with "Cell 18 — V17"
    # Cell 38 starts with "Cell 19 — Final Diagnostics"
    if "Cell 16 — Infer Perch on hidden test" in cell_code:
        print(f"  Skipping cell {i} (test inference) and all subsequent cells")
        break
    code_parts.append(cell_code)

print(f"Using {len(code_parts)} cells for training")

# --- Apply patches ---
full_code = "\n\n# --- CELL BOUNDARY ---\n\n".join(code_parts)

# 1. Remove pip install commands (we already have deps)
full_code = re.sub(r'^!pip install.*$', '# [PATCHED] pip install removed', full_code, flags=re.MULTILINE)

# 2. Patch paths: Kaggle -> local
full_code = full_code.replace(
    'Path("/kaggle/input/competitions/birdclef-2026")',
    f'Path("{PROJECT_ROOT}/data/raw")'
)
full_code = full_code.replace(
    '/kaggle/input/competitions/birdclef-2026',
    f'{PROJECT_ROOT}/data/raw'
)
full_code = full_code.replace(
    'Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")',
    f'Path("{PROJECT_ROOT}/perch_v2/models/perch_v2")'
)
full_code = full_code.replace(
    'Path("/kaggle/input/perch-meta")',
    f'Path("{PROJECT_ROOT}/data/processed/perch_cache")'
)
full_code = full_code.replace(
    'Path("/kaggle/working/perch_cache")',
    f'Path("{PROJECT_ROOT}/data/processed/perch_cache")'
)
full_code = full_code.replace(
    '/kaggle/input/birdclef-2026-cvlb-assets-0911/perch_v2_no_dft.onnx',
    f'{PROJECT_ROOT}/data/external/perch_v2_no_dft.onnx'
)
full_code = full_code.replace(
    '/kaggle/input/perch-meta/perch_v2_no_dft.onnx',
    f'{PROJECT_ROOT}/data/external/perch_v2_no_dft.onnx'
)
full_code = full_code.replace(
    '/kaggle/input/birdclef-2026-cvlb-assets-0911/wheels',
    f'{PROJECT_ROOT}/data/external/wheels'
)
full_code = full_code.replace(
    '/kaggle/working/',
    f'{PROJECT_ROOT}/data/processed/perch_cache/'
)

# 2b. Force TF to CPU only and neutralize TF/PyTorch conflict
# The segfault occurs during loss.backward() because TF's numpy behavior mode
# (tf.experimental.numpy.experimental_enable_numpy_behavior) interferes with
# PyTorch autograd. Since we only use TF for Perch loading (and we have cache),
# we disable it entirely.
full_code = 'import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"\n' + full_code

# Disable TF numpy behavior
full_code = full_code.replace(
    'tf.experimental.numpy.experimental_enable_numpy_behavior()',
    '# [PATCHED] TF numpy behavior disabled\n# tf.experimental.numpy.experimental_enable_numpy_behavior()'
)

# Replace TF import with a stub — we never use TF since we have Perch cache
full_code = full_code.replace(
    'import tensorflow as tf',
    '# [PATCHED] TF replaced with stub to prevent segfault in PyTorch backward\nimport types; tf = types.ModuleType("tf"); tf.__version__ = "STUB"'
)

# After Perch cache is loaded, aggressively clean up TF to free memory
full_code = full_code.replace(
    '_perch_infer = None\n    _perch_backend = "none (using cache)"',
    '_perch_infer = None\n    _perch_backend = "none (using cache)"\n    # Aggressively clean up TF\n    import gc; gc.collect()'
)

# 3. Force MODE = "train" (override notebook's "submit" default)
full_code = full_code.replace(
    'MODE = "submit"',
    'MODE = "train"'
)

# 4. Skip submit-mode safety caps (we want FULL V18 config for local training)
# The submit-mode caps block starts with 'if MODE == "submit":' after V18 CFG
# Since MODE = "train", these caps will be skipped automatically

# 5. Patch build_perch_inferencer to not fail when model files are missing
# (we have the cache, so we never actually call the inferencer)
full_code = full_code.replace(
    '_perch_infer, _perch_backend = build_perch_inferencer(MODEL_DIR, ONNX_PATH)',
    '''try:
    _perch_infer, _perch_backend = build_perch_inferencer(MODEL_DIR, ONNX_PATH)
except Exception as _e:
    print(f"WARNING: Perch model not available locally: {_e}")
    print("Will use cached embeddings instead.")
    _perch_infer = None
    _perch_backend = "none (using cache)"'''
)

# 5b. Add debug prints inside train_proto_ssm_single to find segfault location
full_code = full_code.replace(
    '    label_smoothing = cfg.get("label_smoothing", 0.0)',
    '    print("[TRAIN_DEBUG] Entered train_proto_ssm_single"); sys.stdout.flush()\n    label_smoothing = cfg.get("label_smoothing", 0.0)'
)
full_code = full_code.replace(
    '    optimizer = torch.optim.AdamW(',
    '    print("[TRAIN_DEBUG] Creating optimizer..."); sys.stdout.flush()\n    optimizer = torch.optim.AdamW('
)
full_code = full_code.replace(
    '    scheduler = torch.optim.lr_scheduler.OneCycleLR(',
    '    print("[TRAIN_DEBUG] Creating scheduler..."); sys.stdout.flush()\n    scheduler = torch.optim.lr_scheduler.OneCycleLR('
)
full_code = full_code.replace(
    '    for epoch in range(n_epochs):',
    '    print(f"[TRAIN_DEBUG] Starting training loop ({n_epochs} epochs)..."); sys.stdout.flush()\n    for epoch in range(n_epochs):'
)
full_code = full_code.replace(
    '        # === Train ===\n        model.train()',
    '        # === Train ===\n        if epoch == 0: print("[TRAIN_DEBUG] First epoch - model.train()..."); sys.stdout.flush()\n        model.train()'
)
full_code = full_code.replace(
    '        species_out, family_out, _ = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)',
    '        if epoch == 0: print("[TRAIN_DEBUG] First epoch - forward pass..."); sys.stdout.flush()\n        species_out, family_out, _ = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)'
)
full_code = full_code.replace(
    '        optimizer.zero_grad()\n        loss.backward()',
    '        if epoch == 0: print(f"[TRAIN_DEBUG] First epoch - loss={loss.item():.4f}, backward..."); sys.stdout.flush()\n        optimizer.zero_grad()\n        loss.backward()'
)
full_code = full_code.replace(
    '        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n        optimizer.step()',
    '        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n        if epoch == 0: print("[TRAIN_DEBUG] First epoch - optimizer.step()..."); sys.stdout.flush()\n        optimizer.step()'
)
full_code = full_code.replace(
    '        scheduler.step()\n\n        # === SWA accumulation ===',
    '        scheduler.step()\n        if epoch == 0: print("[TRAIN_DEBUG] First epoch completed successfully!"); sys.stdout.flush()\n\n        # === SWA accumulation ==='
)

# 5c. Fix ensemble weight bug: original code overwrites best_w with 0.0
full_code = full_code.replace(
    'best_w, best_auc, weight_results = optimize_ensemble_weight(oof_proto_flat, oof_mlp_flat, y_flat)\n    ENSEMBLE_WEIGHT_PROTO = 0.0',
    'best_w, best_auc, weight_results = optimize_ensemble_weight(oof_proto_flat, oof_mlp_flat, y_flat)\n    ENSEMBLE_WEIGHT_PROTO = best_w'
)

# 6. Remove display() calls (not available outside Jupyter)
full_code = full_code.replace('display(grid_results)', 'print(grid_results)')

# 7. Suppress tqdm.auto import failure in non-notebook env
full_code = full_code.replace(
    'from tqdm.auto import tqdm',
    'try:\n    from tqdm.auto import tqdm\nexcept ImportError:\n    from tqdm import tqdm'
)

# 8. Add artifact saving at the end
save_code = f'''

# ========================================
# SAVE ALL ARTIFACTS FOR KAGGLE INFERENCE
# ========================================
import torch
import joblib
import json as _json
import numpy as np

_ARTIFACT_DIR = Path("{ARTIFACT_DIR}")
_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print("\\n" + "="*60)
print("SAVING ARTIFACTS TO", _ARTIFACT_DIR)
print("="*60)

# 1. ProtoSSM model weights
torch.save(model.state_dict(), _ARTIFACT_DIR / "proto_ssm_final.pth")
print(f"Saved proto_ssm_final.pth")

# 2. ResidualSSM model weights (may be None)
if res_model is not None:
    torch.save(res_model.state_dict(), _ARTIFACT_DIR / "residual_ssm.pth")
    print(f"Saved residual_ssm.pth")
else:
    print("ResidualSSM was None, not saved")

# 3. MLP probe models
joblib.dump(probe_models, _ARTIFACT_DIR / "probe_models.joblib", compress=3)
print(f"Saved probe_models.joblib ({{len(probe_models)}} probes)")

# 4. Embedding scaler
joblib.dump(emb_scaler, _ARTIFACT_DIR / "emb_scaler.joblib")
print(f"Saved emb_scaler.joblib")

# 5. Embedding PCA
joblib.dump(emb_pca, _ARTIFACT_DIR / "emb_pca.joblib")
print(f"Saved emb_pca.joblib")

# 6. Prior tables (convert numpy to list for JSON)
def _np_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {{k: _np_to_list(v) for k, v in obj.items()}}
    elif isinstance(obj, (list, tuple)):
        return [_np_to_list(x) for x in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj

prior_tables_json = _np_to_list(final_prior_tables)
# Convert tuple keys in sh_to_i to string keys
if "sh_to_i" in prior_tables_json:
    prior_tables_json["sh_to_i"] = {{str(k): v for k, v in final_prior_tables["sh_to_i"].items()}}
with open(_ARTIFACT_DIR / "prior_tables.json", "w") as f:
    _json.dump(prior_tables_json, f)
print(f"Saved prior_tables.json")

# 7. Hyperparameters
hyperparams = {{
    "ensemble_weight_proto": float(ENSEMBLE_WEIGHT_PROTO),
    "correction_weight": float(CORRECTION_WEIGHT),
    "temperature": CFG["temperature"],
    "per_class_thresholds": PER_CLASS_THRESHOLDS.tolist() if "PER_CLASS_THRESHOLDS" in dir() else [0.5] * N_CLASSES,
    "frozen_best_probe": CFG["frozen_best_probe"],
    "best_fusion": {{k: float(v) for k, v in CFG["best_fusion"].items()}},
    "rank_aware_power": CFG.get("rank_aware_power", 0.5),
    "delta_shift_alpha": CFG.get("delta_shift_alpha", 0.15),
    "file_level_top_k": CFG.get("file_level_top_k", 2),
    "tta_shifts": CFG.get("tta_shifts", [0, 1, -1]),
    "proxy_reduce": CFG.get("proxy_reduce", "max"),
}}
with open(_ARTIFACT_DIR / "hyperparams.json", "w") as f:
    _json.dump(hyperparams, f, indent=2)
print(f"Saved hyperparams.json")

# 8. Model architecture config (for reconstruction)
model_config = {{
    "proto_ssm": {{k: v for k, v in CFG["proto_ssm"].items()}},
    "residual_ssm": {{k: v for k, v in CFG["residual_ssm"].items()}},
    "n_classes": N_CLASSES,
    "n_windows": N_WINDOWS,
}}
with open(_ARTIFACT_DIR / "model_config.json", "w") as f:
    _json.dump(model_config, f, indent=2)
print(f"Saved model_config.json")

# 9. OOF predictions (for post-processing tuning)
if oof_proto_flat is not None:
    np.savez_compressed(
        _ARTIFACT_DIR / "oof_predictions.npz",
        oof_proto_flat=oof_proto_flat,
        oof_base=oof_base,
        oof_prior=oof_prior,
        y_flat=labels_files.reshape(-1, N_CLASSES).astype(np.float32),
    )
    print(f"Saved oof_predictions.npz")

# 10. Site mapping (needed for test inference)
site_mapping = {{
    "site_to_idx": site_to_idx,
    "n_sites_mapped": n_sites_mapped,
}}
with open(_ARTIFACT_DIR / "site_mapping.json", "w") as f:
    _json.dump(site_mapping, f, indent=2)
print(f"Saved site_mapping.json")

# 11. Taxonomy groups (needed for family head reconstruction)
taxonomy_info = {{
    "n_families": n_families,
    "class_to_family": class_to_family,
    "fam_to_idx": fam_to_idx,
}}
with open(_ARTIFACT_DIR / "taxonomy_info.json", "w") as f:
    _json.dump(taxonomy_info, f, indent=2)
print(f"Saved taxonomy_info.json")

# 12. Class mapping info (needed for inference)
class_info = {{
    "PRIMARY_LABELS": PRIMARY_LABELS,
    "BC_INDICES": BC_INDICES.tolist(),
    "MAPPED_MASK": MAPPED_MASK.tolist(),
    "MAPPED_POS": MAPPED_POS.tolist(),
    "UNMAPPED_POS": UNMAPPED_POS.tolist(),
    "MAPPED_BC_INDICES": MAPPED_BC_INDICES.tolist(),
    "CLASS_NAME_MAP": CLASS_NAME_MAP,
    "SELECTED_PROXY_TARGETS": SELECTED_PROXY_TARGETS,
    "selected_proxy_pos_to_bc": {{str(k): v.tolist() for k, v in selected_proxy_pos_to_bc.items()}},
    "idx_active_texture": idx_active_texture.tolist(),
    "idx_active_event": idx_active_event.tolist(),
    "idx_mapped_active_texture": idx_mapped_active_texture.tolist(),
    "idx_mapped_active_event": idx_mapped_active_event.tolist(),
    "idx_selected_proxy_active_texture": idx_selected_proxy_active_texture.tolist(),
    "idx_selected_prioronly_active_texture": idx_selected_prioronly_active_texture.tolist(),
    "idx_selected_prioronly_active_event": idx_selected_prioronly_active_event.tolist(),
    "idx_unmapped_inactive": idx_unmapped_inactive.tolist(),
}}
with open(_ARTIFACT_DIR / "class_info.json", "w") as f:
    _json.dump(class_info, f, indent=2)
print(f"Saved class_info.json")

# Print artifact sizes
print("\\n" + "="*60)
print("ARTIFACT SIZES:")
print("="*60)
total = 0
for p in sorted(_ARTIFACT_DIR.glob("*")):
    sz = p.stat().st_size
    total += sz
    print(f"  {{p.name:40s}} {{sz/1024:8.1f}} KB")
print(f"  {{'TOTAL':40s}} {{total/1024:8.1f}} KB ({{total/1024/1024:.1f}} MB)")

print("\\n✅ All artifacts saved successfully!")
print(f"Wall time: {{(time.time() - _WALL_START)/60:.1f}} min")
'''

# 9. Compute per-class thresholds from OOF in train mode
# The notebook hardcodes them, but we want to re-compute from our better OOF
threshold_code = '''

# ========================================
# RE-COMPUTE PER-CLASS THRESHOLDS FROM OOF
# ========================================
if oof_proto_flat is not None:
    # Build the full ensemble OOF predictions
    oof_ensemble = (
        ENSEMBLE_WEIGHT_PROTO * oof_proto_flat +
        (1.0 - ENSEMBLE_WEIGHT_PROTO) * oof_mlp_flat
    ).astype(np.float32)

    y_flat = labels_files.reshape(-1, N_CLASSES).astype(np.float32)

    PER_CLASS_THRESHOLDS, per_class_f1 = optimize_per_class_thresholds(
        oof_ensemble, y_flat,
        n_windows=N_WINDOWS,
        thresholds=CFG["threshold_grid"],
    )

    print(f"Re-computed per-class thresholds from OOF:")
    print(f"  Mean: {PER_CLASS_THRESHOLDS.mean():.3f}")
    print(f"  Range: [{PER_CLASS_THRESHOLDS.min():.2f}, {PER_CLASS_THRESHOLDS.max():.2f}]")
else:
    print("No OOF predictions, using default thresholds")
    PER_CLASS_THRESHOLDS = np.full(N_CLASSES, 0.5, dtype=np.float32)
'''

# Threshold computation must come BEFORE artifact saving
full_code += threshold_code
full_code += save_code

# --- Write the patched script ---
script_path = PROJECT_ROOT / "src" / "_protossm_train_patched.py"
with open(script_path, "w") as f:
    f.write(full_code)

print(f"\nPatched script written to {script_path}")
print(f"({len(full_code)} bytes, {full_code.count(chr(10))} lines)")

# --- Execute it as a subprocess ---
print("\n" + "="*60)
print("STARTING LOCAL TRAINING")
print("="*60 + "\n")

import subprocess
result = subprocess.run(
    [sys.executable, "-u", str(script_path)],
    cwd=str(PROJECT_ROOT),
)
sys.exit(result.returncode)
