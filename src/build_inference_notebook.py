#!/usr/bin/env python3
"""
Build inference-only notebook from the original ProtoSSM notebook.
Takes cells needed for inference, replaces training with artifact loading.
"""
import json
import re
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "jupyter" / "protossm" / "birdclef2026-protossm.ipynb"
OUTPUT_DIR = PROJECT_ROOT / "jupyter" / "protossm-pretrained"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read original notebook
with open(NOTEBOOK_PATH) as f:
    nb = json.load(f)

# Extract code cells
code_cells = []
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        code_cells.append("".join(cell["source"]))

print(f"Original notebook: {len(code_cells)} code cells")

# ===== BUILD INFERENCE NOTEBOOK =====
# Cell mapping (from the index list above):
# 0:  Install deps
# 1:  MODE switch
# 2:  Imports + config
# 3:  V18 CFG
# 4:  Cosine scheduler
# 5:  Mixup
# 6:  Focal loss
# 7:  Data loading (labels)
# 8:  Class weights
# 9:  Isotonic calibration
# 10: Ensemble weight sweep
# 11: Perch model + mapping
# 12: Metrics helpers
# 13: Post-processing utilities
# 14: Perch inference function
# 15: Load/compute Perch cache
# 16: Prior tables
# 17: OOF base/prior
# 18: Embedding probe helpers
# 19: ProtoSSMv2 class definition
# 20: Training functions
# 21: Probe tuning
# 22: Freeze probe params
# 23: Fit prior tables
# 24: Fit scaler/PCA
# 25: Main training
# 26: ResidualSSM training
# 27: Diagnostics
# 28: Test Perch inference
# 29: Score fusion
# 30: Post-processing + submission
# 31: Final logging

# For inference, we need:
# - Cell 0: Install deps (KEEP)
# - Cell 1: MODE = submit (KEEP)
# - Cell 2: Imports + config (KEEP)
# - Cell 3: V18 CFG (KEEP)
# - Cell 6: Focal loss (needed for species_focal_loss reference) -- actually not needed for inference
# - Cell 7: Data loading (MODIFY: only taxonomy + sample_sub)
# - Cell 11: Perch + mapping (KEEP)
# - Cell 12: Metrics (KEEP)
# - Cell 13: Post-processing utilities (KEEP)
# - Cell 14: Perch inference function (KEEP)
# - Cell 16: Prior tables functions (KEEP: fit_prior_tables, prior_logits_from_tables, fuse_scores_with_tables)
# - Cell 18: Embedding probe helpers (KEEP: build_class_features, seq_features_1d)
# - Cell 19: ProtoSSMv2 + SelectiveSSM + TemporalCrossAttention (KEEP)
# - Cell 20: Training helpers (KEEP: reshape_to_files, get_file_metadata, build_site_mapping, temporal_shift_tta)
# - NEW: Load artifacts cell
# - Cell 28: Test Perch inference (KEEP)
# - Cell 29: Score fusion (MODIFY: use loaded model, no training)
# - Cell 30: Post-processing (KEEP)
# - Cell 31: Logging (KEEP)

def make_cell(source):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": source if isinstance(source, list) else [source],
    }

# --- Cell 0: Install deps (unchanged) ---
cell_install = code_cells[0]

# --- Cell 1: MODE = submit ---
cell_mode = 'MODE = "submit"\nassert MODE in {"train", "submit"}\nprint("MODE =", MODE)'

# --- Cell 2: Imports + config (unchanged) ---
cell_imports = code_cells[2]

# --- Cell 3: V18 CFG (unchanged) ---
cell_v18 = code_cells[3]

# --- Cell 7: Data loading (stripped to taxonomy + sample_sub only) ---
cell_data = '''# Data loading (inference-only: taxonomy + sample_submission)
taxonomy = pd.read_csv(BASE / "taxonomy.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY_LABELS)

taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)

# We need soundscape labels for prior tables and class mapping
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")
soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)

def parse_soundscape_labels(x):
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\\d+)_(S\\d+)_(\\d{8})_(\\d{6})\\.ogg")

def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {"file_id": None, "site": None, "date": pd.NaT, "time_utc": None, "hour_utc": -1, "month": -1}
    file_id, site, ymd, hms = m.groups()
    dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    return {"file_id": file_id, "site": site, "date": dt, "time_utc": hms,
            "hour_utc": int(hms[:2]), "month": int(dt.month) if pd.notna(dt) else -1}

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))

sc_clean = (
    soundscape_labels.groupby(["filename", "start", "end"])["primary_label"]
    .apply(union_labels).reset_index(name="label_list")
)
sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
sc_clean["row_id"] = sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_clean["end_sec"].astype(str)
meta = sc_clean["filename"].apply(parse_soundscape_filename).apply(pd.Series)
sc_clean = pd.concat([sc_clean, meta], axis=1)

label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
for i, labels in enumerate(sc_clean["label_list"]):
    idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
    if idxs:
        Y_SC[i, idxs] = 1

print(f"sc_clean: {sc_clean.shape}, Y_SC: {Y_SC.shape}, N_CLASSES: {N_CLASSES}")
'''

# --- Cell 11: Perch + mapping (unchanged) ---
cell_perch_mapping = code_cells[11]

# --- Cell 12: Metrics (unchanged) ---
cell_metrics = code_cells[12]

# --- Cell 13: Post-processing utilities (unchanged) ---
cell_postproc = code_cells[13]

# --- Cell 14: Perch inference function (unchanged) ---
cell_perch_infer = code_cells[14]

# --- Cell 16: Prior tables (only function defs, not the execution) ---
cell_prior_funcs = code_cells[16]

# --- Cell 18: Embedding probe helpers (unchanged) ---
cell_probe_helpers = code_cells[18]

# --- Cell 19: ProtoSSMv2 (unchanged) ---
cell_model_def = code_cells[19]

# --- Cell 20: Training functions (extract only helpers needed for inference) ---
# We need: reshape_to_files, get_file_metadata, build_site_mapping, temporal_shift_tta,
#           build_taxonomy_groups
cell_train_helpers = code_cells[20]

# --- NEW: Load artifacts ---
cell_load_artifacts = '''# ========================================
# LOAD PRE-TRAINED ARTIFACTS
# ========================================
import torch
import joblib

# Find pre-trained artifacts directory
import os
PRETRAINED_DIR = None
for _c in [
    "/kaggle/input/birdclef2026-protossm-pretrained",
    "/kaggle/input/datasets/stevewatson999/birdclef2026-protossm-pretrained",
]:
    _p = Path(_c)
    if _p.exists() and (_p / "model_config.json").exists():
        PRETRAINED_DIR = _p
        break
    # Check one level of subdirs (version dirs)
    if _p.exists():
        for _sub in _p.iterdir():
            if _sub.is_dir() and (_sub / "model_config.json").exists():
                PRETRAINED_DIR = _sub
                break
    if PRETRAINED_DIR:
        break

# Debug: show what's in /kaggle/input/ (top level only)
print("Mounted inputs:", sorted(os.listdir("/kaggle/input")))
assert PRETRAINED_DIR is not None, (
    "Dataset not mounted! Add 'birdclef2026-protossm-pretrained' via Kaggle notebook editor > Input > Add Input"
)
print(f"Loading pre-trained artifacts from {PRETRAINED_DIR}")

# 1. Model config
with open(PRETRAINED_DIR / "model_config.json") as f:
    model_config = json.load(f)

# 2. Hyperparameters
with open(PRETRAINED_DIR / "hyperparams.json") as f:
    hyperparams = json.load(f)

ENSEMBLE_WEIGHT_PROTO = hyperparams["ensemble_weight_proto"]
CORRECTION_WEIGHT = hyperparams["correction_weight"]
PER_CLASS_THRESHOLDS = np.array(hyperparams["per_class_thresholds"], dtype=np.float32)
CFG["temperature"] = hyperparams["temperature"]
CFG["frozen_best_probe"] = hyperparams["frozen_best_probe"]
CFG["best_fusion"] = {k: float(v) for k, v in hyperparams["best_fusion"].items()}
CFG["rank_aware_power"] = hyperparams.get("rank_aware_power", 0.4)
CFG["delta_shift_alpha"] = hyperparams.get("delta_shift_alpha", 0.20)
CFG["file_level_top_k"] = hyperparams.get("file_level_top_k", 2)
CFG["tta_shifts"] = hyperparams.get("tta_shifts", [0, 1, -1, 2, -2])
BEST = CFG["best_fusion"]

print(f"  ENSEMBLE_WEIGHT_PROTO = {ENSEMBLE_WEIGHT_PROTO}")
print(f"  CORRECTION_WEIGHT = {CORRECTION_WEIGHT}")
print(f"  TTA shifts = {CFG['tta_shifts']}")

# 3. Taxonomy info
with open(PRETRAINED_DIR / "taxonomy_info.json") as f:
    taxonomy_info = json.load(f)
n_families = taxonomy_info["n_families"]
class_to_family = taxonomy_info["class_to_family"]
fam_to_idx = taxonomy_info["fam_to_idx"]

# 4. Site mapping
with open(PRETRAINED_DIR / "site_mapping.json") as f:
    site_mapping = json.load(f)
site_to_idx = site_mapping["site_to_idx"]
n_sites_mapped = site_mapping["n_sites_mapped"]

# 5. Prior tables
with open(PRETRAINED_DIR / "prior_tables.json") as f:
    prior_tables_raw = json.load(f)

# Convert lists back to numpy arrays
final_prior_tables = {}
for k, v in prior_tables_raw.items():
    if isinstance(v, list):
        final_prior_tables[k] = np.array(v, dtype=np.float32)
    elif isinstance(v, dict) and k == "sh_to_i":
        # Convert string keys back to tuple keys
        final_prior_tables[k] = {eval(sk): sv for sk, sv in v.items()}
    elif isinstance(v, dict):
        # site_to_i, hour_to_i: keep as dict with proper types
        final_prior_tables[k] = {str(sk): int(sv) for sk, sv in v.items()}
    else:
        final_prior_tables[k] = v

print(f"  Prior tables loaded ({len(final_prior_tables)} keys)")

# 6. Reconstruct ProtoSSM model
ssm_cfg = model_config["proto_ssm"]
model = ProtoSSMv2(
    d_input=1536,
    d_model=ssm_cfg["d_model"],
    d_state=ssm_cfg["d_state"],
    n_ssm_layers=ssm_cfg["n_ssm_layers"],
    n_classes=N_CLASSES,
    n_windows=N_WINDOWS,
    dropout=ssm_cfg["dropout"],
    n_sites=ssm_cfg["n_sites"],
    meta_dim=ssm_cfg["meta_dim"],
    use_cross_attn=ssm_cfg.get("use_cross_attn", True),
    cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
).to(DEVICE)
model.init_family_head(n_families, class_to_family)
model.load_state_dict(torch.load(PRETRAINED_DIR / "proto_ssm_final.pth", map_location=DEVICE))
model.eval()
print(f"  ProtoSSM loaded ({model.count_parameters():,} params)")

# 7. Reconstruct ResidualSSM (if available)
res_model = None
res_path = PRETRAINED_DIR / "residual_ssm.pth"
if res_path.exists() and CORRECTION_WEIGHT > 0:
    res_cfg = model_config["residual_ssm"]

    class ResidualSSM(nn.Module):
        def __init__(self, d_input=1536, d_scores=234, d_model=64, d_state=8,
                     n_classes=234, n_windows=12, dropout=0.1, n_sites=20, meta_dim=8):
            super().__init__()
            self.d_model = d_model
            self.n_classes = n_classes
            self.input_proj = nn.Sequential(
                nn.Linear(d_input + d_scores, d_model), nn.LayerNorm(d_model),
                nn.GELU(), nn.Dropout(dropout),
            )
            self.site_emb = nn.Embedding(n_sites, meta_dim)
            self.hour_emb = nn.Embedding(24, meta_dim)
            self.meta_proj = nn.Linear(2 * meta_dim, d_model)
            self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
            self.ssm_fwd = SelectiveSSM(d_model, d_state)
            self.ssm_bwd = SelectiveSSM(d_model, d_state)
            self.ssm_merge = nn.Linear(2 * d_model, d_model)
            self.ssm_norm = nn.LayerNorm(d_model)
            self.ssm_drop = nn.Dropout(dropout)
            self.output_head = nn.Linear(d_model, n_classes)
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)

        def forward(self, emb, first_pass_scores, site_ids=None, hours=None):
            B, T, _ = emb.shape
            x = torch.cat([emb, first_pass_scores], dim=-1)
            h = self.input_proj(x)
            if site_ids is not None and hours is not None:
                site_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
                hour_e = self.hour_emb(hours.clamp(0, 23))
                meta = self.meta_proj(torch.cat([site_e, hour_e], dim=-1))
                h = h + meta.unsqueeze(1)
            h = h + self.pos_enc[:, :T, :]
            residual = h
            h_f = self.ssm_fwd(h)
            h_b = self.ssm_bwd(h.flip(1)).flip(1)
            h = self.ssm_merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = self.ssm_norm(h + residual)
            return self.output_head(h)

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    res_model = ResidualSSM(
        d_input=1536, d_scores=N_CLASSES,
        d_model=res_cfg["d_model"], d_state=res_cfg["d_state"],
        n_classes=N_CLASSES, n_windows=N_WINDOWS,
        dropout=res_cfg["dropout"], n_sites=ssm_cfg["n_sites"], meta_dim=8,
    ).to(DEVICE)
    res_model.load_state_dict(torch.load(res_path, map_location=DEVICE))
    res_model.eval()
    print(f"  ResidualSSM loaded ({res_model.count_parameters():,} params)")
else:
    print("  ResidualSSM: SKIPPED (not available or weight=0)")

# 8. Load MLP probes
probe_models = joblib.load(PRETRAINED_DIR / "probe_models.joblib")
print(f"  MLP probes loaded ({len(probe_models)} classes)")

# 9. Load scaler + PCA
emb_scaler = joblib.load(PRETRAINED_DIR / "emb_scaler.joblib")
emb_pca = joblib.load(PRETRAINED_DIR / "emb_pca.joblib")
print(f"  Scaler + PCA loaded (PCA dim={emb_pca.n_components})")

# 10. Load class info for fuse_scores_with_tables dependencies
with open(PRETRAINED_DIR / "class_info.json") as f:
    class_info = json.load(f)

# Reconstruct numpy arrays used in fuse_scores_with_tables
idx_active_texture = np.array(class_info["idx_active_texture"], dtype=np.int32)
idx_active_event = np.array(class_info["idx_active_event"], dtype=np.int32)
idx_mapped_active_texture = np.array(class_info["idx_mapped_active_texture"], dtype=np.int32)
idx_mapped_active_event = np.array(class_info["idx_mapped_active_event"], dtype=np.int32)
idx_selected_proxy_active_texture = np.array(class_info["idx_selected_proxy_active_texture"], dtype=np.int32)
idx_selected_prioronly_active_texture = np.array(class_info["idx_selected_prioronly_active_texture"], dtype=np.int32)
idx_selected_prioronly_active_event = np.array(class_info["idx_selected_prioronly_active_event"], dtype=np.int32)
idx_unmapped_inactive = np.array(class_info["idx_unmapped_inactive"], dtype=np.int32)
CLASS_NAME_MAP = class_info["CLASS_NAME_MAP"]

# OOF base/prior not needed — we use fuse_scores_with_tables directly with final_prior_tables

_wall_load = (time.time() - _WALL_START) / 60.0
print(f"\\n✅ All artifacts loaded in {_wall_load:.1f} min")
print(f"   Remaining wall time: {90.0 - _wall_load:.1f} min")
'''

# --- Cell 28: Test Perch inference (unchanged) ---
cell_test_perch = code_cells[28]

# --- Cell 29: Score fusion (unchanged - uses loaded model, probes, etc.) ---
cell_fusion = code_cells[29]

# --- Cell 30: Post-processing + submission (unchanged) ---
cell_postproc_submit = code_cells[30]

# --- Cell 31: Final logging (unchanged) ---
cell_logging = code_cells[31]

# ===== ASSEMBLE NOTEBOOK =====
new_cells = [
    make_cell(cell_install),          # 0: Install deps
    make_cell(cell_mode),             # 1: MODE = submit
    make_cell(cell_imports),          # 2: Imports + config
    make_cell(cell_v18),              # 3: V18 CFG
    make_cell(cell_data),             # 4: Data loading
    make_cell(cell_perch_mapping),    # 5: Perch + mapping
    make_cell(cell_metrics),          # 6: Metrics
    make_cell(cell_postproc),         # 7: Post-processing utilities
    make_cell(cell_perch_infer),      # 8: Perch inference function
    make_cell(cell_prior_funcs),      # 9: Prior table functions
    make_cell(cell_probe_helpers),    # 10: Embedding probe helpers
    make_cell(cell_model_def),        # 11: ProtoSSMv2 definition
    make_cell(cell_train_helpers),    # 12: Training helper functions
    make_cell(cell_load_artifacts),   # 13: LOAD ARTIFACTS
    make_cell(cell_test_perch),       # 14: Test Perch inference
    make_cell(cell_fusion),           # 15: Score fusion
    make_cell(cell_postproc_submit),  # 16: Post-processing + submission
    make_cell(cell_logging),          # 17: Final logging
]

notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": new_cells,
}

out_path = OUTPUT_DIR / "birdclef2026-protossm-pretrained.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"\nInference notebook written to {out_path}")
print(f"  {len(new_cells)} cells")

# ===== WRITE KERNEL METADATA =====
metadata = {
    "id": "stevewatson999/birdclef-2026-protossm-pretrained",
    "title": "BirdCLEF 2026 ProtoSSM Pretrained",
    "code_file": "birdclef2026-protossm-pretrained.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": False,
    "competition_sources": ["birdclef-2026"],
    "dataset_sources": [
        "stevewatson999/birdclef2026-protossm-pretrained"
    ],
    "model_sources": [
        "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"
    ],
    "kernel_sources": [
        "ashok205/tf-wheels"
    ]
}

meta_path = OUTPUT_DIR / "kernel-metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Kernel metadata written to {meta_path}")
