"""Build matched A2_SED + post-postproc-ProtoSSM OOF on the 59-file substrate.

Ports the production notebook's postproc pipeline (per-taxon temperature →
file-level top-K scaling → rank-aware scaling → adaptive delta-shift
smoothing → per-class threshold sharpening) and applies it to the raw
ProtoSSM OOF logits. Output mirrors what production does to test predictions
(within the limit that we omit the residual-SSM correction and the
Perch-CV-blend contribution — see new_plan.md §39 for the caveat).

Output: data/matched_oof_v75_postproc.npz with keys
  p_sed     (708, 234)
  p_proto   (708, 234)   post-postproc, sigmoid-scale
  y_true    (708, 234)
  file_ids  (708,)
  filenames (59,)
"""

from pathlib import Path
import numpy as np
import pandas as pd

FT_ROOT = Path(__file__).resolve().parents[1]
A2_PATH     = FT_ROOT / "data" / "a2_a1_5fold_broader_oof.npz"
PROTO_PATH  = FT_ROOT / "data" / "protossm_oof.npz"
THRESH_PATH = FT_ROOT / "data" / "per_class_thresholds.npz"
TAX_PATH    = FT_ROOT.parent / "data" / "raw" / "taxonomy.csv"
OUT_PATH    = FT_ROOT / "data" / "matched_oof_v75_postproc.npz"

# Production postproc CFG values (V18 overrides applied)
T_AVES                 = 1.10
T_TEXTURE              = 0.95
TEXTURE_TAXA           = {"Amphibia", "Insecta"}
FILE_LEVEL_TOP_K       = 2
RANK_AWARE_POWER       = 0.4
DELTA_SHIFT_ALPHA      = 0.20
TOPN_N1_ENABLED        = False
N_WINDOWS              = 12


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# Verbatim from notebook line 988
def file_level_confidence_scale(preds, n_windows=12, top_k=2):
    N, C = preds.shape
    assert N % n_windows == 0
    view = preds.reshape(-1, n_windows, C)
    sorted_view = np.sort(view, axis=1)
    top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
    scaled = view * top_k_mean
    return scaled.reshape(N, C)


# Verbatim from notebook line 1031
def rank_aware_scaling(scores, n_windows=12, power=0.5):
    N, C = scores.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = scores.reshape(n_files, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)
    scale = np.power(file_max, power)
    scaled = view * scale
    return scaled.reshape(N, C)


# Verbatim from notebook line 4837 (inline def in cell 42)
def adaptive_delta_smooth(probs, n_windows, base_alpha=0.20):
    n_files = probs.shape[0] // n_windows
    result = probs.copy()
    view = result.reshape(n_files, n_windows, -1)
    p_view = probs.reshape(n_files, n_windows, -1)
    for i in range(1, n_windows - 1):
        conf = p_view[:, i, :].max(axis=-1, keepdims=True)
        a = base_alpha * (1.0 - conf)
        neighbor_avg = (p_view[:, i-1, :] + p_view[:, i+1, :]) / 2.0
        view[:, i, :] = (1.0 - a) * p_view[:, i, :] + a * neighbor_avg
    return result.reshape(probs.shape)


# Verbatim from notebook line 1111
def apply_per_class_thresholds(scores, thresholds, n_windows=12):
    N, C = scores.shape
    assert C == len(thresholds)
    scaled = np.copy(scores)
    for c in range(C):
        t = thresholds[c]
        mask_above = scores[:, c] > t
        scaled[mask_above, c] = 0.5 + 0.5 * (scores[mask_above, c] - t) / (1 - t + 1e-8)
        scaled[~mask_above, c] = 0.5 * scores[~mask_above, c] / (t + 1e-8)
    return np.clip(scaled, 0, 1)


def main():
    a2    = np.load(A2_PATH, allow_pickle=True)
    proto = np.load(PROTO_PATH, allow_pickle=True)
    thresh = np.load(THRESH_PATH, allow_pickle=True)
    PER_CLASS_THRESHOLDS = thresh["thresholds"].astype(np.float32)

    taxonomy = pd.read_csv(TAX_PATH)
    PRIMARY_LABELS = taxonomy["primary_label"].astype(str).tolist()
    CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
    N_CLASSES = len(PRIMARY_LABELS)

    # --- Step 0: Build matched (file, window) ordering as before ---
    a2_files  = np.array([str(f) for f in a2["filenames"]])
    a2_starts = a2["start_sec"].astype(np.int32)
    a2_sed    = a2["probs_mean"].astype(np.float32)
    a2_y      = a2["y_true"].astype(np.float32)

    proto_files = [str(f) for f in proto["file_list"]]
    proto_logit_3d = proto["oof_mean"].astype(np.float32)   # (59, 12, 234) — LOGITS

    n_files = len(proto_files)
    proto_logit_flat = proto_logit_3d.reshape(n_files * N_WINDOWS, N_CLASSES)

    # --- Step 1: per-taxon temperature → sigmoid (mirrors notebook lines 4806-4820) ---
    class_temperatures = np.ones(N_CLASSES, dtype=np.float32) * T_AVES
    n_texture = 0
    for ci, label in enumerate(PRIMARY_LABELS):
        cn = CLASS_NAME_MAP.get(label, "Aves")
        if cn in TEXTURE_TAXA:
            class_temperatures[ci] = T_TEXTURE
            n_texture += 1
    print(f"[postproc] per-taxon temperature: Aves={T_AVES}, Texture={T_TEXTURE} "
          f"({n_texture} of {N_CLASSES} texture classes)")
    scaled_scores = proto_logit_flat / class_temperatures[None, :]
    probs = sigmoid(scaled_scores)

    # --- Step 2: file-level confidence scaling ---
    print(f"[postproc] file-level scaling, top_k={FILE_LEVEL_TOP_K}")
    probs = file_level_confidence_scale(probs, n_windows=N_WINDOWS, top_k=FILE_LEVEL_TOP_K)
    probs = np.clip(probs, 0.0, 1.0)

    # --- Step 3: rank-aware scaling ---
    print(f"[postproc] rank-aware scaling, power={RANK_AWARE_POWER}")
    probs = rank_aware_scaling(probs, n_windows=N_WINDOWS, power=RANK_AWARE_POWER)
    probs = np.clip(probs, 0.0, 1.0)

    # --- Step 4: adaptive delta-shift smoothing ---
    print(f"[postproc] adaptive delta-shift smoothing, alpha={DELTA_SHIFT_ALPHA}")
    probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS, base_alpha=DELTA_SHIFT_ALPHA)
    probs = np.clip(probs, 0.0, 1.0)

    # --- Step 5: per-class threshold sharpening ---
    print(f"[postproc] per-class threshold sharpening")
    probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)
    # already clipped inside fn

    p_proto_postproc_3d = probs.reshape(n_files, N_WINDOWS, N_CLASSES)

    # --- Step 6: align with A2 SED, dedup A2 duplicates ---
    p_sed_out   = np.zeros((n_files * N_WINDOWS, N_CLASSES), dtype=np.float32)
    p_proto_out = np.zeros_like(p_sed_out)
    y_out       = np.zeros_like(p_sed_out)
    file_ids    = np.zeros(n_files * N_WINDOWS, dtype=np.int64)

    for fi, fname in enumerate(proto_files):
        a2_mask = a2_files == fname
        starts_here = a2_starts[a2_mask]
        sed_here    = a2_sed[a2_mask]
        y_here      = a2_y[a2_mask]
        _, first_idx = np.unique(starts_here, return_index=True)
        order = np.argsort(starts_here[first_idx])
        first_idx = first_idx[order]
        sed_w  = sed_here[first_idx]
        y_w    = y_here[first_idx]
        for wi in range(N_WINDOWS):
            row = fi * N_WINDOWS + wi
            p_sed_out[row]   = sed_w[wi]
            p_proto_out[row] = p_proto_postproc_3d[fi, wi]
            y_out[row]       = y_w[wi]
            file_ids[row]    = fi

    print(f"\nbuilt matched OOF with POST-POSTPROC ProtoSSM:")
    print(f"  rows={p_sed_out.shape[0]}  classes={N_CLASSES}  n_files={n_files}")
    print(f"  p_sed range:           [{p_sed_out.min():.4f}, {p_sed_out.max():.4f}]  mean={p_sed_out.mean():.4f}")
    print(f"  p_proto (post-postproc) range: [{p_proto_out.min():.4f}, {p_proto_out.max():.4f}]  mean={p_proto_out.mean():.4f}")
    print(f"  y_true positives total: {int(y_out.sum())}")

    np.savez_compressed(
        OUT_PATH,
        p_sed=p_sed_out, p_proto=p_proto_out, y_true=y_out, file_ids=file_ids,
        filenames=np.array(proto_files, dtype=object),
    )
    print(f"saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
