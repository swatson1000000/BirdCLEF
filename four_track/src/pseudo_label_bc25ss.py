"""D3 Phase 1: pseudo-label BC2025 Colombia soundscapes with A1 4-fold ensemble.

Iterates over the 9,726 unlabeled 60 s soundscape files in
`data/raw/birdclef_2025/train_soundscapes/`, cuts each into 12 non-overlapping
5 s windows, mirrors the production-notebook chunking (tile 5 s -> 20 s,
mel + PCEN + [0,1] norm + 3-channel repeat), runs each of the 4 A1 JIT folds,
averages the 4 fold sigmoids, and saves:

  four_track/data/processed/pseudo_bc25ss_probs.npz
    probs:     (N_FILES, 12, 234)  float32  mean-of-4-fold sigmoids
    filenames: (N_FILES,)          <U64     relative to soundscape dir

Compute budget: ~15 min on a single GPU (116,712 windows x 2 ms / fold x 4
folds ~= 900 s).

Later phases (§14.11.7):
  Phase 2 — apply threshold tau, write train-folds-compatible rows.
  Phase 3 — extend dataset_a1.py with BC2025_SS_PSEUDO branch.
  Phase 4 — retrain fold-0.
  Phase 5 — JIT export + v57 LB probe.
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from tqdm import tqdm

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
except AttributeError:
    pass
try:
    torch._C._jit_set_nvfuser_enabled(False)
except (AttributeError, RuntimeError):
    pass

HERE       = Path(__file__).resolve().parent
FT_ROOT    = HERE.parent
ROOT       = FT_ROOT.parent
PARENT_SRC = ROOT / "src"

if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))

from config import RAW, PROC                                      # noqa: E402

BC25_SS_DIR = RAW / "birdclef_2025" / "train_soundscapes"
A1_CKPT_DIR = FT_ROOT / "kaggle_datasets" / "a1-effb0-ckpts"
OUT_DIR     = FT_ROOT / "data" / "processed"
OUT_NPZ     = OUT_DIR / "pseudo_bc25ss_probs.npz"

SR              = 32_000
WINDOW_SEC      = 5
WINDOW_SAMPLES  = SR * WINDOW_SEC
N_WINDOWS       = 12
FILE_SAMPLES    = SR * 60
A1_FOLDS        = [0, 1, 2, 4]
A1_DURATION     = 20
A1_N_MELS       = 224
A1_N_FFT        = 4096
A1_HOP          = 1252
A1_F_MIN        = 0
A1_F_MAX        = 16_000
N_CLASSES       = 234

_A1_PCEN_GAIN   = 0.98
_A1_PCEN_SMOOTH = 0.025
_A1_PCEN_BIAS   = 2.0
_A1_PCEN_POWER  = 0.5
_A1_PCEN_EPS    = 1e-6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pcen(mel: torch.Tensor) -> torch.Tensor:
    E = mel.float()
    T_ = E.shape[2]
    M = E[:, :, 0].clone()
    out = torch.empty_like(E)
    bias_r = _A1_PCEN_BIAS ** _A1_PCEN_POWER
    for t in range(T_):
        M = (1.0 - _A1_PCEN_SMOOTH) * M + _A1_PCEN_SMOOTH * E[:, :, t]
        denom = (M + _A1_PCEN_EPS).pow(_A1_PCEN_GAIN)
        out[:, :, t] = (E[:, :, t] / denom + _A1_PCEN_BIAS).pow(_A1_PCEN_POWER) - bias_r
    return out


def _window_to_mel(wav_1d: np.ndarray, mel_transform: T.MelSpectrogram) -> torch.Tensor:
    chunk_len = SR * A1_DURATION
    reps = -(-chunk_len // len(wav_1d))
    wav  = np.tile(wav_1d, reps)[:chunk_len]
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)
    mel   = mel_transform(wav_t)
    out   = _pcen(mel)
    out   = out - out.min()
    peak  = out.max()
    if peak > 0:
        out = out / peak
    return out.repeat(3, 1, 1)


def main() -> None:
    assert BC25_SS_DIR.exists(), f"missing {BC25_SS_DIR}"
    files = sorted(BC25_SS_DIR.glob("*.ogg"))
    print(f"[d3] found {len(files)} BC2025 soundscape files", flush=True)
    assert len(files) > 0

    # Build mel transform once on the correct device
    mel_transform = T.MelSpectrogram(
        sample_rate=SR,
        n_fft=A1_N_FFT,
        hop_length=A1_HOP,
        n_mels=A1_N_MELS,
        f_min=A1_F_MIN,
        f_max=A1_F_MAX,
        power=2.0,
        norm="slaney",
        mel_scale="slaney",
        center=True,
    )

    print(f"[d3] loading {len(A1_FOLDS)} A1 JIT folds from {A1_CKPT_DIR}", flush=True)
    models = []
    for fi in A1_FOLDS:
        p = A1_CKPT_DIR / f"a1_fold{fi}.pt"
        assert p.exists(), f"missing {p}"
        m = torch.jit.load(str(p), map_location=DEVICE).eval()
        models.append(m)
        print(f"  fold{fi}: {p.stat().st_size / 1e6:.1f} MB", flush=True)

    n_files = len(files)
    # Store mean-of-4-fold sigmoids directly to save 4x memory
    probs = np.zeros((n_files, N_WINDOWS, N_CLASSES), dtype=np.float32)
    names = np.array([f.name for f in files], dtype="<U64")

    print(f"[d3] inference: {n_files} files x {N_WINDOWS} windows x "
          f"{len(A1_FOLDS)} folds = {n_files * N_WINDOWS * len(A1_FOLDS)} fwd-passes",
          flush=True)
    t0 = time.time()

    with torch.no_grad():
        for idx, path in enumerate(tqdm(files, desc="BC25SS A1 ensemble")):
            try:
                y, sr_file = sf.read(str(path), dtype="float32", always_2d=False)
            except Exception as ex:
                print(f"  [warn] failed to read {path.name}: {ex}", flush=True)
                continue
            if y.ndim == 2:
                y = y.mean(axis=1)
            if sr_file != SR:
                print(f"  [warn] unexpected sr {sr_file} in {path.name} — skipping",
                      flush=True)
                continue
            if len(y) < FILE_SAMPLES:
                y = np.pad(y, (0, FILE_SAMPLES - len(y)))
            elif len(y) > FILE_SAMPLES:
                y = y[:FILE_SAMPLES]

            file_mels = []
            for w in range(N_WINDOWS):
                seg = y[w * WINDOW_SAMPLES : (w + 1) * WINDOW_SAMPLES]
                file_mels.append(_window_to_mel(seg, mel_transform))
            batch = torch.stack(file_mels).to(DEVICE)             # (12, 3, 224, 512)

            fold_sigmoids = np.zeros((len(A1_FOLDS), N_WINDOWS, N_CLASSES),
                                     dtype=np.float32)
            for fi, m in enumerate(models):
                logits = m(batch)
                fold_sigmoids[fi] = torch.sigmoid(logits).float().cpu().numpy()
            probs[idx] = fold_sigmoids.mean(axis=0)

            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (idx + 1) * (n_files - idx - 1)
                print(f"  [{idx + 1}/{n_files}] elapsed={elapsed / 60:.1f}m  "
                      f"eta={eta / 60:.1f}m  "
                      f"mean_max_prob={probs[:idx + 1].max(axis=2).mean():.3f}",
                      flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, probs=probs, filenames=names)
    elapsed = time.time() - t0
    print(f"\n[d3] wrote {OUT_NPZ.name}  "
          f"shape={probs.shape}  size={OUT_NPZ.stat().st_size / 1e6:.1f} MB",
          flush=True)
    print(f"[d3] total time: {elapsed / 60:.1f} min", flush=True)

    # Coarse stats for the Phase 2 diagnostic
    max_prob = probs.max(axis=2)
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
        kept = int((max_prob > tau).sum())
        print(f"  tau={tau:.2f}  windows>tau: {kept} / {max_prob.size} "
              f"({100 * kept / max_prob.size:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
