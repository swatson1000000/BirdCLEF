---
description: "Use when writing or editing ONNX export, Kaggle inference notebooks, or submission pipelines. Covers export constraints, sliding window inference, and submission requirements."
applyTo: "src/export_onnx.py, jupyter/**"
---

# Inference & ONNX Export Guidelines

## Inference Format

- **Use PyTorch CPU inference directly** — `onnxruntime` is not available in the Kaggle no-internet environment
- Load `.pt` checkpoints on CPU with `map_location="cpu"` — no ONNX or OpenVINO
- Run model in `eval()` + `torch.no_grad()` + `torch.inference_mode()`

## Sliding Window Inference

- Chunk size: 20 seconds, stride: 5 seconds
- Average overlapping frame predictions across chunks
- Output: per-species probability per 5-second window (matches competition format)

## Submission Constraints

- **CPU-only** notebook on Kaggle
- **≤ 90 min** total runtime
- **No internet** access — all models must be in the Kaggle dataset
- Kaggle dataset: `stevewatson999/birdclef2026-models`
- Submission notebook: `jupyter/birdclef2026-inference.ipynb`

## External Models: Aves/BirdAves

- BirdAves embedding model gives +0.01 **public** LB but hurts **private** LB
- Do **not** include in ensemble without a public/private LB gate confirming it helps both
