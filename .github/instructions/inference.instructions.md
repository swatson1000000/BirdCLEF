---
description: "Use when writing or editing ONNX export, Kaggle inference notebooks, or submission pipelines. Covers export constraints, sliding window inference, and submission requirements."
applyTo: "src/export_onnx.py, jupyter/**"
---

# Inference & ONNX Export Guidelines

## ONNX Export

- Export each fold's model to ONNX for CPU inference — **no quantization**
- `dynamic_axes={"input": {0: "batch_size"}}` for variable batch size
- **Never export a `torch.compile()`'d model** — load checkpoint → create fresh model → export
- `eca_nfnet_l0` backbone fails ONNX conversion — use a different backbone

## OpenVINO

- OpenVINO is ~2× faster than ONNX on CPU but causes ~0.01 ROC-AUC drop
- `eca_nfnet_l0` also fails OpenVINO conversion
- Default to ONNX unless inference time is the bottleneck

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
