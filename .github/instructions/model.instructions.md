---
description: "Use when writing or editing the model architecture, SED head, GEM pooling, or backbone selection. Covers BirdSEDModel design and backbone priority."
applyTo: "src/model.py"
---

# Model Architecture Guidelines

## BirdSEDModel Overview

```python
# CNN backbone (timm) → GEM frequency pooling → attention-weighted clip prediction
backbone = timm.create_model(backbone_name, features_only=True, out_indices=(4,))
# SED head: Conv1d attention + classifier over time frames
# clip_prob = weighted_sum(frame_prob * frame_att) / sum(frame_att)
```

- GEM pools the **frequency** dimension, leaving the **time** dimension intact for the SED head
- The SED head produces per-frame probabilities; clip probability = attention-weighted mean over frames
- Output shape: `(B, num_classes)` for clip-level prediction, `(B, T, num_classes)` for frame-level

## Backbone Priority

Choose the first backbone that meets the inference budget:

1. `tf_efficientnet_b0.ns_jft_in1k` — primary baseline, best accuracy/speed balance
2. `efficientvit_b0.r224_in1k` — 2–3× faster ONNX inference, slight accuracy trade-off
3. `regnety_008`
4. `tf_efficientnet_b3`
5. `tf_efficientnet_b4`
6. `eca_nfnet_l0` — **do not use for ONNX/OpenVINO export** (conversion fails)

## ONNX Export Compatibility

- **Never call `torch.compile()` before ONNX export** — they are incompatible in the same session
- Export un-compiled weights: load checkpoint → create fresh model → `torch.onnx.export`
- Use `dynamic_axes={"input": {0: "batch_size"}}` for variable batch size
