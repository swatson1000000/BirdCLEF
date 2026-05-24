# XC v3 pretrain prep (status 2026-05-10)

Forward-looking prep for the XC v3 pretrain lever. Written as a sketch — NOT
yet executed. Triggers based on iNat v2 outcome (see `new_plan.md` PICK UP
HERE decision tree).

## Trigger conditions

Run XC v3 pretrain only if one of these fires:
- **(T1)** iNat v2 (re-pretrain with natural sampling + MixUp) fails to
  meaningfully exceed iNat-prodft v1's 0.7130 fold-0 val_v2 (e.g., < 0.72).
  This signals that sampler + MixUp aren't the missing pieces; the corpus
  itself might be wrong.
- **(T2)** iNat v2 succeeds (>0.7414, beats ImageNet) and we want a
  *different* corpus to ensemble with for diversity.

Skip XC v3 entirely if iNat v2 lands in 0.72-0.7414 — that's the "partial
fix" zone where another corpus is unlikely to bridge the remaining gap
faster than other levers (different backbones, post-processing, stacking).

## Corpus state check (verify before any pretrain dispatch)

Per `BirdCLEF/CLAUDE.md` and `new_plan.md`, XC v3 was downloaded earlier in
2026-04 / 2026-05 timeframe. Locations to verify:

```bash
# Local skynet
ls -la /home/swatson/work/kaggle/BirdCLEF/data/raw/birdclef_2025/   # earlier comp data
ls -la /home/swatson/work/kaggle/BirdCLEF/data/external/             # extra corpora
find /home/swatson/work/kaggle/BirdCLEF/data -maxdepth 3 -name 'xc_v3*' -o -name 'xenocanto*'

# Deepthought
ssh deepthought "find /home/swatson/work/MachineLearning/kaggle/BirdCLEF/data \\
                    -maxdepth 3 -name 'xc_v3*' -o -name 'xenocanto*'"
```

If the corpus exists, the manifest should give us:
- Total clip count (XC v3 was expected to be 308k-570k clips)
- Species distribution
- Audio sample-rate / format diversity

## Pretrain script design (delta from iNat re-pretrain v2)

The iNat v2 pretrain established the production-grade recipe:
- `--ft-recipe production` (lr=5e-4 + warm restarts) when finetuning
- `--natural-sampling` (drop balanced sampler)
- `--mixup-prob 0.5` (A1-style waveform MixUp with element-wise-max labels)
- focal-BCE loss, mixstyle_p=0.5, 25 epochs

XC v3 differs from iNat in two ways that affect script design:
1. **Multi-positive labels available.** XC has an `also` field per clip
   (secondary species heard in the recording). Use this to construct
   multi-positive label tensors directly, instead of relying on MixUp to
   synthesize multi-positive examples post-hoc.
2. **Tighter taxonomic alignment.** XC v3's species set should be closer
   to BC2026's 234 than iNat's 5569. May not need head-drop logic; could
   even keep the full pretrain head if species overlap is >80%.

Sketch `src/pretrain_xc_v3.py`:
- Reuse `_load_waveform`, `_augment`, `waveform_to_mel` from iNat pretrain
- Multi-positive `_build_targets` from XC `primary_label` + `also`
- Sampler choice: same `--natural-sampling` flag (default off, recommended on)
- Same MixUp logic but ADDITIONALLY mixes labels from both samples
  (element-wise max already handles multi-positive)
- Save path: `inat_best_..._xc_v3.pt` or `xc_best_..._<config>.pt`

## What to NOT bring forward from iNat

- `WeightedRandomSampler(1/n_per_class)` — confirmed harmful for transfer
- Single-positive label structure — XC v3 has secondaries, use them
- Gentle finetune recipe — confirmed bug, use production-recipe by default

## Expected wall time

If XC v3 is comparable in size to iNat (~140k-180k clips), pretrain time
should be similar: ~25 epochs × ~40 min/epoch = ~16-20 h on DT 4080.

If significantly larger (~500k clips), scale linearly. Plan around ~24-30 h
in that case; consider whether to reduce to 15-20 epochs given larger N
per epoch.

## Decision tree if XC v3 also underperforms ImageNet

Per the "three converging signals" finding from 2026-05-10 (L2, iNat-gentle,
iNat-production all below ImageNet baseline), pretrain-on-bird-audio may have
a structural ceiling near ImageNet. If XC v3 also lands below 0.74:

- **Don't run a fourth pretrain corpus.** The pattern is the diagnosis.
- Pivot to:
  - **Different backbones** (V2-S 5-fold completion, ConvNeXt at production
    recipe with ImageNet init). Cross-arch diversity for ensembling.
  - **Stacking variants** beyond D2/D1-b (e.g., per-species temperature
    scaling, learned blend weights — different from the killed D2-α/β/γ).
  - **Test-time augmentation / inference-time tricks** (rank fusion
    variants, threshold calibration).

The strategic insight: ImageNet supervised pretrain on a diverse natural-image
corpus may produce features that *generalize* better to BC2026 soundscape
than any focal-bird-audio pretrain can match. We should respect that pattern
rather than fight it.
