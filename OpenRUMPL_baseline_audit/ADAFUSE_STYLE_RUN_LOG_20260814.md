# AdaFuse-style dense heatmap run log (2026-08-14)

## Purpose

The immediate target is to beat AdaFuse under a defensible common protocol,
not to claim that a small RUMPL ray residual is AdaFuse.  The first controlled
step therefore uses the same frozen HRNet dense heatmaps for every method and
tests the part of AdaFuse that is identifiable from its public implementation:

1. dense epipolar heatmap correspondence;
2. a shared per-joint/per-view reliability predictor using appearance and
   cross-view consistency;
3. reliability-weighted fusion followed by the same calibrated robust ray
   intersection used by the heatmap audit.

The source reference is the public AdaFuse implementation at
`reference/adafuse-official` (commit `74fcb7d`).  This is an AdaFuse-style
adaptation, not an official AdaFuse reproduction: the official model trains a
ResNet-152 detector jointly and uses its own crop/data sampling code, whereas
this line freezes our HRNet export so that the comparison to RUMPL/A1D is
fair within the coordinate/heatmap protocol.

## Code and invariance checks

- `adafuse_style_heatmap_fusion.py`: shared reliability MLP and variable-view
  dense fusion;
- `train_dense_geometry_residual_fusion.py --model-kind adafuse_style`: 2D
  coordinate supervision, no test GT, no camera ID;
- `eval_h36m_dense_epipolar_heatmaps.py --fusion-model-kind adafuse_style`:
  strict combinations from the four synchronized views.

Smoke checks passed for V2/V3/V4: finite output, equal weights at
initialization, and random view permutation equivariance (`<2.4e-7` maximum
floating point discrepancy).

## Runs

| run | training view sampling | device | checkpoint | status |
|---|---|---:|---|---|
| AdaFuseStyle-balanced | V2/V3/V4 = 1/1/1 | GPU0 | `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/AdaFuseStyle_balanced_20260813/final.pth` | complete; corrected strict eval complete |
| AdaFuseStyle-v2focus | V2/V3/V4 = 3/1/1 | GPU1 | `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/AdaFuseStyle_v2focus_20260813/final.pth` | complete; corrected strict eval complete |

Both use 6,000 update steps, 32 depth samples, the full 16 training heatmap
shards, and the same S9/S11 validation records as the strict heatmap table.
The strict evaluator enumerates 6 V2, 4 V3, and 1 V4 camera subsets per
synchronized frame; each subset reads only its selected heatmaps.

## Corrected strict results (action-equal All-17, mm)

| method | V2 | V3 | V4 | Δ vs raw |
|---|---:|---:|---:|---|
| raw HRNet heatmap top-1 | 93.281 | 56.952 | 51.536 | — |
| balanced AdaFuse-style | 92.711 | 57.105 | 51.568 | −0.570 / +0.153 / +0.032 |
| V2-focus AdaFuse-style | 92.908 | 57.139 | 51.554 | −0.373 / +0.187 / +0.018 |

The learned gate is therefore not a stable improvement.  It is retained as a
negative-transfer control, not as the proposed method.  The earlier files
under `AdaFuseStyle_{balanced,v2focus}_20260813/full_eval.json` are invalid
because they used a source-only replacement formula and produced 550--830 mm;
only the `corrected_eval/full_eval.json` files above are valid.

## Official-style line-support control

The public AdaFuse code uses two depth endpoints (1--5 m) and samples the
complete horizontal/vertical epipolar line.  We implemented this as
`--support-mode line` and evaluated all strict subsets without training.  The
best fixed controls are:

| method | V2 | V3 | V4 | gain vs raw |
|---|---:|---:|---:|---|
| raw HRNet top-1 | 93.281 | 56.952 | 51.536 | — |
| line dense-add α=1 | 91.743 | **56.852** | 51.414 | 1.538 / 0.100 / 0.122 |
| line dense-PoE α=0.5 | **91.712** | 56.935 | **51.388** | **1.569** / 0.017 / **0.148** |

These are valid same-input controls, not AdaFuse official numbers.  The line
sampler helps V2 substantially but does not explain the multi-view gap; V3/V4
remain about 37--51 mm.  Therefore the next bottleneck is the 2D detector and
joint heatmap training/crop protocol, not merely the depth discretization.

## Decision rule

The result is accepted only if it is lower than the current same-input raw
HRNet heatmap values (V2 93.221, V3 56.707, V4 51.458 mm, action-equal
All-17) and improves the appropriate prior A1D table without changing the
detector or metric.  If it fails, it is recorded as a negative transfer of
the AdaFuse-style weighting under frozen HRNet heatmaps; the next step is not
to stack it on RUMPL blindly, but to inspect whether the missing gain is from
joint detector training, heatmap resolution/crop convention, or the official
AdaFuse two-depth line sampler.

## Current state (2026-08-14)

Training completed without NaNs.  The initial implementation had a source vs.
target reliability-indexing issue; it was corrected before formal evaluation:
weights now describe the source view (`support.sum(dim=0)`) while each target
view receives the corresponding aligned source maps (`support[target,source]`).
The first formal evaluation attempt also hit a duplicated path prefix; this
was fixed without changing any checkpoint.  A second audit then exposed a
more important implementation error: the first fusion formula replaced the
target detector by a weighted average of warped maps, causing 550--830 mm
outputs.  Those JSON files are **invalid and discarded**.  The corrected
formula keeps an explicit target identity path and adds 0.25-weighted source
evidence; strict evaluation is being rerun under
`.../AdaFuseStyle_{balanced,v2focus}_20260813/corrected_eval/`.

The next controlled test will align the public implementation more closely:
the official AdaFuse config uses two endpoints at 1--5 m and samples the full
horizontal/vertical epipolar line.  Our current support uses a 1--10 m,
32-depth sample approximation, so a 1--5 m high-density line-support control
will be evaluated before any further learned-gate tuning.

## A1D + official-style line support

The line support was then connected to the existing identity-preserving A1D
residual module. This combines a paper-backed AdaFuse geometric operation with
our previously validated correction head; no temporal module, image feature,
camera ID, or test GT was added.

| run | sampling | V2 | V3 | V4 | action-equal All-17, mm |
|---|---|---:|---:|---:|---|
| raw HRNet top-1 | — | 93.221 | 56.707 | 51.458 | — |
| A1D-line balanced | 1/1/1 | **87.685** | **52.631** | **46.896** | best unified |
| A1D-line V2-focus | 3/1/1 | **87.565** | 52.793 | 47.244 | V2 specialist |

The corresponding frame-weighted values are 87.626/53.014/47.095 for
balanced and 87.550/53.138/47.412 for V2-focus. Thus the balanced run is the
current unified candidate: improvements over raw are 5.536, 4.076 and 4.562
mm for V2/V3/V4. V2-focus confirms the V2 gain is not a smoke-sample effect,
but its V3/V4 values are slightly worse, so it remains a specialist ablation.

This still does not exceed the official AdaFuse 4-view number. The gap is not
explained by line sampling (fixed line controls only reached
91.712/56.852/51.388). The remaining high-priority difference is the 2D
front-end/training protocol: AdaFuse's public H36M configuration trains a
ResNet-152 pose backbone jointly, uses 384x384 input and 96x96 heatmaps, and
uses GT bounding boxes; our strict heatmap audit freezes an HRNet export at
96x72. Before adding more 3D modules, we will check whether the official
pretrained backbone can be obtained and reproduce this detector protocol.
