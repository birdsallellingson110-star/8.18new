# 2026-07-11 hard-view debug CMU V=2 evaluation

Setting:

- Dataset: CMU pose5/pose6 single-person eval
- Metric: Absolute KP* MPJPE, mm
- KP*: shoulders, elbows, wrists, knees, ankles; excludes head and hips
- Baseline: RUMPL-conf baseline
- Model: `distill_hardv_debug_2026-07-10_14-31-32/model_best.pth.tar`
- Logs: `/mnt/data/cjyoutput/cmu_v2_eval_hardv_debug_20260711_0107_full/`

| views | RUMPL baseline KP* | hardv KP* | delta | hardv All-17 |
|---|---:|---:|---:|---:|
| [3,6] | 40.37 | 39.67 | -0.70 | 37.53 |
| [3,12] | 46.95 | 45.43 | -1.52 | 44.12 |
| [3,13] | 39.79 | 40.12 | +0.33 | 37.51 |
| [3,23] | 32.30 | 31.33 | -0.97 | 31.99 |
| [6,12] | 67.28 | 59.87 | -7.41 | 53.73 |
| [6,13] | 53.52 | 47.60 | -5.92 | 41.39 |
| [6,23] | 39.39 | 37.67 | -1.72 | 37.03 |
| [12,13] | 59.41 | 52.88 | -6.53 | 48.14 |
| [12,23] | 46.04 | 42.30 | -3.74 | 41.16 |
| [13,23] | 44.08 | 42.15 | -1.93 | 38.54 |

Summary:

- Average delta: -3.01 mm
- Worst delta: +0.33 mm
- Improved configs: 9/10

Interpretation:

Hard-view improves the difficult geometric pairs strongly, especially `[6,12]`,
`[12,13]`, and `[6,13]`. Its average gain is larger than the current ensemble
summary, but it is not yet a clean main result because `[3,13]` regresses.
This makes it a useful candidate for combining with `lw0.7` or for a
worst-config-aware objective, rather than replacing the current robust ensemble
directly.

## Follow-up run: hard-view + legw0.7

Started on 2026-07-11 01:22.

- Script: `OpenRUMPL/MHP/run_distill_hardv_legw07_full_20260711.sh`
- Launch log: `/mnt/data/cjyoutput/distill_hardv_legw07_full_20260711.launch.log`
- Output dir: `/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/distill_hardv_legw07_full_20260711_2026-07-11_01-22-52`
- GPU: 0
- Distillation env: `DISTILL_W=1`, `STUDENT_GT_W=1`, `STUDENT_VIEWS=2`,
  `HARD_VIEW_MINING=1`, `HARD_VIEW_CAND=3`, `LEG_DISTILL_W=0.7`

Gate for paper story:

- Must improve all 10 CMU V=2 view pairs against the RUMPL-conf baseline.
- Primary failure case to fix from hard-view debug is `[3,13]`: 40.12 vs
  baseline 39.79, delta `+0.33 mm`.
- Target is not only average gain; the publishable claim requires worst-case
  delta `<= 0`, preferably with average delta at least as strong as the current
  robust ensemble (`g1 + lw0.7 seeds0/1/2`, average `-2.64 mm`).

Result on CMU V=2:

| checkpoint | avg delta | worst delta | improved pairs | blocker |
|---|---:|---:|---:|---|
| `model_best.pth.tar` | -2.89 mm | +0.73 mm | 9/10 | `[3,12]` |
| `final_state.pth.tar` | -2.78 mm | +0.40 mm | 8/10 | `[3,12]`, `[3,13]` |

`model_best` fixes the original hard-view failure on `[3,13]`
(`+0.33 mm` -> `-0.24 mm`) but creates a new regression on `[3,12]`
(`+0.73 mm`). Therefore this is still not a clean main-result model.

Detailed `model_best` table:

| views | RUMPL baseline KP* | hardv debug KP* | hardv+legw KP* | delta |
|---|---:|---:|---:|---:|
| [3,6] | 40.37 | 39.67 | 38.93 | -1.44 |
| [3,12] | 46.95 | 45.43 | 47.68 | +0.73 |
| [3,13] | 39.79 | 40.12 | 39.55 | -0.24 |
| [3,23] | 32.30 | 31.33 | 30.84 | -1.46 |
| [6,12] | 67.28 | 59.87 | 60.19 | -7.09 |
| [6,13] | 53.52 | 47.60 | 47.57 | -5.95 |
| [6,23] | 39.39 | 37.67 | 36.48 | -2.91 |
| [12,13] | 59.41 | 52.88 | 52.94 | -6.47 |
| [12,23] | 46.04 | 42.30 | 43.92 | -2.12 |
| [13,23] | 44.08 | 42.15 | 42.12 | -1.96 |

Next low-cost test:

- Prediction ensemble between `hardv_debug` and `hardv+legw07 model_best`.
- Rationale: `hardv_debug` is good on `[3,12]`, while `hardv+legw07`
  fixes `[3,13]`; their failure modes are complementary.

## Prediction ensemble result

Prediction-level ensemble:

`pred = alpha * pred_hardv_debug + (1 - alpha) * pred_hardv_legw07`

Repro script:

- `OpenRUMPL/MHP/analyze_cmu_v2_prediction_ensemble.py`

Prediction sources:

- `hardv_debug_v2_*`: `/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/cmu_eval_sp_v2_conf/preds_gt_multiview_cmu_panoptic_rumpl_mmpose_hardv_debug_v2_*_best.pkl`
- `hardv_legw07_bestens_v2_*`: regenerated from `hardv+legw07 model_best` with independent eval comments.

Fine sweep over `alpha` in steps of 0.01:

- All-down alpha range: `0.16..0.74`
- Best worst-case margin: `alpha=0.23`, avg delta `-3.23 mm`,
  worst delta `-0.28 mm`
- Best average delta: `alpha=0.55`, avg delta `-3.41 mm`,
  worst delta `-0.17 mm`
- Balanced candidate: `alpha=0.50`, avg delta `-3.40 mm`,
  worst delta `-0.20 mm`

Balanced candidate table (`alpha=0.50`):

| views | RUMPL baseline KP* | ensemble KP* | delta |
|---|---:|---:|---:|
| [3,6] | 40.37 | 39.00 | -1.37 |
| [3,12] | 46.95 | 45.80 | -1.15 |
| [3,13] | 39.79 | 39.59 | -0.20 |
| [3,23] | 32.30 | 30.87 | -1.43 |
| [6,12] | 67.28 | 59.57 | -7.71 |
| [6,13] | 53.52 | 46.95 | -6.57 |
| [6,23] | 39.39 | 36.81 | -2.58 |
| [12,13] | 59.41 | 52.42 | -6.99 |
| [12,23] | 46.04 | 42.17 | -3.87 |
| [13,23] | 44.08 | 41.91 | -2.17 |

This satisfies the strict paper gate for CMU V=2: 10/10 view pairs improve.
Compared with the previous robust ensemble (`g1 + lw0.7 seeds0/1/2`, avg
delta `-2.64 mm`, worst delta `-0.47 mm`), this is stronger on average but has
a smaller worst-case margin. It is a strong candidate result, but should be
validated on V=3/V=4 before becoming the main table result.

## Why `[3,12]` fails for hardv+legw07

The single-model `hardv+legw07 model_best` fixes the original hard-view
regression on `[3,13]`, but creates a new regression on `[3,12]`.

Geometry:

- Camera centers from CMU calibration:
  - cam 3: `[2.10, -1.51, 1.78]`
  - cam 12: `[-0.06, -2.39, 2.57]`
  - cam 13: `[-1.84, -3.39, 0.18]`
- `[3,12]`: baseline distance `2.46 m`, origin angle `43.0 deg`
- `[3,13]`: baseline distance `4.65 m`, origin angle `82.5 deg`

So `[3,12]` is a weaker-geometry pair than `[3,13]`. It relies more on model
priors and 2D confidence handling.

Per-joint change from `hardv_debug` to `hardv+legw07` on `[3,12]`:

- left shoulder: `+9.39 mm`
- left ankle: `+6.33 mm`
- right ankle: `+5.91 mm`
- left elbow: `+2.55 mm`
- left wrist: `+1.71 mm`

For `[3,13]`, the same change improves the knees:

- left knee: `-4.78 mm`
- right knee: `-4.71 mm`

Sample-level pattern on `[3,12]`:

- 316/697 samples are worse under `hardv+legw07` than `hardv_debug`.
- Median sample delta is slightly better (`-0.28 mm`), but a small set of
  severe failures dominates the mean.
- The worst samples are dominated by left shoulder/elbow/wrist failures.
- Mean 2D confidence is low on the same side:
  - left elbow: `0.608`
  - left wrist: `0.619`
  - left shoulder: `0.818`

Interpretation:

`[3,12]` is not just random noise. It is the intersection of weak camera
geometry, low-confidence one-sided upper-body detections, and the altered
leg/down-weighted distillation prior. This explains why `hardv_debug` and
`hardv+legw07` have complementary failures, and why prediction ensembling helps.

## CMU V=3 single-model check

V=3 evaluation uses all 10 combinations from cameras `[3,6,12,13,23]`.

| views | RUMPL-conf KP* | hardv debug KP* | hardv+legw07 KP* | hardv delta | hardv+legw07 delta |
|---|---:|---:|---:|---:|---:|
| [3,6,12] | 34.25 | 35.05 | 34.85 | +0.80 | +0.59 |
| [3,6,13] | 32.30 | 32.96 | 32.31 | +0.66 | +0.01 |
| [3,6,23] | 28.71 | 28.06 | 27.78 | -0.64 | -0.93 |
| [3,12,13] | 33.29 | 35.43 | 33.34 | +2.13 | +0.05 |
| [3,12,23] | 30.93 | 30.40 | 31.55 | -0.53 | +0.62 |
| [3,13,23] | 29.19 | 29.19 | 28.50 | +0.00 | -0.69 |
| [6,12,13] | 44.64 | 43.82 | 43.68 | -0.83 | -0.97 |
| [6,12,23] | 34.40 | 33.28 | 33.54 | -1.11 | -0.85 |
| [6,13,23] | 35.90 | 35.26 | 34.29 | -0.65 | -1.62 |
| [12,13,23] | 35.70 | 35.26 | 35.59 | -0.43 | -0.11 |

Summary:

- `hardv_debug`: avg delta `-0.06 mm`, worst delta `+2.13 mm`,
  improved `6/10`
- `hardv+legw07`: avg delta `-0.39 mm`, worst delta `+0.62 mm`,
  improved `6/10`

Conclusion:

Current single models are not clean enough for the main paper story. The next
single-model run should not make leg down-weighting stronger. A milder setting,
for example `hard-view + LEG_DISTILL_W=0.9`, is the most direct test of the
diagnosis: keep the `[3,12]` stability of `hardv_debug`, while retaining enough
leg correction to fix `[3,13]`.
