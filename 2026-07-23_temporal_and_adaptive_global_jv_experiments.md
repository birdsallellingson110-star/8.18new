# 2026-07-23 Temporal and Adaptive Global-JV Experiments

## Fixed reference

- Audited RUMPL R5 checkpoint:
  `/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999/R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/model_best.pth.tar`
- Strict CMU all-combination averages, All17/KP*: V2 `30.885/35.506`,
  V3 `23.039/25.159`, V4 `20.213/21.698`, V5 `18.746/20.091` mm.
- All experiments below use one unified random-V2-to-V5 model. They are not
  separately trained for a particular camera count.

## T2: absolute-feature post-VFT temporal refinement

- Frozen R5 backbone, nine frames, two per-joint temporal MHSA blocks.
- Only the temporal module is trained. The zero-initialized output starts as
  exact R5 identity.
- AMASS split is grouped by `source_npz`: 2718 training clips and 202
  validation clips from 149 held-out sources, with zero source overlap.
- Validation and checkpoint selection jointly cover V2/V3/V4/V5.
- AMASS best epoch 17, All17: `53.085/30.220/23.310/21.115` mm.
- AMASS static baseline: `71.198/40.402/31.019/29.515` mm.

Real CMU evaluation uses 503 complete nine-frame windows from the `matched`
annotation whose 2D poses, confidences, 3D targets, and camera parameters agree
with the strict official annotations on overlapping records. The
`matched_swapv3` annotation is excluded because it changes the coordinate
system.

| Views | Static All17 | T2 All17 | Delta | Improved combinations |
|---|---:|---:|---:|---:|
| V2 | 36.124 | 42.382 | +6.259 | 0/10 |
| V3 | 25.511 | 31.916 | +6.405 | 0/10 |
| V4 | 22.494 | 28.905 | +6.411 | 0/5 |
| V5 | 21.193 | 27.398 | +6.205 | 0/1 |

Conclusion: T2 learned an AMASS-specific absolute feature correction rather
than transferable motion. It is retained as a negative ablation and cannot be
reported as an accuracy improvement.

Artifacts:

- `/mnt/data/cjyoutput/temporal_exact_20260723/T2_r5_postvft_l9_oks05_grouped_allviews_seed0_20260723/`
- Real CMU result:
  `cmu_all_combinations.json` under the directory above.

## T3: motion-difference temporal refinement

Changes relative to T2:

- Temporal tokens are `F(t) - F(center)`, so absolute AMASS pose appearance
  cannot directly drive a correction.
- The predicted residual is multiplied by a learned motion-energy gate.
- A static sequence has exactly zero residual and therefore exactly recovers
  R5, including after training.
- Residual-to-R5 regularization weight is `0.05`.
- Other data, split, optimizer, frames, depth, seed, and V2-to-V5 training
  protocol are identical to T2.

Variant:
`T3_r5_motiondiff_l9_oks05_grouped_allviews_rp005_seed0_20260723`

Artifacts:

- `/mnt/data/cjyoutput/temporal_exact_20260723/T3_r5_motiondiff_l9_oks05_grouped_allviews_rp005_seed0_20260723/`
- Training service: `rumpl-temporal-t3-motion-20260723.service`
- Automatic CMU evaluation service:
  `rumpl-temporal-t3-cmu-eval-20260723.service`

Decision rule: retain temporal modeling only if real CMU all-combination means
do not regress and the gain is repeatable. Synthetic AMASS improvement alone
is insufficient.

Final real CMU result, All17/KP* delta:

| Views | Delta All17 | Delta KP* | Improved All17 combinations |
|---|---:|---:|---:|
| V2 | +0.119 | +0.109 | 1/10 |
| V3 | +0.097 | +0.109 | 0/10 |
| V4 | +0.096 | +0.104 | 0/5 |
| V5 | +0.096 | +0.102 | 0/1 |

T3 eliminates almost all of T2's approximately 6 mm domain regression, which
validates the diagnosis, but it still does not improve clean CMU MPJPE.
Temporal clean-accuracy experiments are therefore stopped. T3 may only be
revisited for a separately defined jitter/noise robustness experiment.

## J2: view-count-adaptive global joint-view fusion

Evidence motivating J2:

- J1 global-JV without fixed bias improved strict R5 at V3/V4/V5 by
  `-0.155/-0.305/-0.334` mm All17.
- J1 regressed V2 by `+0.165` mm All17 and `+0.396` mm KP*.
- A single scalar residual gate therefore applies too much global correction
  when only two views are available.

J2 keeps one unified random-V2-to-V5 model and makes the residual gate a
continuous function of available camera count. It initializes at `0.05` for V2
and `0.12` for V5, with a learned slope. No confidence bias, geometry bias,
distillation, token removal, or extra loss is enabled, so this isolates the
adaptive gate.

Variant:
`J2_adaptive_globalJV_d2_nobias_g005_012_seed0_20260723`

Artifacts:

- Log:
  `/mnt/data/cjyoutput/baseline_reaudit_20260722/J2_adaptive_globalJV_d2_nobias_g005_012_seed0_20260723.log`
- Training service: `rumpl-j2-adaptive-globaljv-20260723.service`

After training, J2 must be evaluated on all 26 CMU combinations with both
All17 and KP*. The module is retained only if it protects V2 while preserving
the J1 gains at V3-to-V5.

## Next evidence-based branches

1. If T3 transfers to CMU, add confidence stability to the motion gate and
   evaluate jitter/acceleration in addition to MPJPE.
2. If T3 remains neutral or harmful, stop temporal clean-accuracy work; retain
   it only as a robustness ablation.
3. If J2 protects V2, combine adaptive global-JV with a weak VFT geometry bias
   whose strength decreases with camera count.
4. If J2 still hurts V2, the next structural branch is a lightweight
   anatomical prior (bone-vector or symmetric-limb regularization), not a
   larger attention stack.

## S1: SkelSplat-style symmetric-limb regularization

S1 isolates the 3D structural symmetry term from SkelSplat on the strict RUMPL
architecture. For the COCO skeleton, it penalizes squared length differences
between left/right upper arms, forearms, thighs, and shins. The weight is
`0.5` in the meter-based RUMPL objective. No inference-time module, global-JV,
bias, distillation, or temporal component is enabled.

Variant: `S1_rumpl_symmetry_w05_seed0_20260723`

- Log:
  `/mnt/data/cjyoutput/baseline_reaudit_20260722/S1_rumpl_symmetry_w05_seed0_20260723.log`
- Service: `rumpl-s1-symmetry-20260723.service`
- Purpose: determine whether a weak anatomical prior improves ambiguous
  sparse-view poses before combining it with any architecture change.
- Required evaluation: all 26 CMU combinations, All17 and KP*, compared
  directly with strict R5.
