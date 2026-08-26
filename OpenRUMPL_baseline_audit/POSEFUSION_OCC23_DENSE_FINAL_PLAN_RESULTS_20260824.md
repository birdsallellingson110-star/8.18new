# Dense Human3.6M VOC Occ-2/Occ-3 final benchmark

## Frozen decision

This is the Stage-2 primary occlusion benchmark.  The main method rows must use
the complete Stage-1 clean-trained baselines:

- ResNet-152: `GQ-RUMPL -> E2 identity safeguard -> H18`, T=9;
- HRNet-W32: `C2 RUMPL -> E2-C2 -> H18`, T=9.

Direct generator and E2 T=1 rows are ablations only.  No checkpoint, epoch,
temperature or protocol parameter may be selected from occlusion results.

## Dense protocol

- H36M S9/S11; standard damaged S9 sequences already filtered;
- dense `annot_temporal_5_5`, 26,269 synchronized four-camera groups;
- all 6/4/1 camera combinations for V2/V3/V4;
- two Pascal-VOC objects pasted per selected view;
- Occ-2 masks two of four source views; Occ-3 masks three of four;
- object scale uniform in 0.2--0.5 times the shorter person-box side;
- public generator placement/blending and Python RNG, seed 42;
- T=9 centered window, frame stride 5;
- action-equal All-17 absolute MPJPE, no root or Procrustes alignment.

Publication wording: we follow the public Human3.6M-Occ generation procedure,
state the complete parameters and seed in the implementation details, and
report all methods under the same generated inputs.  Generator-recovery and
control-difference diagnostics stay in the internal audit record and are not
expanded in the main paper.

## Planned primary table

| Method/input | Occ-2 V2 | Occ-2 V3 | Occ-2 V4 | Occ-3 V2 | Occ-3 V3 | Occ-3 V4 |
|---|---:|---:|---:|---:|---:|---:|
| Alg. Tri. (ResNet-152) | pending | pending | pending | pending | pending | pending |
| Ours complete (ResNet-152), T=9 | pending | pending | pending | pending | pending | pending |
| Alg. Tri. (HRNet-W32) | pending | pending | pending | pending | pending | pending |
| Ours complete (HRNet-W32), T=9 | pending | pending | pending | pending | pending | pending |

## Temporal ablation table

Every T=1/T=9 pair below will be recomputed on the same 26,269 target frames.

| Input/protocol | E2 center T=1 V2/V3/V4 | complete T=9 V2/V3/V4 | H18 gain |
|---|---:|---:|---:|
| Occ-2 ResNet-152 | pending | pending | pending |
| Occ-3 ResNet-152 | pending | pending | pending |
| Occ-2 HRNet-W32 | pending | pending | pending |
| Occ-3 HRNet-W32 | pending | pending | pending |

## Published four-view comparison candidates

Exact values below are from SkelSplat WACV 2026 Table 4.  The first block is
the strongest comparison scope for our ResNet line; the second block uses each
method's own frontend and is external context rather than matched input.

### ResNet-152 family primary comparison

Temporal context is part of our method and does not change the 2D input
fairness boundary.  The complete T=9 result is therefore placed directly in
the primary comparison table; `T` is reported explicitly.  Its T=1 counterpart
is retained in the ablation table to isolate the temporal gain.

| Method | T | Occ-2 V4 | Occ-3 V4 |
|---|---:|---:|---:|
| Alg. Triangulation (ResNet-152) | 1 | 43.2 | 48.9 |
| RANSAC (as in AdaFuse) | 1 | 33.7 | 38.6 |
| AdaFuse (ResNet-152) | 1 | 27.9 | 31.2 |
| SkelSplat (ResNet-152) | 1 | **24.6** | **27.0** |
| Ours complete (ResNet-152) | 9 | pending | pending |

### Method-specific frontend references

| Method | Occ-2 V4 | Occ-3 V4 |
|---|---:|---:|
| Alg. Triangulation (MeTRAbs) | 36.0 | 39.0 |
| TransFusion | 40.8 | 76.3 |
| MV Pose Fusion | 33.4 | 36.7 |
| SkelSplat (MeTRAbs) | 29.6 | 31.1 |

SkelSplat explicitly states that its ResNet-152 variant uses the same
H36M-trained model as AdaFuse.  Published methods without an exact V2/V3 value
remain in the V4 comparison; no value is inferred from a plot.

## Code and output locations

- full resumable launch:
  `launch_posefusion_occ23_dense_all_20260824.sh`;
- generator:
  `launch_posefusion_occ23_dense_generate_20260824.sh`;
- two frozen frontends:
  `launch_posefusion_occ23_dense_frontends_20260824.sh`;
- complete baseline and Algebraic evaluation:
  `launch_posefusion_occ23_dense_final_eval_20260824.sh`;
- audited collector:
  `collect_posefusion_occ23_dense_final_table_20260824.py`;
- output root:
  `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/`;
- final tables (after completion):
  `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/final_occ23_table.{json,md}`.
