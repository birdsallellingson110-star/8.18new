# Human3.6M-Occluded official V2/V3/V4 benchmark (2026-08-24)

## Decision

GBT Table II is not retained as the primary occlusion comparison because its
white-square side length, mask seed and generator are unavailable.  The very
large apparent V3/V4 advantage under our reconstructed masks is therefore not
treated as method evidence.

The primary few-view occlusion protocol is instead the generator released with
Bragagnolo et al., *Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose
Estimation* (ACVR at ECCV 2024).  The paper's Figure 6 evaluates 4, 3 and 2
available cameras after generating the four-camera test scene with three
randomly occluded views.

## Frozen protocol

- upstream generator commit: `4a2625805d6979e92283ee9c93ee476a1d8c6a82`;
- Human3.6M subjects: S9/S11;
- damaged S9 sequences removed by the existing `isdamaged` filter;
- 2021 synchronized frame groups after the existing stride-64 validation
  sampling;
- 3 of 4 camera views occluded independently per synchronized group;
- 2 random Pascal VOC 2012 objects pasted per occluded view;
- object scale uniform in `0.5--1.0` times the shorter person-box dimension;
- upstream traversal and Python RNG, seed `42`;
- all `6/4/1` camera subsets evaluated for V2/V3/V4;
- action-equal All-17 absolute MPJPE in millimetres, without alignment;
- learned methods are frozen clean-H36M checkpoints, with no occlusion
  fine-tuning or occlusion-based checkpoint/temperature selection.

Manifest:
`/mnt/data/cjyoutput/h36m_occ_official_20260823/occ3/protocol_manifest.json`.

## Comparison scopes

The primary table uses matched coordinates: every geometric baseline and our
fusion model receives exactly the same exported 2D coordinates, confidences and
camera parameters.  It includes confidence DLT, confidence ray intersection,
IRLS and a coordinate-equivalent port of the public AdaFuse RANSAC
(`10` iterations, `20 px` threshold, recorded seed).

Published Pose Fusion Table-4 V4 values are retained in a separate reference
table because they use each method's own frontend.  Figure-6 V2/V3 points are
curves without published numerical values and are not silently digitized into
an exact table.

## Implementation

- official generator adapter:
  `generate_h36m_occ_official_adapter_20260823.py`;
- frontend export:
  `launch_h36m_occ_official_frontends_20260823.sh`;
- matched geometric controls and AdaFuse RANSAC:
  `eval_h36m_controlled_triangulation_20260813.py`;
- frozen direct/E2 evaluation:
  `launch_h36m_occ_official_spatial_eval_20260823.sh`;
- audited table collector:
  `collect_posefusion_occ_v234_table_20260824.py`.

## Single-frame ablation results (not the final temporal baseline)

The two matched-input control jobs completed on all 2021 synchronized groups.
These rows isolate the spatial generator and E2 contribution.  They belong in
the ablation table only: the Stage-1 final methods are the complete
`GQ-RUMPL-E2-H18` ResNet chain and `C2-E2-H18` HRNet chain.  Their official
dense-occlusion evaluation is still required before this protocol has a main
method row.  The exact T=1 ablation table is:

| ResNet-152 matched coordinates | V2 | V3 | V4 |
|---|---:|---:|---:|
| Algebraic confidence DLT | 1249.311 | 240.155 | 203.438 |
| Confidence ray intersection | 259.889 | 143.507 | 129.557 |
| IRLS confidence rays | 259.882 | 118.661 | 87.111 |
| AdaFuse public RANSAC | 1249.732 | 179.786 | 111.966 |
| Frozen direct generator | **208.422** | 132.789 | 124.119 |
| Ours E2 | 209.345 | **100.427** | **64.486** |

| HRNet-W32 matched coordinates | V2 | V3 | V4 |
|---|---:|---:|---:|
| Algebraic confidence DLT | 1700.274 | 578.572 | 295.643 |
| Confidence ray intersection | 428.720 | 160.219 | 118.136 |
| IRLS confidence rays | 426.362 | 143.910 | 86.799 |
| AdaFuse public RANSAC | 3183.242 | 460.475 | 210.466 |
| Frozen direct generator | **298.312** | 138.207 | 106.724 |
| Ours E2 | 299.117 | **122.752** | **80.942** |

Relative to the frozen direct generator, ResNet E2 changes V2/V3/V4 by
`+0.923/-32.362/-59.633 mm`; HRNet E2 changes them by
`+0.804/-15.455/-25.782 mm`.  Against the strongest non-E2 matched control,
ResNet E2 is better by `18.234/22.626 mm` at V3/V4 and HRNet E2 by
`15.455/5.857 mm`.  It does not improve V2.

The V2 failure is consistent with this protocol rather than evidence that the
table was computed incorrectly.  Three of the four source views are occluded,
so a two-camera subset can contain two corrupted observations.  Two-view DLT
and two-view RANSAC have no redundant observation with which to identify the
outlier; this also explains their unbounded tail and very large action means.
E2 can select between its learned/direct and geometric candidates but cannot
manufacture an uncorrupted ray.  V3/V4 introduce redundancy, where E2's
candidate scoring becomes strongly useful.

The frozen two-seed E2 stratification confirms this mechanism:

| Frontend | V2: 1/2 selected views occluded | V2: 2/2 occluded | V3: 2/3 occluded | V3: 3/3 occluded |
|---|---:|---:|---:|---:|
| ResNet-152 | 161.016 | 257.675 | 88.270 | 136.899 |
| HRNet-W32 | 215.084 | 383.150 | 105.571 | 174.295 |

Thus the large V2 mean is not caused by one fixed bad camera: it rises sharply
when the selected subset contains no clean view.  The corresponding audited
files are `resnet152_spatial/e2_stratified.json` and
`hrnet_spatial/e2_stratified.json` under the benchmark `eval` directory.

Published Pose Fusion Table-4 V4 references are `80.7` RANSAC, `127.4`
Algebraic, `41.1` AdaFuse, `96.5` TransFusion and `37.8` Pose Fusion.  ResNet E2
therefore beats the published RANSAC, Algebraic and TransFusion numbers but not
AdaFuse or Pose Fusion; HRNet E2 only clearly beats Algebraic and TransFusion.
These are external references, not matched-input claims, because the published
methods use their own frontend.  Figure 6 includes V2/V3 curves but does not
publish their exact numerical values, so they are not quoted as exact results.

The primary table is single-frame.  Existing H18 uses a dense stride-5 T=9
cache, whereas this official run has 2021 stride-64 groups and independently
sampled masks.  H18 is therefore deferred to a separately generated dense,
temporally audited occlusion split rather than being mixed into this table.

## Provenance of the GBT Algebraic Triangulation rows

GBT's `Alg. Tri. [10]` cites Iskakov et al., *Learnable Triangulation of Human
Pose* (ICCV 2019).  That paper reports the clean filtered-H36M absolute result
for its learned-confidence algebraic branch (`19.2 mm` with the available
cameras), but it does not contain GBT's all-combination V2/V3 table or GBT's
white-square H36M-Occl experiment.  GBT states that it uses the Iskakov
ResNet-152 detector poses and applies Algebraic Triangulation, and separately
applies the same baseline to its HRNet poses.  It then evaluates all camera
combinations.  Consequently, GBT Table-I V2/V3 and all Table-II occlusion
numbers are GBT-side reruns, not numbers copied from the 2019 paper.

The public Iskakov model/code makes the ResNet frontend and solver recoverable,
but GBT does not publish the white-square side length, exact placement code or
random seed.  Its Table-II payload therefore cannot be reproduced exactly from
the paper alone; this is why our official-code occlusion protocol records its
own generator commit and random seed and keeps external GBT values separate.

Final machine-readable and Markdown tables:

- `/mnt/data/cjyoutput/h36m_occ_official_20260823/occ3/eval/posefusion_v234_table.json`;
- `/mnt/data/cjyoutput/h36m_occ_official_20260823/occ3/eval/posefusion_v234_table.md`.
