# Stage 2 preparation: H36M-Occl and Occlusion-Person (2026-08-22)

## 1. Entry condition

The HRNet direct-C2 and high-LR Joint-Query A/B evaluations have finished and
none improves V2/V3/V4 uniformly. Global Joint-Query is therefore rejected for
the HRNet frontend. Stage 2 uses the frozen HRNet C2 + E2-C2 + H18 result
`37.704/29.231/27.219 mm` and the frozen ResNet result below. H36M-Occl is a
zero-shot test and must not be used to tune model weights, select epochs, or
reselect the Stage-1 checkpoints.

The corrected ResNet clean temporal row is already frozen at
`31.215/22.008/19.971 mm` for V2/V3/V4.  It uses a centered T=9 window and is
therefore an internal/non-causal result until a past-only evaluation is added.

## 2. What GBT actually specifies

GBT trains on clean H36M subjects S1/S5/S6/S7/S8 and tests on S9/S11.  Its
H36M-Occl test set is produced by placing a white square over each projected
2D joint independently with probability `0.1`.  The frozen 2D detector is then
run on the occluded image.  It reports absolute, unaligned All-17 MPJPE and
averages all camera combinations:

| GBT Table II | input | V2 | V3 | V4 |
|---|---|---:|---:|---:|
| Algebraic Triangulation | ResNet-152 H36M-FT | 163.3 | 39.5 | 27.9 |
| GBT | ResNet-152 H36M-FT | 39.1 | 33.4 | 31.3 |
| Algebraic Triangulation | HRNet-W32 COCO | 217.3 | 72.4 | 54.1 |
| GBT | HRNet-W32 COCO | 42.3 | 34.5 | 31.6 |

GBT does **not** publish the square side length, random seed or H36M-Occl
generation code.  This is not the same as saying that synthetic H36M
occlusion has no precedent: several papers use an occluded H36M derivative,
but their corruption processes are different (see the literature audit
below).  Our table is therefore named **GBT-described, algebraically
calibrated reconstruction**: it follows every disclosed GBT rule, and freezes
the missing square size/seed by the published Algebraic Triangulation control;
it is not an exact reproduction of an unreleased official payload.
References [31]/[41] establish related synthetic-occlusion practice, but do
not resolve GBT's omitted square parameters.

### 2.1 Literature audit: why the same name does not imply the same test set

The phrase `H36M-Occl`, `H36M-Occluded` or `Human3.6M-Occluded` is used for
multiple, non-equivalent protocols:

| source | public occlusion rule/code | same as GBT's white-square p=0.1? |
|---|---|---|
| GBT (FG 2024) | For each projected 2D joint, independently place a white square with probability 0.1; side length, seed and generator are omitted | **Target rule** |
| Sárándi et al. (IROS 2018) and its official `synthetic-occlusion` code | Random solid shapes, rectangles/circles/bars and Pascal-VOC object cut-outs; the code is image-level augmentation and is not a GBT H36M-Occl generator | No |
| Banik et al. (IJCNN 2023) | Masks selected joints for a specified number of frames (`q_j`, `q_f`, with temporal variants); evaluates single- and multi-frame models | No |
| Bragagnolo et al. (ECCV 2024 workshop) | Public `human3.6m-occluded` generator: two random Pascal-VOC objects in each selected view, three of four views occluded, random object size/location | No |

Thus there **are** many related synthetic-occlusion experiments, and a later
paper can legitimately cite one of those public benchmarks.  We should not,
however, merge their numbers into GBT Table II: the number of corrupted views,
whether the detector sees an occluded RGB image or only a masked 2D tensor,
object/rectangle geometry, and temporal persistence all change the task.
As of this audit, we found no public implementation that specifies the exact
GBT combination and releases the same H36M-Occl files.  This is why the
provenance qualifier is needed, not because GBT's idea is unprecedented.

Primary links: [GBT](https://arxiv.org/abs/2312.17106), [Sárándi et al.](https://arxiv.org/abs/1808.09316), [official synthetic-occlusion code](https://github.com/isarandi/synthetic-occlusion), [Banik et al.](https://arxiv.org/abs/2304.12069), and [Bragagnolo et al. / generator](https://github.com/laurabragagnolo/human3.6m-occluded).

For the Bragagnolo generator, “three of four views occluded” means exactly
three cameras are selected by `random.sample(range(4), 3)` for each processed
frame, while the fourth remains clean.  It does **not** mean that a 2-view
evaluation is clean.  If camera `A` is the clean one and `B/C/D` are occluded,
the six 2-view pairs contain three `(A,occluded)` pairs and three
`(occluded,occluded)` pairs; the four 3-view subsets contain three with one
clean camera and one with no clean camera.  We will report the official
all-combination mean and, separately, this mask-composition breakdown.

GBT is different: every camera is eligible for every joint independently with
marginal probability `0.1`; no camera is forced clean or forced occluded.  For
one joint, the probability that at least one of four views is masked is
`1-0.9^4=34.39%`, while for a particular 2-view pair it is `1-0.9^2=19%`.
Thus the Bragagnolo benchmark is a structured, severe view-level occlusion
test; GBT is a sparse, joint-level stochastic test.  They must be separate
tables, not treated as interchangeable H36M-Occl labels.

## 3. H36M-Occl generation protocol

The same deterministic mask realization must be used for HRNet and ResNet:

1. read the original H36M image and original camera calibration;
2. undistort the complete image with `K_new=K`, as in both clean frontends;
3. undistort the source GT 2D joint locations into the same pixel system;
4. for every `(subject, action, subaction, frame, camera, joint)`, draw an
   independent deterministic Bernoulli variable from a recorded seed.  GBT
   states the marginal probability but not cross-camera/cross-frame
   correlation, so this independence is another explicit reconstruction
   choice;
5. if selected with probability `0.1`, paste a clipped white square centered
   on that joint before detection/cropping.  This is our image-level
   interpretation of GBT's wording, supported by its Sárándi/Zhang citations;
   GBT does not explicitly state whether the mask is inserted in RGB before
   the detector or directly into the 2D tensor, so this insertion point is
   recorded as an implementation assumption rather than a claimed GBT fact;
6. ResNet uses the established annotation box and official 384 crop; HRNet
   reruns YOLOX-X and HRNet-W32 on the same masked full image;
7. export only coordinates, confidences and camera data, plus the joint mask
   and generation metadata for audit.

No old `20% token dropout`, confidence zeroing, limb deletion or direct
keypoint corruption is silently substituted for the image-level main row.
Those are separate robustness ablations because they bypass the frozen image
detector; a direct-2D-mask sensitivity row can be added if an author artifact
clarifies that GBT used that interpretation.

### Resolving the missing square size

Use only the published Algebraic Triangulation control—not our learned model—to
reconstruct the omitted preprocessing parameter:

- sparse S9/S11 validation, probability `0.1`, fixed seed;
- square side candidates `{0.10, 0.15, 0.20}` times the longer annotation-box
  side;
- rerun the ResNet-152 frontend and compute its V2/V3/V4 Algebraic row;
- select the single size closest to GBT's `163.3/39.5/27.9` control;
- freeze that size and seed before running RUMPL/Joint-Query/E2/H18;
- report the size, seed, mask count and mismatch to the published control.

This is protocol reconstruction from a published baseline, not selection on
our model's MPJPE.  A second fixed seed is reported as sensitivity if runtime
allows; the main method cannot choose the more favorable seed.

## 4. Evaluation order

### 4.1 Sparse spatial table

For each frozen frontend and selected clean model, evaluate:

1. confidence-weighted Algebraic Triangulation control;
2. RUMPL/H76 base;
3. `+Global Joint-Query`;
4. `+E2 identity-hinge`;
5. all V2, V3 and V4 combinations, action-equal and frame-weighted metrics.

Report clean and occluded rows together and include degradation
`MPJPE_occl - MPJPE_clean` and Negative View Rate.

### 4.2 Dense temporal table

Generate the 105,076-record dense occluded frontend only after the square size
is frozen.  First evaluate the current centered H18 as an explicitly
non-causal internal robustness result.  For fair comparison with GBT, add a
past-only fixed-lag T=9 evaluation: GBT inputs the latest observation plus
eight past frames and predicts the latest frame.  Centered future-context H18
must not be labelled as a strict GBT temporal comparison.

Report T=`1/2/3/6/9` on four views only after the causal path is verified; GBT
Table VII reference rows are clean `29.4/27.9/27.7/27.3/26.0` and H36M-Occl
`41.5/34.9/34.0/32.5/31.6`.

## 5. Occlusion-Person is a separate dataset

GBT Table III uses the UnrealCV Occlusion-Person dataset with 8 cameras and
reports V2/V3/V4/V8.  The local workspace currently contains AdaFuse code and
configuration but not the dataset payload.  This stage requires a separate
download/data audit and its official train/test sequences.  GBT evaluates all
joints, including joints invisible in every view; AdaFuse/RANSAC values count
only joints visible in at least two views.  We must report both `All joints`
and `>=2-view visible joints` rather than compare mismatched metrics.

Reference Table III values:

| method | V2 | V3 | V4 | V8 |
|---|---:|---:|---:|---:|
| RANSAC | 33.7 | 87.5 | 35.0 | 15.5 |
| ScoreFuse | 32.7 | 25.7 | 21.4 | 15.0 |
| AdaFuse | — | 26.2 | 19.7 | 12.6 |
| GBT HRNet | 30.8 | 22.9 | 19.6 | 14.2 |

## 6. Output contract

All artifacts will be written under
`/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/` with separate
`protocol_calibration`, `hrnet_sparse`, `resnet_sparse`, `hrnet_temporal`,
`resnet_temporal` and `occlusion_person` directories.  Every result must store
the clean checkpoint hash, frontend checkpoint hash, input PKL hash, mask
probability/size/seed, camera combinations, metric scope and causal/non-causal
temporal label.

Primary references:

- `/home/lixiaob/cjy/reference/Geometry-Biased Transformer(1).pdf`
- GBT-described H36M-Occl generation cites Sárándi et al. synthetic occlusion
  and Zhang et al. object-occluded pose work;
- local AdaFuse code:
  `/mnt/data/cjydata/reference_code/adafuse-3d-human-pose/`.

## 7. Executed protocol and first frozen results (2026-08-22)

The paper/code audit found no public GBT implementation or downloadable
H36M-Occl payload.  The public Sárándi implementation was archived at commit
`3d627bbbeb5dd548d3fecb775c869ab08133f422`, but it pastes random Pascal-VOC
objects and therefore is provenance only; it is not silently substituted for
GBT's white-square corruption.  The shared deterministic implementation is:

- `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/h36m_occlusion_protocol_20260822.py`;
- `p=0.1`, seed `20260822`, square side `0.15` times the longer annotation-box
  side, opaque white, applied after full-image undistortion and before the
  frozen detector;
- 13,819 selected joint masks out of 137,428 opportunities (`10.0554%`), with
  pixel-identical output on repeated generation;
- the identical mask identity rule is used by HRNet and ResNet.

The square fraction was frozen using only GBT's published ResNet algebraic
triangulation control.  Candidate fractions `0.10/0.15/0.20` produced
`69.016/32.678/29.505`, `136.743/38.890/35.724`, and
`223.866/52.920/49.120 mm`; the pre-registered equal-cardinality log-RMSE rule
selected `0.15`.  This still differs from GBT's `163.3/39.5/27.9 mm`, so the
the table must say **GBT-described, algebraically calibrated reconstruction**,
not strict official payload reproduction.  The qualifier identifies the
missing implementation parameters; it does not imply that synthetic H36M
occlusion is an original idea of this project.

All models below are the frozen clean Stage-I checkpoints; no weight,
temperature, epoch, mask, or seed was selected from H36M-Occl.

| frozen model, T=1 | input | clean V2/V3/V4 | H36M-Occl V2/V3/V4 | degradation |
|---|---|---:|---:|---:|
| GQ-RUMPL | ResNet-152† | 32.312/25.101/23.536 | 47.750/33.503/31.259 | +15.438/+8.402/+7.723 |
| GQ-RUMPL-E2 | ResNet-152† | 32.319/22.558/20.272 | **47.866/29.236/24.511** | +15.547/+6.678/+4.239 |
| C2 generator | HRNet-W32 | 38.686/30.943/28.629 | 47.213/36.095/32.668 | +8.527/+5.152/+4.039 |
| C2 + E2-C2 | HRNet-W32 | 38.700/29.486/27.274 | **47.252/33.509/29.739** | +8.552/+4.023/+2.465 |

E2 is essentially neutral at V2 (`+0.116 mm` ResNet and `+0.038 mm` HRNet),
but protects multi-view fusion under occlusion: relative to its matched frozen
generator it improves ResNet by `4.267/6.747 mm` at V3/V4 and HRNet by
`2.587/2.928 mm`.  This is a useful robustness result because it was obtained
without occlusion training.  The centered H18 T=9 rows remain pending and
must be reported separately as non-causal; a past-only implementation is
required for a strict comparison to GBT's causal temporal table.

Auditable outputs:

- protocol selection: `/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/protocol_selection.json`;
- ResNet T=1: `/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/selected_f015/resnet_spatial/`;
- HRNet T=1: `/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/selected_f015/hrnet_spatial/`;
- HRNet masked frontend: `/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/selected_f015/hrnet_frontend/`.

## 8. Dense T=9 frozen H18 results (2026-08-22)

The dense stride-5 frontends contain 105,076 validation records and use the
same frozen mask realization as the sparse table.  We reused the clean-trained
H18-lowLR checkpoints and did not train on H36M-Occl.  The current H18 model is
centered (`4` past + center + `4` future frames), so these are robustness
results, not the strict causal GBT protocol yet.

| frozen chain | H36M-Occl T=9 center baseline | H36M-Occl T=9 H18 | gain (mm) |
|---|---:|---:|---:|
| ResNet-152 + E2 | 47.422/29.201/24.473 | **43.416/26.991/22.985** | -4.007/-2.210/-1.488 |
| HRNet-W32 + E2 | 47.741/33.821/29.961 | **43.996/31.944/28.794** | -3.744/-1.877/-1.167 |

The three columns are V2/V3/V4 action-equal All-17 absolute MPJPE in mm.  The
improvement is consistent for both frontends and all camera cardinalities.  A
past-only T=9 model still needs to be trained on clean data before claiming a
strict GBT temporal comparison; the current result is retained as an internal
centered-window robustness row.

Outputs:

- ResNet T=9: `/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/selected_f015/resnet_temporal/evaluation/centered_h18_result.json`;
- HRNet T=9: `/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/selected_f015/hrnet_temporal/evaluation/centered_h18_result.json`;
- dense frontends completion marker: `/mnt/data/cjyoutput/gbt_occlusion_stage_20260822/selected_f015/dense_frontends_COMPLETED`.

## 9. Official-code Human3.6M-Occluded benchmark (2026-08-23)

The GBT white-square table above remains a documented reconstruction because
GBT did not release its generator.  We therefore added a second, independently
reportable robustness table based on the public generator from Bragagnolo et
al., *Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation*
(ACVR/ECCV Workshop 2024), which is also the Human3.6M-Occluded protocol used
by SkelSplat (WACV 2026).

Official source archives:

- generator: `/mnt/data/cjydata/reference_code/human3.6m-occluded`, commit
  `4a26258`;
- SkelSplat: `/mnt/data/cjydata/reference_code/SkelSplat`, commit `16ed180`;
- Pascal VOC 2012 archive MD5:
  `6cd6e144f989b92b3379bac3b3de84fd`.

The generator extracts segmented non-person, non-difficult, non-truncated VOC
objects and keeps masks with at least 500 pixels.  For each selected camera it
pastes two objects; each object's minimum dimension is sampled uniformly from
`0.5--1.0` times the subject-box minimum dimension, and its center is uniform
inside the subject box.  The seed is `42`.  Occ-2 selects exactly two of the
four cameras per synchronized frame; Occ-3 selects exactly three.  Models are
trained only on clean H36M and evaluated zero-shot.

The public main generator cannot be executed literally on our layout and has
an internal frame-step inconsistency: its frame count is already computed
after `::64`, then the main loop applies a second step of 64.  Our adapter
therefore changes traversal only and directly visits all 2,021 synchronized
S9/S11 evaluation frames from the RUMPL pickle.  It imports the official
`load_occluders` and `occlude_with_objects` functions without modification,
converts only the annotation box from `xyxy` to `xywh`, preserves the global
Python RNG/seed, and records every selected camera:

- adapter:
  `OpenRUMPL_baseline_audit/generate_h36m_occ_official_adapter_20260823.py`;
- output root: `/mnt/data/cjyoutput/h36m_occ_official_20260823/`;
- Occ-2: 4,042 generated occluded JPEGs plus 4,042 clean links;
- Occ-3: 6,063 generated occluded JPEGs plus 2,021 clean links;
- frontends:
  `OpenRUMPL_baseline_audit/launch_h36m_occ_official_frontends_20260823.sh`;
- frozen spatial evaluation:
  `OpenRUMPL_baseline_audit/launch_h36m_occ_official_spatial_eval_20260823.sh`.

Occ-3-Hard is not yet generated.  SkelSplat describes it only as using larger
occluders, while the public generator exposes no separate hard-mode scale.
Until an author artifact or clarification provides that value, an inferred
scale sweep must be labelled a reconstruction and cannot be mixed with the
official Occ-2/Occ-3 table.

Published four-view absolute-MPJPE references from SkelSplat Table 4 are:

| method | Occ-2 | Occ-3 | Occ-3-Hard |
|---|---:|---:|---:|
| Algebraic Triangulation, ResNet-152 | 43.2 | 48.9 | 120.4 |
| TransFusion | 40.8 | 76.3 | 96.5 |
| RANSAC | 33.7 | 38.6 | 80.7 |
| Algebraic Triangulation, MeTRAbs | 36.0 | 39.0 | 67.5 |
| AdaFuse | 27.9 | 31.2 | 41.1 |
| Multi-view Pose Fusion | 33.4 | 36.7 | 37.8 |
| SkelSplat, MeTRAbs | 29.6 | 31.1 | 38.1 |
| SkelSplat, ResNet-152 | **24.6** | **27.0** | **34.8** |

These published values are four-view only.  Our primary comparison must also
be four-view absolute MPJPE.  V2/V3 averaged over all camera combinations are
additional analyses and must be stratified by how many selected cameras are
occluded; they are not numbers reported by SkelSplat or Bragagnolo et al.

### 9.1 Protocol correction after complete-history and control audit

The initial directory names `occ2` and `occ3` must not yet be interpreted as
the ordinary SkelSplat Occ-2/Occ-3 rows.  Fetching all 14 commits of the 2024
generator showed that its very first implementation already hard-coded two
objects per occluded view and contains the source comment `occ-hard is 2 obj`.
The original 2024 paper also describes two objects and its `37.8 mm` result is
placed by SkelSplat in the Occ-3-Hard column.  Conversely, neither the
SkelSplat repository, main paper, nor official supplement publishes the
ordinary-mode object count/scale change.

Therefore the generated data are conservatively relabelled as:

- `occ3`: public-repository two-object, three-view **strong/Hard candidate**;
- `occ2`: the same two-object corruption applied to two views, an unreported
  strong two-view analogue;
- `occ{2,3}_count1_candidate`: one-object inference for the ordinary variants,
  which is accepted only if the published algebraic controls are approached.

The frozen spatial results on the two-object strong data are retained as a
robustness stress test, not put into the ordinary SkelSplat comparison columns:

| two-object strong test | chain | direct V2/V3/V4 | +E2 V2/V3/V4 |
|---|---|---:|---:|
| 2 views occluded | HRNet C2 | 215.950/99.038/76.248 | 216.584/80.452/**49.074** |
| 2 views occluded | ResNet GQ | 155.618/100.546/94.793 | 156.019/64.539/**38.689** |
| 3 views occluded | HRNet C2 | 298.312/138.207/106.724 | 299.117/122.752/**80.942** |
| 3 views occluded | ResNet GQ | 208.422/132.789/124.119 | 209.345/100.427/**64.486** |

All values are action-equal All-17 absolute MPJPE in mm.  At V4, E2 lowers
the strong-test error by `27.174/25.782 mm` for HRNet and
`56.104/59.633 mm` for ResNet (two/three occluded views), despite never seeing
occlusion during training.  V2 remains neutral or slightly worse.  The HRNet
E2 two-seed variance becomes large under this OOD corruption (`7--9 mm` at
V3/V4), whereas the ResNet scorer remains much more stable; the paper must
report this rather than presenting only the seed mean.

An additional end-to-end official LT checkpoint control gave V4
`245.253/229.934 mm` on the two-object 2-view/3-view data, not SkelSplat's
`43.2/48.9/120.4` ordinary/hard controls.  Thus even the public two-object
generator adapted to our H36M image subset is not numerically strict enough
to claim the published Hard payload.  Likely causes include the unreleased
ordinary/Hard preprocessing distinction, different extracted frame offsets,
and frontend crop/undistortion details.  The safe label is **official-generator
stress-test reconstruction** until the algebraic control is matched.

### 9.2 Ordinary Occ-2/Occ-3 control alignment

The same official LT evaluator reproduces clean H36M at `25.616 mm`, versus
the SkelSplat table's `24.5 mm` Algebraic row (`+1.116 mm`).  This establishes
that the evaluator/checkpoint is sufficiently close and that the large
ordinary-occlusion mismatch came from the missing generator parameter.

We then selected the ordinary protocol using **only** SkelSplat's published
Algebraic controls (`43.2/48.9 mm` for Occ-2/Occ-3).  Our model results were
not computed during selection.  Both candidates used the scale interval
`0.2--0.5` motivated by the public Sárándi generator's lower scale and the
SkelSplat statement that Hard has larger occluders:

| calibration candidate | Occ-2 LT V4 | Occ-3 LT V4 | joint log-RMSE |
|---|---:|---:|---:|
| one object/view, scale 0.2--0.5 | 34.048 | 35.319 | 0.2851 |
| **two objects/view, scale 0.2--0.5** | **41.855** | **50.716** | **0.0341** |
| published target | 43.2 | 48.9 | — |

The two-object candidate simultaneously differs by only `-1.345/+1.816 mm`,
so it is frozen as the **SkelSplat control-aligned reconstruction** of ordinary
Occ-2/Occ-3.  The resulting interpretation is consistent with all public
evidence: the number of pasted objects remains two, ordinary uses the smaller
`0.2--0.5` scale, and the public 2024/Hard code raises it to `0.5--1.0`.

Frozen ordinary protocol roots:

- Occ-2: `/mnt/data/cjyoutput/h36m_occ_official_20260823/calib_c2_s020_050_occ2/`;
- Occ-3: `/mnt/data/cjyoutput/h36m_occ_official_20260823/calib_c2_s020_050_occ3/`.

These results still should not be called byte-identical official payloads:
the ordinary scale was recovered from published controls because SkelSplat
did not commit it.  The paper should report the two control mismatches next to
our rows, which makes the reconstruction auditable and prevents hidden
protocol selection.

### 9.3 Frozen model results on the selected ordinary protocol

After protocol selection was frozen, both clean-trained input/model chains
were evaluated without any occlusion training, fine-tuning, epoch selection,
temperature selection, or seed selection.  The two E2 checkpoint seeds are
averaged.

| selected ordinary test | frozen chain | direct V2/V3/V4 | +E2 V2/V3/V4 | E2 seed std V2/V3/V4 |
|---|---|---:|---:|---:|
| Occ-2 | HRNet C2 | 55.510/38.286/34.175 | 55.576/33.840/**29.406** | 0.034/0.062/0.137 |
| Occ-2 | ResNet GQ | 50.409/36.722/34.534 | 50.511/29.336/**23.383** | 0.014/0.090/0.044 |
| Occ-3 | HRNet C2 | 64.060/42.003/36.894 | 64.143/37.122/**31.600** | 0.039/0.636/0.822 |
| Occ-3 | ResNet GQ | 61.120/42.832/40.286 | 61.267/34.067/**26.092** | 0.043/0.063/0.039 |

The primary paper comparison is V4 because SkelSplat Table 4 reports only four
views.  On the matched ResNet-152 input family, our GQ+E2 reconstruction row
is `23.383/26.092 mm`, compared with SkelSplat's published
`24.6/27.0 mm`; the margins are `1.217/0.908 mm`.  This is the best row among
the published Table-4 methods on both ordinary variants, but the claim must be
phrased **on our SkelSplat control-aligned reconstruction**, with the
Algebraic mismatch `41.855 vs 43.2` and `50.716 vs 48.9` shown alongside it.
It is not legitimate to call the result an official-payload leaderboard win.

E2's occlusion contribution is much larger than its clean-data gain.  Relative
to the matched frozen direct generator, it improves ResNet V4 by
`11.151 mm` on Occ-2 and `14.194 mm` on Occ-3; HRNet improves by
`4.769/5.294 mm`.  V2 remains neutral (`+0.102/+0.147 mm` for ResNet), which
is expected because two-view tasks contain only one direct hypothesis and one
confidence-triangulation alternative, while V3/V4 provide a richer candidate
set and allow robust per-joint rejection of corrupted-view hypotheses.

Against each chain's matched clean E2 result, the degradation remains small at
V4: ResNet rises from `20.272` to `23.383/26.092 mm`
(`+3.111/+5.820`), and HRNet rises from `27.274` to
`29.406/31.600 mm` (`+2.132/+4.326`) for Occ-2/Occ-3.  This clean-to-corrupted
comparison is the most defensible robustness claim because detector, 3D
weights, candidate scorer and metric are identical within each row.

The official subset-stratified evaluator confirms that degradation tracks the
actual number of corrupted inputs rather than camera identity:

| ResNet GQ+E2 subset | 0 occluded views | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| Occ-2, V2 | 32.420 | 50.921 | 66.961 | — |
| Occ-2, V3 | — | 26.930 | 31.742 | — |
| Occ-3, V2 | — | 52.801 | 69.733 | — |
| Occ-3, V3 | — | — | 32.360 | 39.189 |

All values are action-equal All-17 absolute MPJPE in mm.  Formal outputs:

- protocol selection:
  `/mnt/data/cjyoutput/h36m_occ_official_20260823/ordinary_protocol_selection.json`;
- Occ-2 ResNet:
  `/mnt/data/cjyoutput/h36m_occ_official_20260823/calib_c2_s020_050_occ2/eval/resnet152_spatial/`;
- Occ-3 ResNet:
  `/mnt/data/cjyoutput/h36m_occ_official_20260823/calib_c2_s020_050_occ3/eval/resnet152_spatial/`;
- stratified evaluator:
  `OpenRUMPL_baseline_audit/evaluate_e2_occlusion_stratified_20260823.py`.

### 9.4 Occ-3-Hard reconstruction boundary

SkelSplat publishes a ResNet-152 Algebraic control of `120.4 mm` for
Occ-3-Hard.  The public 2024 scale `0.5--1.0` yields `229.934 mm` on our frame
subset, so it cannot be inserted into the Table-4 Hard column.  A finite
control-only calibration tested:

| two-object Hard scale | Algebraic V4 |
|---|---:|
| 0.3--0.8 | 203.958 |
| **0.4--0.75** | **111.351** |
| 0.4--0.78 | 337.885 |
| public 0.5--1.0 | 229.934 |
| paper target | 120.4 |

Rare maximum-size occluders cause discontinuous triangulation failures, so we
stop after this finite set rather than overfit the test control.  The closest
candidate `0.4--0.75` is frozen with a `-9.049 mm` mismatch and must be called
an **approximate Hard control-aligned reconstruction**.  Its evidence level is
lower than ordinary Occ-2/Occ-3; the mismatch must appear in any table or
caption.  Selected root:
`/mnt/data/cjyoutput/h36m_occ_official_20260823/calib_hard_c2_s040_075_occ3/`.
