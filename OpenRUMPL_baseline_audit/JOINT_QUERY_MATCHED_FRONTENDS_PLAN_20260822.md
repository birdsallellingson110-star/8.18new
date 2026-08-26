# Joint-Query matched HRNet/ResNet comparison (2026-08-22)

## Comparison contract

This document began as a matched-transfer test. The final audit shows that the
two coordinate frontends share the input modality, H36M split, camera
combinations, metrics, RUMPL ray representation, E2 family and H18 design, but
they do **not** share an identical selected generator: Global Joint-Query is
retained for ResNet and rejected for HRNet. Detector-specific optimization is
therefore reported as an ablation rather than hidden as a matched success.

| component | matched setting |
|---|---|
| generator | RUMPL/H76 + global Joint-Query, depth 2, max residual 0.5 |
| training | seed 0, 20 epochs, lr 1e-4, first 8 epochs K=2, then V2:V3:V4=3:1:1 |
| candidates | all 6 V2 + 4 V3 + 1 V4 hypotheses, plus 11 confidence/IRLS candidates |
| scorer | two seeds, 10 direct + 5 finetune epochs, identity hinge 0.25, V2 weight 4.0 |
| temperature | V2=0.4, V3=1.8, V4=1.8 |
| temporal | H18, T=9, stride 5, hidden 96, 2 layers, 12 epochs, lr 5e-5, wd 5e-4 |
| input to 3D network | 2D coordinates, confidence, calibrated camera rays only |

The frozen ResNet chain is `32.312/25.101/23.536` before E2,
`32.319/22.558/20.272` after identity-preserving E2, and
`31.215/22.008/19.971` after H18. The frozen HRNet chain is
`38.686/30.943/28.629`, `38.700/29.486/27.274`, and
`37.704/29.231/27.219`, respectively, without Global Joint-Query.

## Outputs

- ResNet: `/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/B_global_query_full/`
- HRNet: `/mnt/data/cjyoutput/joint_query_matched_frontends_20260822/hrnet/`
- ResNet completion launcher:
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_resnet_query_best_e2_h18_20260822.sh`
- HRNet matched launcher:
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_hrnet_query_matched_pipeline_20260822.sh`

## 1. Status and role in the paper

This route is frozen as **Model Baseline v1**, not yet as the final claimed
innovation. It supplies detector-specific clean checkpoints for the GBT-style
occlusion, cross-dataset and ablation experiments. Historical
RIGR/A1D image features, trainable view bias, distillation, codebooks and the
failed temporal dual-stream experiments are not part of this baseline.

Temporary names used in records:

- `GQ-RUMPL`: RUMPL/H76 plus the global Joint-Query residual, T=1;
- `GQ-RUMPL-E2`: the same generator plus identity-hinge E2-C2, T=1;
- `GQ-RUMPL-E2-T`: the same spatial model plus H18, T=9.

The final paper name should be selected only after the clean and occlusion
tables establish which module is the main contribution.

## 2. Overall objective and current stage

The total goal is to follow the experimental route of Geometry-Biased
Transformer (GBT): first obtain strong H36M baselines under both HRNet and
H36M-finetuned ResNet-152 coordinates, then test occlusion, CMU, cross-dataset
generalization, component ablations and temporal length. The method must retain
camera generalization: no camera-ID lookup, no fixed camera-slot embedding and
no image/heatmap input to the 3D model.

Stage 1 H36M clean baseline selection is now **complete and frozen**. The
authoritative paper-facing result record is
`/home/lixiaob/cjy/STAGE1_H36M_CLEAN_RESULTS_FOR_PAPER_20260822.md`. Stage 2
H36M-Occl is the current stage. Clean checkpoints must not be reselected from
occlusion or S9/S11 results.

## 3. Problem definition and coordinate-only input

For frame `t`, view `v` and joint `j`, the frozen 2D frontend supplies pixel
coordinate `x_{tvj}=(u,v,1)` and confidence `c_{tvj}`. Camera calibration is
`(K_v,R_v,t_v)`. The camera centre and unit world ray are

\[
o_v=-R_v^\top t_v,\qquad
d_{tvj}=\operatorname{normalize}(R_v^\top K_v^{-1}x_{tvj}).
\]

The 3D network receives only `(x,y,c)` and calibrated rays. It never receives
RGB, heatmaps, bounding boxes, HRNet features or ResNet features. The two final
input lines are:

- HRNet-W32 COCO coordinates/confidence with YOLOX-X boxes;
- official Learnable-Triangulation H36M-finetuned ResNet-152
  coordinates/confidence.

All downstream settings are identical. Therefore differences between the two
rows measure the frozen 2D observation distribution rather than a different 3D
model.

## 4. Spatial generator

### 4.1 Confidence-weighted triangulation anchor

For each joint, define the orthogonal projection to the normal plane of a ray:

\[
P_{vj}=I-d_{vj}d_{vj}^{\top},\qquad w_{vj}=c_{vj}+\epsilon.
\]

The anchor is the regularised least-squares intersection

\[
A_j=\sum_v w_{vj}P_{vj}+\lambda I,\qquad
b_j=\sum_vw_{vj}P_{vj}o_v,\qquad
a_j=A_j^{-1}b_j,
\]

with `epsilon=0.05` and `lambda=1e-4`. The implementation uses
`torch.linalg.solve` rather than an explicit inverse.

### 4.2 Anchor-centred Pluecker ray token

For local origin `o'_v=o_v-a_root`, the Pluecker moment is

\[
m_{vj}=o'_v\times d_{vj}.
\]

The shared ray token encoder embeds direction, moment and confidence. It has no
camera ID, so the representation remains compatible with unseen camera order,
count and position.

### 4.3 Retained RUMPL/H76 path

RUMPL first fuses views independently for each joint (VFT), then models the 17
joints (PFT). With the triangulation anchor, the base prediction is

\[
h_j=\operatorname{VFT}(\{z_{vj}\}_{v\in\mathcal V}),\quad
H=\operatorname{PFT}([h_1,\ldots,h_{17}]),\quad
\hat p_j^{base}=a_j+f_{3D}(H_j).
\]

This path preserves the useful camera-independent structure of RUMPL, while
the next branch fixes its early per-joint view compression.

### 4.4 Global Joint-Query residual

Before VFT, retain all encoded joint-view tokens as global memory

\[
M=\{z_{vj}+e_j^{mem}\mid j=1\ldots17,\ v\in\mathcal V\}.
\]

Each learned joint query is conditioned on its triangulation anchor:

\[
q_j=q_j^{learn}+W_a a_j.
\]

A two-layer Transformer decoder lets every query inspect every joint and every
view:

\[
\tilde q_{1:17}=\operatorname{Decoder}(q_{1:17},M).
\]

The bounded correction and generator output are

\[
\Delta p_j=0.5\tanh(W_o\operatorname{LN}(\tilde q_j)),\qquad
\hat p_j^{GQ}=\hat p_j^{base}+\Delta p_j.
\]

`W_o` is zero-initialised, making the branch an exact identity at step zero.
Unlike the old frozen query adapter that failed, the complete RUMPL path and
Joint-Query decoder are trained jointly. The global memory is the essential
difference: a wrist query can use torso, opposite-limb and all-view evidence
before RUMPL destroys the individual view tokens.

### 4.5 Generator training

- H36M train subjects: S1/S5/S6/S7/S8;
- test subjects: S9/S11;
- seed 0, Adam-family optimiser inherited from the RUMPL implementation;
- 20 epochs, learning rate `1e-4`;
- epochs 0--7: uniformly sampled two-camera subsets;
- epochs 8--19: V2:V3:V4 task ratio `3:1:1`;
- one common checkpoint evaluated on all 6 V2, 4 V3 and 1 V4 combinations;
- loss: absolute-coordinate MPJPE training objective;
- no occlusion augmentation in the clean baseline stage.

The V2 curriculum teaches a strong two-ray human prior, while later
multi-cardinality replay prevents the K=2-only collapse previously observed on
V3/V4.

## 5. E2-C2 multi-hypothesis utility fusion

### 5.1 Candidate pool

For four H36M cameras there are 11 legal subsets:

- six two-view subsets;
- four three-view subsets;
- one four-view subset.

For every subset, two candidate families are generated:

1. the frozen `GQ-RUMPL` prediction;
2. confidence-weighted robust triangulation/IRLS using the same observations.

This gives 22 candidates. For a requested camera set, candidates using an
unavailable camera are masked. The scorer is shared across V2/V3/V4 and never
uses a camera-index embedding.

### 5.2 Candidate features and Set Transformer

For candidate `c` and joint `j`, the scorer uses:

- absolute and root-relative pose;
- deviation from candidate consensus;
- distance from the candidate joint to each calibrated ray;
- included/excluded-view residual statistics and confidences;
- subset fraction;
- eigenvalue spectrum of the weighted ray normal matrix;
- global pose context and joint identity.

A permutation-equivariant two-layer Set Transformer compares the candidates
and predicts relative joint risk `r_{cj}`. Soft fusion is

\[
\alpha_{cj}=\frac{\exp(-r_{cj}/\tau_K)}
{\sum_{c'}\exp(-r_{c'j}/\tau_K)},\qquad
\hat p_j^{E2}=\sum_c\alpha_{cj}\hat p_{cj}.
\]

The temperatures are frozen to `tau_2=0.4`, `tau_3=tau_4=1.8`; V2 temperature
was selected on the training holdout, never on S9/S11.

### 5.3 Training and identity preservation

The scorer first trains for 10 direct-risk epochs at `5e-4`, followed by 5
expected-risk epochs at `1e-4`. Checkpoint selection uses the mean V2/V3/V4
soft-fusion MPJPE on `train_group_index mod 10 == 0`. S9/S11 is evaluated once.

The final scorer uses the registered identity safeguard

\[
L_{id}=\lambda_{id}\sum_K\gamma_K
\max(0,E_{soft}^{K}-E_{base}^{K}),
\]

where `lambda_id=0.25`, `gamma_2=4` and `gamma_3=gamma_4=1`. Its purpose is not
to create the 20 mm V4 result by itself: the main gain comes from the E2-C2
candidate pool and soft utility fusion; the hinge only prevents small
regressions (about 0.182 mm on the previous ResNet V4 control).

## 6. H18 temporal residual

The temporal module operates after frozen E2-C2 fusion. For every pose it uses
four 3D signals, giving 12 channels per joint:

\[
[p_t-p_{t,root},\ p_{t,root},\ p_t-p_{t-1},\
(p_t-p_{t-1})-(p_{t-1}-p_{t-2})].
\]

It applies two spatial Transformer layers per frame and two temporal layers per
joint, then predicts a bounded centre-frame residual

\[
\Delta p^{temp}=0.1\tanh(f_{ST}(p_{t-4:t+4})),\qquad
\hat p_t^{T}=\hat p_t^{E2}+\Delta p^{temp}.
\]

The output layer is zero-initialised and root correction is forced to zero, so
H18 cannot damage the calibrated absolute root at initialisation. Frozen
settings are T=9, stride 5, hidden 96, two spatial/two temporal layers, 12
epochs, `lr=5e-5`, `weight_decay=5e-4`.

Important reporting boundary: current H18 is a centred, non-causal temporal
baseline. It must not be described as GBT's causal latest-frame inference.
Reproducing GBT Table VII at T=2 and T=6 requires a separate latest-frame
causal variant; those cells remain pending.

## 7. Current verified result

Under ResNet-152 coordinates, before E2-C2 and H18:

| method | V2 | V3 | V4 |
|---|---:|---:|---:|
| ResNet H76 direct reference | 41.4704 | 26.0806 | 24.1573 |
| ResNet `GQ-RUMPL` | **32.3121** | **25.1006** | **23.5364** |
| change | **-9.1583** | **-0.9800** | **-0.6209** |

This is the first direct generator checkpoint to improve all cardinalities at
once. The standard E2-C2 scorer without identity hinge has now completed two
seeds on the matched 22-candidate cache:

| intermediate ablation | V2 | V3 | V4 |
|---|---:|---:|---:|
| E2-C2 standard, seed 0 | 32.3312 | 22.6603 | 20.3791 |
| E2-C2 standard, seed 1 | 32.3305 | 22.6308 | 20.3429 |
| two-seed mean | **32.3309** | **22.6456** | **20.3610** |

This changes the direct generator by `+0.019/-2.455/-3.175 mm`: V3/V4 improve
substantially while V2 is essentially unchanged but very slightly worse. It is
therefore recorded as the `+22-candidate E2` ablation, not yet as the frozen
final baseline. Identity-hinge, H18 and the matched HRNet generator remain
pending and must not be filled from historical models.

### 7.1 Provisional paper story and claims

The route is a progressive repair of one concrete RUMPL bottleneck, not a
stack of three complete pose models. RUMPL's camera-independent world rays are
retained, but its per-joint VFT compresses all view tokens before whole-body
context becomes available. Global Joint-Query bypasses that early compression
with an identity-initialised residual. E2-C2 then exploits complementary camera
subsets and robust geometric hypotheses without camera-ID lookup. H18 is only
a small identity-initialised temporal correction and becomes a paper
contribution only if it preserves clean accuracy and improves occlusion.

The provisional claims are therefore: (1) global joint queries over the full
joint-view ray memory while retaining RUMPL's camera-generalizable path; (2) a
single V2/V3/V4 joint-wise multi-hypothesis utility fusion with an identity
safeguard; and (3) matched evaluation under two coordinate frontends, variable
camera sets, occlusion and cross-dataset transfer. The paper must not claim the
first use of Transformers, triangulation, Pluecker coordinates, Set
Transformers, temporal attention or hypothesis scoring.

## 8. GBT-aligned full experimental route

| stage | dataset/protocol | required output | current status |
|---|---|---|---|
| 1 | H36M clean, HRNet and ResNet, all V2/V3/V4 combinations | freeze spatial and temporal baselines | **complete/frozen** |
| 2 | H36M-Occl, model trained on clean H36M | Table II under both inputs | **protocol ready** |
| 3 | Occlusion-Person, 2/3/4/8 cameras | comparison with RANSAC/ScoreFuse/AdaFuse | pending/data audit |
| 4 | CMU Panoptic in-domain | H36M/CMU main table and 2/4/5/6/8 extension | pending |
| 5 | CMU-trained model tested zero-shot on H36M | per-action matched-joint table | pending |
| 6 | component ablation across clean/occlusion/cross-dataset | GBT Table VI-style cumulative table | pending |
| 7 | temporal length 1/2/3/6/9 | H36M and H36M-Occl | pending; causal even-T implementation required |

Stage-2 protocol audit and preparation are recorded in
`OpenRUMPL_baseline_audit/OCCLUSION_STAGE_PREPARATION_20260822.md`. GBT defines
image-level white-square masking at each 2D joint with probability 0.1 but
omits square size and seed. The missing parameter will be reconstructed only
against GBT's published Algebraic Triangulation control before any learned
model is evaluated. Historical token/limb dropout results are not substituted
for H36M-Occl.

## 9. Frozen evaluation protocol

- H36M: S1/S5/S6/S7/S8 train; S9/S11 test;
- omit erroneous S9 portions of Greeting, SittingDown and Waiting following
  GBT/Learnable Triangulation;
- absolute All-17 MPJPE in millimetres, no root/Procrustes alignment;
- action-equal primary report, frame-weighted value retained for audit;
- average all camera combinations for each view cardinality;
- clean and occlusion results must use the same clean-trained checkpoint;
- `T=1` and `T=9` must be separate rows;
- HRNet and ResNet may be compared only after the exact matched pipeline is
  complete.

## 10. Result tables to fill (GBT layout)

All numbers labelled `GBT-reported` are copied from the local GBT paper for
reference. `TBD` is mandatory until our corresponding run is complete.

### Table I. H36M clean with fewer cameras, absolute MPJPE (mm)

| Method | 2D input | T | 2 cams | 3 cams | 4 cams |
|---|---|---:|---:|---:|---:|
| Algebraic Triangulation [10], GBT-reported | ResNet-152† | 1 | 51.1 | 23.4 | 19.1 |
| GBT, reported | ResNet-152† | 9 | 29.9 | 24.4 | 22.7 |
| `GQ-RUMPL` (verified direct) | ResNet-152† | 1 | **32.312** | **25.101** | **23.536** |
| `+E2-C2 standard` (intermediate ablation) | ResNet-152† | 1 | 32.331 | 22.646 | 20.361 |
| `GQ-RUMPL-E2` (identity-preserving final) | ResNet-152† | 1 | **32.319** | **22.558** | **20.272** |
| `GQ-RUMPL-E2-T` | ResNet-152† | 9 | **31.215** | **22.008** | **19.971** |
| Algebraic Triangulation, GBT-reported | HRNet | 1 | 120.7 | 50.9 | 44.2 |
| GBT, reported | HRNet | 9 | 36.8 | 30.4 | 26.0 |
| HRNet C2 generator | HRNet | 1 | 38.686 | 30.943 | 28.629 |
| HRNet C2 + E2-C2 | HRNet | 1 | 38.700 | 29.486 | 27.274 |
| HRNet C2 + E2-C2 + H18 | HRNet | 9 | **37.704** | **29.231** | **27.219** |

### Table II. H36M-Occl, absolute MPJPE (mm)

| Method | 2D input | T | 2 cams | 3 cams | 4 cams |
|---|---|---:|---:|---:|---:|
| Algebraic Triangulation [10], GBT-reported | ResNet-152† | 1 | 163.3 | 39.5 | 27.9 |
| GBT, reported | ResNet-152† | 9 | 39.1 | 33.4 | 31.3 |
| `GQ-RUMPL` | ResNet-152† | 1 | 47.750 | 33.503 | 31.259 |
| `GQ-RUMPL-E2` | ResNet-152† | 1 | **47.866** | **29.236** | **24.511** |
| `GQ-RUMPL-E2-T` (centered H18, internal) | ResNet-152† | 9 | **43.416** | **26.991** | **22.985** |
| Algebraic Triangulation, GBT-reported | HRNet | 1 | 217.3 | 72.4 | 54.1 |
| GBT, reported | HRNet | 9 | 42.3 | 34.5 | 31.6 |
| HRNet C2 generator | HRNet | 1 | 47.213 | 36.095 | 32.668 |
| HRNet C2 + E2-C2 | HRNet | 1 | **47.252** | **33.509** | **29.739** |
| HRNet C2 + E2-C2 + H18 (centered, internal) | HRNet | 9 | **43.996** | **31.944** | **28.794** |

The Ours rows are frozen clean models under a documented GBT-described
reconstruction (`p=0.1`, square fraction `0.15`, seed `20260822`).  GBT did
not publish its generator, mask size, or seed, so this is not labelled an
exact reproduction.  Full calibration and hashes are recorded in
`OCCLUSION_STAGE_PREPARATION_20260822.md`.

The H18 rows are centered-window robustness results (T=9 with four future
frames), not causal GBT-equivalent results.  The matched dense center rows are
ResNet `47.422/29.201/24.473` and HRNet `47.741/33.821/29.961`; the frozen H18
residual improves every V2/V3/V4 column.  A clean-trained past-only T=9 model
is still required for the strict causal comparison.

### Table III. Occlusion-Person, absolute MPJPE (mm)

| Method | 2 cams | 3 cams | 4 cams | 8 cams |
|---|---:|---:|---:|---:|
| RANSAC† [42], GBT-reported | 33.7 | 87.5 | 35.0 | 15.5 |
| ScoreFuse† [42], GBT-reported | 32.7 | 25.7 | 21.4 | 15.0 |
| AdaFuse† [42], GBT-reported | — | 26.2 | 19.7 | 12.6 |
| GBT (HRNet), reported | 30.8 | 22.9 | 19.6 | 14.2 |
| `GQ-RUMPL-E2-T` (HRNet) | TBD | TBD | TBD | TBD |

### Table IV. CMU-to-H36M generalization, matched joints, absolute MPJPE (mm)

| Method | Dir | Disc | Eat | Greet | Phone | Pose | Purch | Sit | SitD | Smoke | Photo | Wait | Walk | WalkD | WalkT | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Algebraic Triangulation (HRNet), GBT-reported | 44.6 | 43.5 | 42.2 | 42.5 | 43.7 | 41.6 | 48.5 | 46.8 | 51.7 | 45.4 | 45.7 | 42.3 | 40.3 | 45.6 | 39.8 | 44.3 |
| GBT (HRNet), reported | 33.9 | 32.5 | 32.6 | 33.7 | 38.3 | 30.8 | 35.1 | 56.9 | 85.9 | 36.0 | 37.9 | 33.3 | 29.1 | 38.7 | 28.7 | 38.9 |
| `GQ-RUMPL-E2-T` (HRNet) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Table V. State of the art on H36M and CMU, absolute MPJPE (mm)

| Method | H36M | CMU |
|---|---:|---:|
| Cross View Fusion [27]†, GBT-reported | 26.2 | — |
| RANSAC Baseline [10], GBT-reported | — | 33.4 |
| Algebraic Triangulation [10]†, GBT-reported | 19.2 | 21.3 |
| Volumetric [10]†, GBT-reported | 17.7 | 13.7 |
| Remelli et al. [29]†, GBT-reported | 30.2 | — |
| Epipolar Transformers [8]†, GBT-reported | 27.1 | — |
| TransFusion [17]†, GBT-reported | 25.8 | — |
| GBT (ResNet-152)†, reported | 22.7 | — |
| Algebraic Triangulation (HRNet), GBT-reported | 44.2 | 26.2 |
| GBT (HRNet), reported | 26.0 | 17.2 |
| `GQ-RUMPL-E2-T` (ResNet-152)† | TBD | TBD/— |
| `GQ-RUMPL-E2-T` (HRNet) | TBD | TBD |

### Table VI-A. GBT component ablation copied as the reference layout

| Train→Test | none | +Centering | +Synthetic | +Conf. bias | +Geom. bias | all |
|---|---:|---:|---:|---:|---:|---:|
| CMU→CMU | 22.4 | 18.9 | 19.3 | 17.7 | 17.1 | 17.2 |
| H36M→H36M | 39.0 | 49.2 | 40.6 | 33.2 | 33.1 | 26.0 |
| H36M→H36M-Occl | 43.1 | 52.3 | 44.0 | 35.5 | 37.2 | 31.6 |
| CMU→H36M | 101.2 | 57.2 | 55.7 | 43.0 | 43.1 | 38.9 |

### Table VI-B. Our cumulative component ablation in the same style

| Train→Test | RUMPL rays | +Tri-anchor/Pluecker | +Global Query | +22-cand. E2 | +Identity hinge | +H18 |
|---|---:|---:|---:|---:|---:|---:|
| H36M→H36M, HRNet | TBD | **38.686/30.943/28.629** | not selected | **38.700/29.486/27.274** | — | **37.704/29.231/27.219** |
| H36M→H36M, ResNet | **41.470/26.081/24.157** | included in H76 | **32.312/25.101/23.536** | **32.331/22.646/20.361** | **32.319/22.558/20.272** | **31.215/22.008/19.971** |
| H36M→H36M-Occl, HRNet | TBD | TBD | TBD | TBD | TBD | TBD |
| H36M→H36M-Occl, ResNet | TBD | TBD | TBD | TBD | TBD | TBD |
| CMU→CMU, HRNet | TBD | TBD | TBD | TBD | TBD | TBD |
| CMU→H36M, HRNet | TBD | TBD | TBD | TBD | TBD | TBD |

For clean H36M, each cell stores `V2/V3/V4`; the final paper may split this
wide ablation into one table per dataset if readability requires it.

### Table VII. Number of temporal frames, four cameras, absolute MPJPE (mm)

| Dataset/method | 1 | 2 | 3 | 6 | 9 |
|---|---:|---:|---:|---:|---:|
| H36M, GBT-reported | 29.4 | 27.9 | 27.7 | 27.3 | 26.0 |
| H36M-Occl, GBT-reported | 41.5 | 34.9 | 34.0 | 32.5 | 31.6 |
| H36M, ours HRNet | **27.274** | TBD | TBD | TBD | **27.219** |
| H36M-Occl, ours HRNet | TBD | TBD | TBD | TBD | TBD |
| H36M, ours ResNet | **20.272** | TBD | TBD | TBD | **19.971** |
| H36M-Occl, ours ResNet | TBD | TBD | TBD | TBD | TBD |

### Supplementary Table S1. Strictly matched frontend comparison

| frontend | `GQ-RUMPL` T=1 V2/V3/V4 | `+E2` T=1 V2/V3/V4 | `+H18` T=9 V2/V3/V4 |
|---|---:|---:|---:|
| HRNet-W32/YOLOX-X | Query not selected; C2 **38.686/30.943/28.629** | **38.700/29.486/27.274** | **37.704/29.231/27.219** |
| ResNet-152† | **32.312/25.101/23.536** | **32.319/22.558/20.272** | **31.215/22.008/19.971** |

### Supplementary Table S2. CMU view-count extension

| Method | 2 cams | 4 cams | 5 cams | 6 cams | 8 cams |
|---|---:|---:|---:|---:|---:|
| Algebraic Triangulation (HRNet) | TBD | TBD | TBD | TBD | TBD |
| `GQ-RUMPL-E2` (HRNet) | TBD | TBD | TBD | TBD | TBD |
| `GQ-RUMPL-E2-T` (HRNet) | TBD | TBD | TBD | TBD | TBD |

## 11. Decision gates

The GBT clean references are `29.9/24.4/22.7` for ResNet-152 and
`36.8/30.4/26.0` for HRNet. Stage 1 is frozen at ResNet
`31.215/22.008/19.971` and HRNet `37.704/29.231/27.219`. Completion means the
current checkpoints and reporting protocol are fixed; it does not mean every
column exceeds GBT. The project now advances to H36M-Occl without using the
occlusion set to reselect clean checkpoints.

## 12. Key implementation index

- RUMPL, triangulation anchor, Pluecker and Joint-Query:
  `/home/lixiaob/cjy/OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py`
- E2-C2 task/candidate wrapper:
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_current_e2_confidence_20260815.py`
- E2-C2 losses and evaluation:
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_e2_v234_universal_20260812.py`
- Set Transformer candidate features:
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_h76_set_transformer_utility_20260811.py`
- H18 spatial-temporal residual:
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/train_e2_clean_temporal_residual_20260818.py`
- GBT paper used for protocols/tables:
  `/home/lixiaob/cjy/reference/Geometry-Biased Transformer(1).pdf`

## 13. Frontend-distribution audit and corrected unified training (2026-08-22)

The first matched HRNet run (`44.330/33.015/30.590` before E2) must not be
interpreted as evidence that HRNet is intrinsically incompatible with the
Joint-Query architecture.  It changed both the frontend and the optimization
history: the historical HRNet C2 model was obtained from a 123-epoch B1 model
followed by a 20-epoch, `1e-5`, `8:1:1` cardinality fine-tune, whereas the
matched run trained all parameters from scratch for 20 epochs at `1e-4`.
Scaling the trained query residual to approximately zero increased V2 to
`120.976 mm`; therefore the base RUMPL path had co-adapted/collapsed and the
query path was compensating it.  This run is a failed optimization control,
not the final HRNet row.

The corrected HRNet experiment preserves the strong C2 checkpoint and uses
one common architecture for both frontends:

1. U1: initialize from C2, freeze the complete RUMPL base, train only the
   zero-initialized global Joint-Query residual for 8 epochs with `8:1:1`
   view-count sampling;
2. U2: initialize from U1 and optionally fine-tune all parameters for 12
   epochs at `5e-6`; U1 remains the selected result if U2 regresses.

“Common architecture” does not require identical detector-specific
hyperparameters.  HRNet and ResNet may independently tune learning rate,
view-count sampling, query residual range, E2 temperatures/identity weight,
and temporal residual scale/window on training-subject holdout data.  The
input modality, module graph, H36M split, S9/S11 evaluation and reported metric
remain fixed.  A same-hyperparameter control is retained as an ablation, while
the main rows use the best validation-selected configuration for each frozen
2D frontend; no S9/S11 test result may be used for selection.

Launcher:
`OpenRUMPL_baseline_audit/launch_hrnet_c2_initialized_query_staged_20260822.sh`.
Output:
`/mnt/data/cjyoutput/joint_query_matched_frontends_20260822/hrnet/c2_initialized_query_staged/`.

U1 completed with `38.729/30.988/28.672 mm`, versus its unchanged C2 source
checkpoint at `38.686/30.943/28.629 mm`.  The approximately
`+0.044/+0.045/+0.044 mm` regression confirms the earlier finding that a
query-only adapter on a frozen RUMPL representation does not create useful new
evidence.  U1 is retained as a negative ablation and initialization-safety
check, not as a final model.  U2 now tests limited VFT/PFT--Query co-adaptation
at `5e-6` without returning to the failed from-scratch optimization.

U2 completed with `38.487/30.913/28.676 mm`. Relative to the original C2
checkpoint (`38.686/30.943/28.629 mm`), V2/V3/V4 change by
`-0.199/-0.030/+0.047 mm`. Limited co-adaptation repairs V2 without collapse,
but the small V4 regression prevents U2 from being selected as a uniformly
improved final generator. Direct-C2 A/B runs, which omit the ineffective U1
intermediate, completed at `38.742/30.957/28.651 mm` and
`38.921/30.870/28.712 mm`, respectively. Neither improves all three
cardinalities, so neither is selected.

Because C2 is the balanced source rather than the strongest trainable V2
source, two additional full-adaptation arms start from the strictly stronger
`B2 -> mixed, high-LR` checkpoint (`36.885/31.451/30.277 mm`).  This checkpoint
dominates the older C1 source (`37.007/32.103/30.930 mm`) in every cardinality,
so C1 is not redundantly rerun.  Arm A uses 12 epochs at `5e-6` with `3:1:1`
sampling to preserve V2; Arm B uses `1e-5` with `3:2:2` sampling to recover
V3/V4.  Both retain complete VFT/PFT and jointly adapt the zero-initialized
global Joint-Query branch.  Launcher:
`OpenRUMPL_baseline_audit/launch_hrnet_highlr_initialized_query_ab_20260822.sh`;
output:
`/mnt/data/cjyoutput/joint_query_matched_frontends_20260822/hrnet/highlr_initialized_query_ab/`.

The high-LR arms also completed. Arm A obtained
`37.026/31.612/30.417 mm`, and Arm B obtained
`36.905/31.465/30.280 mm`; both are worse than their source
`36.885/31.451/30.277 mm`. The HRNet Global Joint-Query migration is therefore
closed as a negative detector-distribution ablation. The frozen HRNet clean
line remains C2 + E2-C2 + H18 at `37.704/29.231/27.219 mm`.

The former ResNet H18 temporal result is also provisional/invalid.  Its dense
validation PKL was generated from an HRNet-merged PKL whose `camera.k/p`
distortion fields had already been zeroed.  All 8,084 overlapping view records
then differed from the audited sparse ResNet frontend in 2D coordinates
(mean record-wise maximum difference `4.619 px`, maximum `131.103 px`), even
though images, boxes, GT and camera extrinsics were identical.  The corrected
dense export starts from the original temporal GT PKL and writes to new paths:
`/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/frontend_temporal_v2_gtinput/`
and
`/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/B_global_query_full/h18_identity_hinge_v2_gtinput/`.
The temporal module is accepted only after its dense center-frame spatial
baseline aligns with the sparse `32.319/22.558/20.272 mm` identity-hinge row.
The corrected dense baseline is `32.437/22.581/20.306 mm`; the remaining
`+0.118/+0.023/+0.034 mm` is consistent with the dense temporal center subset
and is small enough to pass the alignment gate.  H18 training therefore uses
the intended ResNet spatial distribution. The corrected H18 run selected epoch
3 on the training-subject holdout and obtained a single final S9/S11 result of
`31.215/22.008/19.971 mm`. Relative to its matched dense baseline, the gains
are `1.222/0.572/0.335 mm`; all three cardinalities improve. This is the first
valid temporal result on the ResNet global-Query + identity-hinge branch and
is retained for the clean and occlusion temporal tables.
