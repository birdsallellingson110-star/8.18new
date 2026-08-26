# ResNet two-view repair plan and execution record (2026-08-22)

## Objective

The primary objective is to remove the ResNet V2 reversal while preserving its
large V3/V4 advantage. A valid model must improve V2, V3 and V4 under the same
coordinate/confidence/ray input protocol. Selecting favorable camera pairs or
training a separate K=2-only model does not satisfy this objective.

## Evidence and current baseline

The ResNet ray anchor is better than the HRNet ray anchor for every camera pair.
The reversal is introduced by the learned H76 fusion: on the low-parallax pairs
0-3 and 1-2 it remains close to the anchor instead of applying the strong human
pose prior learned by the HRNet model. The other four V2 pairs and all V3/V4
subsets improve with ResNet.

| model | V2 | V3 | V4 |
|---|---:|---:|---:|
| ResNet E2-C2 identity-hinge | 40.8620 | 23.1725 | 20.4261 |
| ResNet geometry-uncertainty E2-C2 | **40.6668** | **23.0912** | **20.3207** |
| change | -0.1951 | -0.0813 | -0.1055 |

The geometry token therefore establishes the required monotonic direction, but
its V2 effect is too small. Relative to the HRNet E2-C2 V2 value 38.959 mm, the
remaining ResNet gap is about 1.71 mm.

## Failure constraints carried forward

- K=2-only 300k-update training reached 37.886 mm V2 but destroyed V3/V4
  (62.215/46.217 mm). It proves a stronger two-view pose prior is learnable,
  while also proving that cardinality replay is mandatory.
- Mixed-cardinality 300k-update training improved all three metrics by
  2.771/0.602/0.146 mm in the prior controlled study. Long optimization is a
  valid baseline control, not the final architectural contribution.
- Frozen post-H76 query residuals and small frozen geometry adapters changed
  results by less than useful margins. New branches must train jointly with the
  fusion backbone rather than operate only after a frozen prediction.
- Pre-VFT semantic-graph mixing and naive global-JV attention damaged the ray
  identity and are not repeated.

## Active experiments

### A: long mixed-cardinality geometry H76

- 123 epochs, approximately 299,874 optimizer updates;
- first eight epochs K=2, then V2/V3/V4 replay ratio 3:1:1;
- ResNet-152 coordinates/confidence, tri-anchor, centered Pluecker and the
  ray-normal geometry-uncertainty token;
- same checkpoint evaluated on every V2/V3/V4 combination.

This isolates whether the geometry-aware model is currently under-optimized.
It is running on GPU0 at:
`/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/A_long_mixed_geom/`.

### B: jointly trained global joint-query residual

- H76 remains the main RUMPL path;
- a GBT/MVGFormer-style joint query directly reads all pre-VFT joint-view ray
  tokens and predicts a bounded residual;
- unlike the failed E3 adapter, all parameters are trained jointly from scratch
  on the ResNet input;
- first eight epochs K=2, then 3:1:1 replay protects V3/V4;
- 20-epoch screen; extend only if V2 improves without V3/V4 regression.

This job is queued for GPU1 immediately after the active geometry-E2 H18 run:
`/mnt/data/cjyoutput/gbt_aligned_resnet_20260822/v2_repair/B_global_query_full/`.

## Decision thresholds

The direct H76 reference for the geometry line is 41.0506/25.9369/24.0002 mm.
An experiment is retained only if all three values decrease. V3/V4 regression
larger than 0.2 mm stops that branch even if V2 improves. A retained generator
is then exported to the same E2-C2 scorer; the immediate target is below
38.959 mm V2 while keeping V3/V4 below 23.091/20.321 mm.

The common launcher is:
`/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_resnet_v2_repair_pair_20260822.sh`.

## First successful result: jointly trained global query (2026-08-22)

The B screen has completed. Its strict direct evaluation is:

| method | V2 | V3 | V4 |
|---|---:|---:|---:|
| ResNet-H76 direct reference | 41.4704 | 26.0806 | 24.1573 |
| B: full global joint-query residual | **32.3121** | **25.1006** | **23.5364** |
| change | **-9.1583** | **-0.9800** | **-0.6209** |

This is the first ResNet experiment that lowers all three cardinalities in one
checkpoint. It is a direct H76-path result before E2 scoring; the successful
checkpoint is being passed through the same 22-candidate E2-C2 protocol now at
`v2_repair/B_global_query_full/e2_c2/`.

## Matched final HRNet/ResNet pipeline

The final comparison is now frozen as a strict one-variable experiment. Both
frontends use the same global Joint-Query generator configuration (20 epochs,
first 8 epochs K=2, then 3:1:1), the same 22-candidate identity-hinge E2-C2
scorer and temperatures, and the same H18 T=9 temporal module. Only the frozen
2D coordinate/confidence frontend differs. The full contract and launchers are
recorded in `JOINT_QUERY_MATCHED_FRONTENDS_PLAN_20260822.md`.
