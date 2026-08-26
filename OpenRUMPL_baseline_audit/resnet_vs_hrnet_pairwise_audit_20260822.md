# HRNet / ResNet input swap: pairwise H76 audit

The following numbers are the mean All-17 absolute error (mm) of the frozen
H76 candidate in the validation candidate cache, before E2 scoring. They use
the same S9/S11 cache protocol and the same camera index order.

| camera subset | HRNet | ResNet | ResNet-HRNet |
|---|---:|---:|---:|
| 0,1 | 34.588 | 29.280 | -5.308 |
| 0,2 | 38.591 | 33.522 | -5.069 |
| 0,3 | 48.833 | 64.339 | **+15.506** |
| 1,2 | 47.926 | 69.722 | **+21.796** |
| 1,3 | 31.249 | 25.593 | -5.656 |
| 2,3 | 32.785 | 28.124 | -4.661 |
| 0,1,2 | 32.785 | 27.916 | -4.869 |
| 0,1,3 | 30.805 | 25.468 | -5.337 |
| 0,2,3 | 32.041 | 27.566 | -4.475 |
| 1,2,3 | 29.224 | 24.804 | -4.420 |
| 0,1,2,3 | 28.868 | 24.478 | -4.390 |

The six two-view subsets average 38.995 mm for HRNet and 41.763 mm for
ResNet. The degradation is therefore concentrated in two ResNet pairs,
(0,3) and (1,2); the other four pairs improve by about 5 mm. Every ResNet
three-view subset improves by 4.4--5.3 mm, and the four-view subset improves
by 4.39 mm. This explains why the averaged V2 result is worse while V3/V4
are much better: V2 averages all six pairs and has no redundancy to suppress
the two bad pair configurations, whereas V3/V4 include enough views for the
good hypotheses to dominate.

## Decisive anchor-versus-network check

The fixed camera geometry is not itself the explanation for ResNet becoming
worse than HRNet. A controlled comparison of the confidence/IRLS ray anchor
against the learned H76 output shows where the reversal is introduced:

| pair | HRNet ray anchor | HRNet H76 | ResNet ray anchor | ResNet H76 |
|---|---:|---:|---:|---:|
| 0,1 | 63.731 | 34.588 | 29.175 | 29.280 |
| 0,2 | 73.228 | 38.591 | 33.428 | 33.522 |
| 0,3 | 135.449 | **48.833** | 66.162 | **64.339** |
| 1,2 | 151.847 | **47.926** | 71.947 | **69.722** |
| 1,3 | 59.140 | 31.249 | 25.597 | 25.593 |
| 2,3 | 61.655 | 32.785 | 28.044 | 28.124 |

Lower is better. ResNet improves the raw ray anchor on every pair, including
the two weak pairs: approximately 135/152 mm becomes 66/72 mm. The reversal
appears after the learned H76 fusion. With noisy HRNet input, H76 learned a
strong skeleton/pose correction and reduces the weak pairs by about 87/104 mm.
With accurate ResNet input, H76 collapses to an almost identity correction:
its output is nearly the same as the ray anchor on all six pairs, so it does
not repair the remaining depth ambiguity. The ResNet E2 pool also contains
two nearly duplicate hypotheses (H76 and ray anchor), limiting what candidate
scoring can recover.

This resolves the apparent paradox: the ResNet 2D frontend did not make the
geometry worse. It made triangulation better, but the newly trained RUMPL
fusion failed to learn the geometry-conditioned pose-prior correction that
the HRNet-trained RUMPL had learned. The repair target is therefore the
fusion/residual path and candidate diversity, not the detector coordinates.

## Why GBT can improve the same six pairs

GBT's Table I also averages all six two-camera combinations, so its result is
not explained by selecting easier pairs.  Its ResNet-152 algebraic baseline is
51.1/23.4/19.1 mm for V2/V3/V4, while the complete model is
29.9/24.4/22.7 mm.  This is an explicit trade: the learned model improves V2
by 21.2 mm while becoming 1.0/3.6 mm worse than triangulation at V3/V4.

Our ResNet result is in the opposite, triangulation-dominant regime: H76 is
almost an identity map over its ray anchor, preserving excellent V3/V4 but
failing to inject a strong pose prior for the two weak V2 pairs.  GBT flattens
all joint/view/time tokens into a global encoder, decodes 3D with learned joint
queries, trains with random K=2 views for 300k iterations and T=9, and supplies
confidence/ray-distance biases in every encoder layer.  Consequently it can
infer depth from whole-body and temporal context instead of depending on the
two-ray solution.  The required repair for our line is a condition-adaptive
switch: retain the accurate anchor on well-conditioned pairs and invoke the
learned pose prior only when the ray normal matrix exposes depth ambiguity.

This table is a diagnostic of the frozen candidate, not a claim that the
hinge caused the multi-view gap. Within the ResNet cache, the no-hinge E2-C2
result is 40.9155/23.3164/20.6082 mm and the hinge result is
40.8620/23.1725/20.4261 mm, so the hinge improves all cardinalities slightly.

## Geometry check

The mean absolute sine of the ray angle (the relevant line-intersection
conditioning term) is:

| pair | HRNet | ResNet |
|---|---:|---:|
| 0,1 | 0.679 | 0.675 |
| 0,2 | 0.631 | 0.630 |
| 0,3 | **0.200** | **0.195** |
| 1,2 | **0.154** | **0.149** |
| 1,3 | 0.671 | 0.667 |
| 2,3 | 0.699 | 0.692 |

Thus `(0,3)` and `(1,2)` are near-parallel line configurations (their
oriented ray angle is around 160 degrees, so the effective unoriented angle
is only about 20 degrees). A 4--5 pixel residual can therefore create a much
larger depth error. The ResNet frontend actually has lower 2D error on every
camera (roughly 3.9--5.1 px versus HRNet 8.7--11.1 px), so this is not a
detector-wide accuracy failure. It is the interaction between residual 2D
noise and ill-conditioned camera pairs, which the current H76 path does not
explicitly model.

## Confidence audit

The confidence value in the actual RUMPL input must be distinguished from the
detector's raw score. In the validation PKLs used by the comparison, ResNet's
`joints_2d_conf` is nearly constant (mean 0.5002, standard deviation 0.0089,
range 0.4855--0.5192), whereas HRNet's value is informative (mean 0.8592,
standard deviation 0.1239, range 0.0223--1.0820). Therefore the statement that
the ResNet input has a higher confidence does not hold for the tensor actually
passed to RUMPL. The ResNet 2D points are more accurate, but its confidence
channel cannot distinguish good from bad joints; the confidence-weighted anchor
is consequently close to uniform. This is a secondary, testable issue and is
being kept separate from the geometry-conditioning experiment.

The targeted follow-up `h76_geom_uncertainty/` enables the existing
zero-initialized ray-normal-matrix uncertainty token while keeping all other
ResNet-H76 settings fixed. Its purpose is to test whether explicitly exposing
this condition information removes the two bad-pair failure.
