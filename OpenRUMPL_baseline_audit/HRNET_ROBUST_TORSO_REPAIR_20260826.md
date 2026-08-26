# HRNet robust-torso repair record (2026-08-26)

## Motivation and isolated diagnosis

The established body-canonical frame used the triangulated shoulder pair for
the horizontal axis and the HRNet virtual-neck-to-pelvis vector for the up
axis.  On clean H36M S8, all camera combinations showed that a confidence-
weighted shoulder+hip horizontal axis and shoulder-midpoint-to-hip-midpoint up
axis reduced frame geodesic error against the GT torso frame:

| views | established | robust torso | reduction |
|---|---:|---:|---:|
| V2 | 14.102 deg | 12.586 deg | 1.516 deg |
| V3 | 8.369 deg | 6.346 deg | 2.022 deg |
| V4 | 7.918 deg | 5.582 deg | 2.337 deg |

Audit: `/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_highview_restore_20260826/body_basis_diagnostic_s8.json`.

The implementation is opt-in through
`RUMPL_BODY_CANONICAL_ROBUST_TORSO=1`.  With the option disabled, the new code
is exactly equal to the established path (maximum absolute difference 0).  The
enabled path passed a synthetic SE(3)-equivariance test with maximum absolute
error below `8e-15`.

## S8-only branch selection

All branches started from the same canonical token10 checkpoint, used the
same 8:1:1 view-count schedule, six epochs, LR `1e-6`, pelvis prior, body
regularization `1e-2`, and no synthetic camera replacement.  Only the first
two-epoch token-dropout choice differed.

| candidate | V2 | V3 | V4 | mean V234 |
|---|---:|---:|---:|---:|
| token10 control | 20.735 | 14.566 | 11.867 | 15.723 |
| robust, dropout 0 | **20.194** | **14.453** | **11.741** | **15.463** |
| robust, token10 first 2 epochs | 20.338 | 14.540 | 11.823 | 15.567 |

The predeclared S8 criterion selected `robust_drop0`.

Selection record: `/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_robust_torso_20260826/selection_s8.json`.

## Formal clean S9/S11 generator result

| generator | V2 | V3 | V4 | mean V234 |
|---|---:|---:|---:|---:|
| established token10 | 38.412 | **31.376** | **28.900** | 32.896 |
| selected robust torso | **38.201** | 31.399 | 28.974 | **32.858** |
| robust delta | -0.211 | +0.023 | +0.074 | -0.038 |

The robust generator improves the unified three-cardinality mean and V2, but
does not independently restore V3/V4.  It is therefore a modest structural
repair, not evidence that the complete HRNet gap is solved.

Formal results: `/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_robust_torso_20260826/formal_selected/eval/`.

## Fresh downstream chain

The selected checkpoint is being passed through fresh E2 and the already
selected continuous no-warp H18 path under:

`/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_robust_downstream_20260826/`

No old cache is reused.  Failed time-warp and uncertainty H18 variants are not
repeated.  The comparison target remains the established final HRNet result
`37.392 / 29.501 / 27.713 mm`.
