# Current-dense Algebraic reevaluation for Stage 2

> Frozen on 2026-08-29. Values are absolute action-equal All-17 MPJPE in mm,
> without alignment. Lower is better.

## Why the Algebraic rows were rerun

The previous Algebraic values came from the sparse
`calib_c2_s020_050_occ{2,3}` frontends generated on 2026-08-23/24. The final
complete model uses the dense frontends regenerated on 2026-08-26. A matched
key audit found that the current and previous 2-D observations are not
identical:

- ResNet-152 Occ-2: 6,061 of 8,084 center records changed in `joints_2d`;
- HRNet-W32 Occ-2: 6,056 of 8,084 center records changed in `joints_2d`.

Therefore, the old Algebraic rows are not a strict matched-input baseline for
the final complete model and must not be used in the final table.

## Matched protocol

- current dense Occ-2/Occ-3 frontend observations;
- exactly the frozen 2,021 scored center groups, or 8,084 camera records;
- all 6/4/1 V2/V3/V4 camera combinations;
- ResNet-152: official ICCV 2019 Algebraic LT with learned confidences;
- HRNet-W32: confidence-weighted Algebraic DLT on current HRNet coordinates;
- no outlier clipping, Procrustes alignment, or failed-sample deletion.

## Final values to use

| Method | 2-D input | Occ-2 V2 | V3 | V4 | Occ-3 V2 | V3 | V4 |
|---|---|---:|---:|---:|---:|---:|---:|
| Algebraic Triangulation | ResNet-152 | **114.514** | **45.326** | **40.978** | **128.619** | **56.126** | **49.576** |
| Ours, complete T=9 | ResNet-152 | **45.278** | **25.652** | **21.349** | **51.111** | **27.862** | **22.653** |
| Algebraic Triangulation | HRNet-W32 | **257.538** | **67.881** | **56.514** | **358.543** | **251.528** | **58.507** |
| Ours, complete T=9 | HRNet-W32 | **53.966** | **32.204** | **28.705** | **58.852** | **33.970** | **29.695** |

The high HRNet Occ-3 V3 value is retained. A deterministic independent rerun
produced a byte-identical JSON (`faeff4d...`); the increase is caused by
unclipped geometric failures in particular action/camera combinations, not by
an interrupted evaluator.

## Result files and checksums

Root:
`/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/current_algebraic_reeval_20260828`

- `occ2_resnet152_official_algebraic.json`:
  `88228510dc01f232ac764262f55b04e68bd2d8be2ba3be3b407a0f6def939056`
- `occ3_resnet152_official_algebraic.json`:
  `5a8ffaf862a3e50969ab223be042cdcd7f4e271a750539d2c7ee1d6502fa91d9`
- `occ2_hrnet_algebraic.json`:
  `41af5a9db1eab98b3fab3c4fa0c9a2723663f569ffc58549a03824d2b3e95fcf`
- `occ3_hrnet_algebraic.json`:
  `faeff4d5848bdba5c3cc99d5ac373df609db8d803a3d1af721c701938c740aab`

The exact center-selection adapter is
`select_current_dense_occ_centers_20260828.py`.
