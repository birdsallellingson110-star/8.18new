# VOC Occ-2/Occ-3: old complete T=9 vs canonical complete T=9

Date: 2026-08-26

This is the matched final-to-final comparison requested after Stage-2.  The old
frozen complete baselines were replayed on the exact same dense VOC images,
frontend coordinate PKLs, 2,021 scored center groups, T=9 centered windows,
camera subsets and action-equal All-17 absolute MPJPE metric as the current
canonical baselines.  No frontend was rerun and no occlusion result was used
for model selection.

Negative delta means that the current canonical model has lower MPJPE.

| Input | Setting | Old complete T=9 V2/V3/V4 | Current complete T=9 V2/V3/V4 | Current - old V2/V3/V4 |
|---|---|---:|---:|---:|
| ResNet-152 | Occ-2 | 48.092 / 27.529 / 22.123 | 45.278 / 25.652 / 21.349 | -2.814 / -1.877 / -0.774 |
| ResNet-152 | Occ-3 | 54.351 / 30.527 / 23.734 | 51.111 / 27.862 / 22.653 | -3.239 / -2.665 / -1.081 |
| HRNet-W32 | Occ-2 | 52.561 / 32.479 / 28.270 | 53.966 / 32.204 / 28.705 | +1.405 / -0.275 / +0.435 |
| HRNet-W32 | Occ-3 | 56.937 / 33.820 / 29.341 | 58.852 / 33.970 / 29.695 | +1.914 / +0.150 / +0.354 |

Mean over V2/V3/V4:

| Input | Setting | Old | Current | Current - old |
|---|---|---:|---:|---:|
| ResNet-152 | Occ-2 | 32.581 | 30.760 | -1.822 |
| ResNet-152 | Occ-3 | 36.204 | 33.875 | -2.329 |
| HRNet-W32 | Occ-2 | 37.770 | 38.292 | +0.522 |
| HRNet-W32 | Occ-3 | 40.033 | 40.839 | +0.806 |

Interpretation:

- The camera-generalization revision clearly improves the complete ResNet-152
  chain in every VOC Occ-2/Occ-3 cell.
- It does not improve the complete HRNet chain uniformly.  HRNet Occ-2 V3 is
  0.275 mm better, but the other five cells regress by 0.150--1.914 mm.
- The paper must not claim detector-independent occlusion improvement from the
  canonical revision alone.  It may claim a strong ResNet improvement and
  report the mixed HRNet result as a frontend-dependent limitation.

Old replay outputs:

- `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/occ2/eval_old_baseline_20260826/hrnet/results/final_h18_t9.json`
- `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/occ2/eval_old_baseline_20260826/resnet152/results/final_h18_t9.json`
- `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/occ3/eval_old_baseline_20260826/hrnet/results/final_h18_t9.json`
- `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/occ3/eval_old_baseline_20260826/resnet152/results/final_h18_t9.json`

Current outputs:

- `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/occ2/eval/hrnet/results/final_h18_t9.json`
- `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/occ2/eval/resnet152/results/final_h18_t9.json`
- `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/occ3/eval/hrnet/results/final_h18_t9.json`
- `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/occ3/eval/resnet152/results/final_h18_t9.json`

Old frozen H18 checkpoint hashes:

- HRNet: `9702248146632557e1b87d0e7ef060a0ff77337c906d3a63988dda261089d3e2`
- ResNet-152: `9f5b0b8aa3a587f223754d5b353d0e03598db2ed0e03f0814d0883a1685beca0`

Current frozen H18 checkpoint hashes:

- HRNet: `d0796f9820cb272590878db245e2c9e817f28df1e992045eaff1a4a39a0e3b1b`
- ResNet-152: `b7aaa26ffbef24c9d965e441aa6ac3eaebef471f777c2235090cef6bbdddfaf9`
