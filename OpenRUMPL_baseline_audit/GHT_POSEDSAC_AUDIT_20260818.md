# GHT PoseDSAC candidate audit — 2026-08-18

## Purpose

The public Generalizable Human Pose Triangulation (CVPR 2022) repository was
previously represented only by its canonical whole-pose `ScoreNN` preprocessing
on the existing RUMPL/H76 candidate pool.  That is not the complete public
method: `src/dsac.py::PoseDSAC.__sample_hyp` samples a camera subset separately
for every joint of every pose hypothesis.  This audit isolates that missing
candidate-generation part before any long scorer training.

The script is
`audit_ght_pose_hypotheses_20260818.py`.  It uses the GBT-aligned HRNet
coordinate-only validation pkl (`S9/S11`, 2,021 synchronized frames), known
H36M `K,R,t`, and absolute world-coordinate All-17 MPJPE.  It does not read
3D labels when generating hypotheses; labels are used only for the reported
oracle.  For each V2/V3/V4 camera task it samples 200 hypotheses, with each
joint uniformly choosing one subset from all camera subsets of sizes 2..K,
matching the official PoseDSAC code.  The frozen H76 11-candidate pool and its
22-candidate confidence extension are evaluated independently and as an oracle
union.

## Full validation result (action-equal absolute MPJPE, mm)

| candidate pool | V2 | V3 | V4 |
|---|---:|---:|---:|
| official GHT PoseDSAC random hypotheses (unweighted) | 122.353 | 53.044 | 49.377 |
| official GHT all-view DLT | 122.353 | 64.466 | 78.554 |
| existing H76 11 candidates | 38.686 | 30.943 | 28.629 |
| existing H76 + confidence 22 candidates | 38.682 | 30.941 | 28.627 |
| H76-22 ∪ GHT oracle (theoretical upper bound) | 38.682 | 30.870 | 28.590 |

The V2 number is dominated by the two known near-degenerate camera pairs; GHT
does not repair them.  The random per-joint candidates are also substantially
worse than the learned RUMPL candidates for V3/V4.  Their union can improve the
candidate oracle by only 0.071 mm (V3) and 0.052 mm (V4), so a GHT scorer cannot
produce a meaningful new main-table gain from this pool.  We therefore do not
launch a long, label-selected GHT scorer run.

## Confidence diagnostic

An additional full run applied the official linear DLT row weighting by the
HRNet confidence (not claimed as official GHT):

| candidate pool | V2 | V3 | V4 |
|---|---:|---:|---:|
| confidence-weighted random GHT hypotheses | 122.540 | 51.961 | 47.853 |
| H76-22 ∪ confidence-weighted GHT | 38.682 | 30.860 | 28.575 |

Confidence weighting changes the union upper bound by at most 0.015 mm.  This
confirms that the issue is not simply an omitted confidence multiplier; the
learned RUMPL correction is the useful part for this detector/noise regime.

## Decision

1. Do not replace RUMPL's learned 3D candidate generator with raw GHT DLT.
2. Do not spend a full training run on the official GHT scorer: its candidate
   upper bound is already too weak.
3. Keep the paper-backed GHT result as a negative ablation and preserve the
   reproducible script/output paths:

   - `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/ght_pose_hypothesis_audit_v1/result.json`
   - `/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/ght_pose_hypothesis_audit_conf_v1/result.json`

The next controlled experiment returns to the best learned RUMPL candidate
checkpoint (H1 high-LR, V2=36.885 mm) and changes only the mixed-cardinality
training distribution to balanced V2/V3/V4 sampling.  This tests whether the
remaining V4 gap is optimization/cardinality coverage rather than another
candidate family.  The HRNet input, H76 anchor/centered-Plücker representation,
camera protocol, and evaluation remain unchanged.
