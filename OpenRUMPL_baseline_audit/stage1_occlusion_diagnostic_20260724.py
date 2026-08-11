#!/usr/bin/env python3
"""Stage 1 diagnostic: where does test-time occlusion break RUMPL (V2)?
Uses only pred/gt from the eval pkls. Reports:
  (A) per-joint MPJPE at occ 0.0/0.3/0.6 and the degradation delta,
  (B) bone-length error (structure violation) vs occlusion level.
Decision rule:
  - if degradation concentrates on extremities -> occluded-joint recovery problem
  - if bone-length error grows sharply -> structure-aware module has headroom
"""
import pickle
import numpy as np

COCO17 = ["nose","Leye","Reye","Lear","Rear","Lsho","Rsho","Lelb","Relb",
          "Lwri","Rwri","Lhip","Rhip","Lkne","Rkne","Lank","Rank"]
KP_STAR = (5,6,7,8,9,10,13,14,15,16)
# COCO skeleton bones (parent-child pairs)
BONES = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),
         (5,6),(11,12),(5,11),(6,12)]
BASE = ("/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval/"
        "R5_v2_occ{occ}/multiview_amass_rumpl/multiview_rumpl_999/"
        "crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_"
        "RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5/"
        "preds_gt_multiview_cmu_panoptic_rumpl_mmpose_occ_R5_v2_occ{occ}_"
        "best_R5_v2_occ{occ}_dict.pkl")

def load(occ):
    with open(BASE.format(occ=occ), "rb") as f:
        d = pickle.load(f)
    return np.asarray(d["pred"]), np.asarray(d["gt"])

def bone_len(x):
    return np.stack([np.linalg.norm(x[:,a]-x[:,b],axis=-1) for a,b in BONES],axis=1)

levels = ["0.0","0.3","0.6"]
perjoint = {}
bone_err = {}
for occ in levels:
    pred, gt = load(occ)
    err = np.linalg.norm(pred-gt,axis=-1)*1000.0  # (N,17) mm
    perjoint[occ] = err.mean(axis=0)
    be = np.abs(bone_len(pred)-bone_len(gt))*1000.0  # (N,12) mm
    bone_err[occ] = be.mean(axis=0)

print("="*70)
print("(A) PER-JOINT MPJPE (mm) by occlusion level  [V2]")
print(f"{'joint':<6}{'occ0.0':>9}{'occ0.3':>9}{'occ0.6':>9}{'Δ(0.6-0.0)':>12}")
order = np.argsort(-(perjoint['0.6']-perjoint['0.0']))
for j in order:
    star = "*" if j in KP_STAR else " "
    print(f"{COCO17[j]:<5}{star}{perjoint['0.0'][j]:>9.2f}{perjoint['0.3'][j]:>9.2f}"
          f"{perjoint['0.6'][j]:>9.2f}{perjoint['0.6'][j]-perjoint['0.0'][j]:>12.2f}")
ext = [9,10,15,16]  # wrists, ankles
core = [5,6,11,12]  # shoulders, hips
print(f"\n  extremities(wri/ank) Δ mean = {(perjoint['0.6'][ext]-perjoint['0.0'][ext]).mean():.2f} mm")
print(f"  core(sho/hip)        Δ mean = {(perjoint['0.6'][core]-perjoint['0.0'][core]).mean():.2f} mm")

print("\n"+"="*70)
print("(B) BONE-LENGTH ERROR (mm) by occlusion level  [structure violation]")
print(f"{'bone':<10}{'occ0.0':>9}{'occ0.3':>9}{'occ0.6':>9}{'Δ':>9}")
for i,(a,b) in enumerate(BONES):
    name=f"{COCO17[a]}-{COCO17[b]}"
    print(f"{name:<10}{bone_err['0.0'][i]:>9.2f}{bone_err['0.3'][i]:>9.2f}"
          f"{bone_err['0.6'][i]:>9.2f}{bone_err['0.6'][i]-bone_err['0.0'][i]:>9.2f}")
print(f"\n  mean bone-length error:  occ0.0={bone_err['0.0'].mean():.2f}  "
      f"occ0.3={bone_err['0.3'].mean():.2f}  occ0.6={bone_err['0.6'].mean():.2f} mm")
print(f"  bone-error growth 0->0.6: {bone_err['0.6'].mean()-bone_err['0.0'].mean():.2f} mm "
      f"({100*(bone_err['0.6'].mean()/bone_err['0.0'].mean()-1):.0f}%)")
