"""Pre-process CMU Panoptic pkl: swap axes (y-up → z-up) AND scale cameras cm→m.

Why:
  - CMU Panoptic world frame is y-up (joints_3d y ∈ [-100, 40] cm; y is vertical)
  - Our model was trained on AMASS world frame which is z-up (joints_3d z ∈ [0, 240] cm)
  - The loader's default branch (USE_CMU_CAMERAS_ON_CMU=false) uses db_rec['camera'] raw
    without axis swap and without /100 scaling for cameras
  - So we make the pkl self-consistent in AMASS convention, in cm for joints_3d
    (loader divides /100) and in METERS for cameras (loader uses as-is)

What we apply for each sample:
  joints_3d   (cm world): P @ j3d  (keep cm; loader will /100)
  camera R:   R_new = R_old @ P.T
  camera T:   T_new = (P @ T_old) / 100   (cam center, cm → m)
  camera t:   t_new = -R_new @ T_new       (extrinsic, m)
  joints_2d:  UNCHANGED  (projection invariant to consistent world relabeling+rescaling)

Swap matrix P (right-handed, det=+1):
  new_x =  old_x
  new_y =  old_z
  new_z = -old_y
(CMU panoptic is y-DOWN — positive y = floor direction. Verified by JSON values:
nose y=-158, ankle y=-12, so larger |y| corresponds to "up". AMASS is z-up
with positive z = up. So new_z = -old_y.)
"""

import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm


P = np.array([[1.0,  0.0, 0.0],
              [0.0,  0.0, 1.0],
              [0.0, -1.0, 0.0]], dtype=np.float64)
Pt = P.T


def transform_sample(s):
    s_new = dict(s)
    j3d = np.asarray(s['joints_3d'], dtype=np.float64)
    s_new['joints_3d'] = (P @ j3d.T).T.astype(np.float32)

    cam = dict(s['camera'])
    R_old = np.asarray(cam['R'], dtype=np.float64)
    T_old = np.asarray(cam['T'], dtype=np.float64).reshape(3, 1)
    # Keep cm scale here; the loader will /100 (joints + camera) when output_in_meter is True.
    # Earlier version divided by 100 here, causing a /10000 double-division bug.
    R_new = R_old @ Pt
    T_new = P @ T_old
    t_new = -R_new @ T_new
    cam['R'] = R_new
    cam['T'] = T_new
    cam['t'] = t_new
    # fx, fy, cx, cy, K unchanged
    s_new['camera'] = cam
    return s_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-pkl', required=True)
    ap.add_argument('--out-pkl', required=True)
    args = ap.parse_args()

    print(f'loading {args.in_pkl}')
    with open(args.in_pkl, 'rb') as f:
        d = pickle.load(f)

    out = []
    for s in tqdm(d, desc='swap+scale'):
        out.append(transform_sample(s))

    # Sanity: project new_j3d (in original cm scale → /100 to m) with new cameras (already m) → 2D
    s0 = out[0]
    R = np.asarray(s0['camera']['R'])
    t = np.asarray(s0['camera']['t']).reshape(3, 1)
    T_center = np.asarray(s0['camera']['T']).reshape(3, 1)
    K = np.asarray(s0['camera']['K'])

    print(f'\n[after transform] sample 0:')
    print(f'  joints_3d range (cm): x=[{out[0]["joints_3d"][:,0].min():.1f},{out[0]["joints_3d"][:,0].max():.1f}] '
          f'y=[{out[0]["joints_3d"][:,1].min():.1f},{out[0]["joints_3d"][:,1].max():.1f}] '
          f'z=[{out[0]["joints_3d"][:,2].min():.1f},{out[0]["joints_3d"][:,2].max():.1f}]')
    print(f'  z range should be ≈ vertical (full body height ~150 cm) — was originally CMU\'s y')
    print(f'  camera T (m): {T_center.flatten()}')
    print(f'  verify -R @ T = t: -R@T = {(-R @ T_center).flatten()}, t = {t.flatten()}, match={np.allclose(-R @ T_center, t)}')

    # Reproject: simulate loader's path: joints in m = j3d/100; cameras already in m
    j3d_m = np.asarray(s0['joints_3d']).astype(np.float64) / 100.0
    j_cam = (R @ j3d_m.T + t).T
    j_img = (K @ j_cam.T).T
    j_proj = j_img[:, :2] / j_img[:, 2:3]
    j_stored = np.asarray(s0['joints_2d'])
    err = np.linalg.norm(j_proj - j_stored, axis=1)
    print(f'\n[sanity sample 0] reprojection error after transform: mean={err.mean():.3f} max={err.max():.3f} px')
    print(f'  (Should be ~0 if all transforms are consistent)')

    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nsaved {args.out_pkl}: {len(out)} samples')


if __name__ == '__main__':
    main()
