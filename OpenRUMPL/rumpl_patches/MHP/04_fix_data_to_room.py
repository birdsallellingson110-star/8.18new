"""Apply locate_mesh_in_room equivalent to existing stage_V data (per-clip).

Paper's run_mmpose_02_run.py:510 calls locate_mesh_in_room which:
  1. Subtracts root_loc from vertices (xyz if room_min_z != room_max_z, else only xy)
  2. Adds random augmentation_3d in [room_min_x, room_max_x] × [room_min_y, room_max_y] × [room_min_z, room_max_z]

Our 02_clip_run.py SKIPPED this step. This script applies it offline to existing stage_V chunks.

Also fixes joints_2d_amass projection bug: 02_clip_run.py used `cam['T']` (cam center in world)
where paper uses `cam['t']` (extrinsic translation = -R @ T). The 2D detections (joints_2d_mmpose)
from HRNet are unaffected (rendered with correct mesh transform using `t`).

After per-clip shift by delta = -root_loc + augmentation_3d (uniform across all L frames):
  - joints_3d        += delta
  - smplh_transl     += delta
  - triangulated_3d  += delta
  - camera.T         += delta
  - camera.t          = -R @ camera.T (recomputed)
  - joints_2d_amass   = project(joints_3d_new, R, t_new, K)  # CORRECT projection
  - joints_2d_mmpose  unchanged (rendered with original t which is consistent)
  - confs_2d_mmpose   unchanged

Math sanity: world_to_cam(j+δ, R, -R(T+δ)) = R(j+δ) - R(T+δ) = Rj - RT = world_to_cam(j, R, -RT),
so the rendered 2D positions are invariant to this relabel; rendered images stay valid.

Usage:
  python 04_fix_data_to_room.py --exp clip_full --work-dir /home/.../mhp_workspace
"""

import os
import glob
import pickle
import argparse
import hashlib
import numpy as np
from tqdm import tqdm


def world_to_cam(joints_world, R, t):
    """joints_world (J,3) → joints_cam (J,3); R (3,3) world→cam; t (3,1) or (3,) extrinsic."""
    t = np.asarray(t).reshape(3, 1)
    R = np.asarray(R)
    return (R @ joints_world.T + t).T


def cam_to_image(joints_cam, K):
    """joints_cam (J,3) → joints_image (J,2) via K (3,3)."""
    pi = (np.asarray(K) @ joints_cam.T).T  # (J, 3)
    return pi[:, :2] / pi[:, 2:3]


def project(joints_world, R, t, K):
    return cam_to_image(world_to_cam(joints_world, R, t), K)


def fix_clip(chunk, ci, rng, room_bounds):
    """Apply locate_mesh_in_room to one clip (in-place on chunk dict)."""
    L = chunk['joints_3d'].shape[1]
    V = chunk['joints_2d_mmpose'].shape[2]

    # COCO mid-hip of frame 0 as root_loc (matches paper coco branch: joints_3d[0, 11:13].mean(0))
    root_loc = (chunk['joints_3d'][ci, 0, 11] + chunk['joints_3d'][ci, 0, 12]) / 2.0  # (3,)

    rmin_x, rmax_x, rmin_y, rmax_y, rmin_z, rmax_z = room_bounds

    # augmentation_3d: random in room bounds
    aug = np.array([
        rng.uniform(rmin_x, rmax_x),
        rng.uniform(rmin_y, rmax_y),
        rng.uniform(rmin_z, rmax_z) if rmin_z != rmax_z else 0.0,
    ], dtype=np.float64)

    # Paper locate_mesh_in_room:
    #   if room_min_z != 0 and room_max_z != 0:  vertices -= root_loc (xyz)
    #   else:                                     vertices[:, 0:2] -= root_loc[0:2]  (only xy)
    #   vertices += augmentation_3d (xyz, but z=0 if z bounds are 0)
    if rmin_z != 0.0 and rmax_z != 0.0:
        sub = root_loc.copy()
    else:
        sub = np.array([root_loc[0], root_loc[1], 0.0], dtype=np.float64)
    delta = -sub + aug  # (3,)
    delta = delta.astype(np.float32)

    # Shift joints_3d (broadcasts over L, 17, 3)
    chunk['joints_3d'][ci] = (chunk['joints_3d'][ci].astype(np.float64) + delta).astype(np.float32)

    # Shift SMPL+H transl (per-frame; clip-uniform shift)
    chunk['smplh_transl'][ci] = (chunk['smplh_transl'][ci].astype(np.float64) + delta).astype(np.float32)

    # Shift triangulated_3d_mmpose (if present)
    if 'triangulated_3d_mmpose' in chunk:
        chunk['triangulated_3d_mmpose'][ci] = chunk['triangulated_3d_mmpose'][ci] + delta

    # Cameras: shift T (cam center), recompute t = -R @ T
    for v in range(V):
        cam = chunk['camera_parameters_all'][ci][v]
        T_old = np.asarray(cam['T'], dtype=np.float64).reshape(3, 1)
        R = np.asarray(cam['R'], dtype=np.float64)
        T_new = T_old + delta.reshape(3, 1).astype(np.float64)
        t_new = -R @ T_new
        cam['T'] = T_new
        cam['t'] = t_new

    # Recompute joints_2d_amass with CORRECTED projection (was buggy: used T, should use t)
    # New joints_3d in shifted frame, new cameras in shifted frame
    j3d_clip = chunk['joints_3d'][ci]  # (L, 17, 3)
    for tt in range(L):
        for v in range(V):
            cam = chunk['camera_parameters_all'][ci][v]
            chunk['joints_2d_amass'][ci, tt, v] = project(
                j3d_clip[tt].astype(np.float64), cam['R'], cam['t'], cam['K']
            )

    return delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--in-stage-name', default='stage_V')
    ap.add_argument('--out-stage-name', default='stage_V_room')
    ap.add_argument('--operation-on', nargs='+', default=['train', 'validation'])
    # Room bounds matching the dataset cfg (cmu_panoptic/rumpl_amass/clip_full.yaml)
    ap.add_argument('--room-min-x', default=-2.0, type=float)
    ap.add_argument('--room-max-x',  default=0.3, type=float)
    ap.add_argument('--room-min-y', default=-0.8, type=float)
    ap.add_argument('--room-max-y',  default=0.8, type=float)
    ap.add_argument('--room-min-z',  default=0.0, type=float)
    ap.add_argument('--room-max-z',  default=0.0, type=float)
    ap.add_argument('--seed', default=2026, type=int)
    args = ap.parse_args()

    work_dir = os.path.join(args.work_dir, args.exp)
    bounds = (args.room_min_x, args.room_max_x,
              args.room_min_y, args.room_max_y,
              args.room_min_z, args.room_max_z)
    print(f'[fix] room bounds: x=[{bounds[0]},{bounds[1]}] y=[{bounds[2]},{bounds[3]}] z=[{bounds[4]},{bounds[5]}]')

    for subset in args.operation_on:
        in_dir = os.path.join(work_dir, args.in_stage_name, subset)
        out_dir = os.path.join(work_dir, args.out_stage_name, subset)
        os.makedirs(out_dir, exist_ok=True)
        chunks = sorted(glob.glob(os.path.join(in_dir, '*.pkl')))
        if not chunks:
            print(f'[{subset}] no chunks at {in_dir}')
            continue
        print(f'[{subset}] processing {len(chunks)} chunks → {out_dir}')

        for chunk_path in chunks:
            # Deterministic seed per chunk (hash of basename + global seed)
            name = os.path.basename(chunk_path)
            seed = int(hashlib.md5((name + str(args.seed)).encode()).hexdigest()[:8], 16) % (2**32)
            rng = np.random.RandomState(seed)

            with open(chunk_path, 'rb') as f:
                chunk = pickle.load(f)
            Nc = chunk['joints_3d'].shape[0]

            deltas = []
            for ci in tqdm(range(Nc), desc=f'  {name}', leave=False):
                d = fix_clip(chunk, ci, rng, bounds)
                deltas.append(d)
            deltas = np.array(deltas)

            out_path = os.path.join(out_dir, name)
            with open(out_path, 'wb') as f:
                pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Sanity: report shift stats + new joints_3d range
            j3d = chunk['joints_3d']
            print(f'    {name}: N={Nc} clips, delta mean={deltas.mean(axis=0)}, '
                  f'joints_3d xyz range: '
                  f'x=[{j3d[..., 0].min():.2f}, {j3d[..., 0].max():.2f}], '
                  f'y=[{j3d[..., 1].min():.2f}, {j3d[..., 1].max():.2f}], '
                  f'z=[{j3d[..., 2].min():.2f}, {j3d[..., 2].max():.2f}]')

    print('\nDONE.')


if __name__ == '__main__':
    main()
