"""Clip-aware MHP renderer + HRNet for ST-VFT method.

For each clip (L_clip consecutive AMASS frames):
  - Pick ONE camera setup; place V cameras (random in room, intrinsics from CMU calibs)
  - SMPL+H forward all L frames → vertices + pelvis displacement check (skip if > 1m)
  - For each frame × each view: render mesh image → HRNet → 2D joints; J_regressor → 3D joints
  - Save:
      joints_3d              (L, 17, 3)            from J_regressor_coco @ vertices
      joints_2d_amass        (L, V, 17, 2)         project joints_3d → 2D
      joints_2d_mmpose       (L, V, 17, 2)         HRNet output (NaN where detection fails)
      confs_2d_mmpose        (L, V, 17, 1)
      triangulated_3d_mmpose (L, 17, 3)            optional (each frame independent)
      camera_parameters      list of V dicts       SAME for whole clip (fixed cameras)
      smplh_betas            (16,)                 per-clip
      smplh_global_orient    (L, 3)                per-frame
      smplh_body_pose        (L, 63)               per-frame
      smplh_transl           (L, 3)                per-frame
      smplh_gender           int                   0=m / 1=f / 2=n
      frame_rate             float
      source_npz, start_frame                       traceability

Parallel: --split-index 0..N-1 + --total-splits N divides stage_IV chunks among processes.
Resume: each process saves chunks every --chunk-size clips; on restart, skip already-saved.
"""

import os
import sys
import glob
import json
import argparse
import pickle
import time
import numpy as np
import torch
import trimesh
import imageio
from tqdm import tqdm

from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from mmpose.apis import MMPoseInferencer

from utils import (
    load_all_cameras_cmu, random_camera_in_room,
    cam_to_image, world_to_cam, run_mmpose, get_rotation_matrix
)
from multiviews.triangulate import triangulate_poses


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# COCO joint order from J_regressor_coco — no L/R swap needed (new regressor)
COCO_JOINTS = ['nose','leye','reye','lear','rear','lsho','rsho',
               'lelb','relb','lwri','rwri','lhip','rhip',
               'lkne','rkne','lank','rank']


def smplh_forward_batch(bm_dict, gender, pose, trans, betas):
    """SMPL+H forward for a clip. Returns vertices (L, 6890, 3).

    pose: (L, 156) ndarray
    trans: (L, 3) ndarray
    betas: (16,) ndarray
    gender: int (0=m, 1=f, 2=n)
    """
    bm = bm_dict[gender]
    L = pose.shape[0]
    p = torch.from_numpy(pose).float().to(device)
    t = torch.from_numpy(trans).float().to(device)
    b = torch.from_numpy(betas).float().to(device).unsqueeze(0).expand(L, -1)
    out = bm.forward(
        root_orient=p[:, :3],
        pose_body=p[:, 3:66],
        pose_hand=p[:, 66:156],
        trans=t,
        betas=b,
    )
    return c2c(out.v)  # (L, 6890, 3)


def main():
    ap = argparse.ArgumentParser()
    # I/O
    ap.add_argument('--exp', required=True, type=str)
    ap.add_argument('--work-dir', required=True, type=str)
    ap.add_argument('--support-dir', required=True, type=str)
    ap.add_argument('--image-save-dir', default='/tmp/mhp_images', type=str)
    ap.add_argument('--operation-on', default='train', type=str)
    # Parallel
    ap.add_argument('--split-index', default=0, type=int)
    ap.add_argument('--total-splits', default=1, type=int)
    ap.add_argument('--chunk-size', default=100, type=int, help='clips per output pkl (resume granularity)')
    ap.add_argument('--max-clips', default=-1, type=int, help='per-process clip processing cap; -1 = no cap')
    # Clip / pose
    ap.add_argument('--L-clip', default=27, type=int)
    ap.add_argument('--displacement-thresh', default=1.0, type=float, help='m, max pelvis travel within clip')
    # Cameras
    ap.add_argument('--calib-root-cmu', required=True, type=str)
    ap.add_argument('--calibs-cmu', nargs='+', required=True, type=str)
    ap.add_argument('--n-cameras-per-clip', default=20, type=int)
    ap.add_argument('--camera-location-limit', nargs=6, type=float,
                    default=[-2.7, 2.7, -2.7, 2.7, 0.7, 3.4])
    ap.add_argument('--camera-dist-from-person', default=2.0, type=float)
    ap.add_argument('--room-size', nargs=6, type=float, default=[-0.5, -0.1, -0.2, 0.2, 0, 0])
    # Rendering
    ap.add_argument('--image-width', default=1920, type=int)
    ap.add_argument('--image-height', default=1080, type=int)
    ap.add_argument('--apply-rotation', action='store_true')
    # 2D estimation
    ap.add_argument('--pose2d-model', default='td-hm_hrnet-w32_8xb64-210e_coco-384x288', type=str)
    # Regressor
    ap.add_argument('--amass-data-dir', required=True, type=str)
    ap.add_argument('--regressor', default='coco', choices=['coco', 'h36m', 'both'])
    # Triangulation
    ap.add_argument('--triangulate', action='store_true')
    ap.add_argument('--triangulate-th', default=0.95, type=float)
    # Seed
    ap.add_argument('--seed', default=0, type=int)
    args = ap.parse_args()

    np.random.seed(args.seed + args.split_index)
    torch.manual_seed(args.seed + args.split_index)

    # --- Setup output dirs ---
    work_dir = os.path.join(args.work_dir, args.exp)
    out_dir = os.path.join(work_dir, 'stage_V', args.operation_on)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.image_save_dir, exist_ok=True)

    # --- Load body models (all 3 genders) ---
    bm_dict = {}
    for gid, gname in [(0, 'male'), (1, 'female'), (2, 'neutral')]:
        bm_path = os.path.join(args.support_dir, f'body_models/smplh/{gname}/model.npz')
        bm_dict[gid] = BodyModel(bm_path=bm_path, model_type='smplh', num_betas=16).to(device)

    # --- Load J_regressor (coco for now; h36m field reserved for future) ---
    J_coco = np.load(os.path.join(args.amass_data_dir, 'J_regressor_coco.npy')).astype(np.float64)

    # --- Load CMU camera intrinsics pool ---
    cameras_pool = load_all_cameras_cmu(args.calib_root_cmu, args.calibs_cmu)
    # cameras_pool[node] = list of cam dicts (one per calib)

    # Flatten into a pool of (K, distCoef) for random sampling
    intrinsics_pool = []
    for node, setups in cameras_pool.items():
        for s in setups:
            intrinsics_pool.append(s)
    print(f'Intrinsics pool size: {len(intrinsics_pool)}')

    # --- MMPose ---
    mmpose_inferencer = MMPoseInferencer(args.pose2d_model, device='cuda:0')

    # --- Load stage_IV clip chunks for this split ---
    stage_iv_dir = os.path.join(work_dir, 'stage_IV', args.operation_on)
    all_pt = sorted(glob.glob(os.path.join(stage_iv_dir, 'amass_clips_*.pt')))
    print(f'Found {len(all_pt)} stage_IV chunks in {stage_iv_dir}')

    # split-index / total-splits decides which chunks this process handles
    my_pt = [p for i, p in enumerate(all_pt) if i % args.total_splits == args.split_index]
    print(f'Split {args.split_index}/{args.total_splits}: {len(my_pt)} chunks')

    # Load all my clips
    my_clips = []
    for p in my_pt:
        my_clips.extend(torch.load(p, weights_only=False))
    print(f'Total clips to process: {len(my_clips)}')

    # --- Resume: find latest saved chunk ---
    pattern = f'split_{args.split_index}_chunk_*.pkl'
    saved = sorted(glob.glob(os.path.join(out_dir, pattern)))
    n_done = 0
    for p in saved:
        try:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            n_done += d['joints_3d'].shape[0]
        except Exception:
            pass
    print(f'Resume: skipping first {n_done} clips ({len(saved)} chunks already saved)')

    my_clips = my_clips[n_done:]
    if args.max_clips > 0:
        my_clips = my_clips[:args.max_clips]
        print(f'Capped to first {len(my_clips)} clips (--max-clips)')

    # --- Accumulators per chunk ---
    def empty_acc():
        return {
            'joints_3d': [], 'joints_2d_mmpose': [], 'confs_2d_mmpose': [],
            'joints_2d_amass': [], 'triangulated_3d_mmpose': [],
            'camera_parameters_all': [],
            'smplh_betas': [], 'smplh_global_orient': [], 'smplh_body_pose': [],
            'smplh_transl': [], 'smplh_gender': [],
            'frame_rate': [], 'source_npz': [], 'start_frame': [],
            # per-clip Z rotation angle applied before rendering (radians).
            # Phase 1: to reproduce vertices, apply Rz(z_rotation) to smplh_forward output.
            'z_rotation': [],
        }

    acc = empty_acc()
    n_skipped = 0
    n_processed = 0
    chunk_idx = len(saved)
    imw, imh = args.image_width, args.image_height

    bar = tqdm(my_clips, desc=f'split {args.split_index}')
    for clip in bar:
        # === SMPL+H forward (all L frames) ===
        try:
            vertices_all = smplh_forward_batch(
                bm_dict, clip['gender'], clip['pose'], clip['trans'], clip['betas'])
        except Exception as e:
            n_skipped += 1
            continue

        # === Random Z-axis rotation augmentation (paper §5.x) ===
        # Apply same random angle to all L frames of the clip — preserves intra-clip motion.
        # This is what paper's `locate_mesh_in_room(rotate=True)` does; we apply it to the
        # full vertex sequence before camera placement so each clip gets a different orientation.
        if args.apply_rotation:
            theta = float(np.random.uniform(0, 2 * np.pi))
            c, s = np.cos(theta), np.sin(theta)
            Rz = np.array([[c, -s, 0.0],
                           [s,  c, 0.0],
                           [0.0, 0.0, 1.0]], dtype=np.float32)
            # vertices_all: (L, 6890, 3); rotate each vertex by Rz around world Z (which is up)
            vertices_all = vertices_all @ Rz.T
        else:
            theta = 0.0

        # Pelvis displacement check (joint 0 of J_regressor_coco is not pelvis = it's nose;
        # use SMPL+H internal J_regressor: the body model's first joint is pelvis (root))
        # bm has 'J_regressor' as buffer; easier to recompute from vertices using SMPL+H regressor
        # For displacement check, we can use any well-defined point — use mesh centroid which is stable
        # Actually cleanest: take mean of hip vertices (consistent across genders)
        # COCO J_regressor: joint 11=l_hip, joint 12=r_hip
        pelvis_first = (J_coco[11] + J_coco[12]) / 2 @ vertices_all[0]
        pelvis_last  = (J_coco[11] + J_coco[12]) / 2 @ vertices_all[-1]
        displacement = float(np.linalg.norm(pelvis_last - pelvis_first))
        if displacement > args.displacement_thresh:
            n_skipped += 1
            continue

        # === Place cameras (fixed for whole clip) ===
        L = vertices_all.shape[0]
        # Use mid-clip mean vertex as "person_location" to aim cameras at
        person_loc_mid = vertices_all[L // 2].mean(axis=0)
        cameras_clip = []
        for _ in range(args.n_cameras_per_clip):
            intrinsics = intrinsics_pool[np.random.randint(len(intrinsics_pool))]
            # random_camera_in_room returns (R: world→cam, T: cam center, t: -R @ T)
            R, T, t = random_camera_in_room(
                args.camera_location_limit, args.room_size,
                False,
                camera_dist_from_person=args.camera_dist_from_person,
                person_location=person_loc_mid,
            )
            distCoef = [-0.287016, 0.182978, 1.91352e-06, 0.000618877, -0.0471994]
            k = np.array([distCoef[0], distCoef[1], distCoef[4]])
            p = np.array([distCoef[2], distCoef[3]])
            cam_dict = {
                'K': intrinsics['K'].copy(),
                'fx': float(intrinsics['fx']), 'fy': float(intrinsics['fy']),
                'cx': float(intrinsics['cx']), 'cy': float(intrinsics['cy']),
                'k': k, 'p': p,
                'R': R.astype(np.float64),
                'T': T.astype(np.float64),
                't': t.astype(np.float64),
            }
            cameras_clip.append(cam_dict)

        # Build MeshViewers (one per camera, intrinsics fixed for whole clip)
        # Original RUMPL approach: keep pyrender camera at origin; transform mesh per render
        mv_list = []
        for cam in cameras_clip:
            mv = MeshViewer(width=imw, height=imh, use_offscreen=True,
                            fx=cam['fx'], fy=cam['fy'], cx=cam['cx'], cy=cam['cy'])
            mv.set_cam_trans(trans=[0, 0, 0])
            mv_list.append(mv)

        # === Per-frame rendering + HRNet ===
        joints_3d_clip = np.zeros((L, 17, 3), dtype=np.float32)
        joints_2d_amass_clip = np.zeros((L, args.n_cameras_per_clip, 17, 2), dtype=np.float64)
        joints_2d_mmpose_clip = np.full((L, args.n_cameras_per_clip, 17, 2), np.nan, dtype=np.float64)
        confs_clip = np.zeros((L, args.n_cameras_per_clip, 17, 1), dtype=np.float64)
        tri_clip = np.zeros((L, 17, 3), dtype=np.float64) if args.triangulate else None

        try:
            faces = c2c(bm_dict[clip['gender']].f)
            for tt in range(L):
                vertices = vertices_all[tt]
                joints_3d_clip[tt] = (J_coco @ vertices).astype(np.float32)

                for v in range(args.n_cameras_per_clip):
                    cam = cameras_clip[v]
                    mv = mv_list[v]
                    R_v = cam['R']
                    t_v = cam['t']
                    # === Same vertex transform as original RUMPL 02 ===
                    # Move mesh to pyrender camera frame (camera at origin looking -Z)
                    Rt = np.eye(4)
                    Rt[:3, :3] = -R_v
                    Rt[:3, 3] = -t_v.T.flatten()
                    Rt[0, :] = -Rt[0, :]  # flip x axis
                    vertices_t = vertices @ Rt[:3, :3].T + Rt[:3, 3]
                    mesh = trimesh.Trimesh(
                        vertices=vertices_t, faces=faces,
                        vertex_colors=np.tile(colors['grey'], (vertices_t.shape[0], 1)),
                        process=False)
                    mv.set_static_meshes([mesh])
                    img = mv.render(render_wireframe=False)
                    # Project amass joints to 2D (using world-frame joints + camera params)
                    joints_world = joints_3d_clip[tt]
                    joints_cam = world_to_cam(joints_world[None, :], R_v, cam['T'])
                    pts_img = cam_to_image(joints_cam, cam['K'])
                    joints_2d_amass_clip[tt, v] = pts_img[0]
                    # HRNet
                    try:
                        pts_mm, sc_mm = run_mmpose(img, mmpose_inferencer,
                                                    convert_to_h36m=False, return_coco=False)
                        joints_2d_mmpose_clip[tt, v] = pts_mm
                        confs_clip[tt, v] = sc_mm.reshape(-1, 1)
                    except Exception:
                        pass

                # Triangulation (per frame, optional)
                if args.triangulate:
                    confs_t = confs_clip[tt].squeeze(-1)
                    pts_t = joints_2d_mmpose_clip[tt]
                    if not np.isnan(pts_t).all():
                        try:
                            tri = triangulate_poses(cameras_clip, pts_t, confs_t,
                                                     conf_threshold=args.triangulate_th)
                            tri_clip[tt] = tri[0] if tri.ndim == 3 else tri
                        except Exception:
                            tri_clip[tt] = np.nan
        except Exception as e:
            print(f'  per-frame loop crash: {e}, skipping clip')
            for mv in mv_list:
                try: mv.viewer.delete()
                except Exception: pass
            n_skipped += 1
            continue

        # Cleanup mesh viewers (release GPU mem)
        for mv in mv_list:
            try: mv.viewer.delete()
            except Exception: pass

        # === Accumulate ===
        acc['joints_3d'].append(joints_3d_clip)
        acc['joints_2d_mmpose'].append(joints_2d_mmpose_clip)
        acc['confs_2d_mmpose'].append(confs_clip)
        acc['joints_2d_amass'].append(joints_2d_amass_clip)
        if args.triangulate:
            acc['triangulated_3d_mmpose'].append(tri_clip)
        acc['camera_parameters_all'].append(cameras_clip)
        acc['smplh_betas'].append(clip['betas'].astype(np.float32))
        acc['smplh_global_orient'].append(clip['pose'][:, :3].astype(np.float32))
        acc['smplh_body_pose'].append(clip['pose'][:, 3:66].astype(np.float32))
        acc['smplh_transl'].append(clip['trans'].astype(np.float32))
        acc['z_rotation'].append(np.float32(theta))
        acc['smplh_gender'].append(int(clip['gender']))
        acc['frame_rate'].append(float(clip.get('frame_rate', 60.0)))
        acc['source_npz'].append(str(clip.get('source_npz', '')))
        acc['start_frame'].append(int(clip.get('start_frame', 0)))

        n_processed += 1
        bar.set_postfix(processed=n_processed, skipped=n_skipped, chunk=chunk_idx)

        # Save chunk
        if n_processed % args.chunk_size == 0:
            save_chunk(acc, out_dir, args.split_index, chunk_idx)
            chunk_idx += 1
            acc = empty_acc()

    # Save final partial chunk
    if any(len(v) > 0 for v in acc.values()):
        save_chunk(acc, out_dir, args.split_index, chunk_idx)

    print(f'\nDONE. processed={n_processed} skipped={n_skipped} (skip rate {100*n_skipped/(n_processed+n_skipped+1e-9):.1f}%)')


def save_chunk(acc, out_dir, split_idx, chunk_idx):
    """Save accumulator as a pkl chunk. Convert lists → np.array where applicable."""
    if len(acc['joints_3d']) == 0:
        return
    out = {
        'joints_3d':             np.stack(acc['joints_3d']),
        'joints_2d_mmpose':      np.stack(acc['joints_2d_mmpose']),
        'confs_2d_mmpose':       np.stack(acc['confs_2d_mmpose']),
        'joints_2d_amass':       np.stack(acc['joints_2d_amass']),
        'camera_parameters_all': acc['camera_parameters_all'],
        'smplh_betas':           np.stack(acc['smplh_betas']),
        'smplh_global_orient':   np.stack(acc['smplh_global_orient']),
        'smplh_body_pose':       np.stack(acc['smplh_body_pose']),
        'smplh_transl':          np.stack(acc['smplh_transl']),
        'smplh_gender':          np.array(acc['smplh_gender'], dtype=np.int8),
        'frame_rate':            np.array(acc['frame_rate'], dtype=np.float32),
        'source_npz':            acc['source_npz'],
        'start_frame':           np.array(acc['start_frame'], dtype=np.int32),
    }
    if len(acc['triangulated_3d_mmpose']) > 0:
        out['triangulated_3d_mmpose'] = np.stack(acc['triangulated_3d_mmpose'])

    path = os.path.join(out_dir, f'split_{split_idx}_chunk_{chunk_idx:04d}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  saved: {path}  ({out["joints_3d"].shape[0]} clips)')


if __name__ == '__main__':
    main()
