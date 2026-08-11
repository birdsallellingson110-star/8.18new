"""Flatten clip-mode stage_V_room → flat RUMPL pkls, with clip-level 90/10 train/val split.

Each chunk pkl has shape (N_clips, L, ...). We flatten to (N_clips*L, ...) and then split
clips (not frames) into train/val to avoid intra-clip leakage.

Output:
  <out_dir>/amass_mmpose_joints_train.pkl       (90% of clips, flattened)
  <out_dir>/amass_mmpose_joints_validation.pkl  (10% of clips, flattened)
"""

import os
import glob
import pickle
import argparse
import numpy as np
from tqdm import tqdm


def flatten_chunk(chunk):
    """Flatten one stage_V_room chunk dict (N_clips, L, ...) → per-frame fields (N_clips*L, ...)."""
    Nc = chunk['joints_3d'].shape[0]
    L = chunk['joints_3d'].shape[1]
    Nf = Nc * L

    out = {}
    out['joints_3d']        = chunk['joints_3d'].reshape(Nf, 17, 3).astype(np.float32)
    out['joints_2d_mmpose'] = chunk['joints_2d_mmpose'].reshape(Nf, *chunk['joints_2d_mmpose'].shape[2:])
    out['confs_2d_mmpose']  = chunk['confs_2d_mmpose'].reshape(Nf, *chunk['confs_2d_mmpose'].shape[2:])
    out['joints_2d_amass']  = chunk['joints_2d_amass'].reshape(Nf, *chunk['joints_2d_amass'].shape[2:])
    if 'triangulated_3d_mmpose' in chunk:
        out['triangulated_3d_mmpose'] = chunk['triangulated_3d_mmpose'].reshape(Nf, 17, 3)
    # Cameras: shared across L frames per clip; expand per-frame for the loader
    out['camera_parameters_all'] = []
    for ci in range(Nc):
        cams = chunk['camera_parameters_all'][ci]
        for _ in range(L):
            out['camera_parameters_all'].append(cams)
    # views_used per frame: [0, 1, ..., V-1]
    V = chunk['joints_2d_mmpose'].shape[2]
    out['camera_setup_used'] = np.zeros(Nf, dtype=np.int64)
    out['views_used'] = np.tile(np.arange(V, dtype=np.int64)[None, :], (Nf, 1))
    # placeholders (loader will reach for these in some branches)
    out['body_depth_all_gt']     = np.zeros((Nf, 0), dtype=np.float64)
    out['body_depth_all_mmpose'] = np.zeros((Nf, 0), dtype=np.float64)

    # Also keep clip-level provenance per frame (useful for ST-VFT Phase 1 grouping)
    out['_clip_id']      = np.repeat(np.arange(Nc, dtype=np.int64), L)
    out['_frame_in_clip'] = np.tile(np.arange(L, dtype=np.int64), Nc)
    return out


def concat_frame_dicts(dicts):
    """Concatenate flat-per-chunk dicts along frame axis."""
    if len(dicts) == 1:
        return dicts[0]
    merged = {}
    arr_keys  = [k for k, v in dicts[0].items() if isinstance(v, np.ndarray)]
    list_keys = [k for k, v in dicts[0].items() if isinstance(v, list)]
    for k in arr_keys:
        merged[k] = np.concatenate([d[k] for d in dicts], axis=0)
    for k in list_keys:
        merged[k] = []
        for d in dicts:
            merged[k].extend(d[k])
    return merged


def select_indices(flat, idx):
    """Return a new flat dict containing only rows at `idx` (numpy index)."""
    sub = {}
    for k, v in flat.items():
        if isinstance(v, np.ndarray):
            sub[k] = v[idx]
        else:
            sub[k] = [v[i] for i in idx]
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--in-stage-name', default='stage_V_room')
    ap.add_argument('--out-dir-name', default='datasets/clip_full_room_flat')
    ap.add_argument('--subset', default='train', help='which stage_V subset to flatten+split')
    ap.add_argument('--val-frac', default=0.1, type=float)
    ap.add_argument('--seed', default=42, type=int)
    args = ap.parse_args()

    work_dir = os.path.join(args.work_dir, args.exp)
    out_root = os.path.join(work_dir, args.out_dir_name)
    os.makedirs(out_root, exist_ok=True)

    in_glob = os.path.join(work_dir, args.in_stage_name, args.subset, '*.pkl')
    chunks = sorted(glob.glob(in_glob))
    if not chunks:
        raise FileNotFoundError(f'no chunks at {in_glob}')
    print(f'flatten: {len(chunks)} chunks from {in_glob}')

    # Step 1: flatten each chunk individually, track clip ids
    flat_per_chunk = []
    chunk_clip_counts = []
    clip_id_offset = 0
    for c in tqdm(chunks, desc='flatten'):
        with open(c, 'rb') as f:
            chunk = pickle.load(f)
        flat = flatten_chunk(chunk)
        # Offset clip ids so they're globally unique
        flat['_clip_id'] = flat['_clip_id'] + clip_id_offset
        Nc = chunk['joints_3d'].shape[0]
        clip_id_offset += Nc
        chunk_clip_counts.append(Nc)
        flat_per_chunk.append(flat)

    merged = concat_frame_dicts(flat_per_chunk)
    Nf = merged['joints_3d'].shape[0]
    total_clips = clip_id_offset
    print(f'  total: {Nf} frames, {total_clips} clips')

    # Step 2: clip-level 90/10 split (deterministic, seeded shuffle of clip ids)
    rng = np.random.RandomState(args.seed)
    all_clip_ids = np.arange(total_clips, dtype=np.int64)
    rng.shuffle(all_clip_ids)
    n_val_clips = int(round(total_clips * args.val_frac))
    val_clip_ids = set(all_clip_ids[:n_val_clips].tolist())
    train_clip_ids = set(all_clip_ids[n_val_clips:].tolist())

    clip_ids_per_frame = merged['_clip_id']
    is_val   = np.isin(clip_ids_per_frame, list(val_clip_ids))
    is_train = ~is_val
    train_idx = np.where(is_train)[0]
    val_idx   = np.where(is_val)[0]
    print(f'  split: train={len(train_idx)} frames ({len(train_clip_ids)} clips), '
          f'val={len(val_idx)} frames ({len(val_clip_ids)} clips)')

    # Step 3: write
    for split_name, idx in [('train', train_idx), ('validation', val_idx)]:
        sub = select_indices(merged, idx)
        # Drop internal-only fields for the RUMPL loader
        sub.pop('_clip_id', None)
        sub.pop('_frame_in_clip', None)
        out_path = os.path.join(out_root, f'amass_mmpose_joints_{split_name}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(sub, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'    saved {out_path}: {sub["joints_3d"].shape[0]} frames, '
              f'keys={sorted(sub.keys())}')

    print('\nDONE.')


if __name__ == '__main__':
    main()
