"""Clip-aware AMASS dataset creator for ST-VFT method.

Replaces RUMPL's `run_mmpose_01_create_dataset.py` which samples discrete poses.
This script samples CLIPS of L_clip consecutive frames with stride to avoid overlap.

Output: stage_IV/<subset>/amass_clips_<i>.pt — each .pt is a list of dicts:
    {
      'pose':          (L_clip, 156) float32  # AMASS pose (52 axis-angle joints)
      'trans':         (L_clip, 3)   float32  # root translation per frame
      'betas':         (16,)         float32  # body shape, per-clip (same subject)
      'gender':        int           # 0='m', 1='f', 2='n'
      'source_npz':    str
      'start_frame':   int
      'frame_rate':    float
    }

Sampling: for each .npz sequence of N frames, take central 80%, generate stride-13
clip starts, then random subsample with keep_rate to control dataset size.
"""

import os
import sys
import glob
import json
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm


# Paper's 23 train subsets vs CMU for valid/test
amass_splits = {
    'validation': ['CMU'],
    'test':       ['CMU'],
    'train': ['Eyes_Japan_Dataset', 'ACCAD', 'DFaust_67', 'BMLhandball',
              'BioMotionLab_NTroje', 'SFU', 'Transitions_mocap', 'TCD_handMocap',
              'TotalCapture', 'KIT', 'MPI_HDM05', 'HumanEva', 'MPI_mosh',
              'BMLmovi', 'SOMA', 'MPI_Limits', 'WEIZMANN', 'EKUT',
              'SSM_synced', 'GRAB', 'DanceDB', 'HUMAN4D', 'CNRS'],
}

GENDER_MAP = {'male': 0, 'female': 1, 'neutral': 2,
              "b'male'": 0, "b'female'": 1, "b'neutral'": 2}


def gender_to_int(g):
    s = str(g).strip().replace('"', '').replace("'", '').replace('b', '')
    if s.startswith('m'): return 0
    if s.startswith('f'): return 1
    return 2


def sample_clips_from_npz(npz_path, L_clip, stride, keep_rate, central_pct=0.8, rng=None):
    """Return list of clip dicts from a single .npz sequence.

    - central_pct: drop first/last (1-central_pct)/2 of sequence
    - stride: spacing between clip starts (controls overlap; stride=L_clip//2 → ≤50% overlap)
    - keep_rate: fraction of valid starts to actually keep (controls total volume)
    """
    if rng is None:
        rng = random
    try:
        cdata = np.load(npz_path)
    except Exception as e:
        return []
    N = len(cdata['poses'])
    if N < L_clip + 10:
        return []

    margin = int((1 - central_pct) / 2 * N)
    valid_starts = list(range(margin, N - L_clip - margin + 1, stride))
    if len(valid_starts) == 0:
        return []

    keep_n = max(1, int(len(valid_starts) * keep_rate))
    keep_n = min(keep_n, len(valid_starts))
    sampled = sorted(rng.sample(valid_starts, keep_n))

    framerate = float(cdata.get('mocap_framerate', 60.0))
    gender = gender_to_int(cdata['gender'])
    betas = cdata['betas'][:16].astype(np.float32)  # truncate to 16 if more

    clips = []
    for start in sampled:
        clip = {
            'pose':        cdata['poses'][start:start + L_clip].astype(np.float32),
            'trans':       cdata['trans'][start:start + L_clip].astype(np.float32),
            'betas':       betas,
            'gender':      gender,
            'source_npz':  npz_path,
            'start_frame': int(start),
            'frame_rate':  framerate,
        }
        clips.append(clip)
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amass-data-dir', required=True, type=str)
    ap.add_argument('--work-dir', required=True, type=str)
    ap.add_argument('--exp', required=True, type=str)
    ap.add_argument('--L-clip', default=27, type=int)
    ap.add_argument('--stride', default=None, type=int, help='default = L_clip // 2')
    ap.add_argument('--keep-rate', default=0.5, type=float,
                    help='fraction of stride-spaced starts to keep (controls volume)')
    ap.add_argument('--n-splits', default=4, type=int, help='# stage_IV chunks per subset split')
    ap.add_argument('--train-datasets', nargs='+', default=None)
    ap.add_argument('--valid-datasets', nargs='+', default=None)
    ap.add_argument('--operation-on', nargs='+', default=['train', 'validation'])
    ap.add_argument('--seed', default=0, type=int)
    args = ap.parse_args()

    if args.stride is None:
        args.stride = args.L_clip // 2

    if args.train_datasets is not None:
        amass_splits['train'] = args.train_datasets
    if args.valid_datasets is not None:
        amass_splits['validation'] = args.valid_datasets

    work_dir = os.path.join(args.work_dir, args.exp)
    os.makedirs(work_dir, exist_ok=True)

    rng = random.Random(args.seed)

    log_path = os.path.join(work_dir, f'{args.exp}.log')
    log = open(log_path, 'a')
    def L(msg):
        print(msg); log.write(msg + '\n'); log.flush()

    L(f'[{args.exp}] L_clip={args.L_clip} stride={args.stride} keep_rate={args.keep_rate}')
    L(f'  amass_dir={args.amass_data_dir}')
    L(f'  work_dir={work_dir}')

    for split_name in args.operation_on:
        subsets = amass_splits.get(split_name, [])
        L(f'\n=== {split_name}: {len(subsets)} subsets ===')

        all_clips = []
        per_subset_counts = {}
        for ds_name in subsets:
            ds_path = os.path.join(args.amass_data_dir, ds_name)
            npz_paths = sorted(glob.glob(os.path.join(ds_path, '*/*_poses.npz')))
            if not npz_paths:
                L(f'  [skip] {ds_name}: 0 .npz')
                continue
            L(f'  {ds_name}: {len(npz_paths)} .npz')

            subset_clips = []
            for p in tqdm(npz_paths, desc=f'  scan {ds_name}', leave=False):
                cs = sample_clips_from_npz(p, args.L_clip, args.stride, args.keep_rate, rng=rng)
                subset_clips.extend(cs)
            per_subset_counts[ds_name] = len(subset_clips)
            L(f'    → {len(subset_clips)} clips')
            all_clips.extend(subset_clips)

        rng.shuffle(all_clips)
        L(f'\n{split_name} total: {len(all_clips)} clips across {len(per_subset_counts)} subsets')
        for k, v in per_subset_counts.items():
            L(f'  {k}: {v}')

        # Split into n_splits chunks for parallel 02 processing
        out_dir = os.path.join(work_dir, 'stage_IV', split_name)
        os.makedirs(out_dir, exist_ok=True)
        splits = np.linspace(0, len(all_clips), args.n_splits + 1, dtype=int)
        for i in range(args.n_splits):
            s, e = splits[i], splits[i + 1]
            if s == e:
                continue
            chunk = all_clips[s:e]
            out_path = os.path.join(out_dir, f'amass_clips_{i}.pt')
            torch.save(chunk, out_path)
            L(f'  saved {out_path}: {len(chunk)} clips')

    log.close()
    print('\nDONE.')


if __name__ == '__main__':
    main()
