"""Build a flat RUMPL AMASS dataset with Human3.6M joint definitions.

This reuses the already generated temporal clip data:
  - 2D MMPOSE detections, confidences, and cameras from stage_V_room/train/*.pkl
  - H36M 3D joints from train_h36m3d/*_h36m3d.pkl

It flattens clip frames into single-frame samples so the existing
multiview_amass_rumpl loader can train without a temporal dataset class.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np


COCO2H36M = {
    1: 12,  # rhip
    2: 14,  # rkne
    3: 16,  # rank
    4: 11,  # lhip
    5: 13,  # lkne
    6: 15,  # lank
    9: 0,   # nose
    11: 5,  # lsho
    12: 7,  # lelb
    13: 9,  # lwri
    14: 6,  # rsho
    15: 8,  # relb
    16: 10, # rwri
}


def coco_to_h36m(points: np.ndarray, confs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map COCO-17 arrays to the H36M-17 order used by RUMPL.

    points: (..., 17, 2)
    confs:  (..., 17, 1)
    """
    out = np.zeros_like(points)
    out_conf = np.zeros_like(confs)

    for h36m_i, coco_i in COCO2H36M.items():
        out[..., h36m_i, :] = points[..., coco_i, :]
        out_conf[..., h36m_i, :] = confs[..., coco_i, :]

    head = points[..., 0:5, :].mean(axis=-2)
    head_conf = confs[..., 0:5, :].mean(axis=-2)
    out[..., 10, :] = head
    out_conf[..., 10, :] = head_conf

    neck = points[..., 3:7, :].mean(axis=-2)
    neck_conf = confs[..., 3:7, :].mean(axis=-2)
    out[..., 8, :] = neck
    out_conf[..., 8, :] = neck_conf

    root = points[..., 11:13, :].mean(axis=-2)
    root_conf = confs[..., 11:13, :].mean(axis=-2)
    out[..., 0, :] = root
    out_conf[..., 0, :] = root_conf

    out[..., 7, :] = (neck + root) / 2.0
    out_conf[..., 7, :] = (neck_conf + root_conf) / 2.0
    return out.astype(np.float32), out_conf.astype(np.float32)


def flatten_chunk(clip_path: str, h36m_dir: str) -> dict:
    with open(clip_path, "rb") as f:
        clip = pickle.load(f)

    h_path = Path(h36m_dir) / (Path(clip_path).stem + "_h36m3d.pkl")
    with open(h_path, "rb") as f:
        h36m = pickle.load(f)

    joints_3d = h36m["joints_3d_h36m"].astype(np.float32)  # (N,T,17,3)
    joints_2d, confs = coco_to_h36m(
        clip["joints_2d_mmpose"].astype(np.float32),
        clip["confs_2d_mmpose"].astype(np.float32),
    )
    joints_2d_gt, _ = coco_to_h36m(
        clip["joints_2d_amass"].astype(np.float32),
        np.ones_like(clip["confs_2d_mmpose"], dtype=np.float32),
    )

    n_clip, n_frames = joints_3d.shape[:2]
    flat_n = n_clip * n_frames
    cams = []
    for i in range(n_clip):
        cams.extend([clip["camera_parameters_all"][i]] * n_frames)

    return {
        "joints_3d": joints_3d.reshape(flat_n, 17, 3),
        "joints_2d_mmpose": joints_2d.reshape(flat_n, joints_2d.shape[2], 17, 2),
        "confs_2d_mmpose": confs.reshape(flat_n, confs.shape[2], 17, 1),
        "joints_2d_amass": joints_2d_gt.reshape(flat_n, joints_2d_gt.shape[2], 17, 2),
        "camera_parameters_all": cams,
        "camera_setup_used": np.zeros(flat_n, dtype=np.int64),
        "views_used": np.tile(np.arange(joints_2d.shape[2], dtype=np.int64), (flat_n, 1)),
    }


def concat_parts(parts: list[dict]) -> dict:
    out = {}
    for key in [
        "joints_3d",
        "joints_2d_mmpose",
        "confs_2d_mmpose",
        "joints_2d_amass",
        "camera_setup_used",
        "views_used",
    ]:
        out[key] = np.concatenate([p[key] for p in parts], axis=0)
    out["camera_parameters_all"] = []
    for p in parts:
        out["camera_parameters_all"].extend(p["camera_parameters_all"])
    out["triangulated_3d_mmpose"] = None
    return out


def save_split(paths: list[str], h36m_dir: str, out_file: str) -> None:
    parts = []
    for i, path in enumerate(paths, 1):
        parts.append(flatten_chunk(path, h36m_dir))
        if i % 25 == 0 or i == len(paths):
            n = sum(p["joints_3d"].shape[0] for p in parts)
            print(f"  loaded {i}/{len(paths)} chunks, buffered {n} frames")
    data = concat_parts(parts)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {out_file}")
    print(f"        joints_3d={data['joints_3d'].shape}  "
          f"joints_2d={data['joints_2d_mmpose'].shape}  "
          f"cameras={len(data['camera_parameters_all'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", default="/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train")
    ap.add_argument("--h36m-dir", default="/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train_h36m3d")
    ap.add_argument("--out-dir", default="/mnt/data/cjydata/mhp_workspace/clip_full_h36m_flat/datasets/official_combined_h36m_flat")
    ap.add_argument("--val-chunks", type=int, default=30)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.clip_dir, "*.pkl")))
    if not paths:
        raise FileNotFoundError(f"No clip pkl files found in {args.clip_dir}")
    if len(paths) <= args.val_chunks:
        raise ValueError("--val-chunks must be smaller than the number of clip chunks")

    train_paths = paths[:-args.val_chunks]
    val_paths = paths[-args.val_chunks:]
    print(f"[split] train chunks={len(train_paths)} validation chunks={len(val_paths)}")

    save_split(train_paths, args.h36m_dir, os.path.join(args.out_dir, "amass_mmpose_joints_train.pkl"))
    save_split(val_paths, args.h36m_dir, os.path.join(args.out_dir, "amass_mmpose_joints_validation.pkl"))


if __name__ == "__main__":
    main()
