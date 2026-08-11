#!/usr/bin/env python3
"""Convert downloaded H36M h5/JPG bundle to RUMPL train or validation PKL.

Protocol (aligned with standard H36M multi-view generalization benchmarks):
  - Train subjects: S1, S5, S6, S7, S8
  - Test subjects:  S9, S11
  - Base temporal grid: every 5 original frames (H36M-Toolbox bundle, the ``5`` in
    ``annot_filtered_5_64``)
  - Train sampling: all complete 4-camera sync groups on that grid (no extra stride)
  - Val sampling:   every 13 groups on that grid (~64 original frames; 5*13=65)
  - Val only: drop known damaged S9 sequences
  - 15 test actions; train uses all action sequences present in train subjects
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np


CAMERA_SERIALS = ["54138969", "55011271", "58860488", "60457274"]
TRAIN_SUBJECTS = (1, 5, 6, 7, 8)
VAL_SUBJECTS = (9, 11)
NAME_RE = re.compile(
    r"^S(?P<subject>\d+)_(?P<take>.+)\.(?P<camera>\d{8})_"
    r"(?P<frame>\d{6})\.jpg$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        required=True,
    )
    parser.add_argument(
        "--prepared-root",
        default="/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared",
    )
    parser.add_argument(
        "--toolbox-root",
        default=(
            "/mnt/data/cjydata/datasets/h36m_rumpl_official/"
            "toolbox/H36M-Toolbox"
        ),
    )
    parser.add_argument(
        "--group-stride",
        type=int,
        default=None,
        help="Override group stride (default: 1 train, 13 validation).",
    )
    parser.add_argument("--dataset-name", default="annot_filtered_5_64")
    parser.add_argument(
        "--skip-symlinks",
        action="store_true",
        help="Do not create eval-layout symlinks (pkl only).",
    )
    return parser.parse_args()


def load_take_mapping(toolbox_root, subjects):
    old_cwd = os.getcwd()
    try:
        os.chdir(toolbox_root)
        sys.path.insert(0, str(toolbox_root))
        from metadata import load_h36m_metadata

        metadata = load_h36m_metadata()
        mapping = {}
        for subject in subjects:
            for action in range(2, 17):
                for subaction in (1, 2):
                    base = metadata.get_base_filename(
                        f"S{subject}",
                        str(action),
                        str(subaction),
                        CAMERA_SERIALS[0],
                    )
                    take = base.rsplit(".", 1)[0].replace(" ", "_")
                    mapping[(subject, take)] = (action, subaction)
        return mapping
    finally:
        os.chdir(old_cwd)


def is_damaged(subject, action, subaction):
    return subject == 9 and (action, subaction) in {(5, 2), (10, 2), (13, 1)}


def ensure_eval_symlink(h36m_root: Path, item: dict) -> str:
    eval_image = (
        f"s_{item['subject']:02d}_act_{item['action']:02d}_"
        f"subact_{item['subaction']:02d}_imgid_{item['frame']:06d}/"
        f"cam_{item['camera_index']}_{item['frame']:06d}.jpg"
    )
    source_path = (h36m_root / "images" / item["image"]).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"missing source image: {source_path}")
    eval_path = h36m_root / "images" / eval_image
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    if eval_path.is_symlink():
        if eval_path.resolve() == source_path:
            return eval_image
        eval_path.unlink()
    elif eval_path.exists():
        if eval_path.resolve() == source_path:
            return eval_image
        raise FileExistsError(f"eval path exists and is not a symlink: {eval_path}")
    eval_path.symlink_to(source_path)
    return eval_image


def build_record(item, h5_file, camera_data, h36m_root, create_symlinks):
    subject = item["subject"]
    camera_index = item["camera_index"]
    R, T, focal, center_cam, radial, tangential, _ = camera_data[
        (subject, camera_index)
    ]
    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(3, 1)
    focal = np.asarray(focal, dtype=np.float64).reshape(2)
    center_cam = np.asarray(center_cam, dtype=np.float64).reshape(2)
    row_index = item["row_index"]
    joints_camera = np.asarray(h5_file["S"][row_index], dtype=np.float64)
    joints_2d = np.asarray(h5_file["part"][row_index], dtype=np.float64)
    joints_world = joints_camera @ R + T.reshape(1, 3)
    crop_center = np.asarray(h5_file["center"][row_index], dtype=np.float64)
    crop_scale_scalar = float(h5_file["scale"][row_index])
    crop_scale = np.array([crop_scale_scalar, crop_scale_scalar], dtype=np.float64)
    half_size = crop_scale_scalar * 100.0

    camera = {
        "R": R,
        "T": T,
        "t": -R @ T,
        "fx": float(focal[0]),
        "fy": float(focal[1]),
        "cx": float(center_cam[0]),
        "cy": float(center_cam[1]),
        "K": np.array(
            [
                [focal[0], 0.0, center_cam[0]],
                [0.0, focal[1], center_cam[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "k": np.asarray(radial, dtype=np.float64).reshape(3),
        "p": np.asarray(tangential, dtype=np.float64).reshape(2),
    }
    if create_symlinks:
        eval_image = ensure_eval_symlink(h36m_root, item)
    else:
        eval_image = item["image"]
    return {
        "image": eval_image,
        "source_image": item["image"],
        "joints_2d": joints_2d,
        "joints_2d_conf": np.ones((17, 1), dtype=np.float64),
        "joints_3d": joints_world,
        "joints_3d_camera": joints_camera,
        "joints_vis": np.ones((17, 3), dtype=np.float64),
        "video_id": item["subject"] * 1000 + item["action"] * 10 + item["subaction"],
        "image_id": item["frame"],
        "subject": item["subject"],
        "action": item["action"],
        "subaction": item["subaction"],
        "camera_id": item["camera_index"] - 1,
        "source": "h36m",
        "camera": camera,
        "center": crop_center,
        "scale": crop_scale,
        "box": np.array(
            [
                crop_center[0] - half_size,
                crop_center[1] - half_size,
                crop_center[0] + half_size,
                crop_center[1] + half_size,
            ],
            dtype=np.float64,
        ),
    }


def main():
    args = parse_args()
    is_train = args.split == "train"
    subjects = TRAIN_SUBJECTS if is_train else VAL_SUBJECTS
    group_stride = args.group_stride if args.group_stride is not None else (
        1 if is_train else 13
    )

    prepared_root = Path(args.prepared_root)
    h36m_root = prepared_root / "h36m"
    annot_root = h36m_root / "annot"
    toolbox_root = Path(args.toolbox_root)

    list_name = "train_images.txt" if is_train else "valid_images.txt"
    h5_name = "train.h5" if is_train else "valid.h5"
    pkl_name = "h36m_train.pkl" if is_train else "h36m_validation.pkl"

    take_mapping = load_take_mapping(toolbox_root, subjects)
    camera_to_index = {serial: i + 1 for i, serial in enumerate(CAMERA_SERIALS)}
    with (toolbox_root / "camera_data.pkl").open("rb") as handle:
        camera_data = pickle.load(handle)

    with (annot_root / list_name).open() as handle:
        image_names = [line.strip() for line in handle if line.strip()]

    parsed = []
    groups = OrderedDict()
    for row_index, image_name in enumerate(image_names):
        match = NAME_RE.match(image_name)
        if not match:
            raise ValueError(f"无法解析图像名: {image_name}")
        subject = int(match.group("subject"))
        if subject not in subjects:
            continue
        take = match.group("take")
        camera_serial = match.group("camera")
        frame = int(match.group("frame"))
        action, subaction = take_mapping[(subject, take)]
        camera_index = camera_to_index[camera_serial]
        item = {
            "row_index": row_index,
            "image": image_name,
            "subject": subject,
            "action": action,
            "subaction": subaction,
            "camera_index": camera_index,
            "frame": frame,
        }
        parsed.append(item)
        if not is_train and is_damaged(subject, action, subaction):
            continue
        key = (subject, action, subaction, frame)
        groups.setdefault(key, {})[camera_index] = row_index

    complete_groups = [
        (key, rows)
        for key, rows in groups.items()
        if all(camera_index in rows for camera_index in range(1, 5))
    ]
    selected_groups = complete_groups[::group_stride]
    selected_rows = {
        row_index for _, rows in selected_groups for row_index in rows.values()
    }

    records = []
    missing_images = 0
    with h5py.File(annot_root / h5_name, "r") as h5_file:
        for item in parsed:
            if item["row_index"] not in selected_rows:
                continue
            source_path = h36m_root / "images" / item["image"]
            if not source_path.is_file():
                missing_images += 1
                continue
            try:
                records.append(
                    build_record(
                        item,
                        h5_file,
                        camera_data,
                        h36m_root,
                        create_symlinks=not args.skip_symlinks,
                    )
                )
            except FileNotFoundError:
                missing_images += 1

    output_dir = h36m_root / "data" / "datasets" / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / pkl_name
    with output_file.open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "protocol": {
            "train_subjects": list(TRAIN_SUBJECTS),
            "test_subjects": list(VAL_SUBJECTS),
            "base_frame_step": 5,
            "val_equivalent_original_stride": 64,
            "group_stride_on_base_grid": group_stride,
            "train_uses_all_sync_groups": is_train,
            "val_drops_damaged_s9_sequences": not is_train,
        },
        "split": args.split,
        "subjects": list(subjects),
        "source_rows": len(image_names),
        "parsed_rows": len(parsed),
        "complete_groups": len(complete_groups),
        "group_stride": group_stride,
        "selected_groups": len(selected_groups),
        "selected_records": len(records),
        "skipped_missing_images": missing_images,
        "output": str(output_file),
    }
    summary_name = (
        "train_conversion_summary.json"
        if is_train
        else "conversion_summary.json"
    )
    (output_dir / summary_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if missing_images:
        raise SystemExit(
            f"{missing_images} records skipped: extract train-subject JPG tars first."
        )


if __name__ == "__main__":
    main()
