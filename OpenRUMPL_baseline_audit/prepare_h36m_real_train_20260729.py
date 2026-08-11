#!/usr/bin/env python3
"""Convert downloaded H36M train split (h5/JPG) to RUMPL train PKL."""

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
NAME_RE = re.compile(
    r"^S(?P<subject>\d+)_(?P<take>.+)\.(?P<camera>\d{8})_"
    r"(?P<frame>\d{6})\.jpg$"
)


def parse_args():
    parser = argparse.ArgumentParser()
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
        default=1,
        help="Keep every N-th complete multiview group (1 = keep all).",
    )
    parser.add_argument("--dataset-name", default="annot_filtered_5_64")
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


def build_record(item, h5_file, camera_data, h36m_root):
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
    eval_image = (
        f"s_{item['subject']:02d}_act_{item['action']:02d}_"
        f"subact_{item['subaction']:02d}_imgid_{item['frame']:06d}/"
        f"cam_{item['camera_index']}_{item['frame']:06d}.jpg"
    )
    eval_path = h36m_root / "images" / eval_image
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = h36m_root / "images" / item["image"]
    if not eval_path.exists():
        eval_path.symlink_to(source_path)
    return {
        "image": eval_image,
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
    prepared_root = Path(args.prepared_root)
    h36m_root = prepared_root / "h36m"
    annot_root = h36m_root / "annot"
    toolbox_root = Path(args.toolbox_root)

    take_mapping = load_take_mapping(toolbox_root, TRAIN_SUBJECTS)
    camera_to_index = {serial: i + 1 for i, serial in enumerate(CAMERA_SERIALS)}
    with (toolbox_root / "camera_data.pkl").open("rb") as handle:
        camera_data = pickle.load(handle)

    with (annot_root / "train_images.txt").open() as handle:
        image_names = [line.strip() for line in handle if line.strip()]

    parsed = []
    groups = OrderedDict()
    for row_index, image_name in enumerate(image_names):
        match = NAME_RE.match(image_name)
        if not match:
            raise ValueError(f"无法解析图像名: {image_name}")
        subject = int(match.group("subject"))
        if subject not in TRAIN_SUBJECTS:
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
        key = (subject, action, subaction, frame)
        groups.setdefault(key, {})[camera_index] = row_index

    complete_groups = [
        (key, rows)
        for key, rows in groups.items()
        if all(camera_index in rows for camera_index in range(1, 5))
    ]
    selected_groups = complete_groups[:: args.group_stride]
    selected_rows = {
        row_index for _, rows in selected_groups for row_index in rows.values()
    }

    records = []
    with h5py.File(annot_root / "train.h5", "r") as h5_file:
        for item in parsed:
            if item["row_index"] not in selected_rows:
                continue
            records.append(build_record(item, h5_file, camera_data, h36m_root))

    output_dir = h36m_root / "data" / "datasets" / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "h36m_train.pkl"
    with output_file.open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "split": "train",
        "subjects": list(TRAIN_SUBJECTS),
        "source_rows": len(image_names),
        "parsed_rows_train_subjects": len(parsed),
        "complete_groups": len(complete_groups),
        "group_stride": args.group_stride,
        "selected_groups": len(selected_groups),
        "selected_records": len(records),
        "output": str(output_file),
    }
    summary_file = output_dir / "train_conversion_summary.json"
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
