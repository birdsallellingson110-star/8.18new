#!/usr/bin/env python3
"""Convert an existing CMU COCO-17 record file to RUMPL H36M-17 order.

This is a deterministic skeleton adapter, not a learned target-domain module.
It converts the detected 2-D points, confidences, visibility flags and 3-D
ground truth with the same COCO-to-H36M virtual-joint construction used by the
RUMPL/MHP preparation code.  Images and camera calibration are left unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


COCO_TO_H36M = {
    1: 12,  # right hip
    2: 14,  # right knee
    3: 16,  # right ankle
    4: 11,  # left hip
    5: 13,  # left knee
    6: 15,  # left ankle
    9: 0,   # nose
    11: 5,  # left shoulder
    12: 7,  # left elbow
    13: 9,  # left wrist
    14: 6,  # right shoulder
    15: 8,  # right elbow
    16: 10, # right wrist
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_values(values: np.ndarray) -> np.ndarray:
    """Map an array ending in ``(17, D)`` without changing its dtype."""
    values = np.asarray(values)
    if values.shape[-2] != 17:
        raise ValueError(f"expected 17 COCO joints, got {values.shape}")
    out = np.zeros_like(values)
    for destination, source in COCO_TO_H36M.items():
        out[..., destination, :] = values[..., source, :]

    # Match OpenRUMPL/MHP/07_build_h36m_flat_from_clip.py exactly.
    head = values[..., 0:5, :].mean(axis=-2)
    neck = values[..., 3:7, :].mean(axis=-2)
    root = values[..., 11:13, :].mean(axis=-2)
    out[..., 10, :] = head
    out[..., 8, :] = neck
    out[..., 0, :] = root
    out[..., 7, :] = (neck + root) / 2.0
    return out


def convert_record(source: dict) -> dict:
    record = copy.deepcopy(source)
    for key in ("joints_2d", "joints_3d", "joints_3d_conf"):
        if key in record and record[key] is not None:
            record[key] = convert_values(np.asarray(record[key]))
    if "joints_2d_conf" in record and record["joints_2d_conf"] is not None:
        record["joints_2d_conf"] = convert_values(
            np.asarray(record["joints_2d_conf"])
        )
    if "joints_vis" in record and record["joints_vis"] is not None:
        record["joints_vis"] = convert_values(np.asarray(record["joints_vis"]))
    record["keypoint_standard"] = "h36m17_virtual_from_coco17"
    record["skeleton_adapter"] = "OpenRUMPL_MHP_07_COCO2H36M"
    return record


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    with input_path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("expected a non-empty list of CMU records")

    converted = [convert_record(record) for record in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(converted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_path)

    metadata = {
        "purpose": "deterministic CMU COCO17 to RUMPL H36M17 skeleton adapter",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "records": len(converted),
        "uses_target_labels_for_learning": False,
        "changes_images_or_cameras": False,
        "mapping": {str(key): value for key, value in COCO_TO_H36M.items()},
        "virtual_joints": {
            "root_0": "mean(COCO left/right hip 11:13)",
            "belly_7": "mean(root, neck)",
            "neck_8": "mean(COCO ears and shoulders 3:7)",
            "head_10": "mean(COCO nose/eyes/ears 0:5)",
        },
    }
    manifest = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
