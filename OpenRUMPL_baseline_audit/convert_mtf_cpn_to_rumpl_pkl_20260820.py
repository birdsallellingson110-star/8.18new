#!/usr/bin/env python3
"""Convert the official MTF/CPN H36M cache to the RUMPL annotation format.

The MTF package is not a drop-in RUMPL annotation file.  Its seven channels
are ``GT-2D(2), CPN-2D(2), camera-3D(3)`` and its visibility is stored in a
separate ``score.pkl``.  This converter deliberately reads only channels
2:4 (and, for C2, the separate score file), maps the VideoPose3D joint order
to RUMPL's order, and writes two ordinary RUMPL pkl files:

* C1: CPN coordinates, confidence fixed to one (CPN-XY);
* C2: the same coordinates, with the official MTF/CPN score (CPN-XYC).

The original RUMPL records are retained for camera/3-D annotations.  CPN
coordinates are converted from VideoPose3D's normalized 1000-pixel frame and
optionally undistorted into the K-with-zero-distortion coordinate system used
by the prepared RUMPL files.  No MTF camera-space 3-D channel is copied.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# MTF/VideoPose3D: root, left leg, right leg, ...; RUMPL stores right leg first.
MTF_TO_RUMPL = np.asarray(
    [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    dtype=np.int64,
)

ACTION_ALIASES = {
    2: ("Directions",),
    3: ("Discussion",),
    4: ("Eating",),
    5: ("Greeting",),
    6: ("Phoning", "Phone"),
    7: ("Photo",),
    8: ("Posing", "Pose"),
    9: ("Purchases", "Purchase"),
    10: ("Sitting",),
    11: ("SittingDown",),
    12: ("Smoking", "Smoke"),
    13: ("Waiting", "Wait"),
    14: ("WalkDog",),
    15: ("Walking", "Walk"),
    16: ("WalkTogether", "WalkTwo"),
}

# MTF uses 1000 x 1000 except cameras 0 and 3, whose H36M height is 1002.
CAMERA_HEIGHTS = np.asarray([1002.0, 1000.0, 1000.0, 1002.0], dtype=np.float64)
CAMERA_WIDTH = 1000.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mtf-data-dir", required=True)
    p.add_argument("--train-pkl", required=True)
    p.add_argument("--validation-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--no-undistort", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--split", choices=("train", "validation", "both"), default="both")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_numpy(value) -> np.ndarray:
    # MTF npz files contain pickled torch tensors.  This remains CPU-only.
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_mtf(data_dir: Path):
    positions = {}
    for subject in (1, 5, 6, 7, 8, 9, 11):
        path = data_dir / f"h36m_sub{subject}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        loaded = np.load(path, allow_pickle=True)
        payload = loaded["positions_2d"].item()[f"S{subject}"]
        positions[f"S{subject}"] = {
            str(action): [as_numpy(camera) for camera in cameras]
            for action, cameras in payload.items()
        }
    score_path = data_dir / "score.pkl"
    with score_path.open("rb") as f:
        scores = pickle.load(f)
    return positions, scores


def action_base(name: str) -> str:
    return name.rstrip().split(" ")[0]


def choose_action_key(subject_actions, action_id: int, subaction: int, max_image_id: int):
    aliases = ACTION_ALIASES.get(int(action_id), ())
    candidates = [
        key for key in subject_actions
        if action_base(key) in aliases or key in aliases
    ]
    if not candidates:
        raise KeyError(
            f"no MTF action for subject S{subject}, H36M action {action_id}"
        )
    valid = [
        key for key in candidates
        if len(subject_actions[key][0]) >= int(max_image_id)
    ]
    if not valid:
        lengths = {key: len(subject_actions[key][0]) for key in candidates}
        raise IndexError(
            f"image_id {max_image_id} exceeds MTF candidates {lengths}"
        )
    # Sparse RUMPL records have image_id as the one-based original frame id.
    # Choosing the shortest sequence that contains the largest id resolves
    # H36M subaction 1/2 without guessing from the subaction label.
    return min(valid, key=lambda key: len(subject_actions[key][0]) - max_image_id)


def cpn_score_key(subject: int, action_key: str, camera_id: int, scores):
    key = f"S{subject}_{action_key}.{camera_id}"
    if key in scores:
        return key
    # A few third-party score dumps remove spaces around the subaction.
    compact = f"S{subject}_{action_key.replace(' ', '')}.{camera_id}"
    if compact in scores:
        return compact
    raise KeyError(key)


def normalized_to_pixel(points: np.ndarray, camera_id: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (points + np.asarray([1.0, CAMERA_HEIGHTS[camera_id] / CAMERA_WIDTH])) * (
        CAMERA_WIDTH / 2.0
    )


def undistort(points: np.ndarray, record: dict) -> np.ndarray:
    camera = record["camera"]
    k = np.asarray(record.get("camera_original_distortion_k", [0, 0, 0]), dtype=np.float64)
    p = np.asarray(record.get("camera_original_distortion_p", [0, 0]), dtype=np.float64)
    distortion = np.asarray([k[0], k[1], p[0], p[1], k[2]], dtype=np.float64)
    K = np.asarray(camera["K"], dtype=np.float64)
    return cv2.undistortPoints(
        points.astype(np.float64)[:, None, :], K, distortion, P=K
    ).reshape(-1, 2)


def convert_split(
    source_path: Path,
    output_dir: Path,
    split: str,
    positions,
    scores,
    apply_undistort: bool,
    overwrite: bool,
):
    with source_path.open("rb") as f:
        source = pickle.load(f)
    if not isinstance(source, list):
        raise TypeError(f"RUMPL annotation must be a list: {source_path}")

    grouped = defaultdict(list)
    for record in source:
        grouped[
            (
                int(record["subject"]),
                int(record["action"]),
                int(record["subaction"]),
            )
        ].append(record)

    converted = {"C1": [], "C2": []}
    sequence_manifest = {}
    audit_rows = []
    for group_key, records in grouped.items():
        subject, action_id, subaction = group_key
        max_image_id = max(int(item["image_id"]) for item in records)
        subject_actions = positions[f"S{subject}"]
        action_key = choose_action_key(
            subject_actions, action_id, subaction, max_image_id
        )
        action_arrays = subject_actions[action_key]
        if len(action_arrays) != 4:
            raise ValueError(f"expected four cameras for {action_key}")
        sequence_manifest["%d/%d/%d" % group_key] = {
            "mtf_action": action_key,
            "max_image_id": max_image_id,
            "length": len(action_arrays[0]),
        }
        for record in records:
            camera_id = int(record["camera_id"])
            frame_index = int(record["image_id"]) - 1
            if frame_index < 0 or frame_index >= len(action_arrays[camera_id]):
                raise IndexError(
                    f"frame {record['image_id']} not in {action_key} camera {camera_id}"
                )
            mtf_row = action_arrays[camera_id][frame_index]
            if mtf_row.shape[-1] < 4:
                raise ValueError(f"MTF row has no predicted CPN channels: {mtf_row.shape}")
            # Only the official CPN prediction channels are used.  The first
            # two are GT 2D and the last three are camera-space 3D; both are
            # intentionally ignored here.
            cpn = normalized_to_pixel(mtf_row[:, 2:4], camera_id)
            cpn = cpn[MTF_TO_RUMPL]
            if apply_undistort:
                cpn = undistort(cpn, record)

            score_key = cpn_score_key(subject, action_key, camera_id, scores)
            score_row = np.asarray(scores[score_key])[frame_index]
            if score_row.shape != (17,):
                raise ValueError(f"bad CPN score shape {score_key}: {score_row.shape}")
            score_row = score_row[MTF_TO_RUMPL]
            if not np.isfinite(cpn).all() or not np.isfinite(score_row).all():
                raise ValueError(f"non-finite CPN row {group_key} frame {frame_index}")

            for variant in ("C1", "C2"):
                item = copy.deepcopy(record)
                item["joints_2d"] = cpn.astype(np.float32)
                item["joints_2d_conf"] = (
                    np.ones((17, 1), dtype=np.float32)
                    if variant == "C1"
                    else score_row.astype(np.float32)[:, None]
                )
                item["source_2d_protocol"] = (
                    "MTF-official-CPN-XY-undistorted-v1"
                    if apply_undistort
                    else "MTF-official-CPN-XY-raw-v1"
                )
                item["source_2d_keypoint_model"] = "CPN-ft-H36M"
                item["source_2d_detector_model"] = "VideoPose3D/MTF-official-CPN"
                item["source_2d_lower_body_swap"] = False
                item["source_2d_undistorted_full_image"] = bool(apply_undistort)
                item["camera_2d_coordinate_system"] = (
                    "undistorted_K_equals_K" if apply_undistort else "MTF_raw_distorted_K"
                )
                item["cpn_source_score"] = (
                    "constant_one" if variant == "C1" else "MTF_score.pkl"
                )
                converted[variant].append(item)

            audit_rows.append(
                {
                    "subject": subject,
                    "action": action_id,
                    "subaction": subaction,
                    "camera_id": camera_id,
                    "image_id": int(record["image_id"]),
                    "mtf_action": action_key,
                    "mtf_frame_index": frame_index,
                    "score_key": score_key,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for variant, records in converted.items():
        path = output_dir / f"h36m_{split}_{variant}.pkl"
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        with path.open("wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        outputs[variant] = {"path": str(path), "records": len(records), "sha256": sha256(path)}

    return outputs, sequence_manifest, audit_rows


def main():
    cli = parse_args()
    data_dir = Path(cli.mtf_data_dir).resolve()
    output_dir = Path(cli.output_dir).resolve()
    positions, scores = load_mtf(data_dir)
    all_manifest = {
        "method": "MTF official CPN to RUMPL coordinate-only conversion",
        "mtf_data_dir": str(data_dir),
        "mtf_files": {
            name: {"sha256": sha256(data_dir / name), "bytes": (data_dir / name).stat().st_size}
            for name in [*(f"h36m_sub{s}.npz" for s in (1, 5, 6, 7, 8, 9, 11)), "score.pkl"]
        },
        "joint_mapping_mtf_to_rumpl": MTF_TO_RUMPL.tolist(),
        "source_channels_used": "positions_2d[...,2:4] only",
        "source_channels_ignored": "GT-2D positions_2d[...,0:2] and camera-space 3D positions_2d[...,4:7]",
        "score_source": "score.pkl; C1 uses ones, C2 uses score.pkl",
        "coordinate_conversion": "VideoPose3D normalized_screen_coordinates inverse, width=1000, camera heights=[1002,1000,1000,1002]",
        "undistort": not cli.no_undistort,
        "splits": {},
    }
    split_sources = []
    if cli.split in ("train", "both"):
        split_sources.append(("train", Path(cli.train_pkl)))
    if cli.split in ("validation", "both"):
        split_sources.append(("validation", Path(cli.validation_pkl)))
    for split, source in split_sources:
        result, sequences, rows = convert_split(
            source.resolve(), output_dir, split, positions, scores,
            not cli.no_undistort, cli.overwrite
        )
        all_manifest["splits"][split] = {
            "source": str(source.resolve()),
            "source_sha256": sha256(source.resolve()),
            "outputs": result,
            "sequence_mapping": sequences,
            "rows": len(rows),
        }
        (output_dir / f"{split}_frame_map.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(all_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(all_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
