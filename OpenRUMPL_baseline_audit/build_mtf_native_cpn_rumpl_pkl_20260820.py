#!/usr/bin/env python3
"""Build self-contained RUMPL CPN annotations from official MTF sequences.

The public MTF cache has synchronized CPN 2-D and camera-space 3-D targets,
but its sequence frame ids are not identical to the image ids in our older
RUMPL detector pkl for every training action.  This script therefore does not
join those two frame tables.  It uses MTF's own sequence rows and only borrows
the subject/camera calibration from a prepared RUMPL pkl.

The MTF 3-D fields are used only to create the supervised target:
the root row is an absolute camera-space point and all other rows are
root-relative.  The model input is *only* CPN ``positions_2d[...,2:4]`` and,
for C2, ``score.pkl``.  The 3-D fields are never written to an input feature.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


MTF_TO_RUMPL = np.asarray(
    [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    dtype=np.int64,
)
ACTION_ALIASES = {
    "Directions": 2, "Discussion": 3, "Eating": 4, "Greeting": 5,
    "Phoning": 6, "Photo": 7, "Posing": 8, "Purchases": 9,
    "Sitting": 10, "SittingDown": 11, "Smoking": 12, "Waiting": 13,
    "WalkDog": 14, "WalkTogether": 16, "Walking": 15,
}
CAMERA_HEIGHTS = np.asarray([1002.0, 1000.0, 1000.0, 1002.0], dtype=np.float64)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mtf-data-dir", required=True)
    p.add_argument("--camera-template-pkl", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-stride", type=int, default=5)
    p.add_argument("--validation-stride", type=int, default=65)
    p.add_argument("--no-undistort", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_templates(paths):
    cameras = {}
    for path in paths:
        with Path(path).open("rb") as f:
            records = pickle.load(f)
        for record in records:
            key = (int(record["subject"]), int(record["camera_id"]))
            cameras.setdefault(key, copy.deepcopy(record))
        del records
    missing = [
        (subject, camera)
        for subject in (1, 5, 6, 7, 8, 9, 11)
        for camera in range(4)
        if (subject, camera) not in cameras
    ]
    if missing:
        raise KeyError(f"camera templates missing: {missing}")
    return cameras


def load_mtf(data_dir):
    positions = {}
    for subject in (1, 5, 6, 7, 8, 9, 11):
        path = data_dir / f"h36m_sub{subject}.npz"
        payload = np.load(path, allow_pickle=True)["positions_2d"].item()[f"S{subject}"]
        positions[subject] = {
            str(action): [as_numpy(camera) for camera in cameras]
            for action, cameras in payload.items()
        }
    with (data_dir / "score.pkl").open("rb") as f:
        scores = pickle.load(f)
    return positions, scores


def mtf_score(scores, subject, action, camera, frame):
    key = f"S{subject}_{action}.{camera}"
    if key not in scores:
        key = f"S{subject}_{action.replace(' ', '')}.{camera}"
    if key not in scores:
        raise KeyError(key)
    value = np.asarray(scores[key])[frame]
    if value.shape != (17,):
        raise ValueError(f"bad score shape {key}: {value.shape}")
    return value[MTF_TO_RUMPL].astype(np.float32)


def cpn_pixels(row, camera_id, template, undistort):
    # Official VideoPose3D normalization: [0,w] -> [-1,1], preserving h/w.
    cpn = (row[:, 2:4] + np.asarray([1.0, CAMERA_HEIGHTS[camera_id] / 1000.0])) * 500.0
    cpn = cpn[MTF_TO_RUMPL].astype(np.float64)
    if undistort:
        camera = template["camera"]
        k = np.asarray(template["camera_original_distortion_k"], dtype=np.float64)
        p = np.asarray(template["camera_original_distortion_p"], dtype=np.float64)
        dist = np.asarray([k[0], k[1], p[0], p[1], k[2]], dtype=np.float64)
        cpn = cv2.undistortPoints(
            cpn[:, None, :], np.asarray(camera["K"], dtype=np.float64), dist,
            P=np.asarray(camera["K"], dtype=np.float64),
        ).reshape(-1, 2)
    return cpn.astype(np.float32)


def camera_to_world(points_cam_mm, camera):
    R = np.asarray(camera["R"], dtype=np.float64)
    # Prepared H36M records keep camera centre in ``T`` and the actual
    # world->camera translation in ``t``; RUMPL's cam_to_world consumes t.
    t = np.asarray(camera["t"], dtype=np.float64).reshape(3)
    return ((R.T @ (points_cam_mm.T - t[:, None])).T).astype(np.float32)


def action_info(action_key, ordinal):
    base = action_key.split(" ")[0]
    if base not in ACTION_ALIASES:
        raise KeyError(f"unknown MTF action {action_key}")
    # The numeric subaction is only a grouping key.  MTF action names contain
    # suffixes such as ``Greeting 2`` that are not always the RUMPL label.
    # A deterministic ordinal keeps repeated base actions separate without
    # relying on an unreliable image-id join.
    return ACTION_ALIASES[base], int(ordinal)


def build_split(subject_positions, scores, templates, split, stride, undistort):
    if stride < 1:
        raise ValueError("stride must be positive")
    records = {"C1": [], "C2": []}
    seq_manifest = {}
    allowed_subjects = {1, 5, 6, 7, 8} if split == "train" else {9, 11}
    for subject, actions in subject_positions.items():
        if int(subject) not in allowed_subjects:
            continue
        by_base = {}
        for action_key in actions:
            by_base.setdefault(action_key.split(" ")[0], []).append(action_key)
        for base, keys in by_base.items():
            # Sort by the literal MTF key so the ordinal is stable across runs.
            for ordinal, action_key in enumerate(sorted(keys), start=1):
                arrays = actions[action_key]
                if len(arrays) != 4:
                    raise ValueError(f"{subject}/{action_key}: expected four cameras")
                lengths = [len(camera) for camera in arrays]
                nframes = min(lengths)
                action_id, subaction = action_info(action_key, ordinal)
                seq_id = f"S{subject}/{action_key}"
                seq_manifest[seq_id] = {
                    "subject": subject,
                    "action_id": action_id,
                    "subaction": subaction,
                    "lengths": lengths,
                    "frames_used": len(range(0, nframes, stride)),
                    "stride": stride,
                }
                for frame in range(0, nframes, stride):
                    row_by_camera = [np.asarray(arrays[camera][frame]) for camera in range(4)]
                    # MTF stores root absolute and all other joints root-relative.
                    cam3d = np.stack(
                        [
                            np.concatenate(
                                [
                                    row[0, 4:7].astype(np.float64)[None],
                                    row[0, 4:7].astype(np.float64)[None]
                                    + row[1:, 4:7].astype(np.float64),
                                ],
                                axis=0,
                            )
                            for row in row_by_camera
                        ],
                        axis=0,
                    )  # view, MTF-joint, xyz in metres
                    if not np.isfinite(cam3d).all():
                        raise ValueError(f"non-finite 3-D target {seq_id} frame {frame}")
                    for camera_id in range(4):
                        template = templates[(subject, camera_id)]
                        cpn = cpn_pixels(row_by_camera[camera_id], camera_id, template, undistort)
                        cpn_conf = mtf_score(scores, subject, action_key, camera_id, frame)
                        cam_mm = (cam3d[camera_id][MTF_TO_RUMPL] * 1000.0).astype(np.float32)
                        world_mm = camera_to_world(cam_mm, template["camera"])
                        for variant in ("C1", "C2"):
                            item = copy.deepcopy(template)
                            # Keep the canonical RUMPL/H36M filename grammar.
                            # Its evaluators parse the action id from split('_')[3],
                            # so a dataset-specific prefix (e.g. ``mtf_cpn/``)
                            # would make otherwise valid records unevaluable.
                            item["image"] = (
                                f"s_{subject:02}_act_{action_id:02}_"
                                f"subact_{subaction:02}_imgid_{frame + 1:06}/cam_{camera_id + 1}.jpg"
                            )
                            item["joints_2d"] = cpn
                            item["joints_2d_conf"] = (
                                np.ones((17, 1), dtype=np.float32)
                                if variant == "C1" else cpn_conf[:, None]
                            )
                            item["joints_3d_camera"] = cam_mm
                            item["joints_3d"] = world_mm
                            item["joints_vis"] = np.ones((17, 3), dtype=np.float32)
                            item["subject"] = int(subject)
                            item["action"] = int(action_id)
                            item["subaction"] = int(subaction)
                            item["camera_id"] = int(camera_id)
                            item["video_id"] = int(subject * 100000 + action_id * 1000 + subaction)
                            item["image_id"] = int(frame + 1)
                            item["center"] = np.asarray([500.0, 500.0], dtype=np.float32)
                            item["scale"] = np.asarray([1.0, 1.0], dtype=np.float32)
                            item["box"] = np.asarray([0.0, 0.0, 1000.0, 1000.0], dtype=np.float32)
                            item["source"] = "h36m"
                            item["source_2d_protocol"] = (
                                "MTF-official-CPN-XY-undistorted-v1"
                                if undistort else "MTF-official-CPN-XY-raw-v1"
                            )
                            item["source_2d_keypoint_model"] = "CPN-ft-H36M"
                            item["source_2d_detector_model"] = "VideoPose3D/MTF-official-CPN"
                            item["source_2d_lower_body_swap"] = False
                            item["source_2d_undistorted_full_image"] = bool(undistort)
                            item["camera_2d_coordinate_system"] = (
                                "undistorted_K_equals_K" if undistort else "MTF_raw_distorted_K"
                            )
                            item["cpn_source_score"] = (
                                "constant_one" if variant == "C1" else "MTF_score.pkl"
                            )
                            records[variant].append(item)
    return records, seq_manifest


def main():
    cli = parse_args()
    out = Path(cli.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    templates = load_templates(cli.camera_template_pkl)
    positions, scores = load_mtf(Path(cli.mtf_data_dir).resolve())
    manifest = {
        "method": "native MTF official CPN -> RUMPL annotations",
        "mtf_data_dir": str(Path(cli.mtf_data_dir).resolve()),
        "camera_templates": [str(Path(p).resolve()) for p in cli.camera_template_pkl],
        "joint_mapping_mtf_to_rumpl": MTF_TO_RUMPL.tolist(),
        "input_channels": "positions_2d[...,2:4] only; score.pkl only for C2 confidence",
        "ignored_channels": "positions_2d[...,0:2] GT-2D and positions_2d[...,4:7] camera-space 3D are never input",
        "target_construction": "MTF root absolute + non-root root-relative camera 3D, then camera calibration to world",
        "undistort": not cli.no_undistort,
        "splits": {},
    }
    split_cfg = (("train", cli.train_stride), ("validation", cli.validation_stride))
    for split, stride in split_cfg:
        records, sequences = build_split(
            positions, scores, templates, split, stride, not cli.no_undistort
        )
        split_manifest = {"stride": stride, "sequences": sequences, "outputs": {}}
        for variant, values in records.items():
            path = out / f"h36m_{split}_{variant}.pkl"
            if path.exists() and not cli.overwrite:
                raise FileExistsError(path)
            with path.open("wb") as f:
                pickle.dump(values, f, protocol=pickle.HIGHEST_PROTOCOL)
            split_manifest["outputs"][variant] = {
                "path": str(path), "records": len(values),
                "sha256": sha256(path),
            }
        manifest["splits"][split] = split_manifest
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
