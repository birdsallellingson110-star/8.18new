#!/usr/bin/env python3
"""Audit CPN RUMPL annotations and run the two coordinate-level evaluators.

Evaluator A is absolute world MPJPE; evaluator B is Protocol-1 style
root-relative MPJPE.  Both are run on the same all-four-view groups, with
frame-weighted and action-equal aggregation.  This is deliberately a
zero-training audit of the converted input, not a learned-model result.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


ACTION_NAMES = {
    2: "Direction", 3: "Discuss", 4: "Eating", 5: "Greet",
    6: "Phone", 7: "Photo", 8: "Pose", 9: "Purchase",
    10: "Sitting", 11: "SittingDown", 12: "Smoke", 13: "Wait",
    14: "WalkDog", 15: "Walk", 16: "WalkTwo",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--c1-pkl", required=True)
    p.add_argument("--c2-pkl", required=True)
    p.add_argument("--hrnet-pkl", default=None)
    p.add_argument("--output", required=True)
    return p.parse_args()


def load(path):
    with Path(path).open("rb") as f:
        value = pickle.load(f)
    if not isinstance(value, list):
        raise TypeError(path)
    return value


def projection_matrix(camera):
    R = np.asarray(camera["R"], dtype=np.float64)
    # ``t`` is the actual world->camera translation.  ``T`` stores the camera
    # centre and is only converted internally by cam_to_world.
    t = np.asarray(camera["t"], dtype=np.float64).reshape(3, 1)
    K = np.asarray(camera["K"], dtype=np.float64)
    return K @ np.concatenate((R, t), axis=1)


def triangulate(points, cameras):
    rows = []
    for point, camera in zip(points, cameras):
        P = projection_matrix(camera)
        x, y = point
        rows.extend((x * P[2] - P[0], y * P[2] - P[1]))
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64), full_matrices=False)
    X = vh[-1]
    if abs(X[3]) < 1e-10:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def world_from_camera(X, camera):
    R = np.asarray(camera["R"], dtype=np.float64)
    t = np.asarray(camera["t"], dtype=np.float64).reshape(3)
    return (R.T @ (np.asarray(X, dtype=np.float64).T - t[:, None])).T


def group_records(records):
    groups = defaultdict(list)
    for item in records:
        key = (
            int(item["subject"]), int(item["action"]), int(item["subaction"]),
            int(item["image_id"]),
        )
        groups[key].append(item)
    return groups


def action_equal(values, actions):
    rows = []
    for action in sorted(np.unique(actions)):
        selected = values[actions == action]
        rows.append(float(selected.mean()))
    return float(np.mean(rows)) if rows else float("nan")


def evaluate_groups(records, variant):
    groups = group_records(records)
    if not groups:
        raise ValueError("empty records")
    errors_abs, errors_rel, actions = [], [], []
    reproj, cross_world = [], []
    malformed = []
    for key, items in groups.items():
        if len(items) != 4 or sorted(int(x["camera_id"]) for x in items) != [0, 1, 2, 3]:
            malformed.append({"key": key, "count": len(items)})
            continue
        items = sorted(items, key=lambda x: int(x["camera_id"]))
        points = np.stack([np.asarray(x["joints_2d"], dtype=np.float64) for x in items])
        cameras = [x["camera"] for x in items]
        estimates = np.stack([triangulate(points[:, joint], cameras) for joint in range(17)])
        # Reconstruct the absolute target from the first camera.  The legacy
        # prepared pkl's ``joints_3d`` field is not consistently absolute;
        # joints_3d_camera + calibration is the audited source of truth.
        target = world_from_camera(items[0]["joints_3d_camera"], items[0]["camera"])
        errors_abs.append(np.linalg.norm(estimates - target, axis=-1)[:, None])
        rel_est = estimates - estimates[0:1]
        rel_target = target - target[0:1]
        errors_rel.append(np.linalg.norm(rel_est - rel_target, axis=-1)[:, None])
        actions.append(int(key[1]))
        # Pinhole re-projection in the converted undistorted K coordinate system.
        group_world = []
        for item in items:
            K = np.asarray(item["camera"]["K"], dtype=np.float64)
            Xc = np.asarray(item["joints_3d_camera"], dtype=np.float64)
            xy = (K @ Xc.T).T
            xy = xy[:, :2] / xy[:, 2:3]
            reproj.append(np.linalg.norm(xy - item["joints_2d"], axis=-1))
            group_world.append(world_from_camera(Xc, item["camera"]))
        cross_world.append(np.stack(group_world))
    if malformed:
        raise ValueError(f"malformed four-view groups: {malformed[:3]}")
    abs_values = np.concatenate(errors_abs, axis=1).T
    rel_values = np.concatenate(errors_rel, axis=1).T
    action_values = np.asarray(actions)
    reproj_values = np.concatenate(reproj)
    world_values = np.stack(cross_world)
    # camera-level world consistency is a direct calibration/frame audit
    world_spread = np.linalg.norm(
        world_values - world_values.mean(axis=1, keepdims=True), axis=-1
    )
    return {
        "variant": variant,
        "groups": len(groups),
        "records": len(records),
        "input_coordinate_reprojection_mean_px": float(reproj_values.mean()),
        "input_coordinate_reprojection_p95_px": float(np.percentile(reproj_values, 95)),
        "cross_camera_world_consistency_mean_mm": float(world_spread.mean()),
        "absolute_world_all17_mm": {
            "frame_weighted": float(abs_values.mean()),
            "action_equal": action_equal(abs_values, action_values),
        },
        "protocol1_root_relative_all17_mm": {
            "frame_weighted": float(rel_values.mean()),
            "action_equal": action_equal(rel_values, action_values),
        },
        "per_action_absolute_action_equal_mm": {
            ACTION_NAMES.get(action, str(action)): float(abs_values[action_values == action].mean())
            for action in sorted(np.unique(action_values))
        },
        "per_action_protocol1_action_equal_mm": {
            ACTION_NAMES.get(action, str(action)): float(rel_values[action_values == action].mean())
            for action in sorted(np.unique(action_values))
        },
    }


def audit_pair(c1, c2):
    if len(c1) != len(c2):
        raise ValueError(f"C1/C2 record count differs: {len(c1)} vs {len(c2)}")
    c1_keys = [(x["subject"], x["action"], x["subaction"], x["image_id"], x["camera_id"]) for x in c1]
    c2_keys = [(x["subject"], x["action"], x["subaction"], x["image_id"], x["camera_id"]) for x in c2]
    if c1_keys != c2_keys:
        raise ValueError("C1/C2 frame or camera ordering differs")
    coord_delta = []
    c1_conf = []
    c2_conf = []
    source_channels = set()
    for a, b in zip(c1, c2):
        coord_delta.append(np.abs(np.asarray(a["joints_2d"]) - np.asarray(b["joints_2d"])))
        c1_conf.append(np.asarray(a["joints_2d_conf"]))
        c2_conf.append(np.asarray(b["joints_2d_conf"]))
        source_channels.add(a.get("source_2d_protocol"))
        if "positions_2d" in a:
            raise ValueError("raw MTF positions_2d leaked into record")
    c1_conf = np.concatenate(c1_conf).reshape(-1)
    c2_conf = np.concatenate(c2_conf).reshape(-1)
    return {
        "records": len(c1),
        "coordinates_identical_max_abs_px": float(np.max(coord_delta)),
        "c1_confidence_min_max_mean": [float(c1_conf.min()), float(c1_conf.max()), float(c1_conf.mean())],
        "c2_confidence_min_max_mean": [float(c2_conf.min()), float(c2_conf.max()), float(c2_conf.mean())],
        "c2_confidence_fraction_lt_0_5": float((c2_conf < 0.5).mean()),
        "source_protocols": sorted(source_channels),
        "c1_is_constant_one": bool(np.allclose(c1_conf, 1.0)),
    }


def main():
    cli = parse_args()
    c1 = load(cli.c1_pkl)
    c2 = load(cli.c2_pkl)
    result = {
        "audit": audit_pair(c1, c2),
        "C1": evaluate_groups(c1, "CPN-XY"),
        "C2": evaluate_groups(c2, "CPN-XYC (same coordinates; score only)"),
        "protocol": {
            "absolute": "world MPJPE, all four-view groups, frame-weighted and action-equal",
            "protocol1": "root-relative MPJPE, same groups, frame-weighted and action-equal",
            "note": "zero-training DLT audit; not a learned RUMPL result",
        },
    }
    if cli.hrnet_pkl:
        hrnet = load(cli.hrnet_pkl)
        result["HRNet_reference_own_frames"] = evaluate_groups(hrnet, "prepared HRNet")
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
