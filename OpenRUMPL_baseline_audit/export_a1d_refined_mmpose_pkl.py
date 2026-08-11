#!/usr/bin/env python3
"""Bake A1D dense cross-view 2D corrections into new RUMPL mmpose PKLs (H0)."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from dense_geometry_residual_fusion import DenseGeometryResidualFusion
from eval_h36m_dense_epipolar_heatmaps import (
    DenseHeatmapStore,
    a1d_corrected_coco,
)
from eval_h36m_sparse_epipolar_topk import (
    DIRECT_COCO_JOINTS,
    build_four_view_groups,
    camera_parameters,
    coco_to_h36m,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True, help="GT annot PKL")
    parser.add_argument(
        "--base-mmpose-pkl",
        required=True,
        help="Existing legswap mmpose PKL to deepcopy metadata from",
    )
    parser.add_argument("--dense-shards", nargs="+", required=True)
    parser.add_argument("--a1d-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--a1d-depth-min", type=float, default=1.0)
    parser.add_argument("--a1d-depth-max", type=float, default=10.0)
    parser.add_argument("--a1d-depth-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    with open(args.base_mmpose_pkl, "rb") as handle:
        base_records = pickle.load(handle)
    if len(records) != len(base_records):
        raise ValueError(
            f"length mismatch: gt={len(records)} base={len(base_records)}"
        )

    store = DenseHeatmapStore(args.dense_shards)
    payload = torch.load(args.a1d_checkpoint, map_location=device)
    model = DenseGeometryResidualFusion().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    depths = torch.linspace(
        args.a1d_depth_min,
        args.a1d_depth_max,
        args.a1d_depth_samples,
        device=device,
    )
    joint_ids = torch.as_tensor(
        DIRECT_COCO_JOINTS, dtype=torch.long, device=device
    )

    groups = build_four_view_groups(records)
    output_records = [copy.deepcopy(item) for item in base_records]
    touched = np.zeros(len(records), dtype=bool)
    deltas = []
    started = time.time()

    for group_id, indices in enumerate(groups, start=1):
        data = store.get(indices)
        group_records = [records[index] for index in indices]
        camera = [camera_parameters(record) for record in group_records]
        coco = a1d_corrected_coco(
            model,
            depths,
            joint_ids,
            data["heatmaps"],
            data["decoded_keypoints"],
            data["input_center"],
            data["input_scale"],
            [item[0] for item in camera],
            [item[1] for item in camera],
            np.stack([item[2] for item in camera]),
            device,
        )
        scores = data["decoded_scores"].astype(np.float64)
        for view, record_index in enumerate(indices):
            joints, confidence = coco_to_h36m(coco[view], scores[view])
            joints = joints.astype(np.float32)
            confidence = confidence.astype(np.float32).reshape(17, 1)
            old = np.asarray(
                base_records[record_index]["joints_2d"], dtype=np.float32
            )
            deltas.append(float(np.linalg.norm(joints - old, axis=-1).mean()))
            output_records[record_index]["joints_2d"] = joints
            output_records[record_index]["joints_2d_conf"] = confidence
            touched[record_index] = True

        if group_id % args.log_every == 0 or group_id == len(groups):
            print(
                json.dumps(
                    {
                        "groups_done": group_id,
                        "groups_total": len(groups),
                        "mean_joint_delta_px": float(np.mean(deltas)),
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )

    untouched = int((~touched).sum())
    if untouched:
        # Damaged / incomplete frames keep the base mmpose 2D unchanged.
        print(
            json.dumps(
                {
                    "untouched_records": untouched,
                    "touched_records": int(touched.sum()),
                }
            ),
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(output_records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "saved": str(output),
                "records": len(output_records),
                "groups": len(groups),
                "mean_joint_delta_px": float(np.mean(deltas)) if deltas else 0.0,
                "elapsed_seconds": time.time() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
