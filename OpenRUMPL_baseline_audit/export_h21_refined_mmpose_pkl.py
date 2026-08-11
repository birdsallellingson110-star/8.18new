#!/usr/bin/env python3
"""Bake H21 pose-query 2D refinements into new RUMPL mmpose PKLs (H35).

Modes:
  h21       - refine from raw detector 2D / raw RUMPL anchor
  a1d_h21   - first apply A1D dense fusion, then H21 around the A1D anchor
"""

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
from eval_h23_rumpl_pose_query_anchor import (
    DIRECT_COCO,
    DIRECT_H36M,
    rumpl_anchor_h36m,
)
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
from iterative_pose_query_refiner import IterativePoseQueryRefiner
from pose_query_geometry import (
    heatmap_to_image,
    image_to_heatmap,
    project_world_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--base-mmpose-pkl", required=True)
    parser.add_argument("--dense-shards", nargs="+", required=True)
    parser.add_argument("--h21-checkpoint", required=True)
    parser.add_argument(
        "--mode",
        choices=("h21", "a1d_h21"),
        default="a1d_h21",
    )
    parser.add_argument("--a1d-checkpoint")
    parser.add_argument("--a1d-depth-min", type=float, default=1.0)
    parser.add_argument("--a1d-depth-max", type=float, default=10.0)
    parser.add_argument("--a1d-depth-samples", type=int, default=64)
    parser.add_argument("--anchor-confidence-epsilon", type=float, default=0.05)
    parser.add_argument("--anchor-regularization", type=float, default=1e-4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "a1d_h21" and not args.a1d_checkpoint:
        raise ValueError("--mode a1d_h21 requires --a1d-checkpoint")

    device = torch.device(args.device)
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    with open(args.base_mmpose_pkl, "rb") as handle:
        base_records = pickle.load(handle)
    if len(records) != len(base_records):
        raise ValueError("gt/base length mismatch")

    store = DenseHeatmapStore(args.dense_shards)
    h21_payload = torch.load(args.h21_checkpoint, map_location=device)
    h21 = IterativePoseQueryRefiner().to(device)
    h21.load_state_dict(h21_payload["model"])
    h21.eval()
    h21_joint_ids = torch.as_tensor(DIRECT_COCO, dtype=torch.long, device=device)

    a1d_model = None
    a1d_depths = None
    a1d_joint_ids = None
    if args.mode == "a1d_h21":
        a1d_payload = torch.load(args.a1d_checkpoint, map_location=device)
        a1d_model = DenseGeometryResidualFusion().to(device)
        a1d_model.load_state_dict(a1d_payload["model"])
        a1d_model.eval()
        a1d_depths = torch.linspace(
            args.a1d_depth_min,
            args.a1d_depth_max,
            args.a1d_depth_samples,
            device=device,
        )
        a1d_joint_ids = torch.as_tensor(
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
        n_views = len(indices)
        raw_coco = data["decoded_keypoints"].astype(np.float64)
        coco_confidence = data["decoded_scores"].astype(np.float64)
        heatmaps_np = data["heatmaps"][:, DIRECT_COCO].astype(np.float32)
        _, _, height, width = heatmaps_np.shape

        if args.mode == "a1d_h21":
            camera = [camera_parameters(record) for record in group_records]
            seed_coco = a1d_corrected_coco(
                a1d_model,
                a1d_depths,
                a1d_joint_ids,
                data["heatmaps"],
                raw_coco,
                data["input_center"],
                data["input_scale"],
                [item[0] for item in camera],
                [item[1] for item in camera],
                np.stack([item[2] for item in camera]),
                device,
            )
        else:
            seed_coco = raw_coco

        seed_h36m = []
        seed_conf = []
        for view in range(n_views):
            joints, conf = coco_to_h36m(seed_coco[view], coco_confidence[view])
            seed_h36m.append(joints)
            seed_conf.append(conf)
        seed_h36m = np.stack(seed_h36m)
        seed_conf = np.stack(seed_conf)

        anchor = rumpl_anchor_h36m(
            group_records,
            seed_h36m,
            seed_conf,
            args.anchor_confidence_epsilon,
            args.anchor_regularization,
        )
        query_image = project_world_points(
            group_records, anchor[DIRECT_H36M]
        )
        query_hm_np = image_to_heatmap(
            query_image,
            data["input_center"],
            data["input_scale"],
            width,
            height,
        )
        detector_hm_np = image_to_heatmap(
            seed_coco[:, DIRECT_COCO],
            data["input_center"],
            data["input_scale"],
            width,
            height,
        )
        direct_confidence = coco_confidence[:, DIRECT_COCO]
        with torch.no_grad():
            refined_hm, _ = h21(
                torch.as_tensor(heatmaps_np, dtype=torch.float32, device=device),
                torch.as_tensor(query_hm_np, dtype=torch.float32, device=device),
                torch.as_tensor(
                    detector_hm_np, dtype=torch.float32, device=device
                ),
                torch.as_tensor(
                    direct_confidence, dtype=torch.float32, device=device
                ),
                h21_joint_ids,
            )
        refined_coco = seed_coco.copy()
        refined_coco[:, DIRECT_COCO] = heatmap_to_image(
            refined_hm.cpu().numpy(),
            data["input_center"],
            data["input_scale"],
            width,
            height,
        )

        for view, record_index in enumerate(indices):
            joints, confidence = coco_to_h36m(
                refined_coco[view], coco_confidence[view]
            )
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
                        "mode": args.mode,
                        "groups_done": group_id,
                        "groups_total": len(groups),
                        "mean_joint_delta_px": float(np.mean(deltas)),
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "untouched_records": int((~touched).sum()),
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
                "mean_joint_delta_px": float(np.mean(deltas)) if deltas else 0.0,
                "elapsed_seconds": time.time() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
