#!/usr/bin/env python3
"""Fair variable-cardinality evaluation of the official PPT checkpoint.

Every V2/V3/V4 camera subset is passed through PPT independently.  This is
essential because taking two output heatmaps from a four-view forward pass
would leak the two omitted views through cross-view attention.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms

from core.config import config, update_config
from core.inference import get_final_preds
from dataset.multiview_h36m import MultiViewH36M
import models

from eval_h36m_sparse_epipolar_topk import (
    ACTION_NAMES,
    KP_STAR,
    camera_parameters,
    pixels_to_rays,
    robust_intersection,
    target_world_metres,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--views", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--irls-iterations", type=int, default=5)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def summarize(
    errors: dict[str, dict[str, list[float]]],
    kp_errors: dict[str, list[float]],
) -> dict:
    output = {}
    for method, action_values in errors.items():
        per_action = {
            action: float(np.mean(values))
            for action, values in sorted(action_values.items())
        }
        flat = [
            value
            for action_group in action_values.values()
            for value in action_group
        ]
        output[method] = {
            "frame_weighted_all17_mm": float(np.mean(flat)),
            "action_equal_all17_mm": float(np.mean(list(per_action.values()))),
            "frame_weighted_kp_star_mm": float(np.mean(kp_errors[method])),
            "per_action_all17_mm": per_action,
        }
    return output


def main() -> None:
    args = parse_args()
    update_config(args.cfg)
    device = torch.device(args.device)
    model = models.multiview_ppt.get_multiview_pose_net(
        config, is_train=False
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    dataset = MultiViewH36M(
        config,
        config.DATASET.TEST_SUBSET,
        False,
        transforms.Compose([transforms.ToTensor(), normalize]),
    )
    if args.limit_groups:
        dataset.grouping = dataset.grouping[: args.limit_groups]
        dataset.group_size = len(dataset.grouping)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    requested = tuple(sorted(set(args.views)))
    errors = {
        n_views: defaultdict(lambda: defaultdict(list))
        for n_views in requested
    }
    kp_errors = {
        n_views: defaultdict(list) for n_views in requested
    }

    group_offset = 0
    with torch.no_grad():
        for batch_index, (inputs, _, _, metadata) in enumerate(loader):
            batch_size = inputs[0].shape[0]
            inputs = [
                tensor.to(device, non_blocking=True) for tensor in inputs
            ]
            centers = [
                item["cam_center"].float().to(device, non_blocking=True)
                for item in metadata
            ]
            rays = [
                item["rays"].float().to(device, non_blocking=True)
                for item in metadata
            ]
            for n_views in requested:
                for combination in itertools.combinations(range(4), n_views):
                    outputs = model(
                        [inputs[index] for index in combination],
                        centers=[centers[index] for index in combination],
                        rays=[rays[index] for index in combination],
                        ratio=0.7,
                        fuse=True,
                    )
                    coordinates = []
                    peaks = []
                    for local_view, camera_index in enumerate(combination):
                        heatmap = outputs[local_view].cpu().numpy()
                        coordinate, peak = get_final_preds(
                            config,
                            heatmap,
                            metadata[camera_index]["center"].numpy(),
                            metadata[camera_index]["scale"].numpy(),
                        )
                        coordinates.append(coordinate[:, :, :2])
                        peaks.append(peak[:, :, 0])
                    coordinates = np.stack(coordinates, axis=1)
                    peak_logits = np.stack(peaks, axis=1)
                    # Released PPT checkpoint and our RUMPL H36M annotations
                    # use opposite right/left leg channel conventions.
                    coordinates[:, :, 1:4], coordinates[:, :, 4:7] = (
                        coordinates[:, :, 4:7].copy(),
                        coordinates[:, :, 1:4].copy(),
                    )
                    peak_logits[:, :, 1:4], peak_logits[:, :, 4:7] = (
                        peak_logits[:, :, 4:7].copy(),
                        peak_logits[:, :, 1:4].copy(),
                    )
                    peak_weights = np.exp(
                        peak_logits
                        - peak_logits.max(axis=1, keepdims=True)
                    )

                    for sample in range(batch_size):
                        group_index = group_offset + sample
                        record_indices = [
                            dataset.grouping[group_index][camera_index]
                            for camera_index in combination
                        ]
                        group_records = [
                            dataset.db[index] for index in record_indices
                        ]
                        camera_data = [
                            camera_parameters(record)
                            for record in group_records
                        ]
                        intrinsics = [item[0] for item in camera_data]
                        rotations = [item[1] for item in camera_data]
                        camera_centers = np.stack(
                            [item[2] for item in camera_data]
                        )
                        directions = np.stack(
                            [
                                pixels_to_rays(
                                    coordinates[sample, view],
                                    intrinsics[view],
                                    rotations[view],
                                )
                                for view in range(n_views)
                            ]
                        )
                        target = target_world_metres(group_records[0])
                        action = ACTION_NAMES[
                            int(group_records[0]["action"])
                        ]
                        for method, weights in (
                            (
                                "ppt_uniform_irls",
                                np.ones((n_views, 17), dtype=np.float64),
                            ),
                            (
                                "ppt_peak_irls",
                                peak_weights[sample].astype(np.float64),
                            ),
                        ):
                            prediction = np.stack(
                                [
                                    robust_intersection(
                                        camera_centers,
                                        directions[:, joint],
                                        weights[:, joint],
                                        args.irls_iterations,
                                    )
                                    for joint in range(17)
                                ]
                            )
                            joint_error = np.linalg.norm(
                                prediction - target, axis=-1
                            ) * 1000.0
                            errors[n_views][method][action].append(
                                float(joint_error.mean())
                            )
                            kp_errors[n_views][method].append(
                                float(joint_error[list(KP_STAR)].mean())
                            )
            group_offset += batch_size
            if (
                (batch_index + 1) % 10 == 0
                or group_offset == len(dataset)
            ):
                print(
                    f"groups {group_offset}/{len(dataset)}",
                    flush=True,
                )

    result = {
        "cfg": args.cfg,
        "checkpoint": args.checkpoint,
        "groups": len(dataset),
        "protocol": (
            "independent forward pass for every camera subset; "
            "RUMPL Table-2 target/action/robust-ray evaluation"
        ),
        "results": {},
    }
    for n_views in requested:
        summary = summarize(errors[n_views], kp_errors[n_views])
        combinations = len(list(itertools.combinations(range(4), n_views)))
        result["results"][f"V{n_views}"] = {
            "views": n_views,
            "groups": len(dataset) * combinations,
            "methods": summary,
        }
        for method, metrics in summary.items():
            print(
                f"V{n_views} {method}: "
                f"All={metrics['action_equal_all17_mm']:.3f} "
                f"KP*={metrics['frame_weighted_kp_star_mm']:.3f}",
                flush=True,
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
