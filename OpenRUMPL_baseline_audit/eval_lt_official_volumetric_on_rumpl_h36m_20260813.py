#!/usr/bin/env python3
"""Controlled subset evaluation of the official ICCV'19 Volumetric LT model.

Every V2/V3/V4 camera combination is processed independently.  Its volume
center comes from the official Algebraic LT prediction using only that subset,
so evaluation never leaks an unselected view through the precomputed pelvis.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

LT_ROOT = Path("/home/lixiaob/cjy/reference/learnable-triangulation-official")
AUDIT_ROOT = Path("/home/lixiaob/cjy/OpenRUMPL_baseline_audit")
sys.path.insert(0, str(LT_ROOT))
sys.path.insert(0, str(AUDIT_ROOT))

from eval_lt_official_on_rumpl_h36m_20260813 import (  # noqa: E402
    ACTION_NAMES,
    LT_TO_RUMPL,
    RUMPL_FROM_PRED_LT,
    RUMPLH36MImages,
    summarize,
)
from mvn.models.triangulation import (  # noqa: E402
    AlgebraicTriangulationNet,
    VolumetricTriangulationNet,
)
from mvn.utils import cfg, multiview, op  # noqa: E402
from mvn.utils.multiview import Camera  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--vol-checkpoint", required=True)
    parser.add_argument("--alg-checkpoint", required=True)
    parser.add_argument("--vol-config", default=str(
        LT_ROOT / "experiments/human36m/eval/human36m_vol_softmax.yaml"
    ))
    parser.add_argument("--alg-config", default=str(
        LT_ROOT / "experiments/human36m/eval/human36m_alg.yaml"
    ))
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--views", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument(
        "--group-indices-json",
        help=(
            "Optional RUMPL evaluation selection manifest. Only its exact "
            "grouping_index sequence is evaluated."
        ),
    )
    parser.add_argument(
        "--predictions-output",
        help=(
            "Optional compressed NPZ containing per-frame predictions and "
            "targets for controlled fusion diagnostics."
        ),
    )
    return parser.parse_args()


def load_strict(model: torch.nn.Module, path: str) -> int:
    state = torch.load(path, map_location="cpu")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(incompatible)
    return len(state)


def cameras_for_subset(batch: dict, combo: tuple[int, ...]) -> list[list[Camera]]:
    """Return the view-major nested Camera list required by official code."""
    result = []
    batch_size = batch["camera_R"].shape[0]
    for view in combo:
        camera_batch = []
        for row in range(batch_size):
            camera_batch.append(Camera(
                batch["camera_R"][row, view].numpy(),
                batch["camera_t"][row, view].numpy(),
                batch["camera_K"][row, view].numpy(),
                batch["camera_dist"][row, view].numpy(),
            ))
        result.append(camera_batch)
    return result


def main() -> None:
    args = parse_args()
    if not set(args.views).issubset({2, 3, 4}):
        raise ValueError("--views must be chosen from 2 3 4")
    torch.set_grad_enabled(False)
    device = torch.device(args.device)

    alg_config = cfg.load_config(args.alg_config)
    alg_config.model.backbone.init_weights = False
    algebraic = AlgebraicTriangulationNet(alg_config, device=device).to(device).eval()
    alg_tensors = load_strict(algebraic, args.alg_checkpoint)

    vol_config = cfg.load_config(args.vol_config)
    vol_config.model.backbone.init_weights = False
    volumetric = VolumetricTriangulationNet(vol_config, device=device).to(device).eval()
    vol_tensors = load_strict(volumetric, args.vol_checkpoint)

    dataset = RUMPLH36MImages(
        args.pkl, args.image_root, args.limit_groups, undistort=True
    )
    if args.group_indices_json:
        if len(args.views) != 1:
            raise ValueError("selection manifest requires exactly one --views value")
        manifest = json.loads(Path(args.group_indices_json).read_text())
        combination_count = len(list(itertools.combinations(range(4), args.views[0])))
        grouping_indices = [
            int(group["grouping_index"]) for group in manifest["groups"]
        ]
        group_indices = list(dict.fromkeys(
            index // combination_count for index in grouping_indices
        ))
        if not group_indices:
            raise ValueError("selection manifest contains no groups")
        if min(group_indices) < 0 or max(group_indices) >= len(dataset.groups):
            raise ValueError(
                f"selection indices outside dataset range 0..{len(dataset.groups) - 1}"
            )
        selected_groups = [dataset.groups[index] for index in group_indices]
        manifest_groups = manifest["groups"]
        if len(manifest_groups) != len(selected_groups) * combination_count:
            raise ValueError("selection manifest does not contain full frame chunks")
        for position, (_, records) in enumerate(selected_groups):
            manifest_chunk = manifest_groups[
                position * combination_count:(position + 1) * combination_count
            ]
            manifest_images = {
                image for group in manifest_chunk for image in group["images"]
            }
            dataset_images = {record["image"] for record in records}
            if manifest_images != dataset_images:
                raise ValueError(
                    f"selection frame {position} image mismatch: "
                    f"manifest={manifest_images}, dataset={dataset_images}"
                )
        dataset.groups = selected_groups
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    combinations = {
        views: list(itertools.combinations(range(4), views))
        for views in args.views
    }
    # Volumetric output is supervised in the official target LT order.  Keep a
    # semantic audit against the historically plausible backbone-channel order
    # and select neither based on headline MPJPE silently.
    mappings = {
        "official_target_lt_to_rumpl": LT_TO_RUMPL,
        "backbone_prediction_lt_to_rumpl_control": RUMPL_FROM_PRED_LT,
    }
    errors = {
        (views, combo, mapping_name, metric): []
        for views, combos in combinations.items()
        for combo in combos
        for mapping_name in mappings
        for metric in ("absolute", "relative")
    }
    prediction_batches = {
        (views, combo, mapping_name): []
        for views, combos in combinations.items()
        for combo in combos
        for mapping_name in mappings
    }
    uncertainty_batches = {
        (views, combo, mapping_name, statistic): []
        for views, combos in combinations.items()
        for combo in combos
        for mapping_name in mappings
        for statistic in ("variance_mm2", "entropy", "peak_probability")
    }
    target_batches = []
    actions_all = []

    for batch_index, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        projections = batch["projections"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        batch_size, n_views = images.shape[:2]

        # Official Algebraic observations are computed once, then each camera
        # subset gets its own DLT pose/pelvis initializer.
        flat_images = images.reshape(-1, *images.shape[2:])
        heatmaps, _, raw_confidences, _ = algebraic.backbone(flat_images)
        keypoints_2d, _ = op.integrate_tensor_2d(
            heatmaps * algebraic.heatmap_multiplier, algebraic.heatmap_softmax
        )
        heatmap_height, heatmap_width = heatmaps.shape[-2:]
        keypoints_2d[..., 0] *= images.shape[-1] / heatmap_width
        keypoints_2d[..., 1] *= images.shape[-2] / heatmap_height
        keypoints_2d = keypoints_2d.reshape(batch_size, n_views, 17, 2)
        raw_confidences = raw_confidences.reshape(batch_size, n_views, 17)

        for views, combos in combinations.items():
            for combo in combos:
                selected = torch.as_tensor(combo, device=device)
                subset_conf = raw_confidences.index_select(1, selected)
                subset_conf = subset_conf / subset_conf.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-12) + 1e-5
                algebraic_pose = multiview.triangulate_batch_of_points(
                    projections.index_select(1, selected),
                    keypoints_2d.index_select(1, selected),
                    subset_conf,
                )
                vol_batch = {
                    "cameras": cameras_for_subset(batch, combo),
                    "pred_keypoints_3d": algebraic_pose.detach().cpu().numpy(),
                }
                (
                    prediction_lt,
                    _,
                    probability_volumes,
                    _,
                    _,
                    coordinate_volumes,
                    _,
                ) = volumetric(
                    images.index_select(1, selected),
                    projections.index_select(1, selected),
                    vol_batch,
                )
                if args.predictions_output:
                    flat_probability = probability_volumes.flatten(2)
                    flat_coordinates = coordinate_volumes.flatten(1, 3)
                    # The official softmax integral returns a normalized 64^3
                    # posterior for every joint.  E[||X||^2]-||E[X]||^2 is
                    # computed without materializing a joint x voxel x xyz
                    # tensor, and therefore adds only scalar uncertainty data
                    # to the exported predictions.
                    expected_square_norm = torch.einsum(
                        "bjv,bv->bj",
                        flat_probability,
                        flat_coordinates.square().sum(-1),
                    )
                    variance_lt = (
                        expected_square_norm - prediction_lt.square().sum(-1)
                    ).clamp_min(0.0)
                    entropy_lt = -(
                        flat_probability
                        * flat_probability.clamp_min(1e-12).log()
                    ).sum(-1)
                    peak_lt = flat_probability.max(-1).values
                for mapping_name, mapping in mappings.items():
                    prediction = prediction_lt[:, mapping]
                    if args.predictions_output:
                        prediction_batches[(views, combo, mapping_name)].append(
                            prediction.cpu().numpy()
                        )
                        uncertainty_batches[
                            (views, combo, mapping_name, "variance_mm2")
                        ].append(variance_lt[:, mapping].cpu().numpy())
                        uncertainty_batches[
                            (views, combo, mapping_name, "entropy")
                        ].append(entropy_lt[:, mapping].cpu().numpy())
                        uncertainty_batches[
                            (views, combo, mapping_name, "peak_probability")
                        ].append(peak_lt[:, mapping].cpu().numpy())
                    absolute = torch.linalg.vector_norm(
                        prediction - targets, dim=-1
                    ).mean(-1)
                    relative = torch.linalg.vector_norm(
                        (prediction - prediction[:, :1])
                        - (targets - targets[:, :1]), dim=-1
                    ).mean(-1)
                    errors[(views, combo, mapping_name, "absolute")].append(
                        absolute.cpu().numpy()
                    )
                    errors[(views, combo, mapping_name, "relative")].append(
                        relative.cpu().numpy()
                    )
        actions_all.append(batch["action"].numpy())
        if args.predictions_output:
            target_batches.append(targets.cpu().numpy())
        print(
            f"processed={min((batch_index + 1) * args.batch_size, len(dataset))}/"
            f"{len(dataset)}",
            flush=True,
        )

    actions = np.concatenate(actions_all)
    output = {
        "protocol": {
            "model": "official ICCV 2019 Volumetric Learnable Triangulation",
            "vol_checkpoint": os.path.abspath(args.vol_checkpoint),
            "alg_checkpoint_for_subset_pelvis": os.path.abspath(args.alg_checkpoint),
            "strict_load": True,
            "vol_state_tensors": vol_tensors,
            "alg_state_tensors": alg_tensors,
            "input": "RUMPL H36M annotation box; official 384x384 transform; undistorted",
            "pelvis_initialization": "official Algebraic LT, current camera subset only",
            "groups": len(dataset),
            "group_indices_json": (
                os.path.abspath(args.group_indices_json)
                if args.group_indices_json else None
            ),
        },
        "results": {},
    }
    for mapping_name in mappings:
        output["results"][mapping_name] = {}
        for views, combos in combinations.items():
            output["results"][mapping_name][f"V{views}"] = {}
            for metric in ("absolute", "relative"):
                combo_errors = [
                    np.concatenate(errors[(views, combo, mapping_name, metric)])
                    for combo in combos
                ]
                summary = summarize(
                    np.concatenate(combo_errors), np.tile(actions, len(combos))
                )
                summary["per_combination"] = {
                    "-".join(str(index + 1) for index in combo): summarize(err, actions)
                    for combo, err in zip(combos, combo_errors)
                }
                output["results"][mapping_name][f"V{views}"][metric] = summary
                print(
                    f"{mapping_name} V{views} {metric}: "
                    f"action_equal={summary['action_equal_mm']:.3f} "
                    f"frame={summary['frame_weighted_mm']:.3f}", flush=True,
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)

    if args.predictions_output:
        arrays = {
            "targets": np.concatenate(target_batches),
            "actions": actions,
        }
        for (views, combo, mapping_name), batches in prediction_batches.items():
            combo_name = "_".join(str(index + 1) for index in combo)
            arrays[f"{mapping_name}_V{views}_{combo_name}"] = np.concatenate(
                batches
            )
        for (
            views, combo, mapping_name, statistic
        ), batches in uncertainty_batches.items():
            combo_name = "_".join(str(index + 1) for index in combo)
            arrays[
                f"{mapping_name}_{statistic}_V{views}_{combo_name}"
            ] = np.concatenate(batches)
        predictions_path = Path(args.predictions_output)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_predictions = predictions_path.with_suffix(
            predictions_path.suffix + ".tmp"
        )
        with open(temporary_predictions, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary_predictions, predictions_path)


if __name__ == "__main__":
    main()
