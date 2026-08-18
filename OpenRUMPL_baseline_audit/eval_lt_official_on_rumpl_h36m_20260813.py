#!/usr/bin/env python3
"""Controlled evaluation of the official ICCV'19 Algebraic LT checkpoint.

The released network and weights are kept intact.  This adapter only presents
our synchronized H36M validation images with the paper's image transform:
undistort, annotation-box crop, 384x384 resize, and ImageNet normalization.
All camera subsets are evaluated from the same frozen heatmaps/confidences.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

LT_ROOT = Path("/home/lixiaob/cjy/reference/learnable-triangulation-official")
sys.path.insert(0, str(LT_ROOT))

from mvn.models.triangulation import AlgebraicTriangulationNet  # noqa: E402
from mvn.utils import cfg, multiview, op  # noqa: E402
from mvn.utils.img import crop_image, normalize_image, resize_image  # noqa: E402
from mvn.utils.multiview import Camera  # noqa: E402


ACTION_NAMES = {
    2: "Direction", 3: "Discuss", 4: "Eating", 5: "Greet", 6: "Phone",
    7: "Photo", 8: "Pose", 9: "Purchase", 10: "Sitting",
    11: "SittingDown", 12: "Smoke", 13: "Wait", 14: "WalkDog",
    15: "Walk", 16: "WalkTwo",
}

# LT label source joints:
# (3,2,1,6,7,8,0,12,13,15,27,26,25,17,18,19,14).
# RUMPL source joints:
# (0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27).
LT_TO_RUMPL = np.asarray(
    [6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10],
    dtype=np.int64,
)
RUMPL_TO_LT = np.argsort(LT_TO_RUMPL)
# Our H36M-Toolbox PKL has the known lower-body semantic swap relative to the
# LT preprocessing.  An exhaustive 17x17 heatmap/GT pixel audit finds the
# released prediction channels match target LT channels exactly as below;
# channels 6..16 remain identity.  This is a label conversion, not a learned
# or fitted coordinate correction.
PRED_LT_TO_TARGET_LT = np.asarray(
    [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    dtype=np.int64,
)
# Source prediction channel for each destination RUMPL joint.
RUMPL_FROM_PRED_LT = PRED_LT_TO_TARGET_LT[LT_TO_RUMPL]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(
        LT_ROOT / "experiments/human36m/eval/human36m_alg.yaml"
    ))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--predictions-output",
        help="Optionally export every subset prediction, targets and actions.",
    )
    parser.add_argument(
        "--export-rumpl-pkl",
        help="Optionally export LT 2D coordinates/raw confidences in RUMPL order.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument(
        "--group-indices-json",
        help="Select grouping indices in the exact order of a RUMPL manifest.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--undistort", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--uniform-confidences",
        action="store_true",
        help="Ablate LT's learned view/joint confidences while keeping its 2D points.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Run the backbone and export observations without triangulation metrics.",
    )
    return parser.parse_args()


def distortion_from_record(record: dict) -> np.ndarray:
    camera = record["camera"]
    radial = np.asarray(camera.get("k", np.zeros(3))).reshape(-1)
    tangential = np.asarray(camera.get("p", np.zeros(2))).reshape(-1)
    return np.asarray([
        radial[0], radial[1], tangential[0], tangential[1], radial[2]
    ], dtype=np.float64)


def group_records(records: list[dict]) -> list[tuple[tuple[int, ...], list[dict]]]:
    grouped: OrderedDict[tuple[int, ...], dict[int, dict]] = OrderedDict()
    for record in records:
        key = (
            int(record["subject"]), int(record["action"]),
            int(record["subaction"]), int(record["image_id"]),
        )
        grouped.setdefault(key, {})[int(record["camera_id"])] = record
    return [
        (key, [by_camera[index] for index in range(4)])
        for key, by_camera in grouped.items()
        if set(by_camera) == {0, 1, 2, 3}
    ]


class RUMPLH36MImages(Dataset):
    def __init__(
        self, pkl_path: str, image_root: str, limit_groups: int, undistort: bool,
        group_indices_json: str | None = None,
    ) -> None:
        with open(pkl_path, "rb") as handle:
            records = pickle.load(handle)
        self.groups = group_records(records)
        if group_indices_json:
            with open(group_indices_json, "r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            selected = []
            for position, entry in enumerate(manifest["groups"]):
                index = int(entry["grouping_index"])
                key, group = self.groups[index]
                images = [record["image"] for record in group]
                if images != entry["images"]:
                    raise ValueError(
                        f"manifest/database alignment failed at group {position}"
                    )
                selected.append((key, group))
            self.groups = selected
        if limit_groups:
            self.groups = self.groups[:limit_groups]
        self.image_root = Path(image_root)
        self.undistort = undistort

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        cv2.setNumThreads(0)
        key, records = self.groups[index]
        images, projections = [], []
        camera_rotations, camera_translations = [], []
        camera_intrinsics, camera_distortions = [], []
        for record in records:
            path = self.image_root / record["image"]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            camera_data = record["camera"]
            intrinsic = np.asarray(camera_data["K"], dtype=np.float64)
            if self.undistort:
                image = cv2.undistort(
                    image, intrinsic, distortion_from_record(record), None, intrinsic
                )

            # The official label file stores int16 boxes.  Mirror its PIL crop
            # semantics instead of passing fractional coordinates.
            bbox = tuple(int(round(float(value))) for value in record["box"])
            image = crop_image(image, bbox)
            before_resize = image.shape[:2]
            if min(before_resize) <= 0:
                raise ValueError(f"empty bbox {bbox} for {path}")
            image = resize_image(image, (384, 384))
            image = normalize_image(image).astype(np.float32)

            camera = Camera(
                np.asarray(camera_data["R"], dtype=np.float64),
                np.asarray(camera_data["t"], dtype=np.float64),
                intrinsic,
                distortion_from_record(record),
            )
            camera.update_after_crop(bbox)
            camera.update_after_resize(before_resize, (384, 384))
            images.append(np.transpose(image, (2, 0, 1)))
            projections.append(camera.projection.astype(np.float32))
            camera_rotations.append(camera.R.astype(np.float64))
            camera_translations.append(camera.t.reshape(3).astype(np.float64))
            camera_intrinsics.append(camera.K.astype(np.float64))
            camera_distortions.append(camera.dist.astype(np.float64))

        target_2d_lt = []
        for record in records:
            pixels = np.asarray(record["joints_2d"], dtype=np.float64)
            intrinsic = np.asarray(record["camera"]["K"], dtype=np.float64)
            if self.undistort:
                pixels = cv2.undistortPoints(
                    pixels.reshape(-1, 1, 2), intrinsic,
                    distortion_from_record(record), P=intrinsic,
                ).reshape(-1, 2)
            bbox = tuple(int(round(float(value))) for value in record["box"])
            crop_width = bbox[2] - bbox[0]
            crop_height = bbox[3] - bbox[1]
            pixels[:, 0] = (pixels[:, 0] - bbox[0]) * 384.0 / crop_width
            pixels[:, 1] = (pixels[:, 1] - bbox[1]) * 384.0 / crop_height
            target_lt_order = pixels[RUMPL_TO_LT]
            target_2d_lt.append(target_lt_order[PRED_LT_TO_TARGET_LT])

        return {
            "images": np.stack(images).astype(np.float32),
            "projections": np.stack(projections).astype(np.float32),
            "camera_R": np.stack(camera_rotations),
            "camera_t": np.stack(camera_translations),
            "camera_K": np.stack(camera_intrinsics),
            "camera_dist": np.stack(camera_distortions),
            "target": np.asarray(records[0]["joints_3d"], dtype=np.float32),
            "target_2d_lt": np.stack(target_2d_lt).astype(np.float32),
            "action": int(key[1]),
            "key": np.asarray(key, dtype=np.int64),
            "bboxes": np.asarray([
                [int(round(float(value))) for value in record["box"]]
                for record in records
            ], dtype=np.float32),
        }


def summarize(errors: np.ndarray, actions: np.ndarray) -> dict:
    per_action = {
        ACTION_NAMES[action]: float(errors[actions == action].mean())
        for action in sorted(ACTION_NAMES)
        if np.any(actions == action)
    }
    return {
        "records": int(len(errors)),
        "frame_weighted_mm": float(errors.mean()),
        "action_equal_mm": float(np.mean(list(per_action.values()))),
        "per_action_mm": per_action,
    }


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    device = torch.device(args.device)

    config = cfg.load_config(args.config)
    # The full released checkpoint already contains the backbone.
    config.model.backbone.init_weights = False
    model = AlgebraicTriangulationNet(config, device=device).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    # The converted MMPose H36M checkpoint is a standard training checkpoint
    # with a top-level ``state_dict`` and names the deconvolution head
    # ``head.*``.  The upstream LT implementation places that same head under
    # ``backbone.*`` and has an optional learned-confidence branch.  Normalize
    # this packaging here; the 2-D pose weights remain unchanged and the
    # optional confidence branch is intentionally left at its upstream
    # initialization when it is absent from the released pose checkpoint.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    if any(key.startswith("head.") for key in state):
        state = {
            ("backbone." + key[5:] if key.startswith("head.") else key): value
            for key, value in state.items()
        }
    # The public LT H36M backbone file is not a whole triangulation model.
    # Accept both that official backbone checkpoint and a complete LT model
    # checkpoint, while keeping strict loading for the selected format.  The
    # released backbone has 33 final heatmap channels; LT's H36M config uses
    # the first 17, exactly as the upstream loader does.
    if "conv1.weight" in state and "backbone.conv1.weight" not in state:
        backbone_state = dict(state)
        if backbone_state["final_layer.weight"].shape[0] != 17:
            backbone_state["final_layer.weight"] = (
                backbone_state["final_layer.weight"][:17].clone()
            )
            backbone_state["final_layer.bias"] = (
                backbone_state["final_layer.bias"][:17].clone()
            )
        incompatible = model.backbone.load_state_dict(backbone_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            key for key in incompatible.missing_keys
            if not key.startswith("alg_confidences.")
        ]
        if unexpected or missing:
            raise RuntimeError(
                "unexpected LT backbone checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        incompatible = None
    else:
        incompatible = model.load_state_dict(state, strict=False)
        missing = [
            key for key in incompatible.missing_keys
            if not key.startswith("backbone.alg_confidences.")
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "unexpected LT checkpoint mismatch: "
                f"missing={missing}, unexpected={incompatible.unexpected_keys}"
            )
        incompatible = None
    if incompatible is not None and (
        incompatible.missing_keys or incompatible.unexpected_keys
    ):
        raise RuntimeError(incompatible)

    dataset = RUMPLH36MImages(
        args.pkl, args.image_root, args.limit_groups, args.undistort,
        args.group_indices_json,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )

    combinations = {
        views: list(itertools.combinations(range(4), views))
        for views in (2, 3, 4)
    }
    errors = {
        (views, combo, metric): []
        for views, combos in combinations.items()
        for combo in combos
        for metric in ("absolute", "relative")
    }
    predictions = {
        (views, combo): []
        for views, combos in combinations.items()
        for combo in combos
    }
    targets_all = []
    actions_all, confidences_all, pixel_errors_all = [], [], []
    exported_observations: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    pixel_pair_sums = np.zeros((17, 17), dtype=np.float64)
    pixel_pair_count = 0

    for batch_index, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        projections = batch["projections"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        batch_size, n_views = images.shape[:2]

        flat_images = images.reshape(-1, *images.shape[2:])
        heatmaps, _, confidences, _ = model.backbone(flat_images)
        keypoints_2d, _ = op.integrate_tensor_2d(
            heatmaps * model.heatmap_multiplier, model.heatmap_softmax
        )
        heatmap_height, heatmap_width = heatmaps.shape[-2:]
        keypoints_2d[..., 0] *= images.shape[-1] / heatmap_width
        keypoints_2d[..., 1] *= images.shape[-2] / heatmap_height
        keypoints_2d = keypoints_2d.reshape(batch_size, n_views, 17, 2)
        raw_confidences = confidences.reshape(batch_size, n_views, 17)
        if args.uniform_confidences:
            confidences = torch.full_like(raw_confidences, 1.0 / n_views)
        else:
            confidences = raw_confidences / raw_confidences.sum(dim=1, keepdim=True)
        confidences = confidences + 1e-5
        confidences_all.append(confidences.detach().cpu().numpy())
        target_2d_lt = batch["target_2d_lt"].to(device, non_blocking=True)
        pixel_errors_all.append(
            torch.linalg.vector_norm(keypoints_2d - target_2d_lt, dim=-1)
            .cpu().numpy()
        )
        pair_error = torch.linalg.vector_norm(
            keypoints_2d[:, :, :, None, :] - target_2d_lt[:, :, None, :, :],
            dim=-1,
        )
        pixel_pair_sums += pair_error.sum(dim=(0, 1)).cpu().numpy()
        pixel_pair_count += batch_size * n_views
        actions_all.append(batch["action"].numpy())
        if args.predictions_output:
            targets_all.append(targets.cpu().numpy())

        if args.export_rumpl_pkl:
            rumpl_pixels = keypoints_2d[:, :, RUMPL_FROM_PRED_LT].cpu().numpy()
            rumpl_conf = raw_confidences[:, :, RUMPL_FROM_PRED_LT].cpu().numpy()
            bboxes = batch["bboxes"].numpy()
            keys = batch["key"].numpy()
            for row in range(batch_size):
                for view in range(n_views):
                    left, upper, right, lower = bboxes[row, view]
                    original = rumpl_pixels[row, view].copy()
                    original[:, 0] = original[:, 0] * (right - left) / 384.0 + left
                    original[:, 1] = original[:, 1] * (lower - upper) / 384.0 + upper
                    record_key = tuple(int(value) for value in keys[row]) + (view,)
                    exported_observations[record_key] = (
                        original.astype(np.float32),
                        rumpl_conf[row, view, :, None].astype(np.float32),
                    )

        if not args.export_only:
            for views, combos in combinations.items():
                for combo in combos:
                    selected = torch.as_tensor(combo, device=device)
                    prediction_lt = multiview.triangulate_batch_of_points(
                        projections.index_select(1, selected),
                        keypoints_2d.index_select(1, selected),
                        confidences.index_select(1, selected),
                    )
                    prediction = prediction_lt[:, RUMPL_FROM_PRED_LT]
                    if args.predictions_output:
                        predictions[(views, combo)].append(prediction.cpu().numpy())
                    absolute = torch.linalg.vector_norm(prediction - targets, dim=-1).mean(-1)
                    pred_relative = prediction - prediction[:, :1]
                    target_relative = targets - targets[:, :1]
                    relative = torch.linalg.vector_norm(
                        pred_relative - target_relative, dim=-1
                    ).mean(-1)
                    errors[(views, combo, "absolute")].append(absolute.cpu().numpy())
                    errors[(views, combo, "relative")].append(relative.cpu().numpy())

        if batch_index % 10 == 0:
            print(
                f"processed={min((batch_index + 1) * args.batch_size, len(dataset))}/"
                f"{len(dataset)}",
                flush=True,
            )

    actions = np.concatenate(actions_all)
    output = {
        "protocol": {
            "model": "official ICCV 2019 Algebraic Learnable Triangulation",
            "checkpoint": os.path.abspath(args.checkpoint),
            "strict_load": True,
            "input": "RUMPL H36M annotation box; official 384x384 transform",
            "undistort": bool(args.undistort),
            "uniform_confidences": bool(args.uniform_confidences),
            "export_only": bool(args.export_only),
            "groups": len(dataset),
            "group_indices_json": args.group_indices_json,
            "joint_mapping_pred_lt_to_target_lt": PRED_LT_TO_TARGET_LT.tolist(),
            "joint_mapping_rumpl_from_pred_lt": RUMPL_FROM_PRED_LT.tolist(),
        },
        "results": {},
        "confidence": {},
    }
    confidence_array = np.concatenate(confidences_all)
    pixel_error_array = np.concatenate(pixel_errors_all)
    output["confidence"] = {
        "mean": float(confidence_array.mean()),
        "min": float(confidence_array.min()),
        "max": float(confidence_array.max()),
    }
    output["pixel_error_384"] = {
        "mean": float(pixel_error_array.mean()),
        "median": float(np.median(pixel_error_array)),
        "per_lt_joint_mean": pixel_error_array.mean(axis=(0, 1)).tolist(),
        "best_target_lt_channel_per_prediction": (
            pixel_pair_sums / pixel_pair_count
        ).argmin(axis=1).tolist(),
        "pairwise_mean": (pixel_pair_sums / pixel_pair_count).tolist(),
    }
    print(
        f"2D error on 384 crop: mean={pixel_error_array.mean():.3f} "
        f"median={np.median(pixel_error_array):.3f}",
        flush=True,
    )

    if not args.export_only:
        for views, combos in combinations.items():
            output["results"][f"V{views}"] = {}
            for metric in ("absolute", "relative"):
                combo_errors = [
                    np.concatenate(errors[(views, combo, metric)]) for combo in combos
                ]
                # Equal weight for every camera combination and every action.
                stacked_errors = np.concatenate(combo_errors)
                stacked_actions = np.tile(actions, len(combos))
                summary = summarize(stacked_errors, stacked_actions)
                summary["per_combination"] = {
                    "-".join(str(index + 1) for index in combo): summarize(err, actions)
                    for combo, err in zip(combos, combo_errors)
                }
                output["results"][f"V{views}"][metric] = summary
                print(
                    f"V{views} {metric}: action_equal={summary['action_equal_mm']:.3f} "
                    f"frame={summary['frame_weighted_mm']:.3f}",
                    flush=True,
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)

    if args.export_rumpl_pkl:
        if len(exported_observations) != len(dataset) * 4:
            raise RuntimeError(
                f"export observations={len(exported_observations)}, "
                f"expected={len(dataset) * 4}"
            )
        with open(args.pkl, "rb") as handle:
            source_records = pickle.load(handle)
        updated_records = []
        for record in source_records:
            key = (
                int(record["subject"]), int(record["action"]),
                int(record["subaction"]), int(record["image_id"]),
                int(record["camera_id"]),
            )
            if key not in exported_observations:
                continue
            updated = copy.deepcopy(record)
            updated["joints_2d"], updated["joints_2d_conf"] = exported_observations[key]
            updated["joints_2d_source"] = "official_lt_alg_undistorted_annotation_box"
            updated_records.append(updated)
        if len(updated_records) != len(exported_observations):
            raise RuntimeError(
                f"updated records={len(updated_records)}, "
                f"observations={len(exported_observations)}"
            )
        export_path = Path(args.export_rumpl_pkl)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_tmp = export_path.with_suffix(export_path.suffix + ".tmp")
        with export_tmp.open("wb") as handle:
            pickle.dump(updated_records, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(export_tmp, export_path)
        print(f"exported RUMPL PKL records={len(updated_records)} to {export_path}")

    if args.predictions_output:
        prediction_path = Path(args.predictions_output)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_predictions = prediction_path.with_suffix(
            prediction_path.suffix + ".tmp"
        )
        arrays = {
            f"prediction_V{views}_{'_'.join(str(index + 1) for index in combo)}":
            np.concatenate(values).astype(np.float32)
            for (views, combo), values in predictions.items()
        }
        arrays["targets"] = np.concatenate(targets_all).astype(np.float32)
        arrays["actions"] = actions.astype(np.int64)
        with open(temporary_predictions, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary_predictions, prediction_path)


if __name__ == "__main__":
    main()
