#!/usr/bin/env python3
"""Fine-tune H21 through H22's final 3D anchor-correction objective.

H21 was originally trained only against per-view 2D coordinates.  H25 keeps
the RUMPL/H22 prediction frozen, differentiates through calibrated
COCO-to-H36M conversion and ray intersection, and minimizes:

    frozen_H22_prediction + learned_gate * (refined_anchor - old_anchor)

against real-H36M 3D ground truth.  Camera IDs and absolute pose/camera
coordinates are never given to either learned module.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from anchor_delta_gate import (
    FEATURE_NAMES,
    AnchorDeltaGate,
    anchor_gate_features,
)
from differentiable_h36m_geometry import (
    coco_to_h36m_torch,
    heatmap_to_image_torch,
    weighted_ray_anchor_torch,
)
from eval_h23_rumpl_pose_query_anchor import (
    DIRECT_COCO,
    rumpl_anchor_h36m,
)
from eval_h36m_dense_epipolar_heatmaps import DenseHeatmapStore
from eval_h36m_sparse_epipolar_topk import COCO_TO_H36M_DIRECT
from iterative_pose_query_refiner import IterativePoseQueryRefiner
from pose_query_geometry import image_to_heatmap, project_world_points


DIRECT_H36M = np.asarray(
    list(COCO_TO_H36M_DIRECT.values()), dtype=np.int64
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--rumpl-input-pkl", required=True)
    parser.add_argument("--dense-shards", nargs="+", required=True)
    parser.add_argument("--h21-checkpoint", required=True)
    parser.add_argument("--gate-checkpoint", required=True)
    parser.add_argument(
        "--rumpl-prediction-dicts", nargs="+", required=True
    )
    parser.add_argument("--group-manifests", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--h21-learning-rate", type=float, default=3e-5)
    parser.add_argument("--gate-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--view-probabilities", type=float, nargs=3, default=(3, 1, 1)
    )
    parser.add_argument("--two-d-loss-weight", type=float, default=0.002)
    parser.add_argument("--delta-penalty-weight", type=float, default=0.001)
    parser.add_argument("--oracle-loss-weight", type=float, default=0.01)
    parser.add_argument("--gate-drift-weight", type=float, default=0.01)
    parser.add_argument("--hard-case-pixels", type=float, default=4.0)
    parser.add_argument("--maximum-hard-weight", type=float, default=5.0)
    parser.add_argument("--anchor-confidence-epsilon", type=float, default=0.05)
    parser.add_argument("--anchor-regularization", type=float, default=1e-4)
    parser.add_argument("--freeze-gate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def atomic_torch_save(payload: dict, output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def save_models(
    output_dir: Path,
    suffix: str,
    h21: IterativePoseQueryRefiner,
    gate: AnchorDeltaGate,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    common = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "h25_objective": "frozen_H22_plus_gated_differentiable_anchor_delta",
    }
    atomic_torch_save(
        {
            **common,
            "model": h21.state_dict(),
            "direct_coco_joints": DIRECT_COCO.tolist(),
            "direct_h36m_joints": DIRECT_H36M.tolist(),
        },
        output_dir / f"h21_{suffix}.pth",
    )
    atomic_torch_save(
        {
            **common,
            "model": gate.state_dict(),
            "feature_names": FEATURE_NAMES,
        },
        output_dir / f"gate_{suffix}.pth",
    )


def load_prediction_buckets(
    prediction_paths: list[str], manifest_paths: list[str]
) -> dict[int, list[tuple[list[int], np.ndarray, np.ndarray]]]:
    if len(prediction_paths) != len(manifest_paths):
        raise ValueError("prediction/manifest counts differ")
    buckets: dict[
        int, list[tuple[list[int], np.ndarray, np.ndarray]]
    ] = {}
    for prediction_path, manifest_path in zip(
        prediction_paths, manifest_paths
    ):
        with open(prediction_path, "rb") as handle:
            frozen = pickle.load(handle)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        predictions = np.asarray(frozen["pred"], dtype=np.float64)
        targets = np.asarray(frozen["gt"], dtype=np.float64)
        manifest_groups = manifest["groups"]
        if len(predictions) != len(manifest_groups):
            raise ValueError(
                f"{prediction_path}: predictions={len(predictions)}, "
                f"manifest groups={len(manifest_groups)}"
            )
        n_views = len(manifest_groups[0]["record_indices"])
        bucket = buckets.setdefault(n_views, [])
        for index, entry in enumerate(manifest_groups):
            indices = [int(value) for value in entry["record_indices"]]
            if len(indices) != n_views:
                raise ValueError("inconsistent view count in manifest")
            bucket.append((indices, predictions[index], targets[index]))
    if set(buckets) != {2, 3, 4}:
        raise ValueError(f"expected frozen V2/V3/V4, got {buckets.keys()}")
    return buckets


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    with open(args.rumpl_input_pkl, "rb") as handle:
        rumpl_records = pickle.load(handle)
    if len(records) != len(rumpl_records):
        raise ValueError(
            f"base records={len(records)}, RUMPL records={len(rumpl_records)}"
        )
    check_indices = np.linspace(
        0, len(records) - 1, min(len(records), 1000), dtype=np.int64
    )
    for index in check_indices:
        if records[index]["image"] != rumpl_records[index]["image"]:
            raise ValueError(f"record alignment failed at {index}")

    store = DenseHeatmapStore(args.dense_shards)
    prediction_buckets = load_prediction_buckets(
        args.rumpl_prediction_dicts, args.group_manifests
    )

    h21_payload = torch.load(args.h21_checkpoint, map_location=device)
    h21 = IterativePoseQueryRefiner().to(device)
    h21.load_state_dict(h21_payload["model"])
    h21.train()
    gate_payload = torch.load(args.gate_checkpoint, map_location=device)
    if tuple(gate_payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("gate feature schema does not match H25")
    gate = AnchorDeltaGate().to(device)
    gate.load_state_dict(gate_payload["model"])
    gate.train(not args.freeze_gate)
    initial_gate = AnchorDeltaGate().to(device)
    initial_gate.load_state_dict(gate_payload["model"])
    initial_gate.eval()
    for parameter in initial_gate.parameters():
        parameter.requires_grad_(False)
    if args.freeze_gate:
        for parameter in gate.parameters():
            parameter.requires_grad_(False)

    parameter_groups = [
        {
            "params": h21.parameters(),
            "lr": args.h21_learning_rate,
        }
    ]
    if not args.freeze_gate:
        parameter_groups.append(
            {
                "params": gate.parameters(),
                "lr": args.gate_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=args.weight_decay
    )
    h21_joint_ids = torch.as_tensor(
        DIRECT_COCO, dtype=torch.long, device=device
    )
    gate_joint_ids = torch.arange(17, dtype=torch.long, device=device)

    started = time.time()
    log_path = output_dir / "train.jsonl"
    with log_path.open("w", encoding="utf-8") as log_handle:
        for step in range(1, args.steps + 1):
            n_views = random.choices(
                (2, 3, 4), weights=args.view_probabilities, k=1
            )[0]
            indices, frozen_prediction, frozen_target = random.choice(
                prediction_buckets[n_views]
            )
            group_records = [records[index] for index in indices]
            group_rumpl = [rumpl_records[index] for index in indices]
            data = store.get(indices)
            heatmaps_np = data["heatmaps"][:, DIRECT_COCO].astype(
                np.float32
            )
            _, _, height, width = heatmaps_np.shape
            raw_coco = data["decoded_keypoints"].astype(np.float64)
            coco_confidence = data["decoded_scores"].astype(np.float64)
            direct_confidence = coco_confidence[:, DIRECT_COCO]
            detector_hm_np = image_to_heatmap(
                raw_coco[:, DIRECT_COCO],
                data["input_center"],
                data["input_scale"],
                width,
                height,
            )
            h36m_xy = np.stack(
                [
                    np.asarray(item["joints_2d"], dtype=np.float64)
                    for item in group_rumpl
                ]
            )
            h36m_confidence = np.stack(
                [
                    np.asarray(
                        item["joints_2d_conf"], dtype=np.float64
                    ).reshape(17)
                    for item in group_rumpl
                ]
            )
            old_anchor = rumpl_anchor_h36m(
                group_records,
                h36m_xy,
                h36m_confidence,
                args.anchor_confidence_epsilon,
                args.anchor_regularization,
            )
            query_image = project_world_points(
                group_records, old_anchor[DIRECT_H36M]
            )
            query_hm_np = image_to_heatmap(
                query_image,
                data["input_center"],
                data["input_scale"],
                width,
                height,
            )
            target_image = np.stack(
                [
                    np.asarray(item["joints_2d"], dtype=np.float64)[
                        DIRECT_H36M
                    ]
                    for item in group_records
                ]
            )
            target_hm_np = image_to_heatmap(
                target_image,
                data["input_center"],
                data["input_scale"],
                width,
                height,
            )

            heatmaps = torch.as_tensor(
                heatmaps_np, dtype=torch.float32, device=device
            )
            detector_hm = torch.as_tensor(
                detector_hm_np, dtype=torch.float32, device=device
            )
            query_hm = torch.as_tensor(
                query_hm_np, dtype=torch.float32, device=device
            )
            confidence = torch.as_tensor(
                direct_confidence, dtype=torch.float32, device=device
            )
            target_hm = torch.as_tensor(
                target_hm_np, dtype=torch.float32, device=device
            )
            refined_hm, auxiliary = h21(
                heatmaps,
                query_hm,
                detector_hm,
                confidence,
                h21_joint_ids,
            )
            refined_image = heatmap_to_image_torch(
                refined_hm,
                data["input_center"],
                data["input_scale"],
                width,
                height,
            )
            refined_coco = torch.as_tensor(
                raw_coco, dtype=torch.float32, device=device
            ).clone()
            refined_coco[:, DIRECT_COCO] = refined_image
            refined_h36m_xy = coco_to_h36m_torch(refined_coco)
            refined_anchor = weighted_ray_anchor_torch(
                group_records,
                refined_h36m_xy,
                h36m_confidence,
                args.anchor_confidence_epsilon,
                args.anchor_regularization,
            )
            old_anchor_tensor = torch.as_tensor(
                old_anchor, dtype=torch.float64, device=device
            )
            anchor_delta = refined_anchor - old_anchor_tensor

            features_np = anchor_gate_features(
                group_records,
                h36m_xy,
                h36m_confidence,
                old_anchor,
                anchor_delta.detach().cpu().numpy(),
            )
            features = torch.as_tensor(
                features_np, dtype=torch.float32, device=device
            )
            gate_value = gate(features, gate_joint_ids)
            with torch.no_grad():
                initial_gate_value = initial_gate(
                    features, gate_joint_ids
                )
            base = torch.as_tensor(
                frozen_prediction, dtype=torch.float64, device=device
            )
            target = torch.as_tensor(
                frozen_target, dtype=torch.float64, device=device
            )
            corrected = (
                base + gate_value.double()[:, None] * anchor_delta
            )
            joint_error = torch.linalg.norm(corrected - target, dim=-1)
            base_error = torch.linalg.norm(base - target, dim=-1)
            three_d_loss = joint_error.mean().float()

            valid = (
                (target_hm[..., 0] >= 0)
                & (target_hm[..., 0] <= width - 1)
                & (target_hm[..., 1] >= 0)
                & (target_hm[..., 1] <= height - 1)
            )
            detector_error = torch.linalg.norm(
                detector_hm - target_hm, dim=-1
            )
            hard_weight = 1.0 + torch.clamp(
                detector_error / args.hard_case_pixels,
                max=args.maximum_hard_weight - 1.0,
            )
            coordinate_loss = functional.smooth_l1_loss(
                refined_hm, target_hm, reduction="none"
            ).mean(dim=-1)
            effective = valid.float() * hard_weight
            coordinate_loss = (
                coordinate_loss * effective
            ).sum() / effective.sum().clamp_min(1.0)
            delta_penalty = auxiliary["delta"].square().mean()

            numerator = (
                (target - base) * anchor_delta.detach()
            ).sum(dim=-1)
            denominator = (
                anchor_delta.detach().square().sum(dim=-1).clamp_min(1e-8)
            )
            oracle_gate = (numerator / denominator).clamp(0.0, 1.0).float()
            impact_weight = (
                1.0
                + torch.linalg.norm(anchor_delta.detach(), dim=-1)
                .div(0.05)
                .clamp(max=4.0)
            ).float()
            oracle_loss = (
                (gate_value - oracle_gate).square() * impact_weight
            ).mean()
            gate_drift_loss = (
                gate_value - initial_gate_value
            ).square().mean()

            loss = (
                three_d_loss
                + args.two_d_loss_weight * coordinate_loss
                + args.delta_penalty_weight * delta_penalty
                + args.oracle_loss_weight * oracle_loss
                + args.gate_drift_weight * gate_drift_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            h21_gradient_norm = torch.nn.utils.clip_grad_norm_(
                h21.parameters(), 5.0
            )
            if args.freeze_gate:
                gate_gradient_norm = 0.0
            else:
                gate_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    gate.parameters(), 5.0
                )
            optimizer.step()

            if step % args.log_every == 0 or step == 1:
                row = {
                    "step": step,
                    "views": n_views,
                    "loss": float(loss.detach()),
                    "three_d_loss_mm": float(
                        three_d_loss.detach() * 1000.0
                    ),
                    "base_prediction_mpjpe_mm": float(
                        base_error.mean() * 1000.0
                    ),
                    "coordinate_loss": float(coordinate_loss.detach()),
                    "delta_penalty": float(delta_penalty.detach()),
                    "oracle_loss": float(oracle_loss.detach()),
                    "gate_drift_loss": float(gate_drift_loss.detach()),
                    "mean_gate": float(gate_value.mean().detach()),
                    "mean_oracle_gate": float(oracle_gate.mean()),
                    "mean_anchor_delta_mm": float(
                        torch.linalg.norm(anchor_delta.detach(), dim=-1).mean()
                        * 1000.0
                    ),
                    "h21_gradient_norm": float(h21_gradient_norm),
                    "gate_gradient_norm": float(gate_gradient_norm),
                    "elapsed_seconds": time.time() - started,
                }
                print(json.dumps(row), flush=True)
                log_handle.write(json.dumps(row) + "\n")
                log_handle.flush()
            if step % args.save_every == 0:
                save_models(
                    output_dir,
                    f"step{step:06d}",
                    h21,
                    gate,
                    optimizer,
                    step,
                    args,
                )
    save_models(
        output_dir, "final", h21, gate, optimizer, args.steps, args
    )


if __name__ == "__main__":
    main()
