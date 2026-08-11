#!/usr/bin/env python3
"""Train H24's camera-generalizable H21 anchor-delta gate on real H36M."""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from anchor_delta_gate import (
    CONTEXT_FEATURE_NAMES,
    FEATURE_NAMES,
    AnchorDeltaGate,
    SignedResidualContextGate,
    anchor_context_gate_features,
    anchor_gate_features,
)
from eval_h23_rumpl_pose_query_anchor import (
    DIRECT_COCO,
    rumpl_anchor_h36m,
)
from eval_h36m_dense_epipolar_heatmaps import (
    DenseHeatmapStore,
    a1d_corrected_coco,
)
from dense_geometry_residual_fusion import DenseGeometryResidualFusion
from eval_h36m_sparse_epipolar_topk import (
    COCO_TO_H36M_DIRECT,
    DIRECT_COCO_JOINTS,
    build_four_view_groups,
    camera_parameters,
    coco_to_h36m,
    target_world_metres,
)
from iterative_pose_query_refiner import IterativePoseQueryRefiner
from pose_query_geometry import (
    heatmap_to_image,
    image_to_heatmap,
    project_world_points,
)


DIRECT_H36M = np.asarray(
    list(COCO_TO_H36M_DIRECT.values()), dtype=np.int64
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--rumpl-input-pkl", required=True)
    parser.add_argument("--dense-shards", nargs="+", required=True)
    parser.add_argument("--h21-checkpoint", required=True)
    parser.add_argument(
        "--anchor-delta-source",
        choices=("h21", "a1d"),
        default="h21",
        help=(
            "Which anchor refinement produces the residual to gate. 'h21' "
            "uses the iterative pose-query refiner (default); 'a1d' uses the "
            "AdaFuse-style dense cross-view heatmap fusion."
        ),
    )
    parser.add_argument(
        "--a1d-checkpoint",
        help="Required dense-fusion checkpoint when --anchor-delta-source a1d.",
    )
    parser.add_argument("--a1d-depth-min", type=float, default=1.0)
    parser.add_argument("--a1d-depth-max", type=float, default=10.0)
    parser.add_argument("--a1d-depth-samples", type=int, default=64)
    parser.add_argument(
        "--rumpl-prediction-dicts",
        nargs="+",
        help=(
            "Frozen H22 train-split prediction dictionaries. Must be paired "
            "with --group-manifests."
        ),
    )
    parser.add_argument(
        "--group-manifests",
        nargs="+",
        help="Exact record-index manifests paired with frozen predictions.",
    )
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--view-probabilities", type=float, nargs=3, default=(3, 1, 1)
    )
    parser.add_argument("--anchor-confidence-epsilon", type=float, default=0.05)
    parser.add_argument("--anchor-regularization", type=float, default=1e-4)
    parser.add_argument("--oracle-loss-weight", type=float, default=0.02)
    parser.add_argument("--prior-loss-weight", type=float, default=0.002)
    parser.add_argument(
        "--context-features",
        action="store_true",
        help=(
            "Use H26's invariant RUMPL-residual/refined-ray context in "
            "addition to the original H24 gate features."
        ),
    )
    parser.add_argument(
        "--initial-gate-checkpoint",
        help=(
            "Optional H24 gate used to initialize a context gate exactly; "
            "new context-feature weights begin at zero."
        ),
    )
    parser.add_argument(
        "--signed-residual-gate",
        action="store_true",
        help=(
            "Train H27's [-1,1] residual around a frozen H26 context gate. "
            "Requires --context-features and an H26 checkpoint."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def save_checkpoint(
    output: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    feature_names = (
        CONTEXT_FEATURE_NAMES if args.context_features else FEATURE_NAMES
    )
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "feature_names": feature_names,
        "delta_source": args.anchor_delta_source,
        "gate_type": (
            "signed_residual_context"
            if args.signed_residual_gate
            else "positive_sigmoid"
        ),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def initialize_context_gate_from_h24(
    model: AnchorDeltaGate, checkpoint: str, device: torch.device
) -> None:
    payload = torch.load(checkpoint, map_location=device)
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("context initialization requires an H24 gate")
    source = AnchorDeltaGate().to(device)
    source.load_state_dict(payload["model"])
    with torch.no_grad():
        model.joint_embedding.weight.copy_(
            source.joint_embedding.weight
        )
        destination = model.network[0]
        original = source.network[0]
        destination.weight.zero_()
        destination.weight[:, : len(FEATURE_NAMES)].copy_(
            original.weight[:, : len(FEATURE_NAMES)]
        )
        destination.weight[:, len(CONTEXT_FEATURE_NAMES) :].copy_(
            original.weight[:, len(FEATURE_NAMES) :]
        )
        destination.bias.copy_(original.bias)
        for index in (1, 3, 5):
            model.network[index].load_state_dict(
                source.network[index].state_dict()
            )


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
    groups = [
        group
        for group in build_four_view_groups(records)
        if all(index in store for index in group)
    ]
    if not groups:
        raise RuntimeError("no complete train groups")
    prediction_buckets: dict[int, list[tuple[list[int], np.ndarray, np.ndarray]]] = {}
    if bool(args.rumpl_prediction_dicts) != bool(args.group_manifests):
        raise ValueError(
            "provide both --rumpl-prediction-dicts and --group-manifests"
        )
    if args.rumpl_prediction_dicts:
        if len(args.rumpl_prediction_dicts) != len(args.group_manifests):
            raise ValueError("prediction/manifest counts differ")
        for prediction_path, manifest_path in zip(
            args.rumpl_prediction_dicts, args.group_manifests
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
            bucket = prediction_buckets.setdefault(n_views, [])
            for index, entry in enumerate(manifest_groups):
                indices = [int(value) for value in entry["record_indices"]]
                if len(indices) != n_views:
                    raise ValueError("inconsistent view count in manifest")
                bucket.append((indices, predictions[index], targets[index]))
        if set(prediction_buckets) != {2, 3, 4}:
            raise ValueError(
                f"expected frozen V2/V3/V4, got {prediction_buckets.keys()}"
            )

    h21_payload = torch.load(args.h21_checkpoint, map_location=device)
    h21 = IterativePoseQueryRefiner().to(device)
    h21.load_state_dict(h21_payload["model"])
    h21.eval()
    for parameter in h21.parameters():
        parameter.requires_grad_(False)

    a1d_model = None
    a1d_depths = None
    a1d_joint_ids = None
    if args.anchor_delta_source == "a1d":
        if not args.a1d_checkpoint:
            raise ValueError(
                "--anchor-delta-source a1d requires --a1d-checkpoint"
            )
        a1d_payload = torch.load(args.a1d_checkpoint, map_location=device)
        a1d_model = DenseGeometryResidualFusion().to(device)
        a1d_model.load_state_dict(a1d_payload["model"])
        a1d_model.eval()
        for parameter in a1d_model.parameters():
            parameter.requires_grad_(False)
        a1d_depths = torch.linspace(
            args.a1d_depth_min,
            args.a1d_depth_max,
            args.a1d_depth_samples,
            device=device,
        )
        a1d_joint_ids = torch.as_tensor(
            DIRECT_COCO_JOINTS, dtype=torch.long, device=device
        )

    feature_names = (
        CONTEXT_FEATURE_NAMES if args.context_features else FEATURE_NAMES
    )
    if args.signed_residual_gate:
        if not args.context_features or not args.initial_gate_checkpoint:
            raise ValueError(
                "signed residual gate requires context features and H26 init"
            )
        payload = torch.load(
            args.initial_gate_checkpoint, map_location=device
        )
        if (
            tuple(payload.get("feature_names", ()))
            != CONTEXT_FEATURE_NAMES
            or payload.get("gate_type", "positive_sigmoid")
            != "positive_sigmoid"
        ):
            raise ValueError("signed residual initialization requires H26")
        base_gate = AnchorDeltaGate(
            feature_dimension=len(CONTEXT_FEATURE_NAMES)
        ).to(device)
        base_gate.load_state_dict(payload["model"])
        gate_model = SignedResidualContextGate(
            base_gate=base_gate,
            feature_dimension=len(CONTEXT_FEATURE_NAMES),
        ).to(device)
    else:
        gate_model = AnchorDeltaGate(
            feature_dimension=len(feature_names)
        ).to(device)
    if args.initial_gate_checkpoint and not args.signed_residual_gate:
        if not args.context_features:
            payload = torch.load(
                args.initial_gate_checkpoint, map_location=device
            )
            if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
                raise ValueError("initial gate feature schema mismatch")
            gate_model.load_state_dict(payload["model"])
        else:
            initialize_context_gate_from_h24(
                gate_model, args.initial_gate_checkpoint, device
            )
    optimizer = torch.optim.AdamW(
        gate_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    joint_ids = torch.arange(17, dtype=torch.long, device=device)
    h21_joint_ids = torch.as_tensor(
        DIRECT_COCO, dtype=torch.long, device=device
    )
    started = time.time()
    log_path = output_dir / "train.jsonl"
    with log_path.open("w", encoding="utf-8") as log_handle:
        for step in range(1, args.steps + 1):
            n_views = random.choices(
                (2, 3, 4), weights=args.view_probabilities, k=1
            )[0]
            frozen_prediction = None
            frozen_target = None
            if prediction_buckets:
                indices, frozen_prediction, frozen_target = random.choice(
                    prediction_buckets[n_views]
                )
            else:
                group = random.choice(groups)
                combination = random.choice(
                    list(itertools.combinations(range(4), n_views))
                )
                indices = [group[index] for index in combination]
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
            if args.anchor_delta_source == "a1d":
                camera = [
                    camera_parameters(record) for record in group_records
                ]
                a1d_intrinsics = [item[0] for item in camera]
                a1d_rotations = [item[1] for item in camera]
                a1d_centers = np.stack([item[2] for item in camera])
                refined_coco = a1d_corrected_coco(
                    a1d_model,
                    a1d_depths,
                    a1d_joint_ids,
                    data["heatmaps"],
                    raw_coco,
                    data["input_center"],
                    data["input_scale"],
                    a1d_intrinsics,
                    a1d_rotations,
                    a1d_centers,
                    device,
                )
            else:
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
                with torch.no_grad():
                    refined_hm, _ = h21(
                        torch.as_tensor(
                            heatmaps_np, dtype=torch.float32, device=device
                        ),
                        torch.as_tensor(
                            query_hm_np, dtype=torch.float32, device=device
                        ),
                        torch.as_tensor(
                            detector_hm_np,
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.as_tensor(
                            direct_confidence,
                            dtype=torch.float32,
                            device=device,
                        ),
                        h21_joint_ids,
                    )
                refined_coco = raw_coco.copy()
                refined_coco[:, DIRECT_COCO] = heatmap_to_image(
                    refined_hm.cpu().numpy(),
                    data["input_center"],
                    data["input_scale"],
                    width,
                    height,
                )
            refined_h36m_xy = []
            for view in range(n_views):
                joints, _ = coco_to_h36m(
                    refined_coco[view], coco_confidence[view]
                )
                refined_h36m_xy.append(joints)
            refined_h36m_xy = np.stack(refined_h36m_xy)
            refined_anchor = rumpl_anchor_h36m(
                group_records,
                refined_h36m_xy,
                h36m_confidence,
                args.anchor_confidence_epsilon,
                args.anchor_regularization,
            )
            anchor_delta = refined_anchor - old_anchor
            target = (
                frozen_target
                if frozen_target is not None
                else target_world_metres(group_records[0])
            )
            base_prediction = (
                old_anchor
                if frozen_prediction is None
                else frozen_prediction
            )
            if args.context_features:
                features_np = anchor_context_gate_features(
                    group_records,
                    h36m_xy,
                    refined_h36m_xy,
                    h36m_confidence,
                    old_anchor,
                    anchor_delta,
                    base_prediction,
                )
            else:
                features_np = anchor_gate_features(
                    group_records,
                    h36m_xy,
                    h36m_confidence,
                    old_anchor,
                    anchor_delta,
                )

            features = torch.as_tensor(
                features_np, dtype=torch.float32, device=device
            )
            old_tensor = torch.as_tensor(
                old_anchor, dtype=torch.float32, device=device
            )
            delta_tensor = torch.as_tensor(
                anchor_delta, dtype=torch.float32, device=device
            )
            target_tensor = torch.as_tensor(
                target, dtype=torch.float32, device=device
            )
            gate = gate_model(features, joint_ids)
            base_tensor = torch.as_tensor(
                base_prediction, dtype=torch.float32, device=device
            )
            corrected = base_tensor + gate[:, None] * delta_tensor
            joint_error = torch.linalg.norm(
                corrected - target_tensor, dim=-1
            )
            base_error = torch.linalg.norm(
                base_tensor - target_tensor, dim=-1
            )
            numerator = (
                (target_tensor - base_tensor) * delta_tensor
            ).sum(dim=-1)
            denominator = delta_tensor.square().sum(dim=-1).clamp_min(1e-8)
            oracle_gate = (numerator / denominator).clamp(
                -1.0 if args.signed_residual_gate else 0.0,
                1.0,
            )
            impact_weight = (
                1.0
                + torch.linalg.norm(delta_tensor, dim=-1)
                .div(0.05)
                .clamp(max=4.0)
            )
            oracle_loss = (
                (gate - oracle_gate).square() * impact_weight
            ).mean()
            if args.signed_residual_gate:
                with torch.no_grad():
                    prior_gate = gate_model.base_gate(
                        features, joint_ids
                    )
                prior_loss = (gate - prior_gate).square().mean()
            else:
                prior_loss = (gate - 0.25).square().mean()
            loss = (
                joint_error.mean()
                + args.oracle_loss_weight * oracle_loss
                + args.prior_loss_weight * prior_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                gate_model.parameters(), 5.0
            )
            optimizer.step()

            if step % args.log_every == 0 or step == 1:
                row = {
                    "step": step,
                    "views": n_views,
                    "loss": float(loss.detach()),
                    "base_prediction_mpjpe_mm": float(
                        base_error.mean() * 1000.0
                    ),
                    "old_anchor_mpjpe_mm": float(
                        torch.linalg.norm(
                            old_tensor - target_tensor, dim=-1
                        ).mean()
                        * 1000.0
                    ),
                    "gated_anchor_mpjpe_mm": float(
                        joint_error.mean().detach() * 1000.0
                    ),
                    "mean_gate": float(gate.mean().detach()),
                    "mean_oracle_gate": float(oracle_gate.mean()),
                    "oracle_loss": float(oracle_loss.detach()),
                    "prior_loss": float(prior_loss.detach()),
                    "mean_anchor_delta_mm": float(
                        torch.linalg.norm(delta_tensor, dim=-1).mean()
                        * 1000.0
                    ),
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.time() - started,
                }
                print(json.dumps(row), flush=True)
                log_handle.write(json.dumps(row) + "\n")
                log_handle.flush()
            if step % args.save_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_step{step:06d}.pth",
                    gate_model,
                    optimizer,
                    step,
                    args,
                )
    save_checkpoint(
        output_dir / "final.pth",
        gate_model,
        optimizer,
        args.steps,
        args,
    )


if __name__ == "__main__":
    main()
