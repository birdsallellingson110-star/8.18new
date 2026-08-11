#!/usr/bin/env python3
"""Evaluate a strict H21-anchor replacement under a frozen H22 RUMPL model.

H22 predicts ``old_anchor + learned_RUMPL_residual``.  This diagnostic keeps
the trained RUMPL prediction and residual fixed, refines the frozen detector
heatmaps once with H21, recomputes the same confidence-weighted ray
intersection used by H22, and reports::

    H22_prediction + alpha * (refined_anchor - old_anchor)

Thus the only changed quantity is the geometric anchor.  No temporal feature,
camera-ID embedding, synthetic input, or additional RUMPL training is used.
The alpha sweep is diagnostic; a publication result must select/learn alpha on
the training split rather than choose it on validation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from collections import defaultdict
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
from eval_h36m_dense_epipolar_heatmaps import (
    DenseHeatmapStore,
    a1d_corrected_coco,
)
from dense_geometry_residual_fusion import DenseGeometryResidualFusion
from eval_h36m_sparse_epipolar_topk import (
    ACTION_NAMES,
    COCO_TO_H36M_DIRECT,
    DIRECT_COCO_JOINTS,
    KP_STAR,
    build_four_view_groups,
    camera_parameters,
    coco_to_h36m,
    pixels_to_rays,
    solve_ray_intersection,
    target_world_metres,
)
from iterative_pose_query_refiner import IterativePoseQueryRefiner
from pose_query_geometry import (
    heatmap_to_image,
    image_to_heatmap,
    project_world_points,
)


DIRECT_COCO = np.asarray(
    list(COCO_TO_H36M_DIRECT.keys()), dtype=np.int64
)
DIRECT_H36M = np.asarray(
    list(COCO_TO_H36M_DIRECT.values()), dtype=np.int64
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument(
        "--rumpl-input-pkl",
        required=True,
        help=(
            "Exact H36M-17 MMPose PKL consumed by H22 before the evaluator's "
            "lower-body swap."
        ),
    )
    parser.add_argument("--dense-shards", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--gate-checkpoint",
        help="Optional H24 train-split learned anchor-delta gate.",
    )
    parser.add_argument(
        "--a1d-gate-checkpoint",
        help=(
            "Optional A1D-source learned gate. When combined with an h21 "
            "--gate-checkpoint, both per-joint corrections are stacked on the "
            "frozen H22 prediction (H32 combined diagnostic)."
        ),
    )
    parser.add_argument(
        "--prediction-root",
        required=True,
        help="H22 eval root containing V2/V3/V4 prediction dictionaries.",
    )
    parser.add_argument("--views", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument(
        "--query-sources",
        nargs="+",
        choices=("old_anchor", "rumpl_prediction"),
        default=("old_anchor", "rumpl_prediction"),
    )
    parser.add_argument(
        "--anchor-delta-scales",
        type=float,
        nargs="+",
        default=(0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--anchor-confidence-epsilon", type=float, default=0.05)
    parser.add_argument("--anchor-regularization", type=float, default=1e-4)
    parser.add_argument(
        "--limit-per-action",
        type=int,
        default=0,
        help="Representative diagnostic subset; zero evaluates every record.",
    )
    parser.add_argument(
        "--report-oracle-diagnostics",
        action="store_true",
        help=(
            "Report explicitly validation-GT-forbidden per-joint scale "
            "ceilings.  These are diagnostics only, never paper results."
        ),
    )
    parser.add_argument(
        "--a1d-checkpoint",
        help=(
            "Optional A1D dense-heatmap geometry-residual fusion checkpoint. "
            "When set, an AdaFuse-style cross-view 2D correction produces a "
            "second triangulation anchor whose delta to the raw RUMPL anchor "
            "is applied to the frozen H22 prediction (H30 diagnostic)."
        ),
    )
    parser.add_argument("--a1d-depth-min", type=float, default=1.0)
    parser.add_argument("--a1d-depth-max", type=float, default=10.0)
    parser.add_argument("--a1d-depth-samples", type=int, default=64)
    parser.add_argument(
        "--a1d-delta-scales",
        type=float,
        nargs="+",
        default=(0.25, 0.5, 1.0),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def swap_lower_body(pose: np.ndarray) -> np.ndarray:
    swapped = pose.copy()
    swapped[..., 1:4, :] = pose[..., 4:7, :]
    swapped[..., 4:7, :] = pose[..., 1:4, :]
    return swapped


def rumpl_anchor_h36m(
    records: list[dict],
    h36m_xy: np.ndarray,
    h36m_confidence: np.ndarray,
    confidence_epsilon: float,
    regularization: float,
) -> np.ndarray:
    """Reproduce H20/H22's non-IRLS confidence-weighted ray anchor."""
    h36m_xy = np.asarray(h36m_xy, dtype=np.float64)
    h36m_confidence = np.asarray(
        h36m_confidence, dtype=np.float64
    ).reshape(len(records), 17)
    camera_data = [camera_parameters(record) for record in records]
    intrinsics = [item[0] for item in camera_data]
    rotations = [item[1] for item in camera_data]
    centers = np.stack([item[2] for item in camera_data])
    directions = np.stack(
        [
            pixels_to_rays(h36m_xy[view], intrinsics[view], rotations[view])
            for view in range(len(records))
        ]
    )
    return np.stack(
        [
            solve_ray_intersection(
                centers,
                directions[:, joint],
                h36m_confidence[:, joint] + confidence_epsilon,
                reg=regularization,
            )
            for joint in range(17)
        ]
    )


def find_prediction_dictionary(root: Path, n_views: int) -> Path:
    matches = sorted(
        (root / f"V{n_views}").glob("preds_gt_*_dict.pkl")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one V{n_views} prediction dictionary, got {matches}"
        )
    return matches[0]


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


def add_error(
    errors: dict[str, dict[str, list[float]]],
    kp_errors: dict[str, list[float]],
    method: str,
    action: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    joint_error = np.linalg.norm(prediction - target, axis=-1) * 1000.0
    errors[method][action].append(float(joint_error.mean()))
    kp_errors[method].append(float(joint_error[list(KP_STAR)].mean()))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    with open(args.rumpl_input_pkl, "rb") as handle:
        rumpl_input_records = pickle.load(handle)
    if len(rumpl_input_records) != len(records):
        raise ValueError(
            f"base records={len(records)} but RUMPL input records="
            f"{len(rumpl_input_records)}"
        )
    for index, (base_record, rumpl_record) in enumerate(
        zip(records, rumpl_input_records)
    ):
        if base_record["image"] != rumpl_record["image"]:
            raise ValueError(
                f"record {index} image mismatch: {base_record['image']} "
                f"!= {rumpl_record['image']}"
            )
    store = DenseHeatmapStore(args.dense_shards)
    groups = [
        group
        for group in build_four_view_groups(records)
        if all(index in store for index in group)
    ]
    payload = torch.load(args.checkpoint, map_location=device)
    model = IterativePoseQueryRefiner().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    def build_gate(checkpoint_path):
        payload = torch.load(checkpoint_path, map_location=device)
        delta_source = payload.get("delta_source", "h21")
        feature_names = tuple(payload.get("feature_names", ()))
        if feature_names not in (FEATURE_NAMES, CONTEXT_FEATURE_NAMES):
            raise ValueError("gate feature schema does not match evaluator")
        gate_type = payload.get("gate_type", "positive_sigmoid")
        if gate_type == "signed_residual_context":
            model = SignedResidualContextGate(
                feature_dimension=len(feature_names)
            ).to(device)
        elif gate_type == "positive_sigmoid":
            model = AnchorDeltaGate(
                feature_dimension=len(feature_names)
            ).to(device)
        else:
            raise ValueError(f"unsupported gate type: {gate_type}")
        model.load_state_dict(payload["model"])
        model.eval()
        return model, feature_names, delta_source

    gate_model = None
    gate_joint_ids = None
    gate_feature_names = None
    gate_delta_source = "h21"
    a1d_gate_model = None
    a1d_gate_feature_names = None
    if args.gate_checkpoint:
        gate_model, gate_feature_names, gate_delta_source = build_gate(
            args.gate_checkpoint
        )
        gate_joint_ids = torch.arange(17, dtype=torch.long, device=device)
    if args.a1d_gate_checkpoint:
        (
            a1d_gate_model,
            a1d_gate_feature_names,
            a1d_gate_delta_source,
        ) = build_gate(args.a1d_gate_checkpoint)
        if a1d_gate_delta_source != "a1d":
            raise ValueError("--a1d-gate-checkpoint must be an a1d-source gate")
        if a1d_gate_feature_names != CONTEXT_FEATURE_NAMES:
            raise ValueError("--a1d-gate-checkpoint requires context features")
        gate_joint_ids = torch.arange(17, dtype=torch.long, device=device)
    joint_ids = torch.as_tensor(
        DIRECT_COCO, dtype=torch.long, device=device
    )
    a1d_model = None
    a1d_depths = None
    a1d_joint_ids = None
    if args.a1d_checkpoint:
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
    result = {
        "input_pkl": args.input_pkl,
        "rumpl_input_pkl": args.rumpl_input_pkl,
        "checkpoint": args.checkpoint,
        "gate_checkpoint": args.gate_checkpoint,
        "gate_feature_names": gate_feature_names,
        "prediction_root": str(Path(args.prediction_root).resolve()),
        "query_sources": args.query_sources,
        "anchor_delta_scales": args.anchor_delta_scales,
        "diagnostic_alpha_warning": (
            "alpha is a validation diagnostic, not a selected final setting"
        ),
        "oracle_diagnostic_warning": (
            "validation-GT oracle methods are ceilings, not usable results"
            if args.report_oracle_diagnostics
            else None
        ),
        "complete_four_view_groups": len(groups),
        "results": {},
    }

    prediction_root = Path(args.prediction_root)
    for n_views in args.views:
        prediction_path = find_prediction_dictionary(
            prediction_root, n_views
        )
        with open(prediction_path, "rb") as handle:
            frozen = pickle.load(handle)
        frozen_prediction = np.asarray(
            frozen["pred"], dtype=np.float64
        )
        frozen_target = np.asarray(frozen["gt"], dtype=np.float64)
        work = [
            (group, combination)
            for group in groups
            for combination in itertools.combinations(range(4), n_views)
        ]
        if len(frozen_prediction) != len(work):
            raise ValueError(
                f"V{n_views}: H22 predictions={len(frozen_prediction)} "
                f"but grouped combinations={len(work)}"
            )
        indexed_work = list(enumerate(work))
        if args.limit_per_action:
            selected = []
            action_counts: dict[int, int] = defaultdict(int)
            for prediction_index, item in indexed_work:
                group, _ = item
                action_id = int(records[group[0]]["action"])
                if action_counts[action_id] >= args.limit_per_action:
                    continue
                action_counts[action_id] += 1
                selected.append((prediction_index, item))
            indexed_work = selected

        errors = defaultdict(lambda: defaultdict(list))
        kp_errors = defaultdict(list)
        anchor_delta_mm = defaultdict(list)
        learned_gate_values = defaultdict(list)
        oracle_scale_values = defaultdict(list)
        target_alignment_mm = []
        for position, (prediction_index, item) in enumerate(
            indexed_work, start=1
        ):
            group, combination = item
            indices = [group[index] for index in combination]
            group_records = [records[index] for index in indices]
            group_rumpl_inputs = [
                rumpl_input_records[index] for index in indices
            ]
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
            heatmaps = torch.as_tensor(
                heatmaps_np, dtype=torch.float32, device=device
            )
            detector_hm = torch.as_tensor(
                detector_hm_np, dtype=torch.float32, device=device
            )
            confidence = torch.as_tensor(
                direct_confidence, dtype=torch.float32, device=device
            )
            # The selected H22 protocol applies FLIP_LOWER_BODY_KP_TEST after
            # loading this MMPose PKL.  Reproduce that exact input here.
            rumpl_h36m_xy_canonical = np.stack(
                [
                    np.asarray(item["joints_2d"], dtype=np.float64)
                    for item in group_rumpl_inputs
                ]
            )
            rumpl_h36m_confidence_canonical = np.stack(
                [
                    np.asarray(
                        item["joints_2d_conf"], dtype=np.float64
                    ).reshape(17)
                    for item in group_rumpl_inputs
                ]
            )
            old_anchor_canonical = rumpl_anchor_h36m(
                group_records,
                rumpl_h36m_xy_canonical,
                rumpl_h36m_confidence_canonical,
                args.anchor_confidence_epsilon,
                args.anchor_regularization,
            )
            old_anchor = swap_lower_body(old_anchor_canonical[None])[0]
            rumpl_prediction = frozen_prediction[prediction_index]
            target = frozen_target[prediction_index]
            record_target = swap_lower_body(
                target_world_metres(group_records[0])[None]
            )[0]
            target_alignment_mm.append(
                float(np.max(np.abs(record_target - target)) * 1000.0)
            )
            action = ACTION_NAMES[int(group_records[0]["action"])]
            add_error(
                errors,
                kp_errors,
                "frozen_h22",
                action,
                rumpl_prediction,
                target,
            )
            add_error(
                errors,
                kp_errors,
                "old_anchor",
                action,
                old_anchor,
                target,
            )

            if a1d_model is not None:
                n_group_views = len(group_records)
                a1d_camera = [
                    camera_parameters(record) for record in group_records
                ]
                a1d_intrinsics = [item[0] for item in a1d_camera]
                a1d_rotations = [item[1] for item in a1d_camera]
                a1d_centers = np.stack([item[2] for item in a1d_camera])
                a1d_coco = a1d_corrected_coco(
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
                a1d_h36m_xy_canonical = np.stack(
                    [
                        coco_to_h36m(
                            a1d_coco[view], coco_confidence[view]
                        )[0]
                        for view in range(n_group_views)
                    ]
                )
                a1d_anchor_canonical = rumpl_anchor_h36m(
                    group_records,
                    a1d_h36m_xy_canonical,
                    rumpl_h36m_confidence_canonical,
                    args.anchor_confidence_epsilon,
                    args.anchor_regularization,
                )
                a1d_anchor = swap_lower_body(a1d_anchor_canonical[None])[0]
                a1d_delta_canonical = (
                    a1d_anchor_canonical - old_anchor_canonical
                )
                a1d_delta = a1d_anchor - old_anchor
                add_error(
                    errors,
                    kp_errors,
                    "a1d_anchor",
                    action,
                    a1d_anchor,
                    target,
                )
                for scale in args.a1d_delta_scales:
                    add_error(
                        errors,
                        kp_errors,
                        f"h22_plus_{scale:g}x_a1d_delta",
                        action,
                        rumpl_prediction + scale * a1d_delta,
                        target,
                    )
                if (
                    gate_model is not None
                    and gate_delta_source == "a1d"
                    and gate_feature_names == CONTEXT_FEATURE_NAMES
                ):
                    rumpl_prediction_canonical = swap_lower_body(
                        rumpl_prediction[None]
                    )[0]
                    a1d_gate_features_np = anchor_context_gate_features(
                        group_records,
                        rumpl_h36m_xy_canonical,
                        a1d_h36m_xy_canonical,
                        rumpl_h36m_confidence_canonical,
                        old_anchor_canonical,
                        a1d_delta_canonical,
                        rumpl_prediction_canonical,
                    )
                    with torch.no_grad():
                        a1d_gate_canonical = gate_model(
                            torch.as_tensor(
                                a1d_gate_features_np,
                                dtype=torch.float32,
                                device=device,
                            ),
                            gate_joint_ids,
                        ).cpu().numpy()
                    a1d_gate = swap_lower_body(
                        a1d_gate_canonical[None, :, None]
                    )[0, :, 0]
                    add_error(
                        errors,
                        kp_errors,
                        "h22_plus_a1d_learned_gate",
                        action,
                        rumpl_prediction + a1d_gate[:, None] * a1d_delta,
                        target,
                    )
                if args.report_oracle_diagnostics:
                    a1d_num = (
                        (target - rumpl_prediction) * a1d_delta
                    ).sum(axis=-1)
                    a1d_den = np.maximum(
                        np.square(a1d_delta).sum(axis=-1), 1e-12
                    )
                    a1d_raw_scale = a1d_num / a1d_den
                    for label, low, high in (
                        ("clip0to1", 0.0, 1.0),
                        ("clipMinus1to1", -1.0, 1.0),
                    ):
                        a1d_oracle_scale = np.clip(a1d_raw_scale, low, high)
                        add_error(
                            errors,
                            kp_errors,
                            f"a1d_oracle_per_joint_scale_{label}",
                            action,
                            rumpl_prediction
                            + a1d_oracle_scale[:, None] * a1d_delta,
                            target,
                        )

            for query_source in args.query_sources:
                if query_source == "old_anchor":
                    query_pose = old_anchor_canonical
                else:
                    # H21 was trained in the unswapped H36M convention used
                    # by the raw detector exporter.
                    query_pose = swap_lower_body(
                        rumpl_prediction[None]
                    )[0]
                query_image = project_world_points(
                    group_records, query_pose[DIRECT_H36M]
                )
                query_hm_np = image_to_heatmap(
                    query_image,
                    data["input_center"],
                    data["input_scale"],
                    width,
                    height,
                )
                query_hm = torch.as_tensor(
                    query_hm_np, dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    refined_hm, _ = model(
                        heatmaps,
                        query_hm,
                        detector_hm,
                        confidence,
                        joint_ids,
                    )
                refined_coco = raw_coco.copy()
                refined_coco[:, DIRECT_COCO] = heatmap_to_image(
                    refined_hm.cpu().numpy(),
                    data["input_center"],
                    data["input_scale"],
                    width,
                    height,
                )
                refined_h36m_xy_canonical = []
                for view in range(len(group_records)):
                    joints, _ = coco_to_h36m(
                        refined_coco[view], coco_confidence[view]
                    )
                    refined_h36m_xy_canonical.append(joints)
                refined_h36m_xy_canonical = np.stack(
                    refined_h36m_xy_canonical
                )
                # Preserve H22's original confidence inputs.  The tested
                # variable is the H21 coordinate correction only.
                refined_anchor_canonical = rumpl_anchor_h36m(
                    group_records,
                    refined_h36m_xy_canonical,
                    rumpl_h36m_confidence_canonical,
                    args.anchor_confidence_epsilon,
                    args.anchor_regularization,
                )
                refined_anchor = swap_lower_body(
                    refined_anchor_canonical[None]
                )[0]
                anchor_delta_canonical = (
                    refined_anchor_canonical - old_anchor_canonical
                )
                anchor_delta = refined_anchor - old_anchor
                anchor_delta_mm[query_source].append(
                    float(np.linalg.norm(anchor_delta, axis=-1).mean() * 1000.0)
                )
                add_error(
                    errors,
                    kp_errors,
                    f"refined_anchor_query_{query_source}",
                    action,
                    refined_anchor,
                    target,
                )
                for alpha in args.anchor_delta_scales:
                    method = (
                        f"h22_plus_{alpha:g}x_anchor_delta_query_"
                        f"{query_source}"
                    )
                    hybrid = rumpl_prediction + alpha * anchor_delta
                    add_error(
                        errors,
                        kp_errors,
                        method,
                        action,
                        hybrid,
                        target,
                    )
                if (
                    args.report_oracle_diagnostics
                    and query_source == "old_anchor"
                ):
                    numerator = (
                        (target - rumpl_prediction) * anchor_delta
                    ).sum(axis=-1)
                    denominator = np.maximum(
                        np.square(anchor_delta).sum(axis=-1), 1e-12
                    )
                    raw_oracle_scale = numerator / denominator
                    for label, lower, upper in (
                        ("clip0to1", 0.0, 1.0),
                        ("clipMinus1to1", -1.0, 1.0),
                    ):
                        oracle_scale = np.clip(
                            raw_oracle_scale, lower, upper
                        )
                        oracle_scale_values[label].extend(
                            oracle_scale.tolist()
                        )
                        oracle_prediction = (
                            rumpl_prediction
                            + oracle_scale[:, None] * anchor_delta
                        )
                        add_error(
                            errors,
                            kp_errors,
                            (
                                "validation_gt_oracle_per_joint_scale_"
                                f"{label}"
                            ),
                            action,
                            oracle_prediction,
                            target,
                        )
                if (
                    gate_model is not None
                    and gate_delta_source == "h21"
                    and query_source == "old_anchor"
                ):
                    if gate_feature_names == FEATURE_NAMES:
                        gate_features_np = anchor_gate_features(
                            group_records,
                            rumpl_h36m_xy_canonical,
                            rumpl_h36m_confidence_canonical,
                            old_anchor_canonical,
                            anchor_delta_canonical,
                        )
                    else:
                        rumpl_prediction_canonical = swap_lower_body(
                            rumpl_prediction[None]
                        )[0]
                        gate_features_np = anchor_context_gate_features(
                            group_records,
                            rumpl_h36m_xy_canonical,
                            refined_h36m_xy_canonical,
                            rumpl_h36m_confidence_canonical,
                            old_anchor_canonical,
                            anchor_delta_canonical,
                            rumpl_prediction_canonical,
                        )
                    with torch.no_grad():
                        gate_canonical = gate_model(
                            torch.as_tensor(
                                gate_features_np,
                                dtype=torch.float32,
                                device=device,
                            ),
                            gate_joint_ids,
                        ).cpu().numpy()
                    gate = swap_lower_body(
                        gate_canonical[None, :, None]
                    )[0, :, 0]
                    learned_gate_values[query_source].extend(
                        gate.tolist()
                    )
                    gated_hybrid = (
                        rumpl_prediction + gate[:, None] * anchor_delta
                    )
                    add_error(
                        errors,
                        kp_errors,
                        "h22_plus_learned_anchor_gate_query_old_anchor",
                        action,
                        gated_hybrid,
                        target,
                    )
            if position % 250 == 0 or position == len(indexed_work):
                print(
                    f"V{n_views}: {position}/{len(indexed_work)}",
                    flush=True,
                )

        if max(target_alignment_mm, default=0.0) > 1e-3:
            raise ValueError(
                f"V{n_views}: grouped target/H22 target alignment failed, "
                f"max={max(target_alignment_mm):.6f} mm"
            )
        summary = summarize(errors, kp_errors)
        result["results"][f"V{n_views}"] = {
            "views": n_views,
            "records": len(indexed_work),
            "prediction_dictionary": str(prediction_path.resolve()),
            "maximum_target_alignment_mm": max(
                target_alignment_mm, default=0.0
            ),
            "mean_anchor_delta_mm": {
                source: float(np.mean(values))
                for source, values in anchor_delta_mm.items()
            },
            "learned_gate_statistics": {
                source: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                for source, values in learned_gate_values.items()
            },
            "validation_gt_oracle_scale_statistics": {
                label: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "fraction_negative": float(
                        np.mean(np.asarray(values) < 0.0)
                    ),
                    "fraction_zero": float(
                        np.mean(np.asarray(values) == 0.0)
                    ),
                    "fraction_one": float(
                        np.mean(np.asarray(values) == 1.0)
                    ),
                }
                for label, values in oracle_scale_values.items()
            },
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
