#!/usr/bin/env python3
"""Audit camera/world-coordinate dependence of the frozen H76+C2 and E2.

The counterfactual applies the same rigid transform to every camera ray,
candidate pose and target pose.  It therefore changes only the arbitrary
world coordinate frame, not the underlying multiview observation.  A
camera-layout-independent method should obey

    f(R d, R o + t) = R f(d, o) + t.

The script separates two possible failure sources:

* ``generator`` reruns the frozen RUMPL/H76 model on transformed rays;
* ``scorer`` transforms an existing candidate pool analytically and reruns
  only E2, so any error there is caused by candidate scoring/fusion.

No training data or checkpoint is modified.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
RUMPL_REPO = HERE.parent / "OpenRUMPL" / "RUMPL"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RUMPL_REPO / "lib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--tri-anchor-reg", type=float, default=1e-4,
        help="Diagnostic override for the frozen generator's anchor solver.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Zero evaluates the complete cache.",
    )
    parser.add_argument(
        "--components", nargs="+",
        choices=("scorer", "generator", "generator_canonical"),
        default=("scorer", "generator"),
    )
    return parser.parse_args()


def rotation_xyz(degrees_xyz: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = (math.radians(value) for value in degrees_xyz)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return z @ y @ x


def transforms() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    identity = np.eye(3, dtype=np.float32)
    zero = np.zeros(3, dtype=np.float32)
    return {
        "identity": (identity, zero),
        "translate_x_1m": (identity, np.array([1.0, 0.0, 0.0], np.float32)),
        "translate_mixed": (
            identity, np.array([1.2, -0.7, 0.5], np.float32)
        ),
        "yaw_37deg": (rotation_xyz((0.0, 0.0, 37.0)), zero),
        "yaw_90deg": (rotation_xyz((0.0, 0.0, 90.0)), zero),
        "full_rotation": (rotation_xyz((20.0, -15.0, 43.0)), zero),
        "full_rotation_translation": (
            rotation_xyz((20.0, -15.0, 43.0)),
            np.array([1.2, -0.7, 0.5], np.float32),
        ),
    }


def transform_points(points: torch.Tensor, rotation: torch.Tensor,
                     translation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,...j->...i", rotation, points) + translation


def transform_rays(rays: torch.Tensor, rotation: torch.Tensor,
                   translation: torch.Tensor) -> torch.Tensor:
    transformed = rays.clone()
    transformed[..., :3] = torch.einsum(
        "ij,...j->...i", rotation, rays[..., :3]
    )
    transformed[..., 3:6] = transform_points(
        rays[..., 3:6], rotation, translation
    )
    return transformed


def equivariant_ray_anchors(
    rays: torch.Tensor, regularization: float = 1e-4,
    confidence_epsilon: float = 0.05,
) -> torch.Tensor:
    """Confidence-weighted ray intersection regularized about ray centroids.

    Regularizing about the coordinate origin breaks translation equivariance.
    The point-on-ray centroid transforms with the scene, so this equivalent
    Tikhonov system remains SE(3)-equivariant while retaining stability.
    """
    direction = F.normalize(rays[..., :3], dim=-1, eps=1e-7)
    point = rays[..., 3:6]
    confidence = rays[..., 6:7].clamp(0, 1) + confidence_epsilon
    eye = torch.eye(3, device=rays.device, dtype=rays.dtype)
    projection = eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    weighted_projection = confidence.unsqueeze(-1) * projection
    lhs = weighted_projection.sum(dim=2)
    rhs = (weighted_projection @ point.unsqueeze(-1)).sum(dim=2)
    centroid = (
        (confidence * point).sum(dim=2)
        / confidence.sum(dim=2).clamp_min(1e-7)
    )
    lhs = lhs + regularization * eye
    rhs = rhs + regularization * centroid.unsqueeze(-1)
    return torch.linalg.solve(lhs, rhs).squeeze(-1)


def body_canonical_frame(rays: torch.Tensor):
    """Build an SE(3)-equivariant pelvis/shoulder/torso frame from rays."""
    anchors = equivariant_ray_anchors(rays)
    origin = anchors[:, 0]
    # H36M-17 indices: left/right shoulder 11/14 and neck 8.
    x_axis = F.normalize(anchors[:, 14] - anchors[:, 11], dim=-1, eps=1e-7)
    up_hint = anchors[:, 8] - origin
    y_axis = up_hint - (up_hint * x_axis).sum(dim=-1, keepdim=True) * x_axis
    y_axis = F.normalize(y_axis, dim=-1, eps=1e-7)
    z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=-1), dim=-1, eps=1e-7)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-7)
    basis = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    return origin, basis


def canonicalize_rays(rays: torch.Tensor):
    origin, basis = body_canonical_frame(rays)
    canonical = rays.clone()
    canonical[..., :3] = torch.einsum(
        "b...i,bij->b...j", rays[..., :3], basis
    )
    centered_point = rays[..., 3:6] - origin[:, None, None, :]
    canonical[..., 3:6] = torch.einsum(
        "b...i,bij->b...j", centered_point, basis
    )
    return canonical, origin, basis


def canonical_to_world(points: torch.Tensor, origin: torch.Tensor,
                       basis: torch.Tensor) -> torch.Tensor:
    return (
        torch.einsum("b...j,bij->b...i", points, basis)
        + origin[:, None, :]
    )


def mean_joint_error_mm(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(first - second, dim=-1).mean().item() * 1000.0
    )


def percentile_mm(first: torch.Tensor, second: torch.Tensor,
                  percentile: float) -> float:
    values = torch.linalg.vector_norm(first - second, dim=-1).flatten()
    return float(torch.quantile(values, percentile / 100.0).item() * 1000.0)


def load_e2(path: Path, device: torch.device):
    from train_h76_set_transformer_utility_20260811 import (
        SetTransformerJointUtility,
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = SetTransformerJointUtility(
        checkpoint["mean"], checkpoint["std"],
        int(checkpoint["attention_depth"]),
        stage_heads=bool(checkpoint.get("stage_heads", False)),
        canonical_geometry=bool(checkpoint.get("canonical_geometry", False)),
        fixed_metric_normalization=bool(
            checkpoint.get("fixed_metric_normalization", False)
        ),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    combinations = tuple(
        tuple(int(view) for view in item)
        for item in checkpoint["candidate_combinations"]
    )
    return model, combinations, float(checkpoint.get("temperature", 1.8))


def task_spec(
    task_combo: tuple[int, ...], candidate_combinations: tuple[tuple[int, ...], ...],
    device: torch.device,
):
    task_set = set(task_combo)
    available = [
        index for index, combo in enumerate(candidate_combinations)
        if set(combo).issubset(task_set)
    ]
    baseline_global = next(
        index for index, combo in enumerate(candidate_combinations)
        if combo == task_combo
    )
    baseline_local = available.index(baseline_global)
    candidate_masks = torch.zeros(
        len(available), 4, dtype=torch.float32, device=device
    )
    for row, index in enumerate(available):
        candidate_masks[row, list(candidate_combinations[index])] = 1.0
    task_mask = torch.zeros(4, dtype=torch.float32, device=device)
    task_mask[list(task_combo)] = 1.0
    return available, candidate_masks, task_mask, baseline_local


@torch.inference_mode()
def e2_outputs(model, predictions: torch.Tensor, rays: torch.Tensor,
               candidate_combinations, temperature: float):
    outputs = {}
    for task_combo in (
        combo for count in (2, 3, 4)
        for combo in itertools.combinations(range(4), count)
    ):
        available, masks, task_mask, baseline_local = task_spec(
            task_combo, candidate_combinations, predictions.device
        )
        candidates = predictions[:, available]
        score = model(candidates, rays, masks, task_mask)
        score = score - score[..., baseline_local:baseline_local + 1]
        weights = F.softmax(-score / temperature, dim=-1)
        soft = torch.einsum("bjc,bcjd->bjd", weights, candidates)
        hard_index = score.argmin(dim=-1)
        hard = candidates.permute(0, 2, 1, 3).gather(
            2, hard_index[..., None, None].expand(-1, -1, 1, 3)
        ).squeeze(2)
        outputs[task_combo] = {
            "soft": soft,
            "hard": hard,
            "score": score,
            "weights": weights,
            "hard_index": hard_index,
            "baseline": candidates[:, baseline_local],
        }
    return outputs


@torch.inference_mode()
def audit_scorer(
    model, candidate_combinations, temperature, predictions, targets, rays,
    transform_table, batch_size,
):
    accumulators = {
        name: defaultdict(list) for name in transform_table
    }
    for start in range(0, len(predictions), batch_size):
        stop = min(start + batch_size, len(predictions))
        batch_predictions = predictions[start:stop]
        batch_targets = targets[start:stop]
        batch_rays = rays[start:stop]
        reference = e2_outputs(
            model, batch_predictions, batch_rays,
            candidate_combinations, temperature,
        )
        for name, (rotation_np, translation_np) in transform_table.items():
            rotation = torch.as_tensor(
                rotation_np, device=predictions.device, dtype=predictions.dtype
            )
            translation = torch.as_tensor(
                translation_np, device=predictions.device, dtype=predictions.dtype
            )
            transformed_predictions = transform_points(
                batch_predictions, rotation, translation
            )
            transformed_targets = transform_points(
                batch_targets, rotation, translation
            )
            transformed_rays = transform_rays(
                batch_rays, rotation, translation
            )
            counterfactual = e2_outputs(
                model, transformed_predictions, transformed_rays,
                candidate_combinations, temperature,
            )
            for task_combo, original in reference.items():
                stage = f"V{len(task_combo)}"
                changed = counterfactual[task_combo]
                expected_soft = transform_points(
                    original["soft"], rotation, translation
                )
                expected_hard = transform_points(
                    original["hard"], rotation, translation
                )
                expected_baseline = transform_points(
                    original["baseline"], rotation, translation
                )
                accumulators[name][(stage, "soft_equiv")].append(
                    torch.linalg.vector_norm(
                        changed["soft"] - expected_soft, dim=-1
                    ).detach().cpu()
                )
                accumulators[name][(stage, "hard_equiv")].append(
                    torch.linalg.vector_norm(
                        changed["hard"] - expected_hard, dim=-1
                    ).detach().cpu()
                )
                accumulators[name][(stage, "baseline_equiv")].append(
                    torch.linalg.vector_norm(
                        changed["baseline"] - expected_baseline, dim=-1
                    ).detach().cpu()
                )
                accumulators[name][(stage, "soft_gt")].append(
                    torch.linalg.vector_norm(
                        changed["soft"] - transformed_targets, dim=-1
                    ).detach().cpu()
                )
                accumulators[name][(stage, "reference_soft_gt")].append(
                    torch.linalg.vector_norm(
                        original["soft"] - batch_targets, dim=-1
                    ).detach().cpu()
                )
                accumulators[name][(stage, "weight_l1")].append(
                    (changed["weights"] - original["weights"])
                    .abs().mean(dim=-1).detach().cpu()
                )
                accumulators[name][(stage, "hard_flip")].append(
                    (changed["hard_index"] != original["hard_index"])
                    .float().detach().cpu()
                )
                accumulators[name][(stage, "score_abs")].append(
                    (changed["score"] - original["score"])
                    .abs().mean(dim=-1).detach().cpu()
                )
    result = {}
    for name, store in accumulators.items():
        result[name] = {}
        for stage in ("V2", "V3", "V4"):
            values = {
                metric: torch.cat(store[(stage, metric)]).flatten()
                for metric in (
                    "soft_equiv", "hard_equiv", "baseline_equiv", "soft_gt",
                    "reference_soft_gt", "weight_l1", "hard_flip", "score_abs",
                )
            }
            result[name][stage] = {
                "soft_equivariance_mean_mm": float(values["soft_equiv"].mean() * 1000),
                "soft_equivariance_p95_mm": float(
                    torch.quantile(values["soft_equiv"], 0.95) * 1000
                ),
                "hard_equivariance_mean_mm": float(values["hard_equiv"].mean() * 1000),
                "baseline_control_equivariance_mm": float(
                    values["baseline_equiv"].mean() * 1000
                ),
                "reference_soft_mpjpe_mm": float(
                    values["reference_soft_gt"].mean() * 1000
                ),
                "transformed_soft_mpjpe_mm": float(values["soft_gt"].mean() * 1000),
                "mpjpe_change_mm": float(
                    (values["soft_gt"].mean() - values["reference_soft_gt"].mean())
                    * 1000
                ),
                "mean_weight_l1": float(values["weight_l1"].mean()),
                "hard_selection_flip_rate": float(values["hard_flip"].mean()),
                "mean_relative_score_change": float(values["score_abs"].mean()),
            }
    return result


def configure_generator(cfg_path: str, tri_anchor_reg: float):
    # Reconstruct exactly the C2/H76 inference architecture before importing
    # and instantiating the model.  These are architecture switches, not new
    # experimental choices.
    settings = {
        "RUMPL_TRI_ANCHOR": "1",
        "RUMPL_TRI_ANCHOR_REG": str(tri_anchor_reg),
        "RUMPL_TRI_ANCHOR_CONF_EPS": "0.05",
        "RUMPL_PFT_REPEAT_LAST": "1",
        "RUMPL_ANCHOR_CENTERED_RAYS": "1",
        "RUMPL_INPUT_PLUCKER": "1",
        "RUMPL_INPUT_HARMONIC_L": "0",
        "RUMPL_GEOMETRY_UNCERTAINTY_TOKEN": "0",
        "RUMPL_RELATIVE_VIEW_FUSION": "0",
        "RUMPL_SKELETON_VIEW_RELIABILITY": "0",
        "RUMPL_CONFIDENCE_VIEW_BIAS": "0",
        "RUMPL_GEOMETRY_VIEW_BIAS": "0",
        "RUMPL_NORMALIZE_VIEW_CONFIDENCE": "1",
        "RUMPL_JOINT_CONFIDENCE_VIEW_BIAS": "0",
        "RUMPL_JOINT_GEOMETRY_VIEW_BIAS": "0",
        "RUMPL_GBT_SET_DECODER": "0",
        "RUMPL_SKIP_VFT": "0",
        "RUMPL_SKIP_PFT": "0",
        "RUMPL_VFT_DEPTH": "0",
        "GBT_GLOBAL_JV_DEPTH": "0",
        "GBT_LEARNABLE_BIAS": "0",
    }
    os.environ.update(settings)
    from core.config import config, update_config

    update_config(cfg_path)
    config.DATASET.N_VIEWS_TRAIN_TEST_ALL = 4
    config.DATASET.TEST_VIEWS = [1, 2, 3, 4]
    config.GPUS = "0"
    from export_h76_train_subset_hypotheses_20260811 import load_model

    return load_model, settings


@torch.inference_mode()
def audit_generator(
    model, targets, rays, transform_table, batch_size, canonical: bool = False,
):
    task_combinations = tuple(
        combo for count in (2, 3, 4)
        for combo in itertools.combinations(range(4), count)
    )
    stores = {name: defaultdict(list) for name in transform_table}
    for start in range(0, len(rays), batch_size):
        stop = min(start + batch_size, len(rays))
        batch_rays = rays[start:stop]
        batch_targets = targets[start:stop]
        reference = {}
        for combo in task_combinations:
            subset = batch_rays[:, :, list(combo)]
            if canonical:
                canonical_rays, origin, basis = canonicalize_rays(subset)
                canonical_pose = model(canonical_rays, is_training=False)
                reference[combo] = canonical_to_world(
                    canonical_pose, origin, basis
                )
            else:
                reference[combo] = model(subset, is_training=False)
        for name, (rotation_np, translation_np) in transform_table.items():
            rotation = torch.as_tensor(
                rotation_np, device=rays.device, dtype=rays.dtype
            )
            translation = torch.as_tensor(
                translation_np, device=rays.device, dtype=rays.dtype
            )
            transformed_rays = transform_rays(batch_rays, rotation, translation)
            transformed_targets = transform_points(
                batch_targets, rotation, translation
            )
            for combo in task_combinations:
                stage = f"V{len(combo)}"
                subset = transformed_rays[:, :, list(combo)]
                if canonical:
                    canonical_rays, origin, basis = canonicalize_rays(subset)
                    canonical_pose = model(canonical_rays, is_training=False)
                    changed = canonical_to_world(canonical_pose, origin, basis)
                else:
                    changed = model(subset, is_training=False)
                expected = transform_points(
                    reference[combo], rotation, translation
                )
                stores[name][(stage, "equiv")].append(
                    torch.linalg.vector_norm(changed - expected, dim=-1).cpu()
                )
                stores[name][(stage, "gt")].append(
                    torch.linalg.vector_norm(
                        changed - transformed_targets, dim=-1
                    ).cpu()
                )
                stores[name][(stage, "reference_gt")].append(
                    torch.linalg.vector_norm(
                        reference[combo] - batch_targets, dim=-1
                    ).cpu()
                )
    result = {}
    for name, store in stores.items():
        result[name] = {}
        for stage in ("V2", "V3", "V4"):
            equiv = torch.cat(store[(stage, "equiv")]).flatten()
            gt = torch.cat(store[(stage, "gt")]).flatten()
            reference_gt = torch.cat(store[(stage, "reference_gt")]).flatten()
            result[name][stage] = {
                "equivariance_mean_mm": float(equiv.mean() * 1000),
                "equivariance_p95_mm": float(torch.quantile(equiv, 0.95) * 1000),
                "reference_mpjpe_mm": float(reference_gt.mean() * 1000),
                "transformed_mpjpe_mm": float(gt.mean() * 1000),
                "mpjpe_change_mm": float((gt.mean() - reference_gt.mean()) * 1000),
            }
    return result


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cache_path = Path(args.cache).resolve()
    source = np.load(cache_path, allow_pickle=False)
    count = len(source["targets"])
    if args.max_samples:
        count = min(count, args.max_samples)
    predictions = torch.from_numpy(source["predictions"][:count]).to(device)
    targets = torch.from_numpy(source["targets"][:count]).to(device)
    rays = torch.from_numpy(source["rays"][:count]).to(device)
    transform_table = transforms()

    payload = {
        "purpose": "world-coordinate rigid-transform equivariance audit",
        "interpretation": (
            "The physical multiview observation is unchanged. Non-zero error "
            "therefore measures dependence on the arbitrary world frame."
        ),
        "cache": str(cache_path),
        "samples": count,
        "transforms": {
            name: {"rotation": rotation.tolist(), "translation_m": translation.tolist()}
            for name, (rotation, translation) in transform_table.items()
        },
    }

    if "scorer" in args.components:
        e2, candidate_combinations, temperature = load_e2(
            Path(args.e2_checkpoint).resolve(), device
        )
        if predictions.shape[1] != len(candidate_combinations):
            raise ValueError(
                f"cache has {predictions.shape[1]} candidates but checkpoint "
                f"expects {len(candidate_combinations)}"
            )
        payload["scorer"] = {
            "checkpoint": str(Path(args.e2_checkpoint).resolve()),
            "temperature": temperature,
            "candidate_combinations": [list(item) for item in candidate_combinations],
            "results": audit_scorer(
                e2, candidate_combinations, temperature, predictions, targets,
                rays, transform_table, args.batch_size,
            ),
        }

    if "generator" in args.components or "generator_canonical" in args.components:
        load_model, settings = configure_generator(args.cfg, args.tri_anchor_reg)
        generator = load_model(
            Path(args.generator_checkpoint).resolve(), device
        )
        if "generator" in args.components:
            payload["generator"] = {
                "checkpoint": str(Path(args.generator_checkpoint).resolve()),
                "architecture_environment": settings,
                "results": audit_generator(
                    generator, targets, rays, transform_table, args.batch_size,
                ),
            }
        if "generator_canonical" in args.components:
            payload["generator_canonical"] = {
                "checkpoint": str(Path(args.generator_checkpoint).resolve()),
                "architecture_environment": settings,
                "canonical_frame": (
                    "equivariant ray triangulation; pelvis origin; shoulder x-axis; "
                    "pelvis-to-neck orthogonal y-axis"
                ),
                "results": audit_generator(
                    generator, targets, rays, transform_table, args.batch_size,
                    canonical=True,
                ),
            }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
