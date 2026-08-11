#!/usr/bin/env python3
"""Measure the geometry-only ceiling of the current H36M 2D/ray inputs.

This deliberately bypasses the RUMPL network.  It intersects the rays already
produced by ``MultiViewH36M_RUMPL`` and reports absolute and pelvis-relative
MPJPE for every requested number of views.  Running both ``org`` and
``mmpose`` separates a fusion failure from a 2D detector/domain failure.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[1] / "OpenRUMPL" / "RUMPL"
sys.path.insert(0, str(REPO / "lib"))
sys.path.insert(0, str(REPO))

from core.config import config, update_config  # noqa: E402
from dataset.multiview_h36m_rumpl import MultiViewH36M_RUMPL  # noqa: E402


H36M17_NAMES = (
    "root", "rhip", "rknee", "rankle", "lhip", "lknee", "lankle",
    "belly", "neck", "nose", "head", "lshoulder", "lelbow", "lwrist",
    "rshoulder", "relbow", "rwrist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--views", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reg", type=float, default=1e-7)
    parser.add_argument("--conf-eps", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mmpose-type",
        default=None,
        help="override TEST_MMPOSE_TYPE for an exact detector/refiner audit",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("org", "mmpose"),
        default=["org", "mmpose"],
    )
    return parser.parse_args()


def intersect_rays(
    rays: torch.Tensor,
    use_confidence: bool,
    reg: float,
    conf_eps: float,
) -> torch.Tensor:
    """Weighted least-squares intersection for rays shaped (B,J,V,7)."""
    direction = torch.nn.functional.normalize(rays[..., :3].double(), dim=-1)
    point = rays[..., 3:6].double()
    eye = torch.eye(3, dtype=torch.double, device=rays.device)
    projection = eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    if use_confidence:
        weight = rays[..., 6:7].double().clamp(0.0, 1.0) + conf_eps
    else:
        weight = torch.ones_like(rays[..., 6:7], dtype=torch.double)
    weighted_projection = projection * weight.unsqueeze(-1)
    lhs = weighted_projection.sum(dim=2) + reg * eye
    rhs = (weighted_projection @ point.unsqueeze(-1)).sum(dim=2)
    return torch.linalg.solve(lhs, rhs).squeeze(-1).float()


def intersect_rays_irls(
    rays: torch.Tensor,
    reg: float,
    conf_eps: float,
    iterations: int = 5,
) -> torch.Tensor:
    """Confidence + robust Cauchy IRLS, requiring no ground-truth information."""
    direction = torch.nn.functional.normalize(rays[..., :3].double(), dim=-1)
    point = rays[..., 3:6].double()
    base_weight = rays[..., 6:7].double().clamp(0.0, 1.0) + conf_eps
    working_rays = rays.clone()
    pred = intersect_rays(working_rays, True, reg, conf_eps).double()
    for _ in range(iterations):
        offset = pred.unsqueeze(2) - point
        residual = torch.linalg.vector_norm(
            torch.cross(offset, direction, dim=-1), dim=-1
        )
        scale = residual.median(dim=2, keepdim=True).values.clamp_min(0.005)
        robust_weight = 1.0 / (1.0 + (residual / (2.3849 * scale)).square())
        weight = base_weight * robust_weight.unsqueeze(-1)
        eye = torch.eye(3, dtype=torch.double, device=rays.device)
        projection = eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)
        weighted_projection = projection * weight.unsqueeze(-1)
        lhs = weighted_projection.sum(dim=2) + reg * eye
        rhs = (weighted_projection @ point.unsqueeze(-1)).sum(dim=2)
        pred = torch.linalg.solve(lhs, rhs).squeeze(-1)
    return pred.float()


def intersect_rays_oracle_reliability(
    rays: torch.Tensor,
    target: torch.Tensor,
    reg: float,
    conf_eps: float,
) -> torch.Tensor:
    """Diagnostic upper bound: weight views by true point-to-ray error."""
    direction = torch.nn.functional.normalize(rays[..., :3].double(), dim=-1)
    point = rays[..., 3:6].double()
    offset = target.double().unsqueeze(2) - point
    residual = torch.linalg.vector_norm(
        torch.cross(offset, direction, dim=-1), dim=-1
    )
    scale = residual.median(dim=2, keepdim=True).values.clamp_min(0.002)
    reliability = torch.exp(-residual / scale)
    confidence = rays[..., 6].double().clamp(0.0, 1.0) + conf_eps
    weight = confidence * reliability
    eye = torch.eye(3, dtype=torch.double, device=rays.device)
    projection = eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    weighted_projection = projection * weight.unsqueeze(-1).unsqueeze(-1)
    lhs = weighted_projection.sum(dim=2) + reg * eye
    rhs = (weighted_projection @ point.unsqueeze(-1)).sum(dim=2)
    return torch.linalg.solve(lhs, rhs).squeeze(-1).float()


def camera_tuple(dataset: MultiViewH36M_RUMPL, group: list[int]) -> str:
    ids = [int(dataset.db[index]["camera_id"]) + 1 for index in group]
    return "-".join(str(camera_id) for camera_id in sorted(ids))


def evaluate_one(
    base_cfg,
    source: str,
    n_views: int,
    batch_size: int,
    workers: int,
    reg: float,
    conf_eps: float,
    mmpose_type: str | None,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.DATASET.TEST_VIEWS = list(range(1, n_views + 1))
    cfg.DATASET.TEST_ON_ALL_CAMERAS = True
    cfg.DATASET.USE_MMPOSE_VAL = source == "mmpose"
    cfg.DATASET.USE_MMPOSE_TEST = source == "mmpose"
    cfg.DATASET.APPLY_NOISE_MISSING_TEST = False
    if mmpose_type is not None:
        cfg.DATASET.TEST_MMPOSE_TYPE = mmpose_type

    dataset = MultiViewH36M_RUMPL(
        cfg,
        cfg.DATASET.TEST_SUBSET,
        False,
        transform=None,
        is_mmpose=source == "mmpose",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
    )

    accumulators = {
        "uniform": defaultdict(list),
        "confidence": defaultdict(list),
        "robust_irls": defaultdict(list),
        "oracle_reliability": defaultdict(list),
    }
    sample_cursor = 0
    for _, _, target, rays, _, _ in loader:
        batch_n = target.shape[0]
        target = target.float()
        predictions = {
            "uniform": intersect_rays(rays, False, reg, conf_eps),
            "confidence": intersect_rays(rays, True, reg, conf_eps),
            "robust_irls": intersect_rays_irls(rays, reg, conf_eps),
            "oracle_reliability": intersect_rays_oracle_reliability(
                rays, target, reg, conf_eps
            ),
        }
        for method, pred in predictions.items():
            absolute = torch.linalg.vector_norm(pred - target, dim=-1) * 1000.0
            pred_relative = pred - pred[:, :1]
            target_relative = target - target[:, :1]
            relative = (
                torch.linalg.vector_norm(pred_relative - target_relative, dim=-1)
                * 1000.0
            )
            for offset in range(batch_n):
                pair = camera_tuple(dataset, dataset.grouping[sample_cursor + offset])
                accumulators[method][f"pair/{pair}/absolute"].extend(
                    absolute[offset].tolist()
                )
                accumulators[method][f"pair/{pair}/relative"].extend(
                    relative[offset].tolist()
                )
            accumulators[method]["all/absolute"].extend(absolute.flatten().tolist())
            accumulators[method]["all/relative"].extend(relative.flatten().tolist())
            for joint_index, joint_name in enumerate(H36M17_NAMES):
                accumulators[method][f"joint/{joint_name}/absolute"].extend(
                    absolute[:, joint_index].tolist()
                )
                accumulators[method][f"joint/{joint_name}/relative"].extend(
                    relative[:, joint_index].tolist()
                )
        sample_cursor += batch_n

    result = {
        "source": source,
        "views": n_views,
        "samples": len(dataset),
        "methods": {},
    }
    for method, values in accumulators.items():
        result["methods"][method] = {
            key: float(np.mean(errors)) for key, errors in sorted(values.items())
        }
    return result


def main() -> None:
    args = parse_args()
    update_config(args.cfg)
    base_cfg = copy.deepcopy(config)
    results = []
    for source in args.sources:
        for n_views in args.views:
            result = evaluate_one(
                base_cfg,
                source,
                n_views,
                args.batch_size,
                args.workers,
                args.reg,
                args.conf_eps,
                args.mmpose_type,
            )
            results.append(result)
            uniform = result["methods"]["uniform"]
            confidence = result["methods"]["confidence"]
            robust = result["methods"]["robust_irls"]
            oracle = result["methods"]["oracle_reliability"]
            print(
                f"{source:6s} V{n_views}: "
                f"uniform abs/rel={uniform['all/absolute']:.3f}/"
                f"{uniform['all/relative']:.3f} mm; "
                f"conf abs/rel={confidence['all/absolute']:.3f}/"
                f"{confidence['all/relative']:.3f} mm; "
                f"IRLS={robust['all/absolute']:.3f}; "
                f"oracle-weight={oracle['all/absolute']:.3f}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": os.path.abspath(args.cfg),
        "reg": args.reg,
        "conf_eps": args.conf_eps,
        "mmpose_type": args.mmpose_type,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
