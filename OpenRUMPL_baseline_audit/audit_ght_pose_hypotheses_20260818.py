#!/usr/bin/env python3
"""Zero-training audit of the official GHT PoseDSAC hypothesis generator.

The public Generalizable Human Pose Triangulation (CVPR 2022) code samples a
different camera subset for each joint of every pose hypothesis.  Earlier
experiments in this repository used the GHT whole-pose ScoreNN preprocessing
on a frozen 22-candidate RUMPL pool, which is not the official PoseDSAC
candidate generator.  This script isolates that missing part before any
scorer training:

* raw HRNet coordinates/confidences and calibrated cameras are used;
* each hypothesis is generated without 3D labels;
* all 2/3/4-view camera combinations are evaluated with the same
  action-equal absolute All-17 metric used by the RUMPL audit;
* the frozen RUMPL/H76 11- and 22-candidate pools are reported separately and
  as an oracle union, never as a trained or test-time label source.

The generator follows ``src/dsac.py::PoseDSAC.__sample_hyp``: for a task with
K cameras, the per-joint subset is drawn uniformly from all subsets of sizes
2..K, and every subset is triangulated by the official linear DLT routine.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


ACTION_NAMES = {
    2: "Direction", 3: "Discuss", 4: "Eating", 5: "Greet",
    6: "Phone", 7: "Photo", 8: "Pose", 9: "Purchase",
    10: "Sitting", 11: "SittingDown", 12: "Smoke", 13: "Wait",
    14: "WalkDog", 15: "Walk", 16: "WalkTwo",
}
ALL_COMBINATIONS = tuple(
    combo
    for views in (2, 3, 4)
    for combo in itertools.combinations(range(4), views)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-pkl", required=True)
    parser.add_argument("--compare-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hyps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument(
        "--triangulation-weight", choices=("none", "confidence"), default="none",
        help="official GHT uses none; confidence is a separate geometry diagnostic",
    )
    return parser.parse_args()


def action_equal(errors: np.ndarray, actions: np.ndarray) -> float:
    values = [
        float(errors[actions == action].mean())
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]
    return float(np.mean(values))


def load_grouped_pkl(path: Path, max_groups: int = 0) -> dict[str, np.ndarray]:
    """Load synchronized four-camera H36M records in file/group order."""
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"unexpected pkl structure: {path}")

    groups: OrderedDict[tuple[int, int], dict[int, dict]] = OrderedDict()
    for record in records:
        key = (int(record["video_id"]), int(record["image_id"]))
        groups.setdefault(key, {})[int(record["camera_id"])] = record
    if max_groups:
        groups = OrderedDict(list(groups.items())[:max_groups])

    missing = [key for key, cams in groups.items() if set(cams) != {0, 1, 2, 3}]
    if missing:
        raise ValueError(f"{len(missing)} groups do not contain cameras 0..3")

    first = next(iter(groups.values()))
    joints = int(np.asarray(first[0]["joints_2d"]).shape[0])
    if joints != 17:
        raise ValueError(f"expected H36M-17, got {joints}")

    points = np.empty((len(groups), 4, joints, 2), dtype=np.float32)
    confidences = np.empty((len(groups), 4, joints), dtype=np.float32)
    projections = np.empty((len(groups), 4, 3, 4), dtype=np.float32)
    targets = np.empty((len(groups), joints, 3), dtype=np.float32)
    actions = np.empty(len(groups), dtype=np.int16)
    subjects = np.empty(len(groups), dtype=np.int16)
    keys = []
    for row, (key, cams) in enumerate(groups.items()):
        reference = cams[0]
        # H36M annotation coordinates/cameras are millimetres; the RUMPL
        # protocol stores the world pose in metres.
        targets[row] = np.asarray(reference["joints_3d"], dtype=np.float32) / 1000.0
        actions[row] = int(reference["action"])
        subjects[row] = int(reference["subject"])
        keys.append([int(key[0]), int(key[1])])
        for camera_id in range(4):
            record = cams[camera_id]
            points[row, camera_id] = np.asarray(record["joints_2d"], dtype=np.float32)
            confidences[row, camera_id] = np.asarray(
                record["joints_2d_conf"], dtype=np.float32
            ).reshape(joints)
            camera = record["camera"]
            K = np.asarray(camera["K"], dtype=np.float32)
            R = np.asarray(camera["R"], dtype=np.float32)
            t = np.asarray(camera["t"], dtype=np.float32).reshape(3, 1)
            projections[row, camera_id] = K @ np.concatenate((R, t), axis=1)
    return {
        "points": points,
        "confidences": confidences,
        "projections": projections,
        "targets": targets,
        "actions": actions,
        "subjects": subjects,
        "keys": np.asarray(keys, dtype=np.int64),
    }


def triangulate_dlt(
    projections: torch.Tensor, points: torch.Tensor, camera_ids: tuple[int, ...],
    confidences: torch.Tensor | None = None,
) -> torch.Tensor:
    """Official GHT linear DLT, batched over N frames and 17 joints."""
    P = projections[:, list(camera_ids)]  # N,V,3,4
    xy = points[:, list(camera_ids)]  # N,V,J,2
    n, views, joints = xy.shape[:3]
    row0 = xy[..., 0:1, None] * P[:, :, None, 2:3, :] - P[:, :, None, 0:1, :]
    row1 = xy[..., 1:2, None] * P[:, :, None, 2:3, :] - P[:, :, None, 1:2, :]
    A = torch.stack((row0[..., 0, :], row1[..., 0, :]), dim=3).permute(
        0, 2, 1, 3, 4
    )
    if confidences is not None:
        weights = confidences[:, list(camera_ids)].transpose(1, 2)
        A = A * weights[..., None, None]
    A = A.reshape(n * joints, 2 * views, 4)
    _, _, vh = torch.linalg.svd(A, full_matrices=False)
    homogeneous = vh[:, -1]
    denominator = homogeneous[:, 3:4]
    # The SVD solution has an arbitrary global sign.  Clamping with
    # ``clamp_min`` would turn every negative homogeneous scale into a tiny
    # positive number and create metre-scale explosions; the official helper
    # divides by the signed last coordinate.
    safe_denominator = torch.where(
        denominator.abs() < 1e-8,
        torch.where(denominator < 0, -torch.full_like(denominator, 1e-8),
                    torch.full_like(denominator, 1e-8)),
        denominator,
    )
    xyz = homogeneous[:, :3] / safe_denominator
    return xyz.reshape(n, joints, 3) / 1000.0


def stage_tasks(views: int) -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.combinations(range(4), views))


def subset_pool(task: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        subset
        for size in range(2, len(task) + 1)
        for subset in itertools.combinations(task, size)
    )


def generate_oracle(
    triangulated: dict[tuple[int, ...], torch.Tensor],
    target: torch.Tensor,
    actions: np.ndarray,
    task: tuple[int, ...],
    hyps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate official-style per-joint hypotheses and return min errors."""
    subsets = subset_pool(task)
    base = torch.stack([triangulated[subset] for subset in subsets], dim=1)
    choice = rng.integers(0, len(subsets), size=(hyps, 17), dtype=np.int64)
    choice_t = torch.from_numpy(choice).to(base.device)
    one_hot = torch.nn.functional.one_hot(choice_t, num_classes=len(subsets))
    one_hot = one_hot.to(dtype=base.dtype)
    # base is N,C,J,3; one_hot is H,J,C.
    candidates = torch.einsum("hjc,ncjd->nhjd", one_hot, base)
    errors = torch.linalg.vector_norm(candidates - target[:, None], dim=-1)
    errors = errors.mean(dim=-1).min(dim=1).values.detach().cpu().numpy() * 1000.0
    direct = torch.linalg.vector_norm(
        triangulated[task] - target, dim=-1
    ).mean(dim=-1).detach().cpu().numpy() * 1000.0
    return errors.astype(np.float32), direct.astype(np.float32)


def cached_oracle(
    cache: dict[str, np.ndarray], task: tuple[int, ...], extra: int = 0
) -> np.ndarray:
    """Return the frozen H76 pool oracle for one camera task."""
    # Existing cache order is V2 (6), V3 (4), V4 (1), then optional confidence
    # candidates in the same order.
    start = ALL_COMBINATIONS.index(task)
    indices = [start]
    if extra:
        indices.append(11 + start)
    candidates = cache["predictions"][:, indices]
    target = cache["targets"][:, None]
    return np.linalg.norm(candidates - target, axis=-1).mean(axis=-1).min(axis=1) * 1000.0


def main() -> None:
    args = parse_args()
    if args.hyps < 1:
        raise ValueError("--hyps must be positive")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.validation_pkl).resolve()
    cache_path = Path(args.compare_cache).resolve()
    print(f"loading {source}", flush=True)
    arrays = load_grouped_pkl(source, args.max_groups)
    cache_npz = np.load(cache_path)
    cache = {key: cache_npz[key] for key in cache_npz.files}
    if args.max_groups:
        cache = {
            key: value[: len(arrays["targets"])]
            if getattr(value, "ndim", 0) > 0 and len(value) >= len(arrays["targets"])
            else value
            for key, value in cache.items()
        }
    if len(cache["targets"]) != len(arrays["targets"]):
        raise ValueError(
            f"cache/group mismatch: {len(cache['targets'])} vs {len(arrays['targets'])}"
        )
    if not np.allclose(cache["targets"], arrays["targets"], atol=2e-5):
        raise ValueError("cache targets are not aligned with raw validation pkl")
    if not np.array_equal(cache["actions"], arrays["actions"]):
        raise ValueError("cache actions are not aligned with raw validation pkl")

    requested = args.gpu
    device = torch.device(
        f"cuda:{requested}" if torch.cuda.is_available() else "cpu"
    )
    points = torch.from_numpy(arrays["points"]).to(device)
    projections = torch.from_numpy(arrays["projections"]).to(device)
    target = torch.from_numpy(arrays["targets"]).to(device)
    confidence = torch.from_numpy(arrays["confidences"]).to(device)
    print(
        f"groups={len(target)} hyps={args.hyps} device={device} "
        f"pkl_protocol=raw_2d+K[R|t]", flush=True
    )

    start_time = time.time()
    triangulated: dict[tuple[int, ...], torch.Tensor] = {}
    with torch.inference_mode():
        for combo in ALL_COMBINATIONS:
            triangulated[combo] = triangulate_dlt(
                projections, points, combo,
                confidence if args.triangulation_weight == "confidence" else None,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print(f"triangulated {combo}", flush=True)

    rng = np.random.default_rng(args.seed)
    records = []
    for views in (2, 3, 4):
        task_records = []
        for task in stage_tasks(views):
            with torch.inference_mode():
                ght_oracle, direct = generate_oracle(
                    triangulated, target, arrays["actions"], task,
                    args.hyps, rng,
                )
            h76_11 = cached_oracle(cache, task, extra=0)
            h76_22 = cached_oracle(cache, task, extra=1)
            union_11 = np.minimum(ght_oracle, h76_11)
            union_22 = np.minimum(ght_oracle, h76_22)
            task_record = {
                "task_zero_based": list(task),
                "subset_pool": [list(x) for x in subset_pool(task)],
                "official_ght_random_oracle_mm": action_equal(
                    ght_oracle, arrays["actions"]
                ),
                "official_ght_all_view_dlt_mm": action_equal(
                    direct, arrays["actions"]
                ),
                "h76_11_candidate_oracle_mm": action_equal(
                    h76_11, arrays["actions"]
                ),
                "h76_22_candidate_oracle_mm": action_equal(
                    h76_22, arrays["actions"]
                ),
                "union_ght_plus_h76_11_oracle_mm": action_equal(
                    union_11, arrays["actions"]
                ),
                "union_ght_plus_h76_22_oracle_mm": action_equal(
                    union_22, arrays["actions"]
                ),
                "frame_weighted_mm": {
                    "ght": float(ght_oracle.mean()),
                    "direct": float(direct.mean()),
                    "h76_11": float(h76_11.mean()),
                    "h76_22": float(h76_22.mean()),
                    "union_11": float(union_11.mean()),
                    "union_22": float(union_22.mean()),
                },
            }
            task_records.append(task_record)
            records.append((views, task_record, ght_oracle, direct, h76_11, h76_22))
            print(
                f"V{views} task={task} "
                f"GHT={task_record['official_ght_random_oracle_mm']:.3f} "
                f"H76-22={task_record['h76_22_candidate_oracle_mm']:.3f} "
                f"union={task_record['union_ght_plus_h76_22_oracle_mm']:.3f}",
                flush=True,
            )

    summary = {}
    for views in (2, 3, 4):
        selected = [record for stage, record, *_ in records if stage == views]
        def mean_metric(name: str) -> float:
            return float(np.mean([item[name] for item in selected]))
        summary[f"V{views}"] = {
            "official_ght_random_oracle_mm": mean_metric(
                "official_ght_random_oracle_mm"
            ),
            "official_ght_all_view_dlt_mm": mean_metric(
                "official_ght_all_view_dlt_mm"
            ),
            "h76_11_candidate_oracle_mm": mean_metric(
                "h76_11_candidate_oracle_mm"
            ),
            "h76_22_candidate_oracle_mm": mean_metric(
                "h76_22_candidate_oracle_mm"
            ),
            "union_ght_plus_h76_11_oracle_mm": mean_metric(
                "union_ght_plus_h76_11_oracle_mm"
            ),
            "union_ght_plus_h76_22_oracle_mm": mean_metric(
                "union_ght_plus_h76_22_oracle_mm"
            ),
            "tasks": selected,
        }
    result = {
        "experiment": "G0_official_GHT_PoseDSAC_zero_training_oracle",
        "paper": "Generalizable Human Pose Triangulation, CVPR 2022",
        "official_reference": "reference/general-3d-humans-official/src/dsac.py::PoseDSAC",
        "protocol": {
            "input": "GBT-aligned HRNet coordinates/confidence and H36M camera K,R,t",
            "candidate_generation": "per-joint random subset uniformly over sizes 2..K",
            "hypotheses": args.hyps,
            "seed": args.seed,
            "triangulation_weight": args.triangulation_weight,
            "target_units": "metres internally, millimetres in metrics",
            "metric": "action-equal absolute MPJPE All-17; all camera tasks averaged",
            "labels_used_for_generation": False,
        },
        "source": str(source),
        "compare_cache": str(cache_path),
        "groups": len(arrays["targets"]),
        "elapsed_seconds": time.time() - start_time,
        "summary": summary,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output / "COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
