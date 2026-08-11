#!/usr/bin/env python3
"""Export compact HRNet heatmap candidates for real Human3.6M images.

The existing RUMPL PKLs retain only one decoded coordinate and confidence per
joint.  This exporter keeps several local heatmap modes so an epipolar module
can recover a geometrically consistent candidate that was not the monocular
argmax.  Coordinates are stored in the original image coordinate system.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from mmpose.apis import inference_topdown, init_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--record-indices",
        type=int,
        nargs="+",
        help=(
            "Optional explicit PKL record indices.  Useful for exporting "
            "complete multi-view smoke-test groups."
        ),
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--nms-kernel", type=int, default=5)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many records in the selected shard (0=all).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dense-output",
        help="Optional float16 .npy memmap containing every dense heatmap.",
    )
    return parser.parse_args()


def refine_quarter_pixel(
    heatmaps: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the quarter-pixel refinement used by MMPose MSRAHeatmap."""
    _, height, width = heatmaps.shape
    joint_ids = torch.arange(heatmaps.shape[0], device=heatmaps.device)[:, None]
    valid_x = (xs > 1) & (xs < width - 1) & (ys > 0) & (ys < height)
    valid_y = (ys > 1) & (ys < height - 1) & (xs > 0) & (xs < width)
    safe_x = xs.clamp(1, width - 2)
    safe_y = ys.clamp(1, height - 2)
    dx = (
        heatmaps[joint_ids, safe_y, safe_x + 1]
        - heatmaps[joint_ids, safe_y, safe_x - 1]
    )
    dy = (
        heatmaps[joint_ids, safe_y + 1, safe_x]
        - heatmaps[joint_ids, safe_y - 1, safe_x]
    )
    refined_x = xs.float() + torch.sign(dx) * 0.25 * valid_x
    refined_y = ys.float() + torch.sign(dy) * 0.25 * valid_y
    return refined_x, refined_y


def sparse_candidates(
    heatmaps: torch.Tensor,
    input_size: np.ndarray,
    input_center: np.ndarray,
    input_scale: np.ndarray,
    topk: int,
    nms_kernel: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find local heatmap modes and transform them to original image pixels."""
    if nms_kernel % 2 != 1:
        raise ValueError("--nms-kernel must be odd")
    if heatmaps.ndim != 3:
        raise ValueError(f"expected KxHxW heatmaps, got {heatmaps.shape}")
    pooled = functional.max_pool2d(
        heatmaps[None],
        kernel_size=nms_kernel,
        stride=1,
        padding=nms_kernel // 2,
    )[0]
    local_modes = torch.where(heatmaps == pooled, heatmaps, -torch.inf)
    flat = local_modes.flatten(1)
    scores, indices = torch.topk(flat, k=min(topk, flat.shape[1]), dim=1)
    height, width = heatmaps.shape[-2:]
    xs = indices % width
    ys = torch.div(indices, width, rounding_mode="floor")
    refined_x, refined_y = refine_quarter_pixel(heatmaps, xs, ys)
    heatmap_xy = torch.stack((refined_x, refined_y), dim=-1).cpu().numpy()

    input_size = np.asarray(input_size, dtype=np.float32)
    input_center = np.asarray(input_center, dtype=np.float32)
    input_scale = np.asarray(input_scale, dtype=np.float32)
    heatmap_size = np.asarray([width, height], dtype=np.float32)
    input_xy = heatmap_xy / heatmap_size * input_size
    image_xy = (
        input_xy / input_size * input_scale
        + input_center
        - 0.5 * input_scale
    )
    return image_xy.astype(np.float32), scores.cpu().numpy().astype(np.float16)


def resolve_image(record: dict, images_root: Path) -> Path:
    candidates = (
        record["image"],
        record.get("source_image", record["image"]),
    )
    for relative_path in candidates:
        path = images_root / relative_path
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing image; tried: {candidates}")


def main() -> None:
    args = parse_args()
    if args.topk < 1:
        raise ValueError("--topk must be positive")
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)

    model = init_model(args.config, args.checkpoint, device=args.device)
    model.test_cfg["output_heatmaps"] = True
    images_root = Path(args.images_root)
    if args.record_indices is not None:
        selected = list(args.record_indices)
        invalid = [
            index for index in selected
            if index < 0 or index >= len(records)
        ]
        if invalid:
            raise IndexError(f"invalid --record-indices: {invalid[:10]}")
    else:
        selected = list(range(args.shard_id, len(records), args.num_shards))
    if args.limit:
        selected = selected[: args.limit]

    indices = []
    decoded_keypoints = []
    decoded_scores = []
    candidate_xy = []
    candidate_scores = []
    input_centers = []
    input_scales = []
    input_sizes = []
    top1_differences = []
    heatmap_shape = None
    dense_memmap = None
    dense_temporary = None

    for done, index in enumerate(selected, start=1):
        record = records[index]
        image = resolve_image(record, images_root)
        bbox = np.asarray(record["box"], dtype=np.float32)[None]
        samples = inference_topdown(model, str(image), bboxes=bbox)
        if len(samples) != 1:
            raise RuntimeError(f"{image}: expected one pose, got {len(samples)}")
        sample = samples[0]
        heatmaps = sample.pred_fields.heatmaps.detach()
        heatmap_shape = tuple(int(value) for value in heatmaps.shape)
        if args.dense_output and dense_memmap is None:
            dense_output = Path(args.dense_output)
            dense_output.parent.mkdir(parents=True, exist_ok=True)
            dense_temporary = dense_output.with_suffix(
                dense_output.suffix + ".tmp"
            )
            dense_memmap = np.lib.format.open_memmap(
                dense_temporary,
                mode="w+",
                dtype=np.float16,
                shape=(len(selected), *heatmap_shape),
            )
        if dense_memmap is not None:
            dense_memmap[done - 1] = (
                heatmaps.detach().cpu().numpy().astype(np.float16)
            )
        xy, heatmap_scores = sparse_candidates(
            heatmaps,
            sample.metainfo["input_size"],
            sample.metainfo["input_center"],
            sample.metainfo["input_scale"],
            args.topk,
            args.nms_kernel,
        )
        keypoints = np.asarray(
            sample.pred_instances.keypoints[0], dtype=np.float32
        )
        scores = np.asarray(
            sample.pred_instances.keypoint_scores[0], dtype=np.float32
        )
        top1_differences.append(
            np.linalg.norm(xy[:, 0] - keypoints, axis=-1)
        )
        indices.append(index)
        decoded_keypoints.append(keypoints)
        decoded_scores.append(scores)
        candidate_xy.append(xy)
        candidate_scores.append(heatmap_scores)
        input_centers.append(
            np.asarray(sample.metainfo["input_center"], dtype=np.float32)
        )
        input_scales.append(
            np.asarray(sample.metainfo["input_scale"], dtype=np.float32)
        )
        input_sizes.append(
            np.asarray(sample.metainfo["input_size"], dtype=np.float32)
        )
        if done % 250 == 0:
            print(
                f"shard {args.shard_id}: {done}/{len(selected)}",
                flush=True,
            )

    arrays = {
        "record_indices": np.asarray(indices, dtype=np.int32),
        "decoded_keypoints": np.asarray(decoded_keypoints, dtype=np.float32),
        "decoded_scores": np.asarray(decoded_scores, dtype=np.float32),
        "candidate_xy": np.asarray(candidate_xy, dtype=np.float32),
        "candidate_scores": np.asarray(candidate_scores, dtype=np.float16),
        # These preserve the exact MMPose crop transform.  A dense cross-view
        # fusion layer needs them to map an epipolar line in original image
        # pixels to the corresponding HRNet heatmap, without rerunning image
        # preprocessing or approximating the bbox convention.
        "input_center": np.asarray(input_centers, dtype=np.float32),
        "input_scale": np.asarray(input_scales, dtype=np.float32),
        "input_size": np.asarray(input_sizes, dtype=np.float32),
    }
    differences = np.asarray(top1_differences, dtype=np.float32)
    validation = {
        "records": len(indices),
        "heatmap_shape": heatmap_shape,
        "topk": args.topk,
        "nms_kernel": args.nms_kernel,
        "top1_vs_mmpose_mean_px": float(differences.mean()),
        "top1_vs_mmpose_max_px": float(differences.max()),
        "top1_vs_mmpose_p99_px": float(np.quantile(differences, 0.99)),
        "top1_vs_mmpose_over_0_5px": int((differences > 0.5).sum()),
        "top1_vs_mmpose_over_0_5px_fraction": float(
            (differences > 0.5).mean()
        ),
    }
    # Rare boundary/tie cases can follow a different quarter-pixel branch
    # even when the exported heatmap and virtually every decoded joint match.
    # Official MMPose coordinates are stored separately as the identity
    # baseline, so reject systematic drift rather than one isolated maximum.
    if (
        validation["top1_vs_mmpose_mean_px"] > 1e-3
        or validation["top1_vs_mmpose_p99_px"] > 1e-2
        or validation["top1_vs_mmpose_over_0_5px_fraction"] > 1e-4
    ):
        raise RuntimeError(
            "manual heatmap decoding does not reproduce MMPose: "
            f"{validation}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    metadata = output.with_suffix(output.suffix + ".json")
    metadata.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if dense_memmap is not None:
        dense_memmap.flush()
        del dense_memmap
        Path(dense_temporary).replace(Path(args.dense_output))
        print(f"saved dense heatmaps: {args.dense_output}", flush=True)
    print(json.dumps(validation, indent=2), flush=True)
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
