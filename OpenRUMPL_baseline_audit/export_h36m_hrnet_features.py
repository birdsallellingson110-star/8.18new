#!/usr/bin/env python3
"""Export frozen HRNet stage-4 features for a H36M PKL shard.

Unlike the existing heatmap exporter this intentionally stops after
``model.backbone``.  All four HRNet branches are resized to the high
resolution branch and concatenated (32+64+128+256=480 channels), then stored
as float16.  This is the feature input required by the next MVGFormer-style
epipolar refiner; no detector weights or crop convention are changed.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmpose.apis import inference_topdown, init_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-pkl", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--record-indices", type=int, nargs="+")
    p.add_argument(
        "--limit-groups", type=int, default=0,
        help="Export the first N four-view groups (one camera per shard).",
    )
    p.add_argument(
        "--group-indices-file", default="",
        help="Optional .npy/.txt file of complete-group indices to export in file order.",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--feature-output", required=True)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def resolve_image(record, root: Path) -> Path:
    candidates = [record["image"], record.get("source_image", record["image"])]
    for item in candidates:
        path = root / item
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing image: {candidates}")


def main():
    args = parse_args()
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    if args.record_indices is not None:
        selected = list(args.record_indices)
    elif args.group_indices_file:
        group_index_path = Path(args.group_indices_file)
        if group_index_path.suffix == ".npy":
            requested_groups = np.asarray(np.load(group_index_path), dtype=np.int64).reshape(-1)
        else:
            requested_groups = np.asarray(
                [int(line.strip()) for line in group_index_path.read_text().splitlines() if line.strip()],
                dtype=np.int64,
            )
        groups = {}
        for index, record in enumerate(records):
            key = (
                int(record["subject"]), int(record["action"]),
                int(record["subaction"]), int(record["image_id"]),
            )
            groups.setdefault(key, [-1] * 4)[int(record["camera_id"])] = index
        complete = [group for group in groups.values() if min(group) >= 0]
        if len(requested_groups) == 0:
            raise ValueError(f"empty group-indices-file: {group_index_path}")
        if np.any(requested_groups < 0) or np.any(requested_groups >= len(complete)):
            raise ValueError(
                f"group index outside complete-group range [0,{len(complete)}): "
                f"{requested_groups.min()}..{requested_groups.max()}"
            )
        selected_groups = [complete[int(group_id)] for group_id in requested_groups]
        if args.shard_id >= 4:
            raise ValueError("--group-indices-file expects four camera shards")
        selected = [group[args.shard_id] for group in selected_groups]
    elif args.limit_groups:
        groups = {}
        for index, record in enumerate(records):
            key = (
                int(record["subject"]), int(record["action"]),
                int(record["subaction"]), int(record["image_id"]),
            )
            groups.setdefault(key, [-1] * 4)[int(record["camera_id"])] = index
        complete = [group for group in groups.values() if min(group) >= 0]
        complete = complete[:args.limit_groups]
        if args.shard_id >= 4:
            raise ValueError("--limit-groups expects four camera shards")
        selected = [group[args.shard_id] for group in complete]
    else:
        selected = list(range(args.shard_id, len(records), args.num_shards))
    if args.limit:
        selected = selected[:args.limit]
    model = init_model(args.config, args.checkpoint, device=args.device)
    model.eval()
    root = Path(args.images_root)
    feature_memmap = None
    Path(args.feature_output).parent.mkdir(parents=True, exist_ok=True)
    output_indices, output_keypoints, output_scores = [], [], []
    centers, scales, sizes = [], [], []
    from mmengine.dataset import Compose, pseudo_collate
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    captured = []
    def capture(module, inputs, output):
        captured.append(output)
    hook = model.backbone.register_forward_hook(capture)
    with torch.no_grad():
        for done, index in enumerate(selected, 1):
            record = records[index]
            captured.clear()
            bbox = np.asarray(record["box"], dtype=np.float32)[None]
            image_path = str(resolve_image(record, root))
            samples = inference_topdown(model, image_path, bboxes=bbox)
            if len(samples) != 1:
                raise RuntimeError(f"{index}: expected one pose sample")
            sample = samples[0]
            # inference_topdown has already normalized/cropped the input.  Use
            # the exact tensor that the model saw so feature coordinates share
            # the existing input_center/input_scale metadata.
            # ``PoseDataSample`` contains metadata/predictions but not the
            # normalized crop tensor.  Re-run the exact test pipeline for this
            # single record, then pass its preprocessed tensor through HRNet.
            data_info = {
                "img_path": image_path,
                "bbox": np.asarray(record["box"], dtype=np.float32)[None],
                "bbox_score": np.ones(1, dtype=np.float32),
            }
            data_info.update(model.dataset_meta)
            processed = pipeline(data_info)
            batch = pseudo_collate([processed])
            prepared = model.data_preprocessor(batch, False)
            inputs = prepared["inputs"]
            # The hook captures the same feature tuple produced by the one
            # backbone pass inside inference_topdown.  Re-running the data
            # preprocessor here is cheap; the expensive HRNet forward is not
            # repeated.
            if not captured:
                raise RuntimeError(f"{index}: HRNet feature hook did not fire")
            # test_cfg.flip_test=True invokes the backbone twice; use the
            # original (non-flipped) branch, matching the first heatmap pass.
            branches = captured[0]
            high = branches[0]
            resized = [high]
            for branch in branches[1:]:
                resized.append(F.interpolate(branch, size=high.shape[-2:], mode="bilinear", align_corners=False))
            features = torch.cat(resized, dim=1).squeeze(0).cpu().numpy().astype(np.float16)
            if feature_memmap is None:
                feature_memmap = np.lib.format.open_memmap(
                    Path(args.feature_output).with_suffix(Path(args.feature_output).suffix + ".tmp"),
                    mode="w+", dtype=np.float16, shape=(len(selected), *features.shape),
                )
            feature_memmap[done - 1] = features
            output_indices.append(index)
            output_keypoints.append(np.asarray(sample.pred_instances.keypoints[0], dtype=np.float32))
            output_scores.append(np.asarray(sample.pred_instances.keypoint_scores[0], dtype=np.float32))
            centers.append(np.asarray(sample.metainfo["input_center"], dtype=np.float32))
            scales.append(np.asarray(sample.metainfo["input_scale"], dtype=np.float32))
            sizes.append(np.asarray(sample.metainfo["input_size"], dtype=np.float32))
            if done % 100 == 0:
                print(f"shard {args.shard_id}: {done}/{len(selected)}", flush=True)
    hook.remove()
    if feature_memmap is None:
        raise RuntimeError("no records selected")
    feature_memmap.flush()
    Path(feature_memmap.filename).replace(args.feature_output)
    output = {
        "record_indices": np.asarray(output_indices, dtype=np.int32),
        "decoded_keypoints": np.stack(output_keypoints),
        "decoded_scores": np.stack(output_scores),
        "input_center": np.stack(centers), "input_scale": np.stack(scales),
        "input_size": np.stack(sizes),
    }
    np.savez_compressed(args.output, **output)
    metadata = {"records": len(selected), "feature_shape": list(feature_memmap.shape[1:]),
                "feature_output": str(Path(args.feature_output).resolve())}
    Path(str(args.output) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
