#!/usr/bin/env python3
"""Run official MMPose HRNet-W32 detections on a shard of H36M validation."""

import argparse
import pickle
from pathlib import Path

import numpy as np
from mmpose.apis import inference_topdown, init_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    model = init_model(args.config, args.checkpoint, device=args.device)
    predictions = []
    selected = range(args.shard_id, len(records), args.num_shards)
    for done, index in enumerate(selected, start=1):
        record = records[index]
        source_image = record.get("source_image", record["image"])
        image = str(Path(args.images_root) / record["image"])
        if not Path(image).is_file():
            image = str(Path(args.images_root) / source_image)
        if not Path(image).is_file():
            raise FileNotFoundError(
                f"missing image for record {index}: "
                f"tried {record['image']} and {source_image}"
            )
        bbox = np.asarray(record["box"], dtype=np.float32)[None]
        samples = inference_topdown(model, image, bboxes=bbox)
        if len(samples) != 1:
            raise RuntimeError(f"{record['image']}: expected one pose, got {len(samples)}")
        instances = samples[0].pred_instances
        predictions.append(
            (
                index,
                np.asarray(instances.keypoints[0], dtype=np.float32),
                np.asarray(instances.keypoint_scores[0], dtype=np.float32),
            )
        )
        if done % 250 == 0:
            print(f"shard {args.shard_id}: {done} / {len(range(args.shard_id, len(records), args.num_shards))}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {len(predictions)} predictions to {output}")


if __name__ == "__main__":
    main()
