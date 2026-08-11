#!/usr/bin/env python3
"""Run the public RUMPL full-image MMPoseInferencer pipeline on H36M."""

import argparse
import pickle
from pathlib import Path

import numpy as np
from mmpose.apis import MMPoseInferencer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)

    inferencer = MMPoseInferencer(
        pose2d=args.pose_config,
        pose2d_weights=args.pose_checkpoint,
        device=args.device,
        det_model=args.det_config,
        det_weights=args.det_checkpoint,
        det_cat_ids=(0,),
        show_progress=False,
    )
    predictions = []
    selected = range(args.shard_id, len(records), args.num_shards)
    for done, index in enumerate(selected, start=1):
        record = records[index]
        image = Path(args.images_root) / record["image"]
        if not image.is_file():
            source_image = record.get("source_image", record["image"])
            image = Path(args.images_root) / source_image
        if not image.is_file():
            raise FileNotFoundError(f"missing image for record {index}: {image}")

        result = next(
            inferencer(str(image.resolve()), return_datasamples=False, show=False)
        )
        people = result["predictions"][0]
        if not people:
            raise RuntimeError(f"{record['image']}: full-image detector found no person")
        # This deliberately matches preprocess_h36m.py, which consumes
        # mmpose_data[0] when more than one detector proposal is returned.
        person = people[0]
        keypoints = np.asarray(person["keypoints"], dtype=np.float32)
        scores = np.asarray(person["keypoint_scores"], dtype=np.float32)
        if keypoints.shape != (17, 2) or scores.shape != (17,):
            raise RuntimeError(
                f"{record['image']}: unexpected shapes {keypoints.shape}/{scores.shape}"
            )
        predictions.append((index, keypoints, scores))
        if done % 100 == 0:
            print(
                f"shard {args.shard_id}: {done} / {len(selected)}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)
    print(f"wrote {len(predictions)} predictions to {output}")


if __name__ == "__main__":
    main()
