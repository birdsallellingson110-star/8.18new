#!/usr/bin/env python3
"""Merge MMPose shards and convert COCO-17 detections to RUMPL H36M-17."""

import argparse
import copy
import pickle
from pathlib import Path

import numpy as np


MMPOSE2H36M = {
    1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15, 9: 0,
    11: 5, 12: 7, 13: 9, 14: 6, 15: 8, 16: 10,
}


def convert(keypoints, scores, swap_lower_body=False):
    joints = np.zeros((17, 2), dtype=np.float32)
    confidence = np.zeros((17, 1), dtype=np.float32)
    for dst, src in MMPOSE2H36M.items():
        joints[dst] = keypoints[src]
        confidence[dst, 0] = scores[src]
    joints[10] = keypoints[0:5].mean(axis=0)
    confidence[10, 0] = scores[0:5].mean()
    joints[8] = keypoints[3:7].mean(axis=0)
    confidence[8, 0] = scores[3:7].mean()
    joints[0] = keypoints[11:13].mean(axis=0)
    confidence[0, 0] = scores[11:13].mean()
    joints[7] = (joints[8] + joints[0]) / 2
    confidence[7, 0] = (confidence[8, 0] + confidence[0, 0]) / 2
    if swap_lower_body:
        joints[1:4], joints[4:7] = joints[4:7].copy(), joints[1:4].copy()
        confidence[1:4], confidence[4:7] = (
            confidence[4:7].copy(),
            confidence[1:4].copy(),
        )
    return joints, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--swap-lower-body", action="store_true")
    args = parser.parse_args()
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    found = {}
    for shard in args.shards:
        if str(shard).endswith(".npz"):
            payload = np.load(shard)
            predictions = zip(
                payload["record_indices"],
                payload["decoded_keypoints"],
                payload["decoded_scores"],
            )
        else:
            with open(shard, "rb") as handle:
                predictions = pickle.load(handle)
        for index, keypoints, scores in predictions:
            index = int(index)
            if not 0 <= index < len(records):
                raise RuntimeError(f"{shard}: out-of-range record index {index}")
            if index in found:
                raise RuntimeError(f"{shard}: duplicate prediction for record {index}")
            keypoints = np.asarray(keypoints)
            scores = np.asarray(scores)
            if keypoints.shape != (17, 2) or scores.shape != (17,):
                raise RuntimeError(
                    f"{shard}: record {index} has shapes "
                    f"{keypoints.shape}/{scores.shape}, expected (17, 2)/(17,)"
                )
            if not np.isfinite(keypoints).all() or not np.isfinite(scores).all():
                raise RuntimeError(f"{shard}: non-finite prediction at record {index}")
            found[index] = (keypoints, scores)
    if len(found) != len(records):
        missing = sorted(set(range(len(records))) - set(found))
        raise RuntimeError(f"missing {len(missing)} predictions; first: {missing[:10]}")
    output_records = []
    for index, record in enumerate(records):
        updated = copy.deepcopy(record)
        updated["joints_2d"], updated["joints_2d_conf"] = convert(
            *found[index], swap_lower_body=args.swap_lower_body
        )
        output_records.append(updated)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("wb") as handle:
        pickle.dump(output_records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_output.replace(output)
    print(f"wrote {len(output_records)} records to {output}")


if __name__ == "__main__":
    main()
