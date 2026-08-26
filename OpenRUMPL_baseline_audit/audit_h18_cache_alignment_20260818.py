#!/usr/bin/env python3
"""Check that H18 cache and temporal-window metadata use identical frame keys."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def group_keys(path: str):
    records = pickle.load(open(path, "rb"))
    groups = {}
    order = []
    for record in records:
        key = (
            int(record["subject"]), int(record["action"]),
            int(record["subaction"]), int(record["image_id"]),
        )
        if key not in groups:
            groups[key] = set()
            order.append(key)
        groups[key].add(int(record["camera_id"]))
    return order, groups


def check(name, pkl_a, pkl_b, cache):
    print(f"[{name}] loading metadata", flush=True)
    a_order, a_groups = group_keys(pkl_a)
    b_order, b_groups = group_keys(pkl_b)
    source = np.load(cache, allow_pickle=False)
    subjects = source["subjects"]
    actions = source["actions"]
    frames = source["group_indices"]
    print(
        f"  pkl_a groups={len(a_order)} pkl_b groups={len(b_order)} "
        f"cache={len(subjects)} sequential_group_indices="
        f"{np.array_equal(frames, np.arange(len(frames)))}",
        flush=True,
    )
    print(
        f"  camera completeness: a={all(v == {0,1,2,3} for v in a_groups.values())} "
        f"b={all(v == {0,1,2,3} for v in b_groups.values())}",
        flush=True,
    )
    same_order = a_order == b_order
    print(f"  pkl order identical={same_order}", flush=True)
    if not same_order:
        for i, (left, right) in enumerate(zip(a_order, b_order)):
            if left != right:
                print(f"  first pkl mismatch index={i}: {left} != {right}")
                break
    mismatches = []
    for i, key in enumerate(a_order[:len(subjects)]):
        if (int(subjects[i]), int(actions[i])) != key[:2]:
            mismatches.append((i, (int(subjects[i]), int(actions[i])), key))
            if len(mismatches) == 3:
                break
    print(f"  cache subject/action mismatches={mismatches}", flush=True)
    return same_order and not mismatches


def main():
    train_ok = check(
        "train",
        "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl",
        "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl",
        "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2/train_c2_22c.npz",
    )
    val_ok = check(
        "validation H15 cache vs H18 window pkl",
        "/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets_mmpose/annot_temporal_5_5_gbt_yolox_x_score001_fallback_legswap/h36m_validation.pkl",
        "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation/merged/h36m_validation.pkl",
        "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260818/h15_temporal_c2_oracle/validation_c2_22c.npz",
    )
    raise SystemExit(0 if train_ok and val_ok else 1)


if __name__ == "__main__":
    main()
