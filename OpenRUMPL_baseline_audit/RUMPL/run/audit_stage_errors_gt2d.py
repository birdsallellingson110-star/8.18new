#!/usr/bin/env python3
"""Stage-wise 3D error audit under GT/org 2D (or mmpose).

Pipeline stages measured (absolute MPJPE in mm unless noted):
  1) ray_to_gt      : mean distance of GT joint to each view ray (2D+calib consistency)
  2) ray_oracle     : mean over views of closest-point-on-ray to GT, vs GT
                     (best 3D if you only pick depth on each noisy ray, then average)
  3) mid_geom       : skew-line midpoint (dataset middle_points) vs GT
  4) rumpl_pred     : RUMPL network output vs GT
  5) pred_vs_mid    : ||pred - mid|| (how far network moves from geometry)
  Also root-relative (joint0) versions of mid/pred.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

# Match valid_rumpl.py import path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _init_paths  # noqa: E402,F401
from core.config import config, update_config  # noqa: E402
import dataset  # noqa: E402
import models  # noqa: E402


KP_STAR = (5, 6, 7, 8, 9, 10, 13, 14, 15, 16)
JOINT_NAMES = [
    "nose", "leye", "reye", "lear", "rear",
    "lsho", "rsho", "lelb", "relb", "lwri", "rwri",
    "lhip", "rhip", "lkne", "rkne", "lank", "rank",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--model-file", required=True)
    p.add_argument("--n-views", type=int, required=True)
    p.add_argument("--gpus", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--use-mmpose", action="store_true",
                   help="Use mmpose 2D instead of GT/org (default: GT/org)")
    p.add_argument("--max-batches", type=int, default=0,
                   help="0 = full val set")
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def mpjpe_mm(a, b):
    """a,b: (N,J,3) meters -> (N,J) mm"""
    return np.linalg.norm(a - b, axis=-1) * 1000.0


def root_align(x, root=0):
    return x - x[:, root:root + 1, :]


def ray_point_distance(origins, dirs, points):
    """origins/dirs: (..., V, 3), points: (..., 3) -> dist (..., V)"""
    d = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-12)
    v = points[..., None, :] - origins
    t = np.sum(v * d, axis=-1, keepdims=True)
    closest = origins + t * d
    return np.linalg.norm(points[..., None, :] - closest, axis=-1), closest, t[..., 0]


def pairwise_midpoints(origins, dirs):
    """Robust multi-view geometry: mean of all 2-view closest-midpoints.

    Dataset middle_points use n-line solve which is ill-conditioned for V>2
    (can explode to tens of meters). Pairwise average stays stable.
    origins/dirs: (B, J, V, 3) -> mid (B, J, 3)
    """
    b, j, v, _ = origins.shape
    d = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-12)
    mids = []
    for i in range(v):
        for k in range(i + 1, v):
            o1, o2 = origins[:, :, i], origins[:, :, k]
            d1, d2 = d[:, :, i], d[:, :, k]
            w0 = o1 - o2
            a = np.sum(d1 * d1, axis=-1)
            bdot = np.sum(d1 * d2, axis=-1)
            c = np.sum(d2 * d2, axis=-1)
            d_ = np.sum(d1 * w0, axis=-1)
            e = np.sum(d2 * w0, axis=-1)
            denom = a * c - bdot * bdot + 1e-12
            t1 = (bdot * e - c * d_) / denom
            t2 = (a * e - bdot * d_) / denom
            p1 = o1 + t1[..., None] * d1
            p2 = o2 + t2[..., None] * d2
            mids.append(0.5 * (p1 + p2))
    return np.mean(np.stack(mids, axis=0), axis=0)


def summarize(err_nj, name=""):
    return {
        "all17_mm": float(err_nj.mean()),
        "kpstar_mm": float(err_nj[:, KP_STAR].mean()),
        "per_joint_mm": {
            JOINT_NAMES[j]: float(err_nj[:, j].mean()) for j in range(err_nj.shape[1])
        },
    }


def main():
    args = parse_args()
    update_config(args.cfg)
    config.GPUS = args.gpus
    config.WORKERS = args.workers
    config.TEST.MODEL_FILE = args.model_file
    config.DATASET.USE_MMPOSE_VAL = bool(args.use_mmpose)
    config.TEST.BATCH_SIZE = args.batch_size
    # Build/load with training view width first (weighted_mean is view-sized),
    # then switch test combination size — same order as valid_rumpl.py.

    # kill experimental env side-effects
    for k in [
        "RUMPL_KPA", "RUMPL_MULTI_HYP", "RUMPL_POSE_CODEBOOK", "RUMPL_2D_REFINE",
        "GBT_LEARNABLE_BIAS", "GBT_ORACLE_RELIABILITY", "GBT_LEARNED_RELIABILITY",
        "RUMPL_GLOBAL_JOINT_VIEW_FUSION", "RUMPL_TRAIN_STRUCT_OCC", "RUMPL_OCC_JOINT_LOSS",
    ]:
        os.environ[k] = "0"

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval("models." + config.MODEL + ".get_multiview_rumpl_net")(
        config, is_train=False
    )
    state = torch.load(args.model_file, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    gpus = [int(i) for i in config.GPUS.split(",")]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model.eval()

    config.DATASET.N_VIEWS_TRAIN_TEST_ALL = args.n_views

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    ds = eval("dataset." + config.DATASET.TEST_DATASET)(
        config,
        config.DATASET.TEST_SUBSET,
        False,
        transforms.Compose([transforms.ToTensor(), normalize]),
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
    )

    buckets = {
        "ray_to_gt": [],
        "ray_oracle": [],
        "mid_abs": [],          # dataset n-line mid (V=2 OK; V>2 may explode)
        "pair_mid_abs": [],     # robust pairwise mid
        "pair_mid_rel": [],
        "mid_rel": [],
        "pred_abs": [],
        "pred_rel": [],
        "pred_vs_pair_mid": [],
        "tri_improve": [],  # pair_mid - pred (positive => network better than geom)
    }

    n_seen = 0
    with torch.no_grad():
        for bi, (middle_points, closest_points_all, target, rays, meta, joints_2ds) in enumerate(
            loader
        ):
            if args.max_batches and bi >= args.max_batches:
                break
            if config.NETWORK.FEED_CAMERA_CALIBRATION or config.NETWORK.FEED_ONLY_2D:
                output = model(joints_2ds, is_training=False)
            else:
                output = model(rays, is_training=False)

            pred = output.detach().cpu().numpy()  # (B,17,3)
            gt = target.detach().cpu().numpy()
            mid = middle_points.detach().cpu().numpy()[:, :, 0, :]  # (B,17,3)
            r = rays.detach().cpu().numpy()
            dirs = r[..., 0:3]
            origins = r[..., 3:6]

            dist, closest, _ = ray_point_distance(origins, dirs, gt)
            # ray_to_gt: mean over views
            buckets["ray_to_gt"].append(dist.mean(axis=-1) * 1000.0)
            # ray_oracle: mean closest-on-ray across views
            oracle = closest.mean(axis=-2)
            buckets["ray_oracle"].append(mpjpe_mm(oracle, gt))

            pair_mid = pairwise_midpoints(origins, dirs)
            buckets["mid_abs"].append(mpjpe_mm(mid, gt))
            buckets["mid_rel"].append(mpjpe_mm(root_align(mid), root_align(gt)))
            buckets["pair_mid_abs"].append(mpjpe_mm(pair_mid, gt))
            buckets["pair_mid_rel"].append(mpjpe_mm(root_align(pair_mid), root_align(gt)))
            buckets["pred_abs"].append(mpjpe_mm(pred, gt))
            buckets["pred_rel"].append(mpjpe_mm(root_align(pred), root_align(gt)))
            buckets["pred_vs_pair_mid"].append(mpjpe_mm(pred, pair_mid))
            buckets["tri_improve"].append(
                mpjpe_mm(pair_mid, gt) - mpjpe_mm(pred, gt)
            )
            n_seen += pred.shape[0]
            if (bi + 1) % 20 == 0:
                print(f"[stage-audit] batch {bi+1} samples={n_seen}", flush=True)

    stacked = {k: np.concatenate(v, axis=0) for k, v in buckets.items()}
    result = {
        "n_views": args.n_views,
        "n_samples": int(n_seen),
        "input_2d": "mmpose" if args.use_mmpose else "gt_org",
        "model_file": os.path.abspath(args.model_file),
        "stages": {
            "1_ray_to_gt_mm": summarize(stacked["ray_to_gt"]),
            "2_ray_oracle_avg_closest_mm": summarize(stacked["ray_oracle"]),
            "3_mid_geom_abs_mm": summarize(stacked["mid_abs"]),
            "3b_mid_geom_rel_mm": summarize(stacked["mid_rel"]),
            "4_rumpl_pred_abs_mm": summarize(stacked["pred_abs"]),
            "4b_rumpl_pred_rel_mm": summarize(stacked["pred_rel"]),
            "5_pred_vs_mid_mm": summarize(stacked["pred_vs_mid"]),
            "network_improves_geom_mm": {
                "mean_mid_minus_pred": float(stacked["tri_improve"].mean()),
                "frac_joints_network_better": float(
                    (stacked["tri_improve"] > 0).mean()
                ),
            },
        },
        "interpretation_hints": [
            "1_ray_to_gt ~0 => GT/org 2D + calib consistent with GT 3D",
            "2_ray_oracle = residual if only depths on rays (view-avg)",
            "3_mid_geom = skew-line midpoint triangulation proxy",
            "4_rumpl_pred = final network",
            "If 3>>4, network helps geometry; if 4≈3, network≈triangulation; if 4>3, network hurts vs geom",
            "Gap (4 under mmpose) - (4 under gt) ≈ 2D-driven; remaining 4 under gt ≈ non-2D",
        ],
    }

    # compact table for stdout
    s = result["stages"]
    print("\n=== STAGE ERRORS (All-17 mm) ===")
    order = [
        ("1 ray→GT dist", s["1_ray_to_gt_mm"]),
        ("2 ray-oracle", s["2_ray_oracle_avg_closest_mm"]),
        ("3 mid-geom abs", s["3_mid_geom_abs_mm"]),
        ("3 mid-geom rel", s["3b_mid_geom_rel_mm"]),
        ("4 RUMPL abs", s["4_rumpl_pred_abs_mm"]),
        ("4 RUMPL rel", s["4b_rumpl_pred_rel_mm"]),
        ("5 |pred-mid|", s["5_pred_vs_mid_mm"]),
    ]
    for name, m in order:
        print(f"{name:18s}  All={m['all17_mm']:7.3f}  KP*={m['kpstar_mm']:7.3f}")
    ni = s["network_improves_geom_mm"]
    print(
        f"network vs geom: mean(mid-pred)={ni['mean_mid_minus_pred']:+.3f} mm, "
        f"frac better={ni['frac_joints_network_better']:.3f}"
    )

    # top joints by pred abs
    pj = s["4_rumpl_pred_abs_mm"]["per_joint_mm"]
    top = sorted(pj.items(), key=lambda x: -x[1])[:6]
    print("worst joints (RUMPL abs):", ", ".join(f"{k}={v:.1f}" for k, v in top))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", args.output_json)


if __name__ == "__main__":
    main()
