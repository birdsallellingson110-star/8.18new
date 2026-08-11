"""Validate strict H36M MHP stage_V files before training on them.

Accepts either a final split pkl or a temp checkpoint. The checks are designed
to catch expensive-but-silent failures: wrong joint count, wrong view count,
missing cameras, NaNs, empty detections, or implausible H36M room placement.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np


REQUIRED_KEYS = [
    "joints_3d",
    "joints_2d_mmpose",
    "confs_2d_mmpose",
    "joints_2d_amass",
    "triangulated_3d_mmpose",
    "camera_setup_used",
    "views_used",
    "camera_parameters_all",
]

CAM_KEYS = ["K", "R", "T", "t", "fx", "fy", "cx", "cy"]


def as_array(v):
    return np.asarray(v)


def fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def check_file(path: str, min_samples: int | None = None, temp_ok: bool = False) -> None:
    with open(path, "rb") as f:
        d = pickle.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        fail(f"{path}: missing keys {missing}")

    j3 = as_array(d["joints_3d"])
    j2 = as_array(d["joints_2d_mmpose"])
    cf = as_array(d["confs_2d_mmpose"])
    j2gt = as_array(d["joints_2d_amass"])
    tri = as_array(d["triangulated_3d_mmpose"])
    views = as_array(d["views_used"])
    setup = as_array(d["camera_setup_used"])
    cams = d["camera_parameters_all"]

    n = j3.shape[0]
    if min_samples is not None and n < min_samples:
        fail(f"{path}: only {n} samples, expected at least {min_samples}")
    if n == 0:
        fail(f"{path}: empty file")
    if j3.shape != (n, 17, 3):
        fail(f"{path}: bad joints_3d shape {j3.shape}, expected (N,17,3)")
    if j2.shape != (n, 20, 17, 2):
        fail(f"{path}: bad joints_2d_mmpose shape {j2.shape}, expected (N,20,17,2)")
    if cf.shape != (n, 20, 17, 1):
        fail(f"{path}: bad confs_2d_mmpose shape {cf.shape}, expected (N,20,17,1)")
    if j2gt.shape != (n, 20, 17, 2):
        fail(f"{path}: bad joints_2d_amass shape {j2gt.shape}, expected (N,20,17,2)")
    if tri.shape != (n, 17, 3):
        fail(f"{path}: bad triangulated_3d_mmpose shape {tri.shape}, expected (N,17,3)")
    if views.shape != (n, 20):
        fail(f"{path}: bad views_used shape {views.shape}, expected (N,20)")
    if setup.shape != (n,):
        fail(f"{path}: bad camera_setup_used shape {setup.shape}, expected (N,)")
    if len(cams) != n:
        fail(f"{path}: camera_parameters_all len {len(cams)} != N {n}")

    for name, arr in [
        ("joints_3d", j3),
        ("joints_2d_mmpose", j2),
        ("confs_2d_mmpose", cf),
        ("joints_2d_amass", j2gt),
        ("triangulated_3d_mmpose", tri),
    ]:
        if not np.isfinite(arr).all():
            fail(f"{path}: non-finite values in {name}")

    if np.nanmax(cf) <= 0:
        fail(f"{path}: all 2D confidences are zero")
    if np.nanmean(cf > 0.05) < 0.15:
        fail(f"{path}: too few usable 2D confidences, mean valid ratio={np.nanmean(cf > 0.05):.3f}")

    # H36M MHP room placement: pelvis x/y should be sampled inside configured room.
    root_xy = j3[:, 0, :2]
    if root_xy[:, 0].min() < -1.25 or root_xy[:, 0].max() > 1.25:
        fail(f"{path}: pelvis x outside expected H36M room range {root_xy[:,0].min():.3f}..{root_xy[:,0].max():.3f}")
    if root_xy[:, 1].min() < -1.75 or root_xy[:, 1].max() > 2.25:
        fail(f"{path}: pelvis y outside expected H36M room range {root_xy[:,1].min():.3f}..{root_xy[:,1].max():.3f}")

    # Random camera view IDs should be 1..20 for H36M random-camera MHP.
    first_views = views[0].tolist()
    if first_views != list(range(1, 21)):
        fail(f"{path}: unexpected views_used[0]={first_views}")

    for i in range(min(n, 5)):
        if not isinstance(cams[i], list) or len(cams[i]) != 20:
            fail(f"{path}: camera sample {i} is not a 20-camera list")
        for ci, cam in enumerate(cams[i]):
            miss = [k for k in CAM_KEYS if k not in cam]
            if miss:
                fail(f"{path}: camera sample {i} view {ci} missing keys {miss}")
            if np.asarray(cam["K"]).shape != (3, 3):
                fail(f"{path}: camera sample {i} view {ci} bad K shape")
            if np.asarray(cam["R"]).shape != (3, 3):
                fail(f"{path}: camera sample {i} view {ci} bad R shape")

    kind = "temp" if temp_ok else "final"
    print(
        f"[OK] {kind} {path}: N={n}, "
        f"j3={j3.shape}, j2={j2.shape}, conf_valid={np.mean(cf > 0.05):.3f}, "
        f"pelvis_xy=({root_xy[:,0].min():.3f},{root_xy[:,0].max():.3f})/"
        f"({root_xy[:,1].min():.3f},{root_xy[:,1].max():.3f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="pkl path(s) or glob(s)")
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--temp-ok", action="store_true")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        matches = glob.glob(p)
        files.extend(matches if matches else [p])
    if not files:
        fail("no files matched")
    for path in files:
        if not os.path.exists(path):
            fail(f"{path}: does not exist")
        check_file(path, min_samples=args.min_samples, temp_ok=args.temp_ok)


if __name__ == "__main__":
    main()
