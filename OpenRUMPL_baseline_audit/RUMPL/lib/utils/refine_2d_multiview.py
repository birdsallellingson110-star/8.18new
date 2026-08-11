"""Multi-view soft 2D / ray refine for RUMPL.

Env:
  RUMPL_2D_REFINE=1
  RUMPL_2D_REFINE_MODE=soft_fill   # soft | fill_only | soft_fill | hard
  RUMPL_2D_REFINE_STRENGTH=0.5     # max blend toward multi-view consensus
  RUMPL_2D_REFINE_CONF_THR=0.1     # views below this do not vote / are "occluded"
  RUMPL_2D_REFINE_FILL_CONF=0.35   # conf written back for filled occluded joints
  RUMPL_2D_REFINE_MIN_VIEWS=2

Design (avoids anchoring high-conf detections to weak triangulation):
  - Triangulate each joint from views with conf > thr (conf-weighted LS).
  - High-conf views: keep original ray (alpha ~ 0).
  - Low-conf views: soft-blend direction toward reprojected consensus.
  - Occluded views (conf < thr): optionally FILL direction + fill_conf.
"""

from __future__ import annotations

import os

import numpy as np

from utils.calib import closest_point_between_rays_batched


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def refine_enabled() -> bool:
    return os.environ.get("RUMPL_2D_REFINE", "0") == "1"


def soft_refine_rays(
    directions: np.ndarray,
    intersections: np.ndarray,
    confs: np.ndarray,
    *,
    mode: str | None = None,
    strength: float | None = None,
    conf_thr: float | None = None,
    fill_conf: float | None = None,
    min_views: int | None = None,
):
    """Refine per-joint multi-view rays.

    Args:
        directions: (J, V, 3) unit-ish direction vectors
        intersections: (J, V, 3) ray origins (cam centers when IntersectM)
        confs: (J, V, 1) or (J, V) detection confidences

    Returns:
        directions_out (J,V,3), confs_out (same shape as confs)
    """
    mode = mode or os.environ.get("RUMPL_2D_REFINE_MODE", "soft_fill")
    strength = _env_float("RUMPL_2D_REFINE_STRENGTH", 0.5) if strength is None else strength
    conf_thr = _env_float("RUMPL_2D_REFINE_CONF_THR", 0.1) if conf_thr is None else conf_thr
    fill_conf = _env_float("RUMPL_2D_REFINE_FILL_CONF", 0.35) if fill_conf is None else fill_conf
    min_views = _env_int("RUMPL_2D_REFINE_MIN_VIEWS", 2) if min_views is None else min_views

    directions = np.asarray(directions, dtype=np.float64)
    intersections = np.asarray(intersections, dtype=np.float64)
    confs_in = np.asarray(confs, dtype=np.float64)
    squeeze_conf = confs_in.ndim == 3 and confs_in.shape[-1] == 1
    conf = confs_in[..., 0] if squeeze_conf else confs_in.copy()  # (J, V)

    jn, vn, _ = directions.shape
    assert intersections.shape == directions.shape
    assert conf.shape == (jn, vn)

    dirs_out = directions.copy()
    conf_out = conf.copy()

    # Pre-normalize input directions for stable triangulation.
    dir_norm = np.linalg.norm(directions, axis=-1, keepdims=True)
    dir_safe = directions / np.clip(dir_norm, 1e-8, None)

    for j in range(jn):
        vote = conf[j] > conf_thr
        n_vote = int(vote.sum())
        if n_vote < min_views:
            continue

        origins = intersections[j, vote][None, ...]  # (1, Nv, 3)
        dirs = dir_safe[j, vote][None, ...]
        w = conf[j, vote][None, ...]
        # Guard against all-zero weights after mask (shouldn't happen).
        if float(w.sum()) <= 1e-8:
            continue
        try:
            x = closest_point_between_rays_batched(origins, dirs, w)[0]  # (3,)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(x)):
            continue

        for v in range(vn):
            origin = intersections[j, v]
            target = x - origin
            tnorm = np.linalg.norm(target)
            if tnorm < 1e-8:
                continue
            target_dir = target / tnorm
            c = float(conf[j, v])
            occluded = c <= conf_thr

            if mode == "hard":
                dirs_out[j, v] = target_dir
                if occluded:
                    conf_out[j, v] = fill_conf
                continue

            if mode == "fill_only":
                if occluded:
                    dirs_out[j, v] = target_dir
                    conf_out[j, v] = fill_conf
                continue

            # soft / soft_fill: blend weight grows as conf drops
            alpha = float(np.clip(strength * (1.0 - c), 0.0, 1.0))
            if mode == "soft_fill" and occluded:
                dirs_out[j, v] = target_dir
                conf_out[j, v] = fill_conf
            elif alpha > 1e-6:
                blended = (1.0 - alpha) * dir_safe[j, v] + alpha * target_dir
                bnorm = np.linalg.norm(blended)
                if bnorm > 1e-8:
                    dirs_out[j, v] = blended / bnorm

    if squeeze_conf:
        confs_out = conf_out[..., None]
    else:
        confs_out = conf_out
    return dirs_out.astype(directions.dtype, copy=False), confs_out.astype(confs.dtype, copy=False)


def maybe_soft_refine_rays(directions, intersections, confs):
    """No-op unless RUMPL_2D_REFINE=1."""
    if not refine_enabled():
        return directions, confs
    return soft_refine_rays(directions, intersections, confs)
