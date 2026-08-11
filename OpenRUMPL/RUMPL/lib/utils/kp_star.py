# ------------------------------------------------------------------------------
# KP* metric (RUMPL paper Sec. 5.3.1).
#
# KP* = mean Absolute MPJPE over 10 joints: knees, ankles, shoulders, elbows,
# wrists (left + right). Head joints (nose/eyes/ears) and hips are EXCLUDED.
#
# Shared by baseline AND ST-VFT evaluation so the metric is computed identically
# -> baseline / B (gate=0) / C (full ST-VFT) are directly comparable.
#
# Selection is by joint NAME (robust to ordering / index drift); the resolved
# indices are returned so callers can print and sanity-check them.
# ------------------------------------------------------------------------------

import numpy as np

# Canonical KP* joints, in a fixed order. Paper Sec. 5.3.1:
# "knees, ankles, shoulders, elbows, and wrists".
KP_STAR_CANON = (
    "lsho", "rsho",   # shoulders
    "lelb", "relb",   # elbows
    "lwri", "rwri",   # wrists
    "lkne", "rkne",   # knees
    "lank", "rank",   # ankles
)

# Accept both the short names used by multiview_*_rumpl datasets and the long
# COCO names, so the same function works across dataset variants.
_ALIASES = {
    "lsho": {"lsho", "left_shoulder"},  "rsho": {"rsho", "right_shoulder"},
    "lelb": {"lelb", "left_elbow"},     "relb": {"relb", "right_elbow"},
    "lwri": {"lwri", "left_wrist"},     "rwri": {"rwri", "right_wrist"},
    "lkne": {"lkne", "left_knee"},      "rkne": {"rkne", "right_knee"},
    "lank": {"lank", "left_ankle"},     "rank": {"rank", "right_ankle"},
}


def kp_star_indices(joint_names):
    """Resolve the indices of the 10 KP* joints within `joint_names`.

    Returns a list of indices in canonical KP_STAR_CANON order.
    Raises ValueError if any KP* joint is missing or matches more than once
    (guards against silent index misalignment).
    """
    lower = [str(n).lower() for n in joint_names]
    idx = []
    for canon in KP_STAR_CANON:
        aliases = _ALIASES[canon]
        matches = [i for i, n in enumerate(lower) if n in aliases]
        if len(matches) != 1:
            raise ValueError(
                "KP* joint {!r} matched indices {} in {} (expected exactly 1)".format(
                    canon, matches, joint_names))
        idx.append(matches[0])
    return idx


def compute_kp_star(per_joint_mpjpe, joint_names):
    """Compute KP* = mean Abs MPJPE over the 10 KP* joints.

    per_joint_mpjpe : 1D array-like, per-joint Absolute MPJPE in the SAME order
                      as `joint_names` (e.g. list(d.values()) of an mpjpe pkl).
    joint_names     : sequence of joint names matching per_joint_mpjpe order.

    Returns (kp_star_value, used_indices, used_names) where kp_star_value is in
    the same unit as per_joint_mpjpe.
    """
    per_joint_mpjpe = np.asarray(per_joint_mpjpe, dtype=float)
    idx = kp_star_indices(joint_names)
    used_names = [joint_names[i] for i in idx]
    kp_star_value = float(np.nanmean(per_joint_mpjpe[idx]))
    return kp_star_value, idx, used_names
