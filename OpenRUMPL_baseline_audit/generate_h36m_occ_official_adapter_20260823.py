#!/usr/bin/env python3
"""Adapt the official Human3.6M-Occluded generator to RUMPL's H36M pickle.

The upstream repository assumes the h36m-fetch directory layout.  Our H36M
evaluation set stores the same S9/S11 camera images behind RUMPL-relative
paths.  This adapter changes only the data traversal and path handling: VOC
object extraction, object selection, scale, placement, alpha blending, JPEG
encoding, the global Python RNG and seed are all inherited from upstream.

The 2024 public generator hard-codes two objects per occluded view and even
contains the source comment ``occ-hard is 2 obj``; this is the original severe
dataset later reported as Occ-3-Hard.  SkelSplat does not publish its ordinary
Occ-2/Occ-3 change.  The adapter can expose the otherwise identical one-object
mode as an audited inference, but it must be validated against SkelSplat's
published algebraic-triangulation control before being called aligned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from tqdm import tqdm


CAMERA_NAMES = ["54138969", "55011271", "58860488", "60457274"]
PROTOCOL_NAME = "Human3.6M-Occluded-official-generator-RUMPL-adapter-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-pkl", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--pascal-voc-root", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--num-occluded-views", type=int, choices=(2, 3), required=True
    )
    parser.add_argument(
        "--objects-per-occluded-view",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "Two exactly reproduces the public 2024 generator (later used as "
            "Occ-3-Hard). One is the SkelSplat ordinary Occ-2/Occ-3 inference "
            "and must be validated against its published algebraic controls."
        ),
    )
    parser.add_argument(
        "--protocol-label",
        help="Explicit output label; recommended whenever object count is one.",
    )
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=None,
        help="Smoke-test prefix; omit for the complete 2021-frame set.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing generated JPEGs."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep already generated JPEGs while still executing every random "
            "sampling operation, so an interrupted deterministic run resumes "
            "without changing the RNG state of later frames."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def record_key(record: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(record["subject"]),
        int(record["action"]),
        int(record["subaction"]),
        int(record["image_id"]),
    )


def activity_sort_name(record: dict[str, Any], images_root: Path) -> str:
    """Recover the original H36M activity spelling from the symlink target."""
    source = (images_root / str(record["image"])).resolve()
    # Example: S9_Directions_1.54138969_000001.jpg -> Directions_1
    prefix = source.name.split(".", 1)[0]
    subject_prefix = f"S{int(record['subject'])}_"
    if prefix.startswith(subject_prefix):
        return prefix[len(subject_prefix) :]
    # Stable fallback for a dataset without the flattened compatibility links.
    return f"action_{int(record['action']):02d}_subaction_{int(record['subaction']):02d}"


def ordered_groups(
    records: list[dict[str, Any]], images_root: Path
) -> list[tuple[tuple[int, int, int, int], list[dict[str, Any]], str]]:
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record_key(record)].append(record)

    result = []
    for key, group in grouped.items():
        group.sort(key=lambda record: int(record["camera_id"]))
        cameras = [int(record["camera_id"]) for record in group]
        if cameras != [0, 1, 2, 3]:
            raise RuntimeError(f"synchronised group {key} has cameras {cameras}")
        activity = activity_sort_name(group[0], images_root)
        result.append((key, group, activity))

    # Mirrors upstream: subject -> lexicographically sorted activity -> frame.
    result.sort(key=lambda item: (item[0][0], item[2], item[0][3]))
    return result


def xyxy_to_xywh(box: Any) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64).reshape(-1)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError(f"invalid H36M annotation box {box}")
    x1, y1, x2, y2 = box.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"non-positive H36M annotation box {box.tolist()}")
    return np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)


def ensure_clean_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(
            f"refusing to replace non-symlink clean image without --overwrite: {destination}"
        )
    destination.symlink_to(source.resolve())


def jsonable_key(key: tuple[int, int, int, int]) -> dict[str, int]:
    return {
        "subject": key[0],
        "action": key[1],
        "subaction": key[2],
        "image_id": key[3],
    }


def occlude_with_object_count(
    image: np.ndarray,
    occluders: list[np.ndarray],
    bbox_xywh: np.ndarray,
    object_count: int,
    scale_min: float,
    scale_max: float,
    paste_over: Any,
    resize_by_factor: Any,
) -> np.ndarray:
    """Upstream occlude_with_objects with only its hard-coded count exposed."""
    result = image.copy()
    person_size = np.asarray([bbox_xywh[2], bbox_xywh[3]])
    for _ in range(object_count):
        occluder = random.choice(occluders)
        random_scale_factor = random.uniform(scale_min, scale_max)
        occluder_size = np.asarray([occluder.shape[1], occluder.shape[0]])
        scale = random_scale_factor * min(person_size) / min(occluder_size)
        occluder = resize_by_factor(occluder, scale)
        x, y, width, height = bbox_xywh
        center = (random.uniform(x, x + width), random.uniform(y, y + height))
        result = paste_over(im_src=occluder, im_dst=result, center=center)
    return result


def main() -> None:
    args = parse_args()
    for path in (
        args.validation_pkl,
        args.images_root,
        args.pascal_voc_root,
        args.official_repo,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if not (0 < args.scale_min <= args.scale_max):
        raise ValueError("scale range must satisfy 0 < min <= max")

    upstream_aug = args.official_repo / "aug_utils.py"
    upstream_main = args.official_repo / "gen_occluded_h36m.py"
    if not upstream_aug.is_file() or not upstream_main.is_file():
        raise FileNotFoundError("official repository is missing generator sources")
    sys.path.insert(0, str(args.official_repo))
    from aug_utils import (  # type: ignore
        load_occluders,
        occlude_with_objects,
        paste_over,
        resize_by_factor,
    )

    with args.validation_pkl.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise TypeError("validation pickle must contain a non-empty record list")
    groups = ordered_groups(records, args.images_root)
    full_group_count = len(groups)
    if args.limit_groups is not None:
        if args.limit_groups <= 0:
            raise ValueError("--limit-groups must be positive")
        groups = groups[: args.limit_groups]

    args.output_root.mkdir(parents=True, exist_ok=True)
    images_out = args.output_root / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_root / "occlusion_selections.jsonl"
    manifest_path = args.output_root / "protocol_manifest.json"

    print(f"Loading official Pascal VOC occluders from {args.pascal_voc_root}")
    occluders = load_occluders(str(args.pascal_voc_root))
    if not occluders:
        raise RuntimeError("official loader returned no Pascal VOC occluders")
    print(f"Loaded {len(occluders)} segmented non-person occluders")

    random.seed(args.seed)
    selection_rows: list[dict[str, Any]] = []
    generated = linked = 0
    for key, group, activity in tqdm(groups, desc=f"Generating Occ-{args.num_occluded_views}"):
        selected = random.sample(range(4), args.num_occluded_views)
        row = jsonable_key(key)
        row.update(
            {
                "activity_sort_name": activity,
                "occluded_camera_ids": selected,
                "occluded_camera_names": [CAMERA_NAMES[index] for index in selected],
            }
        )
        selection_rows.append(row)

        for record in group:
            camera_id = int(record["camera_id"])
            relative = Path(str(record["image"]))
            source = args.images_root / relative
            destination = images_out / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            if camera_id not in selected:
                ensure_clean_link(source, destination)
                linked += 1
                continue

            # Even when a file already exists, invoke the official function so
            # the global RNG state remains identical for all later samples.
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cv2.imread failed for {source}")
            bbox_xywh = xyxy_to_xywh(record["box"])
            exact_public_mode = (
                args.objects_per_occluded_view == 2
                and args.scale_min == 0.5
                and args.scale_max == 1.0
            )
            if exact_public_mode:
                occluded = occlude_with_objects(image, occluders, bbox_xywh)
            else:
                occluded = occlude_with_object_count(
                    image,
                    occluders,
                    bbox_xywh,
                    args.objects_per_occluded_view,
                    args.scale_min,
                    args.scale_max,
                    paste_over,
                    resize_by_factor,
                )
            if destination.exists() and args.resume and not args.overwrite:
                if destination.is_symlink() or not destination.is_file():
                    raise FileExistsError(
                        f"resume expected a regular generated JPEG: {destination}"
                    )
                generated += 1
                continue
            if destination.exists() and not args.overwrite:
                raise FileExistsError(
                    f"generated image already exists; use --overwrite: {destination}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink():
                destination.unlink()
            if not cv2.imwrite(str(destination), occluded):
                raise RuntimeError(f"cv2.imwrite failed for {destination}")
            generated += 1

    with selection_path.open("w", encoding="utf-8") as handle:
        for row in selection_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "name": PROTOCOL_NAME,
        "variant": (
            args.protocol_label
            or f"Occ-{args.num_occluded_views}-objects{args.objects_per_occluded_view}"
        ),
        "status": "complete" if len(groups) == full_group_count else "smoke-test-prefix",
        "input": {
            "validation_pkl": str(args.validation_pkl.resolve()),
            "validation_pkl_sha256": sha256_file(args.validation_pkl),
            "images_root": str(args.images_root.resolve()),
            "subjects": [9, 11],
            "camera_ids": [0, 1, 2, 3],
            "camera_names": CAMERA_NAMES,
        },
        "output": {
            "images_root": str(images_out.resolve()),
            "selection_jsonl": str(selection_path.resolve()),
            "groups_written": len(groups),
            "full_groups_available": full_group_count,
            "occluded_jpegs": generated,
            "clean_symlinks": linked,
        },
        "randomness": {
            "seed": args.seed,
            "rng": "Python random global state, as in upstream",
            "traversal": "subject, recovered activity name, image_id, camera_id",
        },
        "upstream": {
            "repository": "https://github.com/laurabragagnolo/human3.6m-occluded",
            "local_path": str(args.official_repo.resolve()),
            "commit": git_commit(args.official_repo),
            "aug_utils_sha256": sha256_file(upstream_aug),
            "generator_sha256": sha256_file(upstream_main),
            "reused_without_modification": (
                ["load_occluders", "occlude_with_objects", "paste_over", "resize_by_factor"]
                if args.objects_per_occluded_view == 2
                and args.scale_min == 0.5
                and args.scale_max == 1.0
                else ["load_occluders", "paste_over", "resize_by_factor"]
            ),
        },
        "official_parameters": {
            "occluded_views_per_four_view_frame": args.num_occluded_views,
            "objects_per_occluded_view": args.objects_per_occluded_view,
            "object_scale_uniform_relative_to_person_min_dimension": [
                args.scale_min,
                args.scale_max,
            ],
            "object_center": "uniform inside H36M person annotation box",
            "jpeg_encoding": "cv2.imwrite defaults, matching upstream",
        },
        "adaptation_boundary": {
            "changed": [
                "read RUMPL pickle instead of h36m-fetch folders/npy boxes",
                "convert RUMPL xyxy annotation box to upstream xywh",
                f"iterate exactly the {len(groups)} synchronised S9/S11 input groups",
                "symlink clean views instead of recompressing them",
            ],
            "known_upstream_issue": (
                "public main script derives a frame count after ::64 sampling and then "
                "applies range(..., step=64) again; adapter avoids this inconsistent traversal"
            ),
            "occ_3_hard": (
                "The public 2024 generator hard-codes two objects and contains the source "
                "comment 'occ-hard is 2 obj'. SkelSplat does not publish the ordinary-mode "
                "parameter separately; one-object mode is therefore an audited inference "
                "that requires calibration against published algebraic controls."
            ),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest["output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
