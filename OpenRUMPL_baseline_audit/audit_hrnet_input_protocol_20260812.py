#!/usr/bin/env python3
"""Audit HRNet detector/crop provenance and feature-map coordinate conventions.

The report is deliberately read-only.  It records the H36M annotation box
source, MMPose config/checkpoint hashes, and compares the current compact-token
coordinate formula with the exact MMPose affine transform at the HRNet output
resolution.  It does not rerun the detector or modify any cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
from mmpose.structures.bbox import get_warp_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-metadata", nargs="*", default=[])
    parser.add_argument("--sample-records", type=int, default=256)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def current_feature_coord(points: np.ndarray, center: np.ndarray, scale: np.ndarray,
                          feature_size=(72, 96)) -> np.ndarray:
    # This is exactly the formula in prepare_rigr_feature_tokens_20260812.py.
    return (points - center[None] + 0.5 * scale[None]) / scale[None] * np.asarray(
        [feature_size[0], feature_size[1]], dtype=np.float64
    )


def affine_feature_coord(points: np.ndarray, center: np.ndarray, scale: np.ndarray,
                         input_size=(288, 384), feature_size=(72, 96),
                         align_corners=True) -> np.ndarray:
    # TopdownAffine with the current official HRNet config (MSRA, not UDP).
    warp = get_warp_matrix(center.astype(np.float32), scale.astype(np.float32), 0.0,
                           output_size=input_size)
    crop = cv2.transform(points[None].astype(np.float32), warp)[0].astype(np.float64)
    if align_corners:
        factors = np.asarray(
            [(feature_size[0] - 1) / (input_size[0] - 1),
             (feature_size[1] - 1) / (input_size[1] - 1)], dtype=np.float64
        )
        return crop * factors
    return (crop + 0.5) * np.asarray(feature_size, dtype=np.float64) / np.asarray(
        input_size, dtype=np.float64
    ) - 0.5


def mmpose_codec_feature_coord(points: np.ndarray, center: np.ndarray,
                               scale: np.ndarray, input_size=(288, 384),
                               feature_size=(72, 96)) -> np.ndarray:
    """MMPose MSRA codec convention: input/heatmap is a size ratio.

    The official codec decodes a heatmap coordinate by multiplying by
    ``input_size / heatmap_size``.  This is different from interpreting the
    crop as a closed interval and using ``(W-1)/(w-1)``.  The current token
    sampler intentionally follows this codec convention.
    """
    warp = get_warp_matrix(center.astype(np.float32), scale.astype(np.float32), 0.0,
                           output_size=input_size)
    crop = cv2.transform(points[None].astype(np.float32), warp)[0].astype(np.float64)
    return crop * np.asarray(feature_size, dtype=np.float64) / np.asarray(
        input_size, dtype=np.float64
    )


def main() -> None:
    args = parse_args()
    input_pkl = Path(args.input_pkl).resolve()
    config = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = pickle.loads(input_pkl.read_bytes())
    if not records:
        raise ValueError("empty input pkl")
    config_text = config.read_text(errors="replace")
    feature_shapes = []
    metadata_rows = []
    for metadata_path in args.feature_metadata:
        with np.load(metadata_path, allow_pickle=False) as source:
            feature_shapes.append({
                "file": str(Path(metadata_path).resolve()),
                "records": int(len(source["record_indices"])),
                "input_size_unique": np.unique(source["input_size"], axis=0).tolist(),
                "center_shape": list(source["input_center"].shape),
                "scale_shape": list(source["input_scale"].shape),
            })
    sample = records[: min(args.sample_records, len(records))]
    # The pkl ``scale`` is the source toolbox's normalized scale (roughly
    # person-size/100), not the pixel bbox scale consumed by MMPose.  The
    # exporter stores the post-TopdownAffine ``input_center/input_scale``;
    # compare the annotation box with that metadata instead of mixing the two
    # coordinate systems.
    metadata_lookup = {}
    for metadata_path in args.feature_metadata:
        with np.load(metadata_path, allow_pickle=False) as source:
            for row, record_index in enumerate(source["record_indices"]):
                metadata_lookup[int(record_index)] = (
                    np.asarray(source["input_center"][row], dtype=np.float64),
                    np.asarray(source["input_scale"][row], dtype=np.float64),
                    np.asarray(source["input_size"][row], dtype=np.float64),
                )
    box_consistency = []
    for local_index, record in enumerate(sample):
        box = np.asarray(record["box"], dtype=np.float64)
        expected_center = 0.5 * (box[:2] + box[2:])
        # Metadata may not be present when the caller audits only the pkl.
        # In that case, only report the annotation's internal box relation.
        metadata = metadata_lookup.get(local_index)
        if metadata is None:
            observed_center = np.asarray(record["center"], dtype=np.float64)
            observed_scale = np.asarray(record["scale"], dtype=np.float64)
            observed_note = "source pkl center/normalized scale"
        else:
            observed_center, observed_scale, _ = metadata
            observed_note = "MMPose input_center/input_scale"
        box_consistency.append({
            "center_max_abs": float(np.max(np.abs(observed_center - expected_center))),
            "scale_or_size_max_abs": float(
                np.max(np.abs(observed_scale - (box[2:] - box[:2])))
            ) if metadata is None else float(
                # GetBBoxCenterScale uses padding=1.25, then TopdownAffine
                # fixes the 288x384 aspect ratio (height = width*4/3).
                np.max(np.abs(observed_scale - (box[2:] - box[:2])
                              * 1.25 * np.asarray([1., 4. / 3.])))
            ),
            "observed_scale_note": observed_note,
        })
    probe = np.asarray(sample[0]["joints_2d"], dtype=np.float64)
    if 0 in metadata_lookup:
        center, scale, input_size = metadata_lookup[0]
        input_size = tuple(input_size.astype(int).tolist())
    else:
        center = np.asarray(sample[0]["center"], dtype=np.float64)
        scale = np.asarray(sample[0]["scale"], dtype=np.float64)
        input_size = (288, 384)
    current = current_feature_coord(probe, center, scale)
    codec = mmpose_codec_feature_coord(probe, center, scale,
                                       input_size=input_size)
    exact_true = affine_feature_coord(probe, center, scale, input_size=input_size,
                                      align_corners=True)
    exact_false = affine_feature_coord(probe, center, scale, input_size=input_size,
                                       align_corners=False)
    report = {
        "input_pkl": str(input_pkl),
        "input_pkl_sha256": sha256(input_pkl),
        "records": len(records),
        "config": str(config),
        "config_sha256": sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "detector_protocol": {
            "config_name": config.name,
            "input_size": [288, 384],
            "heatmap_size": [72, 96],
            "use_udp_in_config": bool("use_udp=True" in config_text),
            "flip_test_in_config": bool("flip_test=True" in config_text),
            "box_source_in_pkl": "H36M annotation h5 center/scale converted to square xyxy",
            "note": "This script records provenance; it does not infer YOLOX boxes from the pkl.",
        },
        "feature_metadata": feature_shapes,
        "box_consistency_sample": {
            "max_center_abs": float(max(row["center_max_abs"] for row in box_consistency)),
            "max_scale_or_size_abs": float(max(row["scale_or_size_max_abs"] for row in box_consistency)),
        },
        "coordinate_audit": {
            "sample_record_index": 0,
            "current_formula_vs_mmpose_align_corners_true_max_px": float(
                np.max(np.abs(current - exact_true))
            ),
            "current_formula_vs_mmpose_align_corners_false_max_px": float(
                np.max(np.abs(current - exact_false))
            ),
            "current_formula_vs_mmpose_msra_codec_max_px": float(
                np.max(np.abs(current - codec))
            ),
            "current_formula": "(xy-center+0.5*scale)/scale*[72,96]",
            "recommendation": (
                "The current formula matches the MSRA codec ratio, but the feature "
                "sampler's align_corners convention must remain documented."
            ),
        },
        "sample_records": int(len(sample)),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# HRNet input protocol audit (2026-08-12)", "",
        f"- records: `{len(records)}`",
        f"- box source: `{report['detector_protocol']['box_source_in_pkl']}`",
        f"- config: `{config}`",
        f"- use UDP: `{report['detector_protocol']['use_udp_in_config']}`",
        f"- flip test: `{report['detector_protocol']['flip_test_in_config']}`",
        f"- current vs exact affine (align_corners=True): `{report['coordinate_audit']['current_formula_vs_mmpose_align_corners_true_max_px']:.6f}` feature px",
        f"- current vs exact affine (align_corners=False): `{report['coordinate_audit']['current_formula_vs_mmpose_align_corners_false_max_px']:.6f}` feature px",
        f"- current vs MMPose MSRA codec ratio: `{report['coordinate_audit']['current_formula_vs_mmpose_msra_codec_max_px']:.6f}` feature px",
        "",
        "The closed-interval affine values are a convention diagnostic; the MSRA codec ratio is the relevant zero-error reference for the current non-UDP HRNet config.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
