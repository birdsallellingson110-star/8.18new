#!/usr/bin/env python3
"""Collect the reproducible Human3.6M-Occluded V2/V3/V4 benchmark table.

This collector intentionally keeps two comparison scopes separate:

1. matched-input rows recomputed locally from exactly the same 2D coordinate
   cache and all 6/4/1 camera subsets;
2. published Pose Fusion Table-4 V4 numbers, which used each method's own
   frontend and therefore are external references rather than matched controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VIEWS = ("V2", "V3", "V4")
CONTROL_KEYS = {
    "Algebraic confidence DLT": "pred2d_cache_coordinates_algebraic_confidence",
    "Confidence ray intersection": "pred2d_cache_coordinates_ray_confidence",
    "IRLS confidence rays": "pred2d_cache_coordinates_ray_irls",
    "AdaFuse public RANSAC": "pred2d_cache_coordinates_adafuse_ransac",
}
PUBLISHED_V4 = {
    "RANSAC (Pose Fusion paper)": 80.7,
    "Algebraic Triangulation (Pose Fusion paper)": 127.4,
    "AdaFuse (Pose Fusion paper)": 41.1,
    "TransFusion (Pose Fusion paper)": 96.5,
    "Pose Fusion (Pose Fusion paper)": 37.8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/data/cjyoutput/h36m_occ_official_20260823/occ3"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def row(values: dict[str, float]) -> dict[str, float]:
    return {view: float(values[view]) for view in VIEWS}


def main() -> None:
    args = parse_args()
    manifest = load(args.root / "protocol_manifest.json")
    params = manifest["official_parameters"]
    expected = {
        "objects_per_occluded_view": 2,
        "occluded_views_per_four_view_frame": 3,
        "object_scale_uniform_relative_to_person_min_dimension": [0.5, 1.0],
    }
    for key, value in expected.items():
        if params.get(key) != value:
            raise RuntimeError(f"official protocol mismatch for {key}: {params.get(key)!r}")
    if manifest["randomness"]["seed"] != 42:
        raise RuntimeError("expected frozen official-generator seed 42")

    matched: dict[str, dict[str, dict[str, float]]] = {}
    inputs = {
        "ResNet-152": "resnet152_spatial",
        "HRNet-W32": "hrnet_spatial",
    }
    controls_files = {
        "ResNet-152": args.root / "eval/resnet152_controls_v234.json",
        "HRNet-W32": args.root / "eval/hrnet_controls_v234.json",
    }
    for frontend, spatial_name in inputs.items():
        controls = load(controls_files[frontend])
        direct = load(args.root / f"eval/{spatial_name}/direct_result.json")
        e2 = load(args.root / f"eval/{spatial_name}/e2_result.json")
        if controls["protocol"]["complete_four_view_groups"] != 2021:
            raise RuntimeError(f"{frontend}: incomplete control groups")
        frontend_rows: dict[str, dict[str, float]] = {}
        for label, key in CONTROL_KEYS.items():
            frontend_rows[label] = row({
                view: controls["results"][view][key]["action_equal_all17_mm"]
                for view in VIEWS
            })
        frontend_rows["Frozen direct generator"] = row({
            view: direct["results"][view]["action_equal_all17_mm"] for view in VIEWS
        })
        frontend_rows["Ours E2"] = row(e2["mean_mm"])
        matched[frontend] = frontend_rows

    report = {
        "benchmark": "Human3.6M-Occluded official 2024 generator, V2/V3/V4",
        "protocol": {
            "generator_commit": manifest["upstream"]["commit"],
            "generator_parameters": params,
            "seed": manifest["randomness"]["seed"],
            "groups": manifest["output"]["groups_written"],
            "metric": "action-equal All-17 absolute MPJPE (mm), no alignment",
            "camera_subsets": "all 6/4/1 combinations for V2/V3/V4",
            "training": "all learned rows frozen from clean Human3.6M; no occlusion fine-tuning",
        },
        "matched_input_results_mm": matched,
        "published_v4_reference_mm": PUBLISHED_V4,
        "comparison_boundary": (
            "Matched rows share our exact coordinate cache. Published V4 rows use each "
            "paper method's own frontend and are shown only as external references."
        ),
    }

    lines = [
        "# Human3.6M-Occluded official V2/V3/V4 results",
        "",
        "Protocol: official 2024 generator; 3 of 4 views occluded; 2 Pascal-VOC "
        "objects per occluded view; scale 0.5--1.0; seed 42; 2021 synchronized "
        "S9/S11 groups; all camera subsets; action-equal All-17 absolute MPJPE.",
        "",
    ]
    for frontend, rows in matched.items():
        lines.extend([
            f"## Matched {frontend} coordinates",
            "",
            "| Method | V2 | V3 | V4 |",
            "|---|---:|---:|---:|",
        ])
        for label, values in rows.items():
            lines.append(
                f"| {label} | {values['V2']:.3f} | {values['V3']:.3f} | {values['V4']:.3f} |"
            )
        lines.append("")
    lines.extend([
        "## Published Pose Fusion Table-4 reference (V4 only)",
        "",
        "| Method | V4 |",
        "|---|---:|",
    ])
    for label, value in PUBLISHED_V4.items():
        lines.append(f"| {label} | {value:.1f} |")
    lines.extend([
        "",
        "Published rows used their own frontends and are not merged with the matched-input table.",
        "",
    ])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    args.output_md.write_text("\n".join(lines))
    print(args.output_md)


if __name__ == "__main__":
    main()
