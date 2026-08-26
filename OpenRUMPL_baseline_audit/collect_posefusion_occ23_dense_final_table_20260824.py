#!/usr/bin/env python3
"""Collect the frozen dense Occ-2/Occ-3 temporal, Algebraic and literature tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VIEWS = ("V2", "V3", "V4")
PUBLISHED_V4 = {
    # SkelSplat WACV 2026 Table 4. Keep frontend provenance explicit.
    "matched_resnet152": {
        "Alg. Triangulation (ResNet-152)": (43.2, 48.9),
        "RANSAC (as in AdaFuse)": (33.7, 38.6),
        "AdaFuse (ResNet-152)": (27.9, 31.2),
        "SkelSplat (ResNet-152)": (24.6, 27.0),
    },
    "method_specific_frontend": {
        "Alg. Triangulation (MeTRAbs)": (36.0, 39.0),
        "TransFusion": (40.8, 76.3),
        "MV Pose Fusion": (33.4, 36.7),
        "SkelSplat (MeTRAbs)": (29.6, 31.1),
    },
}


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824"),
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict = {
        "protocol": {
            "dataset": "Human3.6M S9/S11 dense VOC-object occlusion",
            "dense_context_groups": 26269,
            "scored_center_groups": 2021,
            "window": "T=9 centered, frame stride 5",
            "occlusion": "2 VOC objects/view, scale 0.2--0.5, seed 42",
            "metric": "action-equal All-17 absolute MPJPE (mm), no alignment",
            "subsets": "all 6/4/1 V2/V3/V4 camera combinations",
            "training": "clean H36M only; all checkpoints frozen before occlusion evaluation",
        },
        "results": {},
        "published_v4": PUBLISHED_V4,
    }
    for occ in (2, 3):
        variant = f"occ{occ}"
        manifest = load(args.root / variant / "protocol_manifest.json")
        params = manifest["official_parameters"]
        assert manifest["output"]["groups_written"] == 26269
        assert params["occluded_views_per_four_view_frame"] == occ
        assert params["objects_per_occluded_view"] == 2
        assert params["object_scale_uniform_relative_to_person_min_dimension"] == [0.2, 0.5]
        assert manifest["randomness"]["seed"] == 42
        assert manifest["temporal_extension_scoring"]["scored_center_groups"] == 2021
        sparse_root = Path(
            "/mnt/data/cjyoutput/h36m_occ_official_20260823"
        ) / f"calib_c2_s020_050_occ{occ}"
        stage = {}
        for frontend in ("resnet152", "hrnet"):
            base = args.root / variant / "eval" / frontend / "results"
            e2 = load(sparse_root / "eval" / f"{frontend}_spatial" / "e2_result.json")
            temporal = load(base / "final_h18_t9.json")
            if frontend == "resnet152":
                algebraic = load(sparse_root / "eval/official_lt_algebraic_report.json")
                alg = {
                    view: float(algebraic["results"][view]["absolute"]["action_equal_mm"])
                    for view in VIEWS
                }
                alg_name = "official LT learned-confidence Algebraic"
            else:
                algebraic = load(sparse_root / "eval/hrnet_algebraic_v234.json")
                alg = {
                    view: float(
                        algebraic["results"][view][
                            "pred2d_cache_coordinates_algebraic_confidence"
                        ]["action_equal_all17_mm"]
                    )
                    for view in VIEWS
                }
                alg_name = "confidence-weighted Algebraic on HRNet coordinates"
            stage[frontend] = {
                "algebraic_name": alg_name,
                "algebraic_t1_mm": alg,
                "ours_e2_t1_ablation_mm": {
                    view: float(e2["mean_mm"][view]) for view in VIEWS
                },
                "matched_temporal_center_t1_mm": {
                    view: float(
                        temporal["result"][view]["baseline_action_equal_all17_mm"]
                    )
                    for view in VIEWS
                },
                "ours_final_h18_t9_mm": {
                    view: float(
                        temporal["result"][view]["temporal_action_equal_all17_mm"]
                    )
                    for view in VIEWS
                },
                "h18_gain_mm": {
                    view: float(-temporal["result"][view]["delta_mm"])
                    for view in VIEWS
                },
            }
        report["results"][variant] = stage

    lines = [
        "# Dense Human3.6M VOC Occ-2/Occ-3 final results",
        "",
        "All learned checkpoints are frozen clean-H36M models. T=9 is the Stage-1 "
        "complete baseline; T=1 E2 is an ablation only.",
        "",
        "## Final complete baselines (T=9)",
        "",
        "| Input | Occ-2 V2 | Occ-2 V3 | Occ-2 V4 | Occ-3 V2 | Occ-3 V3 | Occ-3 V4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for frontend, label in (("resnet152", "ResNet-152"), ("hrnet", "HRNet-W32")):
        values = [
            report["results"][f"occ{occ}"][frontend]["ours_final_h18_t9_mm"][view]
            for occ in (2, 3) for view in VIEWS
        ]
        lines.append(f"| Ours ({label}), T=9 | " + " | ".join(f"{x:.3f}" for x in values) + " |")

    lines.extend([
        "",
        "## Algebraic Triangulation on the same dense images",
        "",
        "| Input | Occ-2 V2 | Occ-2 V3 | Occ-2 V4 | Occ-3 V2 | Occ-3 V3 | Occ-3 V4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for frontend, label in (("resnet152", "ResNet-152"), ("hrnet", "HRNet-W32")):
        values = [
            report["results"][f"occ{occ}"][frontend]["algebraic_t1_mm"][view]
            for occ in (2, 3) for view in VIEWS
        ]
        lines.append(f"| Alg. Tri. ({label}) | " + " | ".join(f"{x:.3f}" for x in values) + " |")

    lines.extend([
        "",
        "## Matched temporal ablation",
        "",
        "| Input/protocol | T=1 V2/V3/V4 | T=9 V2/V3/V4 | H18 gain V2/V3/V4 |",
        "|---|---:|---:|---:|",
    ])
    for occ in (2, 3):
        for frontend, label in (("resnet152", "ResNet-152"), ("hrnet", "HRNet-W32")):
            row = report["results"][f"occ{occ}"][frontend]
            center = "/".join(f"{row['matched_temporal_center_t1_mm'][v]:.3f}" for v in VIEWS)
            final = "/".join(f"{row['ours_final_h18_t9_mm'][v]:.3f}" for v in VIEWS)
            gain = "/".join(f"{row['h18_gain_mm'][v]:.3f}" for v in VIEWS)
            lines.append(f"| Occ-{occ} {label} | {center} | {final} | {gain} |")

    lines.extend([
        "",
        "## Primary four-view comparison: same ResNet-152 input family",
        "",
        "| Method | T | Occ-2 V4 | Occ-3 V4 |",
        "|---|---:|---:|---:|",
    ])
    for method, values in PUBLISHED_V4["matched_resnet152"].items():
        lines.append(f"| {method} | 1 | {values[0]:.1f} | {values[1]:.1f} |")
    ours_occ2 = report["results"]["occ2"]["resnet152"]["ours_final_h18_t9_mm"]["V4"]
    ours_occ3 = report["results"]["occ3"]["resnet152"]["ours_final_h18_t9_mm"]["V4"]
    lines.append(f"| Ours complete (ResNet-152) | 9 | {ours_occ2:.3f} | {ours_occ3:.3f} |")
    lines.extend([
        "",
        "## Published four-view references: method-specific frontends",
        "",
        "| Method | Occ-2 V4 | Occ-3 V4 |",
        "|---|---:|---:|",
    ])
    for method, values in PUBLISHED_V4["method_specific_frontend"].items():
        lines.append(f"| {method} | {values[0]:.1f} | {values[1]:.1f} |")
    lines.extend([
        "",
        "Published values are from SkelSplat WACV 2026 Table 4. Temporal context is "
        "an explicit method component and Ours T=9 is compared in the same primary "
        "table with a transparent T column. All local rows use the same documented "
        "public Human3.6M-Occ generation procedure, parameters and random seed.",
        "",
    ])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    args.output_md.write_text("\n".join(lines))
    print(args.output_md)


if __name__ == "__main__":
    main()
