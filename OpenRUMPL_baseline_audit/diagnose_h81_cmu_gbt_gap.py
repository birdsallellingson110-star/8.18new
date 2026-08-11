#!/usr/bin/env python3
"""Summarize why H81 zero-shot CMU lags GBT cross-dataset numbers."""

import argparse
import json
from pathlib import Path


GBT_REF = {
    "h36m_train_cmu_test_all17_mm": 17.2,
    "note": "GBT full model, 4 cams (2,13,10,19), T=9, absolute All-17",
}
ALG_TRI_HRNET_CMU = 26.2


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--viewgroups-root", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    root = Path(args.eval_root)
    metrics = {}
    for sub in sorted(root.iterdir()):
        p = sub / "gbt_aligned.json"
        if p.is_file():
            metrics[sub.name] = load_json(p)

    vg = load_json(
        Path(args.viewgroups_root) / "viewgroups_master.json"
    )

    proxy = metrics.get("gbt_proxy_4cam_3_6_13_23", {})
    five = metrics.get("rumpl_5cam_standard", {})

    v2_per = (
        load_json(
            Path(args.viewgroups_root) / "V2_all_combinations" / "viewgroups.json"
        )["per_combination"]
    )
    worst_v2 = sorted(
        v2_per.items(),
        key=lambda kv: kv[1]["absolute"]["all17_mm"],
        reverse=True,
    )[:3]

    report = {
        "gbt_paper_reference": GBT_REF,
        "algebraic_triangulation_hrnet_cmu_mm": ALG_TRI_HRNET_CMU,
        "h81_zero_shot_absolute": metrics,
        "h81_v5_all17_mm": vg["view_groups"]["V5"]["mean_over_combinations"]["absolute"]["all17_mm"],
        "worst_v2_pairs_absolute_all17": [
            {"views": k, **v["absolute"]} for k, v in worst_v2
        ],
        "gap_decomposition": [
            {
                "factor": "Training domain",
                "gbt": "CMU Panoptic train (27 HD) OR H36M with scene centering + synthetic views for CMU generalization",
                "h81": "Human3.6M only (A1D/H21 2D); never CMU finetune",
                "impact": "Primary — GBT Table IV H36M→CMU 17.2 mm uses architecture + centering/synthetic/dropout designed for CMU geometry",
            },
            {
                "factor": "Temporal",
                "gbt": "T=9 frame window; temporal self-attention",
                "h81": "Single-frame RUMPL (T=1)",
                "impact": "Large — occlusions and noisy 2D averaged over time in GBT",
            },
            {
                "factor": "Test cameras",
                "gbt": "Fixed 4 HD: 2, 13, 10, 19",
                "h81": "Local PKL only has 3,6,12,13,23; proxy 4-view uses 3,6,13,23",
                "impact": "Moderate — wide-baseline pairs like [3,12] explode under H81 zero-shot",
            },
            {
                "factor": "2D observations",
                "gbt": "HRNet on CMU (paper); geometry/confidence-biased fusion",
                "h81": "HRNet matched_swapv3 PKL, not H36M A1D/H21 stack; no conf/geom bias in VFT like GBT",
                "impact": "Moderate — domain shift in 2D errors and confidence calibration",
            },
            {
                "factor": "Metric",
                "gbt": "Absolute All-17, no pelvis subtraction",
                "h81": "Same when using gbt_aligned script; relative metric was inflating some reports",
                "impact": "Reporting only",
            },
        ],
        "headline": {},
    }

    if proxy:
        gap = proxy["all17_mm"] - GBT_REF["h36m_train_cmu_test_all17_mm"]
        report["headline"] = {
            "h81_proxy4_all17_mm": proxy["all17_mm"],
            "gbt_all17_mm": GBT_REF["h36m_train_cmu_test_all17_mm"],
            "delta_mm": gap,
            "ratio": proxy["all17_mm"] / GBT_REF["h36m_train_cmu_test_all17_mm"],
        }

    Path(args.output_json).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["headline"], indent=2))
    print("\n--- Gap factors (summary) ---")
    for item in report["gap_decomposition"]:
        print(f"* {item['factor']}: {item['impact']}")


if __name__ == "__main__":
    main()
