#!/usr/bin/env python3
"""Summarize Phase-3 depth/anchor models against the R5 baseline."""
import json
import os

ROOT = "/mnt/data/cjyoutput/baseline_reaudit_20260722"
EVAL = os.path.join(ROOT, "depth_anchor_eval")

R5_FILES = {
    2: os.path.join(ROOT, "R5_workers16_fix_scheduler_exact_seed0_20260722_model_best_summary.json"),
    3: os.path.join(ROOT, "multiview_model_best_eval", "R5_v3_summary.json"),
    4: os.path.join(ROOT, "multiview_model_best_eval", "R5_v4_summary.json"),
    5: os.path.join(ROOT, "multiview_model_best_eval", "R5_v5_summary.json"),
}

MODELS = [
    "D1_tri_anchor_seed0_20260724",
    "D2_ray_depth_aux_w01_seed0_20260724",
    "D3_anchor_depthaux_w01_seed0_20260724",
    "D4_ray_depth_aux_w03_seed0_20260724",
]


def load(path):
    with open(path) as handle:
        return json.load(handle)


def main():
    baseline = {}
    for views, path in R5_FILES.items():
        data = load(path)
        baseline[views] = (data["overall"]["all17_mm"], data["overall"]["kpstar_mm"])

    print(f"{'model':44s} {'V':>2s} {'all17':>8s} {'d_all17':>8s} {'kpstar':>8s} {'d_kpstar':>9s}")
    for model in MODELS:
        for views in (2, 3, 4, 5):
            path = os.path.join(EVAL, f"{model}_v{views}_summary.json")
            if not os.path.exists(path):
                continue
            data = load(path)
            all17 = data["overall"]["all17_mm"]
            kpstar = data["overall"]["kpstar_mm"]
            base_all17, base_kpstar = baseline[views]
            print(
                f"{model:44s} {views:2d} {all17:8.3f} {all17 - base_all17:+8.3f} "
                f"{kpstar:8.3f} {kpstar - base_kpstar:+9.3f}"
            )
        print()

    print("R5 baseline reference:")
    for views in (2, 3, 4, 5):
        base_all17, base_kpstar = baseline[views]
        print(f"  V{views}: all17 {base_all17:.3f}  kpstar {base_kpstar:.3f}")


if __name__ == "__main__":
    main()
