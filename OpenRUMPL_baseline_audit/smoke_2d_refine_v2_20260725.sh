#!/usr/bin/env bash
# Smoke: R5 + 2D refine on V2 clean / occ0.6, compare to existing R5 summaries.
set -uo pipefail
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
root=/mnt/data/cjyoutput/baseline_reaudit_20260722
ckpt=$(find "$root/output/multiview_amass_rumpl/multiview_rumpl_999" -maxdepth 1 -type d -name 'R5_workers16_fix_scheduler_exact_seed0_20260722_*' | sort | tail -1)/model_best.pth.tar
gpu=${1:-1}

chmod +x "$repo/eval_2d_refine_occlusion_20260725.sh"
for occ in 0.0 0.6; do
  tag="R5_2dref_sf_v2_occ${occ}"
  if [ -f "$root/occlusion_eval/${tag}_summary.json" ]; then
    echo "[skip] $tag"
    continue
  fi
  bash "$repo/eval_2d_refine_occlusion_20260725.sh" "$gpu" "$tag" "$ckpt" 2 "$occ" soft_fill 0.5 0.35 \
    > "$root/occlusion_eval/${tag}.log" 2>&1 || echo "[FAIL] $tag"
done

python3 - <<'PY'
import json, os
root="/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval"
print("=== R5 vs R5+2D-refine (V2 All-17 / KP*) ===")
for occ in ("0.0","0.6"):
  r=json.load(open(f"{root}/R5_v2_occ{occ}_summary.json"))["overall"]
  p=f"{root}/R5_2dref_sf_v2_occ{occ}_summary.json"
  if not os.path.isfile(p):
    print(f"occ{occ}: refine missing"); continue
  a=json.load(open(p))["overall"]
  print(f"occ{occ}: R5 {r['all17_mm']:.3f}/{r['kpstar_mm']:.3f}  "
        f"2dref {a['all17_mm']:.3f}/{a['kpstar_mm']:.3f}  "
        f"Δ={a['all17_mm']-r['all17_mm']:+.3f}/{a['kpstar_mm']-r['kpstar_mm']:+.3f}")
PY
