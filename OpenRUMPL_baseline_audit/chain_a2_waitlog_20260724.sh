#!/usr/bin/env bash
# Wait for A2 training END in log, then occlusion-axis eval vs R5/S2a.
set -uo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
outbase=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval
runbase=/mnt/data/cjyoutput/baseline_reaudit_20260722
variant=${1:-A2_structocc_occjl_b2_seed0_20260724}
GPU=${2:-1}
train_log="$runbase/${variant}.log"

echo "[chain-a2] waiting for END in $train_log ..."
while true; do
  if grep -q "^END " "$train_log" 2>/dev/null; then
    echo "[chain-a2] got END $(grep '^END ' "$train_log" | tail -1)"
    break
  fi
  # also accept epoch.txt >= 19 and no train process
  d=$(find "$outbase" -maxdepth 1 -type d -name "${variant}_*" 2>/dev/null | sort | tail -1 || true)
  if [ -n "$d" ] && [ -f "$d/epoch.txt" ]; then
    ep=$(tr -cd '0-9' < "$d/epoch.txt" || true)
    echo "[progress] epoch=${ep:-?} $(date +%H:%M:%S)"
    if [ -n "$ep" ] && [ "$ep" -ge 19 ] && ! pgrep -f "train_rumpl.py.*exp-name ${variant}" >/dev/null 2>&1; then
      echo "[chain-a2] epoch19 + process gone"
      break
    fi
  else
    echo "[progress] waiting $(date +%H:%M:%S)"
  fi
  sleep 120
done

sleep 5
ckpt=$(find "$outbase" -maxdepth 1 -type d -name "${variant}_*" | sort | tail -1)/model_best.pth.tar
if [ ! -f "$ckpt" ]; then echo "no ckpt $ckpt"; exit 1; fi
echo "[ckpt] $ckpt"

for v in 2 3 4 5; do
  for occ in 0.0 0.3 0.6; do
    tag="A2_v${v}_occ${occ}"
    if [ -f "$root/${tag}_summary.json" ]; then echo "[skip] $tag"; continue; fi
    echo "[gpu${GPU}] $tag $(date +%H:%M:%S)"
    bash "$repo/eval_occlusion_single_20260724.sh" "$GPU" "$tag" "$ckpt" "$v" baseline "$occ" \
      > "$root/${tag}.log" 2>&1 || echo "[FAIL] $tag"
  done
done

python3 - <<'PY'
import json, os
root="/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval"
print("############ A2 vs R5 vs S2a (All-17 / KP* mm) ############")
print(f"{'':12} {'R5':>15} {'S2a':>15} {'A2':>15} {'ΔA2-R5':>12} {'ΔA2-S2a':>12}")
for v in (2,3,4,5):
  for occ in ("0.0","0.3","0.6"):
    paths={
      "R5": f"{root}/R5_v{v}_occ{occ}_summary.json",
      "S2a": f"{root}/S2a_v{v}_occ{occ}_summary.json",
      "A2": f"{root}/A2_v{v}_occ{occ}_summary.json",
    }
    if not all(os.path.isfile(p) for p in paths.values()):
      print(f"V{v} occ{occ}: incomplete"); continue
    d={k: json.load(open(p))["overall"] for k,p in paths.items()}
    ra,rk=d["R5"]["all17_mm"], d["R5"]["kpstar_mm"]
    sa,sk=d["S2a"]["all17_mm"], d["S2a"]["kpstar_mm"]
    aa,ak=d["A2"]["all17_mm"], d["A2"]["kpstar_mm"]
    print(f"V{v} occ{occ:3}: {ra:6.2f}/{rk:5.2f}  {sa:6.2f}/{sk:5.2f}  {aa:6.2f}/{ak:5.2f}  "
          f"{aa-ra:+6.2f}/{ak-rk:+5.2f}  {aa-sa:+6.2f}/{ak-sk:+5.2f}")
print("=== chain_a2 done ===")
PY
