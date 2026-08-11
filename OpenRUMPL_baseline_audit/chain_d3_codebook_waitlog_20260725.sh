#!/usr/bin/env bash
# Wait for D3 train END, then evaluate V2–V5 × occ{0,0.3,0.6} vs R5/A2.
set -uo pipefail
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
outbase=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval
runbase=/mnt/data/cjyoutput/baseline_reaudit_20260722
variant=${1:-D3_codebook_a2_seed0_20260725}
GPU=${2:-0}
train_log="$runbase/${variant}.log"

echo "[chain-d3] waiting for END in $train_log ..."
while true; do
  if grep -q "^END " "$train_log" 2>/dev/null; then
    echo "[chain-d3] got END $(grep '^END ' "$train_log" | tail -1)"
    break
  fi
  d=$(find "$outbase" -maxdepth 1 -type d -name "${variant}_*" 2>/dev/null | sort | tail -1 || true)
  if [ -n "$d" ] && [ -f "$d/epoch.txt" ]; then
    ep=$(tr -cd '0-9' < "$d/epoch.txt" || true)
    echo "[progress] epoch=${ep:-?} $(date +%H:%M:%S)"
    if [ -n "$ep" ] && [ "$ep" -ge 19 ] && ! pgrep -f "train_rumpl.py.*exp-name ${variant}" >/dev/null 2>&1; then
      echo "[chain-d3] epoch19 + process gone"; break
    fi
  else
    echo "[progress] waiting $(date +%H:%M:%S)"
  fi
  sleep 120
done

sleep 5
ckpt=$(find "$outbase" -maxdepth 1 -type d -name "${variant}_*" | sort | tail -1)/model_best.pth.tar
[ -f "$ckpt" ] || { echo "no ckpt $ckpt"; exit 1; }
echo "[ckpt] $ckpt"

export RUMPL_POSE_CODEBOOK=1
export RUMPL_ADAFUSE_VW=0 RUMPL_2D_REFINE=0 RUMPL_OCC_JOINT_LOSS=0
export RUMPL_TRAIN_STRUCT_OCC=0

for v in 2 3 4 5; do
  for occ in 0.0 0.3 0.6; do
    tag="D3_v${v}_occ${occ}"
    if [ -f "$root/${tag}_summary.json" ]; then echo "[skip] $tag"; continue; fi
    echo "[gpu${GPU}] $tag $(date +%H:%M:%S)"
    bash "$repo/eval_occlusion_single_20260724.sh" "$GPU" "$tag" "$ckpt" "$v" pose_codebook "$occ" \
      > "$root/${tag}.log" 2>&1 || echo "[FAIL] $tag"
  done
done

python3 - <<'PY'
import json, os
root="/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval"
print("############ D3(UniCodebook-lite) vs R5 vs A2 (All-17 mm) ############")
print(f"{'':12} {'R5':>8} {'A2':>8} {'D3':>8} {'ΔD3-R5':>8} {'ΔD3-A2':>8}")
for v in (2,3,4,5):
  for occ in ("0.0","0.3","0.6"):
    def load(tag):
      p=f"{root}/{tag}_v{v}_occ{occ}_summary.json"
      return json.load(open(p))["overall"]["all17_mm"] if os.path.isfile(p) else None
    r,a,d=load("R5"),load("A2"),load("D3")
    if None in (r,d):
      print(f"V{v} occ{occ}: incomplete"); continue
    a=a if a is not None else float('nan')
    print(f"V{v} occ{occ:3}: {r:8.2f} {a:8.2f} {d:8.2f} {d-r:+8.2f} {d-a:+8.2f}")
print("=== chain_d3 done ===")
PY
