#!/usr/bin/env bash
# Wait for a module-ablation train END, then V2–V5 × occ{0,0.3,0.6} vs R5/A2.
# usage: $0 VARIANT MODE [GPU]
#   MODE: kpa | mh3 | pose_codebook | conf_film
set -uo pipefail
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
outbase=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval
runbase=/mnt/data/cjyoutput/baseline_reaudit_20260722
variant=${1:?usage: $0 VARIANT MODE [GPU]}
mode=${2:?}
GPU=${3:-0}
prefix=${4:-}  # optional tag prefix override; default=MODE upper-ish from variant

# short tag for summaries: first token before _
tag_prefix=${prefix:-$(echo "$variant" | cut -d_ -f1)}
train_log="$runbase/${variant}.log"

echo "[chain-$tag_prefix] waiting for END in $train_log mode=$mode ..."
while true; do
  if grep -q "^END " "$train_log" 2>/dev/null; then
    echo "[chain-$tag_prefix] got END $(grep '^END ' "$train_log" | tail -1)"
    break
  fi
  d=$(find "$outbase" -maxdepth 1 -type d -name "${variant}_*" 2>/dev/null | sort | tail -1 || true)
  if [ -n "$d" ] && [ -f "$d/epoch.txt" ]; then
    ep=$(tr -cd '0-9' < "$d/epoch.txt" || true)
    echo "[progress] epoch=${ep:-?} $(date +%H:%M:%S)"
    if [ -n "$ep" ] && [ "$ep" -ge 19 ] && ! pgrep -f "train_rumpl.py.*exp-name ${variant}" >/dev/null 2>&1; then
      echo "[chain-$tag_prefix] epoch19 + process gone"; break
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

# eval-time: architecture flags only (no train occlusions)
export RUMPL_ADAFUSE_VW=0 RUMPL_2D_REFINE=0 RUMPL_OCC_JOINT_LOSS=0
export RUMPL_TRAIN_STRUCT_OCC=0 RUMPL_KPA=0 RUMPL_MULTI_HYP=1 RUMPL_POSE_CODEBOOK=0 RUMPL_CONF_FILM=0

for v in 2 3 4 5; do
  for occ in 0.0 0.3 0.6; do
    tag="${tag_prefix}_v${v}_occ${occ}"
    if [ -f "$root/${tag}_summary.json" ]; then echo "[skip] $tag"; continue; fi
    echo "[gpu${GPU}] $tag $(date +%H:%M:%S)"
    bash "$repo/eval_occlusion_single_20260724.sh" "$GPU" "$tag" "$ckpt" "$v" "$mode" "$occ" \
      > "$root/${tag}.log" 2>&1 || echo "[FAIL] $tag"
  done
done

python3 - <<PY
import json, os
root="/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval"
tagp="${tag_prefix}"
print(f"############ {tagp} vs R5 vs A2 (All-17 mm) ############")
print(f"{'':12} {'R5':>8} {'A2':>8} {tagp:>8} {'Δ-R5':>8} {'Δ-A2':>8}")
for v in (2,3,4,5):
  for occ in ("0.0","0.3","0.6"):
    def load(tag):
      p=f"{root}/{tag}_v{v}_occ{occ}_summary.json"
      return json.load(open(p))["overall"]["all17_mm"] if os.path.isfile(p) else None
    r,a,m=load("R5"),load("A2"),load(tagp)
    if None in (r,m):
      print(f"V{v} occ{occ}: incomplete"); continue
    a=a if a is not None else float('nan')
    print(f"V{v} occ{occ:3}: {r:8.2f} {a:8.2f} {m:8.2f} {m-r:+8.2f} {m-a:+8.2f}")
print(f"=== chain_{tagp} done ===")
PY
