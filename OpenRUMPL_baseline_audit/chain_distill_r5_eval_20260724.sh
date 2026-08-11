#!/usr/bin/env bash
# Wait for DS0, train DS1, then evaluate both on V2-V5.
set -uo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
outbase=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/distill_r5_eval
log="$root/chain_after_ds0.log"
mkdir -p "$root"

exec > >(tee -a "$log") 2>&1
echo "=== chain start $(date --iso-8601=seconds) ==="

wait_train() {
  local tag=$1
  local train_log="$root/${tag}.train.log"
  echo "[wait] $tag"
  while true; do
    if find "$outbase" -maxdepth 1 -type d -name "${tag}_*" | grep -q .; then
      local d
      d=$(find "$outbase" -maxdepth 1 -type d -name "${tag}_*" | sort | tail -1)
      if [ -f "$d/final_state.pth.tar" ] || [ -f "$d/model_best.pth.tar" ]; then
        echo "[done] $tag -> $d"
        return 0
      fi
    fi
    if ! pgrep -f "exp-name $tag" >/dev/null 2>&1; then
      if [ -f "$train_log" ] && grep -q "END " "$train_log" 2>/dev/null; then
        echo "[done-log] $tag"
        return 0
      fi
      if [ -f "$train_log" ] && grep -q "Traceback\|FAILED" "$train_log" 2>/dev/null; then
        echo "[fail] $tag see $train_log"
        return 1
      fi
    fi
    sleep 120
  done
}

eval_tag() {
  local tag=$1
  local ckpt
  ckpt=$(find "$outbase" -maxdepth 2 -type f -path "*${tag}_*/model_best.pth.tar" | sort | tail -1)
  [ -n "$ckpt" ] || { echo "[skip-eval] no ckpt for $tag"; return 1; }
  for v in 2 3 4 5; do
    [ -f "$root/${tag}_v${v}_summary.json" ] && continue
    echo "[eval] ${tag}_v${v}"
    bash "$repo/eval_exact_multiview_20260723.sh" 0 "${tag}_v${v}" "$ckpt" "$v" baseline \
      > "$root/${tag}_v${v}.log" 2>&1 || echo "[FAIL] ${tag}_v${v}"
  done
}

wait_train DS0_general_seed0_20260724 || exit 1
eval_tag DS0_general_seed0_20260724

if ! find "$outbase" -maxdepth 1 -type d -name 'DS1_hardv_legw09_seed0_20260724_*' | grep -q .; then
  echo "[train] DS1_hardv_legw09_seed0_20260724"
  bash "$repo/run_distill_r5_20260724.sh" 0 DS1_hardv_legw09_seed0_20260724 hardv_legw09 \
    > "$root/DS1_hardv_legw09_seed0_20260724.train.log" 2>&1 \
    || echo "[FAIL train DS1]"
fi

wait_train DS1_hardv_legw09_seed0_20260724 || true
eval_tag DS1_hardv_legw09_seed0_20260724

echo "=== chain end $(date --iso-8601=seconds) ==="
