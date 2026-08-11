#!/usr/bin/env bash
# Fixed distill chain: wait for TRUE training completion (END / final epoch),
# then re-evaluate DS0/DS1 on V2-V5. Does NOT start new training.
set -uo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
outbase=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999
runbase=/mnt/data/cjyoutput/baseline_reaudit_20260722
root="$runbase/distill_r5_eval"
eval_root="$runbase/multiview_model_best_eval"
log="$root/chain_reeval_final_20260724.log"
mkdir -p "$root" "$root/premature_eval_bak"

exec > >(tee -a "$log") 2>&1
echo "=== fixed chain start $(date --iso-8601=seconds) ==="

wait_train_done() {
  local tag=$1
  local train_log="$runbase/${tag}.log"
  local distill_log="$root/${tag}.train.log"
  echo "[wait-final] $tag"
  while true; do
    # Prefer explicit END marker written by run_official_like_*.sh
    if [ -f "$train_log" ] && grep -q "^END " "$train_log" 2>/dev/null; then
      echo "[done-END] $tag ($(grep '^END ' "$train_log" | tail -1))"
      return 0
    fi
    if [ -f "$distill_log" ] && grep -q "^END " "$distill_log" 2>/dev/null; then
      echo "[done-END-distilllog] $tag"
      return 0
    fi
    # Or final_state checkpoint (written only at true end in some configs)
    local d
    d=$(find "$outbase" -maxdepth 1 -type d -name "${tag}_*" 2>/dev/null | sort | tail -1 || true)
    if [ -n "$d" ] && [ -f "$d/final_state.pth.tar" ]; then
      echo "[done-final_state] $tag -> $d"
      return 0
    fi
    # Or epoch.txt reached last epoch (END_EPOCH=20 => last index 19)
    if [ -n "$d" ] && [ -f "$d/epoch.txt" ]; then
      local ep
      ep=$(tr -cd '0-9' < "$d/epoch.txt" || true)
      if [ -n "$ep" ] && [ "$ep" -ge 19 ]; then
        # still require training process gone, to avoid racing mid-save
        if ! pgrep -f "exp-name ${tag}" >/dev/null 2>&1; then
          echo "[done-epoch19] $tag epoch=$ep"
          return 0
        fi
      fi
      echo "[progress] $tag epoch=${ep:-?} $(date +%H:%M:%S)"
    else
      echo "[progress] $tag (no ckpt dir yet) $(date +%H:%M:%S)"
    fi
    # hard fail if process died without END
    if ! pgrep -f "exp-name ${tag}" >/dev/null 2>&1; then
      if [ -f "$train_log" ] && grep -qiE "Traceback|FAILED line=" "$train_log" 2>/dev/null; then
        echo "[fail] $tag see $train_log"
        return 1
      fi
      # process gone but no END yet: keep waiting briefly in case of log flush,
      # then treat missing END as incomplete
      sleep 30
      if ! pgrep -f "exp-name ${tag}" >/dev/null 2>&1; then
        if [ -f "$train_log" ] && grep -q "^END " "$train_log" 2>/dev/null; then
          echo "[done-END-late] $tag"
          return 0
        fi
        if [ -n "$d" ] && [ -f "$d/epoch.txt" ]; then
          ep=$(tr -cd '0-9' < "$d/epoch.txt" || true)
          if [ -n "$ep" ] && [ "$ep" -ge 19 ]; then
            echo "[done-epoch19-noproc] $tag"
            return 0
          fi
        fi
        echo "[warn] $tag process gone without END; keep waiting for END/epoch19"
      fi
    fi
    sleep 120
  done
}

archive_premature() {
  local tag=$1
  for v in 2 3 4 5; do
    for f in \
      "$eval_root/${tag}_v${v}_summary.json" \
      "$root/${tag}_v${v}_summary.json" \
      "$root/${tag}_v${v}.log"
    do
      if [ -f "$f" ]; then
        mv -f "$f" "$root/premature_eval_bak/$(basename "$f").$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
        echo "[archive] $f"
      fi
    done
  done
}

eval_tag_final() {
  local tag=$1
  local gpu=${2:-0}
  local ckpt
  ckpt=$(find "$outbase" -maxdepth 2 -type f -path "*${tag}_*/model_best.pth.tar" | sort | tail -1)
  [ -n "$ckpt" ] || { echo "[skip-eval] no ckpt for $tag"; return 1; }
  echo "[ckpt] $tag -> $ckpt (mtime $(stat -c %y "$ckpt"))"
  archive_premature "$tag"
  for v in 2 3 4 5; do
    echo "[eval-final] ${tag}_v${v}"
    bash "$repo/eval_exact_multiview_20260723.sh" "$gpu" "${tag}_v${v}" "$ckpt" "$v" baseline \
      > "$root/${tag}_v${v}.final.log" 2>&1 || echo "[FAIL] ${tag}_v${v}"
    # copy summary into distill_r5_eval for easy comparison
    if [ -f "$eval_root/${tag}_v${v}_summary.json" ]; then
      cp -f "$eval_root/${tag}_v${v}_summary.json" "$root/${tag}_v${v}_summary.json"
    fi
  done
}

print_vs_r5() {
  local tag=$1
  echo "############ $tag vs R5 (All-17 / KP* mm) ############"
  python3 - <<PY
import json, os
root="$root"
eval_root="$eval_root"
tag="$tag"
r5={2:(30.885,35.506),3:(23.039,25.159),4:(20.213,21.698),5:(18.746,20.091)}
for v in (2,3,4,5):
    paths=[
      f"{root}/{tag}_v{v}_summary.json",
      f"{eval_root}/{tag}_v{v}_summary.json",
    ]
    j=None
    for p in paths:
        if os.path.isfile(p):
            j=p; break
    if not j:
        print(f"V{v}: MISSING"); continue
    d=json.load(open(j))
    a=d["overall"]["all17_mm"]; k=d["overall"]["kpstar_mm"]
    ra,rk=r5[v]
    print(f"V{v}: {tag} {a:.3f}/{k:.3f} | R5 {ra:.3f}/{rk:.3f} | ΔAll={a-ra:+.3f} ΔKP={k-rk:+.3f}")
PY
}

# DS0/DS1 are already training; only wait + final eval.
wait_train_done DS0_general_seed0_20260724 || exit 1
# Use GPU0 for eval if free enough; train may still hold memory — prefer GPU that finished.
# After DS0 done, DS1 may still run; eval DS0 on whichever GPU is freer.
eval_gpu=0
if pgrep -f "exp-name DS1_hardv_legw09_seed0_20260724" >/dev/null 2>&1; then
  # DS1 still training (likely on GPU0 or 1); use the other if possible
  eval_gpu=1
fi
eval_tag_final DS0_general_seed0_20260724 "$eval_gpu"
print_vs_r5 DS0_general_seed0_20260724

wait_train_done DS1_hardv_legw09_seed0_20260724 || true
eval_tag_final DS1_hardv_legw09_seed0_20260724 0
print_vs_r5 DS1_hardv_legw09_seed0_20260724

echo "=== fixed chain end $(date --iso-8601=seconds) ==="
