#!/usr/bin/env bash
# Launch KPA+MH3 in parallel on GPU0/1; queue D3-PCT on whichever frees first.
set -euo pipefail
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
runbase=/mnt/data/cjyoutput/baseline_reaudit_20260722
mkdir -p "$runbase/occlusion_eval"

kpa_v=KPA_a2_seed0_20260725
mh3_v=MH3_a2_seed0_20260725
d3_v=D3PCT_a2_seed0_20260725

chmod +x \
  "$repo/run_kpa_a2_20260725.sh" \
  "$repo/run_mh3_a2_20260725.sh" \
  "$repo/run_d3_pct_a2_20260725.sh" \
  "$repo/chain_module_waitlog_20260725.sh"

# GPU0: KPA
nohup bash "$repo/run_kpa_a2_20260725.sh" 0 "$kpa_v" \
  > "$runbase/${kpa_v}.nohup" 2>&1 &
echo "KPA launcher $!"

# GPU1: MH3
nohup bash "$repo/run_mh3_a2_20260725.sh" 1 "$mh3_v" \
  > "$runbase/${mh3_v}.nohup" 2>&1 &
echo "MH3 launcher $!"

# Chains (wait END then eval). D3 starts after first of KPA/MH3 finishes.
nohup bash -c "
set -uo pipefail
repo='$repo'; runbase='$runbase'
kpa_v='$kpa_v'; mh3_v='$mh3_v'; d3_v='$d3_v'

# start eval chains immediately (they wait on END)
bash \"\$repo/chain_module_waitlog_20260725.sh\" \"\$kpa_v\" kpa 0 KPA &
chain_kpa=\$!
bash \"\$repo/chain_module_waitlog_20260725.sh\" \"\$mh3_v\" mh3 1 MH3 &
chain_mh=\$!

# wait until one train finishes, then launch D3 on free GPU
while true; do
  kpa_done=0; mh_done=0
  grep -q '^END ' \"\$runbase/\${kpa_v}.log\" 2>/dev/null && kpa_done=1
  grep -q '^END ' \"\$runbase/\${mh3_v}.log\" 2>/dev/null && mh_done=1
  if [ \"\$kpa_done\" = 1 ] || [ \"\$mh_done\" = 1 ]; then
    # prefer GPU of finished job; if both, use 0
    if [ \"\$kpa_done\" = 1 ] && [ \"\$mh_done\" != 1 ]; then
      gpu=0
    elif [ \"\$mh_done\" = 1 ] && [ \"\$kpa_done\" != 1 ]; then
      gpu=1
    else
      gpu=0
    fi
    # wait a bit for eval of that GPU to start using memory — better: if only one done,
    # that GPU may be taken by its chain eval. Launch D3 on the OTHER gpu if still training,
    # else on free gpu after checking nvidia.
    if [ \"\$kpa_done\" = 1 ] && [ \"\$mh_done\" != 1 ]; then
      # KPA done → its chain uses GPU0; put D3 on GPU1 while MH3 may still train... conflict.
      # Safer: wait until BOTH trains end, then D3 on GPU0 while evals can share or serialize.
      :
    fi
    break
  fi
  sleep 120
done

# Wait BOTH trains finished so GPUs free for D3 (evals may still run — D3 needs ~8GB)
while true; do
  kpa_ok=0; mh_ok=0
  grep -q '^END ' \"\$runbase/\${kpa_v}.log\" 2>/dev/null && kpa_ok=1
  grep -q '^END ' \"\$runbase/\${mh3_v}.log\" 2>/dev/null && mh_ok=1
  [ \"\$kpa_ok\" = 1 ] && [ \"\$mh_ok\" = 1 ] && break
  echo \"[queue-d3] waiting both trains END kpa=\$kpa_ok mh=\$mh_ok \$(date +%H:%M:%S)\"
  sleep 180
done

# pick least-used GPU
gpu=\$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
echo \"[queue-d3] launching D3-PCT on GPU \$gpu\"
bash \"\$repo/run_d3_pct_a2_20260725.sh\" \"\$gpu\" \"\$d3_v\" \
  > \"\$runbase/\${d3_v}.nohup\" 2>&1 &
# chain D3 after
bash \"\$repo/chain_module_waitlog_20260725.sh\" \"\$d3_v\" pose_codebook \"\$gpu\" D3PCT

wait \$chain_kpa \$chain_mh || true
echo '[launch] all chains finished'
" > "$runbase/launch_kpa_mh3_d3pct_20260725.log" 2>&1 &
echo "orchestrator $!"

sleep 35
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
echo "=== trains ==="
pgrep -af 'train_rumpl.py' | grep -E 'KPA_a2|MH3_a2|D3PCT' || true
echo "=== banners ==="
for v in "$kpa_v" "$mh3_v"; do
  echo "-- $v --"
  grep -E 'KPA-faithful|MH3 fusion|START|Epoch: \[0\]|Traceback|FAILED|train load' \
    "$runbase/${v}.log" 2>/dev/null | grep -v '^+' | head -12 || echo "(log not ready)"
done
