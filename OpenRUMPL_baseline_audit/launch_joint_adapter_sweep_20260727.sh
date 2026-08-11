#!/usr/bin/env bash
set -euo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
root=/mnt/data/cjyoutput/baseline_reaudit_20260722
result_dir="$root/transformer_clean_r5_20260726"

# Two serial queues, one per GPU. The first wave tests the two main hypotheses;
# the second wave provides sparse/strong-decay ablations.
(
  bash "$repo/launch_joint_adapter_variant_20260727.sh" \
    0 T4_adapter_decay15_r5init_seed0_20260727 2,5,8 1.5 1.2
  bash "$repo/launch_joint_adapter_variant_20260727.sh" \
    0 T6_adapter_midonly_r5init_seed0_20260727 5 1.0 1.2
) > "$result_dir/t4_t6_queue_gpu0.log" 2>&1 &
pid0=$!

(
  bash "$repo/launch_joint_adapter_variant_20260727.sh" \
    1 T5_adapter_late_r5init_seed0_20260727 5,8,11 1.0 1.2
  bash "$repo/launch_joint_adapter_variant_20260727.sh" \
    1 T7_adapter_decay20_r5init_seed0_20260727 2,5,8 2.0 1.2
) > "$result_dir/t5_t7_queue_gpu1.log" 2>&1 &
pid1=$!

printf 'gpu0_pid=%s\ngpu1_pid=%s\n' "$pid0" "$pid1" \
  > "$result_dir/joint_adapter_sweep_20260727.pids"
echo "launched gpu0_pid=$pid0 gpu1_pid=$pid1"
