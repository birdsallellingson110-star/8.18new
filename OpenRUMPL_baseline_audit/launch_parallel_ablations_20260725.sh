#!/usr/bin/env bash
# Parallel ablations for AdaFuse-VW / occlusion track.
# Packs multiple RUMPL trains (each ~1.5–2GB) onto free GPU memory.
set -uo pipefail
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
out=/mnt/data/cjyoutput/baseline_reaudit_20260722
mkdir -p "$out" "$out/occlusion_eval"

launch() {
  local gpu=$1 workers=$2 script=$3 variant=$4
  shift 4
  echo "[launch gpu$gpu w$workers] $variant $*"
  (
    # shellcheck disable=SC1091
    nohup bash "$script" "$gpu" "$variant" "$@" \
      > "$out/${variant}.nohup" 2>&1 &
    echo $!
  )
}

# --- helper wrappers that accept WORKERS as last official-like arg ---
# run_b1 uses run_official_like ... "$gpu" "$variant" 0 1 16
# we create thin wrappers with workers=8

cat > /tmp/run_b1_vw_only.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
gpu=$1; variant=$2; workers=${3:-8}
unset STUDENT_GT_W DISTILL_W DISTILL_LAMBDA FEAT_DISTILL_W HARD_VIEW_MINING LEG_DISTILL_W STUDENT_VIEWS || true
export DISTILL_W=0 FEAT_DISTILL_W=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_LEARNED_RELIABILITY=0 RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0 RUMPL_SYMMETRY_LOSS_WEIGHT=0
export RUMPL_OCC_JOINT_LOSS=0 RUMPL_2D_REFINE=0
export RUMPL_TRAIN_STRUCT_OCC=0
export RUMPL_ADAFUSE_VW=1 RUMPL_ADAFUSE_VW_MIX=0.0
echo "B1-ablate: AdaFuse-VW ONLY (no struct occ) variant=$variant"
exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 0 1 "$workers"
EOF

cat > /tmp/run_b1_occ02.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
gpu=$1; variant=$2; workers=${3:-8}
unset STUDENT_GT_W DISTILL_W DISTILL_LAMBDA FEAT_DISTILL_W HARD_VIEW_MINING LEG_DISTILL_W STUDENT_VIEWS || true
export DISTILL_W=0 FEAT_DISTILL_W=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_LEARNED_RELIABILITY=0 RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0 RUMPL_SYMMETRY_LOSS_WEIGHT=0
export RUMPL_OCC_JOINT_LOSS=0 RUMPL_2D_REFINE=0
export RUMPL_TRAIN_STRUCT_OCC=1 RUMPL_TRAIN_STRUCT_OCC_LEVEL=0.2
export RUMPL_ADAFUSE_VW=1 RUMPL_ADAFUSE_VW_MIX=0.0
echo "B1-ablate: AdaFuse-VW + struct_occ=0.2 variant=$variant"
exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 0 1 "$workers"
EOF

cat > /tmp/run_b1_a2_combo.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
gpu=$1; variant=$2; workers=${3:-8}
unset STUDENT_GT_W DISTILL_W DISTILL_LAMBDA FEAT_DISTILL_W HARD_VIEW_MINING LEG_DISTILL_W STUDENT_VIEWS || true
export DISTILL_W=0 FEAT_DISTILL_W=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_LEARNED_RELIABILITY=0 RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0 RUMPL_SYMMETRY_LOSS_WEIGHT=0
export RUMPL_2D_REFINE=0
export RUMPL_TRAIN_STRUCT_OCC=1 RUMPL_TRAIN_STRUCT_OCC_LEVEL=0.4
export RUMPL_OCC_JOINT_LOSS=1 RUMPL_OCC_JOINT_LOSS_BOOST=2.0 RUMPL_OCC_JOINT_LOSS_MODE=soft
export RUMPL_ADAFUSE_VW=1 RUMPL_ADAFUSE_VW_MIX=0.0
echo "Combo: AdaFuse-VW + struct_occ=0.4 + occJL soft×2 variant=$variant"
exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 0 1 "$workers"
EOF

cat > /tmp/run_2dref_train.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
gpu=$1; variant=$2; workers=${3:-8}
unset STUDENT_GT_W DISTILL_W DISTILL_LAMBDA FEAT_DISTILL_W HARD_VIEW_MINING LEG_DISTILL_W STUDENT_VIEWS || true
export DISTILL_W=0 FEAT_DISTILL_W=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_LEARNED_RELIABILITY=0 RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0 RUMPL_SYMMETRY_LOSS_WEIGHT=0
export RUMPL_OCC_JOINT_LOSS=0 RUMPL_ADAFUSE_VW=0
export RUMPL_TRAIN_STRUCT_OCC=1 RUMPL_TRAIN_STRUCT_OCC_LEVEL=0.4
# train-time soft 2D/ray refine (paper-inspired consensus fill)
export RUMPL_2D_REFINE=1
export RUMPL_2D_REFINE_MODE=soft_fill
export RUMPL_2D_REFINE_STRENGTH=0.3
export RUMPL_2D_REFINE_FILL_CONF=0.35
export RUMPL_2D_REFINE_CONF_THR=0.1
echo "2D-refine TRAIN soft_fill s=0.3 + struct_occ=0.4 variant=$variant"
exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_official_like_cmu_seed0_20260722.sh \
  "$gpu" "$variant" 0 1 "$workers"
EOF

chmod +x /tmp/run_b1_vw_only.sh /tmp/run_b1_occ02.sh /tmp/run_b1_a2_combo.sh /tmp/run_2dref_train.sh
# also persist in repo
cp /tmp/run_b1_vw_only.sh "$repo/run_b1_vw_only_20260725.sh"
cp /tmp/run_b1_occ02.sh "$repo/run_b1_structocc02_20260725.sh"
cp /tmp/run_b1_a2_combo.sh "$repo/run_b1_a2_combo_20260725.sh"
cp /tmp/run_2dref_train.sh "$repo/run_2dref_train_structocc_20260725.sh"
chmod +x "$repo"/run_b1_vw_only_20260725.sh "$repo"/run_b1_structocc02_20260725.sh \
  "$repo"/run_b1_a2_combo_20260725.sh "$repo"/run_2dref_train_structocc_20260725.sh

# Wait briefly if DS1 still on last epoch (optional) — pack anyway, 24GB enough
echo "=== launching 4 parallel trains ==="

# GPU0: two ablations (DS1 almost done; ~2GB each)
nohup bash "$repo/run_b1_vw_only_20260725.sh" 0 B1_adafuse_vw_only_seed0_20260725 8 \
  > "$out/B1_adafuse_vw_only_seed0_20260725.nohup" 2>&1 &
echo "gpu0 vw_only pid=$!"

nohup bash "$repo/run_b1_structocc02_20260725.sh" 0 B1_adafuse_vw_structocc02_seed0_20260725 8 \
  > "$out/B1_adafuse_vw_structocc02_seed0_20260725.nohup" 2>&1 &
echo "gpu0 occ02 pid=$!"

# GPU1: combo + train-time 2D refine (B1_occ04 already running with w=16)
nohup bash "$repo/run_b1_a2_combo_20260725.sh" 1 B1A2_adafuse_vw_structocc04_occjl_seed0_20260725 8 \
  > "$out/B1A2_adafuse_vw_structocc04_occjl_seed0_20260725.nohup" 2>&1 &
echo "gpu1 combo pid=$!"

nohup bash "$repo/run_2dref_train_structocc_20260725.sh" 1 C_2dref_train_sf_structocc04_seed0_20260725 8 \
  > "$out/C_2dref_train_sf_structocc04_seed0_20260725.nohup" 2>&1 &
echo "gpu1 2dref-train pid=$!"

sleep 20
echo "=== status ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
pgrep -af 'train_rumpl.py' | grep 'exp-name' | grep -oE 'exp-name [^ ]+' | sort -u
for v in B1_adafuse_vw_only_seed0_20260725 B1_adafuse_vw_structocc02_seed0_20260725 \
         B1A2_adafuse_vw_structocc04_occjl_seed0_20260725 C_2dref_train_sf_structocc04_seed0_20260725 \
         B1_adafuse_vw_structocc04_seed0_20260725 DS1_hardv_legw09_seed0_20260724; do
  if grep -qE 'Epoch: \[0\]\[|Traceback|Error|START ' "$out/${v}.log" 2>/dev/null; then
    echo -n "$v: "
    grep -E 'START |Epoch: \[0\]\[|Traceback|AttributeError' "$out/${v}.log" 2>/dev/null | grep -v '^+' | tail -2 | tr '\n' ' '
    echo
  else
    echo "$v: (no log yet / check nohup)"
    tail -3 "$out/${v}.nohup" 2>/dev/null | tr '\n' ' '; echo
  fi
done
