#!/usr/bin/env bash
# C1/C2 input-only screen: the H76/RUMPL checkpoint and all model flags stay
# fixed.  Only the prepared 2-D annotation directory changes:
# C1=CPN-XY with confidence one, C2=CPN-XYC with MTF score.pkl confidence.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
PREP=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets_mmpose
CPN=/mnt/data/cjyoutput/cpn_rumpl_native_20260820_strict
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
CKPT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/CARD_C2_B1_K2HEAVY_H76_T1_20E_LR1e5_seed0_20260815_2026-08-15_12-02-59/model_best.pth.tar
OUT="${CPN_OUT:-/mnt/data/cjyoutput/cpn_rumpl_native_20260820_strict/frozen_h76_eval}"
mkdir -p "$OUT"
test -s "$CFG" && test -s "$CKPT"

link_variant() {
  local variant="$1"
  local type="mtf_cpn_native_${variant,,}"
  local dir="${PREP}/annot_filtered_5_64_${type}"
  mkdir -p "$dir"
  for split in train validation; do
    local source="${CPN}/h36m_${split}_${variant}.pkl"
    local target="${dir}/h36m_${split}.pkl"
    test -s "$source"
    if [[ -e "$target" || -L "$target" ]]; then
      [[ "$(readlink -f "$target")" == "$(readlink -f "$source")" ]] || {
        echo "mismatched CPN link: $target" >&2; exit 2;
      }
    else
      ln -s "$source" "$target"
    fi
  done
}

run_variant() {
  local variant="$1" gpu="$2"
  local type="mtf_cpn_native_${variant,,}"
  local root="${OUT}/${variant}"
  mkdir -p "$root"
  if [[ -s "$root/COMPLETED" ]]; then return 0; fi
  link_variant "$variant"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONPATH="$AUDIT"
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
  export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1 RUMPL_N_VIEWS_TRAIN_TEST_ALL=4
  export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
  export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
  export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
  export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
  export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1 RUMPL_INPUT_HARMONIC_L=0
  export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0 RUMPL_RELATIVE_VIEW_FUSION=0
  export RUMPL_SKELETON_VIEW_RELIABILITY=0 RUMPL_CONFIDENCE_VIEW_BIAS=0
  export RUMPL_GEOMETRY_VIEW_BIAS=0 RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0
  export RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0 RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0
  export RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0 RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
  export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0 RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
  export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
  export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
  export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
  {
    echo "variant=${variant} type=${type} checkpoint=${CKPT}"
    sha256sum "$CFG" "$CKPT" "${CPN}/h36m_validation_${variant}.pkl"
  } >"$root/manifest.txt"
  cd "$REPO"
  for views in 2 3 4; do
    local eval_dir="$root/V${views}"
    mkdir -p "$eval_dir"
    RUMPL_EVAL_STRICT=1 "$PY" -u run/eval_rumpl_checkpoint.py \
      --cfg "$CFG" --checkpoint "$CKPT" --output-dir "$eval_dir" \
      --workers 8 --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test false \
      --test-on-all-cameras true --n-views-combinations "$views" --model-num-views 4 \
      --test-mmpose-type "$type" >"$eval_dir/eval.log" 2>&1
    pred=$(find "$eval_dir" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
    test -s "$pred"
    "$PY" run/eval_h36m_table2.py --dict-pkl "$pred" \
      --output-json "$eval_dir/table2.json" >"$eval_dir/table2.log" 2>&1
  done
  date --iso-8601=seconds >"$root/COMPLETED"
}

gpu_c1="${CPN_GPU_C1:-0}"
gpu_c2="${CPN_GPU_C2:-1}"
if [[ "${CPN_ONLY_VARIANT:-}" == "C1" ]]; then
  run_variant C1 "$gpu_c1"
  exit 0
elif [[ "${CPN_ONLY_VARIANT:-}" == "C2" ]]; then
  run_variant C2 "$gpu_c2"
  exit 0
fi
run_variant C1 "$gpu_c1" & p0=$!
run_variant C2 "$gpu_c2" & p1=$!
wait "$p0" "$p1"
echo "CPN C1/C2 frozen input-only evaluation complete $(date --iso-8601=seconds)"
