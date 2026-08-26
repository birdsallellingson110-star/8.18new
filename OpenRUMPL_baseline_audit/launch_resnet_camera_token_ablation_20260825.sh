#!/usr/bin/env bash
# Matched ResNet-152 canonical+Global-Joint-Query continuation control.
set -euo pipefail

MODE=${1:?usage: $0 control|token10}
case "${MODE}" in
  control) TOKEN_DROPOUT=0 ;;
  token10) TOKEN_DROPOUT=0.10 ;;
  *) echo "unknown mode: ${MODE}" >&2; exit 2 ;;
esac

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
MODEL_ROOT=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999
BASE=/mnt/data/cjyoutput/camera_generalization_20260824/stage1_h36m_dual_frontend/resnet152/generator
INIT=$(<"${BASE}/checkpoint.txt")
ROOT=/mnt/data/cjyoutput/camera_generalization_20260824/resnet_camera_token_ablation_20260825/${MODE}
TAG=CAMGEN_RESNET_CANON_QUERY_${MODE}_4E_seed0_20260825
TYPE=res152_lt_alg_undistorted_annbox

mkdir -p "${ROOT}"
test -s "${CFG}"; test -s "${INIT}"

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=${AUDIT}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export RUMPL_FIX_SCHEDULER_ORDER=1 RUMPL_RANDOM_VIEW_SUBSET=1
export RUMPL_N_VIEWS_TRAIN_TEST_ALL=4 RUMPL_END_EPOCH=4
export RUMPL_INIT_CHECKPOINT=${INIT} RUMPL_FINETUNE_LR=1e-6
export RUMPL_LR_STEPS=3 RUMPL_SAVE_EVERY_N_EPOCHS=1
export RUMPL_FLIP_LOWER_BODY_KP_TEST=0
export RUMPL_TRI_ANCHOR=1 RUMPL_TRI_ANCHOR_REG=1e-4
export RUMPL_TRI_ANCHOR_CONF_EPS=0.05 RUMPL_PFT_REPEAT_LAST=1
export RUMPL_ANCHOR_CENTERED_RAYS=1 RUMPL_INPUT_PLUCKER=1
export RUMPL_BODY_CANONICAL_FRAME=1 RUMPL_BODY_CANONICAL_REG=1e-4
export RUMPL_BODY_CANONICAL_PELVIS_PRIOR=0
export RUMPL_INPUT_HARMONIC_L=0 RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_RELATIVE_VIEW_FUSION=0 RUMPL_SKELETON_VIEW_RELIABILITY=0
export RUMPL_CONFIDENCE_VIEW_BIAS=0 RUMPL_GEOMETRY_VIEW_BIAS=0
export RUMPL_NORMALIZE_VIEW_CONFIDENCE=1
export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0 RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_GBT_SET_DECODER=0 RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0
export RUMPL_VFT_DEPTH=0 GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0
export GBT_GLOBAL_JV_GATED=0 GBT_LEARNABLE_BIAS=0
export GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export RUMPL_GBT_QUERY_RESIDUAL=1 RUMPL_GBT_QUERY_RESIDUAL_GLOBAL=1
export RUMPL_GBT_QUERY_RESIDUAL_DEPTH=2
export RUMPL_GBT_QUERY_RESIDUAL_MAX_DELTA=0.5
export GBT_TOKEN_DROPOUT=${TOKEN_DROPOUT} GBT_TOKEN_DROPOUT_EPOCHS=4
export RUMPL_TOKEN_DROPOUT=0 RUMPL_GBT_SYNTHETIC_REPLACE_PROB=0
export CAA_LAMBDA=0 DEPRO_LAMBDA=0 REPROJ_LAMBDA=0
export RAY_LAMBDA=0 BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0
export RUMPL_TRAIN_SCOPE=all RUMPL_VIEW_COUNT_WEIGHTS=3,1,1
unset TRAIN_FIXED_NUM_VIEWS TRAIN_FIXED_NUM_VIEWS_EPOCHS

if [[ ! -s "${ROOT}/final_checkpoint.txt" ]]; then
  {
    echo "frontend=ResNet-152 canonical Global-Joint-Query"
    echo "mode=${MODE} token_dropout=${TOKEN_DROPOUT} synthetic_camera=0"
    echo "init=${INIT} epochs=4 lr=1e-6 view_weights=3,1,1"
    sha256sum "${CFG}" "${INIT}"
  } >"${ROOT}/manifest.txt"
  cd "${REPO}"
  "${PY}" -u run/train_rumpl.py --cfg "${CFG}" --gpus 0 --workers 8 \
    --seed 0 --train-mmpose-type "${TYPE}" --test-mmpose-type "${TYPE}" \
    --validate-on-two-datasets 0 --use-mmpose-val 1 \
    --apply-noise-missing 0 --missing-level 0.0 --exp-name "${TAG}" \
    >"${ROOT}/train.log" 2>&1
  model_dir=$(find "${MODEL_ROOT}" -maxdepth 1 -type d -name "${TAG}_*" \
    -print | sort | tail -1)
  checkpoint=${model_dir}/final_state.pth.tar
  test -s "${checkpoint}"
  printf '%s\n' "${model_dir}" >"${ROOT}/model_dir.txt"
  printf '%s\n' "${checkpoint}" >"${ROOT}/final_checkpoint.txt"
fi

checkpoint=$(<"${ROOT}/final_checkpoint.txt")
for views in 2 3 4; do
  eval_dir=${ROOT}/eval/V${views}
  [[ -s "${eval_dir}/table2.json" ]] && continue
  mkdir -p "${eval_dir}"
  RUMPL_EVAL_STRICT=1 "${PY}" -u "${REPO}/run/eval_rumpl_checkpoint.py" \
    --cfg "${CFG}" --checkpoint "${checkpoint}" --output-dir "${eval_dir}" \
    --workers 8 --gpu 0 --use-mmpose-val true \
    --flip-lower-body-kp-test false --test-on-all-cameras true \
    --n-views-combinations "${views}" --model-num-views 4 \
    --test-mmpose-type "${TYPE}" >"${eval_dir}/eval.log" 2>&1
  pred=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  "${PY}" "${REPO}/run/eval_h36m_table2.py" --dict-pkl "${pred}" \
    --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1
done

date --iso-8601=seconds >"${ROOT}/COMPLETED"
echo "[ResNet camera token ablation] complete ${MODE}"
