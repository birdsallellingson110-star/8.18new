#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
RUMPL=/home/lixiaob/cjy/OpenRUMPL/RUMPL
CFG=${BASE}/H12_real_h36m_train.yaml
RUN=${BASE}/output/multiview_h36m_rumpl/multiview_rumpl_999/H12_real_h36m_rumpl_nomodule_fullimageval_seed0_20260729_2026-07-29_18-01-38
CKPT=${RUN}/model_best.pth.tar
OUT=${BASE}/H12_model_best_fullimage_multiview_eval
TYPE=mmpose_hrnet_coco_inferencer_legswap

export CUDA_VISIBLE_DEVICES=1
export RUMPL_FIX_SCHEDULER_ORDER=1
unset GBT_LEARNABLE_BIAS GBT_USE_CONF_BIAS GBT_USE_GEOM_BIAS
unset GBT_GLOBAL_JV_DEPTH GBT_GLOBAL_JV_BIASED GBT_GLOBAL_JV_GATED
unset RUMPL_TRI_ANCHOR RUMPL_KPA

mkdir -p "${OUT}"
test -s "${CKPT}"
cd "${RUMPL}"

for n_views in 2 3 4; do
  eval_dir=${OUT}/V${n_views}
  mkdir -p "${eval_dir}"

  if [[ "${n_views}" -eq 2 ]]; then
    test_views=(1 2)
  elif [[ "${n_views}" -eq 3 ]]; then
    test_views=(1 2 3)
  else
    test_views=(1 2 3 4)
  fi

  "${PY}" -u run/eval_rumpl_checkpoint.py \
    --cfg "${CFG}" \
    --checkpoint "${CKPT}" \
    --output-dir "${eval_dir}" \
    --workers 16 \
    --gpu 0 \
    --use-mmpose-val true \
    --flip-lower-body-kp-test true \
    --test-views "${test_views[@]}" \
    --test-on-all-cameras true \
    --test-mmpose-type "${TYPE}" \
    >"${eval_dir}/eval.log" 2>&1

  prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
  test -n "${prediction}"
  "${PY}" run/eval_h36m_table2.py \
    --dict-pkl "${prediction}" \
    --output-json "${eval_dir}/table2.json" \
    >"${eval_dir}/table2.log" 2>&1
  cat "${eval_dir}/table2.log"
done

