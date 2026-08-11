#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/data/cjycode/open_source_fusion_20260731/PPT/multi-view-PPT
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
OUT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/PPT_official_h36m_protocol_matched
CFG="$AUDIT/configs/ppt_h36m_protocol_matched_4view.yaml"
PKL=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets/annot_filtered_5_64/h36m_validation.pkl
VENV=/home/lixiaob/cjy/rumpl_venv310/bin/python

mkdir -p "$OUT"
while tmux has-session -t cjy_h20_fixed2 2>/dev/null \
   || tmux has-session -t cjy_h20_random24 2>/dev/null; do
    echo "[PPT] waiting for H20 to release GPU 1"
    sleep 30
done

cd "$REPO"
export PPT_H36M_ALREADY_SUBSAMPLED=1
export PPT_H36M_SWAP_LEGS=1
export CUDA_VISIBLE_DEVICES=1
PRED="$OUT/multiview_h36m/multiview_ppt_999/ppt_h36m_protocol_matched_4view/predicted_heatmaps.h5"
if [[ ! -s "$PRED" ]]; then
    "$VENV" -u run/pose2d/valid.py \
        --cfg "$CFG" \
        --gpus 0 \
        --workers 8 \
        --model-file "$REPO/models/multiview_ppt_h36m_official_model_best.pth.tar" \
        > "$OUT/official_pose2d_valid.log" 2>&1
fi
touch "$OUT/official_pose2d_complete.marker"

"$VENV" -u run/pose3d/estimate_tri.py --cfg "$CFG" \
    > "$OUT/official_pose3d_triangulation.log" 2>&1

PYTHONPATH="$REPO/lib:$AUDIT" "$VENV" -u \
    "$AUDIT/eval_h36m_ppt_variable_views.py" \
    --cfg "$CFG" \
    --checkpoint "$REPO/models/multiview_ppt_h36m_official_model_best.pth.tar" \
    --views 2 3 4 \
    --batch-size 4 \
    --workers 8 \
    --device cuda:0 \
    --output "$OUT/PPT_official_RUMPL_table2_V234.json" \
    > "$OUT/RUMPL_table2_V234.log" 2>&1
