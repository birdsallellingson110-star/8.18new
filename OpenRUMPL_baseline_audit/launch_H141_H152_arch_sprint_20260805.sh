#!/usr/bin/env bash
# H141-H152: architecture sprint — module add/remove/replace on fixed HRNet→A1D→H21 input.
# Goal: beat GBT Table-I (V2/V3/V4) starting from H81/H76 fusion stack.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 VARIANT PHYSICAL_GPU" >&2
  echo "Variants: h141_no_pft_repeat h142_relview_w322 h143_gjv2_h81 h144_graph_res_h81" >&2
  echo "          h145_no_tri_anchor h146_set_decoder_h76 h147_vft_mask04 h148_jv1_biased_h81" >&2
  echo "          h149_gate_gjv2_w322 h150_gate_relview_w322 h151_bone_ray01 h152_shallow_vft" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit

export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export BASE_OVERRIDE=${ROOT}/H141_H152_arch_sprint
export SEED_OVERRIDE=0
export RUMPL_WORKERS=8

# H81-style geometry input (fixed pipeline)
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_ANCHOR_CENTER_PER_JOINT=0
export RUMPL_INPUT_HARMONIC_L=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export RUMPL_PER_JOINT_RESIDUAL_GATE=0
export RUMPL_POST_PFT_GRAPH_RESIDUAL=0
export RUMPL_JOINT_SPECIFIC_HEAD=0
export RUMPL_RELATIVE_VIEW_FUSION=0
export RUMPL_GBT_SET_DECODER=0
export GBT_LEARNABLE_BIAS=0
export GBT_USE_CONF_BIAS=0
export GBT_USE_GEOM_BIAS=0
export GBT_GLOBAL_JV_DEPTH=0
export GBT_GLOBAL_JV_BIASED=0
export GBT_GLOBAL_JV_GATED=0
export VFT_FULL_RANDOM_MASK=0
export RUMPL_PFT_REPEAT_LAST=1
export RUMPL_TRI_ANCHOR=1
export REPROJ_LAMBDA=0
export RAY_LAMBDA=0
export BONE_LAMBDA=0
export MONO_W=0

# Multi-view curriculum (V3/V4 aware, keep V2 mass)
export TRAIN_FIXED_NUM_VIEWS=2
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
export RUMPL_RANDOM_VIEW_SUBSET=1
export RUMPL_VIEW_COUNT_WEIGHTS=3,2,2
export RUMPL_N_VIEWS_TRAIN_TEST_ALL="${RUMPL_N_VIEWS_TRAIN_TEST_ALL:-4}"
export RUMPL_EVAL_STRICT="${RUMPL_EVAL_STRICT:-0}"

export RUMPL_STACK_FROM=H81

case "${variant}" in
  h141_no_pft_repeat)
    export CODE_OVERRIDE=H141
    export TAG_OVERRIDE=H141_H81_noPftRepeatLast_w322_ftH81_workers8_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; remove public PFT double-last-block quirk"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_PFT_REPEAT_LAST=0
    ;;
  h142_relview_w322)
    export CODE_OVERRIDE=H142
    export TAG_OVERRIDE=H142_H81_relViewFusion_w322_ftH81_workers8_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; relative view fusion before VFT (GIFFormer-style)"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_RELATIVE_VIEW_FUSION=1
    ;;
  h143_gjv2_h81)
    export CODE_OVERRIDE=H143
    export TAG_OVERRIDE=H143_H81_globalJV2_rezero_w322_ftH81_workers8_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; ReZero global joint-view depth=2 (H123 mechanism on H81)"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export GBT_GLOBAL_JV_DEPTH=2
    export GBT_GLOBAL_JV_GATED=1
    ;;
  h144_graph_res_h81)
    export CODE_OVERRIDE=H144
    export TAG_OVERRIDE=H144_H81_postPftGraphRes_w322_ftH81_workers8_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; skeleton message-passing after PFT (H82 on H81)"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_POST_PFT_GRAPH_RESIDUAL=1
    ;;
  h145_no_tri_anchor)
    export CODE_OVERRIDE=H145
    export TAG_OVERRIDE=H145_H81_noTriAnchor_w322_ftH81_workers8_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; disable triangulation-residual anchor (simpler fusion)"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_TRI_ANCHOR=0
    ;;
  h146_set_decoder_h76)
    export CODE_OVERRIDE=H146
    export TAG_OVERRIDE=H146_H76_gbtSetDecoder_w322_ftH76_workers8_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H76 ft; replace VFT path with GBT set encoder-decoder"
    export RUMPL_STACK_FROM=H76
    export RUMPL_GBT_SET_DECODER=1
    export RUMPL_GBT_SET_DEPTH=3
    export RUMPL_GBT_SET_DECODER_DEPTH=2
    export RUMPL_TRI_ANCHOR=0
    ;;
  h147_vft_mask04)
    export CODE_OVERRIDE=H147
    export TAG_OVERRIDE=H147_H81_vftMask04_w322_ftH81_workers8_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; VFT full random view mask 0.4 (MTF/GIFFormer)"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export VFT_FULL_RANDOM_MASK=0.4
    export VFT_MASK_MIN_VIEWS=2
    ;;
  h148_jv1_biased_h81)
    export CODE_OVERRIDE=H148
    export TAG_OVERRIDE=H148_H81_globalJV1_confgeom_w322_ftH81_workers8_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; global JV depth=1 with conf+geom bias (not VFT bias)"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export GBT_GLOBAL_JV_DEPTH=1
    export GBT_GLOBAL_JV_BIASED=1
    export GBT_GLOBAL_JV_GATED=1
    ;;
  h149_gate_gjv2_w322)
    export CODE_OVERRIDE=H149
    export TAG_OVERRIDE=H149_H81_gateGlobalJV2_w322_ftH81_workers8_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; per-joint gate + global JV2 (combine best prior lines)"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export GBT_GLOBAL_JV_DEPTH=2
    export GBT_GLOBAL_JV_GATED=1
    ;;
  h150_gate_relview_w322)
    export CODE_OVERRIDE=H150
    export TAG_OVERRIDE=H150_H81_gateRelView_w322_ftH81_workers8_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; per-joint gate + relative view fusion"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_RELATIVE_VIEW_FUSION=1
    ;;
  h151_bone_ray01)
    export CODE_OVERRIDE=H151
    export TAG_OVERRIDE=H151_H81_boneRay01_w322_ftH81_workers8_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; bone+ray auxiliary losses (no attention bias)"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export BONE_LAMBDA=0.01
    export RAY_LAMBDA=0.01
    ;;
  h152_shallow_vft)
    export CODE_OVERRIDE=H152
    export TAG_OVERRIDE=H152_H81_singlePftBlock_w322_ftH81_workers8_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; no repeat-last + token dropout 0.1 on views"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_PFT_REPEAT_LAST=0
    export GBT_TOKEN_DROPOUT=0.1
    export GBT_TOKEN_DROPOUT_EPOCHS=999
    ;;
  *)
    echo "unsupported variant: ${variant}" >&2
    exit 2
    ;;
esac

# shellcheck source=/dev/null
source "${AUDIT}/experiment_should_skip.sh"
if experiment_should_skip_train "${TAG_OVERRIDE}" 2>/dev/null; then
  echo "[${CODE_OVERRIDE}] skip (${TAG_OVERRIDE} in skip registry)"
  exit 0
fi

exec bash "${AUDIT}/launch_H59_h58_balanced_full_20260802.sh" "${physical_gpu}"
