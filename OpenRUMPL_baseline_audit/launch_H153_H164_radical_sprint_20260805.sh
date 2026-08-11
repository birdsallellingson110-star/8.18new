#!/usr/bin/env bash
# H153-H164: radical fusion ablations (skip VFT/PFT, shallow VFT, mean-fuse).
# Rationale: H81 wins with gate+tri-anchor; stacked GBT/VFT/temporal modules regressed.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 VARIANT PHYSICAL_GPU" >&2
  echo "Variants: h153_skip_vft h154_skip_vft_pft h155_jv2_skip_vft h156_vft1" >&2
  echo "          h157_no_tri_skip_vft h158_tri_only_no_pft h159_jv2_skip_vft_pft" >&2
  echo "          h160_h76_skip_vft h161_vft1_skip_pft h162_skip_vft_graph" >&2
  echo "          h163_h76_set_dec h164_skip_vft_relview" >&2
  exit 2
fi

variant=$1
physical_gpu=$2
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit

export H21_OVERRIDE=${ROOT}/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export BASE_OVERRIDE=${ROOT}/H153_H164_radical_sprint
export SEED_OVERRIDE=0
export RUMPL_WORKERS=12

export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_PER_JOINT_RESIDUAL_GATE=0
export RUMPL_TRI_ANCHOR=1
export RUMPL_PFT_REPEAT_LAST=1
export RUMPL_SKIP_VFT=0
export RUMPL_SKIP_PFT=0
export RUMPL_VFT_DEPTH=0
export RUMPL_RELATIVE_VIEW_FUSION=0
export RUMPL_POST_PFT_GRAPH_RESIDUAL=0
export RUMPL_GBT_SET_DECODER=0
export GBT_GLOBAL_JV_DEPTH=0
export GBT_GLOBAL_JV_GATED=0
export TRAIN_FIXED_NUM_VIEWS=2
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=8
export RUMPL_RANDOM_VIEW_SUBSET=1
export RUMPL_VIEW_COUNT_WEIGHTS=3,2,2
export RUMPL_N_VIEWS_TRAIN_TEST_ALL="${RUMPL_N_VIEWS_TRAIN_TEST_ALL:-4}"
export RUMPL_EVAL_STRICT="${RUMPL_EVAL_STRICT:-0}"
export RUMPL_STACK_FROM=H81

case "${variant}" in
  h153_skip_vft)
    export CODE_OVERRIDE=H153
    export TAG_OVERRIDE=H153_H81_skipVft_meanFuse_w322_ftH81_workers12_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; drop VFT transformer, conf-weighted mean fuse"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_SKIP_VFT=1
    ;;
  h154_skip_vft_pft)
    export CODE_OVERRIDE=H154
    export TAG_OVERRIDE=H154_H81_skipVftPft_minimal_w322_ftH81_workers12_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; mean fuse + no PFT (tri-anchor+gate+head only)"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_SKIP_VFT=1
    export RUMPL_SKIP_PFT=1
    ;;
  h155_jv2_skip_vft)
    export CODE_OVERRIDE=H155
    export TAG_OVERRIDE=H155_H81_globalJV2_skipVft_w322_ftH81_workers12_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; global JV2 then mean fuse (no VFT)"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export GBT_GLOBAL_JV_DEPTH=2
    export GBT_GLOBAL_JV_GATED=1
    export RUMPL_SKIP_VFT=1
    ;;
  h156_vft1)
    export CODE_OVERRIDE=H156
    export TAG_OVERRIDE=H156_H81_vftDepth1_w322_ftH81_workers12_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; single VFT block (depth=1)"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_VFT_DEPTH=1
    export RUMPL_PFT_REPEAT_LAST=0
    ;;
  h157_no_tri_skip_vft)
    export CODE_OVERRIDE=H157
    export TAG_OVERRIDE=H157_H81_noTri_skipVft_w322_ftH81_workers12_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H81 ft; no tri-anchor + mean fuse"
    export RUMPL_TRI_ANCHOR=0
    export RUMPL_SKIP_VFT=1
    export RUMPL_ANCHOR_CENTERED_RAYS=0
    ;;
  h158_tri_only_no_pft)
    export CODE_OVERRIDE=H158
    export TAG_OVERRIDE=H158_H81_triOnly_skipVftPft_w322_ftH81_workers12_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="tri-anchor + mean fuse, no gate no PFT"
    export RUMPL_SKIP_VFT=1
    export RUMPL_SKIP_PFT=1
    ;;
  h159_jv2_skip_vft_pft)
    export CODE_OVERRIDE=H159
    export TAG_OVERRIDE=H159_H81_jv2_skipVftPft_w322_ftH81_workers12_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="global JV2 + mean fuse + skip PFT"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export GBT_GLOBAL_JV_DEPTH=2
    export GBT_GLOBAL_JV_GATED=1
    export RUMPL_SKIP_VFT=1
    export RUMPL_SKIP_PFT=1
    ;;
  h160_h76_skip_vft)
    export CODE_OVERRIDE=H160
    export TAG_OVERRIDE=H160_H76_skipVft_meanFuse_w322_ftH76_workers12_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H76 ft; mean fuse replaces VFT"
    export RUMPL_STACK_FROM=H76
    export RUMPL_SKIP_VFT=1
    ;;
  h161_vft1_skip_pft)
    export CODE_OVERRIDE=H161
    export TAG_OVERRIDE=H161_H81_vft1_skipPft_w322_ftH81_workers12_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="1-layer VFT + skip PFT + gate"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_VFT_DEPTH=1
    export RUMPL_SKIP_PFT=1
    ;;
  h162_skip_vft_graph)
    export CODE_OVERRIDE=H162
    export TAG_OVERRIDE=H162_H81_skipVft_graphRes_w322_ftH81_workers12_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="mean fuse + skeleton graph after PFT path"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_SKIP_VFT=1
    export RUMPL_POST_PFT_GRAPH_RESIDUAL=1
    ;;
  h163_h76_set_dec)
    export CODE_OVERRIDE=H163
    export TAG_OVERRIDE=H163_H76_setDecoder_w322_ftH76_workers12_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="H76 ft; GBT set-decoder replaces VFT"
    export RUMPL_STACK_FROM=H76
    export RUMPL_GBT_SET_DECODER=1
    export RUMPL_TRI_ANCHOR=0
    ;;
  h164_skip_vft_relview)
    export CODE_OVERRIDE=H164
    export TAG_OVERRIDE=H164_H81_skipVft_relView_w322_ftH81_workers12_seed0_20260805
    export CONTROL_NOTE_OVERRIDE="relative view + mean fuse (no VFT token attn)"
    export RUMPL_PER_JOINT_RESIDUAL_GATE=1
    export RUMPL_RELATIVE_VIEW_FUSION=1
    export RUMPL_SKIP_VFT=1
    ;;
  *)
    echo "unsupported: ${variant}" >&2
    exit 2
    ;;
esac

if [[ -n "${RUMPL_TAG_OVERRIDE:-}" ]]; then
  export TAG_OVERRIDE="${RUMPL_TAG_OVERRIDE}"
fi

source "${AUDIT}/experiment_should_skip.sh"
if experiment_should_skip_train "${TAG_OVERRIDE}" 2>/dev/null; then
  echo "[${CODE_OVERRIDE}] skip (${TAG_OVERRIDE})"
  exit 0
fi

exec bash "${AUDIT}/launch_H59_h58_balanced_full_20260802.sh" "${physical_gpu}"
