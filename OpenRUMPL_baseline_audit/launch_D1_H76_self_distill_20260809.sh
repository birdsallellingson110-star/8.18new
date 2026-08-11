#!/usr/bin/env bash
# D1/D2: H76 real-H36M full-view -> hard two-view self-distillation.
# The teacher is the same RUMPL model under no-grad with all four cameras; the
# optional fixed_eval variant also disables DropPath/Dropout for teacher and
# candidates.  The student is trained on the hardest two-camera subset.  This
# is an end-to-end training-loss change only: inference remains one H76 model.
set -euo pipefail

variant=${1:?usage: $0 <legw09|legw07|fixed_eval_legw07|dualhard_aux05_legw07> <physical_gpu>}
physical_gpu=${2:?usage: $0 <legw09|legw07|fixed_eval_legw07|dualhard_aux05_legw07> <physical_gpu>}
case "${variant}" in
  legw09) legw=0.9; distill_w=1.0; aux_w=0.0; label=hardV_legw09; teacher_eval=0 ;;
  legw07) legw=0.7; distill_w=1.0; aux_w=0.0; label=hardV_legw07; teacher_eval=0 ;;
  legw07_retry) legw=0.7; distill_w=1.0; aux_w=0.0; label=hardV_legw07_retry; teacher_eval=0 ;;
  low03_legw07) legw=0.7; distill_w=0.3; aux_w=0.0; label=hardV_dw03_legw07; teacher_eval=0 ;;
  fixed_eval_legw07) legw=0.7; distill_w=1.0; aux_w=0.0; label=hardV_fixedEval_legw07; teacher_eval=1 ;;
  dualhard_aux05_legw07) legw=0.7; distill_w=1.0; aux_w=0.5; label=hardV_legw07_aux05; teacher_eval=0 ;;
  *) echo "unknown variant: ${variant}" >&2; exit 2 ;;
esac

AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
H21=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H46_H48_root_cause/H48_H21_a1d_matched_v2focus_reg005/final.pth

export CODE_OVERRIDE="D1_H76_selfDistill_${label}_seed0"
export BASE_OVERRIDE="${ROOT}/D1_D2_H76_self_distill_20260809"
export TAG_OVERRIDE="D1_H76_selfDistill_${label}_w322_workers12_seed0_20260809"
export CONTROL_NOTE_OVERRIDE="H76 centered-Plucker; full-view teacher to hard 2-view student; distill_w=${distill_w}; aux_multik_w=${aux_w}; leg_distill_w=${legw}; teacher_eval=${teacher_eval}; H76 end-to-end fine-tune"
export SEED_OVERRIDE=0
export RUMPL_STACK_FROM=H76
export TYPE_OVERRIDE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
export H21_OVERRIDE="${H21}"
export RUMPL_FINETUNE_LR=1e-4
export RUMPL_END_EPOCH=8
export RUMPL_WORKERS=12

# H76 representation and all optional branches are explicit.  In particular,
# do not inherit a bias/adapter flag from an interactive shell.
export RUMPL_TRI_ANCHOR=1
export RUMPL_ANCHOR_CENTERED_RAYS=1
export RUMPL_INPUT_PLUCKER=1
export RUMPL_PFT_REPEAT_LAST=1
export RUMPL_RANDOM_VIEW_SUBSET=1
export RUMPL_VIEW_COUNT_WEIGHTS=3,1,1
export TRAIN_FIXED_NUM_VIEWS=2
export TRAIN_FIXED_NUM_VIEWS_EPOCHS=3
unset RUMPL_DISABLE_FIXED_VIEW_CURRICULUM || true
export RUMPL_TRAIN_SCOPE=all
export RUMPL_SKELETON_VIEW_RELIABILITY=0
export RUMPL_CONFIDENCE_VIEW_BIAS=0
export RUMPL_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_VIEW_BIAS=0
export RUMPL_JOINT_GEOMETRY_VIEW_BIAS=0
export RUMPL_JOINT_CONFIDENCE_TOKEN_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_TOKEN_RESIDUAL=0
export RUMPL_JOINT_CONFIDENCE_CONDITIONAL_RESIDUAL=0
export RUMPL_JOINT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_POST_PFT_GEOMETRY_CONDITIONAL_RESIDUAL=0
export RUMPL_PER_JOINT_RESIDUAL_GATE=0
export RUMPL_POST_PFT_GRAPH_RESIDUAL=0
export RUMPL_JOINT_SPECIFIC_HEAD=0
export RUMPL_RELATIVE_VIEW_FUSION=0
export RUMPL_GEOMETRY_UNCERTAINTY_TOKEN=0
export GBT_GLOBAL_JV_DEPTH=0 GBT_GLOBAL_JV_BIASED=0 GBT_GLOBAL_JV_GATED=0
export RUMPL_GBT_SET_DECODER=0
export GBT_LEARNABLE_BIAS=0 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export VFT_FULL_RANDOM_MASK=0
export RUMPL_SKIP_VFT=0 RUMPL_SKIP_PFT=0 RUMPL_VFT_DEPTH=0
export REPROJ_LAMBDA=0 RAY_LAMBDA=0 BONE_LAMBDA=0 MONO_W=0 MONO_GT_W=0
export DEPRO_LAMBDA=0 CAA_LAMBDA=0

# Full-view teacher -> hard-view student.  Distillation is only a training
# objective; no teacher checkpoint or extra view is needed at inference.
export DISTILL_W="${distill_w}"
export DISTILL_TEACHER_EVAL="${teacher_eval}"
export STUDENT_GT_W=1
export FEAT_DISTILL_W=0
export STUDENT_VIEWS=2
export HARD_VIEW_MINING=1
export HARD_VIEW_CAND=6
export AUX_MULTIK_W="${aux_w}"
export AUX_MULTIK_MIN=3
export AUX_MULTIK_MAX=4
export LEG_DISTILL_W="${legw}"

exec bash "${AUDIT}/launch_H59_h58_balanced_full_20260802.sh" "${physical_gpu}"
