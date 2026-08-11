#!/usr/bin/env bash
# Resolve RUMPL_INIT_CHECKPOINT from RUMPL_STACK_FROM={H76|H81} (or explicit RUMPL_INIT_CHECKPOINT).
set -euo pipefail

ROOT=${RUMPL_STACK_ROOT:-/mnt/data/cjyoutput/open_source_fusion_audit_20260731}

if [[ -n "${RUMPL_INIT_CHECKPOINT:-}" ]]; then
  test -s "${RUMPL_INIT_CHECKPOINT}"
  export RUMPL_INIT_CHECKPOINT
  return 0 2>/dev/null || exit 0
fi

parent=${RUMPL_STACK_FROM:-}
case "${parent}" in
  H76|h76)
    ckpt_file="${ROOT}/H76_h50_centered_plucker/checkpoints/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803.txt"
    ;;
  H81|h81)
    ckpt_file="${ROOT}/H81_H83_targeted_pft/checkpoints/H81_H76_perJointResidualGate_workers12_seed0_20260803.txt"
    ;;
  '')
    return 0 2>/dev/null || exit 0
    ;;
  *)
    echo "rumpl_stack_from_parent: unknown RUMPL_STACK_FROM=${parent}" >&2
    exit 2
    ;;
esac

if [[ ! -s "${ckpt_file}" ]]; then
  echo "rumpl_stack_from_parent: missing ${ckpt_file}" >&2
  exit 1
fi
export RUMPL_INIT_CHECKPOINT="$(tr -d '\r\n' <"${ckpt_file}")"
test -s "${RUMPL_INIT_CHECKPOINT}"
export RUMPL_INIT_CHECKPOINT
