#!/usr/bin/env bash
# Recover post-training evaluation if train_rumpl exits non-zero after saving.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 EXPERIMENT_TAG LAUNCHER [LAUNCHER_ARGS...]" >&2
  exit 2
fi

tag=$1
launcher=$2
shift 2

while pgrep -f "run/train_rumpl.py .*--exp-name ${tag}($| )" >/dev/null; do
  sleep 30
done

exec bash "${launcher}" "$@"
