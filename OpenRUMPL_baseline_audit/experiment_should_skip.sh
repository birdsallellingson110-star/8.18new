#!/usr/bin/env bash
# Source this file, then call experiment_should_skip_train TAG or experiment_should_skip_variant VAR.
set -euo pipefail

SKIP_REG="${EXPERIMENT_SKIP_REGISTRY:-/mnt/data/cjyoutput/open_source_fusion_audit_20260731/EXPERIMENT_SKIP_REGISTRY_20260805.json}"

_experiment_skip_py() {
  local mode=$1 arg=$2
  SKIP_REG="${SKIP_REG}" MODE="${mode}" ARG="${arg}" python3 - <<'PY'
import json, os, sys
from pathlib import Path
reg = Path(os.environ["SKIP_REG"])
if not reg.is_file():
    sys.exit(1)
data = json.loads(reg.read_text())
mode = os.environ["MODE"]
arg = os.environ["ARG"]
if mode == "tag":
    tags = set(data.get("skip_train_tags") or [])
    retrain = set(data.get("retrain_duplicate_tags") or [])
    arg = os.environ["ARG"]
    if arg in tags or arg in retrain:
        sys.exit(0)
    sys.exit(1)
if mode == "variant":
    alias = (data.get("launch_alias_skip") or {}).get(arg)
    if alias and alias in set(data.get("skip_train_tags") or []):
        sys.exit(0)
    sys.exit(1)
sys.exit(1)
PY
}

# Returns 0 if training should be skipped, 1 if should run.
experiment_should_skip_train() {
  local tag=$1
  _experiment_skip_py tag "${tag}"
}

experiment_should_skip_variant() {
  local variant=$1
  _experiment_skip_py variant "${variant}"
}

experiment_refresh_skip_registry() {
  python3 /home/lixiaob/cjy/OpenRUMPL_baseline_audit/scan_experiment_skip_registry_20260805.py
}
