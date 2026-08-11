#!/usr/bin/env bash
set -euo pipefail

# G4: conf+geom with fusion-query geometry and a smaller geometry init.
exec /home/lixiaob/cjy/OpenRUMPL_baseline_audit/run_exact_rumpl_gbt_20260723.sh \
  0 G4_gbt_fusion_geom005_exact_seed0_20260724 1 0.05 1 1 0.1
