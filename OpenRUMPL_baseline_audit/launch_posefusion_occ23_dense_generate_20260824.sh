#!/usr/bin/env bash
# Generate the frozen dense stride-5 VOC-object Occ-2/Occ-3 evaluation images.
# This is a temporal extension of the public Human3.6M-Occluded generator.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
GT=${DATA}/data/datasets/annot_temporal_5_5/h36m_validation.pkl
IMAGES=${DATA}/images
VOC=/mnt/data/cjydata/datasets/pascal_voc2012/VOCdevkit/VOC2012
UPSTREAM=/mnt/data/cjydata/reference_code/human3.6m-occluded
ROOT=/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824

for required in "${GT}" "${IMAGES}" "${VOC}" "${UPSTREAM}"; do
  test -e "${required}"
done
mkdir -p "${ROOT}/logs"

run_variant() {
  local views=$1
  local variant=occ${views}
  local out=${ROOT}/${variant}
  if [[ -s "${out}/protocol_manifest.json" ]]; then
    "${PY}" - "${out}/protocol_manifest.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["status"] == "complete"
assert p["output"]["groups_written"] == 26269
assert p["official_parameters"]["objects_per_occluded_view"] == 2
assert p["official_parameters"]["object_scale_uniform_relative_to_person_min_dimension"] == [0.2, 0.5]
assert p["randomness"]["seed"] == 42
PY
    echo "[dense ${variant}] existing complete manifest verified"
    return
  fi
  "${PY}" -u "${AUDIT}/generate_h36m_occ_official_adapter_20260823.py" \
    --validation-pkl "${GT}" --images-root "${IMAGES}" \
    --pascal-voc-root "${VOC}" --official-repo "${UPSTREAM}" \
    --output-root "${out}" --num-occluded-views "${views}" \
    --objects-per-occluded-view 2 --scale-min 0.2 --scale-max 0.5 \
    --seed 42 --protocol-label "public-generator-derived-occ${views}-dense-s020-050" \
    --resume \
    >"${ROOT}/logs/generate_${variant}.log" 2>&1
  echo "[dense ${variant}] generated"
}

run_variant 2 & p2=$!
run_variant 3 & p3=$!
failed=0
wait "${p2}" || failed=1
wait "${p3}" || failed=1
(( failed == 0 ))
date --iso-8601=seconds >"${ROOT}/generation_COMPLETED"
echo "[dense VOC Occ-2/Occ-3 generation] complete"
