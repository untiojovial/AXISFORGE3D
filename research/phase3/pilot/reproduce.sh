#!/usr/bin/env bash
# Usage from repository root: bash research/phase3/pilot/reproduce.sh DATA_DIR
# Assumes pinned OBJ/STLs, manifest and externally stored checkpoint are available.
set -euo pipefail
data=$(realpath "${1:?external data directory required}")
binary=$(realpath "${AXISFORGE_BIN:-target/release/axisforge}")
python_bin=${PYTHON:-python}
protocol=research/phase3/pilot
vision="$data/vision-development-v3"
catalogues="$data/pilot-expanded/catalogs"
run="$data/run-development"
"${python_bin}" -m research_grasping.phase3.vision_pipeline --root "$data" --out "$vision" \
  --checkpoint "$data/weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt" --resolution 96
mkdir -p "$catalogues"
mapfile -t ids < <("$python_bin" -c 'import json,sys;print("\n".join(row["object_id"] for row in json.load(open(sys.argv[1]))["objects"] if row["split"] != "test"))' "$data/manifest.json")
for id in "${ids[@]}"; do
  if [[ ! -f "$catalogues/$id.json" ]]; then
    "$python_bin" -m research_grasping.phase3.prepare_pilot --manifest "$data/manifest.json" \
      --binary "$binary" --vision-archive "$vision/summary.npz" --output "$catalogues" --object "$id" --max-candidates 48
  fi
done
catalogue_args=()
for id in "${ids[@]}"; do catalogue_args+=("$catalogues/$id.json"); done
# Preserve the lexicographic order of training objects in the published run.
mapfile -t catalogue_args < <(printf '%s\n' "${catalogue_args[@]}" | LC_ALL=C sort)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$python_bin" -m research_grasping.phase3.pilot_cli --data "$data" --protocol "$protocol" \
  --binary "$binary" --out "$run" --catalogues "${catalogue_args[@]}" --budget 10 --seeds 11 29 47 \
  --methods random sobol_projected target_local contextual multitask_fixed rgpe equal no_evidence adaptive
"$python_bin" -m research_grasping.phase3.summarize_pilot "$run" --out "$data/results-development"
