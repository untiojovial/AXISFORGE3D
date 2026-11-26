#!/usr/bin/env bash
# Picks each of the 16 YCB objects of the GRASPA benchmark layouts with the
# demo arm and gripper, using only the product's command line.
#
#   tools/verify/ycb_pick.sh target/release/axisforge /path/to/empty/folder
#
# The meshes come from the pybullet-object-models project (LGPL-2.1; the YCB
# models themselves are CC BY 4.0; see THIRD_PARTY_NOTICES.md). They are downloaded into the folder given,
# which should be new and empty: downloaded files are only read as mesh data.
set -euo pipefail
cli=$(realpath "$1")
work=$2
mkdir -p "$work"
cd "$work"
# Pinned on 2026-11-26: this commit reproduces the Phase IV objects exactly (same world bounds and candidate
# counts for all 16; see research/phase7_extension/ARTIFACT_REDISTRIBUTION_REGISTER.md). The original Phase IV
# download used the branch head of that time, whose commit was not recorded.
commit=57cc4007607f6bcbb0a5364ccab7636faf7a91e4
base=https://raw.githubusercontent.com/eleramp/pybullet-object-models/$commit/pybullet_object_models/ycb_objects
"$cli" demo --no-parts --out empty.axisforge > /dev/null
for name in Banana ChipsCan CrackerBox FoamBrick GelatinBox Hammer MasterChefCan MediumClamp \
            MustardBottle Pear PottedMeatCan PowerDrill Scissors Strawberry TennisBall TomatoSoupCan; do
  mkdir -p "$name"
  [ -s "$name/model.urdf" ] || curl -sf -o "$name/model.urdf" "$base/Ycb$name/model.urdf"
  mesh=$(grep -o 'mesh filename="[^"]*"' "$name/model.urdf" | head -1 | sed 's/mesh filename="//; s/"$//')
  [ -s "$name/visual.obj" ] || curl -sf -o "$name/visual.obj" "$base/Ycb$name/$mesh"
  mass=$(grep -o 'mass value="[^"]*"' "$name/model.urdf" | head -1 | sed 's/mass value="//; s/"$//')
  "$cli" add-object empty.axisforge "$name/visual.obj" --name "$name" --units m --mass "$mass" \
      --at 0.42 0 0.42 --out "$name/cell.axisforge" > /dev/null
  start=$(date +%s.%N)
  report=$("$cli" plan "$name/cell.axisforge" "$name" 2>/dev/null)
  summary=${report%%$'\n'*}
  end=$(date +%s.%N)
  printf '%s  (%.1f s)\n' "$summary" "$(echo "$end - $start" | bc)"
done
