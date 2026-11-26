#!/usr/bin/env bash
# Run inside the image built from reproducibility/Dockerfile.repro.
set -euo pipefail
cd /work
echo "== statistical reproduction and extension tests"
PYTHON=/venv/bin/python bash reproducibility/reproduce_statistics.sh
echo "== native planner: learned trees against scikit-learn fixture (cargo test)"
cargo test --release --locked -p axisforge-grasping --test learned_fixture 2>&1 | grep -E "test result|passed|failed"
if [ -d /data/pool ]; then
  echo "== planner sample: native decisions on mounted EGAD projects against the committed validation"
  /venv/bin/python - <<'PY'
import json, subprocess
from pathlib import Path
val = json.loads(Path('research/phase7_extension/results/native_inference_validation.json').read_text())['level2']
names = sorted(n for n in val if n.startswith('egad_') and (Path('/data/pool')/n/'cell.axisforge').exists())
ok = bad = 0
for n in names:
    for mode in ('joint', 'hybrid'):
        rep = json.loads(subprocess.run(['target/release/axisforge', 'plan', '--json', '--nr-grasps', '1', '--learned',
                                         'research/phase7_extension/native/learned_order_phase7.json', '--order', mode,
                                         f'/data/pool/{n}/cell.axisforge', n], capture_output=True, text=True, check=True).stdout)
        got = [rep['candidatesExamined'], rep['plans'][0]['candidate']['label'] if rep['plans'] else None]
        if got == val[n]['modes'][mode]['native']: ok += 1
        else: bad += 1; print('MISMATCH', n, mode, got, val[n]['modes'][mode]['native'])
print(f'planner sample: {ok} identical decisions, {bad} mismatches on {len(names)} objects')
raise SystemExit(1 if bad else 0)
PY
fi
echo "docker validation finished"
