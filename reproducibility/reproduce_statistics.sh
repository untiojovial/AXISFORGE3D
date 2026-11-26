#!/usr/bin/env bash
# Partial reproduction from a clean checkout: recomputes, from the committed per-object outputs only, the
# statistical results of Phase VII and of the extension, and checks them against the committed JSON files.
# It does not re-run planning or physics (that needs the datasets; see REPRODUCIBILITY.md, "full reproduction").
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
out=$(mktemp -d)
echo "== P2 statistical audit"
$PY tools/research/phase7_extension/p2_audit.py --results research/phase7/results --out "$out/p2.json" > /dev/null
$PY - "$out/p2.json" <<'EOF'
import json, sys
new = json.load(open(sys.argv[1])); old = json.load(open('research/phase7_extension/results/p2_statistical_audit.json'))
keys = ('paired_table', 'pre_specified', 'wald_mle', 'tango', 'newcombe_method10', 'primary_family_full_precision', 'fragility_pre_specified')
bad = [k for k in keys if new[k] != old[k]]
print('identical' if not bad else f'DIFFERENT: {bad}'); sys.exit(1 if bad else 0)
EOF
echo "== corrections addendum (A, E, F)"
$PY tools/research/phase7_extension/corrections_addendum.py --results research/phase7/results --out "$out/add.json" > /dev/null
cmp -s "$out/add.json" research/phase7_extension/results/corrections_addendum.json && echo identical || { echo DIFFERENT; exit 1; }
echo "== start verification of the Phase VII figures"
$PY tools/research/phase7_extension/verify_start.py --out "$out/start.json" > /dev/null 2>&1 || true
$PY - "$out/start.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); print('all quoted figures match' if d['all_match'] else 'MISMATCH'); sys.exit(0 if d['all_match'] else 1)
EOF
echo "== extension tests (statistics, exported models, licences, fallback bound)"
$PY -m pytest -q tests/phase7_extension
echo "done: partial reproduction (statistics from committed per-object data) succeeded"
