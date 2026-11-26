"""Build the new EGAD set (positions 300-499 of the Phase VII seeded draw) with the frozen Phase VII preparation rules.

    python -I tools/research/phase7_extension/build_newset.py --egad EGAD_DIR --selection research/phase7/egad_selection.json \
        --binary target/release/axisforge --out NEW_POOL --positions 300 500 --jobs 1

This is a restart-safe wrapper around tools/research/phase7/egad_prepare.py: it imports that file's build_one,
the frozen function, unchanged, with the same scale rule, stable pose, cell placement and CoACD settings. Each
object's row is kept in NEW_POOL/rows/<name>.json, and an object whose row exists is not rebuilt. built.json
has the same format as Phase VII's. Mesh preparation only: no planning, physics or model is run here.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools/research/phase7'))
import egad_prepare as ep  # noqa: E402


def one(task):
    out = Path(task[2])
    row_path = out/'rows'/f'{task[1]}.json'
    if row_path.exists():
        return json.loads(row_path.read_text())
    row = ep.build_one(task)
    row_path.write_text(json.dumps(row))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('egad', 'selection', 'binary', 'out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--positions', type=int, nargs=2, required=True); p.add_argument('--jobs', type=int, default=1)
    a = p.parse_args()
    sel = json.loads(Path(a.selection).read_text())
    out = Path(a.out); (out/'rows').mkdir(parents=True, exist_ok=True)
    empty = out/'empty.axisforge'
    if not empty.exists():
        subprocess.run([a.binary, 'demo', '--no-parts', '--out', str(empty)], check=True, capture_output=True)
    find = {q.name: q for q in Path(a.egad).glob('**/egad_*_set/*.obj')}
    lo, hi = a.positions
    tasks = [(find[r['file']], f"egad_{r['cell']}", str(out), a.binary, str(empty)) for r in sel['primary'][lo:hi]]
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for row in pool.map(one, tasks):
            rows.append(row)
            print(row['name'], row.get('kept'), row.get('collision_pieces'), row.get('seconds'), flush=True)
    (out/'built.json').write_text(json.dumps(dict(rows=rows, empty_sha256=ep.sha256(empty), positions=[lo, hi], evaluation_set=False,
                                                  builder='tools/research/phase7_extension/build_newset.py (egad_prepare.build_one)'),
                                             indent=1) + '\n')


if __name__ == '__main__':
    main()
