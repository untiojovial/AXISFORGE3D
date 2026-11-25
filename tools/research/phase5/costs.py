"""Time and storage costs of building and using the library.

    python tools/research/phase5/costs.py --catalogues 80=cat80/ ... --names lib.txt test.txt \
        --render-logs DIR ... --rows test.json --dynamics dynamics.json --library-files f1 f2 --out costs.json
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import catalogue  # noqa: E402


def describe(values):
    v = np.asarray(values, float)
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                total=float(v.sum())) if len(v) else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--catalogues', nargs='+', required=True); p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--render-logs', nargs='*', default=[]); p.add_argument('--rows'); p.add_argument('--dynamics')
    p.add_argument('--library-files', nargs='*', default=[]); p.add_argument('--out', required=True)
    a = p.parse_args()
    names = [n for f in a.names for n in Path(f).read_text().split()]
    out = dict(parts=len(names), catalogues={})
    for spec in a.catalogues:
        g, d = spec.split('=', 1)
        metas = [catalogue.load(Path(d)/f'{n}.npz')['meta'] for n in names]
        sizes = [(Path(d)/f'{n}.npz').stat().st_size for n in names]
        out['catalogues'][g] = dict(generate_s=describe([m['generate_s'] for m in metas]),
                                    exhaustive_solve_s=describe([m['solve_s'] for m in metas]),
                                    candidates=describe([m['candidates'] for m in metas]),
                                    npz_bytes=describe(sizes))
    seconds = []
    for d in a.render_logs:
        for f in Path(d).glob('*.log'):
            for line in f.read_text().splitlines():
                m = re.match(r'^\S+ ([0-9.]+) s$', line.strip())
                if m:
                    seconds.append((f.stem.rstrip('0123456789'), float(m.group(1))))
    for kind in sorted({k for k, _ in seconds}):
        out[f'render_{kind}_s'] = describe([s for k, s in seconds if k == kind])
    if a.rows:
        doc = json.loads(Path(a.rows).read_text())
        out['train_s'] = describe([g['train_s'] for g in doc['groups']])
        out['query_all_orderings_s'] = describe([r['query_s'] for r in doc['rows'] if 'query_s' in r])
        out['evaluation_wall_s'] = doc['wall_s']
    if a.dynamics:
        dyn = json.loads(Path(a.dynamics).read_text())
        per = [v['sim_seconds']/max(v['sims'], 1) for v in dyn['parts'].values() if 'sims' in v]
        out['dynamic_trial_s'] = describe(per)
        out['dynamic_wall_s'] = dyn['wall_s']
    out['library_files_bytes'] = {Path(f).name: Path(f).stat().st_size for f in a.library_files}
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'catalogues'}, indent=1)[:2000])


if __name__ == '__main__':
    main()
