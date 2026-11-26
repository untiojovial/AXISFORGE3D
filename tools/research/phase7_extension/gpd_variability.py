"""Run-to-run variability of GPD's scores on identical inputs (GPD's shadow computation is random by design).

    python tools/research/phase7_extension/gpd_variability.py --first gpd_dev/ --second gpd_dev_timing/ --catalogues cat80/ \
        --out research/phase7_extension/results/gpd_variability.json

GPD's HandSet::shadowVoxelsToPoints draws its jitter from std::mt19937 seeded by std::random_device
(src/gpd/candidate/hand_set.cpp at commit 6327f20, lines 191-192), and HandSet::fastrand uses a static seed that
advances between calls. Scores therefore differ between runs even with the same point clouds and candidates.
This script compares two complete runs on the same development parts (the second run also records stage timings).
The library is used unmodified.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import gpd_evaluate as G  # noqa: E402
from research_grasping.phase5 import catalogue  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('first', 'second', 'catalogues', 'out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    rows = {}
    for f in sorted(Path(a.second).glob('*.json')):
        n = f.stem
        s2 = json.loads(f.read_text()); s1 = json.loads((Path(a.first)/f.name).read_text())
        x, y = np.asarray(s1['scores']), np.asarray(s2['scores'])
        cat = catalogue.load(Path(a.catalogues)/f'{n}.npz')
        o1, _ = G.gpd_orders(cat, x); o2, _ = G.gpd_orders(cat, y)
        f1, f2 = C.first(cat, o1), C.first(cat, o2)
        rows[n] = dict(score_correlation=float(np.corrcoef(x, y)[0, 1]), max_abs_difference=float(np.abs(x - y).max()),
                       nf=[f1[0], f2[0]], same_first_grasp=f1 == f2, stage_seconds=s2.get('gpd_stage_seconds'),
                       render_seconds=s2['render_seconds'], gpd_seconds=s2['gpd_seconds'])
    nf1 = [r['nf'][0] for r in rows.values()]; nf2 = [r['nf'][1] for r in rows.values()]
    st = {k: float(np.median([r['stage_seconds'][k] for r in rows.values()])) for k in ('cloud_preprocess', 'images', 'classify')}
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), objects=len(rows),
               score_correlation=dict(min=min(r['score_correlation'] for r in rows.values()),
                                      median=float(np.median([r['score_correlation'] for r in rows.values()]))),
               same_first_grasp=int(sum(r['same_first_grasp'] for r in rows.values())),
               N_F_gmean_run1=C.gmean(nf1), N_F_gmean_run2=C.gmean(nf2),
               stage_seconds_median=st, render_seconds_median=float(np.median([r['render_seconds'] for r in rows.values()])),
               per_object=rows)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('per_object', 'note')}, indent=1))


if __name__ == '__main__':
    main()
