"""The first viable grasp of every ordering the Phase VI analyses look at.

    python tools/research/phase6/first_grasps.py --catalogues cat80/ --dev research/phase6/dev_parts.json \
        --priors priors.npz --phase5-dynamics research/phase5/results/dynamics.json.gz --out first_grasps.json

Orderings: default, learned and AFAT (Phase V), every configuration of the
fusion grid (M3-M8), and the first grasps the Phase V dynamic test used
(default, learned, afat, box, DINOv3). Which grasp comes first depends only on
the ordering and the planner verdicts, not on any dynamic outcome, so this
list can be fixed before the second sweep pass.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import catalogue  # noqa: E402
from research_grasping.phase6 import fusion  # noqa: E402


def first(cat, order):
    hits = np.flatnonzero(cat['feasible'][order])
    return str(cat['label'][order[hits[0]]])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('catalogues', 'dev', 'priors', 'phase5-dynamics', 'out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    pri = np.load(a.priors)
    d5 = json.load(gzip.open(a.phase5_dynamics))['parts']
    out = {}
    for n in dev:
        cat = catalogue.load(Path(a.catalogues)/f'{n}.npz')
        pl = pri[f'{n}|learned'].astype(float); pt = pl/pl.max()
        pa = pri[f'{n}|afat'].astype(float)
        labels = {first(cat, catalogue.order(cat)), first(cat, catalogue.order(cat, pt, 2.)),
                  first(cat, catalogue.order(cat, pa/pa.max(), 2.))}
        for f, grid in fusion.GRID.items():
            for cfg in grid:
                labels.add(first(cat, fusion.order(cat, fusion.config_score(f, cfg, cat, pt))))
        labels |= {d5[n]['methods'][m]['first'] for m in d5[n]['methods'] if d5[n]['methods'][m].get('first')}
        out[n] = sorted(labels)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(sum(len(v) for v in out.values()), 'grasps in', len(out), 'parts')


if __name__ == '__main__':
    main()
