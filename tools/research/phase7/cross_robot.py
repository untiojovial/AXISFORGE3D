"""Cross-gripper generalisation of the feasibility classifier (PHASE7_PROTOCOL.md, I.7).

    python tools/research/phase7/cross_robot.py --store store.json --cat80 cat80/ --cat40 cat40/ --cat120 cat120/ \
        --folds mlp/folds.json --out cross_robot_results.json

Same grouped folds as the representation experiment (316 parts). For each target gripper (40, 120 mm):
zero-shot = gradient boosting (Phase V settings) trained on the 80 mm catalogues of the training parts;
specific = the same model trained on the target gripper's catalogues of the training parts. Both order
the held-out parts' target-gripper catalogues as M1 does. A separate model per gripper is not
transfer; only the zero-shot column is.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402
from research_grasping.phase5.stats import _boot  # noqa: E402


def nf(cat, p):
    o = catalogue.order(cat, p/max(float(p.max()), 1e-12), 2.)
    h = np.flatnonzero(cat['feasible'][o])
    return int(h[0])+1 if len(h) else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('store', 'cat80', 'cat40', 'cat120', 'folds', 'out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    store = json.loads(Path(a.store).read_text())
    folds = json.loads(Path(a.folds).read_text())['folds']
    names = sorted(n for f in folds for n in f)
    dirs = {80: a.cat80, 40: a.cat40, 120: a.cat120}
    cats = {g: {n: catalogue.load(Path(d)/f'{n}.npz') for n in names} for g, d in dirs.items()}
    X = {g: {n: bl.features(cats[g][n], store[n]['low'], store[n]['high']) for n in names} for g in dirs}
    out = dict(note='Development parts (316), grouped folds of the representation experiment; HistGradientBoosting seed 0.',
               grippers={})
    preds = {g: {'zero-shot from 80 mm': {}, 'trained for this gripper': {}, 'default': {}} for g in (40, 120)}
    for test in folds:
        train = [n for n in names if n not in set(test)]
        m80 = bl.fit(np.vstack([X[80][n] for n in train]), np.concatenate([cats[80][n]['feasible'] for n in train]).astype(int), 0)
        for g in (40, 120):
            mg = bl.fit(np.vstack([X[g][n] for n in train]), np.concatenate([cats[g][n]['feasible'] for n in train]).astype(int), 0)
            for n in test:
                preds[g]['zero-shot from 80 mm'][n] = nf(cats[g][n], m80.predict_proba(X[g][n])[:, 1])
                preds[g]['trained for this gripper'][n] = nf(cats[g][n], mg.predict_proba(X[g][n])[:, 1])
                preds[g]['default'][n] = nf(cats[g][n], np.zeros(len(X[g][n])))
    for g in (40, 120):
        viable = [n for n in names if preds[g]['default'][n] is not None]
        row = dict(parts=len(names), viable_parts=len(viable), methods={}, contrasts={})
        for m, d in preds[g].items():
            v = np.array([d[n] for n in viable], float)
            row['methods'][m] = dict(N_F_gmean=round(float(np.exp(np.log(v).mean())), 3),
                                     success_at_1=round(float(np.mean([d[n] == 1 for n in viable])), 4) if viable else None)
        for x, y in (('zero-shot from 80 mm', 'trained for this gripper'), ('zero-shot from 80 mm', 'default'),
                     ('trained for this gripper', 'default')):
            d = np.log([preds[g][x][n] for n in viable]) - np.log([preds[g][y][n] for n in viable])
            row['contrasts'][f'{x} / {y}'] = dict(ratio=round(float(np.exp(d.mean())), 4), ci95=[round(z, 4) for z in _boot(d, 95)])
        out['grippers'][f'{g} mm'] = row
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['grippers'], indent=1)[:2500])


if __name__ == '__main__':
    main()
