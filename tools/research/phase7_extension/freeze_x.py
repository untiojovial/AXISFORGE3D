"""Fit and export the selected method X = Joint-multi (DEVELOPMENT_SELECTION_RULE.md, results/v3_models.json).

    python tools/research/phase7_extension/freeze_x.py --v3sweep v3sweep/ --dev research/phase6/dev_parts.json \
        --catalogues cat80/ --store store.json --out-dir research/phase7_extension/frozen

X orders candidates by S = p~ * P(robust at all three torsions | feasible, g) * max(Q_cen, 0.001)^0.3:

- P(F): the frozen Phase VII model (research/phase7/frozen/p_feasible.pkl, hash-checked), unchanged;
- P(robust all3 | F): the rule's gradient boosting (research_grasping/phase5/baselines.fit, seed 0, Phase V
  hyperparameters) trained on y_all3 of every v3-sweep simulation of the 151 development parts. The training set
  is the sweep's seeded sample plus the first grasps of the five reference orders; the extra simulations of the
  variants' first grasps are not used, as in the selection;
- beta = 0.3 and the 0.001 floor (the historical form).

Outputs: p_robust_all3.pkl, learned_order_x.json (exact-bit trees for the native planner) and freeze_x_fit.json
(training size, label rate, hashes).
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402
import v3_models as vm  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('v3sweep', 'dev', 'catalogues', 'store', 'out-dir'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    out = Path(a.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    store = json.loads(Path(a.store).read_text())
    X, y, parts = [], [], 0
    for n in dev:
        f = Path(a.v3sweep)/f'{n}.json'
        trials = json.loads(f.read_text())['trials']
        cat = catalogue.load(Path(a.catalogues)/f'{n}.npz')
        F = bl.features(cat, store[n]['low'], store[n]['high'])
        idx = {str(l): i for i, l in enumerate(cat['label'])}
        got = 0
        for label, t in trials.items():
            lab = vm.labels_of(t)
            if lab:
                X.append(F[idx[label]]); y.append(lab['y_all3']); got += 1
        parts += got > 0
    X, y = np.asarray(X), np.asarray(y, int)
    model = bl.fit(X, y, 0)
    pkl = out/'p_robust_all3.pkl'
    pkl.write_bytes(pickle.dumps(model))
    h = hashlib.sha256(pkl.read_bytes()).hexdigest()
    js = out/'learned_order_x.json'
    subprocess.run([sys.executable, str(ROOT/'tools/research/phase7_extension/export_models.py'),
                    '--feasibility', str(ROOT/'research/phase7/frozen/p_feasible.pkl'), '--robustness', str(pkl),
                    '--alpha', '0.8', '--beta', '0.3', '--out', str(js)], check=True)
    info = dict(method='X = Joint-multi', training_parts=parts, training_candidates=int(len(y)), positive_rate=float(y.mean()),
                model=dict(file=str(pkl.relative_to(ROOT)), sha256=h, kind='HistGradientBoostingClassifier (baselines.classifier, seed 0)',
                           n_iter=int(model.n_iter_)),
                exported=dict(file=str(js.relative_to(ROOT)), sha256=hashlib.sha256(js.read_bytes()).hexdigest()),
                feasibility=dict(file='research/phase7/frozen/p_feasible.pkl',
                                 sha256=hashlib.sha256((ROOT/'research/phase7/frozen/p_feasible.pkl').read_bytes()).hexdigest()),
                beta=0.3, q_floor=1e-3)
    (out/'freeze_x_fit.json').write_text(json.dumps(info, indent=1) + '\n')
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
