"""FORTIS-R Joint under the v3 criterion and the torsional-friction grid (development).

    python tools/research/phase7/joint_v3.py --dev research/phase6/dev_parts.json --catalogues cat80/ \
        --collision COLLISION_DIR --projects POOL --store store.json --priors priors.npz \
        --sweep research/phase7/results/sweep_v2_extended.json.gz --physics research/phase7/results/physics_dev.json.gz \
        --beta-from research/phase7/results/robustness_benchmark.json --out joint_v3.json.gz --jobs 3

Joint = p * h * Q_cen^beta with h = P(H|F) (gradient boosting, Phase VI outer folds, out of fold) and
beta chosen per outer fold by the protocol criterion (taken from robustness_benchmark.json). Its first
viable grasps that physics_dev.json.gz has not simulated are run with v3 in the nine conditions and the
four torsion values; the result adds a 'Joint' entry to the core methods.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402
from research_grasping.phase6 import fusion  # noqa: E402

sys.path.insert(0, str(ROOT/'tools/research/phase7'))
import physics_dev as pdev  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('dev', 'catalogues', 'collision', 'projects', 'store', 'priors', 'sweep', 'physics', 'beta-from', 'out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--jobs', type=int, default=3)
    a = p.parse_args()
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    store = json.loads(Path(a.store).read_text())
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in dev}
    pz = np.load(a.priors)
    pri = {n: pz[f'{n}|learned'].astype(float)/pz[f'{n}|learned'].max() for n in dev}
    sw = json.load(gzip.open(a.sweep, 'rt'))
    std = sw['standard']
    ph = json.load(gzip.open(a.physics, 'rt'))
    betas = json.loads(Path(a.beta_from).read_text())['joint_beta_chosen']['Joint[hgb]']
    order = np.random.default_rng(20261130).permutation(len(dev))
    folds = [order[f::5] for f in range(5)]
    X = {n: bl.features(cats[n], store[n]['low'], store[n]['high']) for n in dev}
    joint = {}
    for k, test in enumerate(folds):
        train = [dev[i] for i in np.setdiff1d(np.arange(len(dev)), test)]
        Xt, yt = [], []
        for n in train:
            idx = np.flatnonzero(cats[n]['feasible'])
            Xt.append(X[n][idx])
            yt.append([all(c in sw['parts'][n][str(cats[n]['label'][i])] and sw['parts'][n][str(cats[n]['label'][i])][c][0]
                           for c in std) for i in idx])
        model = bl.fit(np.vstack(Xt), np.concatenate(yt).astype(int), 0)
        for i in test:
            n = dev[i]
            h = model.predict_proba(X[n])[:, 1]
            s = pri[n]*h*np.maximum(fusion.quality(cats[n], 'cen'), 1e-3)**betas[k]
            o = fusion.order(cats[n], s)
            joint[n] = str(cats[n]['label'][o[np.flatnonzero(cats[n]['feasible'][o])[0]]])
    tasks = []
    for n in dev:
        have = ph['trials'][n].get(joint[n], {})
        need = [t for t in pdev.TORSIONS if str(t) not in have]
        if need:
            tasks.append((n, [(joint[n], need)]))
    args = argparse.Namespace(store=a.store, projects=a.projects, collision=a.collision, catalogues=a.catalogues)
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=pdev.init, initargs=(args,)) as pool:
        for part, out, s in pool.map(pdev.job, tasks):
            for lab, tr in out.items():
                ph['trials'][part].setdefault(lab, {}).update(tr)
            print(part, round(s), flush=True)
    for n in dev:
        ph['core'][n]['Joint'] = joint[n]
    ph['joint_betas_by_fold'] = betas
    with gzip.open(a.out, 'wt') as f:
        json.dump(ph, f, separators=(',', ':'))
    print(len(tasks), 'parts simulated')


if __name__ == '__main__':
    main()
