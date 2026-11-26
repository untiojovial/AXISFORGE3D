"""Shared helpers of the Phase VII extension scripts (development orders, outcomes, statistics)."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402
from research_grasping.phase6 import fusion  # noqa: E402

STANDARD = ('nominal', 'mass x2', 'mass x0.5', 'friction 0.3', 'pose noise 1', 'pose noise 2')
LOADS = ('load 0.5 kg', 'load 1 kg', 'load 2 kg')
KS = (1, 5, 10, 20, 50)


def dev_oof_orders(dev, cats, store, pri, sw, betas):
    """Out-of-fold development orders of M0, M1, M4, M9 and Joint, exactly as
    tools/research/phase7/negative_transfer.py builds them (same folds, models and betas)."""
    std = sw['standard']
    perm = np.random.default_rng(20261130).permutation(len(dev))
    folds = [perm[f::5] for f in range(5)]
    X = {n: bl.features(cats[n], store[n]['low'], store[n]['high']) for n in dev}
    orders, h_oof = {n: {} for n in dev}, {}
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
            n = dev[i]; cat = cats[n]
            h = model.predict_proba(X[n])[:, 1]; h_oof[n] = h
            orders[n]['Joint'] = fusion.order(cat, pri[n]*h*np.maximum(fusion.quality(cat, 'cen'), 1e-3)**betas[k])
            orders[n]['M9'] = fusion.order(cat, pri[n]*h)
    for n in dev:
        cat = cats[n]
        orders[n]['M0'] = catalogue.order(cat)
        orders[n]['M1'] = catalogue.order(cat, pri[n], 2.)
        orders[n]['M4'] = fusion.order(cat, fusion.config_score('M4', dict(alpha=.8, q='cen'), cat, pri[n]))
    return orders, h_oof


def load_dev(dev_json, catalogues, store_json, priors_npz, sweep_gz, physics_gz):
    dev = sorted(json.loads(Path(dev_json).read_text())['parts'])
    store = json.loads(Path(store_json).read_text())
    cats = {n: catalogue.load(Path(catalogues)/f'{n}.npz') for n in dev}
    pz = np.load(priors_npz)
    pri = {n: pz[f'{n}|learned'].astype(float)/pz[f'{n}|learned'].max() for n in dev}
    sw = json.load(gzip.open(sweep_gz, 'rt'))
    ph = json.load(gzip.open(physics_gz, 'rt'))
    orders, h = dev_oof_orders(dev, cats, store, pri, sw, ph['joint_betas_by_fold'])
    mism = [(n, m) for n in dev for m in ('M0', 'M1', 'M4', 'M9', 'Joint') if first(cats[n], orders[n][m])[1] != ph['core'][n][m]]
    if mism:
        raise SystemExit(f'development orders do not reproduce the Phase VII first grasps: {mism[:5]}')
    return dict(dev=dev, store=store, cats=cats, pri=pri, sw=sw, ph=ph, orders=orders, h=h)


def first(cat, order):
    """(N_F, label of the first viable candidate) of an order, or (None, None)."""
    pos = np.flatnonzero(cat['feasible'][order])
    return (int(pos[0]) + 1, str(cat['label'][order[pos[0]]])) if len(pos) else (None, None)


def interleave(cat, a, b):
    seen, out = set(), []
    for k in range(len(a)):
        for x in (int(a[k]), int(b[k])):
            if x not in seen:
                seen.add(x); out.append(x)
    return np.array(out)


def held(trials, which, torsion='0.005', conds=None):
    """1 if every condition of the group held under v3, 0 otherwise; None if a trial is missing."""
    t = trials.get(torsion) if trials else None
    conds = conds or (STANDARD if which == 'robust' else LOADS)
    if not t or any(c not in t for c in conds):
        return None
    return int(all(t[c]['held_v3'] for c in conds))


def gmean(x):
    return float(np.exp(np.mean(np.log(np.asarray(x, float)))))


def boot_ratio(a, b, seed=20261211, n=10000):
    d = np.log(np.asarray(a, float)/np.asarray(b, float))
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return dict(ratio=float(np.exp(d.mean())), ci95=[float(np.exp(np.percentile(m, 2.5))), float(np.exp(np.percentile(m, 97.5)))],
                fewer=int((d < 0).sum()), more=int((d > 0).sum()), ties=int((d == 0).sum()), objects=int(len(d)))


def boot_diff(a, b, seed=20261211, n=10000):
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return dict(difference=float(d.mean()), ci95=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                favourable=int((d > 0).sum()), unfavourable=int((d < 0).sum()), objects=int(len(d)))
