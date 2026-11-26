"""Phase VII mandatory representation experiment: train every variant out of fold.

    python tools/research/phase7/mlp_benchmark.py --catalogues cat80/ --store store.json \
        --library research/phase5/library_parts.txt --test research/phase6/dev_parts.json \
        --openclip a.npz --openclip b.npz --out-dir mlp/ --jobs 2 [--only L2 L8 ...]

Variants, blocks, architecture, folds and seeds as fixed in PHASE7_PROTOCOL.md (I.3). For every
(variant, seed) the out-of-fold feasibility probabilities of all 316 development parts are written
to <out-dir>/<variant>-s<seed>.npz with training time, parameters, epochs and inference time.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402
from research_grasping.phase7 import representation as rep  # noqa: E402

V = dict(geometry=False, visual=None, linear=False)
VARIANTS = {
    'L0': dict(kind='hgb'), 'L1': dict(kind='logreg'),
    'L2': dict(kind='mlp', geometry=True), 'L2v': dict(kind='mlp'),
    'L3': dict(kind='mlp', visual='dinov2', linear=True), 'L4': dict(kind='mlp', visual='dinov3', linear=True),
    'L5': dict(kind='mlp', visual='dinov2'), 'L6': dict(kind='mlp', visual='dinov3'),
    'L7': dict(kind='mlp', geometry=True, visual='dinov2'), 'L8': dict(kind='mlp', geometry=True, visual='dinov3'),
    'L9': dict(kind='mlp', geometry=True, visual='vit-random-s0'),
    'L10': dict(kind='mlp', geometry=True, visual='dinov3', linear=True),
    'L11': dict(kind='mlp', geometry=True, visual='openclip'),
    'L8p': dict(kind='mlp', geometry=True, visual='dinov3-permuted'),
    # pre-declared sensitivity and ranking-loss variants
    'L2-k8': dict(kind='mlp', geometry=True, latent=8), 'L2-k32': dict(kind='mlp', geometry=True, latent=32),
    'L6-k8': dict(kind='mlp', visual='dinov3', latent=8), 'L6-k32': dict(kind='mlp', visual='dinov3', latent=32),
    'L8-k8': dict(kind='mlp', geometry=True, visual='dinov3', latent=8),
    'L8-k32': dict(kind='mlp', geometry=True, visual='dinov3', latent=32),
    'L2-rank': dict(kind='mlp', geometry=True, loss='listwise'),
    'L8-rank': dict(kind='mlp', geometry=True, visual='dinov3', loss='listwise'),
}
SEEDS = (0, 1, 2)
STA = {}


def load_data(a):
    store = json.loads(Path(a.store).read_text())
    library = [s.strip() for s in Path(a.library).read_text().splitlines() if s.strip()]
    tests = json.loads(Path(a.test).read_text())
    test = sorted(set(tests['parts']) | set(tests['nonviable']))
    names = sorted(library + test)
    oc = {}
    for path in a.openclip:
        z = np.load(path, allow_pickle=True)
        oc.update({str(n): v for n, v in zip(z['names'], z['vectors'])})
    perm = np.random.default_rng(7).permutation(len(names))
    data = {}
    for n in names:
        cat = catalogue.load(Path(a.catalogues)/f'{n}.npz')
        F = bl.features(cat, store[n]['low'], store[n]['high'])
        desc = store[n]['descriptors']
        data[n] = dict(v=F[:, rep.V_COLUMNS].astype(np.float32), y=cat['feasible'].astype(np.float32),
                       x19=F.astype(np.float32),
                       u=np.concatenate([F[0, rep.EXTENT_COLUMNS], np.asarray(desc['d2'], float)]).astype(np.float32),
                       dv={k: np.asarray(desc[k], np.float32) for k in ('dinov2', 'dinov3', 'vit-random-s0')})
        data[n]['dv']['openclip'] = np.asarray(oc[n], np.float32)
    for i, n in enumerate(names):
        data[n]['dv']['dinov3-permuted'] = data[names[perm[i]]]['dv']['dinov3']
    return names, data, library


def init(a):
    names, data, library = load_data(a)
    groups = rep.name_groups(names)
    STA.update(a=a, names=names, data=data, groups=groups, folds=rep.grouped_folds(names, groups))


def job(task):
    variant, seed = task
    s = STA
    a, names, data, groups, folds = s['a'], s['names'], s['data'], s['groups'], s['folds']
    cfg = VARIANTS[variant]
    out, info = {}, []
    for k, test in enumerate(folds):
        train = [n for n in names if n not in set(test)]
        tg = sorted({groups[n] for n in train})
        rng = np.random.default_rng(1000 + k)
        val_g = set(rng.choice(tg, max(1, int(round(.15*len(tg)))), replace=False))
        inner_train = [n for n in train if groups[n] not in val_g]
        inner_val = [n for n in train if groups[n] in val_g]
        start = time.perf_counter()
        if cfg['kind'] in ('hgb', 'logreg'):
            X = np.vstack([data[n]['x19'] for n in train]); y = np.concatenate([data[n]['y'] for n in train])
            if cfg['kind'] == 'hgb':
                model = bl.fit(X, y.astype(int), seed)
                params = None
            else:
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000)).fit(X, y.astype(int))
                params = int(X.shape[1] + 1)
            train_s = time.perf_counter() - start
            t0 = time.perf_counter()
            for n in test:
                out[n] = model.predict_proba(data[n]['x19'])[:, 1].astype(np.float32)
            pred_s = time.perf_counter() - t0
            info.append(dict(fold=k, train_s=round(train_s, 2), predict_s=round(pred_s, 4), params=params))
        else:
            vis = cfg.get('visual')
            sub = {n: dict(v=data[n]['v'], y=data[n]['y'], u=data[n]['u'],
                           d=data[n]['dv'][vis] if vis else None) for n in names}
            model = rep.MLPModel(geometry=cfg.get('geometry', False), visual=vis is not None,
                                 linear_visual=cfg.get('linear', False), loss=cfg.get('loss', 'bce'),
                                 latent=cfg.get('latent', rep.LATENT), seed=seed)
            model.fit(sub, inner_train, inner_val)
            pred = model.predict(sub, test)
            out.update({n: p.astype(np.float32) for n, p in pred.items()})
            info.append(dict(fold=k, train_s=round(model.train_seconds, 2), predict_s=round(model.predict_seconds, 4),
                             params=model.params, epochs=model.epochs))
    path = Path(a.out_dir)/f'{variant}-s{seed}.npz'
    np.savez_compressed(path, **out)
    path.with_suffix('.json').write_text(json.dumps(dict(variant=variant, seed=seed, config=cfg, folds=info), indent=1))
    return variant, seed, sum(i['train_s'] for i in info)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('catalogues', 'store', 'library', 'test', 'out-dir'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--openclip', action='append', required=True)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--only', nargs='*')
    a = p.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    tasks = []
    for v, cfg in VARIANTS.items():
        if a.only and v not in a.only:
            continue
        for s in (SEEDS if cfg['kind'] == 'mlp' else (0,)):
            if not (Path(a.out_dir)/f'{v}-s{s}.npz').exists():
                tasks.append((v, s))
    names, data, _ = load_data(a)
    groups = rep.name_groups(names)
    folds = rep.grouped_folds(names, groups)
    (Path(a.out_dir)/'folds.json').write_text(json.dumps(dict(seed=20261201, folds=folds, groups=groups), indent=1))
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=init, initargs=(a,)) as pool:
        for v, s, t in pool.map(job, tasks):
            print(v, s, round(t), flush=True)


if __name__ == '__main__':
    main()
