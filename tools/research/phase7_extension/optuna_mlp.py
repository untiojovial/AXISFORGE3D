"""Exploratory hyperparameter search for the feasibility MLPs (development only, nested grouped CV).

    python tools/research/phase7_extension/optuna_mlp.py --catalogues cat80/ --store store.json \
        --library research/phase5/library_parts.txt --test research/phase6/dev_parts.json \
        --openclip a.npz --openclip b.npz --out-dir optuna/ --trials 30 --families geometry dinov3 ... --jobs 3
    python tools/research/phase7_extension/optuna_mlp.py ... --summarise --mlp-dir MLP_DIR \
        --out research/phase7_extension/results/optuna_mlp.json

Design (fixed before running; budgets reduced from the request for compute and stated here):

- data: the 316 development parts and features of tools/research/phase7/mlp_benchmark.py (blocks v, u, d);
- outer CV: the same 5 folds grouped by product name as the Phase VII MLP benchmark;
- inner: within each outer training set, one grouped hold-out (15 % of the name groups, seed 2000+k)
  for early stopping and the objective; nothing of the outer test fold is seen by the search;
- search (Optuna 4.1 TPE, seed 20261212, MedianPruner after 5 trials and 3 epochs), per family and
  outer fold, --trials trials over: latent 8-256 (log), width 16-512 (log), hidden layers 1-3,
  activation ReLU/GELU/SiLU, dropout 0-0.3, learning rate 1e-5-3e-3 (log), weight decay 1e-6-1e-2 (log),
  batch 1024/2048/4096/8192, Adam/AdamW, residual connections yes/no;
- training rows during the search: at most 256 candidates per part (seeded), at most 30 epochs,
  patience 4; the objective is the geometric-mean N_F of the M1-style order (preliminary + 2 p~,
  research_grasping/phase5/catalogue.order) on the inner hold-out's viable parts;
- confirmation: the best configuration of each outer fold refitted on all rows of the inner training
  parts (the inner hold-out again for early stopping, at most 60 epochs) with 5 seeds; out-of-fold predictions of
  all 316 parts are scored with the Phase VII metrics (AUC, log-loss, ECE, N_F, Success@k);
- families: candidate-only (v), geometry (v+u), dinov2 / dinov3 / openclip (v+d), combined
  (v+u+dinov3), linear (v+u+linear dinov3 branch), random-vit (v+u+untrained ViT), permuted
  (v+u+dinov3 of another part). Gradient boosting (L0) and the fixed-architecture MLPs come from
  research/phase7/results/mlp_benchmark.json (same folds), as references.

Every trial (parameters, value, state, seconds, epochs), each fold's selected configuration, the
checkpoints (torch state dicts, hashed) and the out-of-fold predictions are written to --out-dir.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools/research/phase7'))
from research_grasping.phase5 import catalogue  # noqa: E402
from research_grasping.phase7 import representation as rep  # noqa: E402
import mlp_benchmark as mb  # noqa: E402

FAMILIES = {'candidate-only': dict(u=False, d=None), 'geometry': dict(u=True, d=None),
            'dinov2': dict(u=False, d='dinov2'), 'dinov3': dict(u=False, d='dinov3'), 'openclip': dict(u=False, d='openclip'),
            'combined': dict(u=True, d='dinov3'), 'linear': dict(u=True, d='dinov3', linear=True),
            'random-vit': dict(u=True, d='vit-random-s0'), 'permuted': dict(u=True, d='dinov3-permuted')}
SEARCH_ROWS, SEARCH_EPOCHS, FINAL_EPOCHS, PATIENCE, SEEDS = 256, 30, 60, 4, (0, 1, 2, 3, 4)
STA = {}


def build(dims, fam, hp):
    import torch
    from torch import nn
    act = dict(relu=nn.ReLU, gelu=nn.GELU, silu=nn.SiLU)[hp['activation']]

    class Block(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.lin, self.act, self.drop = nn.Linear(i, o), act(), nn.Dropout(hp['dropout'])
            self.res = hp['residual'] and i == o

        def forward(self, x):
            y = self.drop(self.act(self.lin(x)))
            return x + y if self.res else y

    def tower(i, o):
        layers, w = [], i
        for _ in range(hp['layers']):
            layers.append(Block(w, hp['width'])); w = hp['width']
        layers.append(nn.Linear(w, o))
        return nn.Sequential(*layers)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.grasp = tower(dims['v'], hp['latent'])
            width = hp['latent']
            self.geom = tower(dims['u'], hp['latent']) if fam['u'] else None
            width += hp['latent'] if fam['u'] else 0
            if fam['d']:
                self.vis = nn.Linear(dims['d'], hp['latent']) if fam.get('linear') else tower(dims['d'], hp['latent'])
                width += hp['latent']
            else:
                self.vis = None
            self.head = nn.Sequential(act(), tower(width, 1))

        def forward(self, v, u, d):
            z = [self.grasp(v)]
            if self.geom is not None:
                z.append(self.geom(u))
            if self.vis is not None:
                z.append(self.vis(d))
            return self.head(torch.cat(z, 1)).squeeze(1)
    return Net()


def tensors(data, parts, fam, stats, rows=None, seed=0):
    import torch
    rng = np.random.default_rng(seed)
    V, Y, I = [], [], []
    for i, p in enumerate(parts):
        n = len(data[p]['y'])
        sel = np.arange(n) if rows is None or n <= rows else np.sort(rng.choice(n, rows, replace=False))
        V.append(data[p]['v'][sel]); Y.append(data[p]['y'][sel]); I.append(np.full(len(sel), i))
    out = dict(v=torch.tensor(((np.vstack(V) - stats['v'][0])/stats['v'][1]).astype(np.float32)), y=torch.tensor(np.concatenate(Y)),
               idx=torch.tensor(np.concatenate(I)), counts=[len(x) for x in Y])
    if fam['u']:
        out['u'] = torch.tensor(((np.vstack([data[p]['u'] for p in parts]) - stats['u'][0])/stats['u'][1]).astype(np.float32))
    if fam['d']:
        out['d'] = torch.tensor(((np.vstack([data[p]['dv'][fam['d']] for p in parts]) - stats['d'][0])/stats['d'][1]).astype(np.float32))
    return out


def fit_stats(data, parts, fam):
    st = dict(v=(np.vstack([data[p]['v'] for p in parts]).mean(0), np.vstack([data[p]['v'] for p in parts]).std(0) + 1e-8))
    if fam['u']:
        u = np.vstack([data[p]['u'] for p in parts]); st['u'] = (u.mean(0), u.std(0) + 1e-8)
    if fam['d']:
        d = np.vstack([data[p]['dv'][fam['d']] for p in parts]); st['d'] = (d.mean(0), d.std(0) + 1e-8)
    return st


def forward(net, T, rows):
    u = T['u'][T['idx'][rows]] if 'u' in T else None
    d = T['d'][T['idx'][rows]] if 'd' in T else None
    return net(T['v'][rows], u, d)


def predict(net, T):
    import torch
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(forward(net, T, torch.arange(len(T['y'])))).numpy()
    return np.split(p, np.cumsum(T['counts'])[:-1])


def nf_gmean(cats, parts, probs):
    nf = []
    for p, pr in zip(parts, probs):
        cat = cats[p]
        if not cat['feasible'].any():
            continue
        o = catalogue.order(cat, pr/max(pr.max(), 1e-12), 2.)
        nf.append(int(np.flatnonzero(cat['feasible'][o])[0]) + 1)
    return float(np.exp(np.mean(np.log(nf)))) if nf else np.inf


def train(data, cats, fam, hp, train_parts, val_parts, rows, epochs, seed, trial=None):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed); torch.set_num_threads(1)
    st = fit_stats(data, train_parts, fam)
    T = tensors(data, train_parts, fam, st, rows, seed)
    Vt = tensors(data, val_parts, fam, st)
    dims = dict(v=T['v'].shape[1], u=T['u'].shape[1] if 'u' in T else 0, d=T['d'].shape[1] if 'd' in T else 0)
    net = build(dims, fam, hp)
    Opt = torch.optim.AdamW if hp['optimizer'] == 'adamw' else torch.optim.Adam
    opt = Opt(net.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    rng = np.random.default_rng(seed)
    best, best_state, bad, ep = np.inf, None, 0, 0
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(T['y']))
        for s in range(0, len(order), hp['batch']):
            b = torch.tensor(order[s:s+hp['batch']])
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(forward(net, T, b), T['y'][b]).backward()
            opt.step()
        val = nf_gmean(cats, val_parts, predict(net, Vt))
        if trial is not None:
            import optuna
            trial.report(val, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if val < best - 1e-9:
            best, bad, best_state = val, 0, {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return net, st, best, ep + 1


def sample(trial):
    return dict(latent=trial.suggest_int('latent', 8, 256, log=True), width=trial.suggest_int('width', 16, 512, log=True),
                layers=trial.suggest_int('layers', 1, 3), activation=trial.suggest_categorical('activation', ['relu', 'gelu', 'silu']),
                dropout=trial.suggest_float('dropout', 0., .3), lr=trial.suggest_float('lr', 1e-5, 3e-3, log=True),
                weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
                batch=trial.suggest_categorical('batch', [1024, 2048, 4096, 8192]),
                optimizer=trial.suggest_categorical('optimizer', ['adam', 'adamw']), residual=trial.suggest_categorical('residual', [False, True]))


def init(a):
    names, data, library = mb.load_data(a)
    groups = rep.name_groups(names)
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in names}
    STA.update(a=a, names=names, data=data, groups=groups, folds=rep.grouped_folds(names, groups), cats=cats)


def job(task):
    import optuna
    import torch
    fam_name, k = task
    s = STA
    a, names, data, groups, folds, cats = s['a'], s['names'], s['data'], s['groups'], s['folds'], s['cats']
    out = Path(a.out_dir)/f'{fam_name}-fold{k}.json'
    if out.exists():
        return fam_name, k, 'cached'
    fam = FAMILIES[fam_name]
    test = set(folds[k])
    train_all = [n for n in names if n not in test]
    tg = sorted({groups[n] for n in train_all})
    rng = np.random.default_rng(2000 + k)
    val_g = set(rng.choice(tg, max(1, int(round(.15*len(tg)))), replace=False))
    inner_train = [n for n in train_all if groups[n] not in val_g]
    inner_val = [n for n in train_all if groups[n] in val_g]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=20261212 + k),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    log = []

    def objective(trial):
        hp = sample(trial)
        t0 = time.perf_counter()
        try:
            _, _, val, ep = train(data, cats, fam, hp, inner_train, inner_val, SEARCH_ROWS, SEARCH_EPOCHS, 0, trial)
        finally:
            trial.set_user_attr('seconds', time.perf_counter() - t0)
        trial.set_user_attr('epochs', ep)
        return val
    t_search = time.perf_counter()
    study.optimize(objective, n_trials=a.trials, catch=(RuntimeError, ValueError))
    t_search = time.perf_counter() - t_search
    for t in study.trials:
        log.append(dict(number=t.number, state=t.state.name, value=t.value, params=t.params, seconds=t.user_attrs.get('seconds'),
                        epochs=t.user_attrs.get('epochs'), last_step=max(t.intermediate_values) if t.intermediate_values else None))
    hp = dict(study.best_params)
    test_parts = sorted(test)
    preds, ckpts, fits = {}, [], []
    for seed in SEEDS:
        t0 = time.perf_counter()
        net, st, val, ep = train(data, cats, fam, hp, inner_train, inner_val, None, FINAL_EPOCHS, seed)
        fit_s = time.perf_counter() - t0
        Tt = tensors(data, test_parts, fam, st)
        t1 = time.perf_counter(); pr = predict(net, Tt); pred_s = time.perf_counter() - t1
        preds[seed] = {p: x for p, x in zip(test_parts, pr)}
        ck = Path(a.out_dir)/'checkpoints'/f'{fam_name}-fold{k}-s{seed}.pt'
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(hp=hp, family=fam, state=net.state_dict(), stats={b: [x.tolist() for x in v] for b, v in st.items()}), ck)
        ckpts.append(dict(file=ck.name, sha256=hashlib.sha256(ck.read_bytes()).hexdigest()))
        fits.append(dict(seed=seed, inner_val_nf=val, epochs=ep, fit_seconds=fit_s, predict_ms_per_part=1e3*pred_s/len(test_parts),
                         params=int(sum(x.numel() for x in net.parameters()))))
    np.savez_compressed(Path(a.out_dir)/f'{fam_name}-fold{k}-oof.npz', **{f'{p}|s{seed}': preds[seed][p] for seed in SEEDS for p in test_parts})
    out.write_text(json.dumps(dict(family=fam_name, fold=k, inner_train=len(inner_train), inner_val=len(inner_val), test=len(test_parts),
                                   search_seconds=t_search, best_params=hp, best_value=study.best_value, trials=log,
                                   confirmation=fits, checkpoints=ckpts), indent=1))
    return fam_name, k, round(t_search)


def summarise(a):
    from sklearn.metrics import log_loss, roc_auc_score
    names, data, _ = mb.load_data(a)
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in names}
    groups = rep.name_groups(names); folds = rep.grouped_folds(names, groups)
    res = {}
    for fam in FAMILIES:
        fold_docs = [Path(a.out_dir)/f'{fam}-fold{k}.json' for k in range(5)]
        if not all(f.exists() for f in fold_docs):
            continue
        docs = [json.loads(f.read_text()) for f in fold_docs]
        oof = {}
        for k in range(5):
            z = np.load(Path(a.out_dir)/f'{fam}-fold{k}-oof.npz')
            for key in z.files:
                p, s = key.rsplit('|s', 1)
                oof.setdefault(int(s), {})[p] = z[key]
        per_seed = []
        for s, pr in sorted(oof.items()):
            y = np.concatenate([data[n]['y'] for n in names]); q = np.clip(np.concatenate([pr[n] for n in names]), 1e-7, 1-1e-7)
            nf = []
            for n in names:
                cat = cats[n]
                if cat['feasible'].any():
                    o = catalogue.order(cat, pr[n]/max(pr[n].max(), 1e-12), 2.)
                    nf.append(int(np.flatnonzero(cat['feasible'][o])[0]) + 1)
            bins = np.minimum((q*10).astype(int), 9)
            ece = float(sum(abs(y[bins == b].mean() - q[bins == b].mean())*(bins == b).mean() for b in range(10) if (bins == b).any()))
            nf = np.array(nf)
            per_seed.append(dict(seed=s, auc=float(roc_auc_score(y, q)), log_loss=float(log_loss(y, q)), ece=ece,
                                 N_F_gmean=float(np.exp(np.log(nf).mean())), **{f'success@{kk}': float((nf <= kk).mean()) for kk in (1, 5, 10, 20, 50)}))
        agg = {m: float(np.mean([r[m] for r in per_seed])) for m in per_seed[0] if m != 'seed'}
        agg.update({f'{m}_sd': float(np.std([r[m] for r in per_seed])) for m in ('auc', 'N_F_gmean')})
        trials = [t for d in docs for t in d['trials']]
        res[fam] = dict(metrics=agg, per_seed=per_seed, best_params_by_fold=[d['best_params'] for d in docs],
                        trials=dict(total=len(trials), complete=sum(t['state'] == 'COMPLETE' for t in trials),
                                    pruned=sum(t['state'] == 'PRUNED' for t in trials), failed=sum(t['state'] == 'FAIL' for t in trials)),
                        search_seconds=sum(d['search_seconds'] for d in docs),
                        fit_seconds_mean=float(np.mean([c['fit_seconds'] for d in docs for c in d['confirmation']])),
                        predict_ms_per_part=float(np.mean([c['predict_ms_per_part'] for d in docs for c in d['confirmation']])),
                        params_mean=float(np.mean([c['params'] for d in docs for c in d['confirmation']])))
    ref = json.loads(Path(a.mlp_ref).read_text())['variants']
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), development_only=True, exploratory=True,
               families=res, references={k: {m: ref[k].get(m) for m in ('auc', 'log_loss', 'ece', 'N_F_gmean', 'success@1', 'train_s_per_fold',
                                                                        'predict_ms_per_part', 'params')} for k in ('L0', 'L1', 'L2', 'L8', 'L10') if k in ref})
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    for f, v in res.items():
        print(f, {m: round(x, 4) for m, x in v['metrics'].items()})
    print({k: (v['N_F_gmean'], v['auc']) for k, v in out['references'].items()})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('catalogues', 'store', 'library', 'test', 'out-dir'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--openclip', action='append', required=True)
    p.add_argument('--trials', type=int, default=30); p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--families', nargs='+', default=list(FAMILIES))
    p.add_argument('--summarise', action='store_true'); p.add_argument('--out')
    p.add_argument('--mlp-ref', default=str(ROOT/'research/phase7/results/mlp_benchmark.json'))
    a = p.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    if a.summarise:
        return summarise(a)
    tasks = [(f, k) for f in a.families for k in range(5)]
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=init, initargs=(a,)) as pool:
        for f, k, st in pool.map(job, tasks):
            print(f, k, st, flush=True)


if __name__ == '__main__':
    main()
