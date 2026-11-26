"""v3 robustness models, fusion variants and the development selection (DEVELOPMENT_SELECTION_RULE.md).

    python tools/research/phase7_extension/v3_models.py needed  --v3sweep v3sweep/ --extra v3extra/ COMMON... --out needed.json
    python tools/research/phase7_extension/v3_models.py analyse --v3sweep v3sweep/ --extra v3extra/ COMMON... \
        --out research/phase7_extension/results/v3_models.json
    COMMON = --dev research/phase6/dev_parts.json --catalogues cat80/ --store store.json --priors priors.npz \
             --sweep research/phase7/results/sweep_v2_extended.json.gz --physics research/phase7/results/physics_dev_with_joint.json.gz

Development only, out of fold (the Phase VII development folds). Labels come from the v3 sweep (fresh v3
simulations of sampled viable candidates); the models, variants, metrics and the selection rule are those
committed in research/phase7_extension/DEVELOPMENT_SELECTION_RULE.md before this script was run. 'needed' lists
the first grasps of every variant that have no v3 trials yet (to be simulated with v3_sweep.py
--labels-json); 'analyse' refuses to select while any first-grasp outcome is missing.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from research_grasping.phase5 import baselines as bl  # noqa: E402
from research_grasping.phase6 import fusion  # noqa: E402
from research_grasping.phase7 import representation as rep  # noqa: E402

TORS = ('0.001', '0.005', '0.02')
EXT = ('friction 0.7', 'com +5 mm x', 'com -5 mm x')
MODELS = ('GB-v2 (historical)', 'GB-v3', 'HGB-v3', 'LogReg-v3', 'MLP-v3', 'GB-all3', 'GB-load')


def labels_of(trials):
    """Label dict of one candidate's sweep record, or None if incomplete."""
    try:
        y_tor = {t: int(all(trials[t][c]['held_v3'] for c in C.STANDARD)) for t in TORS}
        r = [int(trials[t][c]['held_v3']) for t in TORS for c in C.STANDARD]
        return dict(y_std=y_tor['0.005'], y_tor=y_tor, y_all3=int(all(y_tor.values())), r_mean=float(np.mean(r)),
                    y_ext=int(all(trials['0.005'][c]['held_v3'] for c in EXT)),
                    y_load=int(all(trials['0.005'][c]['held_v3'] for c in C.LOADS)))
    except KeyError:
        return None


def load_labels(a, dev):
    lab, sampled = {}, {}
    man = json.loads((Path(a.v3sweep)/'manifest.json').read_text())['parts']
    for n in dev:
        lab[n] = {}
        for d in (Path(a.v3sweep), Path(a.extra)):
            f = d/f'{n}.json'
            if f.exists():
                for label, t in json.loads(f.read_text())['trials'].items():
                    y = labels_of(t)
                    if y:
                        lab[n][label] = y
        sampled[n] = set(man[n]['sampled']) if n in man else set()
    return lab, sampled


def fit_models(D, lab, sampled, train, k):
    """Every P(H|F) model of the rule, fitted on the training parts' simulated viable candidates."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, Y, P = [], {'y_std': [], 'y_all3': [], 'y_load': []}, []
    for n in train:
        cat = D['cats'][n]; lab_idx = {str(l): i for i, l in enumerate(cat['label'])}
        F = D['X'][n]
        for label, y in lab[n].items():
            X.append(F[lab_idx[label]]); P.append(n)
            for key in Y:
                Y[key].append(y[key])
    X = np.asarray(X); Y = {k_: np.asarray(v, int) for k_, v in Y.items()}
    t0 = time.perf_counter()
    m = {'GB-v3': bl.fit(X, Y['y_std'], 0), 'GB-all3': bl.fit(X, Y['y_all3'], 0), 'GB-load': bl.fit(X, Y['y_load'], 0),
         'HGB-v3': HistGradientBoostingClassifier(random_state=0).fit(X, Y['y_std']),
         'LogReg-v3': make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000)).fit(np.nan_to_num(X), Y['y_std'])}
    secs = time.perf_counter() - t0
    # geometric MLP (Phase VII architecture, blocks v and u), seeds 0-2, early stopping on 15 % of the training parts
    data = {}
    for n in train:
        cat = D['cats'][n]; lab_idx = {str(l): i for i, l in enumerate(cat['label'])}
        rows = [lab_idx[l] for l in lab[n]]
        if rows:
            F = D['X'][n][rows]
            data[n] = dict(v=F[:, rep.V_COLUMNS].astype(np.float32), y=np.array([lab[n][l]['y_std'] for l in lab[n]], np.float32), u=D['U'][n])
    parts = sorted(data)
    rng = np.random.default_rng(3000 + k)
    val = set(rng.choice(parts, max(1, int(round(.15*len(parts)))), replace=False))
    m['MLP-v3'] = [rep.MLPModel(geometry=True, seed=s).fit(data, [p for p in parts if p not in val], sorted(val)) for s in range(3)]
    return m, dict(training_candidates=len(X), training_parts=len(set(P)), positives_y_std=int(Y['y_std'].sum()), fit_seconds_trees=secs)


def predict_all(D, models, n, h_hist):
    F = D['X'][n]
    out = {'GB-v2 (historical)': h_hist}
    for name in ('GB-v3', 'GB-all3', 'GB-load', 'HGB-v3'):
        out[name] = models[name].predict_proba(F)[:, 1]
    out['LogReg-v3'] = models['LogReg-v3'].predict_proba(np.nan_to_num(F))[:, 1]
    d = {n: dict(v=F[:, rep.V_COLUMNS].astype(np.float32), y=np.zeros(len(F), np.float32), u=D['U'][n])}
    out['MLP-v3'] = np.mean([mm.predict(d, [n])[n] for mm in models['MLP-v3']], 0)
    return out


def nondominated_rank(p, h, q):
    pts = np.stack([p, h, q], 1)
    rank = np.zeros(len(p), int)
    remaining = np.arange(len(p)); r = 0
    while len(remaining):
        sub = pts[remaining]
        dom = np.array([np.any(np.all(sub >= x, 1) & np.any(sub > x, 1)) for x in sub])
        rank[remaining[~dom]] = r; remaining = remaining[dom]; r += 1
    return rank


def variant_orders(D, n, H, k):
    cat = D['cats'][n]; pt = D['pri'][n]; q = fusion.quality(cat, 'cen'); qb = lambda b: np.maximum(q, 1e-3)**b  # noqa: E731
    o = dict(D['orders'][n])  # M0, M1, M4, M9, Joint (historical, out of fold)
    h = H['GB-v3']
    o['Joint-v3'] = fusion.order(cat, pt*h*qb(.3))
    for name in ('HGB-v3', 'LogReg-v3', 'MLP-v3'):
        o[f'Joint-v3[{name}]'] = fusion.order(cat, pt*H[name]*qb(.3))
    o['Joint-multi'] = fusion.order(cat, pt*H['GB-all3']*qb(.3))
    o['Joint-load'] = fusion.order(cat, pt*h*H['GB-load']*qb(.3))
    o['arithmetic'] = fusion.order(cat, (pt + h)/2*qb(.3))
    o['geometric'] = fusion.order(cat, np.sqrt(pt*h)*qb(.3))
    o['lexicographic'] = fusion.order(cat, (h >= .5).astype(float) + pt/2.)  # h >= 0.5 first, then p~ (p~/2 < 1)
    nd = nondominated_rank(pt, h, q)
    o['multi-objective'] = fusion.order(cat, (nd.max() - nd + (pt*h)/(1 + 1e-9))/(nd.max() + 1))
    o['ablation: Joint-v3 beta 0'] = fusion.order(cat, pt*h)
    o['ablation: no h, beta 0.3'] = fusion.order(cat, pt*qb(.3))
    o['ablation: Joint-v3 beta 0.1'] = fusion.order(cat, pt*h*qb(.1))
    o['ablation: Joint-v3 beta 0.5'] = fusion.order(cat, pt*h*qb(.5))
    o['hybrid Joint-v3/M0'] = C.interleave(cat, o['Joint-v3'], o['M0'])
    return o


SELECTABLE = ('M1', 'M4', 'M9', 'Joint', 'Joint-v3', 'Joint-v3[HGB-v3]', 'Joint-v3[LogReg-v3]', 'Joint-v3[MLP-v3]', 'Joint-multi',
              'Joint-load', 'arithmetic', 'geometric', 'lexicographic', 'multi-objective')
COMPLEXITY = {'M1': 0, 'M4': 1, 'M9': 2, 'Joint': 3, 'Joint-v3': 3, 'Joint-multi': 3, 'Joint-v3[HGB-v3]': 4, 'Joint-v3[LogReg-v3]': 4,
              'Joint-load': 5, 'arithmetic': 6, 'geometric': 6, 'lexicographic': 6, 'multi-objective': 7, 'Joint-v3[MLP-v3]': 8}


def run(a):
    D = C.load_dev(a.dev, a.catalogues, a.store, a.priors, a.sweep, a.physics)
    dev = D['dev']
    D['X'] = {n: bl.features(D['cats'][n], D['store'][n]['low'], D['store'][n]['high']) for n in dev}
    D['U'] = {n: np.concatenate([D['X'][n][0, rep.EXTENT_COLUMNS], np.asarray(D['store'][n]['descriptors']['d2'], float)]).astype(np.float32)
              for n in dev}
    lab, sampled = load_labels(a, dev)
    perm = np.random.default_rng(20261130).permutation(len(dev))
    folds = [perm[f::5] for f in range(5)]
    preds_sim, firsts, orders_all, fit_info = {}, {}, {}, []
    for k, test in enumerate(folds):
        train = [dev[i] for i in np.setdiff1d(np.arange(len(dev)), test)]
        models, info = fit_models(D, lab, sampled, train, k)
        fit_info.append(dict(fold=k, **info))
        for i in test:
            n = dev[i]; cat = D['cats'][n]
            H = predict_all(D, models, n, D['h'][n])
            idx = {str(l): j for j, l in enumerate(cat['label'])}
            preds_sim[n] = {label: {m: float(H[m][idx[label]]) for m in MODELS} for label in lab[n]}
            o = variant_orders(D, n, H, k)
            orders_all[n] = o
            firsts[n] = {m: C.first(cat, v) for m, v in o.items()}
        print('fold', k, info, flush=True)
    return D, lab, sampled, preds_sim, firsts, orders_all, fit_info


def needed(a):
    D, lab, sampled, preds, firsts, orders, info = run(a)
    want = {}
    for n, f in firsts.items():
        miss = sorted({lab_ for nf, lab_ in f.values() if lab_ and lab_ not in lab[n]})
        if miss:
            want[n] = miss
    Path(a.out).write_text(json.dumps(want, indent=0))
    print('parts with missing first grasps', len(want), 'candidates', sum(map(len, want.values())))


def prob_metrics(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    bins = np.minimum((p*10).astype(int), 9)
    rel = [dict(bin=b, n=int((bins == b).sum()), mean_p=float(p[bins == b].mean()), rate=float(y[bins == b].mean())) for b in range(10) if (bins == b).any()]
    return dict(n=len(y), positives=int(y.sum()), auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p)),
                ece=float(sum(abs(r['rate']-r['mean_p'])*r['n'] for r in rel)/len(y)), reliability=rel)


def analyse(a):
    D, lab, sampled, preds, firsts, orders, info = run(a)
    dev = D['dev']
    missing = {n: [l_ for _, l_ in f.values() if l_ and l_ not in lab[n]] for n, f in firsts.items()}
    missing = {n: v for n, v in missing.items() if v}
    if missing:
        raise SystemExit(f'first grasps without v3 trials on {len(missing)} parts: run needed + v3_sweep --labels-json first')
    # probability metrics on held-out simulated candidates (random sample only, so the metric is not tilted by first grasps)
    pm = {}
    for m in MODELS:
        for target in ('y_std', 'y_all3', 'y_load'):
            ys, ps = [], []
            for n in dev:
                for label in lab[n]:
                    if label in sampled[n]:
                        ys.append(lab[n][label][target]); ps.append(preds[n][label][m])
            pm.setdefault(m, {})[target] = prob_metrics(ys, ps)
    # ordering metrics
    methods = list(next(iter(firsts.values())))
    om = {}
    for m in methods:
        nf = [firsts[n][m][0] for n in dev]
        out = lambda key: [lab[n][firsts[n][m][1]][key] for n in dev]  # noqa: E731
        # N_FR among simulated candidates: rank, in this order, of the first simulated candidate that is robust (y_std)
        nfr = []
        for n in dev:
            sim = [j for j, l_ in enumerate(D['cats'][n]['label'][orders[n][m]]) if str(l_) in lab[n]]
            pos = [r+1 for r, j in enumerate(sim) if lab[n][str(D['cats'][n]['label'][orders[n][m][j]])]['y_std']]
            nfr.append(pos[0] if pos else len(sim) + 1)
        om[m] = dict(N_F_gmean=C.gmean(nf), success_at={kk: float(np.mean([x <= kk for x in nf])) for kk in C.KS},
                     robust_y_std=float(np.mean(out('y_std'))), robust_y_all3=float(np.mean(out('y_all3'))),
                     robust_by_torsion={t: float(np.mean([lab[n][firsts[n][m][1]]['y_tor'][t] for n in dev])) for t in TORS},
                     r_mean=float(np.mean(out('r_mean'))), extended=float(np.mean(out('y_ext'))), load=float(np.mean(out('y_load'))),
                     N_FR_simulated_gmean=C.gmean(nfr))
    # paired contrasts against M1 and M0 on the primary criterion
    contr = {}
    for m in methods:
        for ref in ('M1', 'M0'):
            if m == ref:
                continue
            a_ = [lab[n][firsts[n][m][1]]['y_all3'] for n in dev]; b_ = [lab[n][firsts[n][ref][1]]['y_all3'] for n in dev]
            contr[f'{m} vs {ref}'] = dict(y_all3=C.boot_diff(a_, b_),
                                          y_std=C.boot_diff([lab[n][firsts[n][m][1]]['y_std'] for n in dev],
                                                            [lab[n][firsts[n][ref][1]]['y_std'] for n in dev]),
                                          N_F=C.boot_ratio([firsts[n][m][0] for n in dev], [firsts[n][ref][0] for n in dev]))
    # the selection rule
    guard = 1.10*om['M1']['N_F_gmean']
    eligible = [m for m in SELECTABLE if om[m]['N_F_gmean'] <= guard]
    best_rate = max(om[m]['robust_y_all3'] for m in eligible)
    tied = [m for m in eligible if om[m]['robust_y_all3'] >= best_rate - .01]
    cand = min(tied, key=lambda m: (COMPLEXITY[m], -om[m]['robust_y_all3']))
    adv = om[cand]['robust_y_all3'] - om['M1']['robust_y_all3']
    final = cand if cand != 'M1' and adv >= .02 else 'M1'
    selection = dict(rule='research/phase7_extension/DEVELOPMENT_SELECTION_RULE.md', efficiency_guard_N_F=guard, eligible=eligible,
                     best_y_all3=best_rate, within_one_point=tied, candidate=cand, advantage_over_M1_pp=100*adv, final=final,
                     statement=('no robustness variant beat M1 by the required 2 points in development' if final == 'M1' else
                                f'{final} beat M1 by {100*adv:.1f} points on y_all3 in development (optimistic: selected on the same parts)'))
    res = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), parts=len(dev),
               labels=dict(simulated_candidates=sum(map(len, lab.values())), sampled=sum(len(s) for s in sampled.values()),
                           y_std_rate_sampled=float(np.mean([lab[n][l_]['y_std'] for n in dev for l_ in lab[n] if l_ in sampled[n]])),
                           y_all3_rate_sampled=float(np.mean([lab[n][l_]['y_all3'] for n in dev for l_ in lab[n] if l_ in sampled[n]]))),
               fits=info, probability_metrics=pm, ordering=om, contrasts=contr, selection=selection,
               first_grasps={n: {m: list(v) for m, v in f.items()} for n, f in firsts.items()})
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    for m in methods:
        v = om[m]
        print(f"{m:32s} N_F {v['N_F_gmean']:.3f} std {v['robust_y_std']:.3f} all3 {v['robust_y_all3']:.3f} load {v['load']:.3f} ext {v['extended']:.3f}")
    print(json.dumps(selection, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('needed', 'analyse'):
        s = sub.add_parser(name)
        for k in ('dev', 'catalogues', 'store', 'priors', 'sweep', 'physics', 'v3sweep', 'extra', 'out'):
            s.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    dict(needed=needed, analyse=analyse)[a.cmd](a)


if __name__ == '__main__':
    main()
