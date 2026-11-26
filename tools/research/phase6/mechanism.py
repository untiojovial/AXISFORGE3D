"""Stage B: why did the learned ordering pick less robust grasps in Phase V?

    python tools/research/phase6/mechanism.py --catalogues cat80/ --store store.json \
        --library research/phase5/library_parts.txt --dev research/phase6/dev_parts.json \
        --phase5-dynamics research/phase5/results/dynamics.json.gz --priors priors.npz \
        [--sweep sweep.json.gz] --out robustness_mechanism.json --ranking-out ranking_quality_analysis.json

B1 failure modes by condition (Phase V trials, with the masses they really had),
B2 which candidate feature predicts holding, B3 whether the method effect goes
through that feature, B4 why the learned order picks it, B5 the same first
grasps under the v2 physics with the intended masses (needs --sweep).
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402
from research_grasping.phase6 import mechanism as mech  # noqa: E402

V1_REAL = {'nominal': 'nominal (0.1 kg)', 'mass x2': '2 kg (labelled "mass x2" in Phase V)',
           'mass x0.5': '0.5 kg (labelled "mass x0.5" in Phase V)', 'friction 0.3': 'friction 0.3',
           'pose noise 1': 'pose noise 1', 'pose noise 2': 'pose noise 2'}
METHODS = ('default', 'learned', 'afat', 'box', 'visual:dinov3')


def r4(x):
    return None if x is None else round(float(x), 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('catalogues', 'store', 'library', 'dev', 'phase5-dynamics', 'priors', 'out', 'ranking-out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--sweep')
    a = p.parse_args()
    store = json.loads(Path(a.store).read_text())
    library = [s.strip() for s in Path(a.library).read_text().splitlines() if s.strip()]
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in library+dev}
    d5 = json.load(gzip.open(a.phase5_dynamics))['parts']
    pri = np.load(a.priors)
    out = dict(note='Stage B is explanatory (exploratory). Unit of analysis: the part; intervals are bootstraps over parts.')

    # ---- B1: failure modes in the Phase V trials (v1 physics) ----
    b1 = {}
    for cond, real in V1_REAL.items():
        b1[real] = {}
        for m in METHODS:
            modes = [mech.failure_mode(d5[n]['methods'][m]['first_trials'][cond]) for n in dev]
            b1[real][m] = {k: modes.count(k) for k in ('held', 'dropped', 'pivoted or slid')}
    out['B1_failure_modes_phase5_v1'] = b1

    # ---- B2: which feature predicts holding (v1, unique first grasps) ----
    seen, X, co, H, G = set(), [], [], {c: [] for c in V1_REAL}, []
    for gi, n in enumerate(dev):
        cat = cats[n]
        labels = list(cat['label'])
        F = bl.features(cat, store[n]['low'], store[n]['high'])
        for m in METHODS:
            r = d5[n]['methods'][m]
            if (n, r['first']) in seen:
                continue
            seen.add((n, r['first']))
            i = labels.index(r['first'])
            X.append(F[i]); co.append(float(np.nan_to_num(cat['center_offset'][i]))); G.append(gi)
            for c in V1_REAL:
                H[c].append(int(r['first_trials'][c]['held']))
    X, co, G = np.array(X), np.array(co), np.array(G)
    robust = np.all(np.array([H[c] for c in V1_REAL]), axis=0).astype(int)
    from sklearn.metrics import roc_auc_score
    b2 = dict(grasps=int(len(X)), parts=len(dev), by_condition={})
    names = list(bl.FEATURE_NAMES)
    j_co = names.index('centre_offset')
    for c, y in [*((V1_REAL[c], np.array(H[c])) for c in ('mass x2', 'mass x0.5', 'nominal')), ('robust (all six)', robust)]:
        if 0 < y.sum() < len(y):
            uni = {f: r4(max(roc_auc_score(y, X[:, j]), 1-roc_auc_score(y, X[:, j]))) for j, f in enumerate(names)}
            top = sorted(uni.items(), key=lambda kv: -kv[1])[:6]
            auc_co, _ = mech.grouped_auc(co[:, None], y, G)
            auc_all, _ = mech.grouped_auc(X, y, G)
            auc_wo, _ = mech.grouped_auc(np.delete(X, j_co, 1), y, G)
            b2['by_condition'][c] = dict(rate=r4(y.mean()), univariate_auc_top=dict(top),
                                         logistic_grouped_cv_auc=dict(centre_offset_only=r4(auc_co), all_19=r4(auc_all),
                                                                      all_but_centre_offset=r4(auc_wo)),
                                         centre_offset_mm_held=r4(np.median(co[y == 1])*1e3),
                                         centre_offset_mm_failed=r4(np.median(co[y == 0])*1e3) if (y == 0).any() else None)
    out['B2_feature_vs_holding_v1'] = b2

    # ---- B3: does the method effect go through the lever arm? (v1) ----
    def first(n, m):
        cat = cats[n]
        i = list(cat['label']).index(d5[n]['methods'][m]['first'])
        return i, float(np.nan_to_num(cat['center_offset'][i]))
    rob = {m: np.array([all(t['held'] for t in d5[n]['methods'][m]['first_trials'].values()) for n in dev]) for m in METHODS}
    held2 = {m: np.array([d5[n]['methods'][m]['first_trials']['mass x2']['held'] for n in dev]) for m in METHODS}
    cos = {m: np.array([first(n, m)[1] for n in dev]) for m in METHODS}
    pooled = np.concatenate([cos['default'], cos['learned']])
    cuts = np.percentile(pooled, [100/3, 200/3])
    strata = {}
    for k, (lo, hi) in enumerate([(-1, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 9)]):
        row = {}
        for m in ('default', 'learned'):
            sel = (cos[m] > lo) & (cos[m] <= hi)
            row[m] = dict(grasps=int(sel.sum()), robust=r4(rob[m][sel].mean()) if sel.any() else None,
                          held_2kg=r4(held2[m][sel].mean()) if sel.any() else None)
        strata[f'tercile {k+1} ({max(lo, 0)*1e3:.0f}-{min(hi, .2)*1e3:.0f} mm)'] = row
    # predicted robustness from the lever arm alone, fitted on every first grasp of every method (out of fold)
    yy = np.concatenate([rob[m] for m in METHODS]).astype(int)
    xx = np.concatenate([cos[m] for m in METHODS])
    gg = np.concatenate([np.arange(len(dev)) for _ in METHODS])
    _, pred = mech.grouped_auc(xx[:, None], yy, gg)
    pm = {m: pred[i*len(dev):(i+1)*len(dev)] for i, m in enumerate(METHODS)}
    out['B3_method_effect_v1'] = dict(
        robust_rate={m: r4(rob[m].mean()) for m in METHODS},
        held_2kg_rate={m: r4(held2[m].mean()) for m in METHODS},
        centre_offset_mm_median={m: r4(np.median(cos[m])*1e3) for m in METHODS},
        learned_minus_default_robust=mech.bootstrap_diff(rob['learned'], rob['default']),
        learned_minus_default_robust_predicted_from_lever_arm=mech.bootstrap_diff(pm['learned'], pm['default']),
        learned_minus_default_centre_offset_mm=mech.bootstrap_diff(cos['learned']*1e3, cos['default']*1e3),
        mcnemar_robust_learned_vs_default=mech.mcnemar(rob['learned'], rob['default']),
        strata_by_centre_offset=strata)

    # ---- B4: why the learned order picks larger lever arms ----
    terms = {m: [mech.preliminary_terms(cats[n], first(n, m)[0]) for n in dev] for m in ('default', 'learned')}
    term_diff = {k: mech.bootstrap_diff([t[k] for t in terms['learned']], [t[k] for t in terms['default']])
                 for k in ('preliminary', 'centre', 'width', 'above', 'force', 'centre_offset_mm')}
    rho, same, rank_def_in_learned, tau = [], 0, [], []
    from scipy.stats import kendalltau, spearmanr
    for n in dev:
        cat = cats[n]
        f = np.flatnonzero(cat['feasible'])
        p = pri[f'{n}|learned'][f]
        c = np.nan_to_num(cat['center_offset'][f])
        if len(f) > 2 and np.ptp(p) > 0 and np.ptp(c) > 0:
            rho.append(spearmanr(p, c)[0])
        i0, i1 = first(n, 'default')[0], first(n, 'learned')[0]
        same += i0 == i1
        o1 = catalogue.order(cat, pri[f'{n}|learned']/max(pri[f'{n}|learned'].max(), 1e-12), 2.)
        o0 = catalogue.order(cat)
        feas1 = [i for i in o1 if cat['feasible'][i]]
        rank_def_in_learned.append(feas1.index(i0)+1)
        pos0 = {i: k for k, i in enumerate(o0)}
        pos1 = {i: k for k, i in enumerate(o1)}
        if len(f) > 2:
            tau.append(kendalltau([pos0[i] for i in f], [pos1[i] for i in f])[0])
    # library: feasibility against the lever arm and height
    co_l = np.concatenate([np.nan_to_num(cats[n]['center_offset']) for n in library])
    h_l = np.concatenate([bl.features(cats[n], store[n]['low'], store[n]['high'])[:, names.index('rel_z')] for n in library])
    y_l = np.concatenate([cats[n]['feasible'] for n in library])
    co_bins = [0, .01, .02, .03, .045, .06, .08, .12, 1]
    h_bins = [0, .25, .5, .75, 1.01]
    out['B4_why_learned_picks_longer_lever_arms'] = dict(
        preliminary_terms_learned_minus_default=term_diff,
        same_first_grasp_parts=int(same),
        spearman_p_vs_centre_offset_within_viable=dict(median=r4(np.median(rho)), q25=r4(np.percentile(rho, 25)),
                                                         q75=r4(np.percentile(rho, 75)), parts=len(rho)),
        default_first_grasp_rank_among_viable_in_learned_order=dict(
            median=float(np.median(rank_def_in_learned)), mean=r4(np.mean(rank_def_in_learned))),
        library_viable_rate_by_centre_offset_mm={f'{co_bins[k]*1e3:.0f}-{co_bins[k+1]*1e3:.0f}': dict(
            candidates=int(((co_l >= co_bins[k]) & (co_l < co_bins[k+1])).sum()),
            viable_rate=r4(y_l[(co_l >= co_bins[k]) & (co_l < co_bins[k+1])].mean()))
            for k in range(len(co_bins)-1)},
        library_viable_rate_by_relative_height={f'{h_bins[k]:.2f}-{h_bins[k+1]:.2f}': dict(
            candidates=int(((h_l >= h_bins[k]) & (h_l < h_bins[k+1])).sum()),
            viable_rate=r4(y_l[(h_l >= h_bins[k]) & (h_l < h_bins[k+1])].mean()))
            for k in range(len(h_bins)-1)})
    ranking = dict(
        note='Default (M0) against the learned prior (M1) on the 151 development parts, 80 mm, experiment B.',
        kendall_tau_orders_over_viable=dict(median=r4(np.median(tau)), q25=r4(np.percentile(tau, 25)),
                                            q75=r4(np.percentile(tau, 75))),
        first_viable_preliminary_terms_mean={m: {k: r4(np.mean([t[k] for t in terms[m]]))
                                                  for k in ('preliminary', 'centre', 'width', 'above', 'force',
                                                            'centre_offset_mm')} for m in terms},
        learned_minus_default=term_diff,
        same_first_grasp_parts=int(same),
        default_first_grasp_rank_among_viable_in_learned_order=out['B4_why_learned_picks_longer_lever_arms'][
            'default_first_grasp_rank_among_viable_in_learned_order'],
        preliminary_score_formula='(0.15 width + 0.2 force + 0.25 centre + 0.3 from-above)/0.9; the prior is added with weight 2')

    # ---- B5: the same first grasps under v2 with the intended masses ----
    if a.sweep:
        sw = json.load(gzip.open(a.sweep, 'rt'))
        std, loads = sw['standard'], ['load 0.5 kg', 'load 1 kg', 'load 2 kg']
        b5 = dict(conditions=list(sw['conditions']), rates={}, mcnemar_learned_vs_default={})
        res = {}
        for m in METHODS:
            rows = [sw['parts'][n][d5[n]['methods'][m]['first']] for n in dev]
            res[m] = dict(robust=np.array([all(r[c][0] for c in std) for r in rows]),
                          load=np.array([all(r[c][0] for c in loads) for r in rows]),
                          held2kg=np.array([r['load 2 kg'][0] for r in rows]),
                          **{c: np.array([r[c][0] for r in rows]) for c in sw['conditions']})
            b5['rates'][m] = {**{c: r4(res[m][c].mean()) for c in sw['conditions']},
                              'robust (six standard)': r4(res[m]['robust'].mean()),
                              'load tolerant (0.5, 1, 2 kg)': r4(res[m]['load'].mean())}
        for k in ('robust', 'load', 'held2kg', 'mass x2'):
            b5['mcnemar_learned_vs_default'][k] = dict(**mech.mcnemar(res['learned'][k], res['default'][k]),
                                                      difference=mech.bootstrap_diff(res['learned'][k], res['default'][k]))
        out['B5_v2_intended_masses'] = b5
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    Path(a.ranking_out).write_text(json.dumps(ranking, indent=1) + '\n')


if __name__ == '__main__':
    main()
