"""Stage C: simple fusions of feasibility and quality, as fixed in PHASE6_PROTOCOL.md.

    python tools/research/phase6/fusion_bench.py --catalogues cat80/ --dev research/phase6/dev_parts.json \
        --priors priors.npz --sweep sweep.json.gz --protocol research/phase6/protocol.json --out fusion_benchmark.json

Every ordering is computed with the exact planner-order simulator; the first
viable grasp's dynamic outcome is looked up in the v2 sweep. Families M3-M8 are
tuned by nested cross-validation by part (criterion: mean log N_FR) and every
part is scored with the configuration chosen without it.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402
from research_grasping.phase5.stats import _boot, holm  # noqa: E402
from research_grasping.phase6 import fusion, mechanism as mech  # noqa: E402

FAMILIES = ('M3', 'M4', 'M5', 'M6', 'M7', 'M8')
LOADS = ('load 0.5 kg', 'load 1 kg', 'load 2 kg')


def outcome(cat, order, dyn, standard):
    """N_F, robust first, N_FR, load-tolerant first, nominal-held first, turn of the first (nominal)."""
    labels = cat['label']
    nf, first, nfr = None, None, None
    for k, i in enumerate(order):
        if not cat['feasible'][i]:
            continue
        d = dyn[str(labels[i])]
        # pass 1 stops at a candidate's first failure: a missing condition means an earlier one failed
        robust = all(c in d and d[c][0] for c in standard)
        if nf is None:
            nf, first = k+1, d
        if robust:
            nfr = k+1
            break
    return dict(nf=nf, robust=int(all(c in first and first[c][0] for c in standard)), nfr=nfr,
                load=int(all(first[c][0] for c in LOADS)), nominal=int(first['nominal'][0]),
                turn=first['nominal'][3])


def ratio(a, b):
    """Paired geometric-mean ratio a/b over parts, with bootstrap CI and Wilcoxon."""
    d = np.log(np.asarray(a, float)/np.asarray(b, float))
    nz = d[d != 0]
    return dict(ratio=round(float(np.exp(d.mean())), 4), ci95=[round(x, 4) for x in _boot(d, 95)], parts=int(len(d)),
                better=int((d < 0).sum()), worse=int((d > 0).sum()),
                wilcoxon_p=float(st.wilcoxon(nz).pvalue) if len(nz) else 1.)


def gmean(x):
    return round(float(np.exp(np.mean(np.log(x)))), 3)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('catalogues', 'dev', 'priors', 'sweep', 'protocol', 'out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    proto = json.loads(Path(a.protocol).read_text())['fusion']
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    pri = np.load(a.priors)
    sw = json.load(gzip.open(a.sweep, 'rt'))
    standard = sw['standard']
    res = {}  # (method, config index) -> per-part outcomes
    configs = {f: fusion.GRID[f] for f in FAMILIES}
    rand_nf = []
    for n in dev:
        cat = catalogue.load(Path(a.catalogues)/f'{n}.npz')
        dyn = sw['parts'][n]
        pl = pri[f'{n}|learned'].astype(float); pt = pl/pl.max()
        pa = pri[f'{n}|afat'].astype(float); at = pa/pa.max()
        for name, order in (('M0', catalogue.order(cat)), ('M1', catalogue.order(cat, pt, 2.)),
                            ('M2', catalogue.order(cat, at, 2.))):
            res.setdefault((name, 0), []).append(outcome(cat, order, dyn, standard))
        for f in FAMILIES:
            for j, cfg in enumerate(configs[f]):
                s = fusion.config_score(f, cfg, cat, pt)
                res.setdefault((f, j), []).append(outcome(cat, fusion.order(cat, s), dyn, standard))
        e = []
        for s in range(20):
            u = bl.random_order(cat, s)
            order = catalogue.order(cat, (u + 2*cat['makeable'])/3, fusion.WEIGHT)
            e.append(int(np.flatnonzero(cat['feasible'][order])[0])+1)
        rand_nf.append(float(np.exp(np.mean(np.log(e)))))
    P = len(dev)
    has_robust = np.array([any(all(c in v and v[c][0] for c in standard) for v in sw['parts'][n].values()) for n in dev])

    def arr(key, k, field):
        return np.array([r[field] if r[field] is not None else np.nan for r in res[(key, k)]], float)

    # ---- nested cross-validation (outer folds by part) ----
    order = np.random.default_rng(proto['tuning']['seed']).permutation(P)
    folds = [order[f::proto['tuning']['outer_folds']] for f in range(proto['tuning']['outer_folds'])]
    cv, chosen = {}, {}
    for f in FAMILIES:
        out = {k: np.full(P, np.nan) for k in ('nf', 'robust', 'nfr', 'load', 'nominal', 'turn')}
        chosen[f] = []
        for k, test in enumerate(folds):
            train = np.setdiff1d(np.arange(P), test)
            tr = train[has_robust[train]]
            crit = [np.mean(np.log(arr(f, j, 'nfr')[tr])) for j in range(len(configs[f]))]
            best = int(np.argmin(crit))
            chosen[f].append(dict(fold=k, config=configs[f][best], criterion=round(float(crit[best]), 4)))
            for key in out:
                out[key][test] = arr(f, best, key)[test]
        cv[f] = out
    ref = {m: {k: arr(m, 0, k) for k in ('nf', 'robust', 'nfr', 'load', 'nominal', 'turn')} for m in ('M0', 'M1', 'M2')}
    allm = {**ref, **cv}

    def summary(o):
        hr = has_robust
        return dict(N_F_gmean=gmean(o['nf']), robust_rate=round(float(o['robust'].mean()), 4),
                    N_FR_gmean=gmean(o['nfr'][hr]), load_tolerant_rate=round(float(o['load'].mean()), 4),
                    nominal_held_rate=round(float(o['nominal'].mean()), 4),
                    turn_deg_median=round(float(np.nanmedian(o['turn'])), 2))
    doc = dict(note='Development (pilot) results: the 151 Phase V test parts are development data in Phase VI. '
                    'M3-M8 are outer-fold estimates of nested cross-validation; nothing here is confirmatory.',
               parts=P, parts_with_a_robust_viable_grasp=int(has_robust.sum()),
               reference=dict(random_within_gate_N_F_gmean=gmean(rand_nf),
                              random_within_gate_over_M0=ratio(rand_nf, ref['M0']['nf'])),
               summary={m: summary(o) for m, o in allm.items()}, chosen=chosen)
    # ---- contrasts ----
    hr = has_robust
    con = {'N_F': [], 'N_FR': [], 'robust': [], 'load_tolerant': []}
    for f in ('M2', *FAMILIES):
        for b in ('M0', 'M1'):
            con['N_F'].append(dict(a=f, b=b, **ratio(allm[f]['nf'], allm[b]['nf'])))
            con['N_FR'].append(dict(a=f, b=b, **ratio(allm[f]['nfr'][hr], allm[b]['nfr'][hr])))
            for key, field in (('robust', 'robust'), ('load_tolerant', 'load')):
                mc = mech.mcnemar(allm[f][field], allm[b][field])
                con[key].append(dict(a=f, b=b, rate_a=round(float(allm[f][field].mean()), 4),
                                     rate_b=round(float(allm[b][field].mean()), 4),
                                     difference=mech.bootstrap_diff(allm[f][field], allm[b][field]),
                                     mcnemar_only_a=mc['only_a'], mcnemar_only_b=mc['only_b'], wilcoxon_p=mc['p']))
    for key in con:
        holm(con[key])
    con['M1_vs_M0'] = dict(N_F=ratio(ref['M1']['nf'], ref['M0']['nf']),
                           N_FR=ratio(ref['M1']['nfr'][hr], ref['M0']['nfr'][hr]),
                           robust=dict(difference=mech.bootstrap_diff(ref['M1']['robust'], ref['M0']['robust']),
                                       **mech.mcnemar(ref['M1']['robust'], ref['M0']['robust'])),
                           load_tolerant=dict(difference=mech.bootstrap_diff(ref['M1']['load'], ref['M0']['load']),
                                              **mech.mcnemar(ref['M1']['load'], ref['M0']['load'])))
    doc['contrasts'] = con
    # ---- decision rule ----
    rule = proto['decision_rule']
    decisions = {}
    for f in FAMILIES:
        r0 = next(c for c in con['robust'] if c['a'] == f and c['b'] == 'M0')
        n0 = next(c for c in con['N_F'] if c['a'] == f and c['b'] == 'M0')
        n1 = next(c for c in con['N_F'] if c['a'] == f and c['b'] == 'M1')
        c1 = r0['difference']['ci95'][0] > -.05
        c2 = n0['ci95'][1] < 1
        c3 = n1['ci95'][1] <= 1.25
        decisions[f] = dict(robustness_noninferior_to_M0=c1, faster_than_M0=c2, keeps_M1_speed=c3, meets_rule=c1 and c2 and c3,
                            robust_minus_M0_ci95=r0['difference']['ci95'], N_F_vs_M0_ci95=n0['ci95'], N_F_vs_M1_ci95=n1['ci95'])
    doc['decision_rule'] = dict(rule=rule, families=decisions)
    # ---- descriptive grid (all parts; not used for selection) ----
    doc['grid'] = {f: [dict(config=cfg, N_F_gmean=gmean(arr(f, j, 'nf')), robust_rate=round(float(arr(f, j, 'robust').mean()), 4),
                            N_FR_gmean=gmean(arr(f, j, 'nfr')[hr]), load_tolerant_rate=round(float(arr(f, j, 'load').mean()), 4))
                       for j, cfg in enumerate(configs[f])] for f in FAMILIES}
    doc['per_part'] = {m: {n: dict(nf=int(o['nf'][i]), robust=int(o['robust'][i]),
                                   nfr=None if np.isnan(o['nfr'][i]) else int(o['nfr'][i]), load=int(o['load'][i]))
                           for i, n in enumerate(dev)} for m, o in allm.items()}
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')


if __name__ == '__main__':
    main()
