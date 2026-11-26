"""Why did Joint lose load tolerance against the original planner? (analysis of stored v3 telemetry, no new simulation)

    python tools/research/phase7_extension/load_inversion.py --egad-conf CONF --dev research/phase6/dev_parts.json \
        --dev-catalogues cat80/ --dev-store store.json --out research/phase7_extension/results/load_inversion.json

Data: the v3 trials of the first grasps already simulated, i.e. EGAD primary (Phase VII confirmation,
exploratory here) and development (physics_dev_with_joint). For each load (0.5, 1, 2 kg, torsion 0.005 m)
and method (M0, M1, Joint) the failure reasons are split as the v3 criterion defines them: unstable start,
rise < 80 mm (dropped), slip >= 20 mm, rotation from the grip > 15 degrees, pads not touching. On the objects
where M0 and Joint disagree under load, the two first grasps are compared on the 19 planner features
(width, approach and closing directions, relative position of the grasp centre, centre offset, ...).
Paired differences with bootstrap CIs; descriptive, no hypothesis test.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402

LOADS = ('load 0.5 kg', 'load 1 kg', 'load 2 kg')
METHODS = ('M0', 'M1', 'Joint')


def reason(r):
    if r.get('unstable'):
        return 'unstable'
    if r['held_v3']:
        return 'held'
    if r['rise'] < .08:
        return 'dropped (rise < 80 mm)'
    if r['slip'] >= .02:
        return 'slip >= 20 mm'
    if r.get('turn_from_grip_deg', 0) > 15:
        return 'rotation > 15 deg'
    if not r.get('touching', True):
        return 'contact lost'
    return 'other'


def analyse(rows, trials_of, feats_of, names):
    out = dict(objects=len(names), reasons={}, slip_mm={}, turn_deg={}, disagreement={})
    for load in LOADS:
        out['reasons'][load], out['slip_mm'][load], out['turn_deg'][load] = {}, {}, {}
        for m in METHODS:
            rs = [trials_of(n)[rows[n][m]]['0.005'][load] for n in names]
            cnt = {}
            for r in rs:
                cnt[reason(r)] = cnt.get(reason(r), 0) + 1
            out['reasons'][load][m] = cnt
            ok = [r for r in rs if not r.get('unstable')]
            out['slip_mm'][load][m] = dict(median=float(np.median([1e3*r['slip'] for r in ok])), p90=float(np.percentile([1e3*r['slip'] for r in ok], 90)))
            out['turn_deg'][load][m] = dict(median=float(np.median([r.get('turn_from_grip_deg', 0) for r in ok])),
                                           p90=float(np.percentile([r.get('turn_from_grip_deg', 0) for r in ok], 90)))
    # objects where M0 and Joint disagree on load tolerance (all three loads)
    tol = lambda n, m: all(trials_of(n)[rows[n][m]]['0.005'][l_]['held_v3'] for l_ in LOADS)  # noqa: E731
    m0_only = [n for n in names if tol(n, 'M0') and not tol(n, 'Joint')]
    j_only = [n for n in names if tol(n, 'Joint') and not tol(n, 'M0')]
    same = [n for n in names if rows[n]['M0'] == rows[n]['Joint']]
    out['disagreement'] = dict(m0_tolerant_joint_not=len(m0_only), joint_tolerant_m0_not=len(j_only), same_first_grasp=len(same))
    for label, group in (('m0_tolerant_joint_not', m0_only), ('all_objects', names)):
        if len(group) < 3:
            continue
        diffs = {}
        for i, f in enumerate(bl.FEATURE_NAMES):
            a_ = [feats_of(n, rows[n]['Joint'])[i] for n in group]; b_ = [feats_of(n, rows[n]['M0'])[i] for n in group]
            a_, b_ = np.nan_to_num(np.asarray(a_, float)), np.nan_to_num(np.asarray(b_, float))
            d = C.boot_diff(a_, b_)
            diffs[f] = dict(joint_mean=float(a_.mean()), m0_mean=float(b_.mean()), difference=d['difference'], ci95=d['ci95'])
        out[f'features_Joint_minus_M0[{label}]'] = diffs
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('egad-conf', 'dev', 'dev-catalogues', 'dev-store', 'out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--egad-results', default='research/phase7/results')
    p.add_argument('--dev-physics', default='research/phase7/results/physics_dev_with_joint.json.gz')
    a = p.parse_args()
    R = Path(a.egad_results)
    orows = json.loads((R/'confirmation_orders.json').read_text())['rows']
    trials = json.load(gzip.open(R/'confirmation_trials.json.gz', 'rt'))
    store = json.loads((Path(a.egad_conf)/'store.json').read_text())
    egad = sorted(n for n in orows if not n.startswith('egad_eval_') and orows[n]['status'] == 'viable' and n in trials)
    ecats = {}

    def efeat(n, label):
        if n not in ecats:
            cat = catalogue.load(Path(a.egad_conf)/'cat80'/f'{n}.npz')
            ecats[n] = ({str(l): i for i, l in enumerate(cat['label'])}, bl.features(cat, store[n]['low'], store[n]['high']))
        idx, F = ecats[n]
        return F[idx[label]]
    erows = {n: {m: orows[n][m]['first'] for m in METHODS} for n in egad}
    res = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '),
               egad_primary_exploratory=analyse(erows, lambda n: trials[n]['trials'], efeat, egad))
    ph = json.load(gzip.open(a.dev_physics, 'rt'))
    dstore = json.loads(Path(a.dev_store).read_text())
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    dcats = {}

    def dfeat(n, label):
        if n not in dcats:
            cat = catalogue.load(Path(a.dev_catalogues)/f'{n}.npz')
            dcats[n] = ({str(l): i for i, l in enumerate(cat['label'])}, bl.features(cat, dstore[n]['low'], dstore[n]['high']))
        idx, F = dcats[n]
        return F[idx[label]]
    drows = {n: {m: ph['core'][n][m] for m in METHODS} for n in dev}
    res['development'] = analyse(drows, lambda n: ph['trials'][n], dfeat, dev)
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    for k in ('egad_primary_exploratory', 'development'):
        print(k, json.dumps(res[k]['reasons']['load 2 kg']), res[k]['disagreement'])
        key = 'features_Joint_minus_M0[m0_tolerant_joint_not]'
        if key in res[k]:
            top = sorted(res[k][key].items(), key=lambda kv: -abs(kv[1]['difference'])/(abs(kv[1]['m0_mean'])+1e-9))[:6]
            for f, v in top:
                print('   ', f, round(v['joint_mean'], 4), round(v['m0_mean'], 4), [round(x, 4) for x in v['ci95']])


if __name__ == '__main__':
    main()
