"""GPD's classifier as an examination order: N_F, Success@k, v3 robustness and load tolerance, cost.

    python tools/research/phase7_extension/gpd_evaluate.py needed --gpd-dev gpd_dev/ --gpd-egad gpd_egad/ \
        --dev research/phase6/dev_parts.json --catalogues cat80/ --store store.json --priors priors.npz \
        --sweep research/phase7/results/sweep_v2_extended.json.gz \
        --physics research/phase7/results/physics_dev_with_joint.json.gz --v3sweep v3sweep/ \
        --egad-conf CONF --out-dev needed_dev.json --out-egad needed_egad.json
    python tools/research/phase7_extension/gpd_evaluate.py analyse ... --extra-dev extra_dev/ --extra-egad extra_egad/ \
        --out research/phase7_extension/results/external_baselines_v2.json

Level: ranking-controlled (the same AxisForge candidates, gripper, gate and checks; GPD only orders them).
Orders compared on the same objects:

- GPD+gate: GPD scores with the planner's stroke gate first (research_grasping/phase6/fusion.order), the
  primary GPD order, because every AxisForge order puts makeable candidates first;
- GPD raw: descending GPD score, no gate (sensitivity);
- M0 (original planner), M1, M4, M9, Joint, the hybrid of Joint and M0, and the hybrid of GPD+gate and M0.

Development: the 151 dynamic parts, AxisForge orders out of fold (tools/research/phase7_extension/common.py
reproduces the Phase VII first grasps exactly). EGAD (exploratory: EGAD was already evaluated in Phase VII): the
frozen models' orders from the confirmation. Robustness and load tolerance are the v3 criterion at torsion
0.005 m; GPD first grasps not simulated before are simulated by extra_physics.py with the same code and
conditions. Cost: rendering two depth images, GPD preprocessing and classification of every candidate (one
thread), per object.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from research_grasping.phase5 import catalogue  # noqa: E402
from research_grasping.phase6 import fusion  # noqa: E402

METHODS = ('M0', 'M1', 'M4', 'M9', 'Joint')


def gpd_orders(cat, s):
    s = np.asarray(s, float)
    gate = fusion.order(cat, (s - s.min())/(s.max() - s.min()) if s.max() > s.min() else np.zeros_like(s))
    raw = np.argsort(-s, kind='stable')
    return gate, raw


def load_scores(d, n):
    f = Path(d)/f'{n}.json'
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return r if 'scores' in r else None


def dev_table(a):
    D = C.load_dev(a.dev, a.catalogues, a.store, a.priors, a.sweep, a.physics)
    rows = {}
    for n in D['dev']:
        r = load_scores(a.gpd_dev, n)
        if r is None:
            continue
        cat = D['cats'][n]
        g, raw = gpd_orders(cat, r['scores'])
        o = dict(D['orders'][n], **{'GPD+gate': g, 'GPD raw': raw})
        o['hybrid Joint/M0'] = C.interleave(cat, o['Joint'], o['M0'])
        o['hybrid GPD/M0'] = C.interleave(cat, g, o['M0'])
        rows[n] = dict(first={m: C.first(cat, v) for m, v in o.items()}, cost=dict(render=r['render_seconds'], gpd=r['gpd_seconds']),
                       cloud_points=r['cloud_points'])
    return D, rows


def egad_table(a):
    conf = Path(a.egad_conf)
    orows = json.loads((conf/'orders.json').read_text())['rows']
    rows = {}
    for n, r0 in orows.items():
        if r0['status'] != 'viable':
            continue
        r = load_scores(a.gpd_egad, n)
        if r is None:
            continue
        cat = catalogue.load(conf/'cat80'/f'{n}.npz')
        g, raw = gpd_orders(cat, r['scores'])
        first = {m: (r0[m]['nf'], r0[m]['first']) for m in METHODS}
        first['GPD+gate'] = C.first(cat, g); first['GPD raw'] = C.first(cat, raw)
        first['hybrid GPD/M0'] = C.first(cat, C.interleave(cat, g, catalogue.order(cat)))
        h = r0.get('hybrid') or r0.get('hybrid_interleave_Joint_M0')
        if h and 'nf' in h:
            first['hybrid Joint/M0'] = (h['nf'], h.get('first'))
        rows[n] = dict(first=first, cost=dict(render=r['render_seconds'], gpd=r['gpd_seconds']), cloud_points=r['cloud_points'],
                       set='evaluation' if n.startswith('egad_eval_') else 'primary')
    return rows


def dev_trials(a, D, n):
    """Trials by label for a development part: Phase VII physics, then the v3 sweep, then extra simulations."""
    out = {}
    for src in (Path(a.v3sweep)/f'{n}.json', Path(a.extra_dev)/f'{n}.json' if getattr(a, 'extra_dev', None) else None):
        if src and src.exists():
            out.update(json.loads(src.read_text())['trials'])
    out.update(D['ph']['trials'].get(n, {}))
    return out


def needed(a):
    D, dev = dev_table(a)
    need_dev = {}
    for n, r in dev.items():
        t = dev_trials(a, D, n)
        labs = {r['first'][m][1] for m in ('GPD+gate', 'GPD raw', 'hybrid GPD/M0') if r['first'][m][1]}
        miss = [lab for lab in labs if C.held(t.get(lab), 'robust') is None or C.held(t.get(lab), 'load') is None]
        if miss:
            need_dev[n] = sorted(miss)
    egad = egad_table(a)
    trials = json.load(gzip.open(Path(a.egad_trials), 'rt'))
    need_egad = {}
    for n, r in egad.items():
        t = trials.get(n, {}).get('trials', {})
        labs = {r['first'][m][1] for m in ('GPD+gate', 'GPD raw', 'hybrid GPD/M0') if r['first'][m][1]}
        miss = [lab for lab in labs if C.held(t.get(lab), 'robust') is None]
        if miss:
            need_egad[n] = sorted(miss)
    Path(a.out_dev).write_text(json.dumps(need_dev, indent=0)); Path(a.out_egad).write_text(json.dumps(need_egad, indent=0))
    print('development parts with GPD scores', len(dev), 'needing physics', len(need_dev), sum(map(len, need_dev.values())))
    print('EGAD viable objects with GPD scores', len(egad), 'needing physics', len(need_egad), sum(map(len, need_egad.values())))


def summarise(rows, outcome):
    """rows: {n: {'first': {m: (nf, label)}}}; outcome(n, label, which) -> 0/1/None."""
    names = sorted(rows)
    methods = list(next(iter(rows.values()))['first'])
    out = {}
    for m in methods:
        nf = [rows[n]['first'][m][0] for n in names]
        rob = [outcome(n, rows[n]['first'][m][1], 'robust') for n in names]
        load = [outcome(n, rows[n]['first'][m][1], 'load') for n in names]
        out[m] = dict(objects=len(names), N_F_gmean=C.gmean(nf), N_F_median=float(np.median(nf)), N_F_max=int(max(nf)),
                      success_at={k: float(np.mean([x <= k for x in nf])) for k in C.KS},
                      robust_v3=float(np.mean([x for x in rob if x is not None])) if any(x is not None for x in rob) else None,
                      load_tolerant_v3=float(np.mean([x for x in load if x is not None])) if any(x is not None for x in load) else None,
                      missing_outcomes=int(sum(x is None for x in rob)))
    contrasts = {}
    for m in ('GPD+gate', 'GPD raw', 'hybrid GPD/M0'):
        for ref in ('M0', 'M1', 'M9', 'Joint'):
            nf = C.boot_ratio([rows[n]['first'][m][0] for n in names], [rows[n]['first'][ref][0] for n in names])
            pair = [(outcome(n, rows[n]['first'][m][1], 'robust'), outcome(n, rows[n]['first'][ref][1], 'robust')) for n in names]
            pair = [p for p in pair if None not in p]
            lp = [(outcome(n, rows[n]['first'][m][1], 'load'), outcome(n, rows[n]['first'][ref][1], 'load')) for n in names]
            lp = [p for p in lp if None not in p]
            contrasts[f'{m} vs {ref}'] = dict(N_F_ratio=nf, robust_difference=C.boot_diff(*zip(*pair)) if pair else None,
                                              load_difference=C.boot_diff(*zip(*lp)) if lp else None)
    cost = dict(render_s=describe([rows[n]['cost']['render'] for n in names]), gpd_s=describe([rows[n]['cost']['gpd'] for n in names]),
                cloud_points=describe([sum(rows[n]['cloud_points']) for n in names]))
    return dict(methods=out, contrasts=contrasts, cost=cost)


def describe(v):
    v = np.asarray(v, float)
    return dict(median=float(np.median(v)), mean=float(v.mean()), p90=float(np.percentile(v, 90)), max=float(v.max()))


def analyse(a):
    D, dev = dev_table(a)

    def dev_outcome(n, label, which):
        return C.held(dev_trials(a, D, n).get(label), which) if label else None
    egad = {} if a.skip_egad else egad_table(a)
    trials = {} if a.skip_egad else json.load(gzip.open(Path(a.egad_trials), 'rt'))

    def egad_outcome(n, label, which):
        if not label:
            return None
        t = trials.get(n, {}).get('trials', {}).get(label)
        if t is None and Path(a.extra_egad, f'{n}.json').exists():
            t = json.loads(Path(a.extra_egad, f'{n}.json').read_text())['trials'].get(label)
        return C.held(t, which)
    res = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '),
               gpd=dict(source='https://github.com/atenpas/gpd', commit=a.gpd_commit, classifier='LeNet on 15-channel grasp images; GPD built without Caffe, OpenVINO or OpenCV DNN, i.e. its own Eigen implementation (EigenClassifier) on the CPU, one thread',
                        adapter='tools/research/phase7_extension/gpd_adapter (scores given candidates; GPD library unmodified)'),
               development=summarise(dev, dev_outcome),
               egad_exploratory=('not completed: GPD scoring of the already-evaluated EGAD objects was stopped after 21 of 255 '
                                 'objects to free compute for the new confirmation; no result is reported for it') if a.skip_egad else
               {s: summarise({n: r for n, r in egad.items() if r['set'] == s}, egad_outcome) for s in ('primary', 'evaluation')},
               presentation="Ranking-controlled comparison using GPD's pretrained classifier on a shared candidate catalogue.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    for k in ('development',):
        for m, v in res[k]['methods'].items():
            print(k, m, round(v['N_F_gmean'], 3), v['success_at'], v['robust_v3'], v['load_tolerant_v3'], v['missing_outcomes'])
    for s, blk in (res['egad_exploratory'].items() if isinstance(res['egad_exploratory'], dict) else []):
        for m, v in blk['methods'].items():
            print('egad', s, m, round(v['N_F_gmean'], 3), v['success_at'][1], v['robust_v3'], v['missing_outcomes'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('needed', 'analyse'):
        s = sub.add_parser(name)
        for k in ('gpd-dev', 'gpd-egad', 'dev', 'catalogues', 'store', 'priors', 'sweep', 'physics', 'v3sweep', 'egad-conf'):
            s.add_argument(f'--{k}', required=True)
        s.add_argument('--egad-trials', default='research/phase7/results/confirmation_trials.json.gz')
        s.add_argument('--extra-dev'); s.add_argument('--extra-egad')
        if name == 'needed':
            s.add_argument('--out-dev', required=True); s.add_argument('--out-egad', required=True)
        else:
            s.add_argument('--out', required=True); s.add_argument('--gpd-commit', required=True)
            s.add_argument('--skip-egad', action='store_true')
    a = p.parse_args()
    dict(needed=needed, analyse=analyse)[a.cmd](a)


if __name__ == '__main__':
    main()
