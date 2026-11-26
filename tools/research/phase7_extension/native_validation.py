"""Equivalence of the native learned orders with the scikit-learn / Python research pipeline.

    python tools/research/phase7_extension/native_validation.py --binary target/release/axisforge \
        --models research/phase7_extension/native/learned_order_phase7.json \
        --dev research/phase6/dev_parts.json --library research/phase5/library_parts.txt --dev-catalogues cat80/ \
        --dev-store store.json --dev-projects POOL --egad-pool EGAD_POOL --egad-conf CONF \
        --out research/phase7_extension/results/native_inference_validation.json --jobs 1

Tolerances fixed before running (stated in the output): features within 1e-12 absolute, probabilities
within 1e-9 absolute, and identical decisions (first viable grasp and number of examined candidates) for
every object and mode. Level 1 (research-candidates --learned): features and probabilities of every
candidate of every development part (316) and EGAD object (345) against bl.features and
predict_proba. Level 2 (plan --learned --order MODE --nr-grasps 1): the first grasp and the number of
examined candidates of every viable object and mode against the Python order simulator; for EGAD the
reference is the confirmation's own orders.json (frozen models), for development the same frozen
models applied in Python here. The frozen P(F) was trained on all development parts, so development
probabilities are in-sample: this checks equivalence, not accuracy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402
from research_grasping.phase6 import fusion  # noqa: E402

FEATURE_TOL, PROB_TOL = 1e-12, 1e-9
MODES = {'m1': 'M1', 'm4': 'M4', 'm9': 'M9', 'joint': 'Joint', 'hybrid': 'hybrid'}


def py_orders(cat, pt, h):
    o = {'M0': catalogue.order(cat), 'M1': catalogue.order(cat, pt, 2.),
         'M4': fusion.order(cat, fusion.config_score('M4', dict(alpha=.8, q='cen'), cat, pt)),
         'M9': fusion.order(cat, pt*h), 'Joint': fusion.order(cat, pt*h*np.maximum(fusion.quality(cat, 'cen'), 1e-3)**.3)}
    seen, hyb = set(), []
    for k in range(len(o['Joint'])):
        for x in (o['Joint'][k], o['M0'][k]):
            if int(x) not in seen:
                seen.add(int(x)); hyb.append(int(x))
    o['hybrid'] = np.array(hyb)
    return o


def first(cat, order):
    pos = np.flatnonzero(cat['feasible'][order])
    return (int(pos[0])+1, str(cat['label'][order[pos[0]]])) if len(pos) else (None, None)


def run(args, timeout=900):
    return json.loads(subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout,
                                     env={'RAYON_NUM_THREADS': '1', 'PATH': '/usr/bin:/bin'}).stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('binary', 'models', 'dev', 'library', 'dev-catalogues', 'dev-store', 'dev-projects', 'egad-pool', 'egad-conf', 'out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--frozen', default=str(ROOT/'research/phase7/frozen'))
    p.add_argument('--robustness-pkl', help='robustness model to validate instead of the Phase VII one (e.g. the extension\'s X)')
    p.add_argument('--modes', nargs='+', default=list(MODES))
    a = p.parse_args()
    pf = pickle.loads((Path(a.frozen)/'p_feasible.pkl').read_bytes())
    ph = pickle.loads(Path(a.robustness_pkl or Path(a.frozen)/'p_robust_given_feasible.pkl').read_bytes())
    modes = {m: MODES[m] for m in a.modes}
    devj = json.loads(Path(a.dev).read_text())
    library = [s.strip() for s in Path(a.library).read_text().splitlines() if s.strip()]
    dev_store = json.loads(Path(a.dev_store).read_text())
    conf = Path(a.egad_conf)
    egad_store = json.loads((conf/'store.json').read_text())
    egad_rows = json.loads((conf/'orders.json').read_text())['rows']
    objects = [('development', n, Path(a.dev_catalogues)/f'{n}.npz', Path(a.dev_projects)/dev_store[n]['project'], dev_store[n])
               for n in sorted(set(library) | set(devj['parts']) | set(devj['nonviable']))]
    objects += [('egad', n, conf/'cat80'/f'{n}.npz', Path(a.egad_pool)/egad_store[n]['project'], egad_store[n])
                for n in sorted(egad_rows) if (conf/'cat80'/f'{n}.npz').exists()]

    def level1(obj):
        kind, n, catp, proj, st = obj
        cat = catalogue.load(catp)
        out = run([a.binary, 'research-candidates', '--learned', a.models, str(proj), n])
        L = out['learned']
        labels = [r['candidate']['label'] for r in out['records']]
        assert labels == [str(x) for x in cat['label']], n
        Fp = bl.features(cat, st['low'], st['high'])
        Fn = np.asarray(L['features'], float)
        pp, hp = pf.predict_proba(Fp)[:, 1], ph.predict_proba(Fp)[:, 1]
        pn, hn = np.asarray(L['feasibility']), np.asarray(L['robustness'])
        pn_py = pf.predict_proba(Fn)[:, 1]  # sklearn on the native features: isolates inference from features
        b = np.asarray(L['bounds'])
        return n, dict(kind=kind, candidates=len(labels), feature_max_abs=float(np.abs(Fn-Fp).max()),
                       bounds_max_abs=float(np.abs(b - np.array([st['low'], st['high']])).max()),
                       prob_max_abs=float(max(np.abs(pn-pp).max(), np.abs(hn-hp).max())),
                       prob_mean_abs=float(np.abs(pn-pp).mean()), inference_vs_sklearn_same_features=float(np.abs(pn-pn_py).max()),
                       model_load_s=L['modelLoadSeconds'], inference_s=L['inferenceSeconds'])

    def level2(obj):
        kind, n, catp, proj, st = obj
        cat = catalogue.load(catp)
        if not cat['feasible'].any():
            return n, None
        F = bl.features(cat, st['low'], st['high'])
        pt = bl.scores(pf, F); h = ph.predict_proba(F)[:, 1]
        po = py_orders(cat, pt, h)
        res = {}
        for mode, key in modes.items():
            rep = run([a.binary, 'plan', '--json', '--nr-grasps', '1', '--learned', a.models, '--order', mode, str(proj), n])
            native = (rep['candidatesExamined'], rep['plans'][0]['candidate']['label'] if rep['plans'] else None)
            ref = first(cat, po[key])
            if kind == 'egad' and not a.robustness_pkl:
                r = egad_rows[n][key]
                ref_conf = (r['nf'], r['first'])
            else:
                ref_conf = ref
            res[mode] = dict(native=list(native), python=list(ref), confirmation_reference=list(ref_conf),
                             match=native == ref == ref_conf, learned_seconds=rep['timings']['learnedSeconds'])
        return n, dict(kind=kind, modes=res)

    l1 = {}
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for n, r in pool.map(level1, objects):
            l1[n] = r
    print('level 1 done', len(l1), flush=True)
    l2 = {}
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for n, r in pool.map(level2, [o for o in objects if o[0] == 'egad' or o[1] in set(devj['parts'])]):
            if r:
                l2[n] = r
    print('level 2 done', len(l2), flush=True)
    by = lambda kind, key: [v[key] for v in l1.values() if v['kind'] == kind]
    summary = {}
    for kind in ('development', 'egad'):
        summary[kind] = dict(objects=len(by(kind, 'candidates')), candidates=int(sum(by(kind, 'candidates'))),
                             feature_max_abs=max(by(kind, 'feature_max_abs')), bounds_max_abs=max(by(kind, 'bounds_max_abs')),
                             prob_max_abs=max(by(kind, 'prob_max_abs')), prob_mean_abs=float(np.mean(by(kind, 'prob_mean_abs'))),
                             inference_vs_sklearn_same_features_max=max(by(kind, 'inference_vs_sklearn_same_features')),
                             within_feature_tolerance=bool(max(by(kind, 'feature_max_abs')) <= FEATURE_TOL),
                             within_probability_tolerance=bool(max(by(kind, 'prob_max_abs')) <= PROB_TOL))
        objs = [v for v in l2.values() if v['kind'] == kind]
        summary[kind]['decisions'] = {mode: dict(objects=len(objs), matches=sum(v['modes'][mode]['match'] for v in objs),
                                                 mismatches=[n for n, v in l2.items() if v['kind'] == kind and not v['modes'][mode]['match']])
                                      for mode in modes}
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), tolerances=dict(feature_abs=FEATURE_TOL, probability_abs=PROB_TOL,
               decisions='identical'), models=a.models, robustness_pkl=a.robustness_pkl, modes=list(modes),
               reference=('Python replay with the same pickles (no confirmation reference exists for a new model)' if a.robustness_pkl
                          else 'Python replay and the Phase VII confirmation orders.json'),
               summary=summary, level1=l1, level2=l2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
