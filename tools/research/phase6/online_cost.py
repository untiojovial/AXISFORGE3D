"""Online cost of the native planner with and without priors (stage G, partial).

    python tools/research/phase6/online_cost.py --binary target/release/axisforge --dev research/phase6/dev_parts.json \
        --catalogues cat80/ --store store.json --projects POOL --priors priors.npz \
        --fusion research/phase6/results/fusion_benchmark.json --out cost_profile_online.json [--parts 60]

One process at a time, one thread (RAYON_NUM_THREADS=1), nothing else running.
For each part and ordering, `axisforge plan --json --nr-grasps 1` runs once
after a warm-up of the binary ("first") and three more times ("repeat"); the
wall time includes process start, loading the cell and generating every
candidate, as an online call would. The examined count must equal the
simulator's. Orderings: default (M0), learned (M1) and the fusion family that
the development decision favours, with the configuration most often chosen
across its outer folds.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import catalogue  # noqa: E402
from research_grasping.phase6 import fusion  # noqa: E402


def pct(x):
    x = np.asarray(x, float)
    return {f'p{q}': round(float(np.percentile(x, q)), 4) for q in (50, 90, 95, 99)} | dict(mean=round(float(x.mean()), 4))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('binary', 'dev', 'catalogues', 'store', 'projects', 'priors', 'fusion', 'out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--parts', type=int, default=0, help='first N development parts by name (0 = all)')
    p.add_argument('--family')
    a = p.parse_args()
    env = dict(os.environ, RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1')
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    if a.parts:
        dev = dev[:a.parts]
    store = json.loads(Path(a.store).read_text())
    pri = np.load(a.priors)
    fb = json.loads(Path(a.fusion).read_text())
    fam = a.family
    if fam is None:
        ok = [f for f, d in fb['decision_rule']['families'].items() if d['meets_rule']]
        fam = ok[0] if ok else min(fb['summary'], key=lambda m: fb['summary'][m]['N_FR_gmean'] if m.startswith('M') and m not in ('M0', 'M1', 'M2') else 1e9)
    cfg = Counter(json.dumps(c['config'], sort_keys=True) for c in fb['chosen'][fam]).most_common(1)[0][0]
    cfg = json.loads(cfg)
    tmp = Path(tempfile.mkdtemp())

    def run(args):
        start = time.perf_counter()
        out = subprocess.run(args, check=True, capture_output=True, text=True, env=env).stdout
        return time.perf_counter() - start, json.loads(out)
    first_part = dev[0]
    run([a.binary, 'plan', '--json', '--nr-grasps', '1', str(Path(a.projects)/store[first_part]['project']), first_part])
    rows = []
    for n in dev:
        cat = catalogue.load(Path(a.catalogues)/f'{n}.npz')
        pl = pri[f'{n}|learned'].astype(float); pt = pl/pl.max()
        s = fusion.config_score(fam, cfg, cat, pt)
        orders = {'M0': (None, None, catalogue.order(cat)),
                  'M1': (pt, 2.0, catalogue.order(cat, pt, 2.)),
                  fam: (fusion.planner_prior(cat, s), fusion.WEIGHT, fusion.order(cat, s))}
        for m, (prior, weight, order) in orders.items():
            hits = np.flatnonzero(cat['feasible'][order])
            simulated = int(hits[0])+1
            args = [a.binary, 'plan', '--json', '--nr-grasps', '1']
            if prior is not None:
                path = tmp/f'{n}-{m}.json'
                path.write_text(json.dumps(dict(weight=weight, candidates=[
                    dict(label=str(lab), score=float(v)) for lab, v in zip(cat['label'], prior)])))
                args += ['--prior', str(path)]
            args += [str(Path(a.projects)/store[n]['project']), n]
            t0, rep = run(args)
            reps = [run(args)[0] for _ in range(3)]
            rows.append(dict(part=n, method=m, first_s=round(t0, 4), repeat_s=[round(x, 4) for x in reps],
                             examined=rep['candidatesExamined'], simulated=simulated,
                             match=rep['candidatesExamined'] == simulated))
    out = dict(note='Wall time of one `axisforge plan --nr-grasps 1` call per part and ordering, one thread, '
                    'including process start, cell loading and candidate generation. Development parts.',
               family=fam, config=cfg, parts=len(dev), mismatches=sum(not r['match'] for r in rows), methods={})
    for m in {r['method'] for r in rows}:
        rr = [r for r in rows if r['method'] == m]
        out['methods'][m] = dict(first_s=pct([r['first_s'] for r in rr]),
                                 repeat_median_s=pct([np.median(r['repeat_s']) for r in rr]),
                                 examined=pct([r['examined'] for r in rr]))
    base = {r['part']: np.median(r['repeat_s']) for r in rows if r['method'] == 'M0'}
    for m in out['methods']:
        if m == 'M0':
            continue
        d = np.log([np.median(r['repeat_s'])/base[r['part']] for r in rows if r['method'] == m])
        out['methods'][m]['time_ratio_vs_M0'] = dict(ratio=round(float(np.exp(d.mean())), 4),
                                                    better=int((d < 0).sum()), worse=int((d > 0).sum()))
    # seconds per examined candidate beyond the fixed cost (slope of time on examined, all runs)
    x = np.array([r['examined'] for r in rows], float); y = np.array([np.median(r['repeat_s']) for r in rows])
    slope, intercept = np.polyfit(x, y, 1)
    out['fixed_s'] = round(float(intercept), 4)
    out['seconds_per_examined_candidate'] = round(float(slope), 6)
    out['rows'] = rows
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'})[:1500])


if __name__ == '__main__':
    main()
