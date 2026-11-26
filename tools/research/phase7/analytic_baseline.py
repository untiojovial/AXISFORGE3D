"""Classical analytic ranking baseline: force closure, then wrench-space epsilon (ranking-controlled level).

    python tools/research/phase7/analytic_baseline.py collect --binary target/release/axisforge --store store.json \
        --projects POOL --names names.txt --out-dir epsilon/ --jobs 4
    python tools/research/phase7/analytic_baseline.py evaluate --epsilon-dir epsilon/ --catalogues cat80/ \
        --dev research/phase6/dev_parts.json --sweep sweep.json.gz --phase6 research/phase6/results/fusion_benchmark.json \
        --out external_baselines.json

This is not a reproduction of a published system: it is the classical analytic criterion (force closure and
the largest wrench-space ball, computed by the planner's own wrench-space measure for the pad contacts)
used as a ranking over the same candidates, with the stroke gate first, so that only the ranking differs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import catalogue  # noqa: E402
from research_grasping.phase6 import fusion, mechanism as mech  # noqa: E402

spec = importlib.util.spec_from_file_location('fb6', ROOT/'tools/research/phase6/fusion_bench.py')
fb6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(fb6)


def collect(a):
    store = json.loads(Path(a.store).read_text())
    names = [s.strip() for s in Path(a.names).read_text().splitlines() if s.strip()]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    def one(n):
        path = out/f'{n}.json'
        if path.exists():
            return n, 'cached'
        start = time.perf_counter()
        rec = json.loads(subprocess.run([a.binary, 'research-candidates', '--static-quality',
                                         str(Path(a.projects)/store[n]['project']), n],
                                        check=True, capture_output=True, text=True, timeout=3600).stdout)
        rows = {r['candidate']['label']: [r['quality']['quality'], int(r['quality']['forceClosure']), r['qualityElapsedS']]
                for r in rec}
        path.write_text(json.dumps(dict(seconds=round(time.perf_counter()-start, 2), rows=rows)))
        return n, round(time.perf_counter()-start, 1)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for n, s in pool.map(one, names):
            print(n, s, flush=True)


def evaluate(a):
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    sw = json.load(gzip.open(a.sweep, 'rt'))
    std = sw['standard']
    p6 = json.loads(Path(a.phase6).read_text())['per_part']
    rows, missing, secs = {}, {}, []
    parts = [n for n in dev if (Path(a.epsilon_dir)/f'{n}.json').exists()]
    for n in parts:
        cat = catalogue.load(Path(a.catalogues)/f'{n}.npz')
        e = json.loads((Path(a.epsilon_dir)/f'{n}.json').read_text())
        secs.append(e['seconds'])
        eps = np.array([e['rows'][str(lab)][0] for lab in cat['label']], float)
        fc = np.array([e['rows'][str(lab)][1] for lab in cat['label']], float)
        s = fc + np.clip(eps, 0, None)/max(float(np.clip(eps, 0, None).max()), 1e-12)
        o = fusion.order(cat, s)
        first = str(cat['label'][o[np.flatnonzero(cat['feasible'][o])[0]]])
        if 'load 2 kg' not in sw['parts'][n][first]:
            missing.setdefault(n, []).append(first)
            continue
        rows[n] = fb6.outcome(cat, o, sw['parts'][n], std)
    if missing:
        Path(a.out).with_name('epsilon_missing_first_grasps.json').write_text(json.dumps(missing, indent=1))
        raise SystemExit(f'{sum(map(len, missing.values()))} first grasps lack load conditions')
    out = dict(note='Development parts with the epsilon export; ranking-controlled comparison (same candidates, gate '
                    'first). Classical analytic criterion reimplemented with the planner\'s own wrench-space measure; '
                    'not a reproduction of a published system. v2 labels.',
               parts=len(parts), epsilon_seconds_per_part=dict(median=float(np.median(secs)), mean=float(np.mean(secs))),
               methods={}, contrasts={})
    ms = {'epsilon (force closure, then epsilon)': rows, **{k: {n: p6[k][n] for n in parts} for k in ('M0', 'M1', 'M4')}}
    for m, r in ms.items():
        nf = np.array([r[n]['nf'] for n in parts], float)
        out['methods'][m] = dict(N_F_gmean=round(float(np.exp(np.log(nf).mean())), 3),
                                 robust_rate=round(float(np.mean([r[n]['robust'] for n in parts])), 4),
                                 load_tolerant_rate=round(float(np.mean([r[n]['load'] for n in parts])), 4),
                                 success_at_1=round(float(np.mean(nf == 1)), 4))
    for b in ('M0', 'M1', 'M4'):
        A, B = rows, ms[b]
        out['contrasts'][f'epsilon vs {b}'] = dict(
            N_F=fb6.ratio([A[n]['nf'] for n in parts], [B[n]['nf'] for n in parts]),
            robust=mech.bootstrap_diff([A[n]['robust'] for n in parts], [B[n]['robust'] for n in parts]),
            load_tolerant=mech.bootstrap_diff([A[n]['load'] for n in parts], [B[n]['load'] for n in parts]))
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['methods'], indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    for k in ('binary', 'store', 'projects', 'names', 'out-dir'):
        c.add_argument(f'--{k}', required=True)
    c.add_argument('--jobs', type=int, default=4)
    e = sub.add_parser('evaluate')
    for k in ('epsilon-dir', 'catalogues', 'dev', 'sweep', 'phase6', 'out'):
        e.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    collect(a) if a.cmd == 'collect' else evaluate(a)


if __name__ == '__main__':
    main()
