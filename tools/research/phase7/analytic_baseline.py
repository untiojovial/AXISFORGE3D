"""Classical analytic ranking baseline: force closure, then wrench-space epsilon (ranking-controlled level).

    python tools/research/phase7/analytic_baseline.py collect --binary target/release/axisforge --store store.json \
        --projects POOL --names names.txt --out-dir epsilon/ --jobs 4
    python tools/research/phase7/analytic_baseline.py evaluate --epsilon-dir epsilon/ --catalogues cat80/ \
        --dev research/phase6/dev_parts.json --physics research/phase7/results/physics_dev_with_joint.json.gz \
        --negative-transfer research/phase7/results/negative_transfer_analysis.json --store store.json --projects POOL \
        --collision COLLISION_DIR --out epsilon_baseline.json

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
    """v3 outcomes (nine conditions, torsion 0.005) of the epsilon order's first viable grasp, against M0, M1, M4
    and FORTIS-R Joint on the same development parts (their first grasps are in physics_dev_with_joint)."""
    import argparse as ap
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, str(ROOT/'tools/research/phase7'))
    import physics_dev as pdev
    ph = json.load(gzip.open(a.physics, 'rt'))
    parts = sorted(n for n in json.loads(Path(a.dev).read_text())['parts'] if (Path(a.epsilon_dir)/f'{n}.json').exists())
    first, nf, secs = {}, {}, []
    for n in parts:
        cat = catalogue.load(Path(a.catalogues)/f'{n}.npz')
        e = json.loads((Path(a.epsilon_dir)/f'{n}.json').read_text())
        secs.append(e['seconds'])
        eps = np.array([e['rows'][str(lab)][0] for lab in cat['label']], float)
        fc = np.array([e['rows'][str(lab)][1] for lab in cat['label']], float)
        s = fc + np.clip(eps, 0, None)/max(float(np.clip(eps, 0, None).max()), 1e-12)
        o = fusion.order(cat, s)
        pos = int(np.flatnonzero(cat['feasible'][o])[0])
        first[n], nf[n] = str(cat['label'][o[pos]]), pos + 1
    tasks = [(n, [(first[n], [.005])]) for n in parts if '0.005' not in ph['trials'][n].get(first[n], {})]
    args = ap.Namespace(store=a.store, projects=a.projects, collision=a.collision, catalogues=a.catalogues)
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=pdev.init, initargs=(args,)) as pool:
        for part, out, _ in pool.map(pdev.job, tasks):
            for lab, tr in out.items():
                ph['trials'][part].setdefault(lab, {}).update(tr)
    std, loads = ph['standard'], ph['loads']

    def held(n, lab, which):
        t = ph['trials'][n][lab]['0.005']
        return int(all(t[c]['held_v3'] for c in (std if which == 'robust' else loads)))
    nt = json.loads(Path(a.negative_transfer).read_text())['per_part_N_F']
    rows = {'epsilon (force closure, then epsilon)': {n: dict(nf=nf[n], label=first[n]) for n in parts}}
    for m in ('M0', 'M1', 'M4', 'Joint'):
        rows[m] = {n: dict(nf=nt[n][m], label=ph['core'][n][m]) for n in parts}
    out = dict(note='Classical analytic ranking (force closure first, then the largest wrench-space ball of the pad contacts, '
                    'computed by the planner\'s own measure) used as an examination order over the same candidates, stroke gate '
                    'first: the ranking-controlled level. A reimplementation of a classical criterion, not a reproduction of a '
                    'published system. v3 physics, torsion 0.005, nine conditions. Development parts (seeded subset, seed 20261208).',
               parts=len(parts), simulated_now=len(tasks),
               epsilon_seconds_per_part=dict(median=round(float(np.median(secs)), 1), mean=round(float(np.mean(secs)), 1)),
               methods={}, contrasts={})
    for m, r in rows.items():
        out['methods'][m] = dict(N_F_gmean=round(float(np.exp(np.mean(np.log([r[n]['nf'] for n in parts])))), 3),
                                 robust_v3=round(float(np.mean([held(n, r[n]['label'], 'robust') for n in parts])), 4),
                                 load_tolerant_v3=round(float(np.mean([held(n, r[n]['label'], 'load') for n in parts])), 4),
                                 success_at_1=round(float(np.mean([r[n]['nf'] == 1 for n in parts])), 4))
    E = rows['epsilon (force closure, then epsilon)']
    for b in ('M0', 'M1', 'M4', 'Joint'):
        B = rows[b]
        out['contrasts'][f'epsilon vs {b}'] = dict(
            N_F=fb6.ratio([E[n]['nf'] for n in parts], [B[n]['nf'] for n in parts]),
            robust=mech.bootstrap_diff([held(n, E[n]['label'], 'robust') for n in parts], [held(n, B[n]['label'], 'robust') for n in parts]),
            load_tolerant=mech.bootstrap_diff([held(n, E[n]['label'], 'load') for n in parts], [held(n, B[n]['label'], 'load') for n in parts]))
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
    for k in ('epsilon-dir', 'catalogues', 'dev', 'physics', 'negative-transfer', 'store', 'projects', 'collision', 'out'):
        e.add_argument(f'--{k}', required=True)
    e.add_argument('--jobs', type=int, default=1)
    a = p.parse_args()
    collect(a) if a.cmd == 'collect' else evaluate(a)


if __name__ == '__main__':
    main()
