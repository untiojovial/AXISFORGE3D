"""Run the new confirmation of PHASE7X_PROTOCOL.md (EGAD positions 300-499), stage by stage, each stage once.

    python tools/research/phase7_extension/confirm_x.py prepare  --pool NEW_POOL --binary BIN --out-dir OUT --jobs 3
    python tools/research/phase7_extension/confirm_x.py orders   --pool NEW_POOL --binary BIN --out-dir OUT --freeze FREEZE.json --jobs 3
    python tools/research/phase7_extension/confirm_x.py physics  --pool NEW_POOL --out-dir OUT --jobs 3
    python tools/research/phase7_extension/confirm_x.py timing   --pool NEW_POOL --binary BIN --out-dir OUT --freeze FREEZE.json --cpu 3
    python tools/research/phase7_extension/confirm_x.py analyse  --out-dir OUT --freeze FREEZE.json --out research/phase7_extension/results/confirmation_x.json

Stages:

- **prepare.** The frozen Phase VII catalogue procedure (egad_confirm.prepare): one exhaustively labelled
  80 mm catalogue per object, plus store.json and status.json.
- **orders.** Decisions of the native planner, `plan --json --nr-grasps 1 [--learned F --order MODE]`, for
  every method of the freeze. That gives N_F, the first grasp and Success@k. The Python replay (catalogue order,
  fusion orders) is computed too, as a check; mismatches are recorded. GPD+gate comes from the adapter scores
  in OUT/gpd. One JSON per object (restart-safe).
- **physics.** v3 trials of every method's first grasp: the nine conditions at torsion 0.005 m for all methods,
  and also at 0.001 and 0.02 m for the methods named in the freeze's torsion grid. One JSON per object.
- **timing.** Full cold-process cost per object and mode, 3 repetitions, one thread pinned to one CPU, seeded
  random order (integrated_benchmark.py logic).
- **analyse.** The protocol's tests, read from the freeze record, which is hash-checked before use.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools/research/phase7')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from research_grasping.phase5 import catalogue, dynamics as v1  # noqa: E402
from research_grasping.phase6 import physics as v2  # noqa: E402
from research_grasping.phase7 import physics as v3  # noqa: E402

CONDITIONS = (('nominal', {}), ('mass x2', dict(mass=.2)), ('mass x0.5', dict(mass=.05)), ('friction 0.3', dict(friction=.3)),
              ('pose noise 1', dict(noise_seed=1)), ('pose noise 2', dict(noise_seed=2)),
              ('load 0.5 kg', dict(mass=.5)), ('load 1 kg', dict(mass=1.)), ('load 2 kg', dict(mass=2.)))
BUDGETS = (1, 5, 10, 20, 50)
STA = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_freeze(path):
    fr = json.loads(Path(path).read_text())
    for f, h in fr['files_sha256'].items():
        if sha(ROOT/f) != h:
            raise SystemExit(f'frozen file changed: {f}')
    return fr


# ------------------------------------------------------------------------------------------ prepare
def prepare(a):
    import egad_confirm as ec
    ec.prepare(argparse.Namespace(pool=a.pool, binary=a.binary, out_dir=a.out_dir, jobs=a.jobs))


# ------------------------------------------------------------------------------------------- orders
def plan_decision(binary, proj, name, models=None, mode=None):
    args = [binary, 'plan', '--json', '--nr-grasps', '1', str(proj), name]
    if models:
        args[3:3] = ['--learned', str(ROOT/models), '--order', mode]
    rep = json.loads(subprocess.run(args, check=True, capture_output=True, text=True, timeout=3600,
                                    env={'RAYON_NUM_THREADS': '1', 'PATH': '/usr/bin:/bin'}).stdout)
    return rep['candidatesExamined'], (rep['plans'][0]['candidate']['label'] if rep['plans'] else None)


def orders_one(task):
    n, a_ = task
    out = Path(a_['out_dir'])/'orders'/f'{n}.json'
    if out.exists():
        return n, 'cached'
    fr = STA['freeze']
    store = json.loads((Path(a_['out_dir'])/'store.json').read_text())
    proj = Path(a_['pool'])/store[n]['project']
    cat = catalogue.load(Path(a_['out_dir'])/'cat80'/f'{n}.npz')
    viable = bool(cat['feasible'].any())
    row = dict(viable=viable, candidates=int(len(cat['label'])), methods={})
    for name, m in fr['methods'].items():
        if m['kind'] == 'native':
            t0 = time.perf_counter()
            nf, first = plan_decision(a_['binary'], proj, n, m.get('models'), m.get('mode'))
            row['methods'][name] = dict(nf=nf if first else None, first=first, examined=nf, seconds=time.perf_counter()-t0)
    # Python replay check: M0 = catalogue order; learned methods with the same pickles (scikit-learn)
    o = catalogue.order(cat)
    ref = C.first(cat, o)
    row['replay_M0'] = list(ref)
    row['replay_M0_match'] = (row['methods']['M0']['nf'], row['methods']['M0']['first']) == ref if viable else row['methods']['M0']['first'] is None
    if viable:
        import pickle
        from research_grasping.phase5 import baselines as bl
        from research_grasping.phase6 import fusion
        F = bl.features(cat, store[n]['low'], store[n]['high'])
        pf = pickle.loads((ROOT/fr['replay']['feasibility']).read_bytes())
        pt = bl.scores(pf, F)
        q = np.maximum(fusion.quality(cat, 'cen'), 1e-3)**.3
        replay = {}
        for name, m in fr['methods'].items():
            if m.get('mode') in ('joint', 'hybrid'):
                h = pickle.loads((ROOT/fr['replay']['robustness'][m['models']]).read_bytes()).predict_proba(F)[:, 1]
                jo = fusion.order(cat, pt*h*q)
                replay[name] = C.first(cat, jo if m['mode'] == 'joint' else C.interleave(cat, jo, o))
            elif m.get('mode') == 'm1':
                replay[name] = C.first(cat, catalogue.order(cat, pt, 2.))
        row['replay_mismatches'] = sorted(k for k, v in replay.items()
                                          if (row['methods'][k]['nf'], row['methods'][k]['first']) != v)
    # GPD+gate (secondary)
    g = Path(a_['out_dir'])/'gpd'/f'{n}.json'
    if g.exists() and 'scores' in json.loads(g.read_text()):
        from research_grasping.phase6 import fusion
        s = np.asarray(json.loads(g.read_text())['scores'], float)
        go = fusion.order(cat, (s - s.min())/(s.max() - s.min()) if s.max() > s.min() else np.zeros_like(s))
        nf, first = C.first(cat, go)
        hy = C.first(cat, C.interleave(cat, go, o))
        row['methods']['GPD+gate'] = dict(nf=nf, first=first)
        row['methods']['hybrid GPD/M0'] = dict(nf=hy[0], first=hy[1])
    out.write_text(json.dumps(row))
    return n, 'ok'


def orders(a):
    fr = load_freeze(a.freeze)
    status = json.loads((Path(a.out_dir)/'status.json').read_text())
    names = sorted(n for n, s in status.items() if s['status'] in ('viable', 'no viable candidate'))
    (Path(a.out_dir)/'orders').mkdir(exist_ok=True)
    STA['freeze'] = fr
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=STA.update, initargs=(dict(freeze=fr),)) as pool:
        for n, st in pool.map(orders_one, [(n, dict(out_dir=a.out_dir, pool=a.pool, binary=a.binary)) for n in names]):
            print(n, st, flush=True)


# ------------------------------------------------------------------------------------------ physics
def physics_one(task):
    n, a_, jobs = task
    out = Path(a_['out_dir'])/'trials'/f'{n}.json'
    if out.exists():
        return n, 0.
    store = json.loads((Path(a_['out_dir'])/'store.json').read_text())
    R, t, g, tt = v1.part_pose(Path(a_['pool'])/store[n]['project'], n)
    pieces = [v1.read_obj(p) for p in sorted((Path(a_['pool'])/n/'collision').glob('model_collision_*.obj'),
                                              key=lambda p: int(p.stem.split('_')[-1]))]
    cat = catalogue.load(Path(a_['out_dir'])/'cat80'/f'{n}.npz')
    com, _ = v2.planner_com(cat)
    cands = {c['label']: c for c in catalogue.candidates(cat)}
    t0 = time.perf_counter()
    res = {label: {tor: {c: v3.trial(pieces, R, t, tt, cands[label], g, com=com, torsion=float(tor), **kw) for c, kw in CONDITIONS}
                   for tor in tors} for label, tors in jobs}
    out.write_text(json.dumps(dict(seconds=round(time.perf_counter()-t0, 1), trials=res)))
    return n, time.perf_counter()-t0


def physics(a):
    fr = load_freeze(a.freeze)
    (Path(a.out_dir)/'trials').mkdir(exist_ok=True)
    tasks = []
    for f in sorted((Path(a.out_dir)/'orders').glob('*.json')):
        row = json.loads(f.read_text())
        if not row['viable']:
            continue
        want = {}
        for m, r in row['methods'].items():
            if r.get('first'):
                want.setdefault(r['first'], set()).add('0.005')
                if m in fr['torsion_grid_methods']:
                    want[r['first']].update(('0.001', '0.02'))
        tasks.append((f.stem, dict(out_dir=a.out_dir, pool=a.pool), sorted((lab, sorted(ts)) for lab, ts in want.items())))
    tasks.sort(key=lambda t: -sum(len(ts) for _, ts in t[2]))
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for n, sec in pool.map(physics_one, tasks):
            print(n, round(sec), flush=True)


# ------------------------------------------------------------------------------------------- timing
def timing(a):
    import integrated_benchmark as ib
    fr = load_freeze(a.freeze)
    store = json.loads((Path(a.out_dir)/'store.json').read_text())
    names = sorted(f.stem for f in (Path(a.out_dir)/'orders').glob('*.json'))
    raw_dir = Path(a.out_dir)/'timing'; raw_dir.mkdir(exist_ok=True)
    modes = {name: m for name, m in fr['methods'].items() if m['kind'] == 'native' and name in fr['timed_methods']}
    order_rng = random.Random(fr['timing_seed']); order_rng.shuffle(names)
    for i, n in enumerate(names):
        path = raw_dir/f'{n}.json'
        if path.exists():
            continue
        rnd = random.Random(f"{fr['timing_seed']}-{n}")
        proj = Path(a.pool)/store[n]['project']
        rec = dict(load_before=__import__('os').getloadavg()[0], cold={k: [] for k in modes})
        for _ in range(fr['timing_reps']):
            ms = list(modes); rnd.shuffle(ms)
            for name in ms:
                m = modes[name]
                args = [a.binary, 'plan', '--json', '--nr-grasps', '1', str(proj), n]
                if m.get('models'):
                    args[3:3] = ['--learned', str(ROOT/m['models']), '--order', m['mode']]
                out, wall, cpu, _ = ib.spawn(args, a.cpu)
                rec['cold'][name].append(dict(wall=wall, cpu=cpu, peak_kib=out.get('peakResidentKiB'), examined=out['candidatesExamined'],
                                              first=out['plans'][0]['candidate']['label'] if out['plans'] else None,
                                              timings=out['timings'], project_load=out.get('projectLoadSeconds'),
                                              model_load=out.get('modelLoadSeconds', 0.)))
        rec['load_after'] = __import__('os').getloadavg()[0]
        path.write_text(json.dumps(rec))
        print(i+1, len(names), n, flush=True)


# ------------------------------------------------------------------------------------------ analyse
STD = ('nominal', 'mass x2', 'mass x0.5', 'friction 0.3', 'pose noise 1', 'pose noise 2')
LOADS = ('load 0.5 kg', 'load 1 kg', 'load 2 kg')


def t_logratio(d, margin=1.0):
    """One-sided t test of mean log ratio < log(margin); estimand: geometric-mean ratio; 95 % t CI."""
    from scipy.stats import t as tdist
    d = np.asarray(d, float); n = len(d); se = d.std(ddof=1)/np.sqrt(n)
    stat = (d.mean() - np.log(margin))/se if se > 0 else -np.inf
    q = tdist.ppf(.975, n-1)
    return dict(n=n, gmean_ratio=float(np.exp(d.mean())), ci95=[float(np.exp(d.mean()-q*se)), float(np.exp(d.mean()+q*se))],
                upper_97_5=float(np.exp(d.mean()+q*se)), t=float(stat), p_one_sided=float(tdist.cdf(stat, n-1)) if se > 0 else 0.,
                fewer=int((d < 0).sum()), more=int((d > 0).sum()), ties=int((d == 0).sum()), margin=margin)


def wilcoxon_sens(d):
    from scipy.stats import wilcoxon
    d = np.asarray(d, float); nz = d[d != 0]
    return dict(p_one_sided=float(wilcoxon(nz, alternative='less').pvalue) if len(nz) else 1., zero_differences_dropped=int((d == 0).sum()))


def paired_binary(x, y, margin):
    """Tango score test of p_x - p_y >= margin (one-sided), Tango and Newcombe 95 % CIs, the counts."""
    from scipy.stats import binom, norm
    from p2_audit import newcombe10, tango_interval, tango_z
    x, y = np.asarray(x, int), np.asarray(y, int)
    a = int((x & y).sum()); b = int((x & (1-y)).sum()); c = int(((1-x) & y).sum()); d = int(((1-x) & (1-y)).sum()); n = len(x)
    z = tango_z(b, c, n, margin)
    lo, hi = tango_interval(b, c, n)
    nlo, nhi = newcombe10(a, b, c, d)
    return dict(n=n, both=a, x_only=b, y_only=c, neither=d, difference=(b-c)/n, margin=margin, tango_z=float(z),
                p_one_sided=float(norm.sf(z)), tango_ci95=[float(lo), float(hi)], newcombe_ci95=[float(nlo), float(nhi)],
                mcnemar_superiority_p_one_sided=float(binom.sf(b-1, b+c, .5)) if b+c else 1.,
                mcnemar_two_sided_p=float(min(1., 2*min(binom.cdf(b, b+c, .5), binom.sf(b-1, b+c, .5)))) if b+c else 1.)


def holm(ps):
    keys = list(ps); order = sorted(keys, key=lambda k: ps[k]); out = {}; run = 0.
    for r, k in enumerate(order):
        run = max(run, min(1., (len(keys)-r)*ps[k])); out[k] = run
    return out


def cluster_ci(values, clusters, stat, reps=4000, seed=20261216):
    """Cluster bootstrap 95 % CI (clusters resampled with replacement): sensitivity to dependence between
    neighbouring EGAD cells (cluster = difficulty letter of the cell)."""
    rng = np.random.default_rng(seed)
    keys = sorted(set(clusters)); idx = {k: [i for i, c in enumerate(clusters) if c == k] for k in keys}
    vals = []
    for _ in range(reps):
        pick = rng.choice(len(keys), len(keys))
        rows = [i for j in pick for i in idx[keys[j]]]
        vals.append(stat(rows))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def analyse(a):
    fr = load_freeze(a.freeze)
    out_dir = Path(a.out_dir)
    status = json.loads((out_dir/'status.json').read_text())
    built = json.loads((Path(a.pool)/'built.json').read_text())['rows']
    orders = {f.stem: json.loads(f.read_text()) for f in (out_dir/'orders').glob('*.json')}
    trials = {f.stem: json.loads(f.read_text())['trials'] for f in (out_dir/'trials').glob('*.json')}
    timing = {f.stem: json.loads(f.read_text()) for f in (out_dir/'timing').glob('*.json')}
    X, M0, M1, JH = fr['names']['X'], 'M0', 'M1', 'Joint-hist'
    viable = sorted(n for n, r in orders.items() if r['viable'])
    counts = dict(objects=len(built), prepared=sum(1 for r in built if r.get('kept')), preparation_failures=sum(1 for r in built if not r.get('kept')),
                  catalogued=sum(1 for s in status.values() if s['status'] in ('viable', 'no viable candidate')),
                  catalogue_failures={k: sum(1 for s in status.values() if s['status'] == k) for k in
                                      ('catalogue error', 'planner budget exhausted', 'no candidate generated', 'cell build refused')},
                  viable=len(viable), no_viable_candidate=sum(1 for r in orders.values() if not r['viable']),
                  planning_errors=sum(1 for r in orders.values() for m in r['methods'].values() if m.get('error')),
                  replay_mismatches_M0=sorted(n for n, r in orders.items() if not r.get('replay_M0_match', True)),
                  replay_mismatches_learned=sorted((n, k) for n, r in orders.items() for k in r.get('replay_mismatches', [])))

    def held(n, m, conds, tor='0.005'):
        lab = orders[n]['methods'][m]['first']
        t = trials[n][lab][tor]
        return int(all(t[c]['held_v3'] for c in conds))

    def all3(n, m):
        return int(all(held(n, m, STD, tor) for tor in ('0.001', '0.005', '0.02')))
    unstable = sum(1 for n in trials for lab in trials[n] for tor in trials[n][lab] for c in trials[n][lab][tor].values() if c.get('unstable'))
    counts['simulation_unstable_trials'] = unstable
    counts['simulation_trials'] = sum(len(trials[n][lab][tor]) for n in trials for lab in trials[n] for tor in trials[n][lab])
    nf = lambda m: np.array([orders[n]['methods'][m]['nf'] for n in viable], float)  # noqa: E731
    d1 = np.log(nf(X)/nf(M0)); d3 = np.log(nf(X)/nf(M1))
    timed = sorted(timing)
    T = lambda m: np.array([np.median([c['wall'] for c in timing[n]['cold'][m]]) for n in timed])  # noqa: E731
    d4 = np.log(T(X)/T(M0))
    primary = {
        'C1': dict(hypothesis='X examines fewer candidates than M0 (geometric-mean N_F ratio < 1)', **t_logratio(d1), wilcoxon=wilcoxon_sens(d1)),
        'C2': dict(hypothesis='X v3 robustness non-inferior to M0, margin -5 pp (Tango)',
                   **paired_binary([held(n, X, STD) for n in viable], [held(n, M0, STD) for n in viable], -.05)),
        'C3': dict(hypothesis='X N_F non-inferior to M1, ratio margin 1.25', **t_logratio(d3, 1.25)),
        'C4': dict(hypothesis='full cold-process time of X lower than M0 (geometric-mean ratio < 1)', **t_logratio(d4), wilcoxon=wilcoxon_sens(d4)),
    }
    hp = holm({k: v['p_one_sided'] for k, v in primary.items()})
    for k, v in primary.items():
        v['holm_p'] = hp[k]
    primary['C1']['supported'] = bool(hp['C1'] < .025 and primary['C1']['gmean_ratio'] < 1)
    primary['C2']['supported'] = bool(hp['C2'] < .025 and primary['C2']['tango_ci95'][0] > -.05)
    primary['C3']['supported'] = bool(hp['C3'] < .025 and primary['C3']['upper_97_5'] < 1.25)
    primary['C4']['supported'] = bool(hp['C4'] < .025 and primary['C4']['gmean_ratio'] < 1)
    sec = {
        'S1 load tolerance X vs M0, NI -10 pp': paired_binary([held(n, X, LOADS) for n in viable], [held(n, M0, LOADS) for n in viable], -.10),
        'S2 robust all3 X > M1': paired_binary([all3(n, X) for n in viable], [all3(n, M1) for n in viable], 0.),
        'S3 robust all3 X > M0': paired_binary([all3(n, X) for n in viable], [all3(n, M0) for n in viable], 0.),
    }
    for k in sec:
        sec[k]['p_one_sided'] = sec[k]['p_one_sided'] if 'NI' in k else sec[k]['mcnemar_superiority_p_one_sided']
    hs = holm({k: v['p_one_sided'] for k, v in sec.items()})
    for k, v in sec.items():
        v['holm_p'] = hs[k]
        v['supported'] = bool(hs[k] < .025 and (v['tango_ci95'][0] > v['margin'] if 'NI' in k else v['difference'] > 0))
    descriptive = {
        'X vs Joint-hist robust v3 (two-sided)': paired_binary([held(n, X, STD) for n in viable], [held(n, JH, STD) for n in viable], 0.),
        'X vs Joint-hist load (two-sided)': paired_binary([held(n, X, LOADS) for n in viable], [held(n, JH, LOADS) for n in viable], 0.),
        'X vs M0 robust at torsion 0.001': paired_binary([held(n, X, STD, '0.001') for n in viable], [held(n, M0, STD, '0.001') for n in viable], -.05),
        'X vs M0 robust at torsion 0.02': paired_binary([held(n, X, STD, '0.02') for n in viable], [held(n, M0, STD, '0.02') for n in viable], -.05),
        'X/Joint-hist N_F': t_logratio(np.log(nf(X)/nf(JH))),
    }
    if all('GPD+gate' in orders[n]['methods'] for n in viable):
        g = np.log(nf('GPD+gate')/nf(M0))
        descriptive['GPD+gate/M0 N_F (two-sided reading)'] = t_logratio(g)
        descriptive['GPD+gate vs M0 robust v3'] = paired_binary([held(n, 'GPD+gate', STD) for n in viable], [held(n, M0, STD) for n in viable], 0.)
        descriptive['GPD+gate vs M0 load'] = paired_binary([held(n, 'GPD+gate', LOADS) for n in viable], [held(n, M0, LOADS) for n in viable], 0.)
    hyb = fr['names']['hybrid']
    viol = [n for n in viable if orders[n]['methods'][hyb]['nf'] > min(2*orders[n]['methods'][X]['nf'] - 1, 2*orders[n]['methods'][M0]['nf'])]
    descriptive['fallback'] = dict(objects=len(viable), violations=len(viol), violating=viol,
                                   hybrid_nf_gmean=float(np.exp(np.mean(np.log(nf(hyb))))))
    rates = {m: dict(N_F_gmean=float(np.exp(np.mean(np.log(nf(m))))), success_at={k: float(np.mean(nf(m) <= k)) for k in BUDGETS},
                     robust_v3=float(np.mean([held(n, m, STD) for n in viable])), load=float(np.mean([held(n, m, LOADS) for n in viable])),
                     population_success_at_1=float(np.sum(nf(m) <= 1)/counts['catalogued']))
             for m in orders[viable[0]]['methods'] if all(orders[n]['methods'].get(m, {}).get('first') for n in viable)}
    for m in (X, M0, M1, JH):
        rates[m]['robust_all3'] = float(np.mean([all3(n, m) for n in viable]))
    cl = [n.split('_')[1][0] for n in viable]
    cluster = dict(C1_gmean_ratio=cluster_ci(d1, cl, lambda r: float(np.exp(d1[r].mean()))),
                   C2_difference=cluster_ci(None, cl, lambda r: float(np.mean([held(viable[i], X, STD) - held(viable[i], M0, STD) for i in r]))),
                   C3_gmean_ratio=cluster_ci(d3, cl, lambda r: float(np.exp(d3[r].mean()))),
                   clusters=len(set(cl)))
    tcl = [n.split('_')[1][0] for n in timed]
    cluster['C4_gmean_ratio'] = cluster_ci(d4, tcl, lambda r: float(np.exp(d4[r].mean())))
    tim = {m: dict(median_s=float(np.median(T(m))), p90_s=float(np.percentile(T(m), 90)), p95_s=float(np.percentile(T(m), 95)),
                   p99_s=float(np.percentile(T(m), 99))) for m in timing[timed[0]]['cold']}
    res = dict(protocol=fr['protocol'], freeze_sha256=sha(a.freeze), counts=counts, primary=primary, secondary=sec, descriptive=descriptive,
               rates=rates, timing=tim, timed_objects=len(timed), cluster_sensitivity=cluster,
               load_during_timing=dict(median_before=float(np.median([timing[n]['load_before'] for n in timed])),
                                       max_after=float(max(timing[n]['load_after'] for n in timed))))
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(dict(counts=counts, primary={k: (v.get('gmean_ratio', v.get('difference')), v.get('ci95', v.get('tango_ci95')), v['holm_p'], v['supported'])
                                                 for k, v in primary.items()},
                          secondary={k: (v['difference'], v['tango_ci95'], v['holm_p'], v['supported']) for k, v in sec.items()}), indent=1, default=str))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('prepare', 'orders', 'physics', 'timing', 'analyse'):
        s = sub.add_parser(name)
        s.add_argument('--pool', required=True); s.add_argument('--out-dir', required=True)
        s.add_argument('--binary'); s.add_argument('--freeze'); s.add_argument('--jobs', type=int, default=3); s.add_argument('--cpu', type=int, default=3)
        s.add_argument('--out')
    a = p.parse_args()
    if a.cmd != 'prepare' and a.freeze:
        fr = load_freeze(a.freeze)
        if a.binary and sha(a.binary) != fr['binary_sha256']:
            raise SystemExit('the binary is not the frozen one')
    dict(prepare=prepare, orders=orders, physics=physics, timing=timing, analyse=analyse)[a.cmd](a)


if __name__ == '__main__':
    main()
