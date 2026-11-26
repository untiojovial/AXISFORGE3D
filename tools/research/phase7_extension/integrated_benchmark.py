"""Measured end-to-end cost of the native learned orders against the original planner.

    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/research/phase7_extension/integrated_benchmark.py \
        --binary target/release/axisforge --models research/phase7_extension/native/learned_order_phase7.json \
        --dev-costs research/phase7/results/full_online_costs.json --dev-store store.json --dev-projects POOL \
        --egad-timing research/phase7/results/confirmation_timing.json --egad-store CONF/store.json \
        --egad-projects EGAD_POOL --out research/phase7_extension/results/integrated_runtime_benchmark.json

What is measured (all exploratory, post hoc: the objects were already used in development or in the Phase VII
confirmation):

- objects: the 60 development parts of full_online_costs.json and the 60 timed EGAD objects of
  confirmation_timing.json, i.e. the same objects as the historical cost measurements;
- modes: M0 (the original planner), and the native learned orders m1, m4, m9, joint and hybrid
  (plan --learned --order MODE), every one with --nr-grasps 1, one thread, pinned to one CPU;
- cold: a fresh process per call (process start, project load, model load, planning); --cold-reps
  repetitions, modes in a seeded random order per object and repetition; wall time, user+system CPU and
  peak resident memory (wait4) of the process;
- warm: one process per object and mode planning --warm-calls times (project and models loaded once,
  the situation of a persistent service); the first call and the median of the remaining calls;
- per call: the planner's own stage timings (setup, generation, ranking, the learned stage by step,
  search with the IK/collision/static rejections), the number of examined candidates and the first grasp;
- consistency: the first grasp and number examined of every call must be identical across repetitions.

Statistics per mode against M0, paired by object: geometric mean of the per-object time ratio with a
bootstrap 95 % CI (10 000 resamples of objects, seed 20261211), the number of objects faster/slower, the
median, p90, p95, p99 of absolute times, and the exploratory Wilcoxon signed-rank p of the log ratios.
Load averages before and after each object are recorded; the machine was not otherwise idle unless the
output's 'other_processes' list is empty.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time

import numpy as np
from scipy.stats import wilcoxon

MODES = ('M0', 'm1', 'm4', 'm9', 'joint', 'hybrid')


def spawn(args, cpu):
    """Run args pinned to one CPU; returns (stdout, wall seconds, cpu seconds, peak RSS in MiB)."""
    env = {'RAYON_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'PATH': '/usr/bin:/bin'}
    with tempfile.TemporaryFile() as out, open(os.devnull, 'wb') as err:
        t0 = time.perf_counter()
        pid = os.posix_spawn('/usr/bin/taskset', ['taskset', '-c', str(cpu), *args], env,
                             file_actions=[(os.POSIX_SPAWN_DUP2, out.fileno(), 1), (os.POSIX_SPAWN_DUP2, err.fileno(), 2)])
        _, status, ru = os.wait4(pid, 0)
        wall = time.perf_counter() - t0
        if os.waitstatus_to_exitcode(status) != 0:
            raise RuntimeError(f'{args} exited with {status}')
        out.seek(0)
        return json.loads(out.read()), wall, ru.ru_utime + ru.ru_stime, ru.ru_maxrss/1024


def plan_args(a, project, body, mode, repeat=1):
    args = [a.binary, 'plan', '--json', '--nr-grasps', '1', str(project), body]
    if mode != 'M0':
        args[3:3] = ['--learned', a.models, '--order', mode]
    if repeat > 1:
        args[3:3] = ['--repeat', str(repeat)]
    return args


def summary_call(rep):
    t = rep['timings']
    return dict(examined=rep['candidatesExamined'], first=rep['plans'][0]['candidate']['label'] if rep['plans'] else None,
                project_load=rep.get('projectLoadSeconds'), model_load=rep.get('modelLoadSeconds', 0.),
                setup=t['setupSeconds'], generation=t['generationSeconds'], ranking=t['rankingSeconds'],
                learned=t.get('learnedSeconds', 0.), learned_stages=t.get('learnedStages', {}), search=t['searchSeconds'],
                rejected=t.get('rejectedSeconds', {}),
                planner_total=t['setupSeconds'] + t['generationSeconds'] + t['rankingSeconds'] + t['searchSeconds'])


def boot_gmean(x, rng, n=10000):
    x = np.log(np.asarray(x, float))
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(np.exp(x.mean())), [float(np.exp(np.percentile(m, 2.5))), float(np.exp(np.percentile(m, 97.5)))]


def describe(v):
    v = np.asarray(v, float)
    return dict(median=float(np.median(v)), gmean=float(np.exp(np.log(v).mean())), mean=float(v.mean()),
                p90=float(np.percentile(v, 90)), p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
                max=float(v.max()))


def compare(per, key, rng):
    out = {}
    names = sorted(per)
    for m in MODES[1:]:
        r = [per[n][m][key]/per[n]['M0'][key] for n in names]
        g, ci = boot_gmean(r, rng)
        lr = np.log(r)
        out[m] = dict(ratio_gmean=g, ci95=ci, faster=int((lr < 0).sum()), slower=int((lr > 0).sum()),
                      wilcoxon_p_two_sided=float(wilcoxon(lr).pvalue) if np.any(lr != 0) else 1.,
                      absolute=describe([per[n][m][key] for n in names]))
    out['M0'] = dict(absolute=describe([per[n]['M0'][key] for n in names]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('binary', 'models', 'dev-costs', 'dev-store', 'dev-projects', 'egad-timing', 'egad-store', 'egad-projects', 'out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--cold-reps', type=int, default=3); p.add_argument('--warm-calls', type=int, default=6)
    p.add_argument('--cpu', type=int, default=3); p.add_argument('--seed', type=int, default=20261211)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    dev_store = json.loads(Path(a.dev_store).read_text()); egad_store = json.loads(Path(a.egad_store).read_text())
    dev_names = sorted({r['part'] for r in json.loads(Path(a.dev_costs).read_text())['rows']})
    egad_names = sorted(json.loads(Path(a.egad_timing).read_text())['per_object'])
    objects = [('development', n, Path(a.dev_projects)/dev_store[n]['project']) for n in dev_names]
    objects += [('egad', n, Path(a.egad_projects)/egad_store[n]['project']) for n in egad_names]
    rnd = random.Random(a.seed)
    rnd.shuffle(objects)
    if a.limit:
        objects = objects[:a.limit]
    others = subprocess.run(['ps', '-eo', 'pcpu,comm', '--sort=-pcpu'], capture_output=True, text=True).stdout.splitlines()[1:8]
    raw, t_start = {}, time.time()
    for i, (kind, n, proj) in enumerate(objects):
        load0 = os.getloadavg()[0]
        rec = dict(kind=kind, cold={m: [] for m in MODES}, warm={})
        for rep in range(a.cold_reps):
            modes = list(MODES); rnd.shuffle(modes)
            for m in modes:
                out, wall, cpu, rss = spawn(plan_args(a, proj, n, m), a.cpu)
                rec['cold'][m].append(dict(wall=wall, cpu=cpu, rss_mib=rss, **summary_call(out)))
        modes = list(MODES); rnd.shuffle(modes)
        for m in modes:
            out, wall, cpu, rss = spawn(plan_args(a, proj, n, m, a.warm_calls), a.cpu)
            calls = out['calls']
            rec['warm'][m] = dict(process_wall=wall, rss_mib=rss, project_load=out['projectLoadSeconds'],
                                  model_load=out.get('modelLoadSeconds', 0.),
                                  calls=[dict(wall=c['wallSeconds'], examined=c['candidatesExamined'], first=c['first'],
                                              generation=c['timings']['generationSeconds'],
                                              learned=c['timings'].get('learnedSeconds', 0.),
                                              learned_stages=c['timings'].get('learnedStages', {}),
                                              search=c['timings']['searchSeconds']) for c in calls])
        rec['load_avg_before_after'] = [load0, os.getloadavg()[0]]
        raw[n] = rec
        print(i+1, len(objects), kind, n, f'{time.time()-t_start:.0f}s load {load0:.2f}', flush=True)

    # consistency and per-object summaries
    inconsistent = []
    per = {kind: {} for kind in ('development', 'egad', 'all')}
    for n, rec in raw.items():
        row = {}
        for m in MODES:
            decisions = {(c['examined'], c['first']) for c in rec['cold'][m]} | {(c['examined'], c['first']) for c in rec['warm'][m]['calls']}
            if len(decisions) != 1:
                inconsistent.append([n, m, sorted(map(str, decisions))])
            cold = rec['cold'][m]
            steady = rec['warm'][m]['calls'][1:]
            row[m] = dict(cold_wall=float(np.median([c['wall'] for c in cold])),
                          cold_cpu=float(np.median([c['cpu'] for c in cold])),
                          cold_planner=float(np.median([c['planner_total'] for c in cold])),
                          rss_mib=float(np.max([c['rss_mib'] for c in cold])),
                          warm_first=rec['warm'][m]['calls'][0]['wall'],
                          warm_steady=float(np.median([c['wall'] for c in steady])),
                          generation=float(np.median([c['generation'] for c in cold])),
                          search=float(np.median([c['search'] for c in cold])),
                          learned=float(np.median([c['learned'] for c in cold])),
                          examined=cold[0]['examined'])
        per[rec['kind']][n] = row; per['all'][n] = row
    rng = np.random.default_rng(a.seed)
    results = {}
    for kind, rows in per.items():
        if not rows:
            continue
        results[kind] = dict(objects=len(rows), **{key: compare(rows, key, rng) for key in
                                                   ('cold_wall', 'cold_cpu', 'cold_planner', 'warm_first', 'warm_steady')})
        stage = {}
        for m in MODES:
            calls = [c for n in rows for c in raw[n]['cold'][m]]
            st = {}
            for c in calls:
                for k, v in c['learned_stages'].items():
                    st.setdefault(k, []).append(v)
            stage[m] = dict(generation=describe([c['generation'] for c in calls]), search=describe([c['search'] for c in calls]),
                            setup=describe([c['setup'] for c in calls]), project_load=describe([c['project_load'] for c in calls]),
                            model_load=describe([max(c['model_load'], 1e-9) for c in calls]) if m != 'M0' else None,
                            learned_total=describe([c['learned'] for c in calls]) if m != 'M0' else None,
                            learned_stages={k: describe(np.maximum(v, 1e-9)) for k, v in st.items()},
                            rss_mib=describe([c['rss_mib'] for c in calls]),
                            examined_gmean=float(np.exp(np.mean([np.log(c['examined']) for c in calls]))))
        results[kind]['stages'] = stage
    models = Path(a.models)
    doc = json.loads(models.read_text())
    nodes = sum(len(t['feature']) for part in ('feasibility', 'robustness') if doc.get(part) for t in doc[part]['trees'])
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '),
               environment=dict(machine=platform.processor() or platform.machine(), cpu_model=next(
                   (l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), None),
                   cpus=os.cpu_count(), pinned_cpu=a.cpu, threads='RAYON_NUM_THREADS=1, OMP_NUM_THREADS=1',
                   binary_sha256=hashlib.sha256(Path(a.binary).read_bytes()).hexdigest(),
                   git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
                   other_processes_at_start=others, seconds=round(time.time()-t_start, 1)),
               model=dict(file=str(models), bytes=models.stat().st_size, sha256=hashlib.sha256(models.read_bytes()).hexdigest(),
                          trees=sum(len(doc[k]['trees']) for k in ('feasibility', 'robustness') if doc.get(k)), nodes=nodes),
               design=dict(cold_reps=a.cold_reps, warm_calls=a.warm_calls, seed=a.seed, modes=MODES),
               inconsistent_decisions=inconsistent, results=results, per_object=per['all'], raw=raw)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    short = {k: {key: {m: (v[key][m].get('ratio_gmean'), v[key][m].get('ci95')) for m in MODES[1:]}
                 for key in ('cold_wall', 'warm_steady')} for k, v in results.items()}
    print(json.dumps(short, indent=1)); print('inconsistent', len(inconsistent))


if __name__ == '__main__':
    main()
