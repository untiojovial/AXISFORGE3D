"""Full online cost from receiving a part to returning a grasp (PHASE7_PROTOCOL.md, I.5).

    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/research/phase7/full_online_cost.py \
        --binary target/release/axisforge --dev research/phase6/dev_parts.json --catalogues cat80/ --store store.json \
        --projects POOL --sweep research/phase7/results/sweep_v2_extended.json.gz --parts 60 --out full_online_costs.json \
        [--dino-checkpoint models/dinov3-timm-vits16 ...]

One thread, nothing else running, parts in a seeded random order, methods in a random order per part.
For every part and method, the pipeline as implemented today:
    T_total = T_candidates (axisforge research-candidates, a separate process)
            + T_features (Python) + T_inference (models) + T_prior_file
            + T_plan (axisforge plan --nr-grasps 1 --prior, a separate process, with its stage timings)
M0 has no prior: T_total = T_plan. The first call of each (part, method) is 'first', the next three
'repeat' (file cache warm). Because the prior needs the candidates before planning, the planner generates
them twice; 'integrated' subtracts the planner's own generation time to estimate a single-process
implementation (an engineering estimate, reported separately). Render + DINOv3 encoding of a new part is
timed separately (Phase V recipe: 16 views at 96 px) as the extra cost a visual prior would add. Unit: the part.
Models (P(F): gradient boosting on the 316 development parts; P(H|F): gradient boosting on the 151) are
trained before timing and are not part of the online cost.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402
from research_grasping.phase6 import fusion  # noqa: E402


def pct(x):
    x = np.asarray(x, float)
    return {**{f'p{q}': round(float(np.percentile(x, q)), 4) for q in (50, 90, 95, 99)}, 'mean': round(float(x.mean()), 4)}


def run(args, env):
    t0 = time.perf_counter(); c0 = os.times()
    out = subprocess.run(args, check=True, capture_output=True, text=True, env=env).stdout
    c1 = os.times()
    return time.perf_counter()-t0, (c1.children_user-c0.children_user)+(c1.children_system-c0.children_system), out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('binary', 'dev', 'catalogues', 'store', 'projects', 'sweep', 'out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--parts', type=int, default=60)
    p.add_argument('--beta', type=float, default=.3)
    for k in ('dino-checkpoint', 'dino-weights-sha256', 'dino-config-sha256', 'dino-revision', 'library'):
        p.add_argument(f'--{k}')
    a = p.parse_args()
    env = dict(os.environ, RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1')
    store = json.loads(Path(a.store).read_text())
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    sw = json.load(gzip.open(a.sweep, 'rt'))
    std = sw['standard']
    lib = [s.strip() for s in Path(a.library).read_text().splitlines() if s.strip()] if a.library else []
    train_pf = sorted(set(dev) | set(lib))
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in train_pf}
    X = {n: bl.features(cats[n], store[n]['low'], store[n]['high']) for n in train_pf}
    pf = bl.fit(np.vstack([X[n] for n in train_pf]), np.concatenate([cats[n]['feasible'] for n in train_pf]).astype(int), 0)
    Xh, yh = [], []
    for n in dev:
        idx = np.flatnonzero(cats[n]['feasible'])
        Xh.append(X[n][idx])
        yh.append([all(c in sw['parts'][n][str(cats[n]['label'][i])] and sw['parts'][n][str(cats[n]['label'][i])][c][0] for c in std) for i in idx])
    ph = bl.fit(np.vstack(Xh), np.concatenate(yh).astype(int), 0)
    rng = np.random.default_rng(20261205)
    parts = [dev[i] for i in rng.permutation(len(dev))[:a.parts]]
    tmp = Path(tempfile.mkdtemp())
    first = parts[0]
    run([a.binary, 'plan', '--json', '--nr-grasps', '1', str(Path(a.projects)/store[first]['project']), first], env)  # warm-up
    rows = []
    for n in parts:
        proj = str(Path(a.projects)/store[n]['project'])
        methods = ['M0', 'M1', 'Joint']
        for m in [methods[i] for i in rng.permutation(3)]:
            for rep in range(4):
                r = dict(part=n, method=m, call='first' if rep == 0 else 'repeat')
                if m == 'M0':
                    w, c, out = run([a.binary, 'plan', '--json', '--nr-grasps', '1', proj, n], env)
                    rep_json = json.loads(out)
                    r.update(T_candidates=0., T_features=0., T_inference=0., T_prior_file=0., T_plan=w, cpu_plan=c)
                else:
                    w0, c0, out0 = run([a.binary, 'research-candidates', proj, n], env)
                    t = time.perf_counter()
                    cands = json.loads(out0)
                    labels = [rr['candidate']['label'] for rr in cands]
                    cat = cats[n]
                    assert labels == [str(x) for x in cat['label']], 'catalogue and live export differ'
                    F = bl.features(cat, store[n]['low'], store[n]['high'])
                    t_feat = time.perf_counter() - t
                    t = time.perf_counter()
                    pr = pf.predict_proba(F)[:, 1]
                    pt = pr/max(pr.max(), 1e-12)
                    if m == 'Joint':
                        h = ph.predict_proba(F)[:, 1]
                        s = pt*h*np.maximum(fusion.quality(cat, 'cen'), 1e-3)**a.beta
                        prior, weight = fusion.encoded_prior(cat, fusion.order(cat, s)), fusion.PLANNER_MAX_WEIGHT
                    else:
                        prior, weight = pt, 2.
                    t_inf = time.perf_counter() - t
                    t = time.perf_counter()
                    path = tmp/f'{n}-{m}.json'
                    path.write_text(json.dumps(dict(weight=weight, candidates=[dict(label=l, score=float(v)) for l, v in zip(labels, prior)])))
                    t_file = time.perf_counter() - t
                    w, c, out = run([a.binary, 'plan', '--json', '--nr-grasps', '1', '--prior', str(path), proj, n], env)
                    rep_json = json.loads(out)
                    r.update(T_candidates=w0, cpu_candidates=c0, T_features=t_feat, T_inference=t_inf, T_prior_file=t_file,
                             T_plan=w, cpu_plan=c)
                r['timings'] = rep_json['timings']
                r['examined'] = rep_json['candidatesExamined']
                r['found'] = len(rep_json['plans']) > 0
                r['T_total'] = r['T_candidates']+r['T_features']+r['T_inference']+r['T_prior_file']+r['T_plan']
                r['T_integrated'] = r['T_total'] - (r['timings']['generationSeconds'] if m != 'M0' else 0.)
                rows.append(r)
            print(n, m, round(rows[-1]['T_total'], 3), flush=True)
    dino = None
    if a.dino_checkpoint:
        from research_grasping.phase4 import visual_prior as vp
        from research_grasping.phase4.frozen_models import load_frozen
        enc = load_frozen('dinov3-timm', a.dino_checkpoint, config_sha256=a.dino_config_sha256,
                          weights_sha256=a.dino_weights_sha256, model_revision=a.dino_revision)
        dr = []
        for n in parts[:20]:
            t = time.perf_counter()
            v, f = vp.part_mesh(Path(a.projects)/store[n]['project'], n)
            imgs = vp.render_views(v, f)
            t_render = time.perf_counter() - t
            t = time.perf_counter(); d = vp.describe(enc, imgs); t_enc = time.perf_counter() - t
            cos = float(d @ np.asarray(store[n]['descriptors']['dinov3'], float))
            dr.append(dict(part=n, render_s=t_render, encode_s=t_enc, cosine_to_stored_descriptor=round(cos, 6)))
        dino = dict(parts=len(dr), render_s=pct([d['render_s'] for d in dr]), encode_s=pct([d['encode_s'] for d in dr]),
                    min_cosine_to_stored=min(d['cosine_to_stored_descriptor'] for d in dr),
                    note=f'Phase V descriptor recipe ({vp.RECIPE}; {vp.VIEWS} views at {vp.SIZE} px), DINOv3 ViT-S/16 (timm) '
                         'on CPU, one thread; cosine to the stored Phase V descriptor checks the recipe')
    summary = {}
    for m in ('M0', 'M1', 'Joint'):
        for call in ('first', 'repeat'):
            rr = [r for r in rows if r['method'] == m and r['call'] == call]
            per_part = {}
            for r in rr:
                per_part.setdefault(r['part'], []).append(r)
            agg = lambda key: [float(np.median([x[key] for x in v])) for v in per_part.values()]
            summary[f'{m} {call}'] = {k: pct(agg(k)) for k in ('T_total', 'T_integrated', 'T_candidates', 'T_features',
                                                              'T_inference', 'T_plan')}
            summary[f'{m} {call}']['stages_plan_median_s'] = {
                s: round(float(np.median([x['timings'][s] for x in rr])), 4)
                for s in ('setupSeconds', 'generationSeconds', 'rankingSeconds', 'searchSeconds')}
    base = {r['part']: [] for r in rows}
    ratios = {}
    for m in ('M1', 'Joint'):
        for key in ('T_total', 'T_integrated'):
            d = []
            for n in parts:
                a0 = np.median([r['T_total'] for r in rows if r['part'] == n and r['method'] == 'M0' and r['call'] == 'repeat'])
                a1 = np.median([r[key] for r in rows if r['part'] == n and r['method'] == m and r['call'] == 'repeat'])
                d.append(np.log(a1/a0))
            d = np.array(d)
            from research_grasping.phase5.stats import _boot
            ratios[f'{m} {key} / M0'] = dict(ratio=round(float(np.exp(d.mean())), 4), ci95=[round(x, 4) for x in _boot(d, 95)],
                                            faster_parts=int((d < 0).sum()), slower_parts=int((d > 0).sum()))
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), parts=len(parts), beta=a.beta,
               summary=summary, ratios_vs_M0_repeat=ratios, dino_extra_cost=dino, rows=rows)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('rows', 'note')}, indent=1)[:4000])
    del base


if __name__ == '__main__':
    main()
