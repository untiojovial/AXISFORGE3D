"""Phase VII confirmation on EGAD (PHASE7_PROTOCOL.md, part II). Stages run in this order:

    freeze    train the frozen models on development data only and write their hashes
    prepare   exhaustive candidate catalogue (80 mm) and geometry record of every built EGAD cell
    describe  DINOv3 and untrained-ViT descriptors (Phase V recipe), for the secondary H5 only
    apply     orders of M0, M1, M4, M9, FORTIS-R Joint, the interleaved fallback and the H5 MLPs
    physics   v3 trials of the first viable grasps (nine conditions; torsion grid for M0, M1, Joint)
    articulated  articulated-arm replication of the M0, M1 and Joint first grasps (descriptive)
    timing    full online cost of M0, M1 and Joint on a seeded subset of viable objects (H4)
    analyse   the frozen statistical analysis -> confirmation_results.json

    python tools/research/phase7/egad_confirm.py freeze --library research/phase5/library_parts.txt \
        --dev research/phase6/dev_parts.json --catalogues cat80/ --store store.json \
        --sweep research/phase7/results/sweep_v2_extended.json.gz --out-dir frozen/
    python tools/research/phase7/egad_confirm.py prepare --pool POOL --binary target/release/axisforge --out-dir conf/
    python tools/research/phase7/egad_confirm.py describe --pool POOL --out-dir conf/ --models MODELS_DIR
    python tools/research/phase7/egad_confirm.py apply --frozen frozen/ --out-dir conf/
    python tools/research/phase7/egad_confirm.py physics --pool POOL --out-dir conf/ --jobs 4
    python tools/research/phase7/egad_confirm.py analyse --out-dir conf/ --frozen frozen/ --results research/phase7/results

Nothing here reads a confirmatory outcome before 'analyse', and 'analyse' applies only the analysis
fixed in part II of the protocol.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import baselines as bl, catalogue, dynamics as v1  # noqa: E402
from research_grasping.phase6 import fusion, physics as v2  # noqa: E402
from research_grasping.phase7 import physics as v3, representation as rep  # noqa: E402

ALPHA_M4 = .8
BETA_JOINT = .3
# H5 (secondary): the best DINO + geometry MLP in development (L10, linear DINOv3 branch) against the geometry
# MLP (L2), with its own controls: permuted DINOv3 (L10p) and an untrained ViT (L10r); L8 (nonlinear) is reported.
H5_VARIANTS = {'L2': dict(geometry=True), 'L10': dict(geometry=True, visual='dinov3', linear=True),
               'L10p': dict(geometry=True, visual='dinov3-permuted', linear=True),
               'L10r': dict(geometry=True, visual='vit-random-s0', linear=True), 'L8': dict(geometry=True, visual='dinov3')}
SEEDS = (0, 1, 2)
CONDITIONS = (('nominal', {}), ('mass x2', dict(mass=.2)), ('mass x0.5', dict(mass=.05)), ('friction 0.3', dict(friction=.3)),
              ('pose noise 1', dict(noise_seed=1)), ('pose noise 2', dict(noise_seed=2)),
              ('load 0.5 kg', dict(mass=.5)), ('load 1 kg', dict(mass=1.)), ('load 2 kg', dict(mass=2.)))
STANDARD = tuple(c for c, _ in CONDITIONS[:6])
LOADS = tuple(c for c, _ in CONDITIONS[6:])
METHODS = ('M0', 'M1', 'M4', 'M9', 'Joint')
TORSION_GRID = {'0.005': METHODS, '0.001': ('M0', 'M1', 'Joint'), '0.02': ('M0', 'M1', 'Joint')}
BUDGETS = (1, 5, 10, 20, 50)
CATALOGUE_TIMEOUT_S = 3600


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def mlp_inputs(F, desc, d=None):
    return dict(v=F[:, rep.V_COLUMNS].astype(np.float32),
                u=np.concatenate([F[0, rep.EXTENT_COLUMNS], np.asarray(desc['d2'], float)]).astype(np.float32),
                d=None if d is None else np.asarray(d, np.float32))


# ------------------------------------------------------------------------------------------ freeze
def freeze(a):
    import sklearn
    import torch
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    store = json.loads(Path(a.store).read_text())
    library = [s.strip() for s in Path(a.library).read_text().splitlines() if s.strip()]
    devj = json.loads(Path(a.dev).read_text())
    dev = sorted(devj['parts'])
    names = sorted(set(library) | set(devj['parts']) | set(devj['nonviable']))
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in names}
    F = {n: bl.features(cats[n], store[n]['low'], store[n]['high']) for n in names}
    t = time.perf_counter()
    pf = bl.fit(np.vstack([F[n] for n in names]), np.concatenate([cats[n]['feasible'] for n in names]).astype(int), 0)
    pf_s = time.perf_counter() - t
    sw = json.load(gzip.open(a.sweep, 'rt'))
    std = sw['standard']
    Xh, yh = [], []
    for n in dev:
        idx = np.flatnonzero(cats[n]['feasible'])
        Xh.append(F[n][idx])
        yh.append([all(c in sw['parts'][n][str(cats[n]['label'][i])] and sw['parts'][n][str(cats[n]['label'][i])][c][0]
                       for c in std) for i in idx])
    t = time.perf_counter()
    ph = bl.fit(np.vstack(Xh), np.concatenate(yh).astype(int), 0)
    ph_s = time.perf_counter() - t
    files = {}
    for key, model in (('p_feasible.pkl', pf), ('p_robust_given_feasible.pkl', ph)):
        (out/key).write_bytes(pickle.dumps(model))
        files[key] = sha256(out/key)
    # H5 MLPs on all development parts (inner validation: 15 % of name groups, seed 1000)
    groups = rep.name_groups(names)
    tg = sorted(set(groups.values()))
    rng = np.random.default_rng(1000)
    val_g = set(rng.choice(tg, max(1, int(round(.15*len(tg)))), replace=False))
    inner_train = [n for n in names if groups[n] not in val_g]
    inner_val = [n for n in names if groups[n] in val_g]
    perm = np.random.default_rng(7).permutation(len(names))
    mlp_info = {}
    torch.set_num_threads(1)
    for v, cfg in H5_VARIANTS.items():
        for seed in SEEDS:
            vis = cfg.get('visual')
            sub = {}
            for i, n in enumerate(names):
                desc = store[n]['descriptors']
                if vis == 'dinov3-permuted':
                    d = store[names[perm[i]]]['descriptors']['dinov3']
                elif vis:
                    d = desc[vis]
                else:
                    d = None
                sub[n] = dict(**mlp_inputs(F[n], desc, d), y=cats[n]['feasible'].astype(np.float32))
            model = rep.MLPModel(geometry=cfg.get('geometry', False), visual=vis is not None,
                                 linear_visual=cfg.get('linear', False), seed=seed)
            model.fit(sub, inner_train, inner_val)
            key = f'mlp_{v}_s{seed}.pt'
            torch.save(model.state(), out/key)
            files[key] = sha256(out/key)
            mlp_info[key] = dict(variant=v, seed=seed, config=cfg, epochs=model.epochs, params=model.params,
                                 train_s=round(model.train_seconds, 2))
            print(key, model.epochs, flush=True)
    doc = dict(
        stage='freeze', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), code_commit=git_commit(),
        code_sha256={p: sha256(ROOT/p) for p in ('tools/research/phase7/egad_confirm.py', 'research_grasping/phase5/baselines.py',
                                                 'research_grasping/phase5/catalogue.py', 'research_grasping/phase6/fusion.py',
                                                 'research_grasping/phase6/physics.py', 'research_grasping/phase7/physics.py',
                                                 'research_grasping/phase7/representation.py')},
        environment=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                         torch=torch.__version__),
        training_data=dict(p_feasible=dict(parts=len(names), candidates=int(sum(len(cats[n]['label']) for n in names)),
                                           label='exhaustive planner feasibility at 80 mm'),
                           p_robust_given_feasible=dict(parts=len(dev), candidates=int(sum(len(x) for x in yh)),
                                                        label='Phase VI v2 robust in the six standard conditions',
                                                        sweep_sha256=sha256(a.sweep)),
                           store_sha256=sha256(a.store),
                           catalogue_sha256={n: sha256(Path(a.catalogues)/f'{n}.npz') for n in names}),
        models=dict(p_feasible=dict(file='p_feasible.pkl', kind='gradient boosting (Phase V settings, seed 0)',
                                    features=list(bl.FEATURE_NAMES), train_s=round(pf_s, 2)),
                    p_robust_given_feasible=dict(file='p_robust_given_feasible.pkl', kind='gradient boosting (seed 0)',
                                                 train_s=round(ph_s, 2)),
                    h5_mlps=mlp_info),
        methods=dict(M0='original planner order (preliminary score, makeable first)',
                     M1='preliminary + 2 p~, p~ = P(F)/max', M4=f'p~^{ALPHA_M4} * Q_cen^{round(1-ALPHA_M4, 2)} (fusion order)',
                     M9='p~ * P(H|F) (fusion order)',
                     Joint=f'p~ * P(H|F) * max(Q_cen, 1e-3)^{BETA_JOINT} (fusion order) -- FORTIS-R Joint, final method',
                     hybrid='round robin of Joint and M0 (secondary)'),
        hyperparameters=dict(alpha_M4=ALPHA_M4, beta_joint=BETA_JOINT,
                             beta_joint_rule='protocol I.4 criterion on all 151 development parts (out-of-fold P(H|F)); '
                                             'also the value chosen in 4 of 5 outer folds'),
        files=files)
    (out/'freeze_manifest.json').write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({k: doc[k] for k in ('code_commit', 'files', 'hyperparameters')}, indent=1))


def load_frozen(frozen):
    fz = Path(frozen)
    man = json.loads((fz/'freeze_manifest.json').read_text())
    for key, h in man['files'].items():
        if sha256(fz/key) != h:
            raise SystemExit(f'frozen file {key} does not match its hash')
    import torch
    models = {}
    for key in man['files']:
        if key.endswith('.pt'):
            models[key] = rep.MLPModel.from_state(torch.load(fz/key, weights_only=True))
        else:
            models[key] = pickle.loads((fz/key).read_bytes())  # our own files, checked against the frozen hashes above
    return man, models


# ----------------------------------------------------------------------------------------- prepare
def prepare_one(task):
    binary, project, name, out = task
    path = Path(out)/'cat80'/f'{name}.npz'
    row = dict(name=name)
    if path.exists():
        cat = catalogue.load(path)
        row.update(status='catalogue', candidates=int(cat['meta']['candidates']), feasible=int(cat['meta']['feasible']))
        return row
    start = time.perf_counter()
    try:
        arrays, meta = catalogue.build(binary, project, name)
    except subprocess.TimeoutExpired:
        row.update(status='planner budget exhausted', seconds=round(time.perf_counter()-start, 1))
        return row
    except Exception as e:  # noqa: BLE001 -- every failure is recorded, never dropped
        row.update(status='catalogue error', error=f'{type(e).__name__}: {str(e)[:300]}', seconds=round(time.perf_counter()-start, 1))
        return row
    catalogue.save(path, arrays, meta)
    row.update(status='catalogue', candidates=int(meta['candidates']), feasible=int(meta['feasible']),
               generate_s=meta['generate_s'], solve_s=meta['solve_s'])
    return row


def prepare(a):
    from research_grasping.phase4 import visual_prior as vp
    out = Path(a.out_dir); (out/'cat80').mkdir(parents=True, exist_ok=True)
    built = json.loads((Path(a.pool)/'built.json').read_text())['rows']
    tasks = [(a.binary, str(Path(a.pool)/r['project']), r['name'], str(out)) for r in built if r.get('kept')]
    status = {r['name']: dict(name=r['name'], status='cell build refused', reason=r.get('reason')) for r in built if not r.get('kept')}
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for row in pool.map(prepare_one, tasks):
            status[row['name']] = row
            print(row, flush=True)
    store = {}
    for r in built:
        if not r.get('kept'):
            continue
        v, f = vp.part_mesh(Path(a.pool)/r['project'], r['name'])
        store[r['name']] = dict(project=r['project'], low=v.min(0).tolist(), high=v.max(0).tolist(),
                                descriptors=dict(d2=bl.d2_descriptor(v, f).tolist()))
    for n, row in status.items():
        if row['status'] == 'catalogue':
            if row['candidates'] == 0:
                row['status'] = 'no candidate generated'
            elif row['feasible'] == 0:
                row['status'] = 'no viable candidate'
            else:
                row['status'] = 'viable'
    (out/'store.json').write_text(json.dumps(store) + '\n')
    (out/'status.json').write_text(json.dumps(status, indent=1) + '\n')
    print({s: sum(1 for r in status.values() if r['status'] == s) for s in {r['status'] for r in status.values()}})


# ---------------------------------------------------------------------------------------- describe
def describe_one(task):
    pool, models, name, project = task
    import torch
    torch.set_num_threads(1)
    from research_grasping.phase4 import visual_prior as vp
    from research_grasping.phase4.frozen_models import load_frozen as lf
    enc = {}
    for key, folder in (('dinov3', 'dinov3-timm-vits16'), ('vit-random-s0', 'vit-random-timm-s0')):
        p = Path(models)/folder
        enc[key] = lf('dinov3-timm', str(p), config_sha256=sha256(p/'config.json'), weights_sha256=sha256(p/'model.safetensors'),
                      model_revision=folder)
    v, f = vp.part_mesh(Path(pool)/project, name)
    imgs = vp.render_views(v, f)
    return name, {k: vp.describe(e, imgs).tolist() for k, e in enc.items()}


def describe(a):
    out = Path(a.out_dir)
    store = json.loads((out/'store.json').read_text())
    tasks = [(a.pool, a.models, n, r['project']) for n, r in store.items()]
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for n, d in pool.map(describe_one, tasks):
            store[n]['descriptors'].update(d)
            print(n, flush=True)
    names = sorted(store)
    perm = np.random.default_rng(7).permutation(len(names))
    for i, n in enumerate(names):
        store[n]['descriptors']['dinov3-permuted'] = store[names[perm[i]]]['descriptors']['dinov3']
    (out/'store.json').write_text(json.dumps(store) + '\n')


# ------------------------------------------------------------------------------------------- apply
def interleave(cat, learned, default):
    seen, n = set(), 0
    for k in range(len(learned)):
        for o in (learned, default):
            i = int(o[k])
            if i in seen:
                continue
            seen.add(i); n += 1
            if cat['feasible'][i]:
                return n, i
    return None, None


def apply(a):
    man, models = load_frozen(a.frozen)
    out = Path(a.out_dir)
    store = json.loads((out/'store.json').read_text())
    status = json.loads((out/'status.json').read_text())
    pf, ph = models['p_feasible.pkl'], models['p_robust_given_feasible.pkl']
    rows = {}
    for n, st in sorted(status.items()):
        row = dict(status=st['status'])
        rows[n] = row
        if st['status'] not in ('viable', 'no viable candidate'):
            continue
        cat = catalogue.load(out/'cat80'/f'{n}.npz')
        F = bl.features(cat, store[n]['low'], store[n]['high'])
        t = time.perf_counter()
        pt = bl.scores(pf, F)
        h = ph.predict_proba(F)[:, 1]
        orders = {'M0': catalogue.order(cat), 'M1': catalogue.order(cat, pt, 2.),
                  'M4': fusion.order(cat, fusion.config_score('M4', dict(alpha=ALPHA_M4, q='cen'), cat, pt)),
                  'M9': fusion.order(cat, pt*h),
                  'Joint': fusion.order(cat, pt*h*np.maximum(fusion.quality(cat, 'cen'), 1e-3)**BETA_JOINT)}
        row['inference_s'] = round(time.perf_counter()-t, 4)
        if 'dinov3' in store[n]['descriptors']:
            for v, cfg in H5_VARIANTS.items():
                ps = []
                for seed in SEEDS:
                    m = models[f'mlp_{v}_s{seed}.pt']
                    vis = cfg.get('visual')
                    d = store[n]['descriptors'][vis] if vis else None
                    sub = {n: dict(**mlp_inputs(F, store[n]['descriptors'], d), y=np.zeros(len(F), np.float32))}
                    ps.append(m.predict(sub, [n])[n])
                p = np.mean(ps, 0)
                orders[f'H5:{v}'] = catalogue.order(cat, p/max(float(p.max()), 1e-12), 2.)
        row['candidates'] = int(len(cat['label']))
        for m, o in orders.items():
            pos = np.flatnonzero(cat['feasible'][o])
            row[m] = dict(nf=int(pos[0])+1 if len(pos) else None,
                          first=str(cat['label'][o[pos[0]]]) if len(pos) else None,
                          **{f's@{b}': bool(len(pos) and pos[0] < b) for b in BUDGETS})
        if len(np.flatnonzero(cat['feasible'])):
            nh, idx = interleave(cat, orders['Joint'], orders['M0'])
            row['hybrid'] = dict(nf=nh, first=str(cat['label'][idx]), **{f's@{b}': bool(nh <= b) for b in BUDGETS})
    (out/'orders.json').write_text(json.dumps(dict(freeze=man['files'], rows=rows), indent=1) + '\n')
    print(len(rows), 'objects;', sum(1 for r in rows.values() if r['status'] == 'viable'), 'viable')


# ----------------------------------------------------------------------------------------- physics
STA = {}


def physics_init(a):
    STA.update(vars(a))


def physics_job(task):
    name, project, jobs = task
    s = STA
    path = Path(s['out_dir'])/'trials'/f'{name}.json'
    if path.exists():
        return name, 0.
    R, t, g, tt = v1.part_pose(Path(s['pool'])/project, name)
    pieces = [v1.read_obj(p) for p in sorted((Path(s['pool'])/name/'collision').glob('model_collision_*.obj'),
                                              key=lambda p: int(p.stem.split('_')[-1]))]
    cat = catalogue.load(Path(s['out_dir'])/'cat80'/f'{name}.npz')
    com, _ = v2.planner_com(cat)
    cands = {c['label']: c for c in catalogue.candidates(cat)}
    start = time.perf_counter()
    res = {}
    for label, torsions in jobs:
        res[label] = {}
        for tor in torsions:
            res[label][tor] = {c: v3.trial(pieces, R, t, tt, cands[label], g, com=com, torsion=float(tor), **kw)
                               for c, kw in CONDITIONS}
    path.write_text(json.dumps(dict(seconds=round(time.perf_counter()-start, 1), trials=res)))
    return name, time.perf_counter()-start


def physics(a):
    out = Path(a.out_dir); (out/'trials').mkdir(exist_ok=True)
    orders = json.loads((out/'orders.json').read_text())['rows']
    store = json.loads((out/'store.json').read_text())
    tasks = []
    for n, r in orders.items():
        if r['status'] != 'viable':
            continue
        want = {}
        for tor, ms in TORSION_GRID.items():
            for m in ms:
                want.setdefault(r[m]['first'], set()).add(tor)
        tasks.append((n, store[n]['project'], sorted((lab, sorted(ts)) for lab, ts in want.items())))
    tasks.sort(key=lambda t: -sum(len(ts) for _, ts in t[2]))
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=physics_init, initargs=(a,)) as pool:
        for n, sec in pool.map(physics_job, tasks):
            print(n, round(sec), flush=True)


# ------------------------------------------------------------------------------------- articulated
ART_CONDITIONS = (('nominal', {}), ('mass x2', dict(mass=.2)), ('mass x0.5', dict(mass=.05)), ('friction 0.3', dict(friction=.3)),
                  ('load 0.5 kg', dict(mass=.5)), ('load 1 kg', dict(mass=1.)), ('load 2 kg', dict(mass=2.)))


def articulated_job(task):
    import tempfile
    from research_grasping.phase7 import articulated as art
    name, project, labels = task
    s = STA
    path = Path(s['out_dir'])/'articulated'/f'{name}.json'
    if path.exists():
        return name, 0.
    proj = Path(s['pool'])/project
    R, t, g, tt = v1.part_pose(proj, name)
    pieces = [v1.read_obj(p) for p in sorted((Path(s['pool'])/name/'collision').glob('model_collision_*.obj'),
                                              key=lambda p: int(p.stem.split('_')[-1]))]
    cat = catalogue.load(Path(s['out_dir'])/'cat80'/f'{name}.npz')
    com, _ = v2.planner_com(cat)
    project_doc = art.load_project(proj)
    start = time.perf_counter()
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        for lab in labels:
            prior = Path(tmp)/'prior.json'
            prior.write_text(json.dumps(dict(weight=5., candidates=[dict(label=lab, score=1.)])))
            rep_ = json.loads(subprocess.run([s['binary'], 'plan', '--json', '--nr-grasps', '1', '--prior', str(prior), str(proj), name],
                                             check=True, capture_output=True, text=True,
                                             env={'RAYON_NUM_THREADS': '1', 'PATH': '/usr/bin:/bin'}).stdout)
            if not rep_['plans'] or rep_['plans'][0]['candidate']['label'] != lab:
                res[lab] = dict(error='planner did not return this label first')
                continue
            res[lab] = {c: art.trial(project_doc, pieces, R, t, tt, rep_['plans'][0], g, com=com, **kw) for c, kw in ART_CONDITIONS}
    path.write_text(json.dumps(dict(seconds=round(time.perf_counter()-start, 1), trials=res)))
    return name, time.perf_counter()-start


def articulated(a):
    """Descriptive replication at the articulated level (first grasps of M0, M1 and Joint)."""
    out = Path(a.out_dir); (out/'articulated').mkdir(exist_ok=True)
    orders = json.loads((out/'orders.json').read_text())['rows']
    store = json.loads((out/'store.json').read_text())
    tasks = [(n, store[n]['project'], sorted({r[m]['first'] for m in ('M0', 'M1', 'Joint')}))
             for n, r in sorted(orders.items()) if r['status'] == 'viable' and not n.startswith('egad_eval_')]
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=physics_init, initargs=(a,)) as pool:
        for n, sec in pool.map(articulated_job, tasks):
            print(n, round(sec), flush=True)


# ------------------------------------------------------------------------------------------ timing
def timing(a):
    """H4: full online cost per object, one thread, nothing else running (protocol II.5)."""
    man, models = load_frozen(a.frozen)
    pf, ph = models['p_feasible.pkl'], models['p_robust_given_feasible.pkl']
    out = Path(a.out_dir)
    orders = json.loads((out/'orders.json').read_text())['rows']
    store = json.loads((out/'store.json').read_text())
    viable = sorted(n for n, r in orders.items() if r['status'] == 'viable' and not n.startswith('egad_eval_'))
    rng = np.random.default_rng(20261207)
    names = [viable[i] for i in sorted(rng.permutation(len(viable))[:a.objects])]
    env = dict(os.environ, RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1')
    tmp = out/'timing_tmp'; tmp.mkdir(exist_ok=True)

    def run(args):
        t0 = time.perf_counter()
        o = subprocess.run(args, check=True, capture_output=True, text=True, env=env).stdout
        return time.perf_counter()-t0, o
    proj0 = str(Path(a.pool)/store[names[0]]['project'])
    run([a.binary, 'plan', '--json', '--nr-grasps', '1', proj0, names[0]])  # warm-up
    per, raw = {}, []
    for n in names:
        proj = str(Path(a.pool)/store[n]['project'])
        cat = catalogue.load(out/'cat80'/f'{n}.npz')
        per[n] = {}
        for m in [('M0', 'M1', 'Joint')[i] for i in rng.permutation(3)]:
            tot = []
            for rep_i in range(4):
                if m == 'M0':
                    w, o = run([a.binary, 'plan', '--json', '--nr-grasps', '1', proj, n])
                    tt = w; tc = 0.
                else:
                    tc, o0 = run([a.binary, 'research-candidates', proj, n])
                    t0 = time.perf_counter()
                    labels = [r['candidate']['label'] for r in json.loads(o0)]
                    assert labels == [str(x) for x in cat['label']]
                    F = bl.features(cat, store[n]['low'], store[n]['high'])
                    pt = bl.scores(pf, F)
                    if m == 'Joint':
                        h = ph.predict_proba(F)[:, 1]
                        prior = fusion.encoded_prior(cat, fusion.order(cat, pt*h*np.maximum(fusion.quality(cat, 'cen'), 1e-3)**BETA_JOINT))
                        weight = fusion.PLANNER_MAX_WEIGHT
                    else:
                        prior, weight = pt, 2.
                    path = tmp/f'{n}-{m}.json'
                    path.write_text(json.dumps(dict(weight=weight, candidates=[dict(label=l, score=float(v)) for l, v in zip(labels, prior)])))
                    tp = time.perf_counter()-t0
                    w, o = run([a.binary, 'plan', '--json', '--nr-grasps', '1', '--prior', str(path), proj, n])
                    tt = tc + tp + w
                rep_doc = json.loads(o)
                first = rep_doc['plans'][0]['candidate']['label'] if rep_doc['plans'] else None
                raw.append(dict(object=n, method=m, call=rep_i, T_total=tt, T_candidates=tc, examined=rep_doc['candidatesExamined'],
                                found=first, expected=orders[n][m]['first'], timings=rep_doc['timings']))
                if rep_i:
                    tot.append(tt)
            per[n][m] = float(np.median(tot))
        print(n, {k: round(v, 3) for k, v in per[n].items()}, flush=True)
    mismatch = sum(1 for r in raw if r['found'] != r['expected'])
    (out/'timing.json').write_text(json.dumps(dict(objects=len(names), seed=20261207, per_object=per,
                                                    first_grasp_mismatches=mismatch, rows=raw), indent=1) + '\n')
    print('mismatches', mismatch)


# ----------------------------------------------------------------------------------------- analyse
def analyse(a):
    """The analysis fixed in PHASE7_PROTOCOL.md part II (II.6). Unit: the object; paired by object."""
    from scipy.stats import binom, norm, t as tdist, wilcoxon
    from research_grasping.phase5.stats import _boot
    from research_grasping.phase6.mechanism import bootstrap_diff
    out = Path(a.out_dir)
    od = json.loads((out/'orders.json').read_text())
    rows = od['rows']
    trials = {p.stem: json.loads(p.read_text())['trials'] for p in (out/'trials').glob('*.json')}
    art = {p.stem: json.loads(p.read_text())['trials'] for p in (out/'articulated').glob('*.json')} if (out/'articulated').exists() else {}
    timing = json.loads((out/'timing.json').read_text()) if (out/'timing.json').exists() else None
    ALPHA = .025

    def held(n, m, which, tor='0.005', criterion='held_v3'):
        t = trials[n][rows[n][m]['first']][tor]
        return int(all(t[c][criterion] for c in (STANDARD if which == 'robust' else LOADS)))

    def mcnemar(A, B):
        A, B = np.asarray(A, bool), np.asarray(B, bool)
        b, c = int((A & ~B).sum()), int((~A & B).sum())
        return dict(favourable=b, unfavourable=c, p_one_sided=float(binom.sf(b-1, b+c, .5)) if b+c else 1.,
                    **bootstrap_diff(A.astype(int), B.astype(int)))

    def ratio(A, B):
        d = np.log(np.asarray(A, float)/np.asarray(B, float))
        nz = d[d != 0]
        lo, hi = _boot(d, 95)
        return dict(ratio=round(float(np.exp(d.mean())), 4), ci95=[round(lo, 4), round(hi, 4)],
                    p_one_sided=float(wilcoxon(nz, alternative='less').pvalue) if len(nz) else 1.,
                    objects=int(len(d)), fewer=int((d < 0).sum()), more=int((d > 0).sum()), ties=int((d == 0).sum()))

    def ratio_noninferior(A, B, margin):
        d = np.log(np.asarray(A, float)/np.asarray(B, float))
        se = d.std(ddof=1)/np.sqrt(len(d))
        r = ratio(A, B)
        r.update(margin=margin, p_one_sided=float(tdist.cdf((d.mean()-np.log(margin))/se, len(d)-1)) if se > 0 else 0.,
                 upper_97_5=round(float(np.exp(d.mean()+tdist.ppf(.975, len(d)-1)*se)), 4))
        return r

    def noninferior(A, B, margin):
        d = np.asarray(A, float) - np.asarray(B, float)
        se = d.std(ddof=1)/np.sqrt(len(d)) if d.std() > 0 else 1e-9
        return dict(**bootstrap_diff(A, B), margin=margin, p_one_sided=float(norm.sf((d.mean()+margin)/se)),
                    lower_97_5=round(float(d.mean()-norm.ppf(.975)*se), 4))

    def holm(results):
        keys = list(results)
        ps = [results[k]['p_one_sided'] for k in keys]
        order = np.argsort(ps)
        run = 0.
        for rank, i in enumerate(order):
            run = max(run, min(1., (len(ps)-rank)*ps[i]))
            results[keys[i]]['holm_p'] = round(run, 6)
        return results

    sets = {'primary': sorted(n for n in rows if not n.startswith('egad_eval_')),
            'evaluation set (secondary)': sorted(n for n in rows if n.startswith('egad_eval_'))}
    result = dict(protocol='research/phase7/PHASE7_PROTOCOL.md part II', freeze=od['freeze'], alpha_one_sided=ALPHA, sets={})
    for set_name, names in sets.items():
        if not names:
            continue
        viable = [n for n in names if rows[n]['status'] == 'viable']
        sim = [n for n in viable if n in trials and all(rows[n][m]['first'] in trials[n] for m in METHODS)]
        S = dict(objects=len(names),
                 status_counts={s: sum(1 for n in names if rows[n]['status'] == s) for s in sorted({rows[n]['status'] for n in names})},
                 viable=len(viable), viable_with_dynamics=len(sim), missing_dynamics=sorted(set(viable)-set(sim)))
        S['population_success_at_budget'] = {
            m: {f'success@{b}': round(float(np.mean([bool(rows[n].get(m, {}).get(f's@{b}')) for n in names])), 4) for b in BUDGETS}
            for m in list(METHODS) + ['hybrid']}
        nf = {m: [rows[n][m]['nf'] for n in viable] for m in list(METHODS) + ['hybrid']}
        S['rates_on_viable'] = {m: dict(
            N_F_gmean=round(float(np.exp(np.mean(np.log(nf[m])))), 3),
            robust_v3=round(float(np.mean([held(n, m, 'robust') for n in sim])), 4),
            load_tolerant_v3=round(float(np.mean([held(n, m, 'load') for n in sim])), 4),
            robust_v2=round(float(np.mean([held(n, m, 'robust', criterion='held') for n in sim])), 4),
            load_tolerant_v2=round(float(np.mean([held(n, m, 'load', criterion='held') for n in sim])), 4)) for m in METHODS}
        # ---- primary family: Holm over P1-P3, one-sided familywise alpha 0.025
        prim = {
            'P1: FORTIS-R Joint examines fewer candidates than the original planner (N_F ratio < 1)': ratio(nf['Joint'], nf['M0']),
            'P2: Joint robustness (v3) non-inferior to the original planner, margin 5 pp': noninferior(
                [held(n, 'Joint', 'robust') for n in sim], [held(n, 'M0', 'robust') for n in sim], .05),
            'P3 (H3): Joint N_F non-inferior to the best learned baseline M1, ratio margin 1.25': ratio_noninferior(
                nf['Joint'], nf['M1'], 1.25)}
        holm(prim)
        for k, v in prim.items():
            ok = v['holm_p'] < ALPHA
            if k.startswith('P1'):
                ok = ok and v['ratio'] < 1 and v['ci95'][1] < 1
            if k.startswith('P2'):
                ok = ok and v['lower_97_5'] > -v['margin']
            if k.startswith('P3'):
                ok = ok and v['upper_97_5'] < v['margin']
            v['supported'] = bool(ok)
        p2 = prim['P2: Joint robustness (v3) non-inferior to the original planner, margin 5 pp']
        sup = mcnemar([held(n, 'Joint', 'robust') for n in sim], [held(n, 'M0', 'robust') for n in sim])
        sup['tested'] = p2['supported']
        sup['level'] = ALPHA/3
        sup['supported'] = bool(p2['supported'] and sup['p_one_sided'] < ALPHA/3 and sup['difference'] > 0)
        S['primary'] = prim
        S['P2-superiority (tested only if P2 holds, at 0.025/3)'] = sup
        # ---- secondary family: Holm, one-sided 0.025
        sec = {
            'S1 (H1, historical): M4 load tolerance (v3) higher than M1': mcnemar(
                [held(n, 'M4', 'load') for n in sim], [held(n, 'M1', 'load') for n in sim]),
            'S2 (H2, historical): M9 robustness (v3) higher than M1': mcnemar(
                [held(n, 'M9', 'robust') for n in sim], [held(n, 'M1', 'robust') for n in sim]),
            'S3: Joint load tolerance (v3) higher than M1': mcnemar(
                [held(n, 'Joint', 'load') for n in sim], [held(n, 'M1', 'load') for n in sim]),
            'S4: Joint robustness (v3) higher than M1': mcnemar(
                [held(n, 'Joint', 'robust') for n in sim], [held(n, 'M1', 'robust') for n in sim]),
            'S5: Joint load tolerance (v3) non-inferior to the original planner, margin 10 pp': noninferior(
                [held(n, 'Joint', 'load') for n in sim], [held(n, 'M0', 'load') for n in sim], .10)}
        if timing:
            tn = sorted(timing['per_object'])
            sec['S6 (H4): Joint full online cost lower than the original planner'] = ratio(
                [timing['per_object'][n]['Joint'] for n in tn], [timing['per_object'][n]['M0'] for n in tn])
        h5 = [n for n in viable if 'H5:L10' in rows[n]]
        if h5:
            sec['S7 (H5): DINOv3 linear-branch MLP (L10) examines fewer candidates than the geometry MLP (L2)'] = ratio(
                [rows[n]['H5:L10']['nf'] for n in h5], [rows[n]['H5:L2']['nf'] for n in h5])
        holm(sec)
        for k, v in sec.items():
            ok = v['holm_p'] < ALPHA
            if 'ratio' in v and 'margin' not in v:
                ok = ok and v['ratio'] < 1
            elif 'margin' in v:
                ok = ok and v['lower_97_5'] > -v['margin']
            else:
                ok = ok and v['difference'] > 0
            v['supported'] = bool(ok)
        S['secondary'] = sec
        S['historical_v2_criterion'] = {
            'H1 v2: M4 - M1 load tolerance': mcnemar([held(n, 'M4', 'load', criterion='held') for n in sim],
                                                     [held(n, 'M1', 'load', criterion='held') for n in sim]),
            'H2 v2: M9 - M1 robustness': mcnemar([held(n, 'M9', 'robust', criterion='held') for n in sim],
                                                 [held(n, 'M1', 'robust', criterion='held') for n in sim])}
        if h5:
            k7 = next(k for k in sec if k.startswith('S7'))
            S['h5_controls'] = {f'L10 vs {c}': ratio([rows[n]['H5:L10']['nf'] for n in h5], [rows[n][f'H5:{c}']['nf'] for n in h5])
                                for c in ('L10p', 'L10r', 'L8')}
            S['h5_controls']['L2 vs P(F) gradient boosting (M1)'] = ratio([rows[n]['H5:L2']['nf'] for n in h5],
                                                                         [rows[n]['M1']['nf'] for n in h5])
            S['h5_controls']['rule'] = 'H5 holds only if S7 is supported and L10 < L10p (95 % CI upper limit below 1)'
            S['h5_controls']['H5_supported'] = bool(sec[k7]['supported'] and S['h5_controls']['L10 vs L10p']['ci95'][1] < 1)
        # ---- descriptive
        S['hybrid_interleave_Joint_M0'] = dict(
            N_F_gmean=round(float(np.exp(np.mean(np.log(nf['hybrid'])))), 3),
            bound_violations=int(sum(rows[n]['hybrid']['nf'] > 2*min(rows[n]['Joint']['nf'], rows[n]['M0']['nf']) for n in viable)),
            worst_ratio_to_M0=round(max(rows[n]['hybrid']['nf']/rows[n]['M0']['nf'] for n in viable), 3),
            negative_transfer_objects_Joint=int(sum(rows[n]['Joint']['nf'] > rows[n]['M0']['nf'] for n in viable)))
        S['torsion_sensitivity'] = {}
        for tor in ('0.001', '0.02'):
            ok = [n for n in sim if all(tor in trials[n][rows[n][m]['first']] for m in ('M0', 'M1', 'Joint'))]
            S['torsion_sensitivity'][tor] = {
                'objects': len(ok),
                'P2 Joint - M0 robust': bootstrap_diff([held(n, 'Joint', 'robust', tor) for n in ok], [held(n, 'M0', 'robust', tor) for n in ok]),
                'S3 Joint - M1 load': bootstrap_diff([held(n, 'Joint', 'load', tor) for n in ok], [held(n, 'M1', 'load', tor) for n in ok])}
        S['rotation_threshold_sensitivity'] = {}
        for lim in (5., 10., 20.):
            def hv(n, m, which):
                t = trials[n][rows[n][m]['first']]['0.005']
                return int(all(v3.held_at(t[c], lim) for c in (STANDARD if which == 'robust' else LOADS)))
            S['rotation_threshold_sensitivity'][str(int(lim))] = {
                'P2 Joint - M0 robust': bootstrap_diff([hv(n, 'Joint', 'robust') for n in sim], [hv(n, 'M0', 'robust') for n in sim]),
                'S3 Joint - M1 load': bootstrap_diff([hv(n, 'Joint', 'load') for n in sim], [hv(n, 'M1', 'load') for n in sim])}
        aa = [n for n in sim if n in art and all('error' not in art[n].get(rows[n][m]['first'], {'error': 1}) for m in ('M0', 'M1', 'Joint'))]
        if aa:
            def ha(n, m, which):
                r = art[n][rows[n][m]['first']]
                return int(all(r[c]['held_art'] for c in (('nominal', 'mass x2', 'mass x0.5', 'friction 0.3') if which == 'robust4' else LOADS)))
            S['articulated_replication'] = dict(
                objects=len(aa), note='articulated pilot model (research_grasping/phase7/articulated.py); robust4 = four standard '
                                      'conditions without pose noise; descriptive, outside both families',
                **{f'{x} - {y}, {w}': bootstrap_diff([ha(n, x, w) for n in aa], [ha(n, y, w) for n in aa])
                   for x, y in (('Joint', 'M0'), ('Joint', 'M1')) for w in ('robust4', 'load')})
        result['sets'][set_name] = S
    Path(a.results).mkdir(parents=True, exist_ok=True)
    (Path(a.results)/'confirmation_results.json').write_text(json.dumps(result, indent=1) + '\n')
    for k, v in result['sets'].items():
        print(k, v['objects'], v['status_counts'], v['viable'])
        for kk, vv in v['primary'].items():
            print('  ', kk, {x: vv.get(x) for x in ('ratio', 'difference', 'ci95', 'holm_p', 'supported')})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('freeze')
    for k in ('library', 'dev', 'catalogues', 'store', 'sweep', 'out-dir'):
        f.add_argument(f'--{k}', required=True)
    pr = sub.add_parser('prepare')
    for k in ('pool', 'binary', 'out-dir'):
        pr.add_argument(f'--{k}', required=True)
    pr.add_argument('--jobs', type=int, default=4)
    d = sub.add_parser('describe')
    for k in ('pool', 'out-dir', 'models'):
        d.add_argument(f'--{k}', required=True)
    d.add_argument('--jobs', type=int, default=4)
    ap = sub.add_parser('apply')
    for k in ('frozen', 'out-dir'):
        ap.add_argument(f'--{k}', required=True)
    ph = sub.add_parser('physics')
    for k in ('pool', 'out-dir'):
        ph.add_argument(f'--{k}', required=True)
    ph.add_argument('--jobs', type=int, default=4)
    ar = sub.add_parser('articulated')
    for k in ('pool', 'out-dir', 'binary'):
        ar.add_argument(f'--{k}', required=True)
    ar.add_argument('--jobs', type=int, default=4)
    tm = sub.add_parser('timing')
    for k in ('pool', 'out-dir', 'binary', 'frozen'):
        tm.add_argument(f'--{k}', required=True)
    tm.add_argument('--objects', type=int, default=60)
    an = sub.add_parser('analyse')
    for k in ('out-dir', 'results'):
        an.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    dict(freeze=freeze, prepare=prepare, describe=describe, apply=apply, physics=physics, articulated=articulated,
         timing=timing, analyse=analyse)[a.cmd](a)


if __name__ == '__main__':
    main()
