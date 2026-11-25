"""Visual grasp transfer for the native planner.

A frozen image model (DINOv2, DINOv3 or OpenCLIP) describes a part from 16
rendered views. The parts already solved that look most like a new one lend
their feasible grasps; every candidate of the new part is scored by how close
its geometry is to those grasps, and the scores go to ``axisforge plan
--prior``. The prior only sets the order in which the planner examines
candidates: inverse kinematics, collision and force-closure checks still run
on every candidate before one is accepted, so a wrong prior costs time, never
a bad grasp.

Grasp labels are specific to each mesh ("surface 26 across 7"), so grasps are
compared by a signature instead: approach direction, closing axis, opening
width, and the grasp centre's position in the part's bounding box.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

VIEWS = 16
SIZE = 96
# Fixed before the evaluation was run; not tuned on its results.
NEIGHBOURS = 3
PRIOR_WEIGHT = 2.0
# One unit of squared distance per 30 degrees of approach or closing-axis
# difference, 1 cm of opening, a quarter of the box, or 2 cm of height.
SCALE = dict(approach=1-np.cos(np.deg2rad(30)), closing=np.sin(np.deg2rad(30))**2,
             width=.01, position=.25, height=.02)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _rotation(q):
    x, y, z, w = (float(q[k]) for k in 'xyzw')
    n = np.sqrt(x*x+y*y+z*z+w*w)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError('invalid rotation')
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _transform(pose):
    t = pose['translation']
    return _rotation(pose['rotation']), np.array([t['x'], t['y'], t['z']], float)


def _box_mesh(size):
    h = np.array([size['x'], size['y'], size['z']], float)/2
    v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)*h
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def _lathe(profile, segments=32):
    """Surface of revolution about +Z from (z, radius) points, top to bottom."""
    phi = 2*np.pi*np.arange(segments)/segments
    v = np.array([[r*np.cos(a), r*np.sin(a), z] for z, r in profile for a in phi])
    f = []
    for k in range(len(profile)-1):
        for j in range(segments):
            a, b = k*segments+j, k*segments+(j+1) % segments
            f += [[a, a+segments, b+segments], [a, b+segments, b]]
    return v, np.array(f)


def _primitive(shape):
    kind = shape['kind']
    if kind not in ('box', 'sphere', 'cylinder', 'capsule'):
        raise ValueError(f'unsupported visual shape {kind!r}')
    if kind == 'box':
        return _box_mesh(shape['size'])
    r = float(shape['radius'])
    if kind == 'sphere':
        t = np.linspace(0, np.pi, 13)
        return _lathe([(r*np.cos(x), r*np.sin(x)) for x in t])
    h = float(shape['length'])/2
    if kind == 'cylinder':
        return _lathe([(h, 0), (h, r), (-h, r), (-h, 0)])
    if kind == 'capsule':
        top = [(h+r*np.cos(x), r*np.sin(x)) for x in np.linspace(0, np.pi/2, 7)]
        bottom = [(-h+r*np.cos(x), r*np.sin(x)) for x in np.linspace(np.pi/2, np.pi, 7)]
        return _lathe(top+bottom)
    raise ValueError(f'unsupported visual shape {kind!r}')


def part_mesh(project_path, body):
    """World-frame vertices and triangles of a body's visual geometry, metres."""
    project = json.loads(Path(project_path).read_text())['project']
    if project.get('units', {}).get('length') != 'meters':
        raise ValueError('project lengths must be metres')
    matches = [b for b in project['environment']['bodies'] if b['name'] == body]
    if len(matches) != 1:
        raise ValueError(f'expected one body named {body!r}, found {len(matches)}')
    resources = {r['id']: r for r in project.get('resources', [])}
    rb, tb = _transform(matches[0]['pose'])
    vertices, faces = [], []
    for geometry in matches[0]['geometries']:
        if geometry.get('role', 'both') not in ('visual', 'both'):
            continue
        shape = geometry['shape']
        if shape['kind'] == 'mesh':
            r = resources[shape['resourceId']]
            v = np.asarray(r['vertices'], float).reshape(-1, 3)
            f = np.asarray(r['indices'], np.int64).reshape(-1, 3)
            s = shape.get('scale', {'x': 1, 'y': 1, 'z': 1})
            v = v*np.array([s['x'], s['y'], s['z']], float)
        else:
            v, f = _primitive(shape)
        rg, tg = _transform(geometry['origin'])
        world = (v@rg.T+tg)@rb.T+tb
        faces.append(f+sum(len(x) for x in vertices))
        vertices.append(world)
    if not vertices:
        raise ValueError('body has no visual geometry')
    v = np.vstack(vertices); f = np.vstack(faces)
    if not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
        raise ValueError('invalid mesh')
    return v, f


def render_views(vertices, faces, count=VIEWS, size=SIZE):
    """The Phase III recipe: neutral gray, two-sided shading, 25° orbit."""
    from research_grasping.phase3.render_cpu import cameras, render
    bounds = np.array([vertices.min(0), vertices.max(0)])
    k, transforms = cameras(bounds, count, size, size)
    return [render(vertices, faces, k, t, size, size)['rgb'] for t in transforms]


def describe(encoder, images):
    """Unit mean of the per-view global embeddings."""
    g = np.asarray(encoder.encode(list(images))['global_embeddings'], float)
    v = g.mean(0)
    return v/max(np.linalg.norm(v), 1e-12)


RECIPE = 'Phase III CPU renderer: neutral gray, two-sided shading, 16-view orbit at 25 degrees, 96 px'


def write_views(parts, out, log=print):
    """Render (name, project) parts into a views ZIP: views/<name>/NN.png and views.json."""
    import io
    import time
    import zipfile
    from PIL import Image
    index = []
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, project in parts:
            start = time.perf_counter()
            vertices, faces = part_mesh(project, name)
            images = render_views(vertices, faces)
            for i, im in enumerate(images):
                buf = io.BytesIO(); Image.fromarray(im).save(buf, format='PNG')
                z.writestr(f'views/{name}/{i:02d}.png', buf.getvalue())
            index.append(dict(name=name, views=len(images), size=SIZE,
                              project_sha256=sha256_bytes(Path(project).read_bytes()),
                              seconds=round(time.perf_counter()-start, 2)))
            log(name, f'{index[-1]["seconds"]} s')
        z.writestr('views.json', json.dumps(dict(recipe=RECIPE, parts=index), indent=2))
    return index


def read_views(path):
    """{name: [RGB uint8 arrays]} from a views ZIP."""
    import io
    import zipfile
    from PIL import Image
    out = {}
    with zipfile.ZipFile(path) as z:
        index = json.loads(z.read('views.json'))
        for part in index['parts']:
            name = part['name']
            if '/' in name or name in out or not 1 <= int(part['views']) <= 64:
                raise ValueError('invalid views index')
            out[name] = [np.asarray(Image.open(io.BytesIO(z.read(f'views/{name}/{i:02d}.png'))).convert('RGB'))
                         for i in range(int(part['views']))]
    return out


def write_descriptors(encoder, views_path, out, label):
    """One unit descriptor per part of a views ZIP, with the encoder's provenance."""
    views = read_views(views_path)
    names = sorted(views)
    vectors = np.stack([describe(encoder, views[n]) for n in names]).astype(np.float32)
    np.savez_compressed(out, names=np.array(names), vectors=vectors, model=np.array(label),
                        provenance=np.array(json.dumps(encoder.provenance, sort_keys=True)),
                        views_sha256=np.array(sha256_bytes(Path(views_path).read_bytes())))
    return names


def read_descriptors(paths):
    """{part: {descriptor name: vector}} merged from descriptor files."""
    out = {}
    for path in paths or []:
        with np.load(path, allow_pickle=False) as f:
            model = str(f['model'])
            for n, v in zip(f['names'], f['vectors']):
                out.setdefault(str(n), {})[model] = np.asarray(v, float)
    return out


def axisforge(binary, *args, timeout=900):
    done = subprocess.run([str(binary), *map(str, args)], check=True, capture_output=True,
                          text=True, timeout=timeout)
    return json.loads(done.stdout)


def candidates(binary, project, body):
    """Every candidate the planner would generate, in generator order, unevaluated."""
    return [r['candidate'] for r in axisforge(binary, 'research-candidates', project, body)]


def solve(binary, project, body):
    """Examine every candidate: the feasible grasps and every refusal."""
    return axisforge(binary, 'plan', '--json', '--nr-grasps', 1_000_000, project, body)


def _vec(v):
    return np.array([v['x'], v['y'], v['z']], float)


def signatures(cands, low, high):
    """Signature arrays of grasp candidates relative to a part's bounding box."""
    low = np.asarray(low, float); high = np.asarray(high, float)
    extent = np.maximum(high-low, 1e-6)
    a = np.array([_vec(c['approach']) for c in cands]).reshape(-1, 3)
    k = np.array([_vec(c['closing']) for c in cands]).reshape(-1, 3)
    centre = np.array([_vec(c['center']) for c in cands]).reshape(-1, 3)
    for m in (a, k):
        m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    return dict(approach=a, closing=k, width=np.array([float(c['width']) for c in cands]),
                position=(centre-low)/extent, height=centre[:, 2]-low[2])


def affinity(query, good):
    """Gaussian similarity between each query candidate and each good grasp."""
    d = ((1-query['approach']@good['approach'].T)/SCALE['approach']
         + (1-(query['closing']@good['closing'].T)**2)/SCALE['closing']
         + ((query['width'][:, None]-good['width'][None])/SCALE['width'])**2
         + (((query['position'][:, None]-good['position'][None])**2).sum(-1))/SCALE['position']**2
         + ((query['height'][:, None]-good['height'][None])/SCALE['height'])**2)
    return np.exp(-d/2)


def library_entry(name, project, body, binary, descriptors=None):
    """Solve a part exhaustively and keep what transfer needs."""
    vertices, _ = part_mesh(project, body)
    low, high = vertices.min(0), vertices.max(0)
    report = solve(binary, project, body)
    good = [p['candidate'] for p in report['plans']]
    return dict(name=name, project=str(project), body=body, low=low.tolist(), high=high.tolist(),
                candidates_total=report['candidatesTotal'], feasible=len(good),
                good=[{k: c[k] for k in ('label', 'center', 'approach', 'closing', 'width')} for c in good],
                descriptors={k: np.asarray(v, float).tolist() for k, v in (descriptors or {}).items()},
                project_sha256=sha256_bytes(Path(project).read_bytes()))


def neighbours(method, query, library, k=NEIGHBOURS, seed=0, model=None):
    """(entry, weight) pairs from solved parts other than the query.

    visual: the k most similar descriptors, weighted by cosine;
    box: the k closest bounding boxes (log extents); random: k at random;
    pooled: every solved part equally; self: the query's own grasps (an
    upper bound, for evaluation only).
    """
    if method == 'self':
        return [(query, 1.)] if query['feasible'] else []
    pool = [e for e in library if e['name'] != query['name'] and e['feasible'] > 0]
    if not pool:
        return []
    if method == 'pooled':
        return [(e, 1.) for e in pool]
    if method == 'random':
        rng = np.random.default_rng(seed)
        return [(pool[i], 1.) for i in rng.choice(len(pool), size=min(k, len(pool)), replace=False)]
    if method == 'box':
        size = lambda e: np.log(np.maximum(np.subtract(e['high'], e['low']), 1e-6))
        d = np.array([np.linalg.norm(size(e)-size(query)) for e in pool])
        order = np.argsort(d, kind='stable')[:k]
        return [(pool[i], 1.) for i in order]
    if method == 'visual':
        q = np.asarray(query['descriptors'][model], float)
        sim = np.array([float(np.dot(q, e['descriptors'][model])) for e in pool])
        order = np.argsort(-sim, kind='stable')[:k]
        return [(pool[i], max(sim[i], 1e-6)) for i in order]
    raise ValueError(f'unknown neighbour method {method!r}')


def prior(cands, low, high, chosen, weight=PRIOR_WEIGHT):
    """The --prior document: one score in [0, 1] per candidate label."""
    if not chosen:
        return dict(candidates=[], weight=weight)
    query = signatures(cands, low, high)
    total = np.zeros(len(cands)); norm = 0.
    for entry, w in chosen:
        good = signatures(entry['good'], entry['low'], entry['high'])
        total += w*affinity(query, good).max(1)
        norm += w
    score = total/norm
    if score.max() > 0:
        score = score/score.max()
    best = {}
    for c, s in zip(cands, score):
        best[c['label']] = max(best.get(c['label'], 0.), float(s))
    return dict(candidates=[dict(label=l, score=round(s, 6)) for l, s in best.items()], weight=weight)


def plan_with(binary, project, body, prior_doc, nr_grasps, workdir):
    args = ['plan', '--json', '--nr-grasps', nr_grasps]
    if prior_doc is not None:
        path = Path(workdir)/'prior.json'
        path.write_text(json.dumps(prior_doc))
        args += ['--prior', path]
    report = axisforge(binary, *args, project, body)
    return dict(examined=report['candidatesExamined'], found=len(report['plans']),
                matched=report.get('priorMatched', 0), unmatched=report.get('priorUnmatched', 0))


def evaluate(library, binary, workdir, models=(), nr_grasps=(1, 3), random_seeds=10):
    """Leave one part out: its prior comes only from the other solved parts."""
    rows = []
    for query in library:
        if query['feasible'] == 0:
            continue
        cands = candidates(binary, query['project'], query['body'])
        n, f = len(cands), query['feasible']
        methods = [('default', None, None), ('pooled', None, None), ('box', None, None),
                   *[('visual', None, m) for m in models],
                   *[('random', s, None) for s in range(random_seeds)], ('self', None, None)]
        for method, seed, model in methods:
            doc = None
            if method != 'default':
                chosen = neighbours(method, query, library, seed=seed or 0, model=model)
                doc = prior(cands, query['low'], query['high'], chosen)
            for target in nr_grasps:
                if target > f:
                    continue
                result = plan_with(binary, query['project'], query['body'], doc, target, workdir)
                rows.append(dict(part=query['name'], method=method if model is None else f'visual:{model}',
                                 seed=seed, nr_grasps=target, candidates=n, feasible=f,
                                 random_expectation=target*(n+1)/(f+1),
                                 neighbours=[] if doc is None or method in ('pooled', 'self') else
                                 [e['name'] for e, _ in neighbours(method, query, library, seed=seed or 0, model=model)],
                                 **result))
    return rows


def summarise(rows):
    """Geometric-mean ratio to the default order and median examined, per method."""
    out = {}
    for target in sorted({r['nr_grasps'] for r in rows}):
        sub = [r for r in rows if r['nr_grasps'] == target]
        base = {r['part']: r['examined'] for r in sub if r['method'] == 'default'}
        methods = sorted({r['method'] for r in sub})
        table = {}
        for m in methods + ['random_expectation']:
            per_part = {}
            for r in sub:
                if m == 'random_expectation':
                    if r['method'] == 'default':
                        per_part[r['part']] = r['random_expectation']
                elif r['method'] == m:
                    per_part.setdefault(r['part'], []).append(r['examined'])
            values = {p: float(np.mean(v)) if isinstance(v, list) else float(v) for p, v in per_part.items()}
            ratio = [values[p]/base[p] for p in values if p in base]
            table[m] = dict(parts=len(values), median_examined=float(np.median(list(values.values()))),
                            mean_examined=float(np.mean(list(values.values()))),
                            geometric_mean_ratio_to_default=float(np.exp(np.mean(np.log(ratio)))),
                            fewer_than_default=sum(values[p] < base[p] for p in values if p in base),
                            per_part=values)
        out[str(target)] = table
    return out
