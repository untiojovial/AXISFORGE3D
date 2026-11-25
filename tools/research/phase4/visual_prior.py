"""Visual grasp transfer: render, encode, build a library, write a prior, evaluate.

  views     render each part's 16 views into a ZIP (no model needed)
  encode    turn a views ZIP into descriptors with a frozen model (here or in Colab)
  library   solve parts exhaustively and store their feasible grasps and descriptors
  prior     write the `axisforge plan --prior` file for a new part
  evaluate  leave one part out over a library and compare orderings

Example:
  python tools/research/phase4/visual_prior.py views --part Banana cells/Banana.axisforge --out views.zip
  python tools/research/phase4/visual_prior.py encode --views views.zip --model dinov2 \\
      --checkpoint dinov2-small --revision REV --weights-sha256 W --config-sha256 C \\
      --processor-sha256 P --out dinov2.npz
  python tools/research/phase4/visual_prior.py library --binary target/release/axisforge \\
      --part Banana cells/Banana.axisforge ... --descriptors dinov2.npz --out library.json
  python tools/research/phase4/visual_prior.py prior --library library.json \\
      --binary target/release/axisforge --part Pear cells/Pear.axisforge \\
      --descriptors dinov2.npz --model dinov2 --out pear-prior.json
  target/release/axisforge plan cells/Pear.axisforge Pear --prior pear-prior.json
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase4 import visual_prior as vp  # noqa: E402


def encode(a):
    from research_grasping.phase4.frozen_models import load_frozen
    enc = load_frozen(a.model, a.checkpoint, a.config_sha256, a.weights_sha256, a.processor_sha256,
                      device=a.device, model_revision=a.revision)
    names = vp.write_descriptors(enc, a.views, a.out, a.label or a.model)
    print(f'{len(names)} descriptors from {a.label or a.model} -> {a.out}')


def library(a):
    desc = vp.read_descriptors(a.descriptors)
    entries = []
    for name, project in a.part:
        start = time.perf_counter()
        e = vp.library_entry(name, project, name, a.binary, desc.get(name))
        entries.append(e)
        print(name, f'{e["feasible"]}/{e["candidates_total"]} feasible', f'{time.perf_counter()-start:.1f} s', flush=True)
    Path(a.out).write_text(json.dumps(dict(format='axisforge-visual-library', version=1, parts=entries)) + '\n')


def load_library(path):
    doc = json.loads(Path(path).read_text())
    if doc.get('format') != 'axisforge-visual-library':
        raise ValueError('not a visual grasp library')
    return doc['parts']


def prior(a):
    lib = load_library(a.library)
    (name, project), = a.part
    vertices, _ = vp.part_mesh(project, name)
    desc = vp.read_descriptors(a.descriptors).get(name, {})
    if a.method == 'visual' and a.model not in desc:
        raise SystemExit(f'no {a.model} descriptor for {name}: run views + encode first')
    query = dict(name=name, low=vertices.min(0).tolist(), high=vertices.max(0).tolist(),
                 descriptors=desc, feasible=0)
    chosen = vp.neighbours(a.method, query, lib, k=a.neighbours, model=a.model)
    doc = vp.prior(vp.candidates(a.binary, project, name), query['low'], query['high'], chosen, a.weight)
    doc['neighbours'] = [dict(part=e['name'], weight=round(float(w), 6)) for e, w in chosen]
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
    print('neighbours:', ', '.join(f'{e["name"]} ({w:.3f})' for e, w in chosen) or 'none')


def evaluate(a):
    lib = load_library(a.library)
    for e in lib:
        project = Path(e['project'])
        if not project.is_absolute():
            e['project'] = str(Path(a.projects or Path(a.library).parent)/project)
    with tempfile.TemporaryDirectory() as tmp:
        start = time.perf_counter()
        queries = None if not a.query else set(a.query)
        rows = vp.evaluate(lib, a.binary, tmp, models=a.model or (), random_seeds=a.random_seeds,
                           model_methods=tuple(a.methods), queries=queries, jobs=a.jobs)
    result = dict(protocol=dict(neighbours=vp.NEIGHBOURS, prior_weight=vp.PRIOR_WEIGHT,
                                signature_scale={k: float(v) for k, v in vp.SCALE.items()},
                                random_seeds=a.random_seeds, leave_one_out=True, model_methods=a.methods,
                                queries=sorted(queries) if queries else 'every solved part',
                                library_sha256=vp.sha256_bytes(Path(a.library).read_bytes())),
                  wall_seconds=round(time.perf_counter()-start, 1), summary=vp.summarise(rows), rows=rows)
    Path(a.out).write_text(json.dumps(result, indent=1) + '\n')
    for target, table in result['summary'].items():
        print(f'-- examined until {target} feasible grasp(s)')
        for m, s in table.items():
            print(f'{m:22s} median {s["median_examined"]:7.1f}  geo-mean ratio to default '
                  f'{s["geometric_mean_ratio_to_default"]:.3f}  fewer than default {s["fewer_than_default"]}/{s["parts"]}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    def parts(q, many=True):
        q.add_argument('--part', nargs=2, action='append', required=True, metavar=('NAME', 'PROJECT'),
                       help='body name and the project that holds it' + (' (repeatable)' if many else ''))


    v = sub.add_parser('views'); parts(v); v.add_argument('--out', required=True)
    e = sub.add_parser('encode'); e.add_argument('--views', required=True)
    e.add_argument('--model', required=True, choices=['dinov2', 'dinov3', 'dinov3-timm', 'openclip'])
    e.add_argument('--label', help='descriptor name, the model kind when omitted (e.g. a control)')
    for k in ('checkpoint', 'revision', 'weights-sha256', 'config-sha256', 'processor-sha256'):
        e.add_argument('--' + k, required=k in ('checkpoint', 'weights-sha256'))
    e.add_argument('--device', default='cpu'); e.add_argument('--out', required=True)
    lb = sub.add_parser('library'); lb.add_argument('--binary', required=True); parts(lb)
    lb.add_argument('--descriptors', action='append'); lb.add_argument('--out', required=True)
    pr = sub.add_parser('prior'); pr.add_argument('--library', required=True); pr.add_argument('--binary', required=True)
    parts(pr, many=False); pr.add_argument('--descriptors', action='append')
    pr.add_argument('--method', default='visual', choices=['visual', 'fusion', 'gate', 'box', 'pooled'])
    pr.add_argument('--model', default='dinov2', help='descriptor name')
    pr.add_argument('--neighbours', type=int, default=vp.NEIGHBOURS)
    pr.add_argument('--weight', type=float, default=vp.PRIOR_WEIGHT); pr.add_argument('--out', required=True)
    ev = sub.add_parser('evaluate'); ev.add_argument('--library', required=True); ev.add_argument('--binary', required=True)
    ev.add_argument('--model', action='append', help='descriptor name to evaluate (repeatable)')
    ev.add_argument('--projects', help='folder relative project paths in the library start from '
                    '(the library\'s own folder when omitted)')
    ev.add_argument('--methods', nargs='+', default=['visual'], choices=['visual', 'fusion', 'gate'],
                    help='how descriptors choose neighbours (each with every --model)')
    ev.add_argument('--query', action='append', help='evaluate only this part (repeatable); every solved part when omitted')
    ev.add_argument('--jobs', type=int, default=1, help='parts evaluated at once')
    ev.add_argument('--random-seeds', type=int, default=10); ev.add_argument('--out', required=True)
    a = p.parse_args()
    if a.command == 'prior' and len(a.part) != 1:
        p.error('prior takes exactly one --part')
    dict(views=lambda: vp.write_views(a.part, a.out, log=lambda *x: print(*x, flush=True)), encode=lambda: encode(a), library=lambda: library(a),
         prior=lambda: prior(a), evaluate=lambda: evaluate(a))[a.command]()


if __name__ == '__main__':
    main()
