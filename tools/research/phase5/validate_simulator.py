"""Check the offline ordering simulator against recorded native planner runs.

For every row of the given evaluation files the prior is rebuilt exactly as
the evaluation built it (same library, method, model and seed), the order is
simulated from the part's catalogue, and the number of candidates examined is
compared with what the planner reported.

    python tools/research/phase5/validate_simulator.py --catalogues cat80/ \
        --run EVALUATION.json LIBRARY.json[.gz] [--run ...] --out validation.json
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase4 import visual_prior as vp  # noqa: E402
from research_grasping.phase5 import catalogue  # noqa: E402


def load_json(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        return json.load(f)


def check(evaluation, library, catalogues):
    lib = load_json(library)['parts']
    by_name = {e['name']: e for e in lib}
    rows = load_json(evaluation)['rows']
    cats, cands, mismatches, checked = {}, {}, [], 0
    for row in rows:
        name = row['part']
        if name not in cats:
            cats[name] = catalogue.load(Path(catalogues)/f'{name}.npz')
            cands[name] = catalogue.candidates(cats[name])
        query = by_name[name]
        method, _, model = row['method'].partition(':')
        scores = None
        if method != 'default':
            chosen = vp.neighbours(method, query, lib, seed=row['seed'] or 0, model=model or None)
            scores = catalogue.prior_doc_scores(vp.prior(cands[name], query['low'], query['high'], chosen))
        got, _ = catalogue.examined(cats[name], scores, target=row['nr_grasps'])
        checked += 1
        if got != row['examined']:
            mismatches.append(dict(part=name, method=row['method'], seed=row['seed'], target=row['nr_grasps'],
                                   planner=row['examined'], simulated=got))
    return dict(evaluation=str(evaluation), rows=checked, mismatches=len(mismatches), examples=mismatches[:20])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--catalogues', required=True)
    p.add_argument('--run', nargs=2, action='append', required=True, metavar=('EVALUATION', 'LIBRARY'))
    p.add_argument('--out', required=True)
    a = p.parse_args()
    results = [check(e, l, a.catalogues) for e, l in a.run]
    Path(a.out).write_text(json.dumps(dict(runs=results, rows=sum(r['rows'] for r in results),
                                           mismatches=sum(r['mismatches'] for r in results)), indent=1) + '\n')
    for r in results:
        print(f"{r['evaluation']}: {r['rows']} rows, {r['mismatches']} mismatches")


if __name__ == '__main__':
    main()
