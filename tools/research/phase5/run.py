"""Run Phase V orderings from catalogues (no planner calls; see catalogue.py).

Development, on library parts only, grouped 5-fold cross-validation:
    python tools/research/phase5/run.py dev --store store.json --catalogues 80=cat80/ --out dev.json

Test, new parts against the frozen library, experiments A, B and C, per gripper:
    python tools/research/phase5/run.py test --store store.json --library-names lib.txt \
        --test-names test.txt --catalogues 40=cat40/ 80=cat80/ 120=cat120/ --experiments A B C --out test.json
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import catalogue, engine  # noqa: E402

STATE = {}


def load(store_path, catalogue_dirs, names):
    STATE['store'] = json.loads(Path(store_path).read_text())
    STATE['cats'] = {g: {n: catalogue.load(Path(d)/f'{n}.npz') for n in names} for g, d in catalogue_dirs.items()}


def job(task):
    """One (experiment, gripper, library) group: train once, evaluate its queries."""
    experiment, gripper, library_names, queries, tag, chase = task
    store, cats = STATE['store'], STATE['cats'][gripper]
    start = time.perf_counter()
    library = engine.entries(library_names, store, cats)
    learned, afat = engine.train(library, store, cats)
    train_s = time.perf_counter()-start
    rows = []
    for q in queries:
        t = time.perf_counter()
        for r in engine.evaluate_query(q, library, store, cats, learned, afat, chase):
            r.update(experiment=experiment, gripper=gripper, group=tag, library_size=len(library))
            rows.append(r)
        rows[-1]['query_s'] = round(time.perf_counter()-t, 3)
    return rows, dict(experiment=experiment, gripper=gripper, group=tag, library=len(library), queries=len(queries),
                      train_s=round(train_s, 2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['dev', 'test'])
    p.add_argument('--store', required=True)
    p.add_argument('--catalogues', nargs='+', required=True, help='GRIPPER_MM=DIR')
    p.add_argument('--library-names'); p.add_argument('--test-names')
    p.add_argument('--experiments', nargs='+', default=['B'])
    p.add_argument('--folds', type=int, default=5); p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--chase', nargs='*', default=[], help='methods whose first feasible grasps are kept '
                   '(experiment B, 80 mm) for the dynamic checks')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    dirs = {int(k): v for k, v in (x.split('=', 1) for x in a.catalogues)}
    store = json.loads(Path(a.store).read_text())
    tasks = []
    if a.mode == 'dev':
        names = sorted(store)
        order = np.random.default_rng(20261128).permutation(len(names))
        folds = [[names[i] for i in order[f::a.folds]] for f in range(a.folds)]
        for g in dirs:
            for f, held in enumerate(folds):
                train = [n for n in names if n not in held]
                queries = [n for n in held]
                tasks.append(('dev', g, train, queries, f'fold{f}', ()))
        load(a.store, dirs, names)
    else:
        library = Path(a.library_names).read_text().split()
        tests = Path(a.test_names).read_text().split()
        if set(library) & set(tests):
            raise SystemExit('a test part is in the library')
        load(a.store, dirs, library+tests)
        # Experiment C covers categories with at least five test parts.
        counts = {}
        for q in tests:
            counts[store[q]['category']] = counts.get(store[q]['category'], 0)+1
        for g in dirs:
            for e in a.experiments:
                groups = {}
                for q in tests:
                    if e == 'C' and (store[q]['category'] in ('(none)', 'YCB') or counts[store[q]['category']] < 5):
                        continue
                    key = tuple(engine.library_for(e, q, store, library))
                    groups.setdefault(key, []).append(q)
                for i, (lib, queries) in enumerate(groups.items()):
                    tag = f'{e}{i}' if e != 'C' else f"C:{store[queries[0]]['category']}"
                    # Split big groups so the four workers share the load.
                    chase = tuple(a.chase) if (e, g) == ('B', 80) else ()
                    for k in range(0, len(queries), 25):
                        tasks.append((e, g, list(lib), queries[k:k+25], tag, chase))
    start = time.perf_counter()
    rows, groups = [], []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for r, info in pool.map(job, tasks):
            rows += r; groups.append(info)
            print(info, flush=True)
    Path(a.out).write_text(json.dumps(dict(mode=a.mode, wall_s=round(time.perf_counter()-start, 1),
                                           groups=groups, rows=rows)) + '\n')
    print(len(rows), 'rows ->', a.out)


if __name__ == '__main__':
    main()
