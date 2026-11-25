"""Run the native planner with the Phase V priors and compare with the simulator.

    python tools/research/phase5/confirm_planner.py --store store.json --library-names lib.txt \
        --test-names test.txt --catalogues cat80/ --projects POOL --binary target/release/axisforge \
        --methods default box learned afat --out confirmation.json --jobs 4

Experiment B, 80 mm: the frozen library, the learned classifier and AFAT are
rebuilt exactly as in run.py (fixed seeds, one thread), each prior is written
as an `axisforge plan --prior` file, and the candidates the planner examines
to its first feasible grasp are compared with the simulated count.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import baselines as bl, catalogue, engine  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('store', 'library-names', 'test-names', 'catalogues', 'projects', 'binary', 'out'):
        p.add_argument('--'+k, required=True)
    p.add_argument('--methods', nargs='+', default=['default', 'box', 'learned', 'afat'])
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    store = json.loads(Path(a.store).read_text())
    lib_names = Path(a.library_names).read_text().split(); tests = Path(a.test_names).read_text().split()
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in lib_names+tests}
    library = engine.entries(lib_names, store, cats)
    learned, afat = engine.train(library, store, cats)
    tmp = Path(tempfile.mkdtemp())

    def prior_for(name, method):
        cat = cats[name]; low, high = store[name]['low'], store[name]['high']
        q = engine.query_entry(name, store, cats)
        if method == 'default':
            return None
        if method == 'box':
            return bl.transfer('box', q, library, cat)
        if method == 'learned':
            return bl.scores(learned, bl.features(cat, low, high))
        Xa, Xn = engine.stacked_features(q, library, cat, low, high)
        return bl.scores(afat[0] if method == 'afat' else afat[1], Xa if method == 'afat' else Xn)

    def one(task):
        name, method = task
        cat = cats[name]
        prior = prior_for(name, method)
        simulated, found = catalogue.examined(cat, prior)
        args = [a.binary, 'plan', '--json', '--nr-grasps', '1']
        if prior is not None:
            path = tmp/f'{name}-{method}.json'
            path.write_text(json.dumps(dict(weight=2.0, candidates=[
                dict(label=str(l), score=float(s)) for l, s in zip(cat['label'], prior)])))
            args += ['--prior', str(path)]
        report = json.loads(subprocess.run(args+[str(Path(a.projects)/store[name]['project']), name],
                                           check=True, capture_output=True, text=True).stdout)
        return dict(part=name, method=method, simulated=simulated, planner=report['candidatesExamined'],
                    found=found, planner_found=len(report['plans']) > 0)

    tasks = [(n, m) for n in tests for m in a.methods]
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(one, tasks))
    mismatches = [r for r in rows if (r['simulated'], r['found']) != (r['planner'], r['planner_found'])]
    Path(a.out).write_text(json.dumps(dict(runs=len(rows), mismatches=len(mismatches), examples=mismatches[:20],
                                           rows=rows), indent=1) + '\n')
    print(len(rows), 'planner runs,', len(mismatches), 'differ from the simulator')


if __name__ == '__main__':
    main()
