"""CLI for development-only native screening and chronological transfer pilots."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import subprocess
import time

from .experiment import (DevelopmentData, digest, fingerprint, load, run_policy,
                         save, viewer_report)
from .native_catalog import evaluate_candidate


def native_observation(binary, project, body, candidate_id):
    start = time.monotonic()
    failures = []
    for _ in range(3):
        try:
            reply = evaluate_candidate(binary, project, body, int(candidate_id))
            record = reply['records'][0]
            q = record['quality']
            if q['measure'] != 'wrench_space' or not 0 <= q['score'] <= 1:
                raise ValueError('unexpected native measure')
            feasible = record['plan'] is not None
            return dict(status='ok' if feasible else 'physical_failure',
                        utility=q['score'] if feasible else 0., cost=time.monotonic()-start,
                        details=dict(feasible=feasible, rejection=record['rejection'],
                                     epsilon=q['quality'], score=q['score'], closure=q['forceClosure'],
                                     quality_elapsed_s=record['qualityElapsedS'],
                                     attempt_elapsed_s=record['attemptElapsedS'],
                                     f1_attempts=1, f2_evaluations=1, f3_executions=0, errors=failures))
        except (ValueError, OSError, subprocess.SubprocessError, KeyError) as exc:
            failures.append(str(exc))
    return dict(status='infrastructure_failure', utility=None, cost=time.monotonic()-start,
                details=dict(errors=failures, f3_executions=0))


def cached_observation(cache_dir, binary, project, body, signature, candidate, fidelity):
    # This function runs in a worker process. Labels reach policy only via the
    # requested id, including when a deterministic native measurement is reused.
    from fcntl import LOCK_EX, flock
    filename = Path(cache_dir) / signature / fidelity / (str(int(candidate)) + '.json')
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.with_suffix('.lock').open('a') as lock:
        flock(lock, LOCK_EX)
        if filename.exists():
            result = load(filename)
            result['details']['cached'] = True
            return result
        if fidelity != 'native-f12':
            raise ValueError('unsupported fidelity')
        result = native_observation(binary, project, body, candidate)
        result['details']['cached'] = False
        if result['status'] != 'infrastructure_failure':
            save(filename, result)
        return result


def observer(executor, cache_dir, binary, catalog, fidelity):
    signature = fingerprint([catalog['metadata']['project_sha256'],catalog['metadata']['binary_sha256'],
                             catalog['metadata']['processed_sha256'],catalog['body']])
    def observe(candidate):
        if candidate not in catalog['ids']:
            raise ValueError('unknown candidate')
        return executor.submit(cached_observation, cache_dir, binary, catalog['project'],
                               catalog['body'], signature, candidate, fidelity).result()
    return observe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogues', type=Path, required=True, nargs='+')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True,
                        help='directory containing preregistered lock.json and manifest.prepilot.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=10)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 29, 47])
    parser.add_argument('--methods', nargs='+', default=['random','sobol_projected','target_local','rgpe','equal','adaptive'])
    args = parser.parse_args(argv)
    data = DevelopmentData(args.protocol/'manifest.prepilot.json',load(args.protocol/'lock.json'),args.data)
    catalogues = [data.catalogue(path, args.binary) for path in args.catalogues]
    if len({c['object_id'] for c in catalogues}) != len(catalogues):
        parser.error('duplicate catalogues')
    if min(len(c['ids']) for c in catalogues) < args.budget or args.budget < 2:
        parser.error('budget outside catalogues')
    train = [c for c in catalogues if c['split'] == 'train']
    targets = [c for c in catalogues if c['split'] == 'validation']
    if not train or not targets:
        parser.error('train and validation catalogues required')
    fidelity = 'native-f12'
    args.out.mkdir(parents=True, exist_ok=True)
    # A CLI invocation has exclusive access to its experiment directory.
    from fcntl import flock, LOCK_EX
    with (args.out/'.pilot.lock').open('a') as lock, ProcessPoolExecutor(max_workers=1) as worker:
        flock(lock, LOCK_EX)
        prelock = dict(catalogues=[fingerprint(c) for c in catalogues], manifest=data.manifest_digest,
                       budget=args.budget, seeds=args.seeds, methods=args.methods, fidelity=fidelity)
        selection_file = args.out/'selection-lock.json'
        if selection_file.exists() and load(selection_file) != prelock:
            raise ValueError('pre-query design changed')
        save(selection_file, prelock)
        archive = []
        for catalog in train:
            path = args.out/'history'/f"{catalog['object_id']}.json"
            signature = fingerprint(catalog)
            selected = np.random.default_rng(2025).choice(catalog['ids'], args.budget, replace=False).tolist()
            if path.exists():
                log = load(path)
                if (log['catalogue_digest'] != signature or log['fidelity'] != fidelity
                        or log['selected'] != selected):
                    raise ValueError('training archive definition changed')
            else:
                log = dict(object_id=catalog['object_id'], fidelity=fidelity, catalogue_digest=signature,
                           selected=selected, rows=[])
            observe = observer(worker, args.out/'cache',args.binary,catalog,fidelity)
            for candidate in selected[len(log['rows']):]:
                log['rows'].append(dict(candidate_id=candidate, **observe(candidate)))
                save(path, log)
            if 'completed_at' not in log:
                log['completed_at'] = time.time()
                save(path, log)
            archive.append((catalog, log))
        runfile = args.out/'archive-lock.json'
        archive_lock = {c['object_id']:digest(args.out/'history'/f"{c['object_id']}.json") for c in train}
        config = dict(budget=args.budget, seeds=args.seeds, methods=args.methods, fidelity=fidelity,
                      manifests=data.manifest_digest, catalogues=[fingerprint(c) for c in catalogues])
        if runfile.exists():
            frozen = load(runfile)
            if frozen['archive'] != archive_lock or frozen['config'] != config:
                raise ValueError('frozen history or configuration changed')
        else:
            frozen = dict(archive=archive_lock, config=config, target_time=time.time())
            save(runfile, frozen)
        runs = []
        for catalog in targets:
            observe = observer(worker, args.out/'cache',args.binary,catalog,fidelity)
            for seed in args.seeds:
                for method in args.methods:
                    path = args.out/'runs'/catalog['object_id']/f'{seed}-{method}.json'
                    runs.append(run_policy(data, catalog, archive, method=method, seed=seed,
                             budget=args.budget, initial_count=2, fidelity=fidelity,
                             target_time=frozen['target_time'], observe=observe, checkpoint=path))
        save(args.out/'viewer.json', viewer_report(targets, runs))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
