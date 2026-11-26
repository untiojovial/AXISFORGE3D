"""Peak resident memory of the planner process per order mode (the integrated benchmark's rusage was invalid).

    python tools/research/phase7_extension/memory_benchmark.py --binary target/release/axisforge \
        --models research/phase7_extension/native/learned_order_phase7.json --bench-raw BENCH_RAW_DIR \
        --dev-store store.json --dev-projects POOL --egad-store CONF/store.json --egad-projects EGAD_POOL \
        --objects 24 --out research/phase7_extension/results/memory_benchmark.json

Why a separate run: a child spawned from a Python process inherits, on Linux, the parent's peak RSS in
ru_maxrss across exec. The integrated benchmark therefore recorded the Python parent's peak (about 170 MiB)
for every mode, and those values are discarded. Here the planner reports its own peak (VmHWM from
/proc/self/status, field peakResidentKiB of plan --json). The run uses a seeded subset of the benchmark
objects, one cold call per mode, in a seeded random order.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess

import numpy as np

MODES = ('M0', 'm1', 'm4', 'm9', 'joint', 'hybrid')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('binary', 'models', 'bench-raw', 'dev-store', 'dev-projects', 'egad-store', 'egad-projects', 'out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--objects', type=int, default=24); p.add_argument('--seed', type=int, default=20261214)
    a = p.parse_args()
    ds, es = json.loads(Path(a.dev_store).read_text()), json.loads(Path(a.egad_store).read_text())
    names = sorted(f.stem for f in Path(a.bench_raw).glob('*.json'))
    rnd = random.Random(a.seed)
    names = rnd.sample(names, min(a.objects, len(names)))
    rows = {}
    for n in names:
        proj = Path(a.egad_projects)/es[n]['project'] if n.startswith('egad_') else Path(a.dev_projects)/ds[n]['project']
        modes = list(MODES); rnd.shuffle(modes)
        rows[n] = {}
        for m in modes:
            args = [a.binary, 'plan', '--json', '--nr-grasps', '1', str(proj), n]
            if m != 'M0':
                args[3:3] = ['--learned', a.models, '--order', m]
            rep = json.loads(subprocess.run(['taskset', '-c', '3', *args], check=True, capture_output=True, text=True,
                                            env={'RAYON_NUM_THREADS': '1', 'PATH': '/usr/bin:/bin'}).stdout)
            rows[n][m] = rep['peakResidentKiB']
        print(n, rows[n], flush=True)
    summ = {m: dict(median_mib=float(np.median([r[m] for r in rows.values()])/1024), max_mib=float(max(r[m] for r in rows.values())/1024),
                    median_minus_M0_mib=float(np.median([r[m]-r['M0'] for r in rows.values()])/1024)) for m in MODES}
    model = Path(a.models)
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), objects=len(rows), model_file_bytes=model.stat().st_size,
               summary=summ, per_object_kib=rows)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
