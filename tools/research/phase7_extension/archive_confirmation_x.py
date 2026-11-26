"""Archive the per-object outputs of the second confirmation and write its execution record.

    python tools/research/phase7_extension/archive_confirmation_x.py --pool NEW_POOL --out-dir OUT --logs LOG_DIR \
        --dest research/phase7_extension/results/confirmation_x

Copies, without changing any value:

- status.json, the catalogue status of each object;
- built.json, the preparation rows;
- orders.json.gz, the per-object orders of every method;
- trials.json.gz, every simulated trial;
- timing.json.gz, every cold-process run;
- gpd_first_ranked.json, GPD's scores of the candidates GPD ranked first.

It writes execution_record.json from the run logs: stage start times, freeze checks, container restarts, the
load around timing and the paused jobs. A SHA-256 digest of every archived file goes into MANIFEST.json. The
catalogues and the full GPD score files are large and stay local; they are regenerable with the stage commands.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gz_dir(src, dest):
    docs = {f.stem: json.loads(f.read_text()) for f in sorted(Path(src).glob('*.json'))}
    with gzip.open(dest, 'wt') as fh:
        json.dump(docs, fh, sort_keys=True)
    return len(docs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('pool', 'out-dir', 'logs', 'dest'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    out, dest, logs = Path(a.out_dir), Path(a.dest), Path(a.logs)
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'status.json').write_text((out/'status.json').read_text())
    (dest/'built.json').write_text((Path(a.pool)/'built.json').read_text())
    counts = {name: gz_dir(out/name, dest/f'{name}.json.gz') for name in ('orders', 'trials', 'timing')}
    orders = json.loads(gzip.open(dest/'orders.json.gz', 'rt').read())
    first = {}
    for n, r in orders.items():
        g = r['methods'].get('GPD+gate', {})
        if g.get('first'):
            first[n] = dict(label=g['first'], nf=g.get('nf'))
    (dest/'gpd_first_ranked.json').write_text(json.dumps(first, indent=0, sort_keys=True) + '\n')
    stages = []
    for line in (logs/'confirm_run.log').read_text().splitlines() + (logs/'timing_run.log').read_text().splitlines():
        m = re.match(r'== (.+?) (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)$', line)
        if m:
            stages.append(dict(stage=m.group(1), utc=m.group(2)))
    tl = (logs/'timing_run.log').read_text()
    rec = dict(
        note=__doc__.strip().split('\n\n', 1)[0],
        stages=stages,
        freeze_checks=sum(1 for line in ((logs/'confirm_run.log').read_text() + tl).splitlines() if line.startswith('freeze ok')),
        restarts=[line for line in (logs/'confirm_restarts.txt').read_text().splitlines() if line.strip()] if (logs/'confirm_restarts.txt').exists() else [],
        timing=dict(paused=[line for line in tl.splitlines() if line.startswith('stopped')],
                    load_before=next((line for line in tl.splitlines() if line.startswith('load before')), None),
                    load_after=next((line for line in tl.splitlines() if line.startswith('load after')), None),
                    pinned_cpu=3, threads='OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1'),
        counts=counts)
    (dest/'execution_record.json').write_text(json.dumps(rec, indent=1) + '\n')
    for f in ('confirm_run.log', 'timing_run.log'):
        (dest/f).write_text((logs/f).read_text())
    files = sorted(f for f in dest.iterdir() if f.name != 'MANIFEST.json')
    (dest/'MANIFEST.json').write_text(json.dumps({f.name: dict(sha256=sha(f), bytes=f.stat().st_size) for f in files}, indent=1) + '\n')
    print(json.dumps(dict(counts=counts, stages=len(stages), freeze_checks=rec['freeze_checks'], restarts=len(rec['restarts'])), indent=1))


if __name__ == '__main__':
    main()
