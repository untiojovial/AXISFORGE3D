"""Publish compact paired-method summaries and raw observations from completed runs."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics

from .experiment import load
from .summarize_pilot import summarize_run


def export(run_dir, output):
    runs=[load(p) for p in sorted((run_dir/'runs').glob('*/*.json'))]
    history=[load(p) for p in sorted((run_dir/'history').glob('*.json'))]
    if not runs or not history:raise ValueError('expected completed runs and historical observations')
    summaries=[summarize_run(r) for r in runs]
    if any(r is None for r in summaries):raise ValueError('incomplete trajectory')
    output.mkdir(parents=True,exist_ok=True)
    observations={k:load(run_dir/(k+'.json')) for k in ('selection-lock','archive-lock')}
    observations.update(history=history,runs=runs)
    (output/'observations.json.gz').write_bytes(gzip.compress(json.dumps(observations,separators=(',',':'),
                                               allow_nan=False).encode(),mtime=0))
    groups={}
    for r in summaries:
        groups.setdefault((r['object_id'],r['method']),[]).append(r)
    with (output/'paired.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['object','method','seeds','budget','mean_auc','mean_final','f1_planned','f1_rejected','infrastructure_errors'])
        for (name,method),rows in sorted(groups.items()):
            writer.writerow([name,method,len(rows),rows[0]['budget'],statistics.mean(r['auc'] for r in rows),
                         statistics.mean(r['last'] for r in rows),
                         *[sum(r['statuses'][k] for r in rows) for k in ('ok','physical_failure','infrastructure_failure')]])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();export(args.run,args.out)
