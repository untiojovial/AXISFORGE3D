"""Run three disjoint, resumable shards and validate the final union."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from research_grasping.phase3.experiment import load, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.directory.mkdir(parents=True,exist_ok=True)
    workers=[]
    for index in range(3):
        shard=a.directory/f'shard_{index}.json'
        log=(a.directory/f'shard_{index}.log').open('a')
        args=[sys.executable,'tools/research/phase4/physical_pilot.py','--output',str(shard),
              '--shard-count','3','--shard-index',str(index)]
        workers.append((subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT),log,shard))
    status=[]
    for process,log,shard in workers:
        status.append(process.wait());log.close()
    if any(status):raise RuntimeError(f'worker failures: {status}')
    shards=[load(item[2]) for item in workers]
    keys=('protocol_sha256','script_sha256','simulation_sha256','mujoco_version','shard_count')
    if any(any(shard[k]!=shards[0][k] for k in keys) for shard in shards):raise ValueError('provenance mismatch')
    protocol=load('research/phase4/robotics/physical_protocol.json')
    expected={(obj['object_id'],cid) for obj in protocol['objects'] for cid in obj['candidate_ids']}
    rows=[row for shard in shards for row in shard['rows']]
    observed=[(row['object_id'],row['candidate_id']) for row in rows]
    if len(observed)!=len(expected) or set(observed)!=expected:raise ValueError('shard union mismatch')
    order={(obj['object_id'],cid):i for i,(obj,cid) in enumerate((o,c) for o in protocol['objects'] for c in o['candidate_ids'])}
    rows.sort(key=lambda row:order[(row['object_id'],row['candidate_id'])])
    save(a.output,{**{k:shards[0][k] for k in keys},'rows':rows})


if __name__=='__main__':main()
