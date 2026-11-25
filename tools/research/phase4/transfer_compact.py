"""Extract compact, auditable F1/F2 trajectories without rewriting originals."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    trajectories=[]
    for path in sorted((args.run/'runs').glob('*/*.json')):
        run=json.loads(path.read_text()); definition=run['definition'];rows=run['rows']
        if len(rows)!=definition['budget'] or run.get('pending') is not None:
            raise ValueError('incomplete run')
        compact=[]
        for row in rows:
            details=row.get('details',{})
            compact.append({key:row.get(key) for key in ('candidate_id','status','utility','cost','acquisition','prediction')}
                | {'cached':details.get('cached'),'epsilon':details.get('epsilon')})
        trajectories.append({'definition':definition,'source_sha256':digest(path),'rows':compact})
    result={'lock_sha256':digest(args.run/'lock.json'),'script_sha256':digest(Path(__file__)),
            'trajectories':trajectories}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')

if __name__=='__main__':main()
