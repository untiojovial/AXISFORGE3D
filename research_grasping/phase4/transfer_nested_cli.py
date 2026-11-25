"""Run predeclared nested diagnostics using only locked train histories."""
import argparse
from pathlib import Path
from research_grasping.phase3.experiment import DevelopmentData, digest, fit_history, fingerprint, load, save
from research_grasping.phase4.transfer_pilot import make_history
from research_grasping.phase4.transfer_nested import GRID, nested


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for v in ('data','protocol','catalogues','binary','archive','history-lock','out'):
        p.add_argument('--'+v,type=Path,required=True)
    p.add_argument('--prefix',type=int,default=50);a=p.parse_args()
    expected=load(a.history_lock);data=DevelopmentData(a.protocol/'manifest.prepilot.json',load(a.protocol/'lock.json'),a.data)
    archive=[]
    for path in sorted(a.catalogues.glob('*.json')):
        raw=load(path)
        # This diagnostic deliberately never opens validation catalogues via guard.
        if data.splits.get(raw['object_id'])!='train':continue
        c=data.catalogue(path,a.binary);file=a.archive/(c['object_id']+'.json')
        if digest(file)!=expected.get(c['object_id']):raise ValueError('history checksum mismatch')
        log=load(file)
        if len(log['rows'])<a.prefix:raise ValueError('incomplete train history')
        log=dict(log,selected=log['selected'][:a.prefix],rows=log['rows'][:a.prefix])
        archive.append((c,log))
    if len(archive)!=16 or set(expected)!={c['object_id'] for c,_ in archive}:raise ValueError('incomplete roster')
    history,scaler=fit_history(data,archive,'native-f12');tasks=make_history(archive,history)
    lock={'history_lock_sha256':digest(a.history_lock),'manifest_sha256':data.manifest_digest,
          'binary_sha256':digest(a.binary),'catalogues':{c['object_id']:fingerprint(c) for c,_ in archive},
          'prefix':a.prefix,'grid':[list(x) for x in GRID],
          'code':{str(file):digest(file) for file in [Path(__file__),Path('research_grasping/phase4/transfer_nested.py'),
              Path('research_grasping/phase4/feasibility_policy_v2b.py'),Path('research_grasping/phase4/feasibility_policy.py')]}}
    lock_path=a.out.with_suffix('.lock.json')
    if lock_path.exists():
        if load(lock_path)!=lock:raise ValueError('frozen diagnostic definition changed')
    else:save(lock_path,lock)
    result=nested(tasks);result['lock_sha256']=digest(lock_path);save(a.out,result)

if __name__=='__main__':main()
