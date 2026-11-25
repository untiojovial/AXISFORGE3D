"""Recount the immutable Phase III public traces using only the standard library."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[3]


def read(path):
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as stream:return json.load(stream)
    return json.loads(path.read_text())


def audit():
    base=ROOT/'research/phase3';primary=base/'bo/results/development/observations.json.gz'
    archive=read(primary)
    reasons=collections.Counter();unique={};repetitions=collections.Counter()
    for definition,rows in [(h,h['rows']) for h in archive['history']]+[(r['definition'],r['rows']) for r in archive['runs']]:
        name=definition['object_id']
        for item in rows:
            key=(name,item['candidate_id'])
            detail=item['details'].get('rejection') or {}
            value=(item['status'],item['utility'],detail.get('reason',item['status']))
            if key in unique and value!=unique[key]:raise ValueError('inconsistent oracle')
            unique[key]=value
            repetitions[key]+=1
    reasons.update(v[2] for v in unique.values())
    historical={h['object_id'] for h in archive['history']}
    valid={r['definition']['object_id'] for r in archive['runs']}
    manifest=read(base/'pilot/manifest.prepilot.json')['objects']
    if historical!={o['object_id'] for o in manifest if o['split']=='train'} or valid!={o['object_id'] for o in manifest if o['split']=='validation'}:
        raise ValueError('unexpected evaluated population')
    dyn=read(base/'robot/contact_probe_development.json')
    measurements=[(o['object_id'],m) for o in dyn['objects'] for m in o['measurements']]
    local=read(base/'bo/results/functional/observations.json.gz')
    auc=collections.defaultdict(list)
    for trial in local['runs']:
        best=0.;curve=[]
        for r in trial['rows']:best=max(best,r['utility']);curve.append(best)
        auc[trial['definition']['method']].append(statistics.mean(curve))
    return dict(reference_commit='bc94fda049f47a7c6a2d26a0db08f5b77e599e36',
      input_sha256=hashlib.sha256(primary.read_bytes()).hexdigest(),
      splits=dict(collections.Counter(o['split'] for o in manifest)),
      history_attempts=sum(len(h['rows']) for h in archive['history']),history_success=sum(v[0]=='ok' for (n,_),v in unique.items() if n in historical),
      validation_nominal=sum(len(r['rows']) for r in archive['runs']),validation_nominal_success=sum(item['status']=='ok' for r in archive['runs'] for item in r['rows']),
      validation_unique=sum(n in valid for n,_ in unique),validation_unique_success=sum(v[0]=='ok' for (n,_),v in unique.items() if n in valid),
      rejection_counts=dict(reasons),unique_total=len(unique),zero_success_validation=sorted(n for n in valid if not any(v[0]=='ok' for (k,_),v in unique.items() if k==n)),
      dynamic_selected=len(measurements),dynamic_sustained=sum(m['probe']['sustained'] for _,m in measurements),
      dynamic_sustained_objects=sorted({n for n,m in measurements if m['probe']['sustained']}),
      functional_auc={k:statistics.mean(v) for k,v in auc.items()})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path)
    out=p.parse_args().out
    report=json.dumps(audit(),indent=2)+'\n'
    if out:
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(report)
    else:
        print(report,end='')
