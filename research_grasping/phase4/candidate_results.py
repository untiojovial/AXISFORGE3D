"""Summarize and archive both prespecified selection strategies on development objects."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from research_grasping.phase3.experiment import digest, load, save
from .catalogue_study import report


def summarize(work, destination):
    selection = load(work/'selection.json')
    per_object = report(work)
    expected = selection['objects']
    records = {}
    unique = set()
    fresh = []
    summaries = []
    for split, category in [('train','history'), ('validation','validation_panel')]:
        for method in ('uniform','stratified'):
            selected = {n for n,v in expected.items() if v['split']==split}
            key = f'{category}/{method}'
            records[key] = {}
            for name in sorted(selected):
                path = work/key/(name+'.json')
                item = load(path)
                ids = expected[name]['sampling'][method]
                if (item['selection_sha256'] != digest(work/'selection.json') or item['object_id'] != name
                        or [r['candidate_id'] for r in item['rows']] != ids):
                    raise ValueError(f'Mismatched native observations for {name}')
                records[key][name] = item
                unique.update((name,r['candidate_id']) for r in item['rows'])
                fresh.extend((name,r['candidate_id'],r['cost']) for r in item['rows'] if not r['details']['cached'])
            for budget in (10,25,50):
                rows = [r for r in per_object if (r['category'],r['scheme'],r['budget'])==(category,method,budget)]
                if not rows: continue
                counts = Counter()
                for r in rows: counts.update(r['causes'])
                summaries.append(dict(split=split,method=method,budget=budget,observations=budget*len(rows),
                                      objects=len(rows),feasible=counts['ok'],by_first_outcome=dict(counts),
                                      objects_with_feasible=sum(bool(r['feasible']) for r in rows),
                                      nominal_seconds=sum(r['nominal_seconds'] for r in rows),
                                      fresh_seconds=sum(r['fresh_seconds'] for r in rows)))
    destination.mkdir(parents=True,exist_ok=True)
    archive = destination/'native_observations.json.gz'
    archive.write_bytes(gzip.compress(json.dumps(records,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    result=dict(selection_sha256=digest(work/'selection.json'),observations_sha256=digest(archive),
        aggregate=summaries,by_object=per_object,unique_object_candidates=len(unique),fresh_observations=len(fresh),
        fresh_distinct=len({(n,i) for n,i,_ in fresh}),fresh_oracle_seconds=sum(t for _,_,t in fresh))
    save(destination/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    summarize(args.work,args.out)
