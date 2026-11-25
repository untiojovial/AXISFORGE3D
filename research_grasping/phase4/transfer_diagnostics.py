"""Replay prior ACT gate against no-evidence on the same observed prefixes."""
import argparse
import json
from pathlib import Path
import numpy as np
from research_grasping.phase3.experiment import DevelopmentData, fit_history, fingerprint, load, save
from research_grasping.phase3.transfer_policy import CataloguePolicy


def compare(data, catalog, archive, run):
    definition=run['definition']; history,scaler=fit_history(data,archive,definition['fidelity'])
    if definition['catalogue_digest'] != fingerprint(catalog) or definition['archive_digest'] != scaler.training_digest:
        raise ValueError('mismatched catalog or history')
    common=dict(ids=catalog['ids'],features=scaler.transform(catalog['features']),context=catalog['context'],
        history=history,manifest_splits=data.splits,manifest_digest=fingerprint(data.splits),
        target_id=catalog['object_id'],scaler=scaler,target_time=definition['target_time'],
        budget=definition['budget'],seed=definition['seed'],initial_ids=definition['initial'])
    adaptive=CataloguePolicy(method='adaptive',**common)
    fixed=CataloguePolicy(method='no_evidence',**common)
    rows=[]
    for step,row in enumerate(run['rows']):
        if adaptive.ask()!=row['candidate_id']:
            raise ValueError('adaptive replay selected different candidate')
        idx=adaptive.pending[0]
        logged=row['prediction']; replay=adaptive.pending[2]
        if logged is None or replay is None or not np.allclose(logged['weights'],replay['weights'],rtol=1e-9,atol=1e-10) or not np.allclose([logged['mean'],logged['variance']],[replay['mean'],replay['variance']],rtol=1e-9,atol=1e-10):
            raise ValueError('adaptive replay prediction mismatch')
        available=[j for j,c in enumerate(adaptive.ids) if c not in adaptive.used]
        points=adaptive.features[available]
        a_mean,a_var=adaptive.model.predict(points); b_mean,b_var=fixed.model.predict(points)
        decision=None
        if step>=len(definition['initial']) and len(adaptive.model.y):
            aa=adaptive.model.acquisition(points); bb=fixed.model.acquisition(points)
            decision={'adaptive':adaptive.ids[available[int(np.argmax(aa))]],
                      'no_evidence':adaptive.ids[available[int(np.argmax(bb))]],
                      'max_acquisition_delta':float(np.max(np.abs(aa-bb)))}
        rows.append({'step':step,'candidate_id':row['candidate_id'],
            'weight_l1':float(np.sum(np.abs(adaptive.model.weights-fixed.model.weights))),
            'max_mean_delta':float(np.max(np.abs(a_mean-b_mean))),
            'max_variance_delta':float(np.max(np.abs(a_var-b_var))), 'decision':decision})
        if row['status']!='infrastructure_failure':
            fixed.model.observe(adaptive.features[idx],row['utility'])
        adaptive.tell(row['candidate_id'],status=row['status'],utility=row['utility'],cost=row['cost'])
    return {'object_id':catalog['object_id'],'seed':definition['seed'],'rows':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--catalogues',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=DevelopmentData(a.protocol/'manifest.prepilot.json',load(a.protocol/'lock.json'),a.data)
    catalogues=[data.catalogue(x,a.binary) for x in sorted(a.catalogues.glob('*.json')) if x.name!='pca.json']
    archive=[(c,load(a.run/'history'/(c['object_id']+'.json'))) for c in catalogues if c['split']=='train']
    results=[]
    for c in catalogues:
        if c['split']!='validation': continue
        for run_path in sorted((a.run/'runs'/c['object_id']).glob('*-adaptive.json')):
            results.append(compare(data,c,archive,load(run_path)))
    decisions=[r['decision'] for run in results for r in run['rows'] if r['decision'] is not None]
    summary={'runs':len(results),'compared_decisions':len(decisions),
             'changed_decisions':sum(d['adaptive']!=d['no_evidence'] for d in decisions),
             'max_weight_l1':max((r['weight_l1'] for run in results for r in run['rows']),default=0),
             'max_mean_delta':max((r['max_mean_delta'] for run in results for r in run['rows']),default=0),
             'max_variance_delta':max((r['max_variance_delta'] for run in results for r in run['rows']),default=0),
             'max_acquisition_delta':max((d['max_acquisition_delta'] for d in decisions),default=0),
             'runs_detail':results}
    save(a.out,summary)

if __name__=='__main__': main()
