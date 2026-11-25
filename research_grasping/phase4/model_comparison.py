"""Compare explicitly named model reports on an identical measured panel."""
import argparse
from pathlib import Path

from research_grasping.phase3.experiment import digest,load,save
from .embedding_report import summarize


def compare(paths):
    reports={}
    reference=None
    for path in paths:
        report=load(path)
        identity=(report['panel_lock_sha256'],report['job_sha256'])
        if reference is not None and identity!=reference:
            raise ValueError('different input images or evaluation protocol')
        reference=identity
        label=report['model_id']
        if label in reports:raise ValueError('duplicate model')
        summary=summarize(path)
        control=summary['results']['geometry']['auc']
        if reports and next(iter(reports.values()))['geometry_auc']!=control:
            raise ValueError('geometry control differs')
        reports[label]=dict(geometry_auc=control,source_sha256=digest(path),
                            revision=report['model_revision'],weights_sha256=report['checkpoint_sha256'],
                            results=summary['results'])
    if not reports:raise ValueError('no verified input reports')
    return dict(models=reports,observed_models=len(reports),scope='matched descriptive development data',
                comparison_status='MULTIPLE_MODELS_OBSERVED' if len(reports)>1 else 'SINGLE_MODEL_ONLY')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();save(a.out,compare(a.inputs))
