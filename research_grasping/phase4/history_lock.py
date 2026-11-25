"""Freeze the measured stratified train history before source-model selection."""
import argparse
from pathlib import Path

from research_grasping.phase3.experiment import digest, load, save
from .catalogue_study import REFERENCE, writer_lock


def seal(work):
    selection = load(work/'selection.json')
    expected = {r['object_id'] for r in load(REFERENCE/'manifest.prepilot.json')['objects'] if r['split']=='train'}
    files = {p.stem:p for p in (work/'history/stratified').glob('*.json')}
    if set(files)!=expected: raise ValueError('requires the full training split')
    values={}
    for name,path in files.items():
        result=load(path)
        if (result['selection_sha256']!=digest(work/'selection.json') or result['selected']!=selection['objects'][name]['sampling']['stratified']
                or [row['candidate_id'] for row in result['rows']]!=result['selected']):
            raise ValueError('history and selection differ')
        values[name]=digest(path)
    lock=work/'history_stratified.lock.json'
    with writer_lock(work/'.history_seal.lock'):
        if lock.exists() and load(lock)!=values: raise ValueError('stored history lock differs')
        if not lock.exists(): save(lock,values)
    return values


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    seal(p.parse_args().work)
