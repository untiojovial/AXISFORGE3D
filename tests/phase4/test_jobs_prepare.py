"""Real sealed visual input exercises job preparation without mutating the dataset."""
import json
from pathlib import Path
import numpy as np
import pytest
from research_grasping.phase4.jobs import prepare
from research_grasping.phase4.bundle_io import load_job

DATA=Path('/workspace/axisforge-phase3-data')
OBS=DATA/'vision-development-v3'
OID='ACE_Coffee_Mug_Kristen_16_oz_cup'


@pytest.fixture
def real_input(tmp_path):
    if not (OBS/(OID+'.npz')).is_file() or not (DATA/'manifest.json').is_file():
        pytest.skip('external sealed Phase III visual inputs unavailable')
    import shutil
    source=tmp_path/'observations';source.mkdir()
    shutil.copyfile(OBS/(OID+'.npz'),source/(OID+'.npz'))
    return source


def test_prepare_verifies_real_sealed_input_and_split(real_input,tmp_path):
    output=tmp_path/'job.zip'
    job=prepare(DATA,real_input,output,objects=[OID])
    assert load_job(output)==job
    assert job['objects'][0]['object_id']==OID
    assert [x['view_id'] for x in job['objects'][0]['views']]==list(range(16))
    manifest=json.loads((DATA/'manifest.json').read_text())
    test_id=next(o['object_id'] for o in manifest['objects'] if o['split']=='test')
    with pytest.raises(ValueError,match='development-only'):
        prepare(DATA,real_input,tmp_path/'forbidden.zip',objects=[test_id])


@pytest.mark.parametrize('mutation',['pixel','view_order','metadata'])
def test_prepare_rejects_modified_real_npz(real_input,tmp_path,mutation):
    path=real_input/(OID+'.npz')
    with np.load(path,allow_pickle=False) as archive:
        arrays={key:archive[key].copy() for key in archive.files}
    if mutation=='pixel': arrays['rgb'][0,0,0,0] ^= np.uint8(1)
    elif mutation=='view_order': arrays['rgb'][[0,1]]=arrays['rgb'][[1,0]]
    else: arrays['mesh_sha256']=np.array('0'*64)
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='vision array lock mismatch'):
        prepare(DATA,real_input,tmp_path/'out.zip',objects=[OID])
    assert not (tmp_path/'out.zip').exists()
