import copy
import json
from pathlib import Path

import numpy as np
import pytest

from research_grasping.phase4 import bundle_analysis as analysis
from research_grasping.phase4 import bundle_io
from research_grasping.phase4.portable_panel import visible_coordinates


def test_callback_requires_independent_model_identity(tmp_path, monkeypatch):
    model = {'model_id':'open_clip/ViT-B-32-quickgelu/laion400m_e31','model_revision':'v0.2-weights',
             'checkpoint_sha256':'d867053b2301634007ed9af230bfb1a217ec634f6c0329f04092133ae5c4b89e',
             'processor_sha256':'c4c7c2276f27ec1b7f9e45f1f76dde5176ca32a245eafeadd607a3c7f4c1660e'}
    monkeypatch.setattr(bundle_io,'verify_bundle',lambda *_:copy.deepcopy(model))
    path=tmp_path/'index.json'
    path.write_text(json.dumps(model))
    def call(**params):
        return analysis.verified_index(tmp_path,'bundle','job',params.get('weights'),params.get('rev'),params.get('processor'))
    assert call()==model
    path.write_text('{}')
    with pytest.raises(ValueError,match='index differs'): call()
    model['model_revision']='arbitrary';path.write_text(json.dumps(model))
    with pytest.raises(ValueError,match='independent pins'): call()
    model.update(model_id='dinov2',model_revision='f'*40);path.write_text(json.dumps(model))
    with pytest.raises(ValueError,match='DINO requires'): call()
    assert call(rev='f'*40,weights=model['checkpoint_sha256'],processor=model['processor_sha256'])==model
    model['model_id']='custom';path.write_text(json.dumps(model))
    with pytest.raises(ValueError,match='unsupported'): call()
    with pytest.raises(ValueError,match='Original result'): analysis.run(tmp_path)
    monkeypatch.setattr(analysis,'verified_index',lambda *_: model)
    monkeypatch.setattr(analysis,'PANEL',tmp_path)
    (tmp_path/'lock.json').write_text('{}')
    with pytest.raises(ValueError,match='independent pin'): analysis.run(tmp_path)


def test_affine_padding_duplicate_tokens_and_train_only_pca():
    geom=dict(grid_shape=[1,2],processed_size=[2,4],origin_size=[2,4],
              pixel_transform=np.eye(3).tolist(),content_bounds=[-.5,-.5,1.5,3.5])
    patches=np.array([[[1.,0.],[0.,1.]],[[1.,0.],[0.,1.]]])
    points=[[[0,0],[.1,0],[3,0],[4,0]],[]]
    v,views=analysis.patch_features(points,patches,geom,[2,4])
    assert views==1 and np.allclose(v,[2**-.5]*2)
    assert analysis.patch_features([[],[]],patches,geom,[2,4])[1]==0
    with pytest.raises(ValueError):analysis.patch_features(points,patches,geom,[4,2])
    blocks={'train':np.eye(4),'val':np.ones((3,4))}
    pc=analysis.compressed(['train'],blocks)
    blocks['val']*=500
    assert np.array_equal(pc['train'],analysis.compressed(['train'],blocks)['train'])


def test_visibility_uses_measured_depth_and_camera():
    points=np.array([[0,0,1],[0,0,2],[5,0,1],[0,0,-1.]])
    k=np.eye(3)
    assert visible_coordinates(points,[np.ones((2,2))],k,[np.eye(4)])==[[[0.,0.]]]


def test_real_import_panel_and_file_tamper(tmp_path):
    import os
    root=Path(os.environ.get('AXISFORGE_PHASE4_DATA','/workspace/axisforge-phase4-data'))
    source=root/'openclip-rerun-import'
    if not source.is_dir():pytest.skip('real extracted development bundle is external')
    job=analysis.PANEL/'development-images.zip'
    result=analysis.run(source,bundle_path=root/'openclip-rerun.zip',job_path=job,out=tmp_path/'result.json')
    assert result['status']=='COMPLETED' and result['objects']==22
    runs=result['results']['geometry']['trajectories']
    assert len(runs)==72 and all(len(set(t['selected']))==6 for t in runs)
    assert len(result['results']['global_local']['predictive'])==6
    idx=json.loads((source/'index.json').read_text());row=idx['objects'][0]; name=row['object_id']
    entry={name:{'split':row['split']}}
    with pytest.raises(ValueError,match='unexpected feature object'):analysis.load_inputs(source,idx,{})
    altered=dict(row,file='../file.npz')
    with pytest.raises(ValueError,match='file path'):analysis.load_inputs(source,dict(idx,objects=[altered]),entry)
    with pytest.raises(ValueError,match='changed after import'):
        analysis.load_inputs(source,dict(idx,objects=[dict(row,sha256='0'*64)]),entry)
