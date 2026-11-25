"""Pre-query native catalogue regression and adversarial split/provenance checks."""
import copy
import json
import os
from pathlib import Path
import numpy as np
import pytest
from research_grasping.phase3 import prepare_pilot as p
from research_grasping.phase3.functional_catalogue import contact_geometry,fit_pca


def xy(x,y,z):return {'x':x,'y':y,'z':z}


def candidate(cid=1):
    return {'id':cid,'candidate':{'center':xy(.45,0,.42),'approach':xy(0,0,1),'closing':xy(1,0,0),'width':.06,'contacts':[{'point':xy(.48,0,.42),'normal':xy(-1,0,0)},{'point':xy(.42,0,.42),'normal':xy(1,0,0)}]}}


def test_split_identity_and_postquery_rejected():
    m={'objects':[{'object_id':'a','split':'train'},{'object_id':'b','split':'test'}]}
    assert p.selected_objects(m,['a'])[0]['split']=='train'
    for ids in (['a','a'],['b'],['missing']):
        with pytest.raises(ValueError):p.selected_objects(m,ids)
    with pytest.raises(ValueError):p.selected_objects({'objects':[m['objects'][0],m['objects'][0]]},['a'])
    rec=candidate(); _,f,c=p.catalogue([rec]); assert len(f[0])==16 and len(c['1']['contacts'])==2
    for key in ('quality','plan','rejection','qualityElapsedS','attemptElapsedS'):
        bad=copy.deepcopy(rec);bad[key]=0
        with pytest.raises(ValueError,match='post-evaluation'):p.catalogue([bad])
    with pytest.raises(ValueError,match='duplicate native ID'):p.catalogue([rec,rec])
    with pytest.raises(ValueError):p.catalogue([])
    bad=copy.deepcopy(rec);bad['candidate']['width']=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):p.catalogue([bad])


def test_mesh_bad_stl_and_deterministic_points(tmp_path):
    pytest.importorskip('trimesh');import trimesh
    path=tmp_path/'mesh.stl';path.write_bytes(b'truncated')
    with pytest.raises(ValueError):p.mesh_vertices(path)
    mesh=trimesh.creation.box(extents=[.1,.2,.3]);mesh.export(path);tris=p.mesh_vertices(path)
    first=p.mesh_points(tris);second=p.mesh_points(tris)
    np.testing.assert_array_equal(first[0],second[0]);assert first[0].shape==(256,3)
    with pytest.raises(ValueError):p.mesh_points(np.zeros((1,3,3)))
    raw=bytearray(path.read_bytes());raw[84+12:84+16]=np.float32(np.nan).tobytes();path.write_bytes(raw)
    with pytest.raises(ValueError,match='nonfinite'):p.mesh_vertices(path)


def test_visual_provenance_rejects_mismatch_before_native(tmp_path):
    row={'object_id':'one','split':'train'};manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'objects':[row]})); archive=tmp_path/'summary.npz';provenance=tmp_path/'provenance.json'
    def run():p.prepare(manifest,tmp_path/'never-binary',tmp_path/'out',['one'],vision_path=archive)
    provenance.write_text(json.dumps({'manifest_sha256':'wrong'}))
    with pytest.raises(ValueError,match='provenance manifest'):run()
    provenance.write_text(json.dumps({'manifest_sha256':p.digest(manifest)}))
    np.savez(archive,ids=['one','one'],splits=['train','train'],embedding_4=np.ones((2,512)))
    with pytest.raises(ValueError,match='invalid vision'):run()
    np.savez(archive,ids=['one'],splits=['test'],embedding_4=np.ones((1,512)))
    with pytest.raises(ValueError,match='non-development'):run()
    np.savez(archive,ids=['different'],splits=['train'],embedding_4=np.ones((1,512)))
    with pytest.raises(ValueError,match='missing or mismatched'):run()
    np.savez(archive,ids=['one'],splits=['validation'],embedding_4=np.ones((1,512)))
    with pytest.raises(ValueError,match='missing or mismatched'):run()


def test_functional_degenerate_and_train_only():
    feature=np.zeros(16);feature[3:6]=[0,0,1];feature[6:9]=[1,0,0]
    with pytest.raises(ValueError):contact_geometry({'position':[0,0,0],'contacts':[]},feature)
    feature[3:6]=0
    with pytest.raises(ValueError,match='degenerate approach'):contact_geometry({'position':[0,0,0],'contacts':[[0,0,0]]},feature)
    with pytest.raises(ValueError,match='train-only'):fit_pca([{'split':'validation'}])
    rows=[{'object_id':'a','cid':i,'split':'train','views':1,'local':[1.,1.,1.,1.,1.]} for i in range(5)]
    with pytest.raises(ValueError,match='rank'):fit_pca(rows)
    rows[0]['views']=0
    with pytest.raises(ValueError,match='insufficient visible'):fit_pca(rows)


def test_native_prepare_one_real_train_with_visual_context(tmp_path,monkeypatch):
    data=os.getenv('AXISFORGE_PHASE3_DATA');binary=os.getenv('AXISFORGE_BIN')
    if not data or not binary or not Path(binary).is_file():pytest.skip('requires local development bundle and native binary')
    root=Path(data);name='ACE_Coffee_Mug_Kristen_16_oz_cup';out=tmp_path/'out'
    monkeypatch.setattr(__import__('socket').socket,'connect',lambda *a,**k:(_ for _ in ()).throw(AssertionError('network')))
    p.prepare(root/'manifest.json',binary,out,[name],cap=8,vision_path=root/'vision-development-v3'/'summary.npz')
    result=json.loads((out/(name+'.json')).read_text());assert result['split']=='train' and len(result['ids'])==8 and len(result['features'][0])==16
    with np.load(root/'vision-development-v3'/'summary.npz',allow_pickle=False) as z:
        index=z['ids'].tolist().index(name);np.testing.assert_allclose(result['context'][3:],z['embedding_4'][index])
    assert set(result['ids'])==set(result['candidates'])


def test_pca_missing_object_and_inconsistent_contact_frame():
    rows=[{'object_id':'a','cid':i,'split':'train','views':1,'local':np.eye(6)[i]} for i in range(5)]
    rows.append({'object_id':'b','cid':0,'split':'train','views':0,'local':np.zeros(6)})
    with pytest.raises(ValueError,match='no visible candidates'):fit_pca(rows)
    f=np.zeros(16);f[3:6]=[0,0,1];f[6:9]=[0,0,1]
    with pytest.raises(ValueError,match='degenerate closing'):contact_geometry({'position':[0,0,0],'contacts':[[.1,0,0]]},f)


def test_actual_token_corrupt_depth_rejected(tmp_path):
    data=os.getenv('AXISFORGE_PHASE3_DATA')
    if not data:pytest.skip('requires local development bundle')
    from research_grasping.phase3.contact_similarity import descriptors
    root=Path(data);name='ACE_Coffee_Mug_Kristen_16_oz_cup';cat=json.loads((root/'pilot-expanded'/'catalogs'/(name+'.json')).read_text());path=root/'vision-development-v3'/(name+'.npz')
    result=descriptors(cat,[cat['ids'][0]],path,cat['metadata']['manifest_sha256']);assert len(result)==1 and len(result[0]['local'])==768
    with np.load(path,allow_pickle=False) as z: arrays={key:z[key] for key in z.files}
    arrays['depth']=arrays['depth'].copy();arrays['depth'][0,0,0]=999
    bad=tmp_path/'bad.npz';np.savez_compressed(bad,**arrays)
    with pytest.raises(ValueError,match='vision array lock'):descriptors(cat,[cat['ids'][0]],bad,cat['metadata']['manifest_sha256'])
