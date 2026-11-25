import numpy as np
import pytest
from research_grasping.phase3.functional_catalogue import contact_geometry, fit_pca


def test_geometry_local_frame_and_degeneracy():
    f=np.zeros(16); f[3:6]=[0,0,1]; f[6:9]=[1,0,0]
    c={'position':[0,0,0], 'contacts':[[.1,0,0],[-.1,0,0]]}
    a=contact_geometry(c,f); np.testing.assert_allclose(a[:3],0); np.testing.assert_allclose(a[3:6],[1,0,0])
    r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    f[3:6]=r@f[3:6]; f[6:9]=r@f[6:9]; c['contacts']=(np.array(c['contacts'])@r.T).tolist()
    np.testing.assert_allclose(a,contact_geometry(c,f))
    f[6:9]=f[3:6]
    with pytest.raises(ValueError): contact_geometry(c,f)


def test_train_only_balanced_pca():
    rng=np.random.default_rng(7); rows=[]
    for name,n in [('a',5),('b',9)]:
        for i in range(n): rows.append(dict(object_id=name,cid=i,split='train',views=1,local=rng.normal(size=8)))
    mean,vec,digest=fit_pca(rows); assert vec.shape==(4,8) and len(digest)==64
    np.testing.assert_allclose(mean,(np.mean([x['local'] for x in rows[:5]],axis=0)+np.mean([x['local'] for x in rows[5:]],axis=0))/2)
    rows[0]['split']='validation'
    with pytest.raises(ValueError): fit_pca(rows)


def test_real_guard_rejects_changed_features_and_ids(tmp_path):
    import os, json
    from pathlib import Path
    from research_grasping.phase3.experiment import DevelopmentData, load
    root=os.getenv('AXISFORGE_PHASE3_DATA')
    if not root: pytest.skip('set AXISFORGE_PHASE3_DATA to offline development bundle')
    root=Path(root); protocol=Path(__file__).resolve().parents[2]/'research/phase3/pilot'
    binary=Path(os.getenv('AXISFORGE_BIN',str(protocol.parents[2]/'target/release/axisforge')))
    source=root/'functional-catalogs'; name='2_of_Jenga_Classic_Game'
    data=DevelopmentData(protocol/'manifest.prepilot.json',load(protocol/'lock.json'),root)
    cat=load(source/(name+'.json')); (tmp_path/'pca.json').write_text((source/'pca.json').read_text())
    candidate=tmp_path/(name+'.json'); candidate.write_text(json.dumps(cat))
    assert data.catalogue(candidate,binary)['object_id']==name
    bad=json.loads(json.dumps(cat)); bad['features'][0][-1]+=1
    candidate.write_text(json.dumps(bad))
    with pytest.raises(ValueError): data.catalogue(candidate,binary)
    bad=json.loads(json.dumps(cat)); bad['ids'][0],bad['ids'][1]=bad['ids'][1],bad['ids'][0]
    candidate.write_text(json.dumps(bad))
    with pytest.raises(ValueError): data.catalogue(candidate,binary)
