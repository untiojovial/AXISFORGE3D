"""Optional CPU regression on one pre-authorized development object; never test assets."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pytest

@pytest.fixture
def development():
    path=os.getenv('AXISFORGE_PHASE3_DATA')
    if not path: pytest.skip('requires the local authorized development bundle')
    root=Path(path); rows=json.loads((root/'manifest.json').read_text())['objects']
    row=next(x for x in rows if x['object_id']=='ACE_Coffee_Mug_Kristen_16_oz_cup')
    assert row['split']=='train'
    return root,row


def test_sealed_visual_and_texture_corruption(development,tmp_path):
    from research_grasping.phase3.vision_integrity import verify_visual, verify_texture
    root,row=development; oid=row['object_id']; archive=root/'vision-development-v3'/(oid+'.npz'); texture=(root/row['original_mesh']).parent/'texture.png'
    verify_visual(archive,oid); verify_texture(texture,oid)
    with np.load(archive,allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    arrays['patches']=arrays['patches'].copy(); arrays['patches'][0,0,0,0]+=1
    corrupt=tmp_path/'bad.npz';np.savez_compressed(corrupt,**arrays)
    with pytest.raises(ValueError,match='vision array lock'):verify_visual(corrupt,oid)
    cp=tmp_path/'texture.png';shutil.copy(texture,cp)
    with cp.open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='texture lock'):verify_texture(cp,oid)
    with pytest.raises(ValueError,match='vision array lock'):verify_visual(archive,'not-an-authorized-object')


def test_real_multiview_depth_projection_and_visibility(development):
    from research_grasping.representations.geometry import backproject,project
    from research_grasping.phase3.vision_openclip import visible_patch
    root,row=development
    with np.load(root/'vision-development-v3'/(row['object_id']+'.npz'),allow_pickle=False) as z:
        for depth,mask,pose,grid in zip(z['depth'],z['masks'],z['world_from_camera'],z['patches'],strict=True):
            points,pixels=backproject(depth,z['K'],pose)
            assert len(points)==np.count_nonzero(mask) and len(points)>0
            ix=np.linspace(0,len(points)-1,min(30,len(points)),dtype=int)
            uv,dist=project(points[ix],z['K'],pose)
            np.testing.assert_allclose(uv,pixels[ix],atol=1e-8)
            np.testing.assert_allclose(dist,depth[pixels[ix,1],pixels[ix,0]],atol=1e-8)
            assert visible_patch(points[ix[0]],depth,z['K'],pose,grid) is not None
            camera=pose[:3,3]; occluded=camera+1.2*(points[ix[0]]-camera)
            assert visible_patch(occluded,depth,z['K'],pose,grid,tolerance=.0001) is None


def test_real_source_atlas_samples_on_visible_surface(development):
    pytest.importorskip('trimesh')
    from PIL import Image
    from scipy.spatial import cKDTree
    from research_grasping.representations.geometry import backproject
    from research_grasping.phase3.source_appearance import triangles_uv, atlas_samples
    root,row=development; mesh=root/row['original_mesh']; tris,uv=triangles_uv(mesh); tree=cKDTree(tris.mean(axis=1))
    with Image.open(mesh.parent/'texture.png') as im: atlas=np.asarray(im.convert('RGB'))
    with np.load(root/'vision-development-v3'/(row['object_id']+'.npz'),allow_pickle=False) as z:
        points,_=backproject(z['depth'][0],z['K'],z['world_from_camera'][0])
    indices=np.linspace(0,len(points)-1,20,dtype=int); rgb,valid=atlas_samples(points[indices],tris,uv,atlas,tree)
    assert rgb.shape==(20,3) and rgb.dtype==np.uint8 and valid.sum()>=15


def test_textured_pipeline_skips_sealed_test(tmp_path,monkeypatch):
    """Synthetic control: absent test files ensure accidental access fails."""
    import sys
    from PIL import Image
    from research_grasping.phase3 import source_appearance as a
    pytest.importorskip('trimesh')
    root=tmp_path/'root';root.mkdir(); mesh=root/'sample.obj';mesh.write_text('v 0 0 1\nv 1 0 1\nv 0 1 1\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n')
    Image.fromarray(np.full((2,2,3),120,np.uint8)).save(root/'texture.png')
    rows=[{'object_id':'train','split':'train','texture_availability':True,'original_mesh':'sample.obj','original_sha256':a.digest(mesh),'processed_sha256':'synthetic'}, {'object_id':'sealed','split':'test','texture_availability':True,'original_mesh':'NEVER_OPEN.obj'}]
    (root/'manifest.json').write_text(json.dumps({'objects':rows}));monkeypatch.setattr(a,'PREPILOT_MANIFEST',a.digest(root/'manifest.json'))
    vision=tmp_path/'vision';vision.mkdir();np.savez_compressed(vision/'train.npz',mesh_sha256='synthetic',depth=np.ones((16,1,1)),K=np.eye(3),world_from_camera=np.tile(np.eye(4),(16,1,1)),vectors=np.tile([1.,0.],(16,1)))
    seen=[];monkeypatch.setattr(a,'verify_texture',lambda path,oid:seen.append(('texture',oid)));monkeypatch.setattr(a,'verify_visual',lambda path,oid:seen.append(('vision',oid)))
    monkeypatch.setattr(a,'load_model',lambda path:(None,None));monkeypatch.setattr(a,'encode_images',lambda model,transform,images:np.tile([1.,0.],(len(images),1)))
    out=tmp_path/'out';summary=tmp_path/'summary.json';monkeypatch.setattr(sys,'argv',['run','--root',str(root),'--vision',str(vision),'--checkpoint','unused','--out',str(out),'--summary',str(summary)])
    a.main();result=json.loads(summary.read_text());assert [o['object_id'] for o in result['objects']]==['train'];assert seen==[('texture','train'),('vision','train')]
    with np.load(out/'train.npz',allow_pickle=False) as z:assert z['rgb'].shape==(16,1,1,3) and len(z['points'])==16
    assert not (out/'sealed.npz').exists()


def test_real_pretrained_phase3_replays_view(development):
    import os
    checkpoint=os.getenv('AXISFORGE_OPENCLIP_CHECKPOINT')
    if not checkpoint: pytest.skip('requires the authorized official OpenCLIP checkpoint')
    torch=pytest.importorskip('torch');torch.set_num_threads(4)
    from research_grasping.phase3.vision_openclip import load_model,encode_images,encode_patches
    root,row=development; model,transform=load_model(checkpoint)
    with np.load(root/'vision-development-v3'/(row['object_id']+'.npz'),allow_pickle=False) as z:
        rgb=z['rgb'][0];expected=z['vectors'][0];patches=z['patches'][0]
    actual=encode_images(model,transform,[rgb])[0];local=encode_patches(model,transform,[rgb])[0]
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=1e-6)
    np.testing.assert_allclose(local,patches,atol=2e-5,rtol=1e-5)


def test_render_invalid_camera_mesh_and_empty_visibility():
    from research_grasping.phase3.render_cpu import cameras,look_at,render
    with pytest.raises(ValueError): look_at([0,0,0],[0,0,0])
    with pytest.raises(ValueError): look_at([0,0,0],[0,0,1])
    with pytest.raises(ValueError): cameras([[0,0,0],[1,1,1]],3)
    with pytest.raises(ValueError): cameras([[0,0,0],[0,1,1]],4)
    with pytest.raises(ValueError): render(np.zeros((3,3)),np.array([[0,1,9]]),np.eye(3),np.eye(4),8,8)
    mesh=np.array([[20.,20,1],[21,20,1],[20,21,1]])
    result=render(mesh,np.array([[0,1,2]]),np.eye(3),np.eye(4),8,8)
    assert not result['mask'].any() and np.isnan(result['depth']).all()


def test_openclip_public_input_failures():
    from research_grasping.phase3.vision_openclip import encode_images,encode_patches,pool_views,visible_patch
    with pytest.raises(ValueError): encode_images(None,None,[],batch_size=0)
    with pytest.raises(ValueError): encode_images(None,None,[])
    with pytest.raises(ValueError): encode_patches(None,None,[])
    with pytest.raises(ValueError): encode_patches(None,None,[np.zeros((4,4),dtype=np.uint8)])
    with pytest.raises(ValueError): pool_views(np.zeros((16,5)),4)
    with pytest.raises(ValueError): pool_views(np.ones((8,5)),4)
    with pytest.raises(ValueError): visible_patch([0,0,1],np.ones((4,5)),np.eye(3),np.eye(4),np.ones((7,7,3)))
    with pytest.raises(ValueError): visible_patch([0,0],np.ones((4,4)),np.eye(3),np.eye(4),np.ones((7,7,3)))
