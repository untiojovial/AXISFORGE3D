"""Local integration and negative controls; no network and no sealed test assets."""
import io
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest


def no_network(*args,**kwargs):
    raise AssertionError('network access forbidden in tests')


def test_fetch_pinned_checkpoint_never_downloads_unnecessarily(tmp_path,monkeypatch):
    from research_grasping.phase3 import fetch_openclip as f
    monkeypatch.setattr(f,'urlopen',no_network)
    destination=tmp_path/f.NAME; destination.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='local checkpoint'):f.fetch(tmp_path)
    checkpoint=os.getenv('AXISFORGE_OPENCLIP_CHECKPOINT')
    if checkpoint:
        destination.unlink(); destination.symlink_to(Path(checkpoint)); assert f.fetch(tmp_path)==destination


def test_fetch_checkpoint_failed_download_cleans_partial(tmp_path,monkeypatch):
    from research_grasping.phase3 import fetch_openclip as f
    monkeypatch.setattr(f,'urlopen',lambda *a,**k:io.BytesIO(b'corrupted download'))
    with pytest.raises(ValueError,match='failed verification'):f.fetch(tmp_path)
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(f,'urlopen',no_network)
    with pytest.raises(AssertionError,match='network access'):f.fetch(tmp_path)
    assert not list(tmp_path.iterdir())


def test_dataset_fetch_local_preserves_and_detects_changes(tmp_path,monkeypatch):
    from research_grasping.phase3 import dataset_reproduce as d
    monkeypatch.setattr(d,'urlopen',no_network)
    path=tmp_path/'model.obj'; path.write_bytes(b'original'); expected=d.digest(path)
    d.fetch('unused',path,expected); assert path.read_bytes()==b'original'
    with pytest.raises(ValueError,match='checksum'):d.fetch('unused',path,'0'*64)
    assert path.read_bytes()==b'original'
    with pytest.raises(AssertionError,match='network access'):d.fetch('unused',tmp_path/'missing.obj')


def test_dataset_entrypoint_local_subset_and_sealed_test(tmp_path,monkeypatch):
    trimesh=pytest.importorskip('trimesh')
    from research_grasping.phase3 import dataset_reproduce as d
    root=tmp_path/'data'; mesh=root/'originals'/'dev'/'model.obj';mesh.parent.mkdir(parents=True)
    trimesh.creation.box(extents=[.1,.2,.3]).export(mesh)
    (mesh.parent/'model.xml').write_text('<mujoco/>')
    visual=root/'processed'/'dev'/'visual.stl';visual.parent.mkdir(parents=True);trimesh.load(mesh,force='mesh',process=False).export(visual)
    obj={'object_id':'dev','split':'train','original_mesh':'originals/dev/model.obj','original_sha256':d.digest(mesh),'texture_availability':False,'visual_mesh':'processed/dev/visual.stl','processed_sha256':d.digest(visual)}
    sealed={'object_id':'sealed','split':'test','original_mesh':'NEVER_OPEN.obj'}
    lock=tmp_path/'lock.json';lock.write_text(json.dumps({'objects':[obj,sealed]}));monkeypatch.setattr(d,'LOCK',lock)
    # Existing, byte-locked assets: no export and no assumption of cross-version serialization.
    monkeypatch.setattr(trimesh,'__version__','5.2.2');monkeypatch.setattr(d,'urlopen',no_network)
    monkeypatch.setattr(sys,'argv',['run','--root',str(root),'--splits','train']);d.main()
    assert (root/'manifest.json').read_bytes()==lock.read_bytes() and not (root/'NEVER_OPEN.obj').exists()
    visual.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checksum'):d.main()


def test_vision_pipeline_one_real_dev_mesh_and_cache(tmp_path,monkeypatch):
    source=os.getenv('AXISFORGE_PHASE3_DATA');checkpoint=os.getenv('AXISFORGE_OPENCLIP_CHECKPOINT')
    if not source or not checkpoint:pytest.skip('requires official local assets and weights')
    import torch
    from research_grasping.phase3 import vision_pipeline as v
    monkeypatch.setattr(__import__('socket').socket,'connect',no_network)
    root=Path(source); manifest=json.loads((root/'manifest.json').read_text()); row=next(r for r in manifest['objects'] if r['object_id']=='ACE_Coffee_Mug_Kristen_16_oz_cup')
    assert row['split']=='train'
    target=tmp_path/'data';target.mkdir();dest=target/row['visual_mesh'];dest.parent.mkdir(parents=True);shutil.copy(root/row['visual_mesh'],dest)
    (target/'manifest.json').write_text(json.dumps({'objects':[row,{'object_id':'sealed','split':'test','visual_mesh':'NEVER_OPEN.stl'}]}))
    out=tmp_path/'out';monkeypatch.setattr(sys,'argv',['run','--root',str(target),'--out',str(out),'--checkpoint',checkpoint,'--resolution','48'])
    v.main();cache=out/(row['object_id']+'.npz');first=cache.read_bytes()
    with np.load(out/'summary.npz',allow_pickle=False) as z: assert z['ids'].tolist()==[row['object_id']]
    v.main();assert cache.read_bytes()==first and not (out/'sealed.npz').exists()
    with np.load(cache,allow_pickle=False) as z: arrays={key:z[key] for key in z.files}
    arrays['mesh_sha256']='0'*64;np.savez_compressed(cache,**arrays)
    with pytest.raises(ValueError,match='stale cache'):v.main()


def test_dataset_fetch_stream_is_verified_and_not_overwritten(tmp_path,monkeypatch):
    from research_grasping.phase3 import dataset_reproduce as d
    import hashlib
    payload=b'authorized-test-fixture'; monkeypatch.setattr(d,'urlopen',lambda *a,**k:io.BytesIO(payload))
    target=tmp_path/'nested'/'file.obj'; d.fetch('fixture',target,hashlib.sha256(payload).hexdigest())
    assert target.read_bytes()==payload
    monkeypatch.setattr(d,'urlopen',no_network);d.fetch('unused',target,hashlib.sha256(payload).hexdigest())


def test_dataset_rejects_incompatible_version_before_io(tmp_path,monkeypatch):
    trimesh=pytest.importorskip('trimesh')
    from research_grasping.phase3 import dataset_reproduce as d
    monkeypatch.setattr(trimesh,'__version__','0.0.0');monkeypatch.setattr(sys,'argv',['run','--root',str(tmp_path)])
    with pytest.raises(RuntimeError,match='trimesh==5.2.2'):d.main()
