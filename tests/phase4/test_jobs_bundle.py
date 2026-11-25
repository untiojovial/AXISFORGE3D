import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from PIL import Image
from research_grasping.phase4.jobs import canonical, sha_bytes, MANIFEST_HASH
from research_grasping.phase4.bundle_io import load_job, verify_bundle, install_bundle
from research_grasping.phase4.orchestrator import run_extraction, export_bundle


@pytest.fixture
def job(tmp_path, monkeypatch):
    lock = json.loads((Path(__file__).resolve().parents[2]/'research/phase3/dataset/manifest.lock.json').read_text())
    row = next(r for r in lock['objects'] if r['split']=='train'); oid=row['object_id']
    views=[]; blobs={}
    for i in range(16):
        path=f'images/{oid}/{i:02d}.png'; buffer=io.BytesIO(); Image.fromarray(np.full((96,96,3), i,dtype='uint8')).save(buffer,format='PNG')
        blobs[path]=buffer.getvalue(); views.append({'view_id':i,'path':path,'sha256':sha_bytes(blobs[path])})
    spec={'schema_version':1,'dataset_manifest_sha256':MANIFEST_HASH,'objects':[{'object_id':oid,'split':'train','views':views,'observation_sha256':'x'}]}
    spec['job_id']=sha_bytes(canonical(spec)); path=tmp_path/'job.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('job.json',canonical(spec))
        for k,v in blobs.items(): z.writestr(k,v)
    import research_grasping.phase4.bundle_io as bio
    rgb_lock = tmp_path/'rgb.json'
    pixels = [sha_bytes(np.full((96,96,3), i, dtype='uint8').tobytes()) for i in range(16)]
    rgb_lock.write_bytes(canonical({'views':{oid:pixels}}))
    monkeypatch.setattr(bio, 'RGB_LOCK', rgb_lock)
    monkeypatch.setattr(bio, 'RGB_LOCK_HASH', sha_bytes(rgb_lock.read_bytes()))
    return path


class Fake:
    provenance={'model_id':'unit-test','weights_sha256':'a'*64,'processor':'test-only'}
    def __init__(self): self.calls=0
    def encode(self, images):
        self.calls+=1; n=len(images)
        return {'global_embeddings':np.ones((n,4),dtype='float32'), 'patch_embeddings':np.ones((n,2,3),dtype='float32'), 'token_geometry':{'grid_shape':[1,2]}}


def test_roundtrip_and_resume(job,tmp_path):
    model=Fake(); work=tmp_path/'work'; out=tmp_path/'bundle.zip'
    run_extraction(job,work,model,model_revision='test',batch_size=4)
    assert model.calls==4
    run_extraction(job,work,model,model_revision='test',batch_size=4)
    assert model.calls==4
    with pytest.raises(ValueError,match='incompatible'):
        run_extraction(job,work,model,model_revision='changed',batch_size=4)
    export_bundle(job,work,out); index=verify_bundle(out,job)
    assert len(index['objects'])==1
    install_bundle(out,job,tmp_path/'installed')
    assert (tmp_path/'installed'/index['objects'][0]['file']).exists()
    (work/index['objects'][0]['file']).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='corrupt'):
        run_extraction(job,work,model,model_revision='test',batch_size=4)


def test_reject_zip_traversal(job,tmp_path):
    with zipfile.ZipFile(job,'a') as z: z.writestr('../escape',b'x')
    with pytest.raises(ValueError,match='unsafe'):
        load_job(job)


def test_reject_tampered_feature(job,tmp_path):
    model=Fake(); work=tmp_path/'work'; bundle=tmp_path/'valid.zip'; bad=tmp_path/'bad.zip'
    run_extraction(job,work,model,model_revision='test'); export_bundle(job,work,bundle)
    with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(bad,'w') as dst:
        for name in src.namelist(): dst.writestr(name, b'changed' if name.endswith('.npz') else src.read(name))
    with pytest.raises(ValueError,match='hash mismatch'):
        install_bundle(bad,job,tmp_path/'dest')
    assert not (tmp_path/'dest').exists()


def test_reject_resealed_changed_pixels(job, tmp_path):
    with zipfile.ZipFile(job) as src:
        blobs={n:src.read(n) for n in src.namelist()}
    spec=json.loads(blobs['job.json']); view=spec['objects'][0]['views'][0]
    buf=io.BytesIO(); Image.fromarray(np.full((96,96,3), 255,dtype='uint8')).save(buf,format='PNG')
    blobs[view['path']]=buf.getvalue(); view['sha256']=sha_bytes(buf.getvalue())
    spec.pop('job_id'); spec['job_id']=sha_bytes(canonical(spec)); blobs['job.json']=canonical(spec)
    bad=tmp_path/'edited.zip'
    with zipfile.ZipFile(bad,'w') as z:
        for name,data in blobs.items(): z.writestr(name,data)
    with pytest.raises(ValueError,match='RGB view seal'):
        load_job(bad)


def test_reject_reordered_views(job, tmp_path):
    with zipfile.ZipFile(job) as src:
        blobs={n:src.read(n) for n in src.namelist()}
    spec=json.loads(blobs['job.json']); views=spec['objects'][0]['views']; views[0],views[1]=views[1],views[0]
    spec.pop('job_id'); spec['job_id']=sha_bytes(canonical(spec)); blobs['job.json']=canonical(spec)
    bad=tmp_path/'reordered.zip'
    with zipfile.ZipFile(bad,'w') as z:
        for name,data in blobs.items(): z.writestr(name,data)
    with pytest.raises(ValueError,match='view order'):
        load_job(bad)


def test_concurrent_writer_rejected(job,tmp_path):
    import fcntl
    work=tmp_path/'work'; work.mkdir()
    with (work/'.writer.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='another extraction'):
            run_extraction(job,work,Fake(),model_revision='test')


def test_invalid_declarative_mode():
    from research_grasping.phase4.orchestrator import validate_declarative
    path=Path(__file__).resolve().parents[2]/'research/phase4/automation/development.json'
    config=json.loads(path.read_text()); validate_declarative(config)
    config['mode']='confirmatory'
    with pytest.raises(ValueError,match='BLOCKED'): validate_declarative(config)


def test_retry_install_after_callback_failure_reuses_only_valid_files(job, tmp_path):
    model=Fake(); work=tmp_path/'work'; bundle=tmp_path/'valid.zip'; dest=tmp_path/'installed'
    run_extraction(job,work,model,model_revision='test'); export_bundle(job,work,bundle)
    index=install_bundle(bundle,job,dest)
    # Represents a callback failure after successful installation; retry is safe.
    assert install_bundle(bundle,job,dest) == index
    feature=dest/index['objects'][0]['file']; feature.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='corrupt'):
        install_bundle(bundle,job,dest)


def test_cli_retries_callback_after_interruption(job, tmp_path, monkeypatch, capsys):
    import runpy
    import sys
    import types
    work=tmp_path/'work'; bundle=tmp_path/'valid.zip'; dest=tmp_path/'installed'
    run_extraction(job,work,Fake(),model_revision='test'); export_bundle(job,work,bundle)
    script=Path(__file__).resolve().parents[2]/'tools/research/phase4/import_embedding_bundle.py'
    monkeypatch.setattr(sys,'argv',[str(script),'--job',str(job),'--bundle',str(bundle),'--destination',str(dest),'--expected-checkpoint-sha256','a'*64])
    callback=types.ModuleType('research_grasping.phase4.bundle_analysis')
    attempts=[]
    def run(*args, **kwargs):
        attempts.append(args[0])
        if len(attempts)==1: raise FileNotFoundError('test data temporarily unavailable')
    callback.run=run
    monkeypatch.setitem(sys.modules,'research_grasping.phase4.bundle_analysis',callback)
    with pytest.raises(SystemExit) as blocked:
        runpy.run_path(str(script),run_name='__main__')
    assert blocked.value.code==2 and dest.exists()
    runpy.run_path(str(script),run_name='__main__')
    assert attempts==[dest,dest]
    assert 'ANALYSIS BLOCKED' in capsys.readouterr().out


def test_retry_rejects_symlink_and_unexpected_file(job, tmp_path):
    work=tmp_path/'work'; bundle=tmp_path/'valid.zip'; dest=tmp_path/'installed'
    run_extraction(job,work,Fake(),model_revision='test'); export_bundle(job,work,bundle)
    index=install_bundle(bundle,job,dest)
    extra=dest/'unexpected'; extra.write_text('unexpected')
    with pytest.raises(ValueError,match='does not match'): install_bundle(bundle,job,dest)
    extra.unlink()
    feature=dest/index['objects'][0]['file']; copy=tmp_path/'copy.npz'
    copy.write_bytes(feature.read_bytes()); feature.unlink(); feature.symlink_to(copy)
    with pytest.raises(ValueError,match='symlink'): install_bundle(bundle,job,dest)
