"""Check the scientific identity of the distributed result against its distributed input."""
import copy
import io
import json
from pathlib import Path
import zipfile

import pytest
from PIL import Image

from research_grasping.phase4 import bundle_io
from research_grasping.phase4.jobs import canonical,sha_bytes,sha_file

ROOT=Path(__file__).resolve().parents[2]/'research/phase4'
JOB=ROOT/'embedding_experiment/development-images.zip'
BUNDLE=ROOT/'embedding_experiment/openclip_development.zip'


def write_zip(path,items):
    with zipfile.ZipFile(path,'w') as archive:
        for key,value in items.items():archive.writestr(key,value)


@pytest.fixture(scope='module')
def first_object():
    with zipfile.ZipFile(JOB) as archive:
        job=json.loads(archive.read('job.json'))
        job['objects']=job['objects'][:1]
        images={v['path']:archive.read(v['path']) for v in job['objects'][0]['views']}
    job.pop('job_id')
    job['job_id']=sha_bytes(canonical(job))
    with zipfile.ZipFile(BUNDLE) as archive:
        index=json.loads(archive.read('index.json'))
        index['objects']=[r for r in index['objects'] if r['object_id']==job['objects'][0]['object_id']]
        feature=index['objects'][0]['file']
        data={feature:archive.read(feature)}
    index['job_id']=index['config']['job_id']=job['job_id']
    index['run_id']=sha_bytes(canonical(index['config']))
    return job,images,index,data


def test_published_artifact_matches_the_independent_reproduction_report():
    evidence=json.loads((ROOT/'automation/openclip_reproduction.json').read_text())
    assert sha_file(BUNDLE)==evidence['repeat_bundle_sha256']
    index=bundle_io.verify_bundle(BUNDLE,JOB)
    assert len(index['objects'])==22 and index['job_id']==evidence['job_id']
    assert index['checkpoint_sha256']==evidence['checkpoint_sha256']


@pytest.mark.parametrize('change,reason',[
    ('missing','missing job'),('empty','unexpected job'),('split','invalid object'),('order','view order'),
    ('pixels','RGB view seal'),('mode','invalid image'),('manifest_lock','sealed lock'),('rgb_lock','RGB seal')])
def test_input_cannot_relabel_or_replace_the_views(tmp_path,first_object,monkeypatch,change,reason):
    original,images,_,_=first_object
    job=copy.deepcopy(original);images=dict(images)
    if change=='empty':job['objects']=[]
    elif change=='split':job['objects'][0]['split']='test'
    elif change=='order':job['objects'][0]['views'].reverse()
    elif change in ('pixels','mode'):
        view=job['objects'][0]['views'][0]
        canvas=Image.new('RGB' if change=='pixels' else 'L',(96,96))
        buf=io.BytesIO();canvas.save(buf,format='PNG')
        images[view['path']]=buf.getvalue();view['sha256']=sha_bytes(buf.getvalue())
    elif change in ('manifest_lock','rgb_lock'):
        local=tmp_path/'changed.json';local.write_text('{}')
        monkeypatch.setattr(bundle_io,'LOCK' if change=='manifest_lock' else 'RGB_LOCK',local)
    job.pop('job_id');job['job_id']=sha_bytes(canonical(job))
    if change!='missing':images['job.json']=canonical(job)
    path=tmp_path/'changed.zip';write_zip(path,images)
    with pytest.raises(ValueError,match=reason):bundle_io.load_job(path)


@pytest.mark.parametrize('change,reason',[
    ('missing','missing index'),('other_job','bundle/job'),('no_model','missing model provenance'),
    ('configuration','inconsistent provenance'),('run_id','configuration hash'),('weight_digest','invalid provenance hash'),
    ('split','invalid bundle object'),('empty','incomplete or unexpected'),('no_features','invalid bundle object')])
def test_output_identity_must_match_the_job_even_for_structurally_valid_features(tmp_path,first_object,change,reason):
    job,images,original,data=first_object
    job_path=tmp_path/'job.zip';write_zip(job_path,{**images,'job.json':canonical(job)})
    index=copy.deepcopy(original);files=dict(data)
    if change=='other_job':index['job_id']='another-job'
    elif change=='no_model':index['model_id']=''
    elif change=='configuration':index['config']['model_revision']='unrelated'
    elif change=='run_id':index['run_id']='changed'
    elif change=='weight_digest':
        index['checkpoint_sha256']=index['config']['checkpoint_sha256']='unresolved'
        index['run_id']=sha_bytes(canonical(index['config']))
    elif change=='split':index['objects'][0]['split']='test'
    elif change=='empty':index['objects']=[]
    elif change=='no_features':files.clear()
    if change!='missing':files['index.json']=canonical(index)
    path=tmp_path/'result.zip';write_zip(path,files)
    with pytest.raises(ValueError,match=reason):bundle_io.verify_bundle(path,job_path)
