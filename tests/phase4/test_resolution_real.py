"""One real object CPU recheck, optional external assets, no new test objects."""
import json, os, shutil, sys
from pathlib import Path
import pytest


def test_real_one_view_resolution_cpu(tmp_path,monkeypatch):
    source=os.getenv('AXISFORGE_PHASE3_DATA'); checkpoint=os.getenv('AXISFORGE_OPENCLIP_CHECKPOINT')
    if not source or not checkpoint: pytest.skip('requires authorized local development data and OpenCLIP')
    pytest.importorskip('torch'); pytest.importorskip('trimesh')
    from research_grasping.phase4 import resolution_study as study
    source=Path(source); name='ACE_Coffee_Mug_Kristen_16_oz_cup'
    manifest=json.loads((source/'manifest.json').read_text()); obj=next(x for x in manifest['objects'] if x['object_id']==name); assert obj['split']=='train'
    root=tmp_path/'data';root.mkdir();(root/'manifest.json').write_text(json.dumps({'objects':[obj]}))
    dest=root/obj['visual_mesh'];dest.parent.mkdir(parents=True);shutil.copy(source/obj['visual_mesh'],dest)
    archive=root/'vision-development-v3';archive.mkdir();shutil.copy(source/'vision-development-v3'/(name+'.npz'),archive/(name+'.npz'))
    monkeypatch.setattr(study,'PREPILOT_MANIFEST',study.sha(root/'manifest.json'))
    out=tmp_path/'result.json'; monkeypatch.setattr(sys,'argv',['resolution_study','--root',str(root),'--checkpoint',checkpoint,'--out',str(out)])
    study.main(); result=json.loads(out.read_text()); assert len(result['results'])==1
    artifact=json.loads((Path(__file__).resolve().parents[2]/'research/phase4/vision/resolution_96_224_patches.json').read_text())
    expected=next(x['cosine'] for x in artifact['results'] if x['object_id']==name)
    assert result['results'][0]['cosine']==pytest.approx(expected,abs=2e-6)
    obj['split']='test';(root/'manifest.json').write_text(json.dumps({'objects':[obj]}))
    with pytest.raises(ValueError,match='manifest lock'):study.main()
