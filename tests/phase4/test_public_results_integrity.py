"""The published summary must reject misaligned controls and produce reusable figures."""
import copy
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from research_grasping.phase3.experiment import load, save
from research_grasping.phase4 import embedding_report,model_comparison
from research_grasping.phase4.workflow import ROOT, implementation_inputs, main

PANEL=ROOT/'research/phase4/embedding_experiment/openclip_analysis.json'


@pytest.mark.parametrize('change,reason',[
    ('partial','full verified'),('drop_rep','representations'),('duplicate','duplicate trajectory'),
    ('drop_policy','baseline'),('wrong_seed','unpaired'),('omit_object','unpaired'),
    ('unpaired_initial','initial queries'),('bad_random','random control'),('bad_id','invalid measured')])
def test_invalid_comparisons_never_make_the_figure(tmp_path,change,reason):
    r=load(PANEL)
    block=r['results']['global_local']
    t=block['trajectories'][0]
    if change=='partial':r['status']='PARTIALLY_COMPLETED'
    if change=='drop_rep':r['results'].pop('global')
    if change=='duplicate':block['trajectories'].append(copy.deepcopy(t))
    if change=='drop_policy':block['trajectories']=[v for v in block['trajectories'] if v['method']!='contextual']
    if change=='wrong_seed':t['seed']=999
    if change=='omit_object':block['predictive'].pop(t['object_id'])
    if change=='unpaired_initial':t['selected'][0],t['selected'][1]=t['selected'][1],t['selected'][0]
    if change=='bad_random':next(v for v in block['trajectories'] if v['method']=='random')['auc']+=.01
    if change=='bad_id':t['selected'][0]=t['selected'][1]
    out=tmp_path/'result.json';save(out,r)
    with pytest.raises(ValueError,match=reason):embedding_report.summarize(out)


def test_exported_figure_uses_real_panel_and_cross_model_control(tmp_path):
    subprocess.run([sys.executable,'-m','research_grasping.phase4.embedding_report',
                    '--analysis',str(PANEL),'--out',str(tmp_path)],check=True)
    with Image.open(tmp_path/'openclip_panel.png') as image:
        assert image.width>=1000 and len(image.getbands())>=3
    a=load(PANEL);a['model_id']='test_variant'
    q=tmp_path/'other.json';save(q,a)
    assert model_comparison.compare([PANEL,q])['observed_models']==2
    a['results']['geometry']['trajectories'][1]['auc']+=.03;save(q,a)
    with pytest.raises(ValueError,match='geometry control'):model_comparison.compare([PANEL,q])
    with pytest.raises(ValueError,match='no verified'):model_comparison.compare([])
    dest=tmp_path/'comparison.json'
    subprocess.run([sys.executable,'-m','research_grasping.phase4.model_comparison','--inputs',str(PANEL),'--out',str(dest)],check=True)
    assert load(dest)['comparison_status']=='SINGLE_MODEL_ONLY'


def test_official_workflow_cli_refuses_unregistered_jobs_and_reuses_pinned_outputs(tmp_path,capsys):
    for options in (['--only','unknown'],['--threads','0']):
        with pytest.raises(SystemExit):main(['development','--work',str(tmp_path),*options])
    args=['development','--work',str(tmp_path),'--only','reference']
    assert main(args)==0
    assert json.loads(capsys.readouterr().out)['reference']['status']=='COMPLETED'
    assert main(args)==0
    assert json.loads(capsys.readouterr().out)['reference']['status']=='REUSED'
    path=ROOT/'research_grasping/phase4/bundle_analysis.py'
    deps=implementation_inputs([str(path)])
    assert str(ROOT/'research_grasping/phase4/bundle_io.py') in deps
    assert str(ROOT/'research_grasping/phase3/transfer_policy.py') in deps
