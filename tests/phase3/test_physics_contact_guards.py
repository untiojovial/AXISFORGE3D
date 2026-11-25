"""Phase-III proxy loading and grasp-simulation boundary cases."""
import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('mujoco')
from research_grasping.phase3.gso_contact_probe import source_parts, scene, simulate

ROOT = Path(__file__).resolve().parents[2]
TETRA = b'v 0 0 0\nv .011 0 0\nv 0 .023 0\nv 0 0 .037\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'


def source_fixture(root, mesh_attrs='', geom_attrs=''):
    folder = root/'models'/'fixture'
    folder.mkdir(parents=True)
    (folder/'part.obj').write_bytes(TETRA)
    (folder/'model.xml').write_text(f'<mujoco><asset><mesh name="part" file="part.obj" {mesh_attrs}/></asset>'
                                  f'<worldbody><body><geom type="mesh" mesh="part" {geom_attrs}/></body></worldbody></mujoco>')
    return folder


def test_proxy_loads_tetra_and_rejects_unhandled_transforms(tmp_path):
    folder = source_fixture(tmp_path)
    assets,hulls = source_parts(tmp_path,'fixture')
    assert assets == {'part.obj':TETRA} and len(hulls)==1
    assert hulls[0].shape[1] == 4
    (folder/'model.xml').write_text('<mujoco><asset><mesh name="part" file="part.obj" scale="2 2 2"/></asset>'
                                   '<worldbody><body><geom type="mesh" mesh="part"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError,match='unsupported source collision transform'):
        source_parts(tmp_path,'fixture')


def test_proxy_rejects_path_escape_and_missing_contact_geometry(tmp_path):
    folder = source_fixture(tmp_path)
    (folder/'model.xml').write_text('<mujoco><asset><mesh name="part" file="../part.obj"/></asset>'
                                   '<worldbody><body><geom type="mesh" mesh="part"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError,match='unexpected collision part'):
        source_parts(tmp_path,'fixture')
    (folder/'model.xml').write_text('<mujoco><asset><mesh name="part" file="part.obj"/></asset>'
                                   '<worldbody><body><geom type="mesh" mesh="part" contype="0"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError,match='no contact geometry'):
        source_parts(tmp_path,'fixture')


def test_scene_rejects_unsupported_effector_and_open_control_cannot_hold():
    project = json.loads((ROOT/'examples/pick_and_place_cell.axisforge').read_text())
    pose = {'translation':{'x':.72,'y':-.1,'z':.52},'rotation':{'x':0.,'y':0.,'z':0.,'w':1.}}
    args = (pose,{'part.obj':TETRA},np.zeros(3),np.array([.42,-.1,.42]))
    invalid = json.loads(json.dumps(project))
    invalid['project']['robot']['gripper']['jawJointIds'] = []
    with pytest.raises(ValueError,match='parallel gripper'):
        scene(invalid,*args)
    model,data,opening = scene(project,*args,mass=.1,friction=.5)
    assert model.body_mass[model.body('object').id] == pytest.approx(.1)
    result = simulate(model,data,opening,close_jaws=False)
    assert not result['sustained'] and not result['held']
    assert result['table_hold_fraction'] > .99


def test_probe_main_runs_fixture_and_preserves_narrow_denominator(tmp_path, monkeypatch):
    import hashlib
    import sys
    from types import SimpleNamespace
    from scipy.spatial import ConvexHull
    from research_grasping.phase3 import gso_contact_probe as probe

    source = tmp_path/'source'
    folder = source/'models'/'fixture'
    folder.mkdir(parents=True)
    (folder/'model.obj').write_bytes(TETRA)
    (folder/'part.obj').write_bytes(TETRA)
    (folder/'model.xml').write_text('<mujoco><asset><mesh name="p" file="part.obj"/></asset>'
                                   '<worldbody><body><geom type="mesh" mesh="p"/></body></worldbody></mujoco>')
    run = tmp_path/'run'/'runs'/'fixture'
    run.mkdir(parents=True)
    (run/'observations.json').write_text(json.dumps({'rows':[{'candidate_id':0,'status':'ok'}]}))
    project = ROOT/'examples/pick_and_place_cell.axisforge'
    metadata={'source_origin_m':[0,0,0],'base_placement_world_m':[.42,-.1,.42],
              'project_sha256':hashlib.sha256(project.read_bytes()).hexdigest()}
    class FixtureGuard:
        manifest_digest='synthetic-fixture'
        objects={'fixture':{'original_sha256':hashlib.sha256(TETRA).hexdigest()}}
        def __init__(self,*args): pass
        def catalogue(self,*args):return {'metadata':metadata,'project':str(project),'body':'fixture'}
    monkeypatch.setattr(probe,'DevelopmentData',FixtureGuard)
    monkeypatch.setattr(probe.subprocess,'check_output',lambda *a,**k:probe.REV+'\n')
    monkeypatch.setattr(probe.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0))
    record={'plan':{'fixture':True},'quality':{'quality':.25},'candidate':{'contacts':[
        {'point':{'x':.42,'y':-.1,'z':.42},'normal':{'x':1,'y':0,'z':0}}]},
        'graspPose':{'translation':{'x':.72,'y':-.1,'z':.52},'rotation':{'x':0.,'y':0.,'z':0.,'w':1.}}}
    monkeypatch.setattr(probe,'evaluate_candidate',lambda *a,**k:{'records':[record]})
    (tmp_path/'lock.json').write_text('{}')
    (tmp_path/'cli').write_bytes(b'fixture-cli')
    output=tmp_path/'result.json'
    argv=['probe','--catalogues',str(tmp_path/'catalogs'),'--run',str(tmp_path/'run'),
          '--source',str(source),'--binary',str(tmp_path/'cli'),'--data',str(tmp_path),
          '--protocol',str(tmp_path),'--out',str(output)]
    monkeypatch.setattr(sys,'argv',argv)
    probe.main()
    result=json.loads(output.read_text())
    assert len(result['objects'])==1 and len(result['objects'][0]['measurements'])==1
    measured=result['objects'][0]['measurements'][0]
    assert measured['candidate_id']==0 and measured['f2_epsilon']==.25
    assert not measured['open_jaws_control']['sustained']
    assert len(measured['sensitivity'])==4
