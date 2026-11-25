"""Boundary checks for native catalogue and external geometry inputs."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
trimesh = pytest.importorskip('trimesh')
from research_grasping.phase3 import dataset_mesh, native_catalog
from research_grasping.phase3.physics_frames import transform_from_axisforge


def test_native_catalog_rejects_invalid_id_timeout_and_response(tmp_path,monkeypatch):
    binary=tmp_path/'axisforge';binary.write_bytes(b'fixture')
    project=tmp_path/'project';project.write_text('{}')
    for bad in (-1,True,1.2,'2'):
        with pytest.raises(ValueError,match='nonnegative integer'):
            native_catalog.evaluate_candidate(binary,project,'body',bad)
    with pytest.raises(ValueError,match='timeout'):
        native_catalog.load_catalog(binary,project,'body',timeout_s=0)
    monkeypatch.setattr(native_catalog.subprocess,'run',lambda *args,**kw:SimpleNamespace(stdout='{}'))
    with pytest.raises(ValueError,match='unexpected AxisForge3D response'):
        native_catalog.load_catalog(binary,project,'body')
    monkeypatch.setattr(native_catalog.subprocess,'run',lambda *args,**kw:SimpleNamespace(stdout='[{"id":99}]'))
    with pytest.raises(ValueError,match='candidate identity mismatch'):
        native_catalog.evaluate_candidate(binary,project,'body',2)


def test_mesh_guards_units_format_scale_and_empty_geometry(tmp_path,monkeypatch):
    path=tmp_path/'object.obj';path.write_text('fixture')
    with pytest.raises(ValueError,match='Supported formats'):
        dataset_mesh.inspect_mesh(tmp_path/'object.xyz',units='m')
    with pytest.raises(ValueError,match='Explicit units'):
        dataset_mesh.inspect_mesh(path,units='in')
    with pytest.raises(ValueError,match='Invalid scale limits'):
        dataset_mesh.inspect_mesh(path,units='m',min_extent_m=2,max_extent_m=1)
    monkeypatch.setattr(trimesh,'load',lambda *a,**kw:trimesh.Scene())
    with pytest.raises(ValueError,match='Empty scene'):
        dataset_mesh.inspect_mesh(path,units='m')
    monkeypatch.setattr(trimesh,'load',lambda *a,**kw:trimesh.Trimesh(vertices=[],faces=[],process=False))
    with pytest.raises(ValueError,match='vertices and triangles'):
        dataset_mesh.inspect_mesh(path,units='m')
    corrupt=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,np.nan,0]],faces=[[0,1,2]],process=False)
    monkeypatch.setattr(trimesh,'load',lambda *a,**kw:corrupt)
    with pytest.raises(ValueError,match='Non-finite vertex'):
        dataset_mesh.inspect_mesh(path,units='m')


def test_reject_nonfinite_or_nonunit_pose():
    pose={'translation':dict(zip('xyz',[0.,0.,float('nan')])),
          'rotation':dict(zip('xyzw',[0.,0.,0.,1.]))}
    with pytest.raises(ValueError,match='non-finite'):
        transform_from_axisforge(pose)
    pose['translation']['z']=0
    pose['rotation']['w']=2
    with pytest.raises(ValueError,match='unit length'):
        transform_from_axisforge(pose)


@pytest.mark.parametrize('invalid', ['palm_geometry','joint_kind'])
def test_f3_scene_rejects_unrepresentable_gripper(invalid):
    pytest.importorskip('mujoco')
    from research_grasping.phase3.gso_contact_probe import scene
    root=Path(__file__).resolve().parents[2]
    project=json.loads((root/'examples/pick_and_place_cell.axisforge').read_text())
    robot=project['project']['robot'];gripper=robot['gripper']
    if invalid=='palm_geometry':
        palm=next(link for link in robot['links'] if link['id']==gripper['palmLinkId'])
        geom=next(g for g in palm['geometries'] if g['role']=='collision')
        geom['shape']['kind']='sphere'
        message='box collision geom'
    else:
        joint=next(j for j in robot['joints'] if j['id']==gripper['jawJointIds'][0])
        joint['kind']='revolute'
        message='unsupported jaw joint'
    pose={'translation':{'x':.72,'y':-.1,'z':.52},'rotation':{'x':0.,'y':0.,'z':0.,'w':1.}}
    tetra=b'v 0 0 0\nv .011 0 0\nv 0 .023 0\nv 0 0 .037\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'
    with pytest.raises(ValueError,match=message):
        scene(project,pose,{'fixture.obj':tetra},np.zeros(3),np.array([.42,-.1,.42]))
