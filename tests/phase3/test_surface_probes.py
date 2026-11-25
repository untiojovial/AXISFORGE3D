import numpy as np
import pytest
from scipy.spatial import ConvexHull,cKDTree
from research_grasping.phase3.gso_contact_probe import contact_ray_error
from research_grasping.phase3.contact_similarity import local_descriptor,distances,evaluate
from research_grasping.phase3.source_appearance import triangles_uv,atlas_samples


def test_ray_uses_exterior_of_union():
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    hulls=[ConvexHull(corners).equations,ConvexHull(corners+[1,0,0]).equations]
    assert contact_ray_error(np.array([.9,0,0]),[1,0,0],hulls)==pytest.approx(1.1)
    assert contact_ray_error(np.array([.9,4,0]),[1,0,0],hulls) is None
    assert contact_ray_error(np.array([2.002,0,0]),[1,0,0],hulls)==pytest.approx(.002)


def test_real_surface_uv_visibility(tmp_path):
    pytest.importorskip('trimesh')
    path=tmp_path/'test.obj'
    path.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n')
    triangles,uv=triangles_uv(path)
    atlas=np.array([[[255,0,0],[0,0,0]],[[0,255,0],[0,0,255]]],np.uint8)
    rgb,ok=atlas_samples(np.array([[.9,.01,0],[.01,.9,0],[.1,.1,.04],[2,2,0]]),triangles,uv,atlas,cKDTree(triangles.mean(axis=1)))
    assert ok.tolist()==[True,True,False,False]
    assert rgb[:2].tolist()==[[0,0,255],[255,0,0]]
    path.write_text('f 1/1 2/2 3/3 4/4')
    with pytest.raises(ValueError):triangles_uv(path)


def test_visible_local_and_constant_ties():
    grid=np.zeros((1,7,7,3));grid[:]=[0,2,0]
    surface=np.ones((1,7,7))
    v,n=local_descriptor([[0.,0.,1.]],np.eye(3),surface,[np.eye(4)],grid)
    assert n==1 and np.array_equal(v,[0,1,0])
    assert local_descriptor([],np.eye(3),surface,[np.eye(4)],grid)[1]==0
    feat=dict(geometry=[.7],global_embedding=[0.,1.],local=v)
    rows=[dict(status='ok',details=dict(epsilon=i)) for i in range(4)]
    d=distances([feat]*4,[feat])
    assert d['local_visual'].shape==(1,4)
    result=evaluate([feat]*4,[feat],rows,rows[:1])
    assert result['equal_fusion']['epsilon_predictions']==[1.5]
    assert result['equal_fusion']['distance_vs_epsilon_delta'] is None
