import numpy as np
from research_grasping.phase3.render_cpu import render, cameras
from research_grasping.representations.geometry import project


def test_plane_occlusion_roundtrip():
    v=np.array([[-1,-1,2],[1,-1,2],[1,1,2],[-1,1,2],[-.25,-.25,1],[.25,-.25,1],[0,.25,1]])
    f=np.array([[0,1,2],[0,2,3],[4,5,6]])
    k=np.array([[8.,0,4],[0,8,4],[0,0,1]])
    out=render(v,f,k,np.eye(4),9,9)
    assert out['depth'][4,4] == 1 and out['depth'][1,1] == 2
    uv,z=project(out['points_world'],k,np.eye(4))
    np.testing.assert_allclose(uv,out['pixels'],atol=1e-12)
    np.testing.assert_allclose(z,out['depth'][out['mask']])
    np.testing.assert_allclose(out['normals_world'][4,4],[0,0,1])


def test_cameras_and_scaling():
    b=np.array([[-.1,-.2,-.3],[.1,.2,.3]])
    for n in (1,4,8,16):
        k,ts=cameras(b,n); assert len(ts)==n
        assert all(np.isclose(np.linalg.det(t[:3,:3]),1) for t in ts)
    _,s=cameras(b*2,1); _,t=cameras(b,1)
    np.testing.assert_allclose(s[0][:3,3],2*t[0][:3,3])


def test_sloped_depth_winding_rotation_and_near_plane():
    import pytest
    v=np.array([[-1.,-1,2],[2,-2,4],[0,2,4]])
    k=np.array([[8.,0,4],[0,8,4],[0,0,1]])
    f=np.array([[0,1,2]])
    a=render(v,f,k,np.eye(4),9,9)
    # Projected vertices (0,0),(8,0),(4,8); weights at centre .25,.25,.5.
    np.testing.assert_allclose(a['depth'][4,4], 1/(.25/2+.25/4+.5/4))
    reverse=render(v,np.array([[0,2,1]]),k,np.eye(4),9,9)
    np.testing.assert_allclose(a['depth'],reverse['depth'],equal_nan=True)
    np.testing.assert_allclose(a['normals_world'][4,4],-reverse['normals_world'][4,4])
    angle=.7; rot=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    t=np.eye(4); t[:3,:3]=rot; t[:3,3]=[1,2,3]
    transformed=v@rot.T+t[:3,3]
    b=render(transformed,f,k,t,9,9)
    np.testing.assert_allclose(a['depth'],b['depth'],atol=1e-12,equal_nan=True)
    with pytest.raises(ValueError): render(np.vstack((v,[0,0,-1])),f,k,np.eye(4),9,9)
