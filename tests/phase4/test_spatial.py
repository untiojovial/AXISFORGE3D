import numpy as np
from research_grasping.phase4.spatial import image_affine, visible_patch


def test_resize_crop_padding_and_occlusion():
    depth=np.ones((8,8)); k=np.array([[1.,0,0],[0,1,0],[0,0,1.]])
    grid=np.arange(16).reshape(4,4,1); t=np.eye(4)
    # 8->16, crop 4 pixels, pad 4 pixels: content occupies centre 8x8.
    a=image_affine((8,8),(16,16),(4,4),(4,4))
    args=dict(pixel_transform=a,processed_size=(16,16),content_bounds=(3.5,3.5,11.5,11.5))
    assert visible_patch([2,2,1],depth,k,t,grid,**args)[0]==5
    assert visible_patch([0,0,1],depth,k,t,grid,**args) is None
    assert visible_patch([2,2,2],depth,k,t,grid,**args) is None


def test_rectangular_grid_and_bounds():
    depth=np.ones((8,12)); k=np.eye(3); grid=np.arange(6).reshape(2,3,1)
    kw=dict(pixel_transform=image_affine((8,12),(8,12)),processed_size=(8,12),content_bounds=(-.5,-.5,7.5,11.5))
    assert visible_patch([5,5,1],depth,k,np.eye(4),grid,**kw)[0]==4
    assert visible_patch([12,0,1],depth,k,np.eye(4),grid,**kw) is None


def test_phase3_compatible_and_invalid_depth():
    from research_grasping.phase3.vision_openclip import visible_patch as old
    depth=np.ones((96,96)); k=np.array([[20.,0,48],[0,20,48],[0,0,1.]])
    grid=np.arange(49).reshape(7,7,1); point=[0,0,1]
    kw=dict(pixel_transform=image_affine((96,96),(224,224)),processed_size=(224,224),content_bounds=(-.5,-.5,223.5,223.5))
    assert np.array_equal(visible_patch(point,depth,k,np.eye(4),grid,**kw),old(point,depth,k,np.eye(4),grid))
    assert np.array_equal(visible_patch(point,depth,k,np.eye(4),grid),old(point,depth,k,np.eye(4),grid))
    depth[48,48]=0
    assert visible_patch(point,depth,k,np.eye(4),grid,**kw) is None


def test_invalid_geometry_and_depth_edges():
    import pytest
    d=np.ones((8,8)); k=np.eye(3); t=np.eye(4); p=np.zeros((2,2,3))
    kw=dict(pixel_transform=np.eye(3),processed_size=(8,8),content_bounds=(-.5,-.5,7.5,7.5))
    with pytest.raises(ValueError): image_affine((0,8),(8,8))
    with pytest.raises(ValueError): visible_patch([0,0,1],d,k,t,p,pixel_transform=np.eye(2),processed_size=(8,8),content_bounds=(-.5,-.5,7.5,7.5))
    with pytest.raises(ValueError): visible_patch([0,0,1],d,k,t,p,**dict(kw,processed_size=(7,8)))
    with pytest.raises(ValueError): visible_patch([0,0,1],d,k,t,p,**dict(kw,content_bounds=(0,0,0,1)))
    with pytest.raises(ValueError): visible_patch([0,0,1],d,k,t,p,**dict(kw,tolerance=-1))
    assert visible_patch([0,0,-1],d,k,t,p,**kw) is None
    d[0,0]=np.nan; assert visible_patch([0,0,1],d,k,t,p,**kw) is None
    assert visible_patch([8,0,1],np.ones((8,8)),k,t,p,**kw) is None
