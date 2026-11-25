import numpy as np
import pytest
from research_grasping.phase3.vision_openclip import pool_views, load_model


def test_pool_and_hash_fail_closed(tmp_path):
    v=np.tile([3.,4.],(16,1))
    for n in (1,4,8,16): np.testing.assert_allclose(pool_views(v,n),[.6,.8])
    with pytest.raises(ValueError): pool_views(v,2)
    p=tmp_path/'model.pt'; p.write_bytes(b'wrong')
    with pytest.raises(ValueError): load_model(p)


def test_reject_invalid_rgb():
    from research_grasping.phase3.vision_openclip import encode_images
    with pytest.raises(ValueError): encode_images(None, None, [np.zeros((4,4), np.uint8)])


def test_visible_patch_occlusion_and_bounds():
    from research_grasping.phase3.vision_openclip import visible_patch
    depth=np.ones((7,7)); k=np.array([[1.,0,3],[0,1,3],[0,0,1]])
    patches=np.arange(49).reshape(7,7,1)
    assert visible_patch([0,0,1],depth,k,np.eye(4),patches)[0] == 24
    assert visible_patch([0,0,2],depth,k,np.eye(4),patches) is None
    assert visible_patch([100,0,1],depth,k,np.eye(4),patches) is None
