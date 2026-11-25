import numpy as np
import pytest
from research_grasping.phase3.functional_features import geometric_features


def test_object_frame_relative_features_and_permutation():
    normals=np.array([[1.,0.,0.], [0.,1.,0.]])
    pose, local=geometric_features([0,0,.1], np.eye(3), .04, [.02,.03,.04], normals)
    _, swapped=geometric_features([0,0,.1], np.eye(3), .04, [.02,.03,.04], normals[::-1])
    assert len(pose)==13 and len(local)==10
    assert np.allclose(local, swapped)
    assert np.allclose(local[4:7], [.5,.5,0])


def test_reject_reflection_nonunit_and_invalid_scale():
    with pytest.raises(ValueError): geometric_features([0,0,0], np.diag([-1,1,1]), .04, [1,1,1], [[1,0,0]])
    with pytest.raises(ValueError): geometric_features([0,0,0], np.eye(3), .04, [1,1,1], [[2,0,0]])
    with pytest.raises(ValueError): geometric_features([0,0,0], np.eye(3), 0, [1,1,1], [[1,0,0]])


def test_joint_object_frame_rotation_preserves_local_summary():
    normals=np.array([[1.,0.,0.], [0.,1.,0.]])
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    _, original=geometric_features([.1,0,0], np.eye(3), .04, [.02,.03,.04], normals)
    _, rotated=geometric_features(rotation @ [.1,0,0], rotation, .04, [.02,.03,.04], normals @ rotation.T)
    assert np.allclose(original, rotated)
