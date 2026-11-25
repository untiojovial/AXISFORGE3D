import numpy as np
import pytest
from research_grasping.phase3.physics_frames import grasp_in_object, transform_from_axisforge


def pose(x=0, y=0, z=0, q=(0,0,0,1)):
    return {'translation': dict(zip('xyz', (x,y,z))), 'rotation': dict(zip('xyzw',q))}


def test_identity_and_quarter_turn_roundtrip():
    root = pose(1,2,3,(0,0,2**-.5,2**-.5))
    grasp = pose(1,3,3,(0,0,2**-.5,2**-.5))
    local = grasp_in_object(root, grasp)
    assert np.allclose(local[:3,3], [1,0,0], atol=1e-12)
    assert np.allclose(local[:3,:3], np.eye(3), atol=1e-12)
    assert np.allclose(transform_from_axisforge(root) @ local, transform_from_axisforge(grasp))


def test_reject_invalid_quaternion():
    with pytest.raises(ValueError):
        transform_from_axisforge(pose(q=(0,0,0,2)))
