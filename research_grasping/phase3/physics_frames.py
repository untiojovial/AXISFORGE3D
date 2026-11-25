"""Explicit rigid transforms for the future native-to-dynamics bridge."""
from __future__ import annotations
import numpy as np


def transform_from_axisforge(pose):
    """Convert AxisForge translation and quaternion x,y,z,w to a 4x4 transform."""
    translation = np.array([pose['translation'][k] for k in 'xyz'], dtype=float)
    q = np.array([pose['rotation'][k] for k in 'xyzw'], dtype=float)
    if not np.isfinite(translation).all() or not np.isfinite(q).all():
        raise ValueError('non-finite pose')
    if not np.isclose(np.linalg.norm(q), 1.0, rtol=0, atol=1e-8):
        raise ValueError('quaternion must be unit length')
    x, y, z, w = q
    matrix = np.eye(4)
    matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = translation
    return matrix


def grasp_in_object(object_pose, grasp_pose):
    """Return T_object_grasp = inv(T_world_object) @ T_world_grasp."""
    world_object = transform_from_axisforge(object_pose)
    world_grasp = transform_from_axisforge(grasp_pose)
    return np.linalg.inv(world_object) @ world_grasp
