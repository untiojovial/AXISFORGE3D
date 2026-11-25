"""Pre-evaluation object-frame pose and candidate-local geometric features."""
import numpy as np


def geometric_features(position_m, rotation_object_from_gripper, opening_m,
                       local_extent_m, proposed_normals_object):
    """Return pose and local blocks; normals must be pre-evaluation proposals.

    Rotation columns are gripper axes in the object frame. local_extent_m is
    the size of a preselected geometric region in the gripper frame, in metres.
    Region selection must not use evaluation results. This function does not
    establish cross-object correspondences or discover surface contacts.
    """
    p = np.asarray(position_m, float)
    r = np.asarray(rotation_object_from_gripper, float)
    extent = np.asarray(local_extent_m, float)
    normals = np.asarray(proposed_normals_object, float)
    if (p.shape != (3,) or r.shape != (3, 3) or extent.shape != (3,)
            or normals.ndim != 2 or normals.shape[1] != 3 or len(normals) == 0
            or not all(np.isfinite(x).all() for x in (p, r, extent, normals))
            or not np.isfinite(opening_m) or opening_m <= 0 or np.any(extent <= 0)
            or not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(r), 1, rtol=0, atol=1e-6)
            or not np.allclose(np.linalg.norm(normals, axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError('invalid SI dimensions, object-frame rotation or normals')
    # Mean and second moment are invariant to permutation of proposed normals,
    # and expressed relative to the gripper axes.
    local_normals = normals @ r
    mean = local_normals.mean(axis=0)
    second = (local_normals**2).mean(axis=0)
    pose = np.concatenate((p, r.ravel(), [opening_m]))
    local = np.concatenate(([opening_m], extent, mean, second))
    return pose, local
