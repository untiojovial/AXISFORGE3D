"""Independent SE(3) and pad-face checks for the development contact probe."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

mujoco = pytest.importorskip('mujoco')
from research_grasping.phase3.gso_contact_probe import scene


def matrix(pose):
    """Use scipy directly as independent reference to AxisForge conversion."""
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([pose['rotation'][k] for k in 'xyzw']).as_matrix()
    out[:3, 3] = [pose['translation'][k] for k in 'xyz']
    return out


def test_asymmetric_frames_and_finger_pad_faces():
    project = json.loads((Path(__file__).resolve().parents[2] / 'examples/pick_and_place_cell.axisforge').read_text())
    quaternion = Rotation.from_euler('xyz', [0.31, -0.47, 0.82]).as_quat()
    pose = {'translation': dict(zip('xyz', [0.47, -0.09, 0.51])),
            'rotation': dict(zip('xyzw', quaternion))}
    # The mesh is only needed for valid model construction; this tetrahedron is asymmetric.
    assets = {'fixture.obj': b'v 0 0 0\nv .011 0 0\nv 0 .023 0\nv 0 0 .037\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'}
    source_origin = np.array([.13, -.07, .021])
    placement = np.array([.42, -.10, .42])
    model, data, opening = scene(project, pose, assets, source_origin, placement)
    robot = project['project']['robot']
    gripper = robot['gripper']
    joints = {j['id']: j for j in robot['joints']}
    links = {link['id']: link for link in robot['links']}
    frame = next(f for f in robot['toolFrames'] if f['id'] == gripper['graspFrameId'])
    world_grasp = matrix(pose)
    approach = world_grasp[:3, :3] @ np.array([gripper['approachAxis'][k] for k in 'xyz'])
    expected_palm = world_grasp @ np.linalg.inv(matrix(frame['offset']))
    expected_palm[:3, 3] -= np.array([0, 0, placement[2]]) + .06 * approach
    np.testing.assert_allclose(data.xpos[model.body('palm').id], expected_palm[:3,3], atol=2e-10)
    np.testing.assert_allclose(data.xmat[model.body('palm').id].reshape(3,3), expected_palm[:3,:3], atol=2e-10)
    # An off-axis point catches axis permutations and quaternion convention errors.
    point = np.array([.013, -.021, .007, 1.])
    actual = np.r_[data.xmat[model.body('palm').id].reshape(3,3) @ point[:3] + data.xpos[model.body('palm').id], 1.]
    np.testing.assert_allclose(actual, expected_palm @ point, atol=2e-10)
    closing_axis = expected_palm[:3,:3] @ np.array([gripper['closingAxis'][k] for k in 'xyz'])
    face_centers = []
    for i, joint_id in enumerate(gripper['jawJointIds']):
        joint = joints[joint_id]
        geom = next(g for g in links[joint['childLinkId']]['geometries'] if g['role'] == 'collision')
        displacement = np.eye(4)
        displacement[:3,3] = opening * np.array([joint['axis'][k] for k in 'xyz'])
        expected_geom = expected_palm @ matrix(joint['origin']) @ displacement @ matrix(geom['origin'])
        geom_id = model.geom(f'jaw{i}_geom').id
        actual_rotation = data.geom_xmat[geom_id].reshape(3,3)
        np.testing.assert_allclose(data.geom_xpos[geom_id], expected_geom[:3,3], atol=2e-10)
        np.testing.assert_allclose(actual_rotation, expected_geom[:3,:3], atol=2e-10)
        half_sizes = np.array([geom['shape']['size'][k] for k in 'xyz']) / 2
        # Select the face pointing toward the palm's closing-axis centre.
        local_direction = actual_rotation.T @ closing_axis
        axis = int(np.argmax(abs(local_direction)))
        inward_sign = -np.sign(np.dot(data.geom_xpos[geom_id] - data.xpos[model.body('palm').id], closing_axis) * local_direction[axis])
        local_face = np.zeros(3)
        local_face[axis] = inward_sign * half_sizes[axis]
        face_centers.append(data.geom_xpos[geom_id] + actual_rotation @ local_face)
        np.testing.assert_allclose(face_centers[-1], (expected_geom @ np.r_[local_face,1])[:3], atol=2e-10)
    assert np.dot(face_centers[1] - face_centers[0], closing_axis) != pytest.approx(0)
    expected_object = placement - source_origin - [0,0,placement[2]]
    np.testing.assert_allclose(data.xpos[model.body('object').id], expected_object, atol=1e-12)
