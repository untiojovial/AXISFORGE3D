"""Compare MuJoCo kinematic tree against real AxisForge3D FK at two postures."""
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
mujoco = pytest.importorskip('mujoco')
from research_grasping.phase4.full_robot_kinematics import build_kinematic_model

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / 'target/release/axisforge'
PROJECT = ROOT / 'examples/pick_and_place_cell.axisforge'


@pytest.mark.skipif(not BINARY.exists(), reason='release CLI unavailable')
@pytest.mark.parametrize('joint_values', [{}, {'axis 1': .4, 'axis 2': -.3, 'axis 3': .7, 'gripper left jaw': .02}])
def test_full_robot_kinematics_against_native_cli(joint_values):
    model, names = build_kinematic_model(json.loads(PROJECT.read_text()))
    data = mujoco.MjData(model)
    args = [str(BINARY), 'fk', str(PROJECT)]
    for name, value in joint_values.items():
        args.extend(['--set', f'{name}={value}'])
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, data)
    native = json.loads(subprocess.check_output(args, text=True))
    for name, body in names.items():
        expected = native[name]
        np.testing.assert_allclose(data.xpos[model.body(body).id], expected[:3], atol=1e-10)
        np.testing.assert_allclose(data.xmat[model.body(body).id].reshape(3,3),
                                   Rotation.from_quat(expected[3:]).as_matrix(), atol=1e-10)
