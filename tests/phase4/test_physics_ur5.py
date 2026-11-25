import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
mujoco = pytest.importorskip('mujoco')
from research_grasping.phase4.full_robot_ur5 import JOINTS, INITIAL, load_model, run

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT/'apps/viewer/samples/ur5/robot.urdf'
BINARY = ROOT/'target/release/axisforge'


@pytest.mark.skipif(not BINARY.exists(), reason='native CLI unavailable')
def test_ur5_fk_against_native(tmp_path):
    project = tmp_path/'ur5.axisforge'
    subprocess.run([BINARY,'import-urdf',URDF,'--out',project],check=True,capture_output=True)
    model,limits = load_model(URDF)
    data = mujoco.MjData(model)
    args = [BINARY,'fk',project]
    for name,value in zip(JOINTS,INITIAL):
        args += ['--set',f'{name}={value}']
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model,data)
    native = json.loads(subprocess.check_output(args,text=True))
    for name in ('shoulder_link','upper_arm_link','forearm_link','wrist_1_link','wrist_2_link','wrist_3_link'):
        expected = native[name]
        np.testing.assert_allclose(data.xpos[model.body(name).id],expected[:3],atol=1e-8)
        np.testing.assert_allclose(data.xmat[model.body(name).id].reshape(3,3),Rotation.from_quat(expected[3:]).as_matrix(),atol=1e-8)
    np.testing.assert_allclose(limits,[150,150,150,28,28,28])


def test_ur5_pd_respects_effort_and_control_has_effect():
    active = run(URDF)
    passive = run(URDF, controlled=False)
    torques = np.abs(np.array([r['torque_nm'] for r in active['trace']]))
    assert np.all(torques <= np.asarray(active['effort_limits_nm']) + 1e-12)
    assert all(np.isfinite(r['energy_j']).all() for r in active['trace'])
    assert passive['max_error_rad'] > active['max_error_rad']


def test_exact_model_compensation_uses_current_bias_and_clips():
    from research_grasping.phase4.full_robot_compensated import run as compensated
    model, limits = load_model(URDF)
    data = mujoco.MjData(model)
    joints = [model.joint(name).id for name in JOINTS]
    qadr = [model.jnt_qposadr[j] for j in joints]
    dadr = [model.jnt_dofadr[j] for j in joints]
    data.qpos[qadr] = INITIAL
    mujoco.mj_forward(model, data)
    expected_initial = np.clip(data.qfrc_bias[dadr], -limits, limits)
    result = compensated(URDF)
    np.testing.assert_allclose(result['trace'][0]['bias_nm'], data.qfrc_bias[dadr], atol=1e-10)
    np.testing.assert_allclose(result['trace'][0]['torque_nm'], expected_initial, atol=1e-10)
    torques = np.abs(np.array([row['torque_nm'] for row in result['trace']]))
    assert np.all(torques <= limits + 1e-10)
