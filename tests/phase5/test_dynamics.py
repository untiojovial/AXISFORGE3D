import numpy as np
import pytest

pytest.importorskip('mujoco')
from research_grasping.phase5 import dynamics  # noqa: E402

GRIPPER = dict(fingerWidth=.025, fingerDepth=.045, maxGripForce=90.)


def box(half):
    return np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)*half


def grasp(y=0., width=.04):
    xyz = lambda v: dict(zip('xyz', v))
    return dict(center=xyz([.4, y, .45]), approach=xyz([0, 0, -1]), closing=xyz([0, 1, 0]), width=width)


def run(candidate, **kw):
    pieces = [box(np.array([.02, .02, .03]))]
    return dynamics.trial(pieces, np.eye(3), np.array([.4, 0., .45]), .42, candidate, GRIPPER, **kw)


def test_a_centred_pinch_lifts_the_part():
    result = run(grasp())
    assert result['held'] and result['rise'] > .08 and not result['unstable']


def test_pads_that_miss_the_part_lift_nothing():
    result = run(grasp(y=.1))
    assert not result['held'] and result['rise'] < .01


def test_perturbations_are_fixed_and_deterministic():
    assert [n for n, _ in dynamics.PERTURBATIONS][0] == 'nominal' and len(dynamics.PERTURBATIONS) == 6
    assert run(grasp(), noise_seed=1) == run(grasp(), noise_seed=1)
