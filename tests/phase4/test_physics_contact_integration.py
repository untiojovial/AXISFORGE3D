"""A native cube plan passes through the proxy contact measurement code."""
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
pytest.importorskip('mujoco')
from research_grasping.phase3.gso_contact_probe import scene
from research_grasping.phase4.physics_simulation import simulate
from research_grasping.phase4.physics_interventions import simulate as simulate_intervention

ROOT=Path(__file__).resolve().parents[2]
BINARY=ROOT/'target/release/axisforge'
PROJECT=ROOT/'examples/pick_and_place_cell.axisforge'


def cube_obj():
    vertices=[(-.02,-.02,0),(.02,-.02,0),(.02,.02,0),(-.02,.02,0),
              (-.02,-.02,.04),(.02,-.02,.04),(.02,.02,.04),(-.02,.02,.04)]
    faces=[(1,3,2),(1,4,3),(5,6,7),(5,7,8),(1,2,6),(1,6,5),
           (2,3,7),(2,7,6),(3,4,8),(3,8,7),(4,1,5),(4,5,8)]
    return (''.join('v %s %s %s\n'%v for v in vertices)+''.join('f %s %s %s\n'%f for f in faces)).encode()


@pytest.mark.skipif(not BINARY.exists(),reason='release AxisForge CLI unavailable')
def test_native_candidate_and_proxy_contact_are_observed():
    report=json.loads(subprocess.check_output([BINARY,'plan',PROJECT,'cube','--json'],text=True))
    assert report['plans']
    plan=report['plans'][0]
    args=(json.loads(PROJECT.read_text()),plan['graspPose'],{'cube.obj':cube_obj()},
          np.zeros(3),np.array([.42,-.1,.42]))
    results=[]
    for evaluate in (simulate,simulate_intervention):
        model,data,opening=scene(*args)
        result=evaluate(model,data,opening)
        assert result['numerical_valid']
        assert result['max_object_penetration_all_m'] > 0
        assert not result['sustained']
        results.append(result)
    assert results[0] == results[1]
