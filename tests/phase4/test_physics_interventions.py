import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('mujoco')
from research_grasping.phase3.gso_contact_probe import scene
from research_grasping.phase4.physics_simulation import simulate as baseline
from research_grasping.phase4.physics_interventions import simulate as intervention


def test_zero_perturbation_reproduces_baseline():
    root = Path(__file__).resolve().parents[2]
    project = json.loads((root/'examples/pick_and_place_cell.axisforge').read_text())
    pose = {'translation': {'x': .72, 'y': -.1, 'z': .52}, 'rotation': {'x':0.,'y':0.,'z':0.,'w':1.}}
    assets = {'fixture.obj': b'v 0 0 0\nv .011 0 0\nv 0 .023 0\nv 0 0 .037\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'}
    args=(project,pose,assets,np.zeros(3),np.array([.42,-.1,.42]))
    first, data1, opening1=scene(*args)
    second, data2, opening2=scene(*args)
    assert baseline(first,data1,opening1) == intervention(second,data2,opening2,perturbation_n=0)
