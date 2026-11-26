"""Phase VII: rotation criterion, conditions, folds, fallback bound, frozen MLPs, articulated model."""
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

from research_grasping.phase5 import catalogue
from research_grasping.phase6 import fusion
from research_grasping.phase7 import physics as v3, representation as rep

ROOT = Path(__file__).resolve().parents[2]


def tool(name):
    path = ROOT/'tools'/'research'/'phase7'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'phase7_{name}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cat(n=40, seed=0, feasible_rate=.15):
    rng = np.random.default_rng(seed)
    return {'label': np.array([f'c{i}' for i in range(n)]), 'preliminary': rng.random(n),
            'makeable': rng.random(n) < .8, 'center_offset': rng.random(n)*.05,
            'feasible': rng.random(n) < feasible_rate}


def test_rotation_criterion_needs_contact_and_small_turn():
    ok = dict(held=True, touching=True, turn_from_grip_deg=14.9)
    assert v3.held_at(ok, 15)
    assert not v3.held_at({**ok, 'turn_from_grip_deg': 15.1}, 15)
    assert not v3.held_at({**ok, 'touching': False}, 15)
    assert not v3.held_at({**ok, 'held': False}, 15)
    assert v3.held_at({**ok, 'turn_from_grip_deg': 19.}, 20) and not v3.held_at({**ok, 'turn_from_grip_deg': 9.}, 5)
    assert v3.TURN_LIMIT_DEG == 15.


def test_rotation_threshold_matches_the_linear_tolerance():
    # 15 degrees moves a point 80 mm from the closing axis by about 20 mm (the slip limit)
    assert abs(2*.08*np.sin(np.radians(15)/2) - v3.SLIP_MAX) < .0015


def test_conditions_use_intended_masses_and_keep_loads_apart():
    for mod in (tool('physics_dev'), tool('egad_confirm')):
        c = dict(mod.CONDITIONS)
        assert c['mass x2'] == dict(mass=.2) and c['mass x0.5'] == dict(mass=.05)
        assert [c[f'load {x} kg']['mass'] for x in ('0.5', '1', '2')] == [.5, 1., 2.]
    ec = tool('egad_confirm')
    assert set(ec.STANDARD).isdisjoint(ec.LOADS) and len(ec.STANDARD) == 6 and len(ec.LOADS) == 3


def test_grouped_folds_never_split_a_name_group():
    names = ['Mug_Red_Large', 'Mug_Red_Small', 'Mug_Blue', 'Bowl', 'Bowl_Big', 'Plate', 'Fork', 'Spoon', 'Knife', 'Cup']
    groups = rep.name_groups(names)
    folds = rep.grouped_folds(names, groups, k=3, seed=1)
    assert sorted(n for f in folds for n in f) == sorted(names)
    fold_of = {n: i for i, f in enumerate(folds) for n in f}
    for a in names:
        for b in names:
            if groups[a] == groups[b]:
                assert fold_of[a] == fold_of[b]
    assert groups['Mug_Red_Large'] == groups['Mug_Red_Small'] != groups['Mug_Blue']  # Jaccard 2/4 vs 1/3


@pytest.mark.parametrize('seed', range(20))
def test_interleaving_never_costs_more_than_twice_the_best(seed):
    nt = tool('negative_transfer')
    c = cat(seed=seed)
    if not c['feasible'].any():
        c['feasible'][-1] = True
    rng = np.random.default_rng(seed + 100)
    learned, default = catalogue.order(c, rng.random(len(c['label'])), 2.), catalogue.order(c)
    n_l, n_d = nt.first_viable_pos(c, learned), nt.first_viable_pos(c, default)
    n_h, idx, src = nt.interleave(c, learned, default)
    assert n_h <= 2*min(n_l, n_d)
    assert c['feasible'][idx]
    ec = tool('egad_confirm')
    assert ec.interleave(c, learned, default)[0] == n_h


def test_encoded_prior_reproduces_any_gated_order():
    c = cat(30, seed=3)
    rng = np.random.default_rng(4)
    target = fusion.order(c, rng.random(30))
    prior = fusion.encoded_prior(c, target)
    assert ((prior >= 0) & (prior <= 1)).all()
    assert list(catalogue.order(c, prior, fusion.PLANNER_MAX_WEIGHT)) == list(target)


def test_frozen_mlp_state_round_trips_exactly():
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(0)
    data = {f'p{i}': dict(v=rng.normal(size=(30, 16)).astype(np.float32), y=(rng.random(30) < .3).astype(np.float32),
                          u=rng.normal(size=35).astype(np.float32), d=rng.normal(size=64).astype(np.float32)) for i in range(8)}
    m = rep.MLPModel(geometry=True, visual=True, linear_visual=True, seed=0).fit(data, [f'p{i}' for i in range(6)], ['p6'])
    buf = io.BytesIO(); torch.save(m.state(), buf); buf.seek(0)
    m2 = rep.MLPModel.from_state(torch.load(buf, weights_only=True))
    np.testing.assert_array_equal(m.predict(data, ['p7'])['p7'], m2.predict(data, ['p7'])['p7'])


def test_macro_names_are_latex_safe():
    mt = tool('make_tables')
    M = mt.Macros()
    M('devParts', 151)
    with pytest.raises(AssertionError):
        M('dev2Parts', 1)


def test_articulated_model_builds_the_project_chain():
    mujoco = pytest.importorskip('mujoco')
    from research_grasping.phase7 import articulated as art
    ident = dict(translation=dict(x=0., y=0., z=0.), rotation=dict(x=0., y=0., z=0., w=1.))
    link = lambda i, name, geoms=(): dict(id=i, name=name, geometries=list(geoms), inertial=dict(
        mass=1., centerOfMass=dict(x=0., y=0., z=0.), inertiaDiagonal=dict(x=.01, y=.01, z=.01)))
    pad = dict(name='pad', role='collision', origin=ident, shape=dict(kind='box', size=dict(x=.02, y=.01, z=.04)))
    joint = lambda i, kind, p, c, axis: dict(id=i, name=i, kind=kind, parentLinkId=p, childLinkId=c, origin=ident,
                                             axis=dict(zip('xyz', axis)), limits=dict(lower=-2., upper=2.), maxEffort=100.)
    project = dict(robot=dict(rootLinkId='b', basePose=ident,
                              links=[link('b', 'base'), link('l1', 'l1'), link('palm', 'palm'), link('fl', 'fl', [pad]), link('fr', 'fr', [pad])],
                              joints=[joint('j1', 'revolute', 'b', 'l1', (0, 0, 1)), joint('mount', 'fixed', 'l1', 'palm', (0, 0, 1)),
                                      joint('jl', 'prismatic', 'palm', 'fl', (0, 1, 0)), joint('jr', 'prismatic', 'palm', 'fr', (0, -1, 0))],
                              gripper=dict(palmLinkId='palm', fingerLinkIds=['fl', 'fr'])))
    body, arm, jaws = art.robot_xml(project, '.5 .005 .0001')
    assert arm == ['j1'] and jaws == ['jl', 'jr']
    model = mujoco.MjModel.from_xml_string(f'<mujoco><compiler angle="radian"/><worldbody>{body}</worldbody></mujoco>')
    assert model.njnt == 3 and model.geom('padL').id >= 0 and model.geom('padR').id >= 0
    assert abs(model.jnt_range[model.joint('j1').id][1] - 2.) < 1e-9  # radians, not degrees
