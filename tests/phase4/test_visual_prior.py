import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from research_grasping.phase4 import visual_prior as vp

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT/'target/release/axisforge'


def grasp(label='g', centre=(0, 0, .05), approach=(0, 0, -1), closing=(1, 0, 0), width=.03):
    xyz = lambda v: dict(zip('xyz', map(float, v)))
    return dict(label=label, center=xyz(centre), approach=xyz(approach), closing=xyz(closing), width=width)


BOX = ([-.05, -.05, 0], [.05, .05, .1])


def test_identical_grasps_match_and_the_closing_axis_has_no_sign():
    a = vp.signatures([grasp()], *BOX)
    b = vp.signatures([grasp(closing=(-1, 0, 0))], *BOX)
    assert vp.affinity(a, a)[0, 0] == pytest.approx(1)
    assert vp.affinity(a, b)[0, 0] == pytest.approx(1)


def test_thirty_degrees_of_approach_is_one_unit_of_distance():
    t = np.deg2rad(30)
    a = vp.signatures([grasp()], *BOX)
    b = vp.signatures([grasp(approach=(np.sin(t), 0, -np.cos(t)))], *BOX)
    assert vp.affinity(a, b)[0, 0] == pytest.approx(np.exp(-.5))


def test_positions_are_relative_to_each_parts_box():
    small = vp.signatures([grasp(centre=(0, 0, .05))], *BOX)
    large = vp.signatures([grasp(centre=(0, 0, .1))], [-.1, -.1, 0], [.1, .1, .2])
    assert np.allclose(small['position'], large['position'])
    assert small['height'][0] != large['height'][0]


def entry(name, good=(), descriptor=None, low=BOX[0], high=BOX[1]):
    return dict(name=name, low=list(low), high=list(high), good=list(good), feasible=len(good),
                descriptors={} if descriptor is None else {'dinov2': list(descriptor)})


def test_neighbours_skip_the_query_and_unsolved_parts():
    q = entry('q', [grasp()], [1, 0])
    library = [q, entry('near', [grasp()], [.9, .1]), entry('far', [grasp()], [0, 1]),
               entry('unsolved', [], [1, 0])]
    visual = vp.neighbours('visual', q, library, k=1, model='dinov2')
    assert [e['name'] for e, _ in visual] == ['near']
    assert {e['name'] for e, _ in vp.neighbours('pooled', q, library)} == {'near', 'far'}
    assert vp.neighbours('random', q, library, seed=3) == vp.neighbours('random', q, library, seed=3)
    assert [e['name'] for e, _ in vp.neighbours('self', q, library)] == ['q']
    assert vp.neighbours('box', q, [q]) == []
    with pytest.raises(ValueError):
        vp.neighbours('psychic', q, library)


def test_box_neighbours_follow_the_part_size():
    q = entry('q', low=[0, 0, 0], high=[.1, .1, .1])
    library = [entry('same', [grasp()], low=[0, 0, 0], high=[.1, .1, .11]),
               entry('long', [grasp()], low=[0, 0, 0], high=[.3, .05, .05])]
    assert vp.neighbours('box', q, library, k=1)[0][0]['name'] == 'same'


def test_fusion_and_gate_weigh_likeness_and_size():
    q = entry('q', [grasp()], [1, 0], low=[0, 0, 0], high=[.1, .1, .1])
    unit = lambda c: [c, float(np.sqrt(1-c*c))]
    library = [q,
               entry('look-alike-giant', [grasp()], unit(1), low=[0, 0, 0], high=[.5, .5, .5]),
               entry('same-size-different', [grasp()], unit(0), low=[0, 0, 0], high=[.1, .1, .1]),
               entry('both-close', [grasp()], unit(.8), low=[0, 0, 0], high=[.11, .1, .1]),
               entry('middling', [grasp()], unit(.5), low=[0, 0, 0], high=[.3, .3, .3])]
    pick = lambda method, k: [e['name'] for e, _ in vp.neighbours(method, q, library, k=k, model='dinov2')]
    assert pick('visual', 1) == ['look-alike-giant']
    assert pick('box', 1) == ['same-size-different']
    # visual ranks 0..3: giant, both, middling, same; box ranks: same, both, middling, giant
    assert pick('fusion', 1) == ['both-close']
    assert pick('fusion', 3) == ['both-close', 'look-alike-giant', 'same-size-different']
    # gate: the 2k closest boxes, then the most similar of those
    assert pick('gate', 1) == ['both-close']
    assert pick('gate', 2) == ['look-alike-giant', 'both-close']
    assert all(w == 1. for _, w in vp.neighbours('fusion', q, library, model='dinov2'))


def test_the_prior_favours_the_candidate_like_the_good_grasp():
    cands = [grasp('top'), grasp('side', approach=(1, 0, 0), closing=(0, 1, 0)), grasp('top', width=.031)]
    doc = vp.prior(cands, *BOX, [(entry('n', [grasp()]), 1.)])
    scores = {c['label']: c['score'] for c in doc['candidates']}
    assert set(scores) == {'top', 'side'}  # one score per label, the best one
    assert scores['top'] == 1 and 0 <= scores['side'] < .01
    assert doc['weight'] == vp.PRIOR_WEIGHT
    assert vp.prior(cands, *BOX, []) == dict(candidates=[], weight=vp.PRIOR_WEIGHT)


def project(tmp_path, units='meters', body='part'):
    doc = dict(project=dict(units=dict(length=units), resources=[], environment=dict(bodies=[dict(
        name=body, pose=dict(translation=dict(x=.4, y=0, z=.4), rotation=dict(x=0, y=0, z=0, w=1)),
        geometries=[dict(role='both', origin=dict(translation=dict(x=0, y=0, z=.05),
                                                  rotation=dict(x=0, y=0, z=0, w=1)),
                         shape=dict(kind='box', size=dict(x=.1, y=.02, z=.1)))])])))
    path = tmp_path/'p.axisforge'
    path.write_text(json.dumps(doc))
    return path


def test_part_mesh_is_in_world_metres(tmp_path):
    v, f = vp.part_mesh(project(tmp_path), 'part')
    assert np.allclose(v.min(0), [.35, -.01, .4]) and np.allclose(v.max(0), [.45, .01, .5])
    assert f.shape == (12, 3)
    with pytest.raises(ValueError):
        vp.part_mesh(project(tmp_path), 'other')
    with pytest.raises(ValueError):
        vp.part_mesh(project(tmp_path, units='millimeters'), 'part')


@pytest.mark.parametrize('shape, low, high', [
    (dict(kind='sphere', radius=.02), [-.02]*3, [.02]*3),
    (dict(kind='cylinder', radius=.02, length=.1), [-.02, -.02, -.05], [.02, .02, .05]),
    (dict(kind='capsule', radius=.02, length=.1), [-.02, -.02, -.07], [.02, .02, .07])])
def test_primitives_fill_their_extent(shape, low, high):
    v, f = vp._primitive(shape)
    assert np.allclose(v.min(0), low, atol=1e-9) and np.allclose(v.max(0), high, atol=1e-9)
    assert f.min() >= 0 and f.max() < len(v)
    with pytest.raises(ValueError):
        vp._primitive(dict(kind='torus', radius=1))


def test_views_follow_the_phase_iii_recipe(tmp_path):
    v, f = vp.part_mesh(project(tmp_path), 'part')
    images = vp.render_views(v, f)
    assert len(images) == 16 and images[0].shape == (96, 96, 3) and images[0].dtype == np.uint8
    assert all(im.any() for im in images)


@pytest.mark.skipif(not BINARY.exists(), reason='release CLI not built')
def test_a_transferred_prior_reaches_the_native_planner(tmp_path):
    cell = tmp_path/'demo.axisforge'
    subprocess.run([str(BINARY), 'demo', '--out', str(cell)], check=True, capture_output=True)
    solved = vp.library_entry('cube', cell, 'cube', BINARY)
    assert solved['feasible'] > 0 and solved['candidates_total'] >= solved['feasible']
    v, _ = vp.part_mesh(cell, 'dowel')
    cands = vp.candidates(BINARY, cell, 'dowel')
    doc = vp.prior(cands, v.min(0), v.max(0), [(solved, 1.)])
    assert len(doc['candidates']) == len({c['label'] for c in cands})
    with_prior = vp.plan_with(BINARY, cell, 'dowel', doc, 1, tmp_path)
    without = vp.plan_with(BINARY, cell, 'dowel', None, 1, tmp_path)
    assert with_prior['matched'] == len(doc['candidates']) and with_prior['unmatched'] == 0
    assert with_prior['found'] == without['found'] == 1


def test_a_fixed_camera_distance_keeps_the_scale(tmp_path):
    v, f = vp._box_mesh(dict(x=.05, y=.05, z=.05))
    area = lambda images: np.mean([im.any(-1).sum() for im in images])
    normalised = [area(vp.render_views(v*s, f, count=4)) for s in (1, 2)]
    kept = [area(vp.render_views(v*s, f, count=4, size=128, distance=.6)) for s in (1, 2)]
    assert abs(normalised[1]/normalised[0]-1) < .05
    assert 3.5 < kept[1]/kept[0] < 4.5
    with pytest.raises(ValueError):
        vp.render_views(v*10, f, distance=.1)


def test_descriptors_attach_without_solving_again():
    library = [entry('a', descriptor=[1, 0]), entry('b', descriptor=[0, 1])]
    vp.add_descriptors(library, {'a': {'metric': np.array([.6, .8])}})
    assert library[0]['descriptors'] == {'dinov2': [1, 0], 'metric': [.6, .8]}
    assert library[1]['descriptors'] == {'dinov2': [0, 1]}
