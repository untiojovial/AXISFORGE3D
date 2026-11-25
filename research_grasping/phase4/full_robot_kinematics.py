"""Kinematic-only project tree in MuJoCo, not an arm dynamics model.

Mass and inertia placeholders exist solely for compilation; collision geometry,
actuators and robot dynamics are deliberately not represented here.
"""
from __future__ import annotations
from xml.etree import ElementTree as ET


def _xyz(value):
    return ' '.join(str(value[axis]) for axis in 'xyz')


def _pose(pose):
    q = pose['rotation']
    return {'pos': _xyz(pose['translation']),
            'quat': ' '.join(str(q[axis]) for axis in 'wxyz')}


def build_kinematic_model(project):
    """Build a named, fixed-base MuJoCo model from supported project joints."""
    import mujoco
    robot = project['project']['robot']
    links = {link['id']: link for link in robot['links']}
    children = {key: [] for key in links}
    for joint in robot['joints']:
        if joint['kind'] not in ('fixed', 'revolute', 'prismatic'):
            raise ValueError('unsupported joint kind')
        children[joint['parentLinkId']].append(joint)
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', angle='radian')
    world = ET.SubElement(root, 'worldbody')
    names = {}

    def add(parent, link_id, pose, incoming=None):
        link = links[link_id]
        if link['name'] in names:
            raise ValueError('link names must be unique')
        body_name = f'link_{len(names)}'
        names[link['name']] = body_name
        body = ET.SubElement(parent, 'body', name=body_name, **_pose(pose))
        if incoming is not None and incoming['kind'] != 'fixed':
            kind = 'hinge' if incoming['kind'] == 'revolute' else 'slide'
            joint_name = incoming['name']
            ET.SubElement(body, 'joint', name=joint_name, type=kind, axis=_xyz(incoming['axis']),
                          limited='false')
            ET.SubElement(body, 'inertial', pos='0 0 0', mass='1', diaginertia='.01 .01 .01')
        for joint in children[link_id]:
            add(body, joint['childLinkId'], joint['origin'], joint)

    add(world, robot['rootLinkId'], robot['basePose'])
    if len(names) != len(links):
        raise ValueError('disconnected or unsupported robot tree')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode')), names
