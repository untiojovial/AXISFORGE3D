"""Conditional development probe of native parallel-jaw poses with GSO collision parts.

A free object, two actuated jaws and the project palm interact under gravity.
The robot arm is replaced by two prescribed stages. Source parts and all assumptions
are recorded separately from the native planning utility.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .dataset_reproduce import REV
from .experiment import DevelopmentData, digest, load, save
from .native_catalog import evaluate_candidate
from .physics_frames import transform_from_axisforge


def vec(d):
    return np.array([d[k] for k in 'xyz'])


def nums(x):
    return ' '.join(f'{v:.12g}' for v in x)


def frame_attrs(transform):
    return dict(pos=nums(transform[:3, 3]), quat=nums(np.roll(Rotation.from_matrix(transform[:3,:3]).as_quat(), 1)))


def source_parts(source, object_id):
    folder = source/'models'/object_id
    xml = ET.parse(folder/'model.xml')
    names = {e.get('name'):e for e in xml.findall('./asset/mesh')}
    assets = {}
    hulls = []
    for geom in xml.findall('./worldbody/body/geom'):
        name = geom.attrib['mesh']
        if geom.get('contype') == '0':
            continue
        mesh = names[name]
        if set(mesh.attrib) - {'name','file'} or set(geom.attrib) - {'mesh','type','group'}:
            raise ValueError('unsupported source collision transform')
        path = folder/mesh.attrib['file']
        if path.parent != folder or path.suffix != '.obj':
            raise ValueError('unexpected collision part')
        data = path.read_bytes()
        vertices = np.array([list(map(float,l.split()[1:4])) for l in data.splitlines() if l.startswith(b'v ')])
        hulls.append(ConvexHull(vertices).equations)
        assets[path.name] = data
    if not assets:
        raise ValueError('source has no contact geometry')
    return assets, hulls


def contact_ray_error(point, normal, hulls):
    """Distance along a unit normal to the nearest boundary of the union, metres."""
    intervals = []
    for planes in hulls:
        b = planes[:, :3] @ point + planes[:, 3]
        d = planes[:, :3] @ normal
        parallel = np.abs(d) <= 1e-10
        if np.any(b[parallel] > 1e-8):
            continue
        upper = np.min(-b[d > 1e-10]/d[d > 1e-10], initial=np.inf)
        lower = np.max(-b[d < -1e-10]/d[d < -1e-10], initial=-np.inf)
        if lower <= upper:
            intervals.append((float(lower),float(upper)))
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + 1e-8:
            merged[-1][1] = max(merged[-1][1],hi)
        else:
            merged.append([lo,hi])
    return min((abs(v) for segment in merged for v in segment), default=None)


def scene(project, pose, assets, source_origin, placement, *, mass=.1, friction=.5):
    import mujoco
    robot = project['project']['robot']
    g = robot['gripper']
    joints = {j['id']:j for j in robot['joints']}
    links = {l['id']:l for l in robot['links']}
    frame = next(f for f in robot['toolFrames'] if f['id']==g['graspFrameId'])
    if frame['linkId'] != g['palmLinkId'] or len(g['jawJointIds']) != 2:
        raise ValueError('probe requires a parallel gripper on the palm')
    grasp = transform_from_axisforge(pose)
    # Lower the whole AxisForge cell to put its table plane at z=0.
    grasp[2,3] -= placement[2]
    approach = grasp[:3,:3] @ vec(g['approachAxis'])
    palm_pose = grasp @ np.linalg.inv(transform_from_axisforge(frame['offset']))
    palm_pose[:3,3] -= .06*approach
    root = ET.Element('mujoco')
    ET.SubElement(root,'compiler',inertiafromgeom='auto')
    ET.SubElement(root,'option',timestep='.001',gravity='0 0 -9.81',solver='Newton',iterations='80')
    default=ET.SubElement(root,'default')
    ET.SubElement(default,'geom',friction=f'{friction} .005 .0001',condim='3')
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'geom',name='support',type='plane',size='1 1 .02')
    asset=ET.SubElement(root,'asset')
    obj_pos = placement-source_origin
    obj_pos[2] -= placement[2]
    obj = ET.SubElement(world,'body',name='object',pos=nums(obj_pos))
    ET.SubElement(obj,'freejoint')
    # Same density across component hulls; mass/inertia derive from the union
    # approximation. Overlapping volumes contribute to each part.
    for i, filename in enumerate(assets):
        name=f'part{i}'
        ET.SubElement(asset,'mesh',name=name,file=filename)
        ET.SubElement(obj,'geom',type='mesh',mesh=name,density='1000',group='3')
    lift=ET.SubElement(world,'body',name='lift')
    ET.SubElement(lift,'joint',name='lift_z',type='slide',axis='0 0 1',damping='5')
    ET.SubElement(lift,'inertial',pos='0 0 0',mass='.1',diaginertia='.001 .001 .001')
    stage=ET.SubElement(lift,'body',name='stage')
    ET.SubElement(stage,'joint',name='approach',type='slide',axis=nums(approach),damping='5')
    ET.SubElement(stage,'inertial',pos='0 0 0',mass='.1',diaginertia='.001 .001 .001')
    palm=ET.SubElement(stage,'body',name='palm',**frame_attrs(palm_pose))
    actuators=ET.SubElement(root,'actuator')
    for name,kp,force in [('lift_z',6500,180),('approach',6500,180)]:
        ET.SubElement(actuators,'position',joint=name,kp=str(kp),forcerange=f'-{force} {force}')
    def add_link(body, link_id, mass):
        geoms = [e for e in links[link_id]['geometries'] if e['role']=='collision']
        if len(geoms)!=1 or geoms[0]['shape']['kind']!='box':
            raise ValueError('requires one box collision geom per link')
        geom=geoms[0]
        ET.SubElement(body,'geom',name=body.get('name')+'_geom',type='box',size=nums(vec(geom['shape']['size'])/2),
                      mass=str(mass),**frame_attrs(transform_from_axisforge(geom['origin'])))
    add_link(palm,g['palmLinkId'],links[g['palmLinkId']]['inertial']['mass'])
    for i,joint_id in enumerate(g['jawJointIds']):
        joint=joints[joint_id]
        child = ET.SubElement(palm,'body',name=f'jaw{i}',**frame_attrs(transform_from_axisforge(joint['origin'])))
        if joint['kind']!='prismatic' or joint['parentLinkId']!=g['palmLinkId']:
            raise ValueError('unsupported jaw joint')
        ET.SubElement(child,'joint',name=f'jaw{i}',type='slide',axis=nums(vec(joint['axis'])),
                      range=nums([joint['limits']['lower'],joint['limits']['upper']]),damping='1')
        add_link(child,joint['childLinkId'],links[joint['childLinkId']]['inertial']['mass'])
        force=g['maxGripForce']/2
        ET.SubElement(actuators,'position',joint=f'jaw{i}',kp='2500',forcerange=f'-{force} {force}')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets)
    object_id=model.body('object').id
    ratio=mass/model.body_mass[object_id]
    model.body_mass[object_id]*=ratio
    model.body_inertia[object_id]*=ratio
    data=mujoco.MjData(model)
    mujoco.mj_setConst(model,data)
    for i in range(2):
        data.qpos[model.jnt_qposadr[model.joint(f'jaw{i}').id]]=g['maxOpening']/2
    mujoco.mj_forward(model,data)
    return model,data,g['maxOpening']/2


def simulate(model,data,opening,*,close_jaws=True):
    import mujoco
    object_id=model.body('object').id
    jaw_ids={model.geom(f'jaw{i}_geom').id for i in range(2)}
    obj_geoms=set(np.flatnonzero(model.geom_bodyid==object_id).tolist())
    support=model.geom('support').id
    palm=model.body('palm').id
    bilateral=table_steps=force_bilateral=0
    initial_com=data.xipos[object_id].copy()
    initial_rotation=data.xmat[object_id].reshape(3,3).copy()
    maximum_penetration=0.
    hold_normal_force=np.zeros(2)
    jaw_by_id={model.geom(f'jaw{i}_geom').id:i for i in range(2)}
    dt=float(model.opt.timestep)
    for i in range(1700):
        # settle 0.2 s, approach 0.35, close 0.35, lift 0.4 and hold 0.4.
        t=i*dt
        data.ctrl[0]=.06*np.clip((t-.90)/.40,0,1)
        data.ctrl[1]=.06*np.clip((t-.20)/.35,0,1)
        data.ctrl[2:]=opening*(1-np.clip((t-.55)/.35,0,1)) if close_jaws else opening
        mujoco.mj_step(model,data)
        if i==199:
            baseline=float(data.xipos[object_id,2])
            settle_translation=float(np.linalg.norm(data.xipos[object_id]-initial_com))
            settle_rotation=float(Rotation.from_matrix(
                initial_rotation.T@data.xmat[object_id].reshape(3,3)).magnitude())
        if i==899:
            initial_relative=data.xipos[object_id].copy()-data.xpos[palm]
        if i>=1300:
            contacts=set()
            forces=np.zeros(2)
            for index,contact in enumerate(data.contact[:data.ncon]):
                a,b=contact.geom
                if a in obj_geoms:contacts.add(b)
                if b in obj_geoms:contacts.add(a)
                finger=b if a in obj_geoms else a if b in obj_geoms else None
                if finger in jaw_by_id:
                    wrench=np.empty(6)
                    mujoco.mj_contactForce(model,data,index,wrench)
                    forces[jaw_by_id[finger]]+=max(0.,wrench[0])
                    maximum_penetration=max(maximum_penetration,-float(contact.dist))
            hold_normal_force+=forces
            force_bilateral+=bool(np.all(forces>.01))
            bilateral+=jaw_ids<=contacts
            table_steps+=support in contacts
    lift=float(data.xipos[object_id,2]-baseline)
    slip=float(np.linalg.norm((data.xipos[object_id]-data.xpos[palm])-initial_relative))
    if not np.isfinite(data.qpos).all() or not np.isfinite([slip,lift]).all():
        raise ValueError('nonfinite simulation')
    return dict(lift_m=lift,relative_displacement_m=slip,bilateral_hold_fraction=bilateral/400,
                settle_displacement_m=settle_translation,settle_rotation_rad=settle_rotation,
                mean_hold_normal_force_n=(hold_normal_force/400).tolist(),
                max_hold_penetration_m=maximum_penetration,loaded_bilateral_fraction=force_bilateral/400,
                table_hold_fraction=table_steps/400,held=bool(lift>=.04 and slip<=.02 and bilateral>=360 and table_steps==0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ['catalogues','run','source','binary','data','protocol','out']:
        p.add_argument('--'+field,type=Path,required=True)
    a=p.parse_args()
    if subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()!=REV:
        raise ValueError('unrecognized source revision')
    if subprocess.run(['git','-C',str(a.source),'diff','--quiet','HEAD','--','models']).returncode!=0:
        raise ValueError('provider mesh checkout must match its pinned revision')
    guard=DevelopmentData(a.protocol/'manifest.prepilot.json',load(a.protocol/'lock.json'),a.data)
    import mujoco
    report=dict(mujoco=mujoco.__version__,source_revision=REV,manifest_sha256=guard.manifest_digest,
                mass_kg=.1,friction=.5,scope='development validation; conditional on F1 feasibility',objects=[])
    for folder in sorted((a.run/'runs').iterdir()):
        catalog=guard.catalogue(a.catalogues/(folder.name+'.json'),a.binary)
        ids=sorted({int(row['candidate_id']) for trace in folder.glob('*.json') for row in load(trace)['rows']
                    if row['status']=='ok'})[:4]
        assets,hulls=source_parts(a.source,folder.name)
        objdir=a.source/'models'/folder.name
        if digest(objdir/'model.obj')!=guard.objects[folder.name]['original_sha256']:
            raise ValueError('source visual differs from manifest')
        meta=catalog['metadata']; origin=np.array(meta['source_origin_m']); placement=np.array(meta['base_placement_world_m'])
        entry=dict(object_id=folder.name,measurements=[],proxy_files={k:digest(objdir/k) for k in assets},
                   source_xml_sha256=digest(objdir/'model.xml'),project_sha256=meta['project_sha256'])
        for i in ids:
            record=evaluate_candidate(a.binary,catalog['project'],catalog['body'],i)['records'][0]
            if record['plan'] is None:raise ValueError('native feasibility changed')
            contacts=record['candidate']['contacts']
            errors=[contact_ray_error(vec(c['point'])-placement+origin,vec(c['normal']),hulls) for c in contacts]
            row=dict(candidate_id=i,f2_epsilon=record['quality']['quality'],contacts=len(contacts),
                     visual_to_proxy_along_normal_m=errors,source_proxy_local_check='within_3mm' if
                     errors and all(e is not None and e<=.003 for e in errors) else 'disagreement')
            start=time.monotonic()
            model,data,opening=scene(load(catalog['project']),record['graspPose'],assets,origin,placement)
            row['probe']=simulate(model,data,opening)
            alt,alt_data,alt_opening=scene(load(catalog['project']),record['graspPose'],assets,origin,placement)
            row['open_jaws_control']=simulate(alt,alt_data,alt_opening,close_jaws=False)
            row['sensitivity']=[]
            for mass, friction in ((.05,.5),(.15,.5),(.1,.2),(.1,.8)):
                alt, alt_data, alt_opening = scene(load(catalog['project']), record['graspPose'],
                                          assets, origin, placement, mass=mass, friction=friction)
                row['sensitivity'].append(dict(mass_kg=mass, friction=friction,
                                               probe=simulate(alt,alt_data,alt_opening)))
            row['elapsed_s']=time.monotonic()-start
            entry['measurements'].append(row)
        report['objects'].append(entry)
        save(a.out,report)


if __name__=='__main__':
    main()
