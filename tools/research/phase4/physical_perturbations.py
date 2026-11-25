"""Prespecified post-hoc COM and force perturbations for three diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from research_grasping.phase3.experiment import digest, load
from research_grasping.phase3.gso_contact_probe import source_parts, scene
from research_grasping.phase4.physics_interventions import simulate

CASES=(('ACE_Coffee_Mug_Kristen_16_oz_cup','0'),('Android_Figure_Orange','0'),('Android_Figure_Orange','220'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raw=load(args.input);protocol=load('research/phase4/robotics/physical_protocol.json')
    if raw['protocol_sha256']!=digest('research/phase4/robotics/physical_protocol.json'):raise ValueError('protocol mismatch')
    output=[]
    for name,cid in CASES:
        row=next(r for r in raw['rows'] if (r['object_id'],r['candidate_id'])==(name,cid))
        spec=next(o for o in protocol['objects'] if o['object_id']==name)
        path=Path(protocol['catalog_root'])/(name+'.json')
        if digest(path)!=spec['catalog_sha256']:raise ValueError('catalog changed')
        catalog=load(path);source=Path(protocol['source_root']);folder=source/name
        if digest(catalog['project'])!=spec['project_sha256']:raise ValueError('project changed')
        assets,_=source_parts(source.parent,name)
        if {k:digest(folder/k) for k in assets}!=spec['source_proxy_sha256']:raise ValueError('proxy changed')
        meta=catalog['metadata']; origin=np.asarray(meta['source_origin_m']);placement=np.asarray(meta['base_placement_world_m'])
        project=load(catalog['project']);pose=row['native_detail']['graspPose'];variants={}
        for mode in ('nominal','shifted_com','force'):
            model,data,opening=scene(project,pose,assets,origin,placement)
            obj=model.body('object').id
            inertia=model.body_inertia[obj].copy()
            if np.any(inertia<=0) or 2*np.max(inertia)>np.sum(inertia)+1e-12:raise ValueError('invalid inertia')
            original_com=model.body_ipos[obj].copy()
            if mode=='shifted_com':
                qpos=data.qpos.copy();qvel=data.qvel.copy()
                model.body_ipos[obj,0]+=.01
                mujoco.mj_setConst(model,data)
                data.qpos[:]=qpos;data.qvel[:]=qvel;mujoco.mj_forward(model,data)
            result=simulate(model,data,opening,perturbation_n=.2 if mode=='force' else 0.)
            variants[mode]={'metrics':result,'mass_kg':float(model.body_mass[obj]),'inertia_kg_m2':inertia.tolist(),
                'original_com_m':original_com.tolist(),'com_m':model.body_ipos[obj].tolist(),
                'force_world_n':[.2,0,0] if mode=='force' else [0,0,0],
                'force_interval_s':[1.45,1.55] if mode=='force' else None}
        output.append({'object_id':name,'candidate_id':cid,'variants':variants})
    args.output.write_text(json.dumps({'protocol_sha256':raw['protocol_sha256'],
       'code_sha256':digest('research_grasping/phase4/physics_interventions.py'),'cases':output},indent=2)+'\n')


if __name__=='__main__':main()
