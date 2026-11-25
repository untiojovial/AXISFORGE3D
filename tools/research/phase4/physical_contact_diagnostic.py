"""Post-hoc reproducible contact-pair attribution for three fixed pilot rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from research_grasping.phase3.experiment import digest, load
from research_grasping.phase3.gso_contact_probe import source_parts, scene

CASES = (('ACE_Coffee_Mug_Kristen_16_oz_cup','0'), ('Android_Figure_Orange','0'),
         ('Android_Figure_Orange','220'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    protocol=load('research/phase4/robotics/physical_protocol.json')
    raw=load(a.input)
    if raw['protocol_sha256'] != digest('research/phase4/robotics/physical_protocol.json'):
        raise ValueError('pilot protocol mismatch')
    results=[]
    for name,candidate_id in CASES:
        row=next(r for r in raw['rows'] if (r['object_id'],r['candidate_id'])==(name,candidate_id))
        spec=next(o for o in protocol['objects'] if o['object_id']==name)
        if not row.get('f3') or not row['f3']['nominal']['metrics']['numerical_valid']:
            raise ValueError('nominal is unavailable or numerically invalid')
        catalog_path=Path(protocol['catalog_root'])/(name+'.json')
        if digest(catalog_path)!=spec['catalog_sha256']:raise ValueError('catalog changed')
        catalog=load(catalog_path)
        if digest(catalog['project'])!=spec['project_sha256']:raise ValueError('project changed')
        assets,_=source_parts(Path(protocol['source_root']).parent,name)
        source_dir=Path(protocol['source_root'])/name
        if {k:digest(source_dir/k) for k in assets} != spec['source_proxy_sha256']:
            raise ValueError('proxy changed')
        meta=catalog['metadata']
        model,data,opening=scene(load(catalog['project']),row['native_detail']['graspPose'],assets,
            np.asarray(meta['source_origin_m']),np.asarray(meta['base_placement_world_m']))
        object_id=model.body('object').id
        obj_geoms=set(np.flatnonzero(model.geom_bodyid==object_id).tolist())
        maxima={}
        for i in range(1700):
            t=i*.001
            data.ctrl[0]=.06*np.clip((t-.90)/.40,0,1)
            data.ctrl[1]=.06*np.clip((t-.20)/.35,0,1)
            data.ctrl[2:]=opening*(1-np.clip((t-.55)/.35,0,1))
            mujoco.mj_step(model,data)
            stage='settle' if t<.2 else 'approach' if t<.55 else 'close' if t<.9 else 'lift' if t<1.3 else 'hold'
            for index,contact in enumerate(data.contact[:data.ncon]):
                geom_a,geom_b=contact.geom
                if geom_a not in obj_geoms and geom_b not in obj_geoms:continue
                other=geom_b if geom_a in obj_geoms else geom_a
                other_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(other)) or f'geom_{other}'
                key=f'{stage}:{other_name}'
                depth=max(0.,-float(contact.dist))
                if key not in maxima or depth>maxima[key]['penetration_m']:
                    wrench=np.empty(6);mujoco.mj_contactForce(model,data,index,wrench)
                    maxima[key]={'penetration_m':depth,'time_s':float(data.time),'normal_force_n':float(wrench[0])}
        expected=row['f3']['nominal']['metrics']['max_object_penetration_all_m']
        observed=max((v['penetration_m'] for v in maxima.values()),default=0.)
        if not np.isclose(observed,expected,rtol=0,atol=1e-9):
            raise ValueError(f'cannot reproduce penetration {name}/{candidate_id}: {observed} != {expected}')
        results.append({'object_id':name,'candidate_id':candidate_id,'reproduced_max_penetration_m':observed,
                        'stage_contact_maxima':maxima})
    a.output.write_text(json.dumps({'pilot_protocol_sha256':raw['protocol_sha256'],'cases':results},indent=2)+'\n')


if __name__=='__main__':main()
