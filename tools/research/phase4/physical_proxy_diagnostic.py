"""Post-hoc comparison against a deliberately coarse combined convex hull."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from research_grasping.phase3.experiment import digest, load
from research_grasping.phase3.gso_contact_probe import source_parts, contact_ray_error, scene, vec
from research_grasping.phase4.physics_simulation import simulate

CASES=(('ACE_Coffee_Mug_Kristen_16_oz_cup','0'),('Android_Figure_Orange','0'),('Android_Figure_Orange','220'))


def model_summary(model):
    idx=model.body('object').id
    return {'mass_kg':float(model.body_mass[idx]),'com_local_m':model.body_ipos[idx].tolist(),
            'diagonal_inertia_kg_m2':model.body_inertia[idx].tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=load(a.input);protocol=load('research/phase4/robotics/physical_protocol.json')
    if raw['protocol_sha256']!=digest('research/phase4/robotics/physical_protocol.json'):raise ValueError('protocol mismatch')
    results=[]
    for name,candidate_id in CASES:
        row=next(r for r in raw['rows'] if (r['object_id'],r['candidate_id'])==(name,candidate_id))
        spec=next(x for x in protocol['objects'] if x['object_id']==name)
        catalog_path=Path(protocol['catalog_root'])/(name+'.json')
        if digest(catalog_path)!=spec['catalog_sha256']:raise ValueError('catalog changed')
        catalog=load(catalog_path);source_dir=Path(protocol['source_root'])/name
        if digest(catalog['project'])!=spec['project_sha256']:raise ValueError('project changed')
        assets,hulls=source_parts(Path(protocol['source_root']).parent,name)
        if {k:digest(source_dir/k) for k in assets}!=spec['source_proxy_sha256']:raise ValueError('proxy changed')
        vertices=np.array([list(map(float,line.split()[1:4])) for blob in assets.values()
                           for line in blob.splitlines() if line.startswith(b'v ')])
        hull=ConvexHull(vertices)
        obj=''.join('v %.17g %.17g %.17g\n'%tuple(v) for v in vertices)
        obj+=''.join('f %d %d %d\n'%tuple(face+1) for face in hull.simplices)
        payload=obj.encode(); coarse={'coarse.obj':payload}
        meta=catalog['metadata'];origin=np.array(meta['source_origin_m']);placement=np.array(meta['base_placement_world_m'])
        project=load(catalog['project']);pose=row['native_detail']['graspPose']
        original,_,_=scene(project,pose,assets,origin,placement)
        model,data,opening=scene(project,pose,coarse,origin,placement)
        errors=[contact_ray_error(vec(c['point'])-placement+origin,vec(c['normal']),[hull.equations])
                for c in row['native_detail']['candidate']['contacts']]
        results.append({'object_id':name,'candidate_id':candidate_id,'coarse_sha256':hashlib.sha256(payload).hexdigest(),
            'coarse_hull_volume_m3':float(hull.volume),'original_inertial':model_summary(original),
            'coarse_inertial':model_summary(model),'coarse_contact_ray_errors_m':errors,
            'original_metrics':row['f3']['nominal']['metrics'],'coarse_metrics':simulate(model,data,opening)})
    a.output.write_text(json.dumps({'pilot_protocol_sha256':raw['protocol_sha256'],'cases':results},indent=2)+'\n')


if __name__=='__main__':main()
