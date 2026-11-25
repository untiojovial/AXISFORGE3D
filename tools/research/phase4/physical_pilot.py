"""Run the frozen development-only 14-object physical pilot with resumable rows."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np

from research_grasping.phase3.experiment import DevelopmentData, digest, load, save
from research_grasping.phase3.gso_contact_probe import source_parts, contact_ray_error, scene, vec
from research_grasping.phase3.native_catalog import evaluate_candidate
from research_grasping.phase4.physics_simulation import simulate


def physical(project, pose, assets, origin, placement, *, mass=.1, friction=.5,
             close=True, offset=0., dt=.001, solver='Newton'):
    import mujoco
    pose = json.loads(json.dumps(pose))
    pose['translation']['x'] += offset
    start = time.perf_counter()
    model, data, opening = scene(project, pose, assets, origin, placement, mass=mass, friction=friction)
    model.opt.timestep = dt
    model.opt.solver = {'Newton': mujoco.mjtSolver.mjSOL_NEWTON, 'CG': mujoco.mjtSolver.mjSOL_CG}[solver]
    result = simulate(model, data, opening, close_jaws=close)
    return {'parameters': {'mass_kg': mass, 'friction': friction, 'close_jaws': close,
                           'world_x_offset_m': offset, 'timestep_s': dt, 'solver': solver},
            'elapsed_s': time.perf_counter() - start, 'metrics': result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=Path('research/phase4/robotics/physical_protocol.json'))
    p.add_argument('--binary', type=Path, default=Path('target/release/axisforge'))
    p.add_argument('--data-root', type=Path, default=Path('/workspace/axisforge-phase3-data'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout-s', type=float, default=120.)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    a = p.parse_args()
    if a.shard_count < 1 or not 0 <= a.shard_index < a.shard_count:
        raise ValueError('invalid shard')
    import mujoco
    if mujoco.__version__ != '3.3.7':
        raise ValueError('pilot requires MuJoCo 3.3.7')
    protocol = load(a.protocol)
    protocol_sha = digest(a.protocol)
    binary = a.binary.resolve()
    if digest(binary) != protocol['binary_sha256']:
        raise ValueError('binary hash differs from frozen protocol')
    phase3 = Path('research/phase3/pilot')
    guard = DevelopmentData(phase3 / 'manifest.prepilot.json', load(phase3 / 'lock.json'), a.data_root)
    if guard.manifest_digest != protocol['manifest_sha256']:
        raise ValueError('manifest hash differs')
    script_sha = digest(__file__)
    sim_sha = digest('research_grasping/phase4/physics_simulation.py')
    if a.output.exists():
        output = load(a.output)
        if (output['protocol_sha256'], output['script_sha256'], output['simulation_sha256'], output['shard_count'], output['shard_index']) != (protocol_sha, script_sha, sim_sha, a.shard_count, a.shard_index):
            raise ValueError('resume provenance differs')
    else:
        output = {'protocol_sha256': protocol_sha, 'script_sha256': script_sha,
                  'simulation_sha256': sim_sha, 'mujoco_version': mujoco.__version__,
                  'shard_count': a.shard_count, 'shard_index': a.shard_index, 'rows': []}
    completed = {(row['object_id'], row['candidate_id']) for row in output['rows']}
    for object_index, obj in enumerate(protocol['objects']):
        if object_index % a.shard_count != a.shard_index:
            continue
        name = obj['object_id']
        if guard.splits[name] != 'train':
            raise ValueError('locked non-train object')
        path = Path(protocol['catalog_root']) / (name + '.json')
        if digest(path) != obj['catalog_sha256']:
            raise ValueError('catalogue changed')
        catalog = guard.catalogue(path, binary)
        source_dir = Path(protocol['source_root']) / name
        if digest(source_dir/'model.xml') != obj['source_xml_sha256'] or digest(source_dir/'model.obj') != obj['source_visual_sha256']:
            raise ValueError('source model changed')
        assets, hulls = source_parts(Path(protocol['source_root']).parent, name)
        if {k: digest(source_dir/k) for k in assets} != obj['source_proxy_sha256']:
            raise ValueError('proxy hashes differ')
        project = load(catalog['project'])
        origin = np.asarray(catalog['metadata']['source_origin_m'])
        placement = np.asarray(catalog['metadata']['base_placement_world_m'])
        for candidate_id in obj['candidate_ids']:
            if (name, candidate_id) in completed:
                continue
            row = {'object_id': name, 'family': obj['family'], 'candidate_id': candidate_id,
                   'f1_status': None, 'f2': None, 'f3_status': 'not_evaluated', 'f3': None}
            try:
                evaluation = evaluate_candidate(binary, catalog['project'], catalog['body'], int(candidate_id), a.timeout_s)
                record = evaluation['records'][0]
                row.update(f1_status='feasible' if record['plan'] is not None else 'rejected',
                           f1_rejection=record['rejection'], f2=record['quality'],
                           native_elapsed_s=evaluation['elapsed_s'], native_detail=record)
                errors = [contact_ray_error(vec(c['point']) - placement + origin, vec(c['normal']), hulls)
                          for c in record['candidate']['contacts']]
                row['proxy_contact_normal_errors_m'] = errors
                if not errors or any(v is None or v > .003 for v in errors):
                    row['f3_status'] = 'blocked_proxy_local_disagreement'
                else:
                    kwargs = (project, record['graspPose'], assets, origin, placement)
                    trials = {'nominal': physical(*kwargs),
                              'open_jaws': physical(*kwargs, close=False),
                              'zero_friction': physical(*kwargs, friction=0.),
                              'heavy_mass': physical(*kwargs, mass=2.),
                              'bad_approach': physical(*kwargs, offset=.1),
                              'mass_low': physical(*kwargs, mass=.05),
                              'mass_high': physical(*kwargs, mass=.15),
                              'friction_low': physical(*kwargs, friction=.2),
                              'friction_high': physical(*kwargs, friction=.8)}
                    trials['numerics'] = [physical(*kwargs, dt=dt, solver=solver)
                                          for solver in ('Newton', 'CG') for dt in (.0005,.001,.002)]
                    variants = [value for key, value in trials.items() if key != 'numerics'] + trials['numerics']
                    unstable = sum(not variant['metrics']['numerical_valid'] for variant in variants)
                    row.update(f3_status='completed_with_numerical_failures' if unstable else 'completed',
                               f3=trials, numerical_failures=unstable,
                               f3_interpretation='isolated gripper; not evidence of executable arm motion')
            except Exception as exc:
                row.update(f3_status='infrastructure_error', error_type=type(exc).__name__, error=str(exc))
            output['rows'].append(row)
            save(a.output, output)
            print(name, candidate_id, row['f1_status'], row['f3_status'], flush=True)


if __name__ == '__main__':
    main()
