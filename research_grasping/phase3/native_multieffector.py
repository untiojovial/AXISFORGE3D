"""Pre-specified native three-finger and hand development smoke, no F3."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from research_grasping.phase3.native_catalog import load_catalog, evaluate_candidate

OBJECTS = ('Cole_Hardware_Hammer_Black', 'Craftsman_Grip_Screwdriver_Phillips_Cushion',
           'Razer_Abyssus_Ambidextrous_Gaming_Mouse')
EFFECTORS = ('three-finger', 'hand')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(binary, root, timeout):
    manifest = root / 'manifest.json'
    items = {obj['object_id']: obj for obj in json.loads(manifest.read_text())['objects']}
    output = {'manifest_sha256': sha(manifest), 'binary_sha256': sha(binary),
              'selection': '0, floor(n/2), n-1 in generator order', 'f3_executed': False, 'runs': []}
    for effector in EFFECTORS:
        for object_id in OBJECTS:
            obj = items[object_id]
            if obj['split'] != 'train':
                raise ValueError('development train split only')
            mesh = (root / obj['visual_mesh']).resolve()
            if root not in mesh.parents or sha(mesh) != obj['processed_sha256']:
                raise ValueError('invalid mesh provenance')
            row = {'object_id': object_id, 'effector': effector, 'mesh_sha256': sha(mesh),
                   'mass_kg_assumed': .1, 'scale': obj['physical_scale'], 'status': 'started', 'evaluations': []}
            output['runs'].append(row)
            with tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp) / 'project.axisforge'
                try:
                    row['stage'] = 'demo'
                    subprocess.run([binary, 'demo', '--gripper', effector, '--no-parts', '--out', project],
                                   check=True, capture_output=True, text=True, timeout=timeout)
                    row['stage'] = 'import'
                    subprocess.run([binary, 'add-object', project, mesh, '--at', '0.42', '-0.1', '0.42',
                                    '--units', 'm', '--name', object_id, '--mass', '0.1'],
                                   check=True, capture_output=True, text=True, timeout=timeout)
                    row['stage'] = 'catalog'
                    catalog = load_catalog(binary, project, object_id, timeout)
                    count = len(catalog['records'])
                    row.update(candidate_count=count, project_sha256=catalog['project_sha256'],
                               catalog_elapsed_s=catalog['elapsed_s'])
                    if not count:
                        row['status'] = 'empty_catalog'
                        continue
                    ids = list(dict.fromkeys((0, count // 2, count - 1)))
                    row['selected_ids'] = ids
                    row['stage'] = 'evaluation'
                    for index in ids:
                        try:
                            result = evaluate_candidate(binary, project, object_id, index, timeout)
                            record = result['records'][0]
                            if record['candidate'] != catalog['records'][index]['candidate']:
                                raise ValueError('candidate drift')
                            row['evaluations'].append({'id': index, 'status': 'ok',
                                                       'record': record, 'total_elapsed_s': result['elapsed_s']})
                        except (subprocess.SubprocessError, OSError, ValueError) as exc:
                            row['evaluations'].append({'id': index, 'status': 'infrastructure_error',
                                                       'error': f'{type(exc).__name__}: {exc}'})
                    row['status'] = 'complete'
                except (subprocess.SubprocessError, OSError, ValueError) as exc:
                    row.update(status='infrastructure_error', error=f'{type(exc).__name__}: {exc}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=120.)
    args = parser.parse_args()
    result = run(args.binary.resolve(), args.data_root.resolve(), args.timeout_s)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
