"""Exploratory five-object native checks; no final-test objects are read."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from research_grasping.phase3.native_catalog import load_catalog, evaluate_candidate

OBJECT_IDS = (
    'ACE_Coffee_Mug_Kristen_16_oz_cup',
    'Cole_Hardware_Hammer_Black',
    'Craftsman_Grip_Screwdriver_Phillips_Cushion',
    'Razer_Abyssus_Ambidextrous_Gaming_Mouse',
    'Air_Hogs_Wind_Flyers_Set_Airplane_Red',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=120.0)
    args = parser.parse_args()
    root = args.data_root.resolve()
    binary = args.binary.resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    by_id = {item['object_id']: item for item in manifest['objects']}
    rows = []
    for object_id in OBJECT_IDS:
        obj = by_id[object_id]
        if obj['split'] != 'train':
            raise ValueError(f'{object_id}: only the development train split is allowed')
        mesh = (root / obj['visual_mesh']).resolve()
        if root not in mesh.parents or sha(mesh) != obj['processed_sha256']:
            raise ValueError(f'{object_id}: mesh provenance mismatch')
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'project.axisforge'
            subprocess.run([binary, 'demo', '--no-parts', '--out', project], check=True,
                           capture_output=True, timeout=args.timeout_s)
            subprocess.run([binary, 'add-object', project, mesh, '--at', '0.42', '-0.1', '0.42',
                            '--units', 'm', '--name', object_id, '--mass', '0.1'], check=True,
                           capture_output=True, timeout=args.timeout_s)
            catalog = load_catalog(binary, project, object_id, args.timeout_s)
            count = len(catalog['records'])
            if count < 3:
                raise ValueError(f'{object_id}: fewer than three candidates')
            indices = [0, count // 2, count - 1]
            evaluations = [evaluate_candidate(binary, project, object_id, index, args.timeout_s)
                           for index in indices]
            for index, evaluation in zip(indices, evaluations):
                if evaluation['records'][0]['candidate'] != catalog['records'][index]['candidate']:
                    raise ValueError('candidate drift between catalogue and evaluation')
            rows.append({'object_id': object_id, 'mesh_sha256': sha(mesh), 'split': obj['split'],
                         'scale': obj['physical_scale'], 'mass_kg_assumed': 0.1,
                         'candidate_count': count, 'catalog_elapsed_s': catalog['elapsed_s'],
                         'project_sha256': catalog['project_sha256'], 'selected_ids': indices,
                         'evaluations': [e['records'][0] for e in evaluations]})
    output = {'manifest_sha256': sha(manifest_path), 'binary_sha256': sha(binary),
              'selection': 'generator-order indices 0, floor(n/2), n-1',
              'split': 'train', 'physical_status': 'F3 not evaluated', 'objects': rows}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
