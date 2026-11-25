"""Recreate the pinned GSO subset; never overwrite files that differ from the lock."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REV = '6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c'
BASE = f'https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/{REV}/models/'
LOCK = Path(__file__).resolve().parents[2] / 'research/phase3/dataset/manifest.lock.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, destination, expected=None):
    if not destination.exists():
        with urlopen(url, timeout=60) as response:
            data = response.read()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    if expected and digest(destination) != expected:
        raise ValueError(f'checksum mismatch: {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/axisforge-phase3-data'))
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'], default=['train', 'validation', 'test'])
    args = parser.parse_args()
    import trimesh
    if trimesh.__version__ != '5.2.2':
        raise RuntimeError('Install trimesh==5.2.2 for byte reproducibility')
    lock = json.loads(LOCK.read_text())
    for obj in lock['objects']:
        if obj['split'] not in args.splits:
            continue
        name = obj['object_id']
        orig = args.root / obj['original_mesh']
        fetch(BASE + name + '/model.obj', orig, obj['original_sha256'])
        fetch(BASE + name + '/model.xml', orig.parent / 'model.xml')
        if obj['texture_availability']:
            fetch(BASE + name + '/texture.png', orig.parent / 'texture.png')
        visual = args.root / obj['visual_mesh']
        if not visual.exists():
            visual.parent.mkdir(parents=True, exist_ok=True)
            trimesh.load(orig, force='mesh', process=False).export(visual)
        if digest(visual) != obj['processed_sha256']:
            raise ValueError(f'checksum mismatch: {visual}')
        print(name, obj['split'])
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = args.root / 'manifest.json'
    if manifest.exists() and manifest.read_bytes() != LOCK.read_bytes():
        raise ValueError(f'refusing to replace {manifest}')
    manifest.write_bytes(LOCK.read_bytes())


if __name__ == '__main__':
    main()
