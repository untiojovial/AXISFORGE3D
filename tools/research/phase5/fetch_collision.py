"""Download the published convex collision pieces of GSO parts (pinned revision).

    python tools/research/phase5/fetch_collision.py --clone GSO_BLOBLESS_CLONE --names test.txt --out DIR

The file list comes from the blob-less clone's tree at the pinned revision;
files go to DIR/<name>/model_collision_<i>.obj (DIR should be new and empty)
and are only read as mesh data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

REV = '6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c'
BASE = f'https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/{REV}/'


def fetch(url, attempts=4):
    for attempt in range(attempts):
        try:
            with urlopen(url, timeout=120) as r:
                return r.read()
        except OSError:
            if attempt == attempts-1:
                raise
            time.sleep(2**attempt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--clone', required=True); p.add_argument('--names', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    files = []
    for name in Path(a.names).read_text().split():
        listing = subprocess.run(['git', '-C', a.clone, 'ls-tree', '--name-only', REV, f'models/{name}/'],
                                 check=True, capture_output=True, text=True).stdout.split()
        files += [f for f in listing if Path(f).name.startswith('model_collision_') and f.endswith('.obj')]
    out = Path(a.out)

    def one(f):
        dst = out/Path(f).parent.name/Path(f).name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(fetch(BASE+f))
        return f

    with ThreadPoolExecutor(max_workers=8) as pool:
        done = list(pool.map(one, files))
    print(len(done), 'collision pieces ->', out)


if __name__ == '__main__':
    main()
