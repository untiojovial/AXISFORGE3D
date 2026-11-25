"""Build labelled candidate catalogues for parts, in parallel.

    python tools/research/phase5/build_catalogues.py --binary target/release/axisforge \
        --library library.json[.gz] --projects /path/pool --out catalogues/ --jobs 4
    (or --part NAME PROJECT ...)

Writes one <name>.npz per part (see research_grasping/phase5/catalogue.py)
and skips parts already built for the same project hash.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import catalogue  # noqa: E402


def parts_from_library(path, projects):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        lib = json.load(f)['parts']
    root = Path(projects) if projects else Path(path).parent
    return [(e['name'], str(Path(e['project']) if Path(e['project']).is_absolute() else root/e['project']))
            for e in lib]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--binary', required=True); p.add_argument('--out', required=True)
    p.add_argument('--library'); p.add_argument('--projects')
    p.add_argument('--part', nargs=2, action='append', default=[], metavar=('NAME', 'PROJECT'))
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    parts = list(a.part) + (parts_from_library(a.library, a.projects) if a.library else [])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def one(part):
        name, project = part
        path = out/f'{name}.npz'
        digest = hashlib.sha256(Path(project).read_bytes()).hexdigest()
        if path.exists() and catalogue.load(path)['meta']['project_sha256'] == digest:
            return name, 'cached'
        arrays, meta = catalogue.build(a.binary, project, name)
        catalogue.save(path, arrays, meta)
        return name, f"{meta['feasible']}/{meta['candidates']} in {meta['solve_s']} s"

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for name, status in pool.map(one, parts):
            print(name, status, flush=True)


if __name__ == '__main__':
    main()
