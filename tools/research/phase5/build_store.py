"""Gather what Phase V knows about each part into one store.

    python tools/research/phase5/build_store.py --library LIB.json[.gz] --projects POOL \
        [--new-parts new_parts.json --new-root DIR --new-prefix gso3] \
        --categories fuel_categories.json --descriptors a.npz ... --out store.json

Per part: source (ycb or gso), official GSO category from Gazebo Fuel (YCB
parts get 'YCB'), bounding box, project path, and descriptors: those in the
library, those in the descriptor files, and the metric D2 shape distribution
computed here from the part's mesh.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase4 import visual_prior as vp  # noqa: E402
from research_grasping.phase5 import baselines as bl  # noqa: E402

YCB = {'Banana', 'ChipsCan', 'CrackerBox', 'FoamBrick', 'GelatinBox', 'Hammer', 'MasterChefCan', 'MediumClamp',
       'MustardBottle', 'Pear', 'PottedMeatCan', 'PowerDrill', 'Scissors', 'Strawberry', 'TennisBall', 'TomatoSoupCan'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--library'); p.add_argument('--projects', required=True)
    p.add_argument('--new-parts'); p.add_argument('--new-prefix', default='gso3')
    p.add_argument('--categories', required=True)
    p.add_argument('--descriptors', action='append', default=[])
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path(a.projects)
    categories = json.loads(Path(a.categories).read_text())
    store = {}
    if a.library:
        opener = gzip.open if a.library.endswith('.gz') else open
        with opener(a.library, 'rt') as f:
            for e in json.load(f)['parts']:
                store[e['name']] = dict(project=e['project'], low=e['low'], high=e['high'],
                                        descriptors=dict(e['descriptors']))
    if a.new_parts:
        for name in json.loads(Path(a.new_parts).read_text())['kept']:
            project = f'{a.new_prefix}/{name}/cell.axisforge'
            v, _ = vp.part_mesh(root/project, name)
            store[name] = dict(project=project, low=v.min(0).tolist(), high=v.max(0).tolist(), descriptors={})
    extra = vp.read_descriptors(a.descriptors)
    for name, rec in store.items():
        rec['source'] = 'ycb' if name in YCB else 'gso'
        rec['category'] = 'YCB' if name in YCB else (categories.get(name) or ['(none)'])[0]
        for model, vector in extra.get(name, {}).items():
            rec['descriptors'][model] = [float(x) for x in vector]
        if 'd2' not in rec['descriptors']:
            v, f = vp.part_mesh(root/rec['project'], name)
            rec['descriptors']['d2'] = bl.d2_descriptor(v, f).tolist()
    Path(a.out).write_text(json.dumps(store) + '\n')
    print(len(store), 'parts ->', a.out)


if __name__ == '__main__':
    main()
