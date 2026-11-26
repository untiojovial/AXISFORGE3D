"""Phase VI audit: the 19 classifier features and the data manifest.

    python tools/research/phase6/audit.py --catalogues cat80/ --store store.json \
        --library research/phase5/library_parts.txt --dev research/phase6/dev_parts.json \
        --features-out feature_audit.json --manifest-out phase6_data_manifest.json [--scratch-input NAME=PATH ...]

For every feature: where it comes from, what it costs, whether it exists
before planning, the leakage risk, its range and missing values, its
univariate AUC for feasibility on the library and on the development parts,
and the permutation importance of the Phase V classifier on the development
parts. The manifest lists every Phase VI input and output with its hash.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402

GEN = 'planner candidate generation (`axisforge research-candidates`), 0.4 s per part for all candidates'
BOX = 'part bounding box in the cell (store low/high), computed once from the mesh'
SOURCE = {
    **{f'approach_{c}': ('approach direction of the candidate', GEN) for c in 'xyz'},
    **{f'closing_{c}': ('|closing direction| of the candidate', GEN) for c in 'xyz'},
    'width': ('distance between the two contacts', GEN),
    **{f'rel_{c}': ('candidate centre relative to the part bounding box', f'{GEN}; {BOX}') for c in 'xyz'},
    'height': ('candidate centre height above the bounding-box bottom', f'{GEN}; {BOX}'),
    **{f'log_extent_{c}': ('log of the part bounding-box size (same for all candidates of a part)', BOX) for c in 'xyz'},
    'antipodal_a': ('|normal at contact A . closing|', GEN),
    'antipodal_b': ('|normal at contact B . closing|', GEN),
    'centre_offset': ('distance from the planner centre of mass to the closing line (lever arm); NaN set to 0', GEN),
    'preliminary': ('planner preliminary score: stroke, grip force, centring, approach (uses the cell mass 0.1 kg and mu)', GEN),
    'makeable': ('stroke and grip-force gate (uses the cell mass and the gripper force)', GEN),
}
RISK = {
    'centre_offset': 'low; assumes uniform density (planner centre of mass), not measured mass distribution',
    'preliminary': 'low; depends on the assumed nominal mass and friction of the cell',
    'makeable': 'low; depends on the assumed nominal mass; it is also the planner gate',
    'rel_z': 'low; encodes height above the table, specific to this cell (transfer to other cells untested)',
    'height': 'low; same as rel_z',
}


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('catalogues', 'store', 'library', 'dev', 'features-out', 'manifest-out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--scratch-input', action='append', default=[], help='NAME=PATH of a regenerable input')
    a = p.parse_args()
    store = json.loads(Path(a.store).read_text())
    library = [s.strip() for s in Path(a.library).read_text().splitlines() if s.strip()]
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    assert not set(library) & set(dev), 'a development part is in the library'
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in library+dev}

    def stack(names):
        X = np.vstack([bl.features(cats[n], store[n]['low'], store[n]['high']) for n in names])
        y = np.concatenate([cats[n]['feasible'].astype(int) for n in names])
        g = np.concatenate([[i]*len(cats[n]['label']) for i, n in enumerate(names)])
        nan = np.concatenate([~np.isfinite(np.asarray(cats[n]['center_offset'], float)) for n in names])
        return X, y, g, nan
    XL, yL, _, nanL = stack(library)
    XD, yD, gD, nanD = stack(dev)
    model = bl.fit(XL, yL, 0)
    from sklearn.metrics import roc_auc_score
    base = roc_auc_score(yD, model.predict_proba(XD)[:, 1])
    rng = np.random.default_rng(20261129)
    rows = []
    for j, name in enumerate(bl.FEATURE_NAMES):
        drops = []
        for _ in range(3):
            Xp = XD.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            drops.append(base - roc_auc_score(yD, model.predict_proba(Xp)[:, 1]))
        what, src = SOURCE[name]
        a_lib, a_dev = auc(XL[:, j], yL), auc(XD[:, j], yD)
        rows.append(dict(
            feature=name, meaning=what, source=src,
            cost='no extra cost: read from the generated candidate or the bounding box (microseconds per candidate)',
            available_before_planning=True,
            needs=['candidate generation', 'stroke/force gate'] if name == 'makeable' else ['candidate generation'],
            uses_ik_collision_or_verdict=False,
            leakage_risk=RISK.get(name, 'low'),
            range_library=[round(float(XL[:, j].min()), 4), round(float(XL[:, j].max()), 4)],
            auc_feasible_library=None if a_lib is None else round(max(a_lib, 1-a_lib), 4),
            direction_library=None if a_lib is None else ('higher is more often viable' if a_lib >= .5 else 'lower is more often viable'),
            auc_feasible_development=None if a_dev is None else round(max(a_dev, 1-a_dev), 4),
            permutation_importance_auc=round(float(np.mean(drops)), 4)))
    doc = dict(
        classifier='Phase V learned (HistGradientBoosting, seed 0) trained on the 116 library parts, all candidates',
        development_auc=round(float(base), 4),
        candidates=dict(library=int(len(yL)), library_viable=int(yL.sum()), development=int(len(yD)),
                        development_viable=int(yD.sum())),
        centre_offset_missing=dict(library=int(nanL.sum()), development=int(nanD.sum())),
        label='exhaustive planner verdict (reach, IK, collisions, force closure); used only as the training target on library parts',
        leakage_checks=dict(
            development_in_library=0,
            features_from_ik_collision_or_verdict=0,
            object_level='Phase V near-duplicate audit: 52 name variants and 74 look-alikes among the 200 Phase V test parts; '
                         'Phase V sensitivity kept all decisions without them (research/phase5/near_duplicates.json)',
            cell_specific='all parts share one cell (table, robot, part pose); height features carry cell-specific reach'),
        features=rows)
    Path(a.features_out).write_text(json.dumps(doc, indent=1) + '\n')

    root = Path(__file__).resolve().parents[3]
    files = sorted([*root.glob('research/phase6/**/*'), *root.glob('research_grasping/phase6/*.py'),
                    *root.glob('tools/research/phase6/*.py')])
    manifest = dict(
        note='Phase VI inputs and outputs. Files in the repository are hashed; scratch inputs are regenerated by '
             'the Phase V scripts (tools/research/phase5) from the pinned sources listed in research/phase5.',
        repository={str(f.relative_to(root)): dict(bytes=f.stat().st_size, sha256=sha256(f))
                    for f in files if f.is_file() and f.name != Path(a.manifest_out).name},
        phase5_inputs={str(f.relative_to(root)): sha256(f) for f in sorted(root.glob('research/phase5/**/*'))
                       if f.is_file() and f.suffix in ('.json', '.gz', '.txt', '.npz')},
        scratch_inputs={k: v for k, v in (s.split('=', 1) for s in a.scratch_input)},
        sealed=dict(manifest='research/phase3/dataset/manifest.lock.json', split='test', objects=8, opened=False),
        sets=dict(library=len(library), development=len(dev)))
    Path(a.manifest_out).write_text(json.dumps(manifest, indent=1) + '\n')


if __name__ == '__main__':
    main()
