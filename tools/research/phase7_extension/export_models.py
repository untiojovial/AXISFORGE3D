"""Export scikit-learn HistGradientBoostingClassifier models to the planner's learned-order format.

    python tools/research/phase7_extension/export_models.py --feasibility research/phase7/frozen/p_feasible.pkl \
        --robustness research/phase7/frozen/p_robust_given_feasible.pkl --alpha 0.8 --beta 0.3 \
        --out research/phase7_extension/native/learned_order_phase7.json

Every threshold, leaf value and the baseline are written as the 64-bit pattern of the float, so the
Rust planner (crates/grasping/src/learned.rs) loads them exactly. Only numerical splits are exported
(the models have no categorical features); the script refuses anything else. The pickles are loaded
only after their SHA-256 is checked against PHASE7_MODEL_FREEZE.json when they are the frozen ones.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import baselines as bl  # noqa: E402


def bits(x):
    return struct.unpack('<Q', struct.pack('<d', float(x)))[0]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def export(model):
    from sklearn.ensemble import HistGradientBoostingClassifier
    if not isinstance(model, HistGradientBoostingClassifier) or model.n_trees_per_iteration_ != 1:
        raise SystemExit('only binary HistGradientBoostingClassifier models are supported')
    if getattr(model, 'is_categorical_', None) is not None and model.is_categorical_.any():
        raise SystemExit('categorical features are not supported')
    trees = []
    for (pred,) in model._predictors:
        nd = pred.nodes
        if nd['is_categorical'].any():
            raise SystemExit('categorical split found')
        trees.append(dict(feature=[int(x) for x in nd['feature_idx']], threshold=[bits(x) for x in nd['num_threshold']],
                          left=[int(x) for x in nd['left']], right=[int(x) for x in nd['right']],
                          leaf=[bool(x) for x in nd['is_leaf']], value=[bits(x) for x in nd['value']],
                          missingLeft=[bool(x) for x in nd['missing_go_to_left']]))
    return dict(baseline=bits(model._baseline_prediction.ravel()[0]), trees=trees)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--feasibility', required=True); p.add_argument('--robustness')
    p.add_argument('--alpha', type=float, default=.8); p.add_argument('--beta', type=float, default=.3)
    p.add_argument('--q-floor', type=float, default=1e-3); p.add_argument('--finger-depth', type=float, default=.045)
    p.add_argument('--freeze', default=str(ROOT/'research/phase7/PHASE7_MODEL_FREEZE.json'))
    p.add_argument('--out', required=True)
    a = p.parse_args()
    frozen = json.loads(Path(a.freeze).read_text())['model_files_sha256'] if a.freeze else {}
    for path in filter(None, (a.feasibility, a.robustness)):
        h = sha256(path)
        if Path(path).name in frozen and frozen[Path(path).name] != h:
            raise SystemExit(f'{path} does not match its frozen hash')
    load = lambda path: pickle.loads(Path(path).read_bytes())  # noqa: E731 -- own files, hashes checked above
    doc = dict(format='axisforge-learned-order', version=1, features=list(bl.FEATURE_NAMES),
               feasibility=export(load(a.feasibility)), robustness=export(load(a.robustness)) if a.robustness else None,
               alpha=a.alpha, beta=a.beta, qFloor=a.q_floor, fingerDepth=a.finger_depth,
               source=dict(feasibility=dict(file=a.feasibility, sha256=sha256(a.feasibility)),
                           robustness=dict(file=a.robustness, sha256=sha256(a.robustness)) if a.robustness else None,
                           exporter='tools/research/phase7_extension/export_models.py'))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, separators=(',', ':')) + '\n')
    print(a.out, Path(a.out).stat().st_size, 'bytes;', len(doc['feasibility']['trees']), 'feasibility trees;',
          len(doc['robustness']['trees']) if doc['robustness'] else 0, 'robustness trees')


if __name__ == '__main__':
    main()
