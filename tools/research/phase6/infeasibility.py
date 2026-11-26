"""Early infeasibility detection on the 200 Phase V parts (80 mm), three levels.

    python tools/research/phase6/infeasibility.py --catalogues cat80/ --store store.json \
        --library research/phase5/library_parts.txt --rows research/phase5/results/test_rows.json.gz \
        --out infeasibility_benchmark.json

- demonstrable: no candidate passes the stroke/force gate, so the part has no
  viable grasp among the generated candidates without any IK or collision check;
- estimated: the classifier's highest candidate probability is below a
  threshold chosen on the library (out of fold) so that no viable library part
  would be dropped; reported with the viable development parts it would drop;
- none: everything else needs the planner.
"Infeasible" always means "no viable grasp among the generated candidates".
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('catalogues', 'store', 'library', 'rows', 'out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    store = json.loads(Path(a.store).read_text())
    library = [s.strip() for s in Path(a.library).read_text().splitlines() if s.strip()]
    rows = json.load(gzip.open(a.rows))['rows']
    parts = sorted({r['part'] for r in rows if r['experiment'] == 'B' and r['gripper'] == 80})
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in library+parts}
    feats = {n: bl.features(cats[n], store[n]['low'], store[n]['high']) for n in library+parts}
    # out-of-fold max probability on the library, to set the threshold
    order = np.random.default_rng(20261129).permutation(len(library))
    folds = [[library[i] for i in order[f::5]] for f in range(5)]
    lib_max = {}
    for held in folds:
        train = [n for n in library if n not in held]
        m = bl.fit(np.vstack([feats[n] for n in train]), np.concatenate([cats[n]['feasible'] for n in train]).astype(int), 0)
        for n in held:
            lib_max[n] = float(m.predict_proba(feats[n])[:, 1].max())
    lib_viable = {n: bool(cats[n]['feasible'].any()) for n in library}
    viable_max = [lib_max[n] for n in library if lib_viable[n]]
    threshold = float(min(viable_max))*.999 if viable_max else 0.
    model = bl.fit(np.vstack([feats[n] for n in library]), np.concatenate([cats[n]['feasible'] for n in library]).astype(int), 0)
    out_parts = {}
    for n in parts:
        c = cats[n]
        mk = int(c['makeable'].sum())
        pmax = float(model.predict_proba(feats[n])[:, 1].max())
        viable = bool(c['feasible'].any())
        level = 'demonstrable' if mk == 0 else ('estimated' if pmax < threshold else 'none')
        reasons = {}
        for r in c['reason'][~c['feasible']]:
            reasons[str(r)] = reasons.get(str(r), 0)+1
        out_parts[n] = dict(viable=viable, makeable=mk, max_p=round(pmax, 5), level=level,
                            solve_s=round(float(c.get('solve_s', np.nan)), 3) if 'solve_s' in c else None,
                            main_rejection=max(reasons, key=reasons.get) if reasons else None)
    from sklearn.metrics import roc_auc_score
    y = np.array([not v['viable'] for v in out_parts.values()], int)
    s = -np.array([v['max_p'] for v in out_parts.values()])
    summary = {}
    for lvl in ('demonstrable', 'estimated', 'none'):
        sel = [v for v in out_parts.values() if v['level'] == lvl]
        summary[lvl] = dict(parts=len(sel), nonviable=sum(not v['viable'] for v in sel), viable=sum(v['viable'] for v in sel))
    doc = dict(note='"Infeasible" = no viable grasp among the generated candidates (not proof that the part cannot be grasped). '
                    'Phase V test parts, 80 mm gripper, experiment B.',
               library_threshold_max_p=round(threshold, 5), library_viable_parts=int(sum(lib_viable.values())),
               levels=summary, auc_max_p_for_nonviable=round(float(roc_auc_score(y, s)), 4),
               parts=out_parts)
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps(summary), doc['auc_max_p_for_nonviable'])


if __name__ == '__main__':
    main()
