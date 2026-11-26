"""Per-candidate feasibility probabilities for the Phase VI development parts.

    python tools/research/phase6/priors.py --catalogues cat80/ --store store.json \
        --library research/phase5/library_parts.txt --dev research/phase6/dev_parts.json \
        --phase5-rows research/phase5/results/test_rows.json.gz --out priors.npz

Retrains the Phase V learned classifier and AFAT exactly as experiment B did
(frozen 116-part library, seed 0) and stores, for every candidate of every
development part, the raw probability and the planner prior (probability over
the part's maximum). It checks that the orderings reproduce the Phase V rows
(candidates until the first viable grasp) before writing anything. It also
writes out-of-fold probabilities for the library parts (5 folds, by part),
which the calibration analysis uses.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import baselines as bl, catalogue, engine  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('catalogues', 'store', 'library', 'dev', 'phase5-rows', 'out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    start = time.perf_counter()
    store = json.loads(Path(a.store).read_text())
    library = [s.strip() for s in Path(a.library).read_text().splitlines() if s.strip()]
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in library+dev}
    lib = engine.entries(library, store, cats)
    learned, (afat, _) = engine.train(lib, store, cats, seed=0)
    rows = json.load(gzip.open(a.phase5_rows))['rows']
    ref = {(r['part'], r['method']): r['examined1'] for r in rows
           if r['experiment'] == 'B' and r['gripper'] == 80 and r['method'] in ('default', 'learned', 'afat')}
    out, mismatches = {}, []
    for n in dev:
        cat = cats[n]
        low, high = store[n]['low'], store[n]['high']
        X = bl.features(cat, low, high)
        raw = learned.predict_proba(X)[:, 1]
        q = engine.query_entry(n, store, cats)
        Xa, _ = engine.stacked_features(q, lib, cat, low, high, bl.query_signature(cat, low, high))
        raw_afat = afat.predict_proba(Xa)[:, 1]
        for method, prior in (('default', None), ('learned', bl.scores(learned, X)), ('afat', bl.scores(afat, Xa))):
            e1 = engine.positions(cat, prior, 2.0)[0]
            if e1 != ref[(n, method)]:
                mismatches.append([n, method, e1, ref[(n, method)]])
        out[f'{n}|learned'] = raw.astype(np.float32)
        out[f'{n}|afat'] = raw_afat.astype(np.float32)
    if mismatches:
        raise SystemExit(f'orderings do not reproduce Phase V: {mismatches[:5]} ({len(mismatches)} in all)')
    # out-of-fold library probabilities for calibration (5 folds by part, seed 20261129)
    order = np.random.default_rng(20261129).permutation(len(library))
    folds = [[library[i] for i in order[f::5]] for f in range(5)]
    for f, held in enumerate(folds):
        train = [n for n in library if n not in held]
        X = np.vstack([bl.features(cats[n], store[n]['low'], store[n]['high']) for n in train])
        y = np.concatenate([cats[n]['feasible'].astype(int) for n in train])
        model = bl.fit(X, y, 0)
        for n in held:
            out[f'{n}|oof'] = model.predict_proba(bl.features(cats[n], store[n]['low'], store[n]['high']))[:, 1].astype(np.float32)
            out[f'{n}|fold'] = np.array([f])
    np.savez_compressed(a.out, **out)
    print(f'{len(dev)} development parts reproduce Phase V (default, learned, afat); '
          f'{len(library)} library parts out of fold; {time.perf_counter()-start:.0f} s')


if __name__ == '__main__':
    main()
