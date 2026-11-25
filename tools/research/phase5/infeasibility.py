"""Why parts have no feasible grasp, and what searching them costs.

    python tools/research/phase5/infeasibility.py --names test.txt --catalogues 40=cat40/ 80=cat80/ 120=cat120/ \
        --out infeasibility.json

Per gripper: parts with and without a feasible grasp, the rejection reasons
over all candidates (and over the parts without any), and the planning time
of an exhaustive search (all candidates are examined when none is feasible,
whatever the order).
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import catalogue  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--names', required=True); p.add_argument('--catalogues', nargs='+', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    names = Path(a.names).read_text().split()
    out = {}
    for spec in a.catalogues:
        g, d = spec.split('=', 1)
        parts, reasons_all, reasons_none = [], Counter(), Counter()
        for n in names:
            cat = catalogue.load(Path(d)/f'{n}.npz')
            reasons = Counter(str(r) for r in cat['reason'])
            meta = cat['meta']
            parts.append(dict(part=n, candidates=meta['candidates'], feasible=meta['feasible'], solve_s=meta['solve_s'],
                              per_candidate_ms=round(1000*meta['solve_s']/meta['candidates'], 3),
                              main_reason=reasons.most_common(2)[-1][0] if meta['feasible'] else reasons.most_common(1)[0][0]))
            reasons_all.update(reasons)
            if meta['feasible'] == 0:
                reasons_none.update(reasons)
        none = [x for x in parts if x['feasible'] == 0]
        out[g] = dict(parts=len(parts), viable=len(parts)-len(none), without=len(none),
                      reasons_all={k: v for k, v in reasons_all.most_common()},
                      reasons_without={k: v for k, v in reasons_none.most_common()},
                      main_reason_without=dict(Counter(x['main_reason'] for x in none).most_common()),
                      candidates_without_mean=float(np.mean([x['candidates'] for x in none])) if none else 0.,
                      solve_s_without_mean=float(np.mean([x['solve_s'] for x in none])) if none else 0.,
                      solve_s_mean=float(np.mean([x['solve_s'] for x in parts])),
                      per_candidate_ms_median=float(np.median([x['per_candidate_ms'] for x in parts])),
                      detail=parts)
        print(f"{g} mm: {out[g]['viable']}/{len(parts)} viable; without: {out[g]['main_reason_without']}; "
              f"exhaustive {out[g]['solve_s_mean']:.1f} s mean, {out[g]['per_candidate_ms_median']:.1f} ms/candidate")
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
