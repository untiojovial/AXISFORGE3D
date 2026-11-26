"""Upper bound on what any better ordering (DKL, ADKF-IFT, few-shot adaptation) could save in time.

    python tools/research/phase7_extension/ordering_ceiling.py --benchmark research/phase7_extension/results/integrated_runtime_benchmark.json \
        --out research/phase7_extension/results/ordering_ceiling.json

An oracle order would examine one candidate: the first feasible one. Against the native Joint order it saves
(N_F(Joint) - 1) rejected examinations per object. The cost of one rejected examination is the object's own
median under M0, i.e. the rejected-candidate search time divided by the number rejected. The bound ignores the
oracle's own inference cost, so the real gain of any learned model is smaller. Development and EGAD
(already evaluated) are reported separately.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    raw = json.loads(Path(a.benchmark).read_text())['raw']
    out = {}
    for kind in ('development', 'egad'):
        rows = [x for x in raw.values() if x['kind'] == kind]
        per_rej, saves, totals, rel, excess = [], [], [], [], []
        for x in rows:
            costs = [sum(c['rejected'].values())/(c['examined']-1) for c in x['cold']['M0'] if c['examined'] > 1]
            t_rej = float(np.median(costs)) if costs else 0.
            if costs:
                per_rej.append(t_rej)
            j = x['cold']['joint']
            nf = j[0]['examined']; tot = float(np.median([c['wall'] for c in j]))
            saves.append((nf-1)*t_rej); totals.append(tot); rel.append((nf-1)*t_rej/tot); excess.append(nf-1)
        out[kind] = dict(objects=len(rows), joint_excess_candidates_mean=float(np.mean(excess)),
                         rejected_candidate_cost_ms_median=1e3*float(np.median(per_rej)),
                         oracle_saving_ms_mean=1e3*float(np.mean(saves)), oracle_saving_ms_max=1e3*float(np.max(saves)),
                         joint_total_s_median=float(np.median(totals)), oracle_saving_share_mean=float(np.mean(rel)),
                         oracle_saving_share_max=float(np.max(rel)))
    Path(a.out).write_text(json.dumps(dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), results=out), indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
