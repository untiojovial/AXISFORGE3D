"""Is there room for few-shot adaptation (GP, DKL, ADKF-IFT, BO) during planning? (PHASE7_PROTOCOL.md, I.6)

    python tools/research/phase7/few_shot_analysis.py --fusion research/phase6/results/fusion_benchmark.json \
        --robust research/phase6/results/robustness_prediction.json \
        --cost research/phase6/results/cost_profile_online.json --out few_shot_analysis.json

A support set for a new object can only come from evaluating some of its candidates during planning,
which is exactly what the planner does anyway. If the learned prior already finds the first viable
(or viable and robust) grasp after N candidates, a perfect adaptive model could save at most N - 1
evaluations per part: that is the ceiling for any adaptation, before paying for its own support
evaluations and inference. The ceiling is computed per part from the development results.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def ceiling(nf, per_candidate_s):
    nf = np.asarray([x for x in nf if x is not None], float)
    extra = nf - 1
    return dict(parts=int(len(nf)), share_first_candidate=round(float(np.mean(nf == 1)), 4),
                share_at_most_2=round(float(np.mean(nf <= 2)), 4), share_at_most_5=round(float(np.mean(nf <= 5)), 4),
                mean_extra_candidates=round(float(extra.mean()), 3), median_extra_candidates=float(np.median(extra)),
                p90_extra_candidates=float(np.percentile(extra, 90)),
                ceiling_saving_seconds_mean=round(float(extra.mean()*per_candidate_s), 4))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('fusion', 'robust', 'cost', 'out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--mlp-eval', help='optional mlp_benchmark per-part file (not required)')
    a = p.parse_args()
    fb = json.loads(Path(a.fusion).read_text())['per_part']
    rp = json.loads(Path(a.robust).read_text())['per_part']
    cost = json.loads(Path(a.cost).read_text())
    c = cost['seconds_per_examined_candidate']
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '),
               seconds_per_examined_candidate=c, fixed_seconds_per_call=cost['fixed_s'],
               M1=dict(N_F=ceiling([r['nf'] for r in fb['M1'].values()], c),
                       N_FR=ceiling([r['nfr'] for r in fb['M1'].values()], c)),
               M9=dict(N_F=ceiling([r['nf'] for r in rp.values()], c), N_FR=ceiling([r['nfr'] for r in rp.values()], c)),
               M0=dict(N_F=ceiling([r['nf'] for r in fb['M0'].values()], c)))
    m1 = out['M1']['N_F']
    out['decision'] = dict(
        rule='run few-shot/DKL/ADKF-IFT only if the planner usually needs several evaluations after the learned prior',
        observed=f"M1 finds the first viable grasp at the first candidate in {m1['share_first_candidate']:.0%} of parts and "
                 f"within 2 in {m1['share_at_most_2']:.0%}; a perfect adaptive model could save at most "
                 f"{m1['mean_extra_candidates']:.2f} evaluations per part on average (~{m1['ceiling_saving_seconds_mean']*1e3:.1f} ms), "
                 f"against a fixed cost of {cost['fixed_s']:.2f} s per planning call",
        run_adaptation=bool(m1['mean_extra_candidates'] >= 2))
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['decision'], indent=1)); print(json.dumps(out['M1'], indent=1))


if __name__ == '__main__':
    main()
