"""Paired power simulation for the confirmatory family, from development outcomes.

    python tools/research/phase7/power.py --physics physics_dev.json.gz --fusion research/phase6/results/fusion_benchmark.json \
        --out phase7_statistical_power.json

For each hypothesis, development parts are resampled with replacement (the part is the unit) to a
given number of viable parts; the confirmatory test (exact McNemar, one-sided in the hypothesised
direction, Holm across the family) is applied; power = share of 4 000 replicates that reject.
Effects are taken as observed in development (v3 criterion, torsion 0.005) and, conservatively,
with a third of the favourable discordant pairs turned into ties. Total objects = viable / assumed
viable fraction (0.75 as in GSO, and a conservative 0.55).
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

STANDARD = ('nominal', 'mass x2', 'mass x0.5', 'friction 0.3', 'pose noise 1', 'pose noise 2')
LOADS = ('load 0.5 kg', 'load 1 kg', 'load 2 kg')


def outcome(trials, which, torsion='0.005'):
    t = trials[torsion]
    conds = STANDARD if which == 'robust' else LOADS
    return int(all(t[c]['held_v3'] for c in conds))


def mcnemar_one_sided(b, c):
    """P(X >= b) for X ~ Bin(b + c, 1/2): b = pairs where the hypothesised method wins."""
    n = b + c
    return 1.0 if n == 0 else float(binom.sf(b-1, n, .5))


def simulate(pairs, n, alpha, reps=4000, shrink=0., seed=20261203):
    rng = np.random.default_rng(seed)
    pairs = np.asarray(pairs, int)
    win = (pairs[:, 0] == 1) & (pairs[:, 1] == 0)
    if shrink:
        idx = np.flatnonzero(win)
        drop = rng.choice(idx, int(round(shrink*len(idx))), replace=False)
        pairs = pairs.copy(); pairs[drop, 0] = 0
    hits = 0
    for _ in range(reps):
        s = pairs[rng.integers(0, len(pairs), n)]
        b = int(((s[:, 0] == 1) & (s[:, 1] == 0)).sum()); c = int(((s[:, 0] == 0) & (s[:, 1] == 1)).sum())
        hits += mcnemar_one_sided(b, c) < alpha
    return hits/reps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--physics', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    ph = json.load(gzip.open(a.physics, 'rt'))
    core, trials = ph['core'], ph['trials']
    parts = sorted(core)

    def pairs(m1, m2, which):
        return [(outcome(trials[n][core[n][m1]], which), outcome(trials[n][core[n][m2]], which)) for n in parts]
    hyps = {'H1: M4 > M1, load tolerance (v3)': pairs('M4', 'M1', 'load'),
            'H2: M9 > M1, robustness (v3)': pairs('M9', 'M1', 'robust')}
    if 'Joint' in core[parts[0]]:
        hyps.update({'Joint > M1, load tolerance (v3)': pairs('Joint', 'M1', 'load'),
                     'Joint > M1, robustness (v3)': pairs('Joint', 'M1', 'robust')})
    noninf = {}
    if 'Joint' in core[parts[0]]:
        noninf = {'Joint non-inferior to M0, robustness (v3), margin 5 pp': (pairs('Joint', 'M0', 'robust'), .05),
                  'Joint non-inferior to M0, load tolerance (v3), margin 10 pp': (pairs('Joint', 'M0', 'load'), .10)}
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), development_parts=len(parts), hypotheses={})
    sizes = (100, 150, 200, 250, 300, 400)
    for h, pr in hyps.items():
        pr = np.asarray(pr)
        b = int(((pr[:, 0] == 1) & (pr[:, 1] == 0)).sum()); c = int(((pr[:, 0] == 0) & (pr[:, 1] == 1)).sum())
        row = dict(development_discordant=dict(favourable=b, unfavourable=c, n=len(pr),
                                                rate_a=round(float(pr[:, 0].mean()), 4), rate_b=round(float(pr[:, 1].mean()), 4)),
                   power={})
        for label, shrink in (('observed effect', 0.), ('conservative (a third of favourable discordances removed)', 1/3)):
            row['power'][label] = {}
            for alpha_label, alpha in (('alpha 0.025 (first Holm step at family 2: 0.0125)', .0125), ('alpha 0.025', .025)):
                row['power'][label][alpha_label] = {str(n): round(simulate(pr, n, alpha, shrink=shrink), 3) for n in sizes}
        out['hypotheses'][h] = row
    out['non_inferiority'] = {}
    for h, (pr, margin) in noninf.items():
        pr = np.asarray(pr, float)
        d = pr[:, 0] - pr[:, 1]
        row = dict(development_difference=round(float(d.mean()), 4), margin=margin, power={})
        rng = np.random.default_rng(20261204)
        for alpha_label, alpha in (('one-sided 0.025/3 (Holm, family of 3)', .025/3), ('one-sided 0.025', .025)):
            from scipy.stats import norm
            z = norm.ppf(1-alpha)
            row['power'][alpha_label] = {}
            for n in sizes:
                hits = 0
                for _ in range(4000):
                    s = d[rng.integers(0, len(d), n)]
                    se = s.std(ddof=1)/np.sqrt(n) if s.std() > 0 else 1e-9
                    hits += (s.mean() - z*se) > -margin
                row['power'][alpha_label][str(n)] = round(hits/4000, 3)
        out['non_inferiority'][h] = row
    out['viable_fraction_assumed'] = {'GSO (Phase V)': .755, 'conservative': .55}
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1)[:6000])


if __name__ == '__main__':
    main()
