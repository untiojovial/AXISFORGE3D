"""Power of the new confirmation (EGAD positions 300-499) by resampling observed per-object outcomes.

    python tools/research/phase7_extension/power_new.py --spec power_spec.json --out research/phase7_extension/results/power_new.json

The spec gives, for each primary hypothesis, the source of per-object paired outcomes that stands in for
the new population, and the test of the protocol. It also gives the planning scenarios:

- observed: the effect as observed in the source;
- conservative: the effect shrunk toward the null by the stated factor;
- null: the effect at the margin, which gives the type I error at the boundary.

Every scenario resamples objects with replacement (N_total objects; the viable share is drawn from a binomial
with the observed rate) and applies the protocol's test and the Holm step over the primary family.
Monte Carlo SE is reported. Sources are exploratory (development, or EGAD already evaluated) and are used
for planning only.

Tests implemented: 'wilcoxon_less' (paired log ratio < 0), 'tango_ni' (paired binary difference >= margin,
Tango score test), 'mcnemar_sup' (exact one-sided McNemar), 'ratio_ni_t' (t test of mean log ratio <
log margin).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binom, norm, t as tdist, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_audit import tango_z  # noqa: E402


def test(kind, x, margin=None):
    x = np.asarray(x, float)
    if kind == 'wilcoxon_less':
        d = x[x != 0]
        return float(wilcoxon(d, alternative='less').pvalue) if len(d) else 1.
    if kind == 'ratio_ni_t':
        se = x.std(ddof=1)/np.sqrt(len(x))
        return float(tdist.cdf((x.mean() - np.log(margin))/se, len(x)-1)) if se > 0 else (0. if x.mean() < np.log(margin) else 1.)
    if kind == 'tango_ni':  # x: paired differences in {-1, 0, 1}
        b, c, n = int((x > 0).sum()), int((x < 0).sum()), len(x)
        return float(norm.sf(tango_z(b, c, n, margin)))
    if kind == 'mcnemar_sup':
        b, c = int((x > 0).sum()), int((x < 0).sum())
        return float(binom.sf(b-1, b+c, .5)) if b + c else 1.
    raise ValueError(kind)


def holm(ps):
    order = np.argsort(ps); out = np.empty(len(ps)); run = 0.
    for r, i in enumerate(order):
        run = max(run, min(1., (len(ps)-r)*ps[i])); out[i] = run
    return out


def shrink(x, kind, factor, margin):
    """Shift the per-object effects so that their mean moves toward the null (factor 1 = observed, 0 = at the null
    or the margin), keeping the observed spread. Scaling would leave a t statistic unchanged and is not used."""
    x = np.asarray(x, float)
    if kind == 'wilcoxon_less':
        return x - (1 - factor)*x.mean()
    if kind == 'ratio_ni_t':
        return x - (1 - factor)*(x.mean() - np.log(margin))
    return x  # binary outcomes are shifted by flipping discordant pairs instead (see main)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--spec', required=True); p.add_argument('--out', required=True)
    p.add_argument('--reps', type=int, default=4000); p.add_argument('--seed', type=int, default=20261213)
    a = p.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    rng = np.random.default_rng(a.seed)
    N, viable_rate, alpha = spec['objects'], spec['viable_rate'], spec['alpha_one_sided']
    hyps = spec['primary']
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), spec=spec, scenarios={})
    for scen, factor in spec['scenarios'].items():
        rej = np.zeros((a.reps, len(hyps)), bool)
        for r in range(a.reps):
            nv = rng.binomial(N, viable_rate)
            ps = []
            for h in hyps:
                x = np.asarray(h['per_object'], float)
                n = nv if h.get('unit', 'viable') == 'viable' else N
                if h['test'] in ('tango_ni', 'mcnemar_sup') and factor < 1:
                    # binary: with probability (1 - factor) replace a favourable or unfavourable discordant pair
                    # by its mirror at the rate that puts the mean at the null/margin
                    target = h.get('margin', 0.) if h['test'] == 'tango_ni' else 0.
                    s = rng.choice(x, n)
                    gap = s.mean() - target
                    if gap > 0:
                        pos = np.flatnonzero(s > 0)
                        k = int(round((1-factor)*gap*n/2)) if len(pos) else 0
                        if k:
                            s[rng.choice(pos, min(k, len(pos)), replace=False)] = -1
                    ps.append(test(h['test'], s, h.get('margin')))
                else:
                    s = shrink(x, h['test'], factor, h.get('margin'))
                    s = rng.choice(s, n)
                    ps.append(test(h['test'], s, h.get('margin')))
            rej[r] = holm(np.array(ps)) < alpha
        power = rej.mean(0)
        out['scenarios'][scen] = dict(factor=factor, power={h['name']: dict(power=float(pw), mc_se=float(np.sqrt(pw*(1-pw)/a.reps)))
                                                            for h, pw in zip(hyps, power)},
                                      all_primary=float(rej.all(1).mean()))
        print(scen, {h['name']: round(float(pw), 3) for h, pw in zip(hyps, power)}, 'all', round(float(rej.all(1).mean()), 3))
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
