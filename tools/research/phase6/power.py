"""Sample size for a future confirmatory test, from the development variability.

    python tools/research/phase6/power.py --fusion fusion_benchmark.json --out statistical_power.json

Normal approximations on the part as unit, alpha 0.025 one-sided (the 95% CI
rule of the protocol), power 0.8 and 0.9. Ratios use the per-part log ratio;
proportions use paired differences (discordant pairs). The family is the one
the development decision favours; numbers are planning estimates, not results.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

ALPHA = .025


def n_ratio(sd, effect, margin=None, power=.8):
    """Parts for a paired log-ratio test: superiority (margin None) or non-inferiority/equivalence to margin."""
    zb = norm.ppf(power)
    za = norm.ppf(1-ALPHA)
    gap = abs(effect) if margin is None else np.log(margin) - abs(effect)
    return None if gap <= 0 else int(np.ceil(((za+zb)*sd/gap)**2))


def n_noninferior_prop(d, var, margin=.05, power=.8):
    zb, za = norm.ppf(power), norm.ppf(1-ALPHA)
    gap = d + margin
    return None if gap <= 0 else int(np.ceil(((za+zb)*np.sqrt(var)/gap)**2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fusion', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    f = json.loads(Path(a.fusion).read_text())
    pp = f['per_part']
    parts = sorted(pp['M0'])
    viable_rate = 151/200  # Phase V: viable parts at 80 mm among new GSO parts
    out = dict(note='Planning estimates for a confirmatory test on new objects; not results. Unit: the part. '
                    'alpha 0.025 one-sided; normal approximation.',
               viable_fraction_used=round(viable_rate, 3), families={})
    for fam in [k for k in pp if k not in ('M0',)]:
        row = {}
        for ref in ('M0', 'M1'):
            if fam == ref:
                continue
            lr = np.log([pp[fam][q]['nf']/pp[ref][q]['nf'] for q in parts])
            sd, eff = float(lr.std(ddof=1)), float(lr.mean())
            rob = np.array([pp[fam][q]['robust'] - pp[ref][q]['robust'] for q in parts], float)
            p10, p01 = float((rob > 0).mean()), float((rob < 0).mean())
            var = p10 + p01 - (p10-p01)**2
            row[f'vs_{ref}'] = dict(
                N_F=dict(log_ratio_mean=round(eff, 4), log_ratio_sd=round(sd, 4), ratio=round(float(np.exp(eff)), 4),
                         viable_parts_superiority_80=n_ratio(sd, eff), viable_parts_superiority_90=n_ratio(sd, eff, power=.9),
                         viable_parts_noninferior_1_25_80=n_ratio(sd, eff if eff > 0 else 0., 1.25),
                         viable_parts_equivalence_1_10_if_equal_80=n_ratio(sd, 0., 1.10, power=.9)),
                robust=dict(difference=round(float(rob.mean()), 4), discordant_better=round(p10, 4),
                            discordant_worse=round(p01, 4),
                            viable_parts_noninferior_5pp_80=n_noninferior_prop(float(rob.mean()), var),
                            viable_parts_noninferior_5pp_90=n_noninferior_prop(float(rob.mean()), var, power=.9),
                            viable_parts_noninferior_10pp_80=n_noninferior_prop(float(rob.mean()), var, .10)))
        for k, v in row.items():
            for block in v.values():
                for key in list(block):
                    if key.startswith('viable_parts') and block[key] is not None:
                        block[key.replace('viable_parts', 'total_parts')] = int(np.ceil(block[key]/viable_rate))
        out['families'][fam] = row
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
