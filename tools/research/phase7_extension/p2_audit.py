"""Statistical audit of the Phase VII primary hypothesis P2 (sensitivity analysis; the pre-specified result stands).

    python tools/research/phase7_extension/p2_audit.py --results research/phase7/results \
        --out research/phase7_extension/results/p2_statistical_audit.json

Recomputed from the committed per-object outputs only (confirmation_orders.json, confirmation_trials.json.gz):

1. the paired table of robustness (v3, six standard conditions, torsion 0.005) of Joint vs M0;
2. the pre-specified test (paired-difference z test with the sample SD, ddof 1; margin -5 pp) and its Holm step;
3. alternatives for paired binary data, as sensitivity analyses:
   - Wald test with the multinomial (MLE) variance of the paired difference;
   - Tango's score test for a paired difference (Tango 1998, Statistics in Medicine 17:891-908) at the margin,
     and the score interval obtained by inverting it;
   - Newcombe's hybrid score interval for a paired difference (Newcombe 1998, Statistics in Medicine
     17:2635-2650, method 10), without continuity correction;
4. the full-precision one-sided p-values of the whole primary family and Holm recomputed with them.

The references above are cited for the formulas; they are not yet in references_verified.json.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom, norm, t as tdist, wilcoxon

STANDARD = ('nominal', 'mass x2', 'mass x0.5', 'friction 0.3', 'pose noise 1', 'pose noise 2')


def wilson(x, n, z):
    p = x/n
    c = (p + z*z/(2*n))/(1 + z*z/n)
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    return c-h, c+h


def newcombe10(a, b, c, d, z=1.959963984540054):
    """a: both succeed, b: first only, c: second only, d: neither. Interval for p1 - p2."""
    n = a+b+c+d
    p1, p2 = (a+b)/n, (a+c)/n
    l1, u1 = wilson(a+b, n, z); l2, u2 = wilson(a+c, n, z)
    den = (a+b)*(c+d)*(a+c)*(b+d)
    phi = (a*d - b*c)/np.sqrt(den) if den > 0 else 0.
    th = p1-p2
    dl = np.sqrt((p1-l1)**2 - 2*phi*(p1-l1)*(u2-p2) + (u2-p2)**2)
    du = np.sqrt((u1-p1)**2 - 2*phi*(u1-p1)*(p2-l2) + (p2-l2)**2)
    return th-dl, th+du


def tango_z(b, c, n, theta):
    """Tango's score statistic for H0: p1 - p2 = theta (b = first only, c = second only)."""
    A = 2*n
    B = -b - c + (2*n - b + c)*theta
    C = -c*theta*(1-theta)
    q21 = (np.sqrt(B*B - 4*A*C) - B)/(2*A)
    return (b - c - n*theta)/np.sqrt(n*(2*q21 + theta*(1-theta)))


def tango_interval(b, c, n, z=1.959963984540054):
    f = lambda th, target: tango_z(b, c, n, th) - target
    lo = brentq(f, -1+1e-9, (b-c)/n, args=(z,))
    hi = brentq(f, (b-c)/n, 1-1e-9, args=(-z,))
    return lo, hi


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    R = Path(a.results)
    rows = json.loads((R/'confirmation_orders.json').read_text())['rows']
    trials = json.load(gzip.open(R/'confirmation_trials.json.gz', 'rt'))
    conf = json.loads((R/'confirmation_results.json').read_text())['sets']['primary']
    names = sorted(n for n in rows if not n.startswith('egad_eval_') and rows[n]['status'] == 'viable')

    def robust(n, m):
        t = trials[n]['trials'][rows[n][m]['first']]['0.005']
        return int(all(t[c]['held_v3'] for c in STANDARD))
    J = np.array([robust(n, 'Joint') for n in names]); M = np.array([robust(n, 'M0') for n in names])
    n = len(names)
    a11, b, c, d00 = int((J & M).sum()), int((J & ~M.astype(bool)).sum()), int((~J.astype(bool) & M).sum()), int(((1-J) & (1-M)).sum())
    diff = (J - M).astype(float)
    margin = -.05
    se1 = diff.std(ddof=1)/np.sqrt(n)
    z1 = (diff.mean() - margin)/se1
    pre = dict(method='pre-specified: z test, sample SD of the paired differences (ddof 1)', z=float(z1), p_one_sided=float(norm.sf(z1)),
               lower_97_5=float(diff.mean() - norm.ppf(.975)*se1), se=float(se1))
    var_mle = ((b+c)/n - ((b-c)/n)**2)/n
    z2 = (diff.mean() - margin)/np.sqrt(var_mle)
    wald = dict(method='Wald, multinomial (MLE) variance', z=float(z2), p_one_sided=float(norm.sf(z2)),
                lower_97_5=float(diff.mean() - norm.ppf(.975)*np.sqrt(var_mle)))
    zt = tango_z(b, c, n, margin)
    tlo, thi = tango_interval(b, c, n)
    tango = dict(method='Tango score test at the margin, and its inverted 95 % interval', z=float(zt), p_one_sided=float(norm.sf(zt)),
                 ci95=[float(tlo), float(thi)], supports_noninferiority_at_0_025=bool(norm.sf(zt) < .025 and tlo > margin))
    nlo, nhi = newcombe10(a11, b, c, d00)
    newc = dict(method='Newcombe hybrid score interval (method 10), no continuity correction', ci95=[float(nlo), float(nhi)],
                lower_bound_above_margin=bool(nlo > margin))
    # whole primary family at full precision
    pr = conf['primary']
    k1, k3 = (next(k for k in pr if k.startswith(x)) for x in ('P1', 'P3'))
    nf = lambda m: np.array([rows[x][m]['nf'] for x in names], float)
    d1 = np.log(nf('Joint')/nf('M0')); nz = d1[d1 != 0]
    p1 = float(wilcoxon(nz, alternative='less').pvalue)
    d3 = np.log(nf('Joint')/nf('M1')); se3 = d3.std(ddof=1)/np.sqrt(n)
    p3 = float(tdist.cdf((d3.mean()-np.log(1.25))/se3, n-1))
    ps = {'P1': p1, 'P2': pre['p_one_sided'], 'P3': p3}
    order = sorted(ps, key=ps.get)
    holm, run = {}, 0.
    for rank, k in enumerate(order):
        run = max(run, min(1., (len(ps)-rank)*ps[k])); holm[k] = run
    alt = {}
    for lab, pv in (('Tango', tango['p_one_sided']), ('Wald MLE', wald['p_one_sided'])):
        q = dict(ps, P2=pv); od = sorted(q, key=q.get); r_, h_ = 0., {}
        for rank, k in enumerate(od):
            r_ = max(r_, min(1., (len(q)-rank)*q[k])); h_[k] = r_
        alt[lab] = dict(holm_p_P2=h_['P2'], supported_at_0_025=bool(h_['P2'] < .025))
    mcn = dict(favourable=b, unfavourable=c, superiority_p_one_sided=float(binom.sf(b-1, b+c, .5)))
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), objects=n,
               paired_table=dict(both_robust=a11, joint_only=b, m0_only=c, neither=d00),
               difference=float(diff.mean()), margin=margin, pre_specified=pre, wald_mle=wald, tango=tango, newcombe_method10=newc,
               original_reported=dict(difference=pr[next(k for k in pr if k.startswith('P2'))]['difference'],
                                      lower_97_5=pr[next(k for k in pr if k.startswith('P2'))]['lower_97_5'],
                                      holm_p=pr[next(k for k in pr if k.startswith('P2'))]['holm_p']),
               primary_family_full_precision=dict(p_one_sided=ps, holm=holm,
                                                  stored_holm_p=dict(P1=pr[k1]['holm_p'], P3=pr[k3]['holm_p'])),
               holm_with_alternative_P2=alt, superiority_mcnemar=mcn,
               conclusion=('The pre-specified test supports non-inferiority narrowly (lower bound -4.9 pp vs margin -5). '
                           'Score-based methods for paired binary data do not: see tango and newcombe_method10. '
                           'The pre-specified result is kept as the confirmatory result; the discrepancy is reported.'))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('paired_table', 'difference', 'pre_specified', 'wald_mle', 'tango', 'newcombe_method10',
                                          'primary_family_full_precision', 'holm_with_alternative_P2', 'superiority_mcnemar')}, indent=1))


if __name__ == '__main__':
    main()
