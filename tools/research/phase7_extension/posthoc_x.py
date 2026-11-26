"""Post-hoc descriptions of the second confirmation (not part of the frozen analysis; no decision depends on them).

    python tools/research/phase7_extension/posthoc_x.py --archive research/phase7_extension/results/confirmation_x \
        --result research/phase7_extension/results/confirmation_x.json --out research/phase7_extension/results/confirmation_x_posthoc.json

Written after confirmation_x.json existed, because two results needed a closer description:

1. C4. The geometric-mean time ratio was close to 1, while X was slower on most objects and the frozen one-sided
   Wilcoxon sensitivity pointed the other way. This file gives the per-object ratio distribution, the split into
   viable objects and objects without a viable candidate, and the two-sided Wilcoxon p.
2. C2. The robustness difference rests on few discordant objects. The same fragility check as in the Phase VII
   P2 audit: move one favourable object to "neither", and one "both robust" object to "M0 only".

Everything is labelled post hoc. The confirmatory decisions are those in confirmation_x.json.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_audit import tango_interval, tango_z  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--archive', required=True); p.add_argument('--result', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    arc = Path(a.archive)
    res = json.loads(Path(a.result).read_text())
    timing = json.loads(gzip.open(arc/'timing.json.gz', 'rt').read())
    orders = json.loads(gzip.open(arc/'orders.json.gz', 'rt').read())
    med = lambda n, m: float(np.median([c['wall'] for c in timing[n]['cold'][m]]))  # noqa: E731
    names = sorted(timing)
    r = np.array([med(n, 'Joint-multi')/med(n, 'M0') for n in names])
    viable = np.array([orders[n]['viable'] for n in names])

    def summ(x):
        return dict(n=int(len(x)), gmean=float(np.exp(np.log(x).mean())), median=float(np.median(x)),
                    q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)), slower=int((x > 1).sum()), faster=int((x < 1).sum()),
                    wilcoxon_two_sided_p=float(wilcoxon(np.log(x)).pvalue))
    c4 = dict(all=summ(r), viable=summ(r[viable]), no_viable_candidate=summ(r[~viable]),
              median_seconds=dict(M0_viable=float(np.median([med(n, 'M0') for n in np.array(names)[viable]])),
                                  X_viable=float(np.median([med(n, 'Joint-multi') for n in np.array(names)[viable]])),
                                  M0_no_viable=float(np.median([med(n, 'M0') for n in np.array(names)[~viable]])),
                                  X_no_viable=float(np.median([med(n, 'Joint-multi') for n in np.array(names)[~viable]]))))
    v = res['primary']['C2']; n = v['n']; b, c = v['x_only'], v['y_only']

    def tango(b, c):
        z = tango_z(b, c, n, -.05); lo, hi = tango_interval(b, c, n)
        return dict(x_only=b, y_only=c, p_one_sided=float(norm.sf(z)), ci95=[float(lo), float(hi)], lower_above_margin=bool(lo > -.05))
    c2 = {'as observed': tango(b, c), 'one X-only object becomes neither': tango(b - 1, c),
          'one both-robust object becomes M0-only': tango(b, c + 1), 'two both-robust objects become M0-only': tango(b, c + 2)}
    doc = dict(note=__doc__.strip().split('\n\n', 1)[1], C4_time_ratio_per_object=c4, C2_fragility_tango=c2)
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps(dict(C4={k: (v['gmean'], v['median'], v['slower'], v['faster'], v['wilcoxon_two_sided_p']) for k, v in c4.items() if 'n' in v},
                          C2={k: (round(v['p_one_sided'], 5), [round(x, 4) for x in v['ci95']]) for k, v in c2.items()}), indent=1))


if __name__ == '__main__':
    main()
