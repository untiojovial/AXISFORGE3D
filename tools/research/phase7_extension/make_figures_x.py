"""Write the data of the article's effects figure (Fig. 2) from the stored confirmation results (nothing typed by hand).

    python tools/research/phase7_extension/make_figures_x.py --out publication/intelligent_service_robotics/figures

Writes fig_effects_{ratio,diff}_{1,2}.dat, read by figures/fig_effects.tex (pgfplots). Each row is one
confirmatory effect with its 95 % interval, exactly as in the article's tables:

- first confirmation (research/phase7/results/confirmation_results.json): P1 and P3 (N_F ratios), S6/H4 (full online
  cost of the two-process implementation), P2 (robustness difference) and S5 (load-tolerance difference), with the
  paired bootstrap intervals of that analysis;
- second confirmation (research/phase7_extension/results/confirmation_x.json): C1 and C3 (N_F ratios), C4 (cold-process
  time of the integrated implementation), C2 (robustness), S1 (load tolerance) and S2 (robust at all three torsions,
  against M1), with the t intervals (ratios) and Tango score intervals (differences) of that analysis.

Columns: y (row position in the plot), est, em (est - lower), ep (upper - est), set (1 = first, 2 = second
confirmation). Differences are in percentage points.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OFFSET = 0.17  # vertical offset of the two confirmations within a row


def row(y, est, lo, hi, s):
    return f'{y:.2f} {est:.5f} {est - lo:.5f} {hi - est:.5f} {s}'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--first', default=str(ROOT/'research/phase7/results/confirmation_results.json'))
    p.add_argument('--second', default=str(ROOT/'research/phase7_extension/results/confirmation_x.json'))
    a = p.parse_args()
    c1 = json.loads(Path(a.first).read_text())['sets']['primary']
    pr, se = c1['primary'], c1['secondary']
    first = lambda d, pre: next(v for k, v in d.items() if k.startswith(pre))
    c2 = json.loads(Path(a.second).read_text())
    x, xs = c2['primary'], c2['secondary']
    sec = lambda pre: next(v for k, v in xs.items() if k.startswith(pre))

    ratio = ['y est em ep set']
    # row 3: N_F vs the original planner; row 2: N_F vs M1; row 1: planning time vs the original planner
    for y, v1, v2 in ((3, first(pr, 'P1'), x['C1']), (2, first(pr, 'P3'), x['C3']), (1, first(se, 'S6'), x['C4'])):
        ratio.append(row(y + OFFSET, v1['ratio'], *v1['ci95'], 1))
        ratio.append(row(y - OFFSET, v2['gmean_ratio'], *v2['ci95'], 2))
    diff = ['y est em ep set']
    # row 3: robustness vs the original planner; row 2: load tolerance vs the original planner;
    # row 1: robust at all three torsions vs M1 (second confirmation only)
    for y, v1, v2 in ((3, first(pr, 'P2'), x['C2']), (2, first(se, 'S5'), sec('S1'))):
        diff.append(row(y + OFFSET, 100*v1['difference'], *(100*t for t in v1['ci95']), 1))
        diff.append(row(y - OFFSET, 100*v2['difference'], *(100*t for t in v2['tango_ci95']), 2))
    v = sec('S2')
    diff.append(row(1 - OFFSET, 100*v['difference'], *(100*t for t in v['tango_ci95']), 2))
    out = Path(a.out)
    for name, rows in (('ratio', ratio), ('diff', diff)):
        for s in (1, 2):  # one file per confirmation, so the figure needs no row filter
            keep = [rows[0]] + [r for r in rows[1:] if r.endswith(f' {s}')]
            (out/f'fig_effects_{name}_{s}.dat').write_text('\n'.join(keep) + '\n')
    print('\n'.join(ratio + diff))


if __name__ == '__main__':
    main()
