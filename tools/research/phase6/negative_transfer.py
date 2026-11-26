"""Negative transfer of the Phase VI orderings against the planner's default.

    python tools/research/phase6/negative_transfer.py --fusion fusion_benchmark.json \
        --conditional robustness_prediction.json --out negative_transfer_analysis.json

Per part: an ordering transfers negatively in speed when it examines more
candidates than the default before its first viable grasp, and in robustness
when the default's first viable grasp is robust (or load tolerant) and the
ordering's is not. Rates over the 151 development parts.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fusion', required=True); p.add_argument('--conditional', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    pp = json.loads(Path(a.fusion).read_text())['per_part']
    pp = {**pp, 'M9 P(F)P(H|F)': json.loads(Path(a.conditional).read_text())['per_part']}
    base = pp['M0']
    parts = sorted(base)
    out = dict(note='Development parts; rates over parts. M3-M8 and M9 are outer-fold results.', methods={})
    for m, rows in pp.items():
        if m == 'M0':
            continue
        slower = sum(rows[q]['nf'] > base[q]['nf'] for q in parts)
        faster = sum(rows[q]['nf'] < base[q]['nf'] for q in parts)
        lost_r = sum(base[q]['robust'] and not rows[q]['robust'] for q in parts)
        won_r = sum(rows[q]['robust'] and not base[q]['robust'] for q in parts)
        lost_l = sum(base[q]['load'] and not rows[q]['load'] for q in parts)
        won_l = sum(rows[q]['load'] and not base[q]['load'] for q in parts)
        out['methods'][m] = dict(slower_than_default=round(slower/len(parts), 4), faster_than_default=round(faster/len(parts), 4),
                                 robustness_lost=round(lost_r/len(parts), 4), robustness_gained=round(won_r/len(parts), 4),
                                 load_tolerance_lost=round(lost_l/len(parts), 4), load_tolerance_gained=round(won_l/len(parts), 4))
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    for m, v in out['methods'].items():
        print(m, v)


if __name__ == '__main__':
    main()
