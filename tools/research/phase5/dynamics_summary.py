"""Summarise the MuJoCo checks per method.

    python tools/research/phase5/dynamics_summary.py --dynamics dynamics.json --out dynamics_summary.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import stats  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dynamics', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    parts = json.loads(Path(a.dynamics).read_text())['parts']
    methods = sorted({m for v in parts.values() for m in v.get('methods', {})})
    out, rows = {}, []
    for m in methods:
        rec = [v['methods'][m] for v in parts.values() if m in v.get('methods', {})]
        rec = [r for r in rec if r.get('first')]
        held = [r['first_held'] for r in rec]
        robust = [r['first_robust'] for r in rec]
        counts = [r['first_held_count'] for r in rec]
        dyn = [r['first_dynamic'] for r in rec]
        per = {}
        for name, v in parts.items():
            r = v.get('methods', {}).get(m)
            if r and r.get('first'):
                d = r['first_dynamic']
                rows.append(dict(part=name, method=m, feasible=1, found1=d is not None,
                                 examined1=d['examined'] if d else 10**6))
                per[name] = r['first_trials']
        by_perturbation = {}
        for trials in per.values():
            for k, t in trials.items():
                by_perturbation.setdefault(k, []).append(t['held'])
        out[m] = dict(parts=len(rec), first_held=float(np.mean(held)), first_robust=float(np.mean(robust)),
                      first_held_of_6=float(np.mean(counts)),
                      held_by_perturbation={k: float(np.mean(v)) for k, v in by_perturbation.items()},
                      dynamic_found=float(np.mean([d is not None for d in dyn])),
                      dynamic_examined_median=float(np.median([d['examined'] for d in dyn if d])) if any(dyn) else None)
    # Paired: candidates until the first grasp that holds, parts where both methods found one within the chase.
    contrasts = []
    for m in methods:
        if m == 'default':
            continue
        sub = [r for r in rows if r['method'] in (m, 'default') and r['found1']]
        both = {r['part'] for r in sub if r['method'] == m} & {r['part'] for r in sub if r['method'] == 'default'}
        sub = [r for r in sub if r['part'] in both]
        c = stats.contrast(sub, m, 'default')
        if c:
            contrasts.append(c)
    stats.holm(contrasts)
    Path(a.out).write_text(json.dumps(dict(methods=out, to_first_held_vs_default=contrasts), indent=1) + '\n')
    for m, v in out.items():
        print(f"{m:15s} parts {v['parts']} first grasp holds {v['first_held']:.2f}, robust {v['first_robust']:.2f}, "
              f"{v['first_held_of_6']:.1f}/6; a holding grasp within 20 feasible: {v['dynamic_found']:.2f}, "
              f"median candidates {v['dynamic_examined_median']}")
    for c in contrasts:
        print(f"   to first holding grasp {c['a']} ÷ default: {c['ratio']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] "
              f"n={c['parts']} holm={c['holm_p']:.4f}")


if __name__ == '__main__':
    main()
