"""Analyse Phase V rows: contrasts, budgets, negative transfer, per experiment and gripper.

    python tools/research/phase5/analyse.py --rows test.json --protocol protocol.json --out analysis.json

The protocol names the primary contrasts (Holm across them), the secondary
family (Holm across all of it), the equivalence margin, the experiments and
grippers. Without --protocol, a development summary is printed.
"""
import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import stats  # noqa: E402

SUMMARY = ['default', 'random-order', 'heuristic:com', 'heuristic:narrow', 'heuristic:antipodal',
           'heuristic:topdown', 'learned', 'random', 'box', 'pooled', 'visual:dinov2', 'visual:dinov3',
           'visual:dinov3-metric', 'visual:d2', 'visual:vit-random', 'fusion:dinov3', 'fusion:d2',
           'fusion:vit-random', 'afat', 'afat-novis', 'self']


def summary(rows, label):
    print(f'== {label}')
    viable = sorted({r['part'] for r in rows if r['feasible'] > 0})
    parts = sorted({r['part'] for r in rows})
    print(f'   {len(parts)} parts, {len(viable)} with a feasible grasp')
    out = {}
    for m in SUMMARY:
        c = stats.contrast(rows, m, 'default') if m != 'default' else None
        b = stats.contrast(rows, m, 'box') if m not in ('box',) else None
        bud = stats.budget(rows, m)
        nt = stats.negative_transfer(rows, m) if m != 'default' else None
        out[m] = dict(vs_default=c, vs_box=b, budget=bud, negative=nt)
        print(f"   {m:22s} ÷defecto {c['ratio'] if c else 1:6.3f}  ÷tamaño {b['ratio'] if b else 1:6.3f}"
              f"  éxito@1 {bud['budgets']['1']['rate']:.2f} @10 {bud['budgets']['10']['rate']:.2f}"
              f" @50 {bud['budgets']['50']['rate']:.2f}  peor que defecto {nt['rate'] if nt else 0:.2f}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rows', required=True); p.add_argument('--protocol'); p.add_argument('--out', required=True)
    p.add_argument('--exclude-parts', help='file of part names to leave out (sensitivity analysis)')
    a = p.parse_args()
    rows = json.loads(Path(a.rows).read_text())['rows']
    if a.exclude_parts:
        drop = set(Path(a.exclude_parts).read_text().split())
        rows = [r for r in rows if r['part'] not in drop]
    if not a.protocol:
        groups = sorted({(r['experiment'], r['gripper']) for r in rows})
        res = {f'{e}/{g}': summary(stats.select(rows, e, g), f'{e} {g} mm') for e, g in groups}
        Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
        return
    proto = json.loads(Path(a.protocol).read_text())
    margin = proto['equivalence_margin']
    main_rows = stats.select(rows, proto['primary_experiment'], proto['primary_gripper'])
    primary = []
    for c in proto['primary']:
        r = stats.contrast(main_rows, c['a'], c['b'], target=1, margin=margin)
        r.update(kind=c['kind'], question=c['question'])
        primary.append(r)
    sup = [r for r in primary if r['kind'] == 'superiority']
    stats.holm(sup)
    for r in primary:
        if r['kind'] == 'equivalence':
            r['decision'] = 'equivalent' if r['equivalent'] else 'not shown'
        else:
            r['decision'] = 'different' if r['holm_p'] < .05 and not (r['ci95'][0] <= 1 <= r['ci95'][1]) else 'not shown'
    secondary = []
    for e in proto['experiments']:
        for g in proto['grippers']:
            sub = stats.select(rows, e, g)
            if not sub:
                continue
            for t in (1, 3):
                for a_, b_ in proto['secondary']:
                    if (e, g, t) == (proto['primary_experiment'], proto['primary_gripper'], 1) and \
                            any(c['a'] == a_ and c['b'] == b_ for c in proto['primary']):
                        continue
                    r = stats.contrast(sub, a_, b_, target=t, margin=margin)
                    if r:
                        r.update(experiment=e, gripper=g)
                        secondary.append(r)
    stats.holm(secondary)
    budgets, negative = {}, {}
    for e in proto['experiments']:
        for g in proto['grippers']:
            sub = stats.select(rows, e, g)
            if sub:
                budgets[f'{e}/{g}'] = [stats.budget(sub, m) for m in proto['budget_methods']]
                negative[f'{e}/{g}'] = [stats.negative_transfer(sub, m) for m in proto['budget_methods'] if m != 'default']
    parts = sorted({r['part'] for r in main_rows})
    viable = sorted({r['part'] for r in main_rows if r['feasible'] > 0})
    out = dict(primary=primary, secondary=secondary, budgets=budgets, negative_transfer=negative,
               parts=len(parts), viable_parts=len(viable), margin=margin,
               note='Unit of analysis: the part. Test parts are never in the library, so outcomes are '
                    'independent given the frozen library; rows of one part are not independent.')
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    for r in primary:
        print(f"PRIMARY {r['kind']:11s} {r['a']} vs {r['b']}: {r['ratio']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] "
              f"{r['better']}/{r['equal']}/{r['worse']} of {r['parts']}  p={r['wilcoxon_p']:.4f} "
              f"holm={r.get('holm_p', float('nan')):.4f} tost_p={r['tost_p']:.4f} -> {r['decision']}")
    print(f'{len(parts)} parts, {len(viable)} viable; {len(secondary)} secondary contrasts (Holm-adjusted)')
    for r in secondary:
        if r['experiment'] == proto['primary_experiment'] and r['gripper'] == proto['primary_gripper'] and r['target'] == 1:
            print(f"   {r['a']:22s} vs {r['b']:18s} {r['ratio']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] "
                  f"holm={r['holm_p']:.4f} equiv={r['equivalent']}")



if __name__ == '__main__':
    main()
