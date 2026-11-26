"""Machine-readable addendum to the Phase VII confirmation results (issues A, E and F of the corrections register).

    python tools/research/phase7_extension/corrections_addendum.py --results research/phase7/results \
        --out research/phase7_extension/results/corrections_addendum.json

Reads only committed files; confirmation_results.json itself is never modified.

A  The evaluation-set block of confirmation_results.json carries an S6 (H4) entry that is a copy of the
   primary one: the analysis added S6 to every set from the same timing file, and none of the 60 timed
   objects is in the evaluation set. The correct evaluation-set value is "not measured". Holm over the
   evaluation-set secondary family is recomputed without it.
E  The denominators of every reported quantity, recounted from the files.
F  Holm-adjusted p-values stored as 0 (rounded to 6 decimals) are given at full precision.
"""
import argparse
import gzip
import json
from pathlib import Path


def holm(ps):
    order = sorted(ps, key=ps.get)
    run, out = 0., {}
    for rank, k in enumerate(order):
        run = max(run, min(1., (len(ps)-rank)*ps[k])); out[k] = run
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', required=True); p.add_argument('--out', required=True)
    p.add_argument('--phase6', default='research/phase6/dev_parts.json')
    p.add_argument('--library', default='research/phase5/library_parts.txt')
    a = p.parse_args()
    R = Path(a.results)
    res = json.loads((R/'confirmation_results.json').read_text())['sets']
    rows = json.loads((R/'confirmation_orders.json').read_text())['rows']
    timing = json.loads((R/'confirmation_timing.json').read_text())
    costs = json.loads((R/'full_online_costs.json').read_text())
    dev = json.loads(Path(a.phase6).read_text())
    library = [s.strip() for s in Path(a.library).read_text().splitlines() if s.strip()]
    trials = json.load(gzip.open(R/'confirmation_trials.json.gz', 'rt'))

    ev = res['evaluation set (secondary)']['secondary']
    pr = res['primary']['secondary']
    k6 = next(k for k in ev if k.startswith('S6'))
    timed = sorted(timing['per_object'])
    A = dict(entry=k6, evaluation_set_value_as_stored=ev[k6], primary_value=pr[k6], identical=ev[k6] == pr[k6],
             timed_objects=len(timed), timed_objects_in_evaluation_set=sum(n.startswith('egad_eval_') for n in timed),
             correct_evaluation_set_value='not measured (no evaluation-set object was timed)',
             evaluation_set_holm_without_S6={k: v for k, v in holm({k: v['p_one_sided'] for k, v in ev.items() if k != k6}).items()},
             evaluation_set_holm_as_stored={k: v['holm_p'] for k, v in ev.items()},
             cause='tools/research/phase7/egad_confirm.py adds S6 to each set from the same timing file without filtering by set')
    prim = sorted(n for n in rows if not n.startswith('egad_eval_'))
    evs = sorted(n for n in rows if n.startswith('egad_eval_'))
    viable = lambda names: [n for n in names if rows[n]['status'] == 'viable']
    dyn = lambda names: [n for n in viable(names) if n in trials]
    E = {
        'development parts with labelled catalogues (feature/probability analyses; P(F) training set)':
            len(set(library) | set(dev['parts']) | set(dev['nonviable'])),
        'development dynamic parts (physics, fusion selection; the manuscript\'s \\devParts)': len(dev['parts']),
        'development parts with full online cost measured': costs['parts'],
        'primary EGAD objects (population metrics)': len(prim),
        'primary viable objects (P1-P3, S1-S5, S7; rates conditional on a feasible grasp)': len(viable(prim)),
        'primary viable objects with dynamics': len(dyn(prim)),
        'primary objects without any viable candidate': len(prim) - len(viable(prim)),
        'evaluation-set objects': len(evs), 'evaluation-set viable objects': len(viable(evs)),
        'timed objects (S6, seeded subset of the primary viable objects)': len(timed),
        'timed objects that are primary viable': sum(n in set(viable(prim)) for n in timed)}
    F = {}
    for k, v in res['primary']['primary'].items():
        F[k] = dict(p_one_sided=v['p_one_sided'], holm_p_stored=v['holm_p'])
    full = holm({k: v['p_one_sided'] for k, v in res['primary']['primary'].items()})
    for k in F:
        F[k]['holm_p_full_precision'] = full[k]
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '),
               A_S6_in_evaluation_set=A, E_denominators=E, F_full_precision_holm_primary=F)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1)[:4000])


if __name__ == '__main__':
    main()
