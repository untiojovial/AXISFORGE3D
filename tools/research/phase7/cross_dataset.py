"""Development (GSO) vs confirmation (EGAD) summary: same methods, same physics, different object populations.

    python tools/research/phase7/cross_dataset.py --results research/phase7/results --out research/phase7/results/cross_dataset_results.json

Descriptive only (PHASE7_PROTOCOL.md II.6): the two populations differ (household scans vs evolved shapes
scaled to the gripper) and the development numbers are out of fold, the EGAD numbers frozen. No test.
"""
import argparse
import json
from pathlib import Path

METHODS = ('M0', 'M1', 'M4', 'M9', 'Joint')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    R = Path(a.results)
    pv = json.loads((R/'physical_validation.json').read_text())
    nt = json.loads((R/'negative_transfer_analysis.json').read_text())
    conf = json.loads((R/'confirmation_results.json').read_text())
    dev = {m: dict(N_F_gmean=nt['methods']['M1']['N_F_M0_gmean'] if m == 'M0' else nt['methods'][m]['N_F_gmean'],
                   robust_v3=pv['rates_v3'][m]['robust'], load_tolerant_v3=pv['rates_v3'][m]['load'],
                   robust_v2=pv['rates_v2'][m]['robust'], load_tolerant_v2=pv['rates_v2'][m]['load']) for m in METHODS}
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '),
               development=dict(population='GSO household scans (151 viable development parts, out of fold)', methods=dev,
                                contrasts_v3={k: v for k, v in pv['contrasts_v3'].items()}),
               confirmation={})
    for set_name, S in conf['sets'].items():
        out['confirmation'][set_name] = dict(
            population=f"EGAD {set_name}: {S['viable']} viable of {S['objects']} objects", status_counts=S['status_counts'],
            methods={m: {k: S['rates_on_viable'][m][k] for k in ('N_F_gmean', 'robust_v3', 'load_tolerant_v3', 'robust_v2', 'load_tolerant_v2')}
                     for m in METHODS})
        out['confirmation'][set_name]['difference_EGAD_minus_development'] = {
            m: {k: round(S['rates_on_viable'][m][k]-dev[m][k], 4) for k in ('robust_v3', 'load_tolerant_v3')} for m in METHODS}
        out['confirmation'][set_name]['N_F_ratio_to_M0'] = {
            'development': {m: round(dev[m]['N_F_gmean']/dev['M0']['N_F_gmean'], 4) for m in METHODS},
            'EGAD': {m: round(S['rates_on_viable'][m]['N_F_gmean']/S['rates_on_viable']['M0']['N_F_gmean'], 4) for m in METHODS}}
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['confirmation'], indent=1)[:3000])


if __name__ == '__main__':
    main()
