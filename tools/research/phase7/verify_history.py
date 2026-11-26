"""Check the Phase V and VI figures quoted in the Phase VII request against the stored results.

    python tools/research/phase7/verify_history.py --out phase6_verification.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P5 = ROOT/'research/phase5/results'
P6 = ROOT/'research/phase6/results'


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    checks = []

    def check(what, claimed, found, ok, source):
        checks.append(dict(what=what, claimed=claimed, found=found, ok=bool(ok), source=source))
    an = json.loads((P5/'analysis.json').read_text())
    prim = {(c['a'], c['b']): c for c in an['primary']}
    for (x, y), (r, lo, hi) in {('learned', 'box'): (.565, .462, .685), ('afat', 'learned'): (1.058, .986, 1.142),
                                ('afat', 'afat-novis'): (1.011, .959, 1.070),
                                ('visual:dinov3', 'visual:vit-random'): (.322, .259, .399)}.items():
        c = prim[(x, y)]
        check(f'Phase V {x}/{y}', [r, lo, hi], [round(c['ratio'], 3)] + [round(v, 3) for v in c['ci95']],
              close(c['ratio'], r, .001) and close(c['ci95'][0], lo, .001) and close(c['ci95'][1], hi, .001), 'research/phase5/results/analysis.json')
    sec = [c for c in an['secondary'] if c['a'] == 'learned' and c['b'] == 'default' and c['experiment'] == 'B'
           and c['gripper'] == 80 and c['target'] == 1][0]
    check('Phase V learned/default (B, 80 mm)', [.207, .152, .282], [round(sec['ratio'], 3)] + [round(v, 3) for v in sec['ci95']],
          close(sec['ratio'], .207, .001) and close(sec['ci95'][0], .152, .001) and close(sec['ci95'][1], .282, .001),
          'research/phase5/results/analysis.json')
    fb = json.loads((P6/'fusion_benchmark.json').read_text())
    rp = json.loads((P6/'robustness_prediction.json').read_text())
    for m, (r, lt) in {'M0': (.841, .649), 'M1': (.907, .483), 'M4': (.927, .603)}.items():
        s = fb['summary'][m]
        check(f'Phase VI {m} robust / load tolerant', [r, lt], [s['robust_rate'], s['load_tolerant_rate']],
              close(s['robust_rate'], r, .0006) and close(s['load_tolerant_rate'], lt, .0006), 'research/phase6/results/fusion_benchmark.json')
    s = rp['summary']
    check('Phase VI M9 robust / load tolerant', [.960, .464], [s['robust_rate'], s['load_tolerant_rate']],
          close(s['robust_rate'], .960, .0006) and close(s['load_tolerant_rate'], .464, .0006), 'research/phase6/results/robustness_prediction.json')
    c = next(c for c in fb['contrasts']['load_tolerant'] if c['a'] == 'M4' and c['b'] == 'M1')
    check('Phase VI M4 - M1 load tolerance', [.119, .066, .179], [c['difference']['difference']] + c['difference']['ci95'],
          close(c['difference']['difference'], .119, .0006) and close(c['difference']['ci95'][0], .066, .0006)
          and close(c['difference']['ci95'][1], .179, .0006), 'fusion_benchmark.json')
    c = next(c for c in fb['contrasts']['load_tolerant'] if c['a'] == 'M4' and c['b'] == 'M0')
    check('Phase VI M4 vs M0 load tolerance not shown superior', 'CI includes 0', c['difference']['ci95'],
          c['difference']['ci95'][0] < 0 < c['difference']['ci95'][1], 'fusion_benchmark.json')
    c = rp['contrasts']['vs_M1']['robust']['difference']
    check('Phase VI M9 - M1 robustness', [.053, .020, .093], [c['difference']] + c['ci95'],
          close(c['difference'], .053, .0006) and close(c['ci95'][0], .020, .0006) and close(c['ci95'][1], .093, .0006),
          'robustness_prediction.json')
    oc = json.loads((P6/'cost_profile_online.json').read_text())
    r1 = oc['methods']['M1']['time_ratio_vs_M0']['ratio']; r4 = oc['methods']['M4']['time_ratio_vs_M0']['ratio']
    check('Phase VI time per planner call vs default (about -24 %)', '~0.76', [r1, r4], .70 < r1 < .80 and .70 < r4 < .80,
          'cost_profile_online.json (60 parts; planner call only, prior construction excluded)')
    rnd = fb['reference']['random_within_gate_over_M0']
    check('Phase VI random within gate / default', [1.18, .91, 1.52], [rnd['ratio']] + rnd['ci95'],
          close(rnd['ratio'], 1.18, .006) and close(rnd['ci95'][0], .91, .006) and close(rnd['ci95'][1], 1.52, .006), 'fusion_benchmark.json')
    pv = json.loads((P6/'physics_validation.json').read_text())
    check('Phase VI torsion sensitivity (verdicts changed of 180)', [17, 20],
          [180-pv['torsion_0.001']['agree'], 180-pv['torsion_0.02']['agree']],
          180-pv['torsion_0.001']['agree'] == 17 and 180-pv['torsion_0.02']['agree'] == 20, 'physics_validation.json')
    out = dict(checks=checks, passed=sum(c['ok'] for c in checks), total=len(checks))
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    for c in checks:
        print('OK ' if c['ok'] else 'BAD', c['what'], c['claimed'], c['found'])


if __name__ == '__main__':
    main()
