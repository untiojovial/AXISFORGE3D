"""Development physics reports from physics_dev_with_joint.json.gz (no new simulation).

    python tools/research/phase7/physics_report.py --physics research/phase7/results/physics_dev_with_joint.json.gz \
        --out-dir research/phase7/results

Writes physical_validation.json (v2 vs v3, per condition, mass vs stress), rotation_validation.json
(thresholds 5/10/15/20 degrees) and friction_sensitivity.json (torsion 0.001/0.0025/0.005/0.02 m).
Unit: the development part (151 GSO parts); the first viable grasp of each method; paired bootstrap
(seed 20261128, 10 000 draws) and exact McNemar. Development evidence only: no correction for
multiplicity, nothing here is confirmatory.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase6.mechanism import bootstrap_diff, mcnemar  # noqa: E402
from research_grasping.phase7.physics import held_at  # noqa: E402

METHODS = ('M0', 'M1', 'M4', 'M9', 'Joint')
THRESHOLDS = (5., 10., 15., 20.)
CONTRASTS = (('Joint', 'M0'), ('Joint', 'M1'), ('M4', 'M1'), ('M9', 'M1'), ('M4', 'M0'), ('M9', 'M0'))


def ok(rec, criterion, limit=15.):
    if criterion == 'v2':
        return bool(rec.get('held'))
    return held_at(rec, limit) and not rec.get('unstable', False)


def outcomes(ph, torsion, criterion, limit=15.):
    std, loads = ph['standard'], ph['loads']
    out = {m: {} for m in METHODS}
    for n, core in ph['core'].items():
        for m in METHODS:
            t = ph['trials'][n][core[m]][torsion]
            out[m][n] = dict(robust=int(all(ok(t[c], criterion, limit) for c in std)),
                             load=int(all(ok(t[c], criterion, limit) for c in loads)),
                             **{c: int(ok(t[c], criterion, limit)) for c in std + loads})
    return out


def rates(o, keys):
    parts = sorted(next(iter(o.values())))
    return {m: {k: round(float(np.mean([o[m][n][k] for n in parts])), 4) for k in keys} for m in o}


def contrasts(o, keys=('robust', 'load')):
    parts = sorted(next(iter(o.values())))
    res = {}
    for a, b in CONTRASTS:
        for k in keys:
            A, B = [o[a][n][k] for n in parts], [o[b][n][k] for n in parts]
            res[f'{a} - {b}, {k}'] = dict(**bootstrap_diff(A, B), mcnemar=mcnemar(A, B))
    return res


def method_ranking(r, key):
    return sorted(METHODS, key=lambda m: -r[m][key])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--physics', required=True); p.add_argument('--out-dir', required=True)
    a = p.parse_args()
    ph = json.load(gzip.open(a.physics, 'rt'))
    out_dir = Path(a.out_dir)
    std, loads = ph['standard'], ph['loads']
    parts = sorted(ph['core'])
    common = dict(source=Path(a.physics).name, development_parts=len(parts), unit='part (first viable grasp of each method)',
                  evidence='development (GSO); simulation only, not a real robot; no multiplicity correction',
                  joint_betas_by_fold=ph.get('joint_betas_by_fold'))

    # --- physical_validation.json
    v2, v3 = outcomes(ph, '0.005', 'v2'), outcomes(ph, '0.005', 'v3')
    keys = ['robust', 'load'] + std + loads
    trials = [ph['trials'][n][ph['core'][n][m]]['0.005'][c] for n in parts for m in METHODS for c in std + loads]
    uniq = {}
    for n in parts:
        for lab in set(ph['core'][n].values()):
            for c in std + loads:
                uniq[(n, lab, c)] = ph['trials'][n][lab]['0.005'][c]
    recs = list(uniq.values())
    v2_held = [r for r in recs if r.get('held')]
    turn = np.array([r['turn_from_grip_deg'] for r in v2_held if 'turn_from_grip_deg' in r])
    pv = dict(common, torsion_m=0.005, criterion_v3=f"held_v3 = rise >= 80 mm and slip < 20 mm and rotation from the grip <= "
                                                    f"{ph['turn_limit_deg']} deg and contact kept and not unstable (fixed a priori)",
              mass_vs_stress=dict(
                  note='The 0.1 kg nominal mass is assumed (no measured masses). Mass perturbations 0.05 and 0.2 kg are part of '
                       'robustness; 0.5, 1 and 2 kg are stress tests of the grip (load tolerance), reported separately and '
                       'never merged into robustness. Centre of mass = planner centroid (uniform density assumed).',
                  robustness_conditions=std, stress_conditions=loads),
              rates_v2=rates(v2, keys), rates_v3=rates(v3, keys),
              v2_to_v3=dict(unique_trials=len(recs), v2_held=len(v2_held),
                            v2_held_but_v3_failed=int(sum(1 for r in v2_held if not ok(r, 'v3'))),
                            turn_from_grip_deg_among_v2_held=dict(
                                median=round(float(np.median(turn)), 2), p90=round(float(np.percentile(turn, 90)), 2),
                                share_over_15=round(float((turn > 15).mean()), 4))),
              contrasts_v3=contrasts(v3), contrasts_v2=contrasts(v2),
              articulated_simulation='not run: see articulated_simulator_validation.json',
              trials_counted=len(trials))
    (out_dir/'physical_validation.json').write_text(json.dumps(pv, indent=1) + '\n')

    # --- rotation_validation.json
    rv = dict(common, torsion_m=0.005, thresholds_deg=list(THRESHOLDS), primary_threshold_deg=15.,
              rationale='15 deg was fixed before any v3 result: it moves a point 80 mm from the grip axis by about 20 mm, '
                        'the slip limit. Other thresholds are sensitivity only.', by_threshold={})
    base_rank = None
    for th in THRESHOLDS:
        o = outcomes(ph, '0.005', 'v3', th)
        r = rates(o, ('robust', 'load'))
        row = dict(rates=r, ranking_robust=method_ranking(r, 'robust'), ranking_load=method_ranking(r, 'load'),
                   contrasts=contrasts(o))
        if th == 15.:
            base_rank = r
        rv['by_threshold'][str(int(th))] = row
    for th in THRESHOLDS:
        r = rv['by_threshold'][str(int(th))]['rates']
        rv['by_threshold'][str(int(th))]['kendall_tau_vs_15deg'] = {
            k: round(float(kendalltau([base_rank[m][k] for m in METHODS], [r[m][k] for m in METHODS]).statistic), 3)
            for k in ('robust', 'load')}
    (out_dir/'rotation_validation.json').write_text(json.dumps(rv, indent=1) + '\n')

    # --- friction_sensitivity.json
    fs = dict(common, criterion='v3 (15 deg)', torsions_m=ph['torsions'],
              note='Torsional friction (MuJoCo condim 4) is not calibrated against hardware; 0.005 m is the value used in '
                   'Phases VI-VII. The grid shows how much the conclusions depend on it.', by_torsion={})
    for tor in ph['torsions']:
        o = outcomes(ph, str(tor), 'v3')
        r = rates(o, ('robust', 'load'))
        fs['by_torsion'][str(tor)] = dict(rates=r, ranking_robust=method_ranking(r, 'robust'),
                                          ranking_load=method_ranking(r, 'load'), contrasts=contrasts(o))
    # verdict flips per grasp between torsion values (unique first grasps)
    flips = {}
    for tor in ph['torsions']:
        if str(tor) == '0.005':
            continue
        n_tot = n_flip = 0
        for n in parts:
            for lab in set(ph['core'][n].values()):
                for c in std + loads:
                    r0, r1 = ph['trials'][n][lab]['0.005'][c], ph['trials'][n][lab][str(tor)][c]
                    n_tot += 1; n_flip += ok(r0, 'v3') != ok(r1, 'v3')
        flips[f'{tor} vs 0.005'] = dict(trials=n_tot, verdicts_changed=n_flip, share=round(n_flip/n_tot, 4))
    fs['verdict_changes_vs_0.005'] = flips
    signs = {}
    for key in fs['by_torsion']['0.005']['contrasts']:
        vals = [fs['by_torsion'][str(t)]['contrasts'][key]['difference'] for t in ph['torsions']]
        lo = [fs['by_torsion'][str(t)]['contrasts'][key]['ci95'][0] for t in ph['torsions']]
        hi = [fs['by_torsion'][str(t)]['contrasts'][key]['ci95'][1] for t in ph['torsions']]
        signs[key] = dict(differences=vals, same_sign_all=bool(all(v > 0 for v in vals) or all(v < 0 for v in vals)),
                          ci_excludes_zero_all=bool(all(l > 0 for l in lo) or all(h < 0 for h in hi)))
    fs['contrast_stability'] = signs
    (out_dir/'friction_sensitivity.json').write_text(json.dumps(fs, indent=1) + '\n')
    for name, d in (('physical', pv['rates_v3']), ):
        print(name, json.dumps({m: (d[m]['robust'], d[m]['load']) for m in METHODS}))
    print('v2->v3', pv['v2_to_v3'])
    print('flips', flips)
    for k, v in signs.items():
        print(k, v)
    for th in THRESHOLDS:
        print(th, rv['by_threshold'][str(int(th))]['kendall_tau_vs_15deg'], rv['by_threshold'][str(int(th))]['ranking_robust'])


if __name__ == '__main__':
    main()
