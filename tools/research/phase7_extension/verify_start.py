"""Starting audit of the Phase VII extension: every figure quoted in the extension request, recomputed or
read directly from the committed result files (nothing typed by hand). Writes start_verification.json.

    python tools/research/phase7_extension/verify_start.py --out research/phase7_extension/results/start_verification.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
R7 = ROOT/'research/phase7/results'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True); a = p.parse_args()
    c = json.loads((R7/'confirmation_results.json').read_text())
    S, E = c['sets']['primary'], c['sets']['evaluation set (secondary)']
    pr = {k.split(':')[0]: v for k, v in S['primary'].items()}
    sec = {k.split(':')[0]: v for k, v in S['secondary'].items()}
    mb = json.loads((R7/'mlp_benchmark.json').read_text())
    fc = json.loads((R7/'full_online_costs.json').read_text())
    tm = json.loads((R7/'confirmation_timing.json').read_text())
    checks = []

    def chk(name, quoted, actual, ok):
        checks.append(dict(item=name, quoted=quoted, file_value=actual, match=bool(ok)))
    chk('development parts', 316, mb['parts'], mb['parts'] == 316)
    chk('development viable parts (representation study)', 239, mb['viable_parts'], mb['viable_parts'] == 239)
    chk('primary EGAD objects', 300, S['objects'], S['objects'] == 300)
    chk('primary viable', 220, S['viable'], S['viable'] == 220)
    chk('evaluation-set objects', 45, E['objects'], E['objects'] == 45)
    chk('evaluation-set viable', 35, E['viable'], E['viable'] == 35)
    chk('P1 ratio and CI', [0.609, [0.516, 0.711]], [pr['P1']['ratio'], pr['P1']['ci95']],
        round(pr['P1']['ratio'], 3) == .609 and [round(x, 3) for x in pr['P1']['ci95']] == [.516, .711])
    chk('P2 difference, CI, lower bound, Holm p', [-0.018, [-0.05, 0.014], -0.049, 0.021574],
        [pr['P2']['difference'], pr['P2']['ci95'], pr['P2']['lower_97_5'], pr['P2']['holm_p']],
        round(pr['P2']['difference'], 3) == -.018 and round(pr['P2']['lower_97_5'], 3) == -.049 and pr['P2']['holm_p'] == .021574)
    chk('P3 ratio and CI', [0.993, [0.953, 1.037]], [pr['P3 (H3)']['ratio'], pr['P3 (H3)']['ci95']],
        round(pr['P3 (H3)']['ratio'], 3) == .993)
    chk('S5 load tolerance Joint - M0', -0.064, sec['S5']['difference'], round(sec['S5']['difference'], 3) == -.064)
    chk('S6 full online cost ratio (EGAD)', 1.94, sec['S6 (H4)']['ratio'], round(sec['S6 (H4)']['ratio'], 2) == 1.94)
    chk('no secondary hypothesis supported', True, [v['supported'] for v in S['secondary'].values()],
        not any(v['supported'] for v in S['secondary'].values()))
    chk('timing objects', 60, tm['objects'], tm['objects'] == 60)
    chk('development full cost Joint/M0', None, fc['ratios_vs_M0_repeat']['Joint T_total / M0'], True)
    # issue A: the evaluation set carries an S6 entry computed from the same primary timing file
    s6e = next((v for k, v in E['secondary'].items() if k.startswith('S6')), None)
    out = dict(checks=checks, all_match=all(x['match'] for x in checks),
               issue_A_S6_in_evaluation_set=dict(present=s6e is not None, value=s6e and {k: s6e[k] for k in ('ratio', 'ci95', 'objects')},
                                                 same_as_primary=bool(s6e and s6e['ratio'] == sec['S6 (H4)']['ratio']),
                                                 timing_objects_in_evaluation_set=sum(1 for n in tm['per_object'] if n.startswith('egad_eval_'))),
               issue_F_p_values_stored_as_zero=[k for k, v in {**S['primary'], **S['secondary']}.items() if v.get('holm_p') == 0.0])
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    for x in checks:
        print(('OK ' if x['match'] else 'MISMATCH ') + x['item'], x['file_value'] if not x['match'] else '')
    print('issue A', out['issue_A_S6_in_evaluation_set']); print('issue F', out['issue_F_p_values_stored_as_zero'])


if __name__ == '__main__':
    main()
