"""Write PHASE7X_FREEZE.json: everything the new confirmation depends on, with SHA-256 digests (Git-based freeze).

    python tools/research/phase7_extension/make_freeze_x.py --binary FIXED_BINARY --gpd-adapter ADAPTER \
        --out research/phase7_extension/PHASE7X_FREEZE.json

The record lists the protocol, selection rule, models, analysis and pipeline code, the compiled planner binary
(a copy at a fixed path, used by every stage), the GPD adapter binary, the object manifest (positions 300-499 with
their mesh hashes), seeds, methods, inclusion rules and error handling. confirm_x.py re-checks `files_sha256`
and `binary_sha256` at the start of every stage. The digests are not signatures. The record is committed before
any catalogue or plan of the new set exists.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
FILES = [
    'research/phase7_extension/PHASE7X_PROTOCOL.md', 'research/phase7_extension/DEVELOPMENT_SELECTION_RULE.md',
    'research/phase7_extension/results/v3_models.json', 'research/phase7_extension/results/new_set_untouched.json',
    'research/phase7_extension/results/native_inference_validation_x.json',
    'research/phase7_extension/power/power_egad.json', 'research/phase7_extension/power/power_dev.json',
    'research/phase7_extension/frozen/p_robust_all3.pkl', 'research/phase7_extension/frozen/learned_order_x.json',
    'research/phase7_extension/frozen/freeze_x_fit.json', 'research/phase7_extension/native/learned_order_phase7.json',
    'research/phase7/frozen/p_feasible.pkl', 'research/phase7/frozen/p_robust_given_feasible.pkl',
    'research/phase7/egad_selection.json',
    'tools/research/phase7_extension/confirm_x.py', 'tools/research/phase7_extension/build_newset.py',
    'tools/research/phase7_extension/common.py', 'tools/research/phase7_extension/p2_audit.py',
    'tools/research/phase7_extension/integrated_benchmark.py', 'tools/research/phase7_extension/gpd_baseline.py',
    'tools/research/phase7_extension/gpd_adapter/gpd_score_candidates.cpp', 'tools/research/phase7_extension/v3_models.py',
    'tools/research/phase7_extension/freeze_x.py', 'tools/research/phase7_extension/export_models.py',
    'tools/research/phase7/egad_prepare.py', 'tools/research/phase7/egad_confirm.py',
    'research_grasping/phase5/catalogue.py', 'research_grasping/phase5/baselines.py', 'research_grasping/phase5/dynamics.py',
    'research_grasping/phase6/fusion.py', 'research_grasping/phase6/physics.py', 'research_grasping/phase7/physics.py',
    'crates/grasping/src/learned.rs', 'crates/grasping/src/planner.rs', 'apps/cli/src/main.rs', 'Cargo.lock',
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--binary', required=True); p.add_argument('--gpd-adapter', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *FILES], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    if dirty:
        raise SystemExit(f'commit these files before freezing:\n{dirty}')
    sel = json.loads((ROOT/'research/phase7/egad_selection.json').read_text())
    objects = [dict(position=300 + i, name=f"egad_{r['cell']}", file=r['file'], sha256=r['sha256']) for i, r in enumerate(sel['primary'][300:500])]
    xm = 'research/phase7_extension/frozen/learned_order_x.json'
    hm = 'research/phase7_extension/native/learned_order_phase7.json'
    doc = dict(
        title='Phase VII extension: freeze of the new confirmation (EGAD positions 300-499)',
        nature='Git-based freeze: SHA-256 digests committed before any catalogue, plan, score or simulation of the new set. '
               'Not a third-party signature and not a registry preregistration.',
        code_commit=commit, protocol='research/phase7_extension/PHASE7X_PROTOCOL.md',
        selection=dict(rule='research/phase7_extension/DEVELOPMENT_SELECTION_RULE.md', rule_commit='49eb5bb', result='research/phase7_extension/results/v3_models.json',
                       final='Joint-multi'),
        names=dict(X='Joint-multi', hybrid='hybrid Joint-multi/M0'),
        methods={'M0': dict(kind='native'),
                 'M1': dict(kind='native', models=hm, mode='m1'),
                 'Joint-hist': dict(kind='native', models=hm, mode='joint'),
                 'Joint-multi': dict(kind='native', models=xm, mode='joint'),
                 'hybrid Joint-multi/M0': dict(kind='native', models=xm, mode='hybrid'),
                 'GPD+gate': dict(kind='external', adapter_commit='6327f20eabfcba41a05fdd2e2ba408153dc2e958',
                                  presentation="Ranking-controlled comparison using GPD's pretrained classifier on a shared candidate catalogue.")},
        replay=dict(feasibility='research/phase7/frozen/p_feasible.pkl',
                    robustness={hm: 'research/phase7/frozen/p_robust_given_feasible.pkl', xm: 'research/phase7_extension/frozen/p_robust_all3.pkl'}),
        torsion_grid_methods=['M0', 'M1', 'Joint-hist', 'Joint-multi'],
        timed_methods=['M0', 'M1', 'Joint-hist', 'Joint-multi', 'hybrid Joint-multi/M0'], timing_reps=3, timing_seed=20261217,
        seeds=dict(timing=20261217, cluster_bootstrap=20261216, coacd=0, egad_draw=20261202),
        binary_sha256=sha(a.binary), binary_path=str(a.binary), binary_source_commit=commit,
        gpd_adapter_sha256=sha(a.gpd_adapter),
        objects=objects,
        inclusion='all 200 positions; preparation, catalogue and planner failures are recorded and not replaced; no reserve set',
        errors='unstable simulations count as failures (v3); crashed stages are resumed, not re-sampled; timing is run once',
        files_sha256={f: sha(ROOT/f) for f in FILES})
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
    print(a.out, 'commit', commit[:10], len(doc['files_sha256']), 'files', 'binary', doc['binary_sha256'][:12])


if __name__ == '__main__':
    main()
