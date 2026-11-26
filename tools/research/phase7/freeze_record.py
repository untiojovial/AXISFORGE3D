"""Write research/phase7/PHASE7_MODEL_FREEZE.json from the freeze manifest, the code and the protocol.

    python tools/research/phase7/freeze_record.py --frozen frozen/ --out research/phase7/PHASE7_MODEL_FREEZE.json \
        --copy-to research/phase7/frozen

Signs the frozen confirmation by SHA-256: model files, every code file on the confirmation path, the
protocol, the selection (test manifest) and the inputs the models were trained on. Copies the gradient
boosting models and the geometry-only MLPs into the repository; MLPs whose inputs include DINOv3 or the
untrained ViT stay out (hash only; regenerate with egad_confirm.py freeze), pending the author's decision
on redistributing DINOv3-derived artefacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
CODE = ['tools/research/phase7/egad_prepare.py', 'tools/research/phase7/egad_confirm.py', 'tools/research/phase7/freeze_record.py',
        'research_grasping/phase5/baselines.py', 'research_grasping/phase5/catalogue.py', 'research_grasping/phase5/dynamics.py',
        'research_grasping/phase5/stats.py', 'research_grasping/phase6/fusion.py', 'research_grasping/phase6/physics.py',
        'research_grasping/phase6/mechanism.py', 'research_grasping/phase7/physics.py',
        'research_grasping/phase7/representation.py', 'research_grasping/phase7/articulated.py',
        'research_grasping/phase4/visual_prior.py', 'research_grasping/phase4/frozen_models.py',
        'crates/grasping/src/planner.rs', 'apps/cli/src/main.rs', 'Cargo.lock']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frozen', required=True); p.add_argument('--out', required=True); p.add_argument('--copy-to', required=True)
    a = p.parse_args()
    man = json.loads((Path(a.frozen)/'freeze_manifest.json').read_text())
    for k, h in man['files'].items():
        assert sha(Path(a.frozen)/k) == h, k
    dest = Path(a.copy_to); dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for k in man['files']:
        if k.endswith('.pkl') or k.startswith('mlp_L2_'):
            shutil.copy2(Path(a.frozen)/k, dest/k); copied.append(k)
    shutil.copy2(Path(a.frozen)/'freeze_manifest.json', dest/'freeze_manifest.json')
    sel = json.loads((ROOT/'research/phase7/egad_selection.json').read_text())
    doc = dict(
        title='Phase VII model freeze: FORTIS-R Joint and comparators, before the EGAD confirmation',
        status='frozen before any confirmatory object was planned',
        final_method=dict(name='FORTIS-R Joint', order='p~ * P(H|F) * max(Q_cen, 1e-3)^0.3, planner stroke gate first',
                          main_baseline='M0 (original planner)', best_learned_baseline='M1',
                          physical_criterion='v3: rise >= 80 mm, slip < 20 mm, rotation from the grip <= 15 deg, contact kept; '
                                             'torsion 0.005 m; nine conditions', operational_metric='N_F',
                          configuration='demo cell, 80 mm parallel gripper, part at (0.42, 0, 0.42), assumed mass 0.1 kg'),
        models=man['models'], hyperparameters=man['hyperparameters'], methods=man['methods'],
        normalisation='p~ = P(F)/max over the object; Q_cen = minmax(clip(1 - centre_offset/finger_depth, 0, 1)); '
                      'MLP inputs standardised with training-set statistics stored in each .pt file',
        thresholds=dict(rotation_deg=15, rise_m=.08, slip_m=.02, ratio_margin_P3=1.25, margin_P2=.05, margin_S5=.10,
                        alpha_one_sided=.025),
        training_data=man['training_data'], environment=man['environment'],
        model_files_sha256=man['files'], model_files_in_repository=sorted(copied),
        model_files_not_in_repository={k: 'DINOv3- or untrained-ViT-input MLP; regenerate with egad_confirm.py freeze and '
                                          'check the hash' for k in man['files'] if k not in copied},
        code_sha256={c: sha(ROOT/c) for c in CODE},
        code_commit_at_training=man['code_commit'],
        protocol=dict(file='research/phase7/PHASE7_PROTOCOL.md', sha256=sha(ROOT/'research/phase7/PHASE7_PROTOCOL.md'),
                      part='II'),
        test_manifest=dict(selection_file='research/phase7/egad_selection.json',
                           selection_sha256=sha(ROOT/'research/phase7/egad_selection.json'),
                           archives=dict(l.split()[::-1] for l in (ROOT/'research/phase7/egad_archives.sha256').read_text().splitlines()),
                           primary=[dict(position=i, **{k: r[k] for k in ('cell', 'file', 'sha256')})
                                    for i, r in enumerate(sel['primary'][:300])],
                           evaluation_set=[{k: r[k] for k in ('cell', 'file', 'sha256')} for r in sel['evaluation_set']],
                           excluded_smoke_test_positions=[500, 501, 502], reserve_positions='300-598'),
        statistical_analysis='PHASE7_PROTOCOL.md II.6; implemented in egad_confirm.py analyse',
        failure_handling='PHASE7_PROTOCOL.md II.7 and II.9',
        budget=dict(catalogue_timeout_s=3600, physics='first grasps only; torsion grid for M0, M1 and Joint',
                    timing_objects=60, compute='CPU only, no GPU'))
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
    print('frozen:', len(man['files']), 'model files;', len(copied), 'copied;', 'protocol', doc['protocol']['sha256'][:12])


if __name__ == '__main__':
    main()
