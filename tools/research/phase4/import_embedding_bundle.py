import argparse
from pathlib import Path
from research_grasping.phase4.bundle_io import install_bundle, verify_bundle
p=argparse.ArgumentParser(); p.add_argument('--job',type=Path,required=True); p.add_argument('--bundle',type=Path,required=True); p.add_argument('--destination',type=Path,required=True); p.add_argument('--data-root',type=Path); p.add_argument('--analysis-out',type=Path); p.add_argument('--expected-checkpoint-sha256', help='independently verified authorized checkpoint SHA-256'); p.add_argument('--expected-revision'); p.add_argument('--expected-processor-sha256')
a=p.parse_args(); preview=verify_bundle(a.bundle,a.job)
known_openclip='d867053b2301634007ed9af230bfb1a217ec634f6c0329f04092133ae5c4b89e'
expected=a.expected_checkpoint_sha256
if preview['model_id']=='open_clip/ViT-B-32-quickgelu/laion400m_e31': expected=expected or known_openclip
if not expected or preview['checkpoint_sha256'] != expected: raise SystemExit('BLOCKED: supply independently verified checkpoint SHA-256 matching the bundle')
index=install_bundle(a.bundle,a.job,a.destination)
print('VERIFIED',index['job_id'])
try:
    from research_grasping.phase4.bundle_analysis import run
    run(a.destination, data_root=a.data_root, out=a.analysis_out, bundle_path=a.bundle, job_path=a.job, expected_checkpoint_sha256=a.expected_checkpoint_sha256, expected_revision=a.expected_revision, expected_processor_sha256=a.expected_processor_sha256)
except (FileNotFoundError, ImportError) as exc:
    print('ANALYSIS BLOCKED:',type(exc).__name__,str(exc)); print('Resume by rerunning this import command after restoring the missing dependency or data.')
    raise SystemExit(2) from exc
