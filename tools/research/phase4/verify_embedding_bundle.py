import argparse
from pathlib import Path
from research_grasping.phase4.bundle_io import verify_bundle
p=argparse.ArgumentParser(); p.add_argument('--job',type=Path,required=True); p.add_argument('--bundle',type=Path,required=True)
a=p.parse_args(); result=verify_bundle(a.bundle,a.job); print('VERIFIED', result['job_id'], len(result['objects']))
