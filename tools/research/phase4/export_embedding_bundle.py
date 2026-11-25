import argparse
from pathlib import Path
from research_grasping.phase4.orchestrator import export_bundle
p=argparse.ArgumentParser(); p.add_argument('--job',type=Path,required=True); p.add_argument('--work',type=Path,required=True); p.add_argument('--bundle',type=Path,required=True)
a=p.parse_args(); export_bundle(a.job,a.work,a.bundle); print('VERIFIED',a.bundle)
