import argparse
from pathlib import Path
from research_grasping.phase4.jobs import prepare
p = argparse.ArgumentParser(description='Build a development-only portable RGB job')
p.add_argument('--root', type=Path, required=True)
p.add_argument('--observations', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--object', action='append', dest='objects')
a = p.parse_args()
print(prepare(a.root, a.observations, a.output, objects=a.objects)['job_id'])
