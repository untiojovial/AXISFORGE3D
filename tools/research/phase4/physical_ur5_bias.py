"""Reproduce preregistered exact-model bias comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
from research_grasping.phase4.full_robot_compensated import run


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf',type=Path,default=Path('apps/viewer/samples/ur5/robot.urdf'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result={'urdf_sha256':sha(a.urdf),'code_sha256':sha('research_grasping/phase4/full_robot_compensated.py'),
            'baseline_code_sha256':sha('research_grasping/phase4/full_robot_ur5.py'),
            'mujoco_version':mujoco.__version__,'scope':'exact simulated model bias, no contacts',
            'runs':[run(a.urdf,dt) for dt in (.0005,.001,.002)]}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
