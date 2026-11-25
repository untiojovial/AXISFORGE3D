"""Run the embedded notebook cells offline. Optionally reproduce OpenCLIP in CPU."""
import argparse
import ast
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--openclip',type=Path,help='authorized OpenCLIP checkpoint to execute the inference cell offline')
parser.add_argument('--compare-bundle',type=Path,help='compare each feature file with this independently produced bundle')
parser.add_argument('--report',type=Path)
args=parser.parse_args()
if args.compare_bundle and not args.openclip:parser.error('comparison requires --openclip')
root=Path(__file__).resolve().parents[3]
notebook=root/'notebooks/AxisForge_Phase4_Frozen_Foundation_Models.ipynb'
nb=json.loads(notebook.read_text())
cells=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
for cell in cells:
    if not cell.startswith('%pip'):compile(cell,'<notebook>','exec')
bootstrap=next(c for c in cells if 'payload = ' in c)
tree=ast.parse(bootstrap)
assignment=next(x for x in tree.body if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='payload')
payload=ast.literal_eval(assignment.value)
for name,encoded in payload.items():
    if gzip.decompress(base64.b64decode(encoded))!=(root/name).read_bytes():
        raise ValueError('embedded payload stale: '+name)
report=dict(notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),embedded_files=len(payload),
            pip_executed=False,google_colab_runtime=False,device='cpu',feature_files_compared=0)
with tempfile.TemporaryDirectory(prefix='axisforge-notebook-') as temporary:
    staging=Path(temporary)
    prefix="""import socket, sys, types
def offline(*args,**kwargs): raise RuntimeError('network disabled in notebook verification')
socket.socket.connect=offline
socket.socket.connect_ex=offline
google=types.ModuleType('google')
colab=types.ModuleType('google.colab')
colab.files=types.SimpleNamespace(download=lambda path: __import__('pathlib').Path(path).stat())
sys.modules['google']=google
sys.modules['google.colab']=colab
"""
    selected=[c for c in cells if not c.startswith('%pip')]
    if args.openclip:
        (staging/'vit_b_32-quickgelu-laion400m_e31-d867053b.pt').symlink_to(args.openclip.resolve(strict=True))
    else:
        selected=selected[:3]
    script=staging/'offline_cells.py'
    script.write_text(prefix+'\n'.join(c.replace('/content',temporary) for c in selected))
    subprocess.run([sys.executable,str(script)],cwd=staging,check=True,
                   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    if args.compare_bundle:
        with zipfile.ZipFile(staging/'axisforge-bundle.zip') as ran, zipfile.ZipFile(args.compare_bundle) as previous:
            expected={n for n in previous.namelist() if n.startswith('features/')}
            if expected!={n for n in ran.namelist() if n.startswith('features/')} or len(expected)!=22:
                raise ValueError('incomplete notebook extraction')
            for name in sorted(expected):
                if ran.read(name)!=previous.read(name):raise ValueError('feature discrepancy: '+name)
            report.update(feature_files_compared=len(expected),compared_views=16*len(expected),
                          notebook_run_id=json.loads(ran.read('index.json'))['run_id'])
report['executed_cells']=len(selected)
if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
