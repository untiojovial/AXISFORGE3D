"""Independent check of a built package: no DINOv3-derived arrays, no YCB per-object data, no manuscript, and every
file matches its manifest hash. Exits non-zero on any finding.

    python tools/package/verify_package.py PACKAGE_DIR
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

YCB = ('Banana', 'ChipsCan', 'CrackerBox', 'FoamBrick', 'GelatinBox', 'Hammer', 'MasterChefCan', 'MediumClamp', 'MustardBottle',
       'Pear', 'PottedMeatCan', 'PowerDrill', 'Scissors', 'Strawberry', 'TennisBall', 'TomatoSoupCan')
GEOM = re.compile(r'"(low|high|center|approach|closing|descriptors|vectors)"')
D3 = re.compile(r'"dinov3[^"]*"\s*:\s*\[\s*-?\d')


def main():
    root = Path(sys.argv[1])
    man = json.loads((root/'MANIFEST_INCLUDED.json').read_text())
    findings = []
    listed = {r['path']: r['sha256'] for r in man['files']}
    for path, h in listed.items():
        f = root/path
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != h:
            findings.append(f'hash mismatch or missing: {path}')
        if path.startswith('publication/'):
            findings.append(f'manuscript file included: {path}')
        data = f.read_bytes()
        if path.endswith('.npz'):
            z = np.load(io.BytesIO(data), allow_pickle=True)
            if any('dinov3' in k for k in z.files):
                findings.append(f'DINOv3 array in {path}')
            if 'model' in z.files and 'dinov3' in str(z['model']).lower():
                findings.append(f'DINOv3 model file {path}')
            for k in z.files:
                if k == 'names' or k.endswith('__names'):
                    if any(str(n) in YCB for n in z[k]):
                        findings.append(f'YCB rows in {path}:{k}')
        elif path.endswith(('.json', '.json.gz', '.ipynb')):
            t = (gzip.decompress(data) if path.endswith('.gz') else data).decode('utf-8', 'replace')
            if D3.search(t):
                findings.append(f'DINOv3 numeric field in {path}')
            if any(f'"{y}"' in t for y in YCB) and GEOM.search(t):
                findings.append(f'YCB name together with geometry/descriptor fields in {path}')
        elif path.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if any('dinov3' in n.lower() for n in zf.namelist()):
                    findings.append(f'DINOv3 member in {path}')
    extra = [str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and str(p.relative_to(root)) not in listed
             and p.name not in ('MANIFEST_INCLUDED.json', 'MANIFEST_EXCLUDED.json', 'PACKAGE_README.md')]
    if extra:
        findings.append(f'unlisted files: {extra[:5]}')
    print(json.dumps(dict(files=len(listed), findings=findings), indent=1))
    sys.exit(1 if findings else 0)


if __name__ == '__main__':
    main()
