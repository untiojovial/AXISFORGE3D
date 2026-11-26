"""Automated licence audit of the software dependencies (Rust crates in Cargo.lock, Python packages installed).

    python tools/licenses/audit_dependencies.py --out research/phase7_extension/results/dependency_licences.json

Rust: `cargo metadata --locked` (every third-party crate resolved for the workspace, with the licence
expression its publisher declared). Python: the metadata of the installed distributions (License-Expression,
License, or the classifiers) for the packages the project imports. The script reports what the packages
declare; it does not interpret licence compatibility. Packages whose declared licence is missing, or not in
the permissive allow-list below, are listed for review. None of these dependencies is redistributed in the
repository (Rust crates are fetched by cargo, Python packages by pip); a binary release would have to
include their notices (see THIRD_PARTY_NOTICES.md).
"""
import argparse
import importlib.metadata as md
import json
import re
import subprocess
import sys
from pathlib import Path

PERMISSIVE = {'MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 'ISC', 'Zlib', 'Unlicense', 'Unicode-3.0', 'Unicode-DFS-2016',
              'PSF-2.0', 'Python-2.0', 'MPL-2.0', 'BSL-1.0', 'CC0-1.0', 'Apache-2.0 WITH LLVM-exception'}
PYTHON = ('numpy', 'scipy', 'scikit-learn', 'torch', 'torchvision', 'open_clip_torch', 'timm', 'transformers', 'safetensors',
          'mujoco', 'trimesh', 'coacd', 'matplotlib', 'gpytorch', 'botorch', 'optuna', 'pytest', 'huggingface_hub', 'pillow')


def terms(expr):
    return {t for t in re.split(r'\s+(?:OR|AND)\s+|[()/]', expr or '') if t.strip()}


def rust():
    meta = json.loads(subprocess.run(['cargo', 'metadata', '--format-version', '1', '--locked'], capture_output=True, text=True,
                                     check=True).stdout)
    ws = set(meta['workspace_members'])
    rows = []
    for p in meta['packages']:
        if p['id'] in ws:
            continue
        lic = p.get('license')
        rows.append(dict(name=p['name'], version=p['version'], license=lic, source=p.get('source'),
                         review=not lic or not terms(lic) <= PERMISSIVE))
    return sorted(rows, key=lambda r: r['name'])


def python():
    rows = []
    for name in PYTHON:
        try:
            d = md.distribution(name)
        except md.PackageNotFoundError:
            rows.append(dict(name=name, installed=False)); continue
        m = d.metadata
        lic = m.get('License-Expression') or (m.get('License') or '').split('\n')[0][:120] or None
        cls = [c.split('::')[-1].strip() for c in m.get_all('Classifier') or [] if c.startswith('License ::')]
        rows.append(dict(name=name, version=d.version, license=lic, classifiers=cls,
                         review=not (lic or cls)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r, py = rust(), python()
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), python_executable=sys.executable,
               rust=dict(crates=len(r), needs_review=[x for x in r if x['review']], all=r),
               python=dict(packages=py, needs_review=[x for x in py if x.get('review')]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print('rust crates', len(r), 'needing review', [(x['name'], x['license']) for x in r if x['review']])
    for x in py:
        print(x['name'], x.get('version'), x.get('license'), x.get('classifiers'))


if __name__ == '__main__':
    main()
