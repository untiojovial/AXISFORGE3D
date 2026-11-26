"""Verify the Phase VII freeze record against its own commit and against the current tree (read-only).

    python tools/research/phase7_extension/verify_freeze.py --commit de22a8a --out research/phase7_extension/results/freeze_verification.json

For every file whose SHA-256 is recorded in research/phase7/PHASE7_MODEL_FREEZE.json (code, models in the
repository, the protocol part II and the EGAD selection), the script compares the recorded hash with the
file's bytes at the freeze commit (git show) and in the working tree. A file that differs now but matched at
the freeze is listed as changed after the freeze, with the commits that touched it, so later edits are visible.
Models kept outside the repository (DINOv3-input MLPs) are checked in --external-models if given.
Historical hashes are not modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def git(*a):
    return subprocess.run(['git', *a], capture_output=True, cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--commit', required=True); p.add_argument('--out', required=True)
    p.add_argument('--freeze', default='research/phase7/PHASE7_MODEL_FREEZE.json')
    p.add_argument('--external-models', help='directory holding the frozen models kept outside the repository')
    a = p.parse_args()
    fr = json.loads(git('show', f'{a.commit}:{a.freeze}').stdout)
    recorded = dict(fr['code_sha256'])
    outside = set(fr.get('model_files_not_in_repository', {}))
    recorded.update({f'research/phase7/frozen/{k}': v for k, v in fr['model_files_sha256'].items() if k not in outside})
    recorded[fr['protocol']['file']] = fr['protocol']['sha256']
    recorded[fr['test_manifest']['selection_file']] = fr['test_manifest']['selection_sha256']
    rows = []
    for path, h in sorted(recorded.items()):
        at = git('show', f'{a.commit}:{path}')
        h_at = hashlib.sha256(at.stdout).hexdigest() if at.returncode == 0 else None
        f = ROOT/path
        h_now = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
        touched = git('log', '--format=%h %s', f'{a.commit}..HEAD', '--', path).stdout.decode().splitlines() if h_now != h else []
        rows.append(dict(file=path, recorded=h, matches_at_freeze_commit=h_at == h, matches_now=h_now == h, commits_after_freeze=touched))
    ext = []
    for k in sorted(outside):
        f = Path(a.external_models)/k if a.external_models else None
        h_now = hashlib.sha256(f.read_bytes()).hexdigest() if f and f.exists() else None
        ext.append(dict(file=k, recorded=fr['model_files_sha256'][k], present=h_now is not None, matches=h_now == fr['model_files_sha256'][k]))
    out = dict(external_models=ext, all_external_match=all(e['matches'] for e in ext), note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), freeze_commit=git('rev-parse', a.commit).stdout.decode().strip(),
               freeze_record_sha256_at_commit=hashlib.sha256(git('show', f'{a.commit}:{a.freeze}').stdout).hexdigest(),
               freeze_record_unchanged_now=hashlib.sha256((ROOT/a.freeze).read_bytes()).hexdigest() ==
               hashlib.sha256(git('show', f'{a.commit}:{a.freeze}').stdout).hexdigest(),
               files=len(rows), all_match_at_freeze_commit=all(r['matches_at_freeze_commit'] for r in rows),
               changed_after_freeze=[r for r in rows if not r['matches_now']], rows=rows)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('rows', 'note')}, indent=1))


if __name__ == '__main__':
    main()
