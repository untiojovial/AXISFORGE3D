"""Audit the Git history and remote branches for third-party-licensed files committed without their licence copy.

    python tools/licenses/audit_history.py --out research/phase7_extension/results/history_licence_audit.json

It reads only Git metadata: the commits that added DINOv3-derived files, the commit that added
LICENSES/DINOv3-License.md, and, for every remote-tracking branch, whether its head carries DINOv3-derived files and
the licence copy. Nothing is rewritten. The report states the scope of the gap and the remedies that do not rewrite
history; applying a remedy to a branch other than the working branches needs the author's authorisation.
"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LICENCE = 'LICENSES/DINOv3-License.md'


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=ROOT, check=True).stdout


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    added = {}
    for line in git('log', '--all', '--diff-filter=A', '--format=@%h %aI', '--name-only').splitlines():
        if line.startswith('@'):
            commit, date = line[1:].split()
        elif line and 'dinov3' in line.lower() and line != LICENCE:
            added[line] = dict(first_added_in=commit, date=date)  # log is newest first: the last assignment is the oldest
    lic = git('log', '--all', '--diff-filter=A', '--format=%h %aI', '--', LICENCE).split()
    first_data = min(added.values(), key=lambda r: r['date'])
    window = git('rev-list', '--count', f"{first_data['first_added_in']}..{lic[0]}").strip() if lic else None
    branches = {}
    for ref in git('for-each-ref', '--format=%(refname:short)', 'refs/remotes/origin').split():
        if ref.endswith('/HEAD'):
            continue
        files = git('ls-tree', '-r', '--name-only', ref).splitlines()
        d = [f for f in files if 'dinov3' in f.lower() and f != LICENCE]
        branches[ref] = dict(head=git('rev-parse', '--short', ref).strip(), dinov3_files=len(d), licence_copy=LICENCE in files,
                             gap=bool(d) and LICENCE not in files)
    doc = dict(
        note=__doc__,
        dinov3_files=added, licence_copy_added_in=dict(commit=lic[0], date=lic[1]) if lic else None,
        history_window=dict(first_dinov3_file=first_data, licence_copy=lic[0] if lic else None, commits_between=int(window) if window else None,
                            statement='Between these commits DINOv3-derived embeddings were in the history without a copy of the DINOv3 '
                                      'License. The folder README described the licence terms and the citation duty, but did not include the '
                                      'licence text. Those commits stay in the history; excluding the embeddings from the new package does '
                                      'not remove them from Git history or from any copy already made.'),
        branches=branches,
        remedies_without_rewriting=[
            'A notice on the working branches (README and THIRD_PARTY_NOTICES) that every DINOv3-derived file in any commit or branch '
            'of this repository is governed by the DINOv3 License, with the copy at LICENSES/DINOv3-License.md (done).',
            'With the author\'s authorisation: a fast-forward commit on each branch with a gap that adds LICENSES/DINOv3-License.md '
            'and a short notice (no history rewrite, no force push).',
            'Alternatively, with the author\'s authorisation: archive or delete stale branches through the hosting service; this '
            'removes the branch, not the commits from existing clones or forks.'],
        not_done=['history rewrite', 'force push', 'any change to branches other than the two working branches'])
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(dict(window=doc['history_window']['commits_between'], gaps=[b for b, r in branches.items() if r['gap']]), indent=1))


if __name__ == '__main__':
    main()
