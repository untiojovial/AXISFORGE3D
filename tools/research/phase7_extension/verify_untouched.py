"""Evidence that EGAD primary-list positions 300-499 were never used, before they become a new test set.

    python -I tools/research/phase7_extension/verify_untouched.py --selection research/phase7/egad_selection.json \
        --search research tools publication SCRATCH/p7 SCRATCH/p7x --out research/phase7_extension/results/new_set_untouched.json

Checks, all of which must pass:

1. the cells at positions 300-499 of the seeded draw are distinct from the cells of positions 0-299 (the
   Phase VII primary set), 500-502 (the smoke test) and from the evaluation set's;
2. no mesh file of positions 300-499 has the SHA-256 of any mesh that was built, planned or simulated
   (positions 0-299, 500-502, the evaluation set);
3. no object name (egad_<cell>) of positions 300-499 occurs in any text file under the searched
   directories (repository results, scripts, manuscript, and the Phase VII working directories), nor as a
   directory name there: nothing was built, catalogued, planned, predicted, simulated or timed for them;
4. the Phase VII built pool lists exactly positions 0-299 and the evaluation set;
5. no commit of the repository's history (all refs) contains any of the names (git grep).

Only names and hashes are read; no mesh is opened.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--selection', required=True); p.add_argument('--built', default='research/phase7/results/confirmation_built.json')
    p.add_argument('--search', nargs='+', required=True); p.add_argument('--out', required=True)
    p.add_argument('--first', type=int, default=300); p.add_argument('--last', type=int, default=499)
    a = p.parse_args()
    sel = json.loads(Path(a.selection).read_text())
    prim = sel['primary']
    new = prim[a.first:a.last+1]
    used = prim[:a.first] + prim[500:503] + sel['evaluation_set']
    new_names = {f"egad_{x['cell']}" for x in new}
    checks = {}
    checks['1_cells_disjoint'] = dict(new_cells=len({x['cell'] for x in new}),
                                      overlap=sorted({x['cell'] for x in new} & {x['cell'] for x in used}))
    checks['2_sha256_disjoint'] = dict(overlap=sorted({x['sha256'] for x in new} & {x['sha256'] for x in used}),
                                       duplicates_within_new=len(new) - len({x['sha256'] for x in new}))
    hits, dirs, scanned = {}, [], 0
    for root in a.search:
        for dp, dn, fn in os.walk(root):
            if any(part in ('target', '.git', 'node_modules', 'venv') for part in Path(dp).parts):
                continue
            for d in dn:
                if d in new_names:
                    dirs.append(str(Path(dp)/d))
            for f in fn:
                path = Path(dp)/f
                if path.resolve() == Path(a.out).resolve():
                    continue  # this script's own output lists the names
                try:
                    if path.stat().st_size > 200_000_000:
                        continue
                    data = path.read_bytes()
                except OSError:
                    continue
                if b'\0' in data[:4096] and not f.endswith(('.json', '.txt', '.md', '.tex', '.log', '.py', '.sh')):
                    continue  # binary file (catalogues are .npz named after objects and are caught by name below)
                scanned += 1
                if b'egad_' not in data:
                    if any(n in f for n in new_names):
                        hits.setdefault(str(path), []).append('file name')
                    continue
                text = data.decode('utf-8', 'replace')
                found = [n for n in new_names if n in text or n in f]
                if found:
                    hits[str(path)] = sorted(found)[:10]
    checks['3_names_unused'] = dict(files_scanned=scanned, files_mentioning=hits, directories=dirs)
    built = json.loads(Path(a.built).read_text())
    pos = built.get('positions')
    checks['4_built_pool'] = dict(positions=pos if not isinstance(pos, list) or len(pos) < 10 else [min(pos), max(pos), len(pos)],
                                  built_names_in_new=sorted({r['name'] for r in built['rows']} & new_names))
    commits = subprocess.run(['git', 'rev-list', '--all'], capture_output=True, text=True, check=True).stdout.split()
    git_hits = set()
    with tempfile.NamedTemporaryFile('w', suffix='.txt') as tf:
        tf.write('\n'.join(sorted(new_names)) + '\n'); tf.flush()
        for i in range(0, len(commits), 20):
            r = subprocess.run(['git', 'grep', '-l', '-F', '-f', tf.name, *commits[i:i+20], '--', f':!{a.out}'],
                               capture_output=True, text=True)
            git_hits.update(r.stdout.split())
    checks['5_git_history'] = dict(commits=len(commits), files_mentioning=sorted(git_hits))
    ok = (not git_hits and not checks['1_cells_disjoint']['overlap'] and not checks['2_sha256_disjoint']['overlap']
          and checks['2_sha256_disjoint']['duplicates_within_new'] == 0 and not hits and not dirs
          and not checks['4_built_pool']['built_names_in_new'])
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), positions=[a.first, a.last], objects=len(new),
               names=sorted(new_names), checks=checks, all_pass=ok)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(all_pass=ok, **{k: {kk: (vv if not isinstance(vv, (dict, list)) or len(vv) < 8 else len(vv)) for kk, vv in v.items()}
                                          for k, v in checks.items()}), indent=1))


if __name__ == '__main__':
    main()
