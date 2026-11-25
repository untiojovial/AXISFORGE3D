"""Flag test parts that may be variants of a library part (fixed before solving).

    python tools/research/phase5/near_duplicates.py --store store.json --library-names lib.txt \
        --test-names test.txt --out near_duplicates.json

Two tiers, each against every library part:
  variant    the Jaccard index of the lower-case name words is at least 0.5
             (the same product line: colours, sizes, editions)
  lookalike  the D2 cosine is at least 0.99 and the DINOv3 cosine at least 0.90
             (similar shape and appearance; GSO has many similar boxes and bottles)
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

NAME_JACCARD = .5
D2_COSINE = .99
DINO_COSINE = .90


def words(name):
    return {w for w in re.split(r'[^a-z0-9]+', name.lower()) if len(w) > 1}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--store', required=True); p.add_argument('--library-names', required=True)
    p.add_argument('--test-names', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    store = json.loads(Path(a.store).read_text())
    lib = Path(a.library_names).read_text().split(); test = Path(a.test_names).read_text().split()
    vec = lambda n, m: np.asarray(store[n]['descriptors'][m], float)
    flagged = []
    for t in test:
        hits = []
        for l in lib:
            jac = len(words(t) & words(l))/max(len(words(t) | words(l)), 1)
            d2 = float(vec(t, 'd2')@vec(l, 'd2')); dino = float(vec(t, 'dinov3')@vec(l, 'dinov3'))
            if jac >= NAME_JACCARD or (d2 >= D2_COSINE and dino >= DINO_COSINE):
                hits.append(dict(library=l, name_jaccard=round(jac, 3), d2_cosine=round(d2, 4), dinov3_cosine=round(dino, 4)))
        if hits:
            tiers = sorted({'variant' if h['name_jaccard'] >= NAME_JACCARD else 'lookalike' for h in hits})
            flagged.append(dict(test=t, tiers=tiers, matches=hits))
    Path(a.out).write_text(json.dumps(dict(rule=dict(name_jaccard=NAME_JACCARD, d2_cosine=D2_COSINE,
                                                     dinov3_cosine=DINO_COSINE),
                                           flagged=len(flagged),
                                           variants=sorted(f['test'] for f in flagged if 'variant' in f['tiers']),
                                           lookalikes_only=sorted(f['test'] for f in flagged if f['tiers'] == ['lookalike']),
                                           parts=flagged), indent=1) + '\n')
    v = sum('variant' in f['tiers'] for f in flagged)
    print(len(flagged), 'of', len(test), 'test parts flagged:', v, 'product variants,', len(flagged)-v, 'look-alikes only')


if __name__ == '__main__':
    main()
