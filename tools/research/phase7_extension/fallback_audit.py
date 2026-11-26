"""Audit of the round-robin fallback bound on the native implementation's real decisions (section 15).

    python tools/research/phase7_extension/fallback_audit.py --validation research/phase7_extension/results/native_inference_validation.json \
        --dev-catalogues cat80/ --egad-catalogues CONF/cat80 --out research/phase7_extension/results/fallback_audit.json

Three levels:

1. **Theory.** The bound N_hybrid <= min(2 N_L - 1, 2 N_D) <= 2 min(N_L, N_D) for the round robin L1, D1, L2, D2, ...
   that skips repeats. A candidate is feasible or not regardless of the order, so the hybrid stops no later than
   L's first hit, at step 2 N_L - 1, and no later than D's, at step 2 N_D.
2. **Tests.**
   - exhaustive over orders of 5 candidates (crates/grasping/src/learned.rs);
   - a 2000-catalogue randomized property test of the native order;
   - 3000 random catalogues of the reference interleave (tests/phase7_extension).
3. **Implementation.** On every viable object of the native equivalence run (151 development parts, 255 EGAD
   objects), take the native planner's N_F for joint and hybrid (plan --learned --order ...). Take the original
   planner's N_F from the catalogue order, which is the planner's own order. Then check the bound.

The development numbers use the frozen models, which were trained on those same parts. They are in-sample and
therefore optimistic in N_F; the bound check does not depend on this.

The bound is about **examined candidates** only. It says nothing about time: the learned stage adds inference,
and candidates differ in cost. It says nothing about the robustness or safety of the grasp found. Timing is
measured separately (integrated_runtime_benchmark.json) with no temporal guarantee.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import catalogue  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('validation', 'dev-catalogues', 'egad-catalogues', 'out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    val = json.loads(Path(a.validation).read_text())['level2']
    res, rows = {}, {}
    for n, r in val.items():
        kind = r['kind']
        cat = catalogue.load(Path(a.dev_catalogues if kind == 'development' else a.egad_catalogues)/f'{n}.npz')
        o = catalogue.order(cat)
        n0 = int(np.flatnonzero(cat['feasible'][o])[0]) + 1
        nj, nh = r['modes']['joint']['native'][0], r['modes']['hybrid']['native'][0]
        rows[n] = dict(kind=kind, M0=n0, joint=nj, hybrid=nh, bound=2*min(nj, n0), tight_bound=min(2*nj - 1, 2*n0),
                       violation=nh > 2*min(nj, n0), tight_violation=nh > min(2*nj - 1, 2*n0))
    for kind in ('development', 'egad'):
        rs = [v for v in rows.values() if v['kind'] == kind]
        ratio = np.array([v['hybrid']/v['M0'] for v in rs])
        res[kind] = dict(objects=len(rs), violations=int(sum(v['violation'] for v in rs)),
                         tight_violations=int(sum(v['tight_violation'] for v in rs)),
                         joint_worse_than_M0=int(sum(v['joint'] > v['M0'] for v in rs)),
                         hybrid_worse_than_M0=int(sum(v['hybrid'] > v['M0'] for v in rs)),
                         max_hybrid_over_M0=float(ratio.max()), max_joint_over_M0=float(max(v['joint']/v['M0'] for v in rs)),
                         gmean=dict(M0=float(np.exp(np.mean(np.log([v['M0'] for v in rs])))),
                                    joint=float(np.exp(np.mean(np.log([v['joint'] for v in rs])))),
                                    hybrid=float(np.exp(np.mean(np.log([v['hybrid'] for v in rs]))))))
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), summary=res, per_object=rows)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
