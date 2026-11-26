"""Phase VII extension: statistics reproduce from committed files, exported models equal scikit-learn,
licence texts and the third-party manifest are intact, and the native fallback order keeps its bound."""
import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT/'tools'/'research'/'phase7_extension'
APACHE_SHA256 = 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'  # official LICENSE-2.0.txt


def tool(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- statistics (from committed results only)
def test_p2_audit_reproduces_the_committed_values(tmp_path):
    out = tmp_path/'p2.json'
    subprocess.run([sys.executable, str(EXT/'p2_audit.py'), '--results', str(ROOT/'research/phase7/results'), '--out', str(out)],
                   check=True, capture_output=True)
    new, old = json.loads(out.read_text()), json.loads((ROOT/'research/phase7_extension/results/p2_statistical_audit.json').read_text())
    assert new['paired_table'] == old['paired_table'] == dict(both_robust=196, joint_only=4, m0_only=8, neither=12)
    for k in ('pre_specified', 'tango', 'newcombe_method10', 'wald_mle'):
        assert new[k] == old[k]
    # the pre-specified test is the one stored in confirmation_results.json
    conf = json.loads((ROOT/'research/phase7/results/confirmation_results.json').read_text())['sets']['primary']['primary']
    p2 = next(v for k, v in conf.items() if k.startswith('P2'))
    assert new['pre_specified']['p_one_sided'] == pytest.approx(p2['p_one_sided'], abs=1e-15)
    assert round(new['pre_specified']['lower_97_5'], 4) == p2['lower_97_5']
    # the score methods do not support non-inferiority, and the stored decision is fragile
    assert not new['tango']['supports_noninferiority_at_0_025'] and not new['newcombe_method10']['lower_bound_above_margin']
    assert not any(v['still_supported'] for v in new['fragility_pre_specified'].values())


def test_tango_and_newcombe_are_self_consistent():
    m = tool(EXT/'p2_audit.py', 'p2_audit')
    for b, c, n in ((4, 8, 220), (10, 3, 150), (0, 5, 60), (7, 7, 100)):
        lo, hi = m.tango_interval(b, c, n)
        assert lo < (b - c)/n < hi
        assert m.tango_z(b, c, n, lo) == pytest.approx(1.959963984540054, abs=1e-6)
        assert m.tango_z(b, c, n, hi) == pytest.approx(-1.959963984540054, abs=1e-6)
        a, d = n - b - c - 5, 5
        nlo, nhi = m.newcombe10(a, b, c, d)
        assert nlo < (b - c)/n < nhi and -1 <= nlo and nhi <= 1
    # Wilson interval against its closed form at a textbook value (x = 0 of n = 10: upper z^2/(n+z^2))
    lo, hi = m.wilson(0, 10, 1.959963984540054)
    assert lo == pytest.approx(0, abs=1e-12) and hi == pytest.approx(1.959963984540054**2/(10 + 1.959963984540054**2), rel=1e-12)


def test_corrections_addendum_reproduces(tmp_path):
    out = tmp_path/'a.json'
    subprocess.run([sys.executable, str(EXT/'corrections_addendum.py'), '--results', str(ROOT/'research/phase7/results'), '--out', str(out)],
                   check=True, capture_output=True, cwd=ROOT)
    new = json.loads(out.read_text())
    old = json.loads((ROOT/'research/phase7_extension/results/corrections_addendum.json').read_text())
    assert new == old
    assert new['A_S6_in_evaluation_set']['identical'] and new['A_S6_in_evaluation_set']['timed_objects_in_evaluation_set'] == 0
    f = new['F_full_precision_holm_primary']
    assert all(v['holm_p_full_precision'] > 0 for v in f.values())
    assert new['E_denominators']['primary viable objects (P1-P3, S1-S5, S7; rates conditional on a feasible grasp)'] == 220


# ---------------------------------------------------------------- native model export
def test_exported_trees_equal_scikit_learn_bit_for_bit():
    """The learned-order JSON loaded by the Rust planner reproduces predict_proba of the frozen models."""
    doc = json.loads((ROOT/'research/phase7_extension/native/learned_order_phase7.json').read_text())
    freeze = json.loads((ROOT/'research/phase7/PHASE7_MODEL_FREEZE.json').read_text())['model_files_sha256']
    rng = np.random.default_rng(0)
    for part, fname in (('feasibility', 'p_feasible.pkl'), ('robustness', 'p_robust_given_feasible.pkl')):
        path = ROOT/'research/phase7/frozen'/fname
        assert sha(path) == freeze[fname]
        model = pickle.loads(path.read_bytes())  # frozen, hash-checked project file
        X = rng.normal(size=(400, 19))
        X[::7, 17] = np.nan  # missing values follow the recorded side
        m = doc[part]
        raw = np.full(len(X), np.frombuffer(np.uint64(m['baseline']).tobytes(), np.float64)[0])
        for t in m['trees']:
            thr = np.array(t['threshold'], np.uint64).view(np.float64)
            val = np.array(t['value'], np.uint64).view(np.float64)
            for i, x in enumerate(X):
                node = 0
                while not t['leaf'][node]:
                    v = x[t['feature'][node]]
                    left = t['missingLeft'][node] if np.isnan(v) else v <= thr[node]
                    node = t['left'][node] if left else t['right'][node]
                raw[i] += val[node]
        np.testing.assert_allclose(1/(1+np.exp(-raw)), model.predict_proba(X)[:, 1], rtol=0, atol=1e-12)


# ---------------------------------------------------------------- licences and manifest
def test_licence_texts_are_the_official_ones_and_present():
    assert sha(ROOT/'LICENSE') == APACHE_SHA256 == sha(ROOT/'LICENSES/Apache-2.0.txt')
    for f in ('CC-BY-4.0.txt', 'DINOv3-License.md', 'BSD-3-Clause-EGAD.txt', 'BSD-3-Clause-example-robot-data.txt'):
        assert (ROOT/'LICENSES'/f).stat().st_size > 1000
    assert 'Attribution 4.0 International' in (ROOT/'LICENSES/CC-BY-4.0.txt').read_text()
    assert 'DINOv3 License' in (ROOT/'LICENSES/DINOv3-License.md').read_text()
    cargo = (ROOT/'Cargo.toml').read_text()
    assert 'license = "Apache-2.0"' in cargo and 'license = "Apache-2.0"' in (ROOT/'pyproject.toml').read_text()


def test_third_party_manifest_hashes_match_the_committed_files():
    doc = json.loads((ROOT/'research/phase7_extension/third_party_manifest.json').read_text())
    for path, h in doc['licence_texts_sha256'].items():
        assert sha(ROOT/path) == h, path
    names = set()
    for c in doc['components']:
        for k in ('name', 'version', 'source', 'license', 'evidence', 'redistribution', 'attribution', 'status', 'verified'):
            assert c.get(k), (c['name'], k)
        for path, h in c['affected_sha256'].items():
            assert sha(ROOT/path) == h, path
        names.add(c['name'].split(' ')[0])
    notices = (ROOT/'THIRD_PARTY_NOTICES.md').read_text()
    for n in ('YCB', 'Google', 'EGAD', 'DINOv3', 'DINOv2', 'OpenCLIP', 'GPD', 'UR5', 'three.js'):
        assert n in notices


def test_dinov3_material_is_never_offered_under_apache():
    reg = (ROOT/'research/phase7_extension/ARTIFACT_REDISTRIBUTION_REGISTER.md').read_text()
    line = next(l for l in reg.splitlines() if l.startswith('| Embeddings and values derived from DINOv3'))
    assert 'DINOv3 License' in line and 'never Apache-2.0' in line


# ---------------------------------------------------------------- fallback bound (property test of the reference interleave)
def test_round_robin_bound_holds_on_random_catalogues():
    common = tool(EXT/'common.py', 'ext_common')
    rng = np.random.default_rng(20261211)
    for _ in range(3000):
        n = int(rng.integers(1, 60))
        feas = rng.random(n) < rng.uniform(.01, .5)
        if not feas.any():
            feas[rng.integers(n)] = True
        cat = dict(feasible=feas, label=np.array([str(i) for i in range(n)]))
        a, b = rng.permutation(n), rng.permutation(n)
        h = common.interleave(cat, a, b)
        assert sorted(h.tolist()) == list(range(n))  # a permutation: nothing skipped, nothing repeated
        na, nb, nh = common.first(cat, a)[0], common.first(cat, b)[0], common.first(cat, h)[0]
        assert nh <= 2*min(na, nb)
        assert nh <= min(2*na - 1, 2*nb)
