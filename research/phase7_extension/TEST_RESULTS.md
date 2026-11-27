# Test results (full suites, 2026-11-26)

- **Run.** At commit `6e1d50d` (the code is unchanged up to `633bb27`), in the development container, at low
  priority while the second confirmation was running. The confirmation uses a separate frozen copy of the
  binary; it is not affected.
- **Toolchain.** rustc and cargo 1.97.0; Python 3.12.3 (scratch venv, `reproducibility/requirements-lock.txt`
  for the statistics packages).

| Suite | Command | Result |
|---|---|---|
| Formatting | `cargo fmt --all -- --check` | clean (exit 0) |
| Lints | `cargo clippy --workspace --all-targets --release -- -D warnings` | no warnings (exit 0) |
| Rust tests | `cargo test --workspace --release --no-fail-fast` | **368 passed, 0 failed, 0 ignored** (28 test binaries, including `learned_fixture` and the fallback-bound property tests) |
| Python tests | `pytest tests -q -rs` | **512 passed, 65 skipped, 0 failed**, 1 warning |

**Re-run on 2026-11-27** after the final technical review, which added one test of the table scripts' minus
handling (`tests/phase7/test_phase7.py`) and changed no other code: Python tests **513 passed, 65 skipped, 0 failed**,
1 warning (same skips). The Rust code was not changed, so the Rust suites were not re-run.

The extension's own tests (`tests/phase7_extension`, 12 tests) pass in the clean Docker container as well
(`reproducibility/DOCKER_VALIDATION_RECORD.md`).

## Skipped tests (65): investigated, not counted as passed

Every skip is a test that needs data outside the repository, or an optional package. **No test of Phases V,
VI, VII or the extension was skipped.**

| Count | File | Reason given by the test |
|---|---|---|
| 15 + 1 + 1 | `tests/phase4/test_transfer_integrity_additional.py` | development artifacts not available; functional panel unavailable; completed pilot missing |
| 7 + 1 | `tests/phase4/test_transfer_entrypoints.py` | locked Phase IV development artifacts are not local; Phase III functional run not local |
| 7 | `tests/phase4/test_reach_study.py` | external Phase IV development assets unavailable |
| 4 | `tests/phase4/test_jobs_prepare.py` | external sealed Phase III visual inputs unavailable |
| 4 | `tests/phase3/test_vision_asset_regression.py` | requires the local authorised development bundle |
| 3 | `tests/phase4/test_physics_placement_real.py` | prepared development assets unavailable |
| 3 | `tests/phase3/test_scientific_reports.py` | Phase III development artifacts unavailable |
| 2 + 1 + 1 | `tests/phase4/test_calibration_integration.py` | development (calibration) assets or complete physics unavailable |
| 2 | `tests/phase3/test_development_replay.py` | `AXISFORGE_PHASE3_DATA` not set (offline development bundle) |
| 4 | `tests/phase4/test_candidate_study.py` | external native catalogues, locks or inputs required |
| 1 each | `phase4/test_resolution_real.py`, `test_portable_panel.py`, `test_frozen.py`, `test_bundle_analysis.py`; `phase3/test_visual_entrypoints.py`, `test_prepare_pilot_regressions.py` (2), `test_functional_catalogue.py` | authorised local data, the OpenCLIP checkpoint path, original native data or official weights |
| 1 | `tests/test_research_backends.py` | optional backend `numpyro` not installed (AxisForgeBO; not used by the article) |

**Reading.**

- **The Phase III and IV tests.** They check the integrity of sealed development bundles that are kept outside
  the repository, under the data-sharing terms of the earlier phases. Their results were recorded when those
  phases ran. They cannot be re-run here without those bundles.
- **The `numpyro` test.** It covers an optional probabilistic backend of AxisForgeBO, which the grasp-ordering
  study does not use.
- **What this means for the claims.** None of the skipped tests covers code or data behind a claim of the
  article.

## Warning

One `InputDataWarning` from BoTorch in an AxisForgeBO unit test: a deliberately constant outcome. It is not
related to the article.
