# Phase VII — repository audit (stage A)

Date 2026-11-26. Start point `4c3e8af`, the last Phase VI commit.

- **Branches.** No newer commits on the remote. The work happens on `claude/new-session-nsx8rp`, and the
  same history is published as `research/axisforge-phase7-final-publication`. No rewrite, no force push.
- **Untouched.** Phases I–VI, the Phase III sealed objects and the repository licence.

## 1. Tests (run today)

| Suite | Result |
|---|---|
| `cargo test --workspace --release` | 359 passed, 0 failed (27 binaries); 360 with the Phase VII timing test |
| `cargo clippy --workspace --all-targets --release -- -D warnings` | 0 warnings |
| `pytest tests` | 472 passed, **65 skipped** |
| `ruff --select F` on Phase V–VII code | clean |

The 65 skipped tests are **not** counted as passed. They are skipped for these reasons (Phase VI audit,
unchanged):

- 57 need external Phase III/IV development bundles;
- 4 need the sealed Phase III inputs;
- 3 need authorised weights;
- 1 needs the optional `numpyro`.

## 2. Historical figures

[phase6_verification.json](results/phase6_verification.json) recomputes the 15 numbers that the Phase
VII request quotes, directly from the stored result files. **All 15 match**:

- the five Phase V ratios and their intervals;
- M0, M1, M4 and M9 robustness and load tolerance;
- M4 − M1 load tolerance (+11.9 [6.6, 17.9]), and M4 not shown superior to M0;
- M9 − M1 robustness (+5.3 [2.0, 9.3]);
- the time ratio per planner call (0.76);
- random within the gate (1.18 [0.91, 1.52]);
- the torsion sensitivity (17 and 20 of 180 verdicts).

## 3. Implementation review of Phase VI

These points are not in the Phase VI summary, or are only implicit there:

1. **Rotation was not part of success.** Phase VI's v2 measured the rotation after the lift but did not
   use it. Fixed in v3 (§4, error 4).
2. **The Phase VI online cost leaves out building the prior.** It does not count features, model
   inference, rendering or encoding. The 24 % saving is per planner call only. Phase VII instruments the
   planner by stage and times prior construction (§I.5 of the protocol).
3. **N_FR assumes a simulator in the loop.** It models a system that simulates every viable grasp before
   executing it. It is an evaluation metric, not the cost of a deployed planner without a simulator.
4. **M9's robustness labels come from the same 151 development parts it is evaluated on.** The labels
   are out of fold, so this is valid as development evidence. A confirmatory M9 must be trained on
   development data only and tested on new objects.
5. **New finding (Phase VII local features): the planner's contact points are not on the render mesh.**
   `center ± closing·width/2` lies on the planner's own surface model, a median **6.1 mm** away from the
   render mesh (one part, 300 contacts; measured again for every part in the local-feature run). Any
   image-space correspondence must snap contacts to the render surface. The offset is recorded as a
   feature.
6. **The Phase VI encoded prior is exact.** The fusion order passed to the native planner with weight
   5 reproduced the simulator in 60 of 60 parts.

## 4. Error traceability matrix

| # | Error | Where | Effect | Correction | Conclusions affected | Status |
|---|---|---|---|---|---|---|
| 1 | "mass ×2/×0.5" simulated 2 kg and 0.5 kg | Phase V `dynamics.py` | Phase V robustness 6/6 ≈ "holds 2 kg" | Phase VI v2: 0.2/0.05 kg; heavy loads kept as named stress tests | Phase V RQ7 and the response to reviewers; corrected in the Phase VI erratum | Corrected (Phase VI) |
| 2 | Simulator centre of mass ≠ planner's (median 6.4 mm, max 51 mm) | Phase V `dynamics.py` | Wrong lever arm in simulation | v2/v3 use the planner's centre of mass. It is still the **geometric centroid** (uniform density assumed); the real mass distribution is unknown | Phase V dynamics | Corrected (Phase VI) |
| 3 | Random baseline not inside the stroke gate | Phase V `random-order` | "Default is 3× better than random" | Random within the gate: 1.18 [0.91, 1.52] | Phase V response, objection 4 | Corrected (Phase VI) |
| 4 | Success ignored in-hand rotation (66° counted as held) | Phase V/VI criterion | Over-counts holding | v3: rotation from the grip ≤ 15° and contact kept (fixed a priori) | Phase VI R, N_FR and load tolerance are under the old criterion; Phase VII reports both | Corrected in Phase VII (v3) |
| 5 | Torsional friction uncalibrated (0.005 m) | All dynamics | Heavy-load outcomes change with it (17–20 of 180 verdicts) | Phase VII sensitivity grid 0.001–0.02 m; no hardware calibration | Every load-tolerance claim | Sensitivity in Phase VII; calibration blocked (no hardware) |
| 6 | Contact points ≠ render surface (median 6 mm) | Planner surface model vs render mesh | Naive image projection misses most contacts (19 % visible) | Snap to the render surface (89 % visible) | Only the Phase VII local-feature study | Corrected in Phase VII |

## 5. Licences (as declared by the sources; verify before release)

| Item | Licence (source) | Redistributed in the repository? | Note |
|---|---|---|---|
| AxisForge code | Apache-2.0 (declared in `Cargo.toml` and `pyproject.toml`) | Yes | **No `LICENSE` text at the repository root.** Adding one is the author's decision |
| PyTorch | BSD-3-Clause (package metadata) | No | |
| timm | Apache-2.0 | No | |
| open_clip_torch (code) | MIT | No | The licence of the LAION-400M ViT-B/32 weights is not verified here |
| transformers | Apache-2.0 | No | |
| MuJoCo 3.3.7 | Apache-2.0 per the project (not in the wheel metadata) | No | |
| scikit-learn, NumPy, SciPy | BSD-style (package metadata) | No | |
| trimesh | MIT | No | |
| CoACD 1.0.14 | MIT | No | Used to decompose EGAD meshes for MuJoCo |
| gpytorch | MIT | No | |
| DINOv2 ViT-S/14 | Apache-2.0 (model card) | No | |
| DINOv3 ViT-S/16 (timm copy) | **DINOv3 License** (Meta); accepted by the user in Phase IV | Weights no. **Phase V committed DINOv3 descriptors** (`research/phase5/results/descriptors.npz`) | Whether embeddings count as "derivative works" under §1.b of the licence is a legal question for the author before release |
| Google Scanned Objects | CC-BY 4.0 (Downs et al. 2022) | Names and hashes only | Attribution required |
| YCB models | Not verified in this phase | Names only | Verify before release |
| EGAD | BSD-3 (QUT record and GitHub `LICENSE`) | Names, hashes and derived statistics only | Copyright: D. Morrison, ACRV/QUT 2020 |
| UR5/robot models | Not used in the Phase VII experiments | — | |

## 6. End-of-phase verification (after the confirmation)

| Suite | Result |
|---|---|
| `cargo test --workspace --release --no-fail-fast` | **360 passed, 0 failed** (27 binaries) after the fix below |
| `cargo clippy --workspace --all-targets --release -- -D warnings` | 0 warnings |
| `cargo fmt --check` | clean |
| `pytest tests` | **500 passed, 65 skipped** (472 earlier + 28 new Phase VII tests). The 65 skipped tests are not counted as passed; the reasons are as in §1 |

**Regression found and fixed.**

- **What failed.** The first end-of-phase run failed `planner::tests::planning_is_deterministic`.
- **Cause.** Commit `925a39e` added wall-clock stage timings (`PlanTimings`) to `PlanReport`. The test
  compares two whole reports, so it compared timings too. The plans themselves were identical.
- **Fix.** The test now clears the timings before comparing. Every other field must still repeat
  exactly. No planner behaviour changed, and no result depends on it.
- **Correction to §1.** The count "360 with the Phase VII timing test" in §1 was not re-verified after
  that commit. It is superseded by this section.

**Effect on the freeze record.** The fix edits `crates/grasping/src/planner.rs`, one of the files hashed in
`PHASE7_MODEL_FREEZE.json`, so its current hash no longer matches the record.

- The change is confined to the `#[cfg(test)]` module and is not compiled into the planner binary.
- It was made **after** the confirmatory analysis. The analysis checked every frozen code hash before
  running and found no differences.
- The frozen version is the one at commit `5a821dc` (`git show 5a821dc:crates/grasping/src/planner.rs`).
