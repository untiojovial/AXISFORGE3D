# Phase VII — hypotheses

The request's starting hypotheses H1–H5, how each was treated, and the confirmatory families. These
were fixed in [PHASE7_PROTOCOL.md](PHASE7_PROTOCOL.md) part II before the test was opened. The
development evidence quoted comes from the 151 GSO development parts (dynamics) and the 316 development
parts (feasibility). None of it is confirmatory.

## Starting hypotheses

| Id | Statement (request) | Physical or operational definition used | Development evidence | Role in the confirmation |
|---|---|---|---|---|
| H1 | M4 improves load tolerance over M1 | Load tolerance = held at 0.5, 1 and 2 kg. **v3** criterion (historically v2); torsion 0.005 m | v3: +9.9 pp [4.6, 15.9]. Articulated pilot: −3.3 [−10.6, 4.0] | **S1** (secondary, historical); v2 also reported |
| H2 | M9 improves standard robustness over M1 | Robust = held in the six standard conditions; v3 (historically v2) | v3: +4.0 [−0.7, 8.6]; power 0.43 at 200 viable objects | **S2** (secondary, historical, underpowered) |
| H3 | The final method keeps operational performance non-inferior to the best learned baseline under a predefined margin | N_F(Joint)/N_F(M1), margin 1.25 (examined candidates as the planner-time unit) | Ratio 1.03; power 0.997 at 200 | **P3 (primary)** |
| H4 | The final method reduces the full cost relative to the original planner | T_total per object, receiving the object to the planner's answer, one thread | Smoke test (3 excluded objects): Joint 1.0–1.7 s vs M0 0.5–0.8 s, i.e. probably **not** with the current two-process implementation | **S6** (secondary) |
| H5 | The best DINO + MLP representation adds value over geometry + MLP | N_F of L10 (linear DINOv3 branch) vs L2, plus its permuted control L10p | L8 vs L2 1.014 [0.969, 1.059]; L8 vs permuted 0.999. L10 is the best DINO variant. No visual variant met the development rule | **S7** (secondary); H5 also needs L10 < L10p |

## Primary family

Holm across three tests, one-sided familywise α 0.025, on the primary EGAD set.

| Id | Hypothesis | Margin / direction | Development estimate |
|---|---|---|---|
| P1 | FORTIS-R Joint examines fewer candidates than the original planner | N_F ratio < 1 (CI upper limit < 1) | 0.213 |
| P2 | Joint's robustness (v3) is non-inferior to the original planner's | −5 pp | +9.3 pp [2.7, 15.9] |
| P2-sup | Joint is more robust than the original planner | Tested only if P2 holds, at 0.025/3 | Same; power 0.77 at 200 |
| P3 = H3 | Joint's N_F is non-inferior to M1's | Ratio < 1.25 | 1.03 |

## Secondary family

Holm, one-sided 0.025.

- **S1–S7** as in the table above.
- **S3**: Joint > M1 in load tolerance.
- **S4**: Joint > M1 in robustness.
- **S5**: Joint's load tolerance is non-inferior to M0's, margin 10 pp.

## Why load tolerance is not primary

The articulated development pilot reproduces the floating-gripper verdicts in the standard conditions:
agreement 0.95–0.97, κ 0.74–0.82. It reverses the heavy-load contrasts:

| Contrast | v3 | Articulated |
|---|---|---|
| Joint − M1 load tolerance | +11.3 pp | −3.3 pp |
| Joint − M0 load tolerance | −2.7 pp | −17.2 pp |

Load tolerance depends on the model of the arm and on the uncalibrated torsional friction (18 % of
verdicts change between 0.001 and 0.005 m). It is reported at every level and is never a primary claim.

## Not hypotheses

These are measured and reported, not tested:

- population Success@B;
- the hybrid's worst-case bound;
- torsion and rotation sensitivity;
- the articulated replication;
- the EGAD evaluation set;
- development vs EGAD comparison.

H5 failing would not invalidate the article. The request says so explicitly, and the primary family does
not depend on visual features.
