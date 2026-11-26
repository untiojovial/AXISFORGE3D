# Articulated model v2: development run (not calibrated)

- **Result:** `results/articulated2_development.json`.
- **Raw data:** one file per part, in the run directory.
- **Code:**
  - `tools/research/phase7_extension/articulated2_dev.py`;
  - `research_grasping/phase7x/articulated2.py`.
- **Population:** development only, the 151 dynamic parts.
- **Grasps:** the first viable grasps of M0, M1 and Joint (Phase VII out-of-fold orders), re-planned natively
  to obtain their joint values.
- **Settings:** torsion 0.005 m, no pose noise.
- **Status:** nothing here is calibrated against a real robot, and nothing here changes a hypothesis of
  either confirmation.

## Controls (M0's first grasp, 151 parts each)

| Control | Expected | Lifted | Held |
|---|---|---|---|
| jaws kept open | must not lift | 0 | 0 |
| frictionless contacts | must slip | 0 | 0 |
| gravity off, part mass 2 kg | "any grasp that closed must hold" | 137 | 118 |

**Negative controls.** Both behave as expected.

**Positive control.** It does **not** meet its written expectation: 33 of the 151 grasps were not held with
gravity off.

**Breakdown of the 33.**

- **19 grasps** are also not held in the nominal condition: the grasp did not hold in the first place.
- **14 grasps** held nominally (0.1 kg, gravity on) but lost the part with gravity off. In that control the
  part has 2 kg of inertia, and the ±20 mm, 3 Hz shake through the base joint accelerates it by up to about
  7 m/s². That is a load of the same order as the 1 kg weight condition.
- **3 grasps** held with gravity off but not nominally.

**Reading.** The expectation was badly specified: a 2 kg part under the shake is not a weightless test.
The failures are consistent with inertial loading, but this run does not prove that. The control is
therefore reported as **not passed as specified**. The articulated v2 model is not validated as a
load-tolerance instrument.

## Rates (share of the 151 first grasps held, pilot timing)

| Condition | M0 | M1 | Joint |
|---|---|---|---|
| nominal | 85.4 % | 90.7 % | 94.0 % |
| mass ×2 | 86.8 % | 91.4 % | 92.7 % |
| mass ×0.5 | 87.4 % | 88.1 % | 93.4 % |
| friction 0.3 | 84.1 % | 89.4 % | 91.4 % |
| load 0.5 kg | 83.4 % | 84.8 % | 90.1 % |
| load 1 kg | 73.5 % | 67.5 % | 76.8 % |
| load 2 kg | 49.0 % | 35.1 % | 31.8 % |

## Joint − M0 load tolerance (held at 0.5, 1 and 2 kg), paired bootstrap 95 % CI

| Timing | Difference | 95 % CI |
|---|---|---|
| pilot | −17.2 pp | [−24.5, −10.6] |
| lift 0.5 s | −15.9 pp | [−22.5, −9.3] |
| lift 2.0 s | −17.9 pp | [−25.2, −11.3] |
| close 0.25 s | −17.2 pp | [−24.5, −10.6] |

**Reading.**

- **Load tolerance.** With the arm, Joint's first grasps are less load tolerant than the original planner's.
  The difference holds at every timing variant. Its direction agrees with the Phase VII articulated pilot and
  with the first confirmation's secondary result (Joint less load tolerant than M0).
- **Standard conditions.** Joint's first grasps are held more often than M0's.
- **Telemetry.**
  - no arm-torque saturation;
  - no unexpected link contacts;
  - no palm-table contacts;
  - the median tracking error is about 0.05 rad.

  The grip-force limit is reached in 13–59 % of trials, depending on the condition.
- **What this does not show.** These are development results with an uncalibrated model whose positive
  control did not pass as specified. They describe the simulator, not a robot.

## Effect on the article and the hypotheses

- It changes no hypothesis. Load tolerance was already outside both primary families.
- It appears in the supplement, labelled as an uncalibrated exploratory analysis.
