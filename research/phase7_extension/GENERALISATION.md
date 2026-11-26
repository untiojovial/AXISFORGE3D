# Generalisation: what was tested, under which regime (section 14)

The table classifies every generalisation result produced so far. Its terms are used as follows:

- **zero-shot**: the frozen model is applied with no data from the target;
- **retraining**: a model of the same form is trained on the target condition;
- **in-domain / OOD**: whether the objects come from the same population as the development parts (Google
  Scanned Objects household scans) or not;
- **same robot**: every result uses the same six-axis arm. Gripper strokes are variants of one parallel-jaw
  gripper, not different robots.

| Result | Regime | Objects | Robot / gripper | N_F (geometric mean) | Source | Evidence level |
|---|---|---|---|---|---|---|
| P(F) on its own development parts | in-domain, out of fold | 151 GSO dynamic parts | same arm, 80 mm | M1 1.25 vs M0 6.07 | `research/phase7/results/cross_dataset_results.json` | development |
| Frozen P(F) on EGAD, primary set | **zero-shot, OOD** (evolved shapes, scaled to the gripper) | 220 viable of 300 | same arm, 80 mm | M1 1.28 vs M0 2.09; Joint/M0 ratio 0.61 (0.52–0.71) | Phase VII confirmation (P1) | confirmatory |
| Frozen P(F) on the EGAD evaluation set | zero-shot, OOD | 35 viable of 45 | same arm, 80 mm | M1 1.29 vs M0 2.89 | `cross_dataset_results.json` | secondary, descriptive |
| 80 mm model applied to a 40 mm stroke | **zero-shot to a stroke variant** (not another robot) | 79 viable of 316 development parts | same arm, 40 mm stroke | zero-shot 1.98; retrained 2.11; default 9.39; zero-shot/retrained 0.94 (0.74–1.19) | `cross_robot_results.json` | development, grouped folds |
| 80 mm model applied to a 120 mm stroke | zero-shot to a stroke variant | 254 viable of 316 | same arm, 120 mm stroke | zero-shot 1.30; retrained 1.22; default 4.97; zero-shot/retrained 1.07 (0.99–1.16) | `cross_robot_results.json` | development, grouped folds |
| Robustness model P(H\|F) and Joint on EGAD | zero-shot, OOD | 220 | same arm, 80 mm | robustness non-inferior only narrowly and only at the pre-specified torsion (P2); load tolerance −6.4 pp vs M0 | Phase VII confirmation | confirmatory (P2), secondary (S5) |
| New EGAD positions 300–499 (second confirmation) | zero-shot, same family as the EGAD primary set (**not a new domain**) | 151 viable of 200 | same arm, 80 mm | X 1.17 vs M0 2.60; X/M0 ratio 0.449 (0.356–0.567); X/M1 0.992 (0.930–1.059) | `results/confirmation_x.json` (C1, C3) | confirmatory |
| X's robustness model on the new positions | zero-shot, same family | 151 | same arm, 80 mm | robustness X−M0 +3.3 pp (Tango 0.7–7.5): non-inferior (C2); all-three-torsion robustness vs M1 0.0 pp (S2, not supported; development +25.2 pp) | `results/confirmation_x.json` | confirmatory (C2), secondary (S2, S3) |

## What these results do and do not show

- **Feasibility prior.** It transferred zero-shot from household scans to evolved shapes: confirmatory P1. It
  also transferred to the 40 mm and 120 mm strokes of the same gripper on the development parts. There, the
  zero-shot model was not detectably worse than a model retrained for the stroke; the intervals include
  differences of up to about +16 % or −26 %.
- **Robustness term.** It did not transfer as an advantage. Phase VII: P2 fragile, S5, torsion sensitivity. Second
  confirmation: robustness non-inferior to M0 (C2), but the multi-torsion advantage that selected X was absent (S2).
- **What was not tested.**
  - Other robots: different kinematics, a hand or a suction gripper. The project supports hands, but no
    learned-order experiment used one.
  - Real objects and real sensors; adaptation with support observations (few-shot).
  - Unknown object categories beyond the two datasets.
- **Hard and non-viable objects.**
  - The 80 primary EGAD objects without any viable candidate are part of the population metrics. No ordering
    can change them.
  - Hard objects (large N_F under M0) are where the learned order saves most. They are also where negative
    transfer occurs: Joint needed more candidates than M0 on 14 of the 220 viable objects in Phase VII (`numbers.tex`, `confNegJoint`). The
    round-robin fallback bounds the number of candidates examined.
