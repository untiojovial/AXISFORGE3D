# Data availability (draft)

**Public datasets** (used under the licences their sources declare; not redistributed here):

| Dataset | Reference | Licence | Role | Access |
|---|---|---|---|---|
| EGAD | Morrison et al. 2020 | BSD-3-Clause | Confirmation | Training-set archive sha256 `ba6c6470…`; evaluation-set archive sha256 `f3fff4c7…`. Downloaded from the CSIRO Data Access Portal public resource without access control |
| Google Scanned Objects | Downs et al. 2022 | CC BY 4.0 | Development | Collision models from the dataset's public distribution |
| YCB (16 objects) | — | Not verified in this phase | Development library | — |

**Derived data in the repository** (`research/phase5`–`research/phase7`):

- part lists;
- the frozen EGAD selection order (`egad_selection.json`, with per-mesh hashes);
- protocols and frozen hypotheses;
- per-part development results;
- confirmation results (`research/phase7/results/`);
- frozen model hashes and the gradient-boosting models (`research/phase7/frozen/`).

**Kept local, not committed.** The exhaustive candidate catalogues and the per-trial simulation logs of
the confirmation are large. They are regenerable with the commands in
`research/phase7/reproducibility.md`. They can be archived on request, or in a release once the author
authorises one.

**No human-subject data.**
