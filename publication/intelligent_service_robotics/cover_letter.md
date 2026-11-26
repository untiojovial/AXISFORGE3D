# Cover letter — draft for the author

**Do not send without the author's review. The bracketed fields must be completed by the author.**

---

[Date]

To the Editor-in-Chief, *Intelligent Service Robotics*

Dear Editor,

Please consider our manuscript, *"Learning where to look first in a complete parallel-jaw grasp planner:
a pre-registered evaluation of efficiency, robustness and cost"*, for publication as an original research
article.

**Question.** Service robots that pick unfamiliar objects must plan grasps quickly, and the grasps must
hold. Learned feasibility classifiers can make sampling-based planners examine far fewer candidates.
They also change which grasp is executed. The manuscript asks whether such efficiency gains come at the
expense of dynamic robustness, and whether a simple robustness-aware fusion recovers it.

**Answer.** It answers with an independent, pre-registered confirmation: the method, hypotheses, sample
size and analysis were frozen and signed by hashes before the test objects were planned.

**Main findings.** All are in simulation.

1. The learned order examines 0.61 times as many candidates as the original planner on 300
   independently selected EGAD objects, and the robustness-aware variant costs nothing in efficiency.
2. Robustness was only narrowly non-inferior to the original planner. It depended on uncalibrated
   torsional friction, and the robustness and load-tolerance gains observed in development **did not
   replicate**.
3. The end-to-end cost of the current implementation was higher than the original planner's, because
   candidate generation dominates.
4. Frozen vision foundation-model features (DINOv2, DINOv3, OpenCLIP) gave no measurable gain over
   geometric features against permuted and untrained controls.

**Why it fits the journal.** We report these mixed and negative results deliberately. We believe they
are useful to practitioners who add learned components to classical grasp planners. Code, protocols,
frozen models and per-object results are available as described in the availability statements.

**Prior work.** The work extends the author's master's thesis (Universidad de Zaragoza, 2025), which
explored foundation-model features for grasp optimisation. The thesis is not a peer-reviewed publication.
[Author: confirm whether the journal requires this to be declared, and how.]

**Use of AI.** Generative AI tools (Claude Code and ChatGPT) were used in the preparation of code,
experiments and text. This is described in the AI-usage statement. The author takes full responsibility
for the content.

**Exclusive submission.** The manuscript has not been published and is not under consideration
elsewhere. [Author: confirm.]

Suggested reviewers: [optional; author to decide].

Sincerely,

Eduardo Pérez González
Independent Researcher, Zaragoza, Spain [confirm affiliation]
[e-mail] · [ORCID]
