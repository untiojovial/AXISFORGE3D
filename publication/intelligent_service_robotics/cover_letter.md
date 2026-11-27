# Cover letter (draft for the author)

**Do not send without the author's review.** Two fields can only be filled at submission: the date and the commit of
the submitted version. The author has to confirm the exclusive-submission sentence when submitting.

---

[Date of submission]

To the Editor-in-Chief, *Intelligent Service Robotics*

Dear Editor,

Please consider the manuscript *"Data-Efficient Grasp Candidate Ordering with Learned Feasibility Priors and Bounded
Fallback"* for publication as an original research article, through the subscription route.

**Question.** Service robots that pick unfamiliar objects must plan grasps quickly, and the grasps must hold. A
learned feasibility prior can make a sampling-based planner examine far fewer candidates. The order, however, also
decides which grasp is executed. The manuscript asks three things:

- whether the efficiency gain holds on objects not used for design;
- whether it comes at the expense of dynamic robustness, including robustness across friction values;
- what it costs in end-to-end time, with the learned inference included.

The related-work section separates what was known (learned feasibility orders, robustness-aware ranking, fallbacks
with guarantees) from what this study adds.

**Design.**

- All experiments are in simulation.
- The study uses two confirmations on EGAD objects. In each one, the method, the hypotheses and the analysis were
  frozen in Git, with SHA-256 digests, before any confirmatory object was planned.
- The second confirmation tests a variant selected on development data after the first confirmation.
- The learned models are evaluated bit-exactly inside the planner.

**Main findings.**

1. **Fewer examined candidates.** The learned orders examined 0.61 and 0.45 times as many candidates as the original
   planner, in the first and the second confirmation. In both, the robustness-aware variant was non-inferior in
   examined candidates to the plain learned order.
2. **Robustness.** It was non-inferior to the original planner in the second confirmation. In the first, the result
   was fragile: it depends on the statistical procedure. The robustness gains seen in development did not replicate,
   including the one for which the second variant was selected.
3. **Planning time.** It was not reduced: the time ratio was 1.00 [0.98, 1.02]. A round-robin fallback, an instance
   of a known parallel-execution principle, bounds the number of examined candidates at min(2N_L − 1, 2N_D), and the
   bound was never exceeded.
4. **GPD.** GPD's pretrained classifier, used to rank the same candidates, was a poor examination order for this
   planner.

**Why it fits the journal.** We report mixed and negative results deliberately, because we believe they are useful to
practitioners who add learned components to classical grasp planners. Code, protocols, frozen models and per-object
results are in the public repository https://github.com/untiojovial/Colegio-Rumareda. The submitted version is commit
[commit at submission].

**Relation to the author's master's thesis.**

- **The thesis.** E. Pérez González, *Robot grasping with Bayesian optimization informed with foundational models*,
  Universidad de Zaragoza, 2025, TAZ-TFM-2025-047, https://zaguan.unizar.es/record/152263.
- **What it did.** It studied sequential Bayesian optimisation of grasps for a humanoid hand in simulation, informed
  by DINO features and context.
- **How this study differs.**
  - It is independent research carried out afterwards.
  - It addresses a different problem: a fixed learned examination order for a complete parallel-jaw planner,
    evaluated prospectively.
  - It reuses no text, figure or result of the thesis.
  - It is not affiliated with the Universidad de Zaragoza.
  - The thesis supervisor did not take part directly in it.
- **Status.** The thesis is a degree work deposited in the university repository, not a peer-reviewed publication.

**Public drafts.** The repository is public, so draft versions of the manuscript have been publicly visible since 26
November 2026. For about 40 minutes on that day, the repository's README contained wording that could be read as an
Apache-2.0 grant covering the manuscript (the commit remains in the public history); the manuscript has since been
explicitly excluded from that licence. We would be grateful for
the editor's guidance on whether this needs any action under the journal's preprint policy.

**Use of AI.**

- **Claude Code (Anthropic).** Used substantially, under the author's direction and supervision, to implement the
  software, run the simulations and experiments, assist with the statistical analysis and draft the text.
- **ChatGPT (OpenAI).** Used for methodological, literature, editorial and licence consultation and for critical
  review.
- **Who decided.** The author directed the research, made the final scientific decisions, reviewed the work and takes
  full responsibility for it.
- **Authorship.** No AI system is an author. The use is described in the Methods and in the declarations.

**Exclusive submission.** The manuscript has not been published and is not under consideration elsewhere. [Author to
confirm at submission.]

Suggested reviewers: none.

Sincerely,

Eduardo Pérez González
Independent Researcher, Zaragoza, Spain
eduperez2000g@gmail.com · ORCID https://orcid.org/0009-0007-0303-8839
