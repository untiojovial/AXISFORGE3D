# Cover letter (draft for the author)

**Do not send without the author's review. The author must complete the bracketed fields.**

---

[Date]

To the Editor-in-Chief, *Intelligent Service Robotics*

Dear Editor,

Please consider the manuscript *"Data-Efficient Grasp Candidate Ordering with Learned Feasibility Priors and
Bounded Fallback"* for publication as an original research article, through the subscription route.

**Question.** Service robots that pick unfamiliar objects must plan grasps quickly, and the grasps must hold.
A learned feasibility prior can make a sampling-based planner examine far fewer candidates, but the order
also decides which grasp is executed. The manuscript asks whether the efficiency gain holds on objects not
used for design, whether it comes at the expense of dynamic robustness, and what it costs in time.

**Design.** All experiments are in simulation. The study uses two confirmations on EGAD objects. In each,
the method, the hypotheses and the analysis were frozen in Git, with SHA-256 digests, before any
confirmatory object was planned. The second confirmation tests a variant that was selected on development
data after the first confirmation. The learned models are evaluated bit-exactly inside the planner.

**Main findings.**

1. **Fewer examined candidates.** The learned orders examined 0.61 and 0.45 times as many candidates as the
   original planner, in the first and the second confirmation. In both, the robustness-aware variant was
   non-inferior in examined candidates to the plain learned order.
2. **Robustness.** It was non-inferior to the original planner in the second confirmation. In the first it
   was fragile: the result depends on the statistical procedure. The robustness gains seen in development
   did not replicate, including the one for which the second variant was selected.
3. **Planning time.** It was not reduced: the time ratio was 1.00 [0.98, 1.02]. A round-robin fallback
   bounds the number of examined candidates at min(2N_L − 1, 2N_D), and it was never exceeded.
4. **GPD.** GPD's pretrained classifier, used to rank the same candidates, was a poor examination order for
   this planner.

**Why it fits the journal.** We report mixed and negative results deliberately. We believe they are useful
to practitioners who add learned components to classical grasp planners. Code, protocols, frozen models and
per-object results are available as described in the availability statements.

**Prior work.** The work extends the author's master's thesis (Universidad de Zaragoza, 2025), which used
foundation-model features in Bayesian optimisation of grasps. The thesis is not a peer-reviewed publication.
[Author: confirm the record and how the journal wants it declared.]

**Use of AI.** A large language model coding agent (Claude Code, Anthropic) was used substantially to
implement the software, run the experiments and analyses, and draft the text. ChatGPT (OpenAI) was used for
ideas, design, review, statistics, literature, editorial and licence review. This is described in the
Methods and the declarations. [Author: statement of responsibility.]

**Exclusive submission.** The manuscript has not been published and is not under consideration elsewhere.
[Author: confirm.]

Suggested reviewers: [optional; author to decide].

Sincerely,

Eduardo Pérez González
Independent Researcher, Zaragoza, Spain [confirm affiliation]
[e-mail] · [ORCID]
