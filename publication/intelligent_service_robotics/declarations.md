# Statements and Declarations

The journal requires a "Statements and Declarations" section. Its requirements were verified by ChatGPT on
2026-11-26 and relayed by the author (`JOURNAL_REQUIREMENTS_VERIFIED.md`). The author confirmed the texts below on
27 November 2026 (UTC). The same texts are in `main.tex`, shortened where noted.

- **Funding.** "The author received no financial support for the research, authorship, and publication of this
  article."
- **Competing interests.** "The author declares no competing interests."
- **Ethics approval.** "Not applicable. This study involved computational experiments and robotic simulations using
  publicly available object datasets, without human or animal participants."
  - Checked against the final pipeline on 2026-11-27. The datasets are EGAD, Google Scanned Objects and 16 YCB
    objects: meshes of objects only.
  - The simulations are MuJoCo floating-gripper and articulated-arm models.
  - The images are renders of those meshes.
  - The repository holds no data about people, and no human or animal participated.
- **Consent to participate.** "Not applicable."
- **Consent for publication.** "Not applicable."
- **Materials availability.** "Not applicable."
- **Data availability.** See `data_availability.md`; `main.tex` has the shortened version.
- **Code availability.** See `code_availability.md`.
- **Author contributions.** E.P.G. is the sole author. CRediT roles:
  - Conceptualization;
  - Methodology (selection and approval of the methods);
  - Supervision;
  - Project administration;
  - Writing – review & editing.

  Software, experiment execution, formal analysis and drafting were carried out with generative AI tools under the
  author's direction, and the author takes responsibility for them. The statement does not attribute to the author
  manual work that was done only by automated tools.
- **Use of generative AI.** As in `main.tex`, built from the base text the author provided:

  > The author directed the research, approved the experimental methodologies, made the final scientific decisions,
  > and conducted a human review of the work. Generative AI tools were used substantially under the author's
  > supervision: Claude Code (Anthropic) for software implementation, execution of the simulations and experiments,
  > assistance with statistical analysis, documentation and manuscript drafting; ChatGPT (OpenAI) for methodological
  > and technical consultation, literature consultation, questions on editorial requirements and licences, and
  > critical review with proposals for improvement. Codex (OpenAI) contributed only to a separate
  > Bayesian-optimisation library in the same repository, which the method of this article does not use. No figure
  > was produced by a generative image model. The author retains full responsibility for the scientific content,
  > conclusions, and final submission. No AI system is listed as an author.

  Detail: `ai_usage_disclosure.md`. The human review is recorded in
  `research/phase7_extension/HUMAN_REVIEW_RECORD.md`.
- **Previous work.** "This is independent research carried out after the author's master's thesis (Universidad de
  Zaragoza, 2025), which studied sequential Bayesian optimisation of grasps for a humanoid hand in simulation,
  informed by DINO features and context. No text, figure or result of the thesis is reused, and the present study is
  not affiliated with the Universidad de Zaragoza."
- **Acknowledgements.** None, by the author's decision. The manuscript has no acknowledgements section.
- **Publishing route.** Subscription, with no article processing charge. No open-access option has been selected or
  paid for.

## Fit with Springer Nature's policy on AI (as relayed; to be confirmed on the official page at submission)

| Point | How the declaration meets it |
|---|---|
| Large language models do not meet the authorship criteria | No AI system is listed as an author |
| Substantive use must be documented in Methods (or a suitable alternative section) | Methods paragraph "Use of large language models", plus the declaration above |
| AI-assisted copy editing does not need to be declared | Not relied on: the use was substantive, and it is declared as such |
| Generative-AI images | None: every figure is produced by a plotting script from stored results |
| Human accountability | The author retains full responsibility; the AI tools made no decisions |

The official page could not be read from this environment, so the policy points above come from the requirements
relayed on 2026-11-26.
