# Intelligent Service Robotics: submission requirements

**Source.** https://link.springer.com/journal/11370/submission-guidelines

**How the requirements were obtained.** From this working environment the page redirects to the Springer login
(HTTP 303 to idp.springer.com; re-checked on 2026-11-27 with the same result), so it could not be read here. The requirements below were read from the
official page by ChatGPT on 2026-11-26, without bypassing authentication, and passed on by the author. They are
recorded as information from the official source, relayed. They were not verified directly from this environment.

## Requirements and how the submission meets them (status after the final build, 2026-11-27)

| Requirement (verified by ChatGPT on 2026-11-26 and relayed) | How the submission meets it | Status |
|---|---|---|
| At most **16 pages**, everything included | `manuscript.pdf`: 16 pages, figures, tables and references included; secondary material in the supplement (11 pages) | met |
| Official Springer Nature LaTeX template; `iicol` recommended | `sn-jnl` with `[pdflatex,sn-mathphys-num,iicol]`; no margin or font changes | met |
| Editable sources plus PDF | `.tex`, `.bib`, tables and figures in this folder; `manuscript.pdf`, `supplementary.pdf` | met |
| Abstract of 150–250 words | 235 words (counted from the PDF text) | met |
| 4–6 keywords | 6 | met |
| At most three heading levels | numbered sections and run-in paragraph headings | met |
| Numbered references in square brackets | `sn-mathphys-num` | met |
| Title page: name, affiliation if any, corresponding author's details, ORCID if available | Eduardo Pérez González; Independent Researcher, Zaragoza, Spain; corresponding author, eduperez2000g@gmail.com; ORCID 0009-0007-0303-8839 (confirmed by the author) | met |
| "Statements and Declarations" section | in `main.tex`: funding, competing interests, ethics approval, consent to participate, consent for publication, data, materials, code, author contributions, use of generative AI, previous work; no acknowledgements | met (texts confirmed by the author) |
| LLMs cannot be authors; substantive LLM use documented (Methods) and declared; human responsibility for the content | Methods paragraph "Use of large language models"; declaration built from the author's base text; `ai_usage_disclosure.md`; no generative-AI images | met |
| Hybrid journal: subscription route without APC; optional open access at €2,690 | **Decision: subscription, no APC.** No option selected or paid | decided by the author; confirm at submission |

These requirements were verified by ChatGPT and relayed by the author. They were not read directly from this
environment.

Any requirement not in this list counts as **unverified** and is not assumed. That includes figure file formats,
the supplementary-file size limit, the reviewer-suggestion policy, and the review model (single- or double-blind).
