# Intelligent Service Robotics: submission requirements

**Source.** https://link.springer.com/journal/11370/submission-guidelines

**How the requirements were obtained.** From this working environment the page redirects to the Springer login
(HTTP 303 to idp.springer.com), so it could not be read here. The requirements below were read from the
official page by ChatGPT on 2026-11-26, without bypassing authentication, and passed on by the author. They are
recorded as information from the official source, relayed. They were not verified directly from this environment.

## Requirements and how the submission meets them (status after the final build, 2026-11-26)

| Requirement (verified by ChatGPT on 2026-11-26 and relayed) | How the submission meets it | Status |
|---|---|---|
| At most **16 pages**, everything included | `manuscript.pdf`: 14 pages, figures, tables and references included; secondary material in the supplement | met |
| Official Springer Nature LaTeX template; `iicol` recommended | `sn-jnl` with `[pdflatex,sn-mathphys-num,iicol]`; no margin or font changes | met |
| Editable sources plus PDF | `.tex`, `.bib`, tables and figures in this folder; `manuscript.pdf`, `supplementary.pdf` | met |
| Abstract of 150–250 words | 240 words (counted from the PDF text) | met |
| 4–6 keywords | 6 | met |
| At most three heading levels | numbered sections and run-in paragraph headings | met |
| Numbered references in square brackets | `sn-mathphys-num` | met |
| Title page: name, affiliation if any, corresponding author's details, ORCID if available | name and provisional affiliation; e-mail and ORCID **pending the author**; nothing invented | author input needed |
| "Statements and Declarations" section | in `main.tex` (funding, competing interests, ethics, consent, data, materials, code, author contributions, use of generative AI, previous work); bracketed author fields | author input needed |
| LLMs cannot be authors; substantive LLM use documented (Methods) and declared; human responsibility for the content | Methods paragraph "Use of large language models"; declaration; `ai_usage_disclosure.md`. The statement of responsibility is left for the author to confirm | author confirmation needed |
| Hybrid journal: subscription route without APC; optional open access at €2,690 | **Decision: subscription, no APC.** No option selected or paid | decided by the author; confirm at submission |

These requirements were verified by ChatGPT and relayed by the author. They were not read directly from this
environment.

Any requirement not in this list counts as **unverified** and is not assumed. That includes figure file formats,
the supplementary-file size limit, the reviewer-suggestion policy, and the review model (single- or double-blind).
