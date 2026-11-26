# Intelligent Service Robotics: submission requirements

**Source.** https://link.springer.com/journal/11370/submission-guidelines

**How the requirements were obtained.** From this working environment the page redirects to the Springer login
(HTTP 303 to idp.springer.com), so it could not be read here. The requirements below were read from the
official page by ChatGPT on 2026-11-26, without bypassing authentication, and passed on by the author. They are
recorded as information from the official source, relayed. They were not verified directly from this environment.

## Requirements and how the submission meets them

| Requirement (as relayed) | How the submission meets it | Status |
|---|---|---|
| At most **16 pages**, figures, tables, references and appendices included; exceeding it can lead to rejection without review | Manuscript rebuilt in the Springer template with `iicol` and checked by page count after compilation; secondary analyses moved to the supplementary material, limitations and negative results kept | to verify after the final build (stage 14) |
| Springer Nature LaTeX template recommended, option `iicol` | `main.tex` to be switched from its current class to the Springer Nature template with `iicol` (stage 14) | pending (template files must come from Springer's official template package) |
| Compiled PDF plus editable sources (figures, styles) | `publication/intelligent_service_robotics/` holds the `.tex`, `.bib`, figure PDFs and the build script | in place; final check at stage 14 |
| Abstract of 150–250 words; no undefined abbreviations; no unspecified references | Abstract rewritten and counted | stage 14 |
| 4–6 keywords | To be set in the template's keyword field | stage 14 |
| Decimal heading numbering, at most three levels | Current structure uses sections and paragraphs; checked at stage 14 | stage 14 |
| Numbered citations in square brackets | Springer template reference style (numbered) | stage 14 |
| Title page: name, affiliation if any, corresponding author information, ORCID if available; authors without institutional affiliation are allowed | Placeholders only; **no affiliation is invented** | author input needed (name, e-mail, ORCID; affiliation or "independent researcher") |
| "Statements and Declarations" section with the relevant items (competing interests, funding, …) | Section drafted in `declarations.md`; the entries that depend on the author's real situation stay as placeholders | author input needed |
| LLMs cannot be authors; substantive LLM use must be documented in Methods or another suitable section; the copy-editing exception does not cover this project; the human author is responsible for the content | `ai_usage_disclosure.md` and a Methods paragraph describe the roles of Claude Code and ChatGPT in implementation, experiments, analysis and writing | updated at stage 14 |
| Hybrid journal; the subscription route should have no mandatory APC; no paid option is chosen for the author | Not selected or paid for here; the conditions of the subscription route must be confirmed by the author at submission | author decision |

Any requirement not in this list counts as **unverified** and is not assumed. That includes figure file formats,
the supplementary-file size limit, the reviewer-suggestion policy, and the review model (single- or double-blind).
