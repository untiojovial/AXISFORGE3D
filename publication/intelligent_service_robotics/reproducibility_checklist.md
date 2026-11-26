# Reproducibility checklist

| Item | Status | Where |
|---|---|---|
| Code for every experiment | Yes | `tools/research/phase7/`, `research_grasping/` |
| Exact commands | Yes | Docstrings of each script; `research/phase7/reproducibility.md` |
| Seeds | Yes | Folds 20261201 and 20261130; bootstrap 20261128; EGAD selection 20261202; power 20261203–06; timing subset 20261207; epsilon subset 20261208 |
| Data versions and hashes | Yes | `PHASE7_DATA_MANIFEST.json`, `egad_archives.sha256`, `egad_selection.json` |
| Frozen models and hashes, before the test | Yes | `PHASE7_MODEL_FREEZE.json`, commit `de22a8a` |
| Protocol before the test | Yes | `PHASE7_PROTOCOL.md` part II (same commit) |
| Deviations disclosed | Yes | Protocol II.1 (D1) and II.8 |
| Every selected object reported, including non-viable ones | Yes | `confirmation_results.json` (`status_counts`, population Success@B) |
| Unit of analysis = object, paired | Yes | Protocol II.6 |
| Multiplicity control | Yes | Holm within the primary and the secondary family |
| Effect sizes with CIs, not only p-values | Yes | Results tables |
| Development and confirmation kept separate | Yes | Manuscript Sects. 5 and 6 |
| Hardware / compute | Yes | 4 CPU cores, no GPU; timings specific to the container |
| Tests | Yes | `tests/phase7/test_phase7.py`; full-suite counts in `research/phase7/publication_readiness.md` |
| Real-robot validation | **No** | Simulation only (limitations) |
| Archived release with DOI | **No** | Requires the author's authorisation |
