# Reproducibility checklist

| Item | Status | Where |
|---|---|---|
| Code for every experiment | Yes | `tools/research/phase7/`, `tools/research/phase7_extension/`, `research_grasping/`, `crates/grasping/src/learned.rs` |
| Exact commands | Yes | Docstrings of each script; `research/phase7/reproducibility.md`; `research/phase7_extension/REPRODUCIBILITY.md` |
| Seeds | Yes | **Phase VII:** folds 20261201 and 20261130; bootstrap 20261128; EGAD selection 20261202; power 20261203–06; timing subset 20261207; epsilon subset 20261208. **Extension:** timing 20261217; cluster bootstrap 20261216; CoACD 0; Optuna 20261212 + fold |
| Data versions and hashes | Yes | `PHASE7_DATA_MANIFEST.json`, `egad_archives.sha256`, `egad_selection.json`; `PHASE7X_FREEZE.json` (objects with mesh SHA-256) |
| Frozen models, analysis and binary before each test | Yes | First confirmation: `PHASE7_MODEL_FREEZE.json`, commit `de22a8a`. Second: `PHASE7X_FREEZE.json`, commit `b478289` (36 files, planner binary, GPD adapter) |
| Freeze checked during the run | Yes (second confirmation) | 10 checks, before and after every stage (`results/confirmation_x/execution_record.json`) |
| Deviations disclosed | Yes | Phase VII D1–D4; extension: position 300 prepared before the freeze (protocol §2) |
| Every selected object reported, including non-viable ones | Yes | `confirmation_results.json`; `confirmation_x.json` (`counts`) |
| Unit of analysis = object, paired | Yes | both protocols |
| Multiplicity control | Yes | Holm within each family |
| Effect sizes with CIs; estimands stated; rank tests as sensitivity only | Yes | results tables; `PHASE7X_PROTOCOL.md` §5 |
| Development and confirmation kept separate | Yes | manuscript Sects. 6, 7 and 9 |
| Per-object outputs | Yes | `research/phase7/results/`; `research/phase7_extension/results/confirmation_x/` |
| Hardware and compute | Yes | 4-core Intel Xeon at 2.30 GHz cloud container, no GPU; timing pinned to one CPU, one thread; load recorded |
| Clean-environment reproduction | Partial | slim Docker image: statistics and planner decisions (`reproducibility/DOCKER_VALIDATION_RECORD.md`). The full pipeline was not repeated in a clean environment |
| Tests | Yes | `research/phase7_extension/TEST_RESULTS.md` (368 Rust, 513 Python, 65 skips explained) |
| Real-robot validation | **No** | simulation only (limitations) |
| Archived release with DOI | **No** | requires the author's authorisation; draft metadata in `research/phase7_extension/package/zenodo_metadata.json` |
