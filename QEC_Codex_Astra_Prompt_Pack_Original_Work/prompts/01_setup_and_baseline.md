# Stage 01: Setup, data verification and a real baseline

Follow `AGENTS.md` and the latest `docs/RULES_AUDIT.md`. Complete ONLY this stage. Aim for an actual local LER and a working `decode` function, not a research project.

1. If the official repo is absent, clone it into a clearly named child folder without overwriting the dataset ZIP or project instructions. Use the official repo as the source of truth. If the repo exists, preserve user changes.
2. Safely inspect then extract only necessary archive members to their expected locations after validating paths, symlinks, overwrite risks, schema and expected dataset identity. Keep the original ZIP intact. If archive entries are incompatible with the repo layout, explain and choose a safe mapping.
3. Use Python 3.10+ in a local virtual environment. Install only base dependencies needed for the starter runner; avoid banned decoder packages and unnecessary tutorial extras.
4. Inspect actual data: CSV headers, number of samples, X/Z counts, D0..D69 contents, optional L0..L3, ID uniqueness, Stim circuit files and versions. Report missing Kaggle data clearly.
5. Run the organizer's existing baseline locally (`submission_template/run_local.py`) or minimally implement the documented all-zero `decode` function. Measure X LER, Z LER, combined LER, runtime and determinism. Never fabricate any result.
6. Add minimal smoke tests for `decode` shape/dtype/binary values, empty input, one row, both bases and repeatability. Do not rewrite the official runner.
7. Append measurements to `docs/EXPERIMENT_LOG.md`. Record the exact command and environment.

Acceptance: baseline executes end-to-end on available labelled data; all output-contract checks pass; data splits are understood; no prohibited packages are imported. STOP and explain the exact next action. Do not train a neural network yet. Do not submit without approval.
