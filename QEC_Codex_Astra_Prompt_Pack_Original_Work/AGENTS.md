# QEC Decoder Hackathon: Project Instructions

## Mission and working mode
Build a **rule-compliant, reproducible, accurate, fast** classical decoder for the QEC Decoder Hackathon 2026. It receives detector-event arrays from the `[[18,4,4]]` qLDPC memory circuits and predicts four logical observable flips per shot. This is a time-limited competition. Prioritize a valid baseline and submission artifacts before speculative research or elaborate architectures.

**Work one stage at a time.** Read `START_HERE.md`, `docs/COMPETITION_RULES.md`, and the selected `prompts/NN_*.md` before working. At the end of a stage, STOP and report the acceptance checks, exact files changed, commands executed, measured results, unresolved rule questions, and recommended next stage. Do not begin another stage without an explicit new user instruction.

## Authority and rules

### Originality is a hard requirement
- Our competition solution must be an independently designed decoder, not a renamed or line-by-line rewrite of a known decoder. Do not copy code, pseudocode, architecture diagrams, algorithm-specific parameter settings, or trained weights from competitor repositories, papers, tutorials, or prohibited QEC packages. Never use those packages as a hidden dependency, baseline engine or teacher model.
- Do not make an ordinary reimplementation of belief propagation, OSD, matching, a generic k-nearest-neighbor decoder, or an off-the-shelf neural network the claimed final innovation. Common math (XOR, probability, sparse matrices, classical ML as building blocks) is allowed, but propose and justify OUR OWN decision logic, representation and error-inference strategy.
- Do not claim unprecedented research novelty unless actually supported by a proper literature search. "Original to this project" means independently chosen design and implementation with traceable decisions; it does not mean no one has ever thought of anything similar.
- After baseline submission, run `prompts/03a_original_decoder_design.md` as a separate read-only design stage. Propose at least three original project-specific hypotheses, select one to test, document its mathematics, data flow, failure modes and speed estimate in `docs/DESIGN_DECISIONS.md`, then STOP for the user's approval before writing decoder code.
- Implement only the explicitly approved design in Stage 03B; preserve original reasoning and experiment history. Any new external source needs a source link and explanation of what was learned, with a clear distinction between external ideas and original implementation.
- Before final hand-in run an originality/provenance review. If uncertain whether a concept closely reproduces an existing named decoder, describe that uncertainty and redesign or ask the user instead of masking it.
- Review `docs/ORIGINALITY_POLICY.md` at every design, implementation, experiment and final-audit stage.

1. **The authenticated Kaggle Competition Rules, Overview/Evaluation, Data, Submission interface, and organizer announcements are authoritative.** Check them using available browser access or provided screenshots. If inaccessible, explicitly mark the facts UNVERIFIED; do not pretend you read Kaggle.
2. Read the latest official repository README and submission template: https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026 . If it conflicts with Kaggle, Kaggle wins. The README currently contains duplicated/conflicting submission and timing sections. Do not silently pick one.
3. The public leaderboard is live. The user's screenshot says it is based on approximately 30% of the test data and that the remaining 70% determines final standings. **Use leaderboard feedback only for evaluating whole model versions, not hand-editing individual predictions or trying to infer hidden labels.** Rankings and scores change.
4. Banned QEC decoder implementations, code copied from them, and pretrained decoders made by others: `pymatching`, `ldpc`, `stimbposd`, `beliefmatching`, `tesseract-decoder`, `fusion-blossom`, and equivalents. Tutorials may import banned packages, but our submitted decoder must not. Build the decoder logic ourselves. General-purpose tools such as NumPy, SciPy, Numba, PyTorch, JAX, scikit-learn, generic optimization and graph libraries are permitted in the official repo summary, subject to Kaggle confirmation.
5. Training data may be generated using the provided Stim circuits. Do not use hidden test labels, hand-label predictions, tune individual predictions against leaderboard feedback, or privately share code with other competing teams. Respect Kaggle's team, deadline, submission quota, reproducibility, environment and AI-use/disclosure requirements as verified.
6. Do not upload, publish, submit, delete, overwrite, change Git remotes, or install system-level packages without first explaining the action and getting approval. Do not use Kaggle credentials in scripts or logs. Ordinary local sandbox commands, creating code, and testing in the project are permitted.

## Known technical contract, subject to live-rule verification
- Competition: `[[18,4,4]]` code, five rounds, 70 detector bits, four logical observables, X and Z memory experiments, SI1000 circuit noise at p=0.002.
- Preferred function: `decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray`.
- Inputs typically shape `(N,70)` and bool/0-1. Return `(N,4)` integer/bool 0-1 with no NaNs, same shot order, deterministic, and sensible behavior for N=0.
- Use each row's `basis` to pick its correct X/Z Stim circuit.
- Labelled data: `data/test_set/test_set.csv` from repository; organizer README says it contains 200,000 labelled shots. Confirm on disk before assuming.
- Official README describes `data/kaggle_public/` and `data/kaggle_private/` with CSVs and circuit files, but the *current Kaggle data format may differ.* Inspect both on-disk files and Kaggle Data tab.
- Predicted submission columns are `id,L0,L1,L2,L3` in exact order, with binary values and untouched IDs. Validate against the CURRENT Kaggle `sample_submission.csv`, not an old typo-containing sample. The repo describes two separate submission files but Kaggle UI must be checked for the actual submission procedure.
- Lower LER (binary MAE across all four observables) is better. The README quotes a zero-baseline LER around 0.29. Verify locally.
- Benchmark decoder speed and determinism. The official README contains mutually inconsistent t_cap/alpha passages; do not assert a final value before checking Kaggle.

## Data safety
- The user has already placed one or more `.zip` dataset archives in the `quantum_decoder` project folder. Discover filenames and list archive contents before extracting. Do not re-download data by default. Reject path traversal, absolute paths, symlinks and unexpected overwrite. Do not modify original archives.
- Verify directory layout, CSV schema, row counts, `basis` values, detector bit columns D0..D69, optional labels L0..L3, and associated X/Z circuit files. Do not assume the ZIP contains both public and private files.
- Do not accidentally train on the held-out Kaggle competition datasets. Keep training, independent local validation and official test submissions separate.
- Avoid expensive, open-ended training or downloads. Establish a working, scored baseline first and set resource/time limits before experiments.

## Implementation quality
- Prefer small modules with explicit interfaces, simple functions, unit tests and measurable improvements. No giant one-shot refactors.
- Do not replace the organizer's files in place unnecessarily. Keep original setup, templates and generated artifacts clearly separated.
- For every new decoder: validate output shape/type/range/order; test both bases; test batches including empty, 1, small, and full-size; check repeated-call consistency; measure LER on genuinely unseen samples and runtime. Compare with the best accepted prior candidate. Revert regressions.
- Keep a reproducible experiment log in `docs/EXPERIMENT_LOG.md` or CSV: timestamp, candidate, data split/seed, X LER, Z LER, combined LER, speed, public score if provided, notes. Do not invent scores.
- Optimize for a runnable entry point and verifiable artifacts. Avoid premature GPU dependency. Record dependencies and seed; provide a clean restart/reproduce procedure.

## Human-readable writing style
- Style preference does not override truthful attribution or any required competition disclosure. Produce authentic, concise engineering documentation; never invent authorship, falsely say the code was written without assistance, or remove mandatory provenance notices.
- Write the way a careful engineer documents their own work: concise, plain, specific. Comments should explain non-obvious invariants, mathematical assumptions, edge cases and optimization choices, not narrate obvious code.
- Avoid boilerplate, promotional language, decorative headings, emoji, AI self-references, filler comments, and em dashes in source code and documentation. Keep meaningful explanations and accurate attribution where appropriate.
- Do not add false statements about authorship, conceal required disclosures, or misrepresent copied or third-party material. If Kaggle requires AI-tool disclosure, follow that requirement.

## End-of-stage report template
1. Stage completed and high-level outcome.
2. Files created/changed (paths).
3. Tests/commands run with actual outcomes.
4. Accuracy and speed measurements (or 'not measured').
5. Rules verified, rules still uncertain, and blockers.
6. Suggested next prompt. STOP.
