# QEC Decoder Hackathon: Rules and Source-of-Truth Checklist

Compiled on 2026-11-13 from the public organizer repository and the user's Kaggle screenshot. **This is a working summary, not a replacement for the live Kaggle Rules tab.** Verify organizer revisions before submission.

## Primary references, in precedence order

1. Kaggle Rules (private/authenticated, may require user browser session): https://www.kaggle.com/competitions/quantum-error-correction-decoder/rules
2. Kaggle Overview/Evaluation: https://www.kaggle.com/competitions/quantum-error-correction-decoder/overview
3. Kaggle Data: https://www.kaggle.com/competitions/quantum-error-correction-decoder/data
4. Kaggle leaderboard: https://www.kaggle.com/competitions/quantum-error-correction-decoder/leaderboard
5. Kaggle Discussion/organizer announcements: https://www.kaggle.com/competitions/quantum-error-correction-decoder/discussion
6. Organizer GitHub README: https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026
7. Organizer starter notebook: https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026/blob/main/notebooks/getting_started.ipynb
8. Organizer submission template: https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026/tree/main/submission_template
9. Stim documentation: https://github.com/quantumlib/Stim

## Verified in public organizer README

- Build your own decoder for the `[[18,4,4]]` qLDPC quantum code using Stim circuits. X/Z experiments have 70 detector events and 4 logical observable labels per shot.
- Noise: SI1000 with p=0.002; five error correction rounds.
- Measure logical error rate: fraction of wrong logical observable bits, equivalent to binary MAE over four outputs per shot. Lower is better; zero prediction reported around 0.29.
- The provided function contract is `decode(circuit, detector_events)` and its returned array must be binary with shape `(shots, 4)`.
- Output CSV is `id,L0,L1,L2,L3` in exactly that order. An earlier sample submission had a typo; check the latest one.
- The README describes `data/test_set/` with 200,000 labelled shots and `data/kaggle_public/` / `data/kaggle_private/` with 200,000 unlabelled shots each. Inspect the downloaded ZIP and current Kaggle Data tab instead of assuming they match.
- Prohibited: existing QEC decoder implementations and code copied from them (`pymatching`, `ldpc`, `stimbposd`, `beliefmatching`, `tesseract-decoder`, `fusion-blossom`), as well as other teams' pretrained decoders. Examples of allowed general-purpose libraries include NumPy, SciPy, Numba, PyTorch, JAX, scikit-learn, graph and optimization libraries.
- Allowed: generate more samples using the organizer's Stim circuits; design and train a custom algorithm.
- Not allowed: hand-labelled competition predictions, tuning individual predictions from leaderboard feedback, sharing code privately between teams. Team size up to 4 according to README.
- Top 10 private-leaderboard teams progress to decoder hand-in and runtime check, with determinism/reproduction requirements; competition runtime uses a GPU+CPU machine. Confirm all details on Kaggle.

## Live screenshot evidence (2026-11-13 19:24 India time)

- Competition is actively running; leaderboard has Public and Private tabs.
- Public leaderboard page states approximately 30% of test data is used for the visible standings, with the remaining 70% determining final results. Do not treat public score as the final ranking.
- User already downloaded a dataset ZIP into their local project. Filename and archive contents not yet independently confirmed.

## Unresolved / conflicting details: MUST verify

- The README contains duplicated old/new sections. One lists t_cap=15.4 seconds and alpha=0.2; another calls t_cap provisionally 3 seconds and alpha=0.1. Do not hardcode either without current Kaggle confirmation.
- The README alternately describes a singular submission file and two files (public and private). The live Kaggle UI and Data tab must decide how many files and which upload route are actually required.
- One section says a maximum of 10 submissions per day and up to 2 final selections; confirm the live quota and submission selection process.
- A reference to 'public vs private sets' in the README may not correspond exactly to the Kaggle platform's 30%/70% visible/hidden scoring split. Reconcile with actual data and rule pages.
- The README contains contradictory/provisional presentation deadlines and descriptions. Check the official rules and announcements.
- AI-assisted software development/disclosure is **not established** by the accessible public README summary; check Kaggle Rules for AI-specific restrictions. Do not claim all work was manual if disclosure is required.

## Required rules review in Stage 00

- Confirm whether "independently developed decoder" has any stricter competition-specific originality or similarity-test definitions, and whether generated code is permitted. Our own stricter originality policy applies unless contradictory competition rules must be resolved.

Record each item with `confirmed`, `conflict`, or `unverified`, plus source, timestamp and evidence:

- Competition deadline and timezone; submission quota.
- Whether AI-assisted coding is permitted and whether it must be disclosed.
- Allowed/disallowed libraries, reference implementations, pretrained models and external code.
- Required deliverables: number of CSV files, exact headers, submission flow, source hand-in and trained weights.
- Prohibition on hand-labeling, leaderboard-targeted per-row changes, test leakage.
- LER/MAE scoring and hidden holdout split.
- Speed scoring formula, cap, hard timeout and evaluation hardware.
- Data layout and data-use permissions; handling of X vs Z experiments.
- Team and code-sharing restrictions.
- Finalists, code hand-in, presentation dates and prize eligibility if relevant.

If Kaggle content is inaccessible, STOP after safe local inspection and ask the user for a screenshot or copied text of the relevant Kaggle Rules/Overview pages. Work on the compliant baseline can continue in later stages, but do not invent rule confirmations.
