# Stage 06: Final compliant hand-in and reproducibility audit

Work only on a final candidate already approved by me. Follow AGENTS.md, confirmed Kaggle rules and README/template contract.

Also follow `docs/ORIGINALITY_POLICY.md` and review `docs/DESIGN_DECISIONS.md`. Include an originality audit: verify the central inference logic matches the team's approved design rather than a renamed stock decoder; review provenance of every external idea, code module, trained weight and any borrowed parameter settings. Document any overlap with existing literature accurately, without unsupported novelty claims.

1. Run tests that prove `decode(circuit, detector_events)` gives deterministic binary `(N,4)` outputs for both X/Z circuits, across empty/small/large batches. Ensure rows and IDs remain in original order.
2. Run the organizer's local test harness. Record the actual X/Z/overall LER, repeated-call consistency, wall time and memory. If missing dependencies/hardware prevent any test, report that clearly rather than inventing a pass.
3. Inspect dependency tree and repository imports for banned decoder packages and borrowed/pretrained decoder code. Also check source files and runtime artifacts for inadvertent use of hidden competition labels or per-row hand adjustments.
4. Produce exact requirements versions, decoding source, generation/training scripts, fixed seeds, lightweight trained weights or tables, and a brief reproduce-from-scratch procedure. Check relative paths and that the decoder works without internet at inference time.
5. Validate the final Kaggle CSV file(s) against the ACTUAL current submission format, preserving a known good previous artifact.
6. Prepare a concise, accurate explanation of the method for a human judge: what inputs are, model/algorithm, validation protocol, LER, speed, tradeoffs and original engineering choices. Write it in plain engineering prose. No fabricated claims, no fake 'human-only' authorship; disclose AI assistance if required by official rules.
7. Report unverified rules, blockers and every exact output artifact. Request my approval before any upload, submission or sharing.

STOP after final audit. Do not modify code further unless a concrete failure is found and approved for fixing.
