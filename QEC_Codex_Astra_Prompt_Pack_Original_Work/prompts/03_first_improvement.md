# Stage 03B: Implement our approved original decoder

Read `AGENTS.md`, `docs/ORIGINALITY_POLICY.md`, `docs/DESIGN_DECISIONS.md`, the latest `docs/RULES_AUDIT.md`, and the experiment log. Stage 03A must have ended with user approval of one concept. If there is no documented approval, STOP; do not invent a design or switch to a stock decoder.

Implement ONLY the approved independently designed algorithm from `docs/DESIGN_DECISIONS.md` as a first small, testable candidate. Preserve the last working baseline and submission files.

1. Restate the approved algorithm's mathematical rule, module boundaries, expected complexity and limitations before editing code. If key design ambiguities exist, document and ask for guidance rather than adding arbitrary complexity.
2. Use general-purpose allowed dependencies only and no copied code, pseudocode or tuned constants from third-party QEC decoders, other competitors, tutorials or published decoder implementations. If applying a general mathematical idea, implement the team-specific reasoning independently and cite conceptual inspiration in documentation.
3. Separate data processing, model construction/training if needed, inference, and validation. Match `decode(circuit, detector_events)` with deterministic binary (N,4) outputs for both bases, including empty and singleton batches.
4. Train only on independently generated and permitted labelled samples. Do not learn from Kaggle held-out labels or public scores. Preserve frozen validation seeds.
5. Measure X/Z/combined LER, runtime, peak memory, repeated-call behavior and known failure modes; compare with the same validation split and zero-output baseline.
6. Record the actual results, source provenance, design choices and any deviations in `docs/DESIGN_DECISIONS.md` and `docs/EXPERIMENT_LOG.md`.
7. Promote only if the pre-registered accuracy/runtimes criteria are met. If not, preserve the previous best and explain why, then STOP.

Do not automatically start Stage 04. Do not submit without approval. Never label a clone of an existing decoder an original innovation.
