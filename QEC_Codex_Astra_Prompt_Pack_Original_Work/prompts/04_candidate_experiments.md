# Stage 04: Controlled algorithm experiments

Read AGENTS.md, RULES_AUDIT.md, and experiment log. We already have a valid submission, so only improve it if evidence supports doing so. Work ONE experiment at a time and stop before starting another.

Read `docs/ORIGINALITY_POLICY.md` and `docs/DESIGN_DECISIONS.md` too. Any experiment MUST evolve the approved original decoder based on a specific evidence-backed hypothesis, not silently replace it with an off-the-shelf algorithm. If proposing a fundamentally different architecture, return to Stage 03A for a separate design approval. Record design provenance and conceptual sources.

Choose the single highest-value next experiment based on measured shortcomings. Options, ordered by simplicity and expected implementation risk:
- Better fallback for unseen detector patterns, e.g. nearest syndromes with constrained comparison and measured time cost.
- A compact MLP with 70 binary inputs and four sigmoid outputs, trained per basis with binary cross-entropy and independently generated Stim samples.
- A custom physics-informed model using `problem_info(circuit)` H/L/p, developed from first principles without prohibited decoders. Consider modest BP only if time permits.

Before coding, state one testable hypothesis, a runtime/training budget, dataset/train-validation seeds and quantitative promotion threshold. Reuse frozen validation and previous baseline. Make minimal changes in an isolated candidate path. Time preprocessing, training and inference separately. Treat the four outputs as four potentially correlated logical flips; choose loss and thresholds based on held-out validation, not leaderboard row edits. Reject any regression that cannot be justified by verified final scoring.

Append reproducible results and explain whether to promote. Do not claim a 'winning' result without measured evidence. STOP after this one experiment. The next experiment must be separately requested.
