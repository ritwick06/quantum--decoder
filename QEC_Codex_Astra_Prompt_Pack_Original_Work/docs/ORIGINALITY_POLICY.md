# Originality and provenance policy

This is a binding instruction for the Codex Astra agent working in this project. Follow it together with `AGENTS.md` and the verified Kaggle rules. If an authenticated Kaggle rule conflicts with this document, report the conflict before acting.

## What "our own work" means

1. The team's decoder has a project-specific hypothesis, algorithm design, implementation, experiments and trade-off decisions that we can defend in a review.
2. Do not import, port, wrap, paraphrase or copy an existing QEC decoder implementation from any repository, package, notebook, article or other team. Also do not use a third-party model or decoder to generate labels for imitation training.
3. Do not treat simply renaming a standard algorithm (BP, BP+OSD, MWPM, union-find, generic MLP, kNN or LUT) as an original decoder idea. These can be studied conceptually or used as individual building blocks only where allowed, and only if the team's distinctive logic is materially different and explained.
4. Use independent reasoning based on the provided problem inputs (`D`, the X/Z circuit, detector history, `H`, `L`, `p` where the official APIs expose them). State precisely which features, inference rule, uncertainty handling and runtime optimizations are our design choices.
5. Do not promise novel research or claim the algorithm has never existed. Full literature novelty is not realistically verifiable in this sprint. Accurate claim: "designed and implemented independently for this challenge."
6. Do not conceal AI assistance if disclosure is required. Clean code style is not falsification of authorship.

## Required design gate: BEFORE implementation

Produce `docs/DESIGN_DECISIONS.md`, containing:
- Problem observations from actual labelled training data only, plus fixed independent holdout definition.
- At least three concrete, independently reasoned decoder concepts suited to 70 detectors / 4 logical outcomes and X/Z basis separation.
- For each: a brief mathematical rule or pseudocode, features required, likely ambiguity, inference complexity, potential LER advantage, likely failure modes and how it differs from a plain stock baseline.
- A comparison table and one selected small experiment.
- Explicit links for background concepts that informed the design, separating those concepts from our distinct design decisions.
- A one-page implementation specification with test cases and acceptance threshold.
- User approval. Do NOT proceed from design to code automatically.

## Experimental integrity

- Pre-register hypothesis, seeds, validation samples and success criteria before observing experiment results.
- Keep one simple zero-output fallback for interface sanity, but do not submit it as "an original decoder contribution".
- Record why each modification is made; do not do an unbounded hyperparameter sweep against public Kaggle scores.
- Record package licenses and source provenance. Cite conceptual references in documentation. Do not reuse distinctive source code or parameterization.
- Review local runtime and accuracy against identical frozen validation sets. Preserve the previous verified submission.

## Final provenance checklist

- [ ] Design decisions and limitations are documented
- [ ] All third-party dependencies are general-purpose and allowed under verified rules
- [ ] No disallowed decoder implementation, copied pseudocode/logic or prohibited pretrained model
- [ ] Any conceptual influences are acknowledged accurately
- [ ] Dataset seeds and split boundaries are reproducible
- [ ] No hidden Kaggle labels or leaderboard-driven per-shot adjustments
- [ ] Claims about originality and AI use are truthful and comply with competition rules
- [ ] A judge can follow the original algorithm from the math/pseudocode and reproduce it
