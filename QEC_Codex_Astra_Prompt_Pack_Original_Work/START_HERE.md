# Using Codex Astra for the QEC Decoder Hackathon

## Setup
1. Place this file and `AGENTS.md` directly in the local `quantum_decoder` project folder that you opened in Codex.
2. Copy `docs/` and `prompts/` alongside them. Keep your original dataset ZIP where it is; the first stage inspects it.
3. Open the project folder in Codex. Select GPT-6 Astra High if desired.
4. Use `/plan` and paste the contents of `prompts/00_audit_and_plan.md`. Do not start with `/goal` plus a huge implementation request.
5. After the plan is reviewed, run one prompt at a time, starting with `prompts/01_setup_and_baseline.md`. Read `docs/ORIGINALITY_POLICY.md` for the hard originality requirement.
6. Stage 03A is mandatory: approve the team-designed decoder specification BEFORE Codex writes it.
6. Optionally set a succinct persistent `/goal` (see below) **only if you will keep directing the stages**. Do not let Goal mode run all stages unattended.

## Suggested /goal text

Independently design, implement and validate our team’s own reproducible QEC decoder, not a port, clone or renamed existing decoder. Respect the Kaggle rules and `docs/ORIGINALITY_POLICY.md`. Follow one approved stage at a time, preserve a working submission, measure performance on frozen holdout data, and stop for design approval before implementation.

## Source links
- https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026
- https://www.kaggle.com/competitions/quantum-error-correction-decoder/rules
- https://www.kaggle.com/competitions/quantum-error-correction-decoder/data
- https://www.kaggle.com/competitions/quantum-error-correction-decoder/leaderboard

## Stage ordering
00 Audit and rules plan (read-only except for a rules-audit note)
01 Environment, archive inspection, baseline LER, starter notebook
02 Create a valid submission, validate CSVs, upload only with user approval
03A Design three original approaches, document the selected one, stop for approval
03B Implement the approved original decoder and independently validate it
04 Improve one algorithm at a time under explicit time budget
05 Benchmark, optimize and select the best candidate
06 Final audit, reproducibility, hand-in package and optional final submission

## Daily tactics
- Submit a valid baseline early, subject to Kaggle rules and with user approval.
- Never use banned QEC decoding packages, even indirectly via a tutorial. Do not clone stock decoder implementations or present a renamed existing approach as an original invention.
- Never install tutorial extras merely for the initial notebook if they pull banned implementations.
- Distinguish organizer's `data/kaggle_public`/`private` folder naming from Kaggle's platform-level public/private holdout scoring split.
- Do not claim the live leaderboard score is the final score.
- Keep both the dataset ZIP and the last working candidate intact.
