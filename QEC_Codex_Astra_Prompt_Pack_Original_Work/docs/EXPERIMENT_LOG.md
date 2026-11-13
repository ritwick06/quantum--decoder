# Experiment Log

Keep this append-only. Use actual measured values only. Record how the validation set was generated and its seed. Leave fields as `pending` rather than guessing.

| Timestamp (IST) | Candidate / revision | Train data / seed | Validation data / seed | X LER | Z LER | Combined LER | Runtime for 200k shots | Kaggle public score | Notes |
|---|---|---|---|---|---|---|---|---|---|
| pending | all-zero baseline | none | pending | pending | pending | pending | pending | not submitted | First reproducible benchmark |

## Candidate selection policy
- Compare on the same **frozen independent validation set** whenever possible.
- Preserve the fastest valid working submission until another candidate clearly improves accuracy or final score without violating rules.
- Public Kaggle feedback is secondary confirmation, not a target for adjusting individual predictions.
- Log exact commits/checkpoints, commands, seeds, required artifact files, training duration, and any environment requirements.
