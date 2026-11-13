# Stage 05: Speed benchmark, candidate selection and leaderboard feedback

Use the current verified speed-scoring rules. The organizer README has conflicting t_cap and alpha figures; if Kaggle cannot confirm them, present sensitivity scenarios rather than inventing authority.

1. Benchmark current best and fallback on the same representative batch sizes, including full 100k-shot per-basis decode calls if hardware permits. Reproduce organizer run_local.py timing methodology; measure throughput, peak memory, X/Z LER and deterministic output.
2. Profile only real bottlenecks. Optimize small hot spots using allowed general-purpose numerical tools. Do not install or call prohibited QEC decoders, even for benchmarking within the submitted solution.
3. Record commit, environment, seeds, artifact versions, test results and measured runtime. Confirm no local validation deterioration.
4. If I provide PUBLIC leaderboard scores, append them to the experiment log with exact submission ID, timestamp, candidate hash, local results and public score. Do not target individual hidden rows or reverse-engineer labels from leaderboard values. Do not confuse public leaderboard rank with the final private rank.
5. Recommend which existing candidate to select or submit next. Explain the evidence and any unresolved format/rule concerns.

Do not submit anything or alter the saved best candidate until I explicitly approve it. STOP.
