# Temporal decoding with a midpoint join

## Preregistered experiment

The accepted `temporal_bidir_v1` averages the mass retained by two end-to-end passes. Each can prune a history before it has access to measurements on the far side. In both circuits all fault mechanisms span at most two adjacent slices. Slice 2 contains 113 mechanisms confined to the slice, with an expected fault count of 0.233.

Split at slice 2, counting from zero. Use the incumbent forward tables for slices 0 and 1 and its reverse tables for slices 5, 4 and 3. These two groups contain disjoint mechanisms. Only the slice 2 mechanisms remain. Compute their full distribution of the incoming parity and four signed logical moments by XOR convolution. When the two searches reach the middle, join their retained states by looking up the middle parity implied by the observation and the two boundary masks. Multiply and sum the corresponding moments across pairs. Every retained full history contributes once.

Keep the existing limit of three faults per table and 48 transitions per row. Use 32 states at internal slices and 48 at each join boundary. Rank the final states by accumulated mass times the exact projected likelihood of the observed middle slice from the mechanisms assigned to the opposite side and to the middle. Use the existing 2.5% ambiguity threshold and parity fallback. The 48 by 48 join is bounded. The likelihood is approximate because the two searches prune paths and their tables omit fault subsets.

The hypothesis is that conditioning both sides jointly on the middle slice will improve bitwise LER at a local median of at most 12 seconds over five runs with four workers for 200,000 shots. Compare once with `temporal_bidir_v1` on the original frozen validation seeds `2026111301` and `2026111302`. Require a positive absolute combined LER gain with a positive lower bound of the paired 95% interval and a lower locally penalized LER. Retain the 2 GiB inference memory and 20-minute, 5 GiB build budgets. No labelled fitting or leaderboard feedback. Check the join against a tiny exhaustive model, validate both circuit contracts, and require fresh-process byte-identical CSV reproduction before flagging an upload.

The use of likelihood sums is established in QEC. This experiment reuses the project's own temporal tables and derives the join directly from the circuit fault partition. [Likelihood inference background](https://arxiv.org/abs/2609.39556)

## Result

The midpoint join produced an LER of 0.03581125 versus 0.03410125 for `temporal_bidir_v1`. The paired gain was −0.001710 with a 95% interval of [−0.001906, −0.001514]. The local five-run median was 4.270 seconds for 200,000 shots and the full suite passed 32 tests. The candidate failed promotion. [Frozen validation report](../artifacts/midpoint/validation.json)

The join enforces the complete measured parity for every retained pair of histories. The two sides still prune histories independently, and the frozen result shows that this partition loses accuracy overall.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/build_temporal.py --midpoint
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/evaluate_temporal.py --candidate temporal_midpoint_v1 --replay
```
