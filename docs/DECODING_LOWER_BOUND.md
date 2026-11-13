# Lower bound on detector-only logical error

The requested target is 0.02000 bitwise LER. This diagnostic gives an ideal decoder extra information so its optimal error rate is a lower bound for a decoder that receives the 70 observed bits.

The pinned circuits produce independent binary mechanisms under the Stim detector error model. Assign each mechanism to the first slice it affects in the chosen time direction and group the final two slices together. The blocks contain disjoint mechanisms. Reveal each block's separate detector parity to the ideal decoder. Their XOR gives the actual observation. For a cheaper bound, reveal the actual faults in the three interior blocks. Their logical signs are then known exactly.

For bit $j$ and a revealed block syndrome $s$, define $m_j(s)=P(s,L_j=0)-P(s,L_j=1)$. The mean absolute posterior sign of that block is $\sum_s |m_j(s)|$. The conditional sign for the XOR of independent block contributions is the product of their conditional signs. Therefore the optimal bit error with these reveals is $(1-\prod_b\sum_s |m_{bj}(s)|)/2$. No inference model can benefit from seeing fewer observations. Take the stronger of the forward and reverse oracle bounds for each basis and bit, then average with the contest's equal basis and bit weights.

Compute each signed parity distribution by Walsh–Hadamard transform of the exact independent-fault characteristic function. This is a numerical lower bound for the expected error under the pinned circuit model. It does not specify an exact score on a finite hidden test sample.

The independent check enumerates only fault subsets of size up to three. For each local parity and logical bit, it sums the smaller of the two joint masses of the retained histories. Each joint mass is a sum of nonnegative terms. Omitting histories can only decrease the smaller mass, so this yields a weaker bound without relying on cancellation in the transform. Subtract a small rounding allowance from each local error estimate before combining blocks.

The implementation uses no Kaggle rows or labels. The independent-mechanism semantics come from the [Stim detector error model specification](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md). [Cheng and Pan](https://arxiv.org/abs/2609.39556) discuss optimal likelihood inference and circuit-specific optimality gaps.

## Results

| Revealed information and calculation | X lower bound | Z lower bound | Combined lower bound |
|---|---:|---:|---:|
| Exact boundary parities, all interior faults revealed | 0.01590173 | 0.01469618 | 0.01529895 |
| Exact boundary parities and conservative interior enumeration | 0.02168613 | 0.02046049 | 0.02107331 |
| Exact block parities throughout | 0.02318426 | 0.02198386 | 0.02258406 |

The stronger numerical lower bound of 0.02258406 is above the 0.02000 target. The conservative calculation also exceeds it after retaining at most three faults in each interior block and applying rounding allowances. Even the ideal decoder in this calculation has strictly more information than a competition decoder. This rules out 0.02000 **expected** LER under the supplied circuit model. The actual optimal detector-only LER may be higher than this lower bound. [Full transform report](../artifacts/oracle/full.json), [conservative report](../artifacts/oracle/hybrid.json)

```sh
mkdir -p artifacts/oracle
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/bound_temporal.py \
  --boundary-only --output artifacts/oracle/boundary.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/bound_temporal.py \
  --output artifacts/oracle/full.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/bound_temporal.py \
  --subsets --output artifacts/oracle/subsets.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/bound_temporal.py \
  --combine artifacts/oracle/boundary.json artifacts/oracle/subsets.json \
  --output artifacts/oracle/hybrid.json
```
