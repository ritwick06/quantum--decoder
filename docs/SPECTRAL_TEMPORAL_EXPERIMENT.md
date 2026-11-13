# Full local fault distributions

## Preregistered experiment

In both directions of the incumbent decoder, the transition table represents at most three simultaneous mechanisms per slice. The audited circuit probabilities put the omitted unconditional mass from fault counts alone between 0.00020 and 0.00663 per slice. The missed fraction can be larger conditional on an unusual measured parity. A wider inference beam cannot recover probability discarded before it reads the table.

Replace the fault count cutoff with the full local joint distribution of the two parity masks and four signed logical moments. Keep the existing 48 largest unsigned-mass outgoing masks per incoming row and the incumbent bidirectional inference, lookahead, ambiguity threshold and parity fallback. This isolates the change to how transition mass is estimated. The overall decoder still approximates the posterior through transition selection and beam pruning.

Use the fact that the parity of independent binary faults has a product characteristic function. For the combined detector effect mask $h_m$, logical bit $\ell_{mj}$, and frequency mask $k$, its signed moment in the parity transform is $\prod_m (1-2p_m)^{(k\cdot h_m)\oplus\ell_{mj}}$. Compute the log of this product by a Walsh–Hadamard transform of sparse coefficients and then invert the exponentiated result. The unsigned channel is the same formula with every $\ell_{mj}=0$. Compute one channel at a time. A 28-bit transform uses 2 GiB of double precision workspace. Discard reconstructed unsigned mass at or below $2\times10^{-15}$ and clamp each signed moment to its unsigned mass to handle floating-point cancellation. Verify on a tiny exact enumeration including fault subsets larger than three.

Compare with `temporal_bidir_v1` on the frozen X and Z arrays generated with seeds `2026111301` and `2026111302`. The hypothesis is that restoring the missing mass reduces bitwise LER. Require positive paired gain with a positive lower 95% bound, lower locally penalized LER, and a median at or below 12 seconds over five local runs on four workers. Keep the 2 GiB inference and 20-minute, 5 GiB build budgets. Use only the circuits and check both bases and exact fresh-process CSV reproduction before recommending an upload.

For a candidate that passes frozen validation, check the direction and statistical reliability of the gain on a fresh Stim sample of 250,000 shots per basis, using one sample call each and seeds `2026111311` and `2026111312`. Use a positive paired 95% lower bound as the confirmation threshold.

The transform follows directly from independence of detector error mechanisms. It uses the same project's temporal inference and the established idea of summing likelihoods over fault explanations. [Stim detector error model specification](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md), [likelihood inference background](https://arxiv.org/abs/2609.39556)

## Result

Combined frozen LER changed from 0.03410125 to 0.03409625, four fewer bit errors over 800,000 predictions. The paired 95% interval for the gain was [−0.00006138, 0.00007138]. The candidate failed the accuracy gate. Its local median was 11.31 seconds, and all 34 tests passed. [Frozen validation report](../artifacts/spectral/validation.json)
