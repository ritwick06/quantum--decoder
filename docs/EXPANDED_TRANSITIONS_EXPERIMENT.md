# Expanded transitions for ambiguous shots

## Preregistered experiment

The 128-state beam experiment kept more states but used the same 48 outgoing transitions per table row. The exact final two slices experiment improved accuracy with the 32-state beam, and the spectral experiment showed that the omitted four-fault-and-higher mass made little difference at the same transition count.

Increase transition fanout from 48 to 96 on uncertain shots while retaining a 32-state beam, the same three-fault local enumeration, and the exact terminal likelihood. Reuse the original 48-transition decode for the first pass. Mark a bit for retry when its combined normalized signed moment has magnitude at most 0.4 or its two directional signs disagree. Run the two 96-transition directions on marked shots, and replace only marked bits. The original parity fallback and ambiguity threshold apply to the resulting estimates.

Compare against `temporal_suffix_v1` on the original frozen data and then on 250,000 new simulated shots per basis with seeds `2026111361` and `2026111362` if the frozen gate passes. Require positive paired 95% lower bounds on both samples, a local five-run median at or below 12 seconds for 200,000 shots, peak inference RSS at or below 2 GiB, and a build within 20 minutes and 5 GiB. Require byte-identical replay and a matching code hand-in archive before updating the best-save pointer.

## Results

Frozen LER fell from 0.03405625 to 0.03392625. The candidate corrected 104 net bit errors, 54 in X and 50 in Z. The paired 95% gain interval was [0.00003537, 0.00022463]. The five-run median was 9.75 seconds with 1.59 GiB peak RSS. Building took 36.1 seconds with peak RSS of 1,057 MiB. All 38 tests passed. [Frozen validation](../artifacts/expanded/validation.json)

On 500,000 fresh Stim shots the candidate corrected 235 net bit errors. The paired gain was 0.00011750 with a 95% interval of [0.00005732, 0.00017768]. The candidate's LER on that sample was 0.03392250. [Independent confirmation](../artifacts/expanded/confirmation.json)

The submission was reproduced byte for byte in a fresh process at a different batch size. The extracted code hand-in archive also reproduced all predictions and IDs. [Verification](../artifacts/expanded/verification.json)
