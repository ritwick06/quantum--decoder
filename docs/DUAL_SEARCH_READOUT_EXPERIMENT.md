# Readout using both search widths

The earlier calibrated readout used only the 48-transition estimates. The accepted expanded decoder also computes 96-transition estimates for roughly 8% of shots. Changes in signed moments and retained mass between the searches can reveal sensitivity to truncation.

Keep the expanded decoder's eligibility and search unchanged. On eligible bits, fit a regularized logistic readout using signed moments from both widths, their changes, relative retained mass, and detector counts. Fit per basis and bit with L2 coefficient 0.002 on two million new Stim shots per basis, in batches of 100,000 with seeds `2026111411` and `2026111412`. Apply the readout only where its absolute logit exceeds 0.08. Keep the original expanded decision for the remaining bits.

Compare once with `temporal_expanded_v1` on the frozen sample and require a positive paired 95% lower bound. Measure five-run local median and apply the supplied runtime penalty as a sensitivity calculation. Require a lower penalized local score, 2 GiB peak inference RSS with the incumbent's tables released, and training within 20 minutes and 5 GiB. Confirm any passing accuracy gain on 250,000 fresh shots per basis with seeds `2026111421` and `2026111422`. Require a positive paired lower bound and byte-identical CSV replay and hand-in verification before promotion.

## Result

Frozen LER was 0.03392000 versus 0.03392625 for the expanded decoder, a net gain of five bit errors across 800,000 predictions. X incurred eight more errors and Z corrected thirteen. The paired 95% interval for the absolute LER gain was [−0.00002689, 0.00003939]. The five-run local median was 13.03 seconds, with peak inference RSS of 1.16 GiB. All 42 tests passed. The candidate failed the accuracy gate and did not proceed to independent confirmation or CSV generation. [Frozen report](../artifacts/dual_readout/validation.json)
