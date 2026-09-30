# CN-first ten-case multiplicity and partition investigation

Evidence date: September 29, 2026, America/Chicago (CDT). This follows the user's request to deeply investigate the ten scenario-diverse CliPP1.5/PyClone-VI disagreements.

[Full investigation](../../results/experimental-full29-20260928-v3/performance-cnfirst4k-20260929-v1/deep-10case-investigation-v1/REPORT.md) · [Mechanism figures](../../results/experimental-full29-20260928-v3/performance-cnfirst4k-20260929-v1/deep-10case-investigation-v1/CNfirst10_multiplicity_mechanisms.pdf) · [Original case selection](CNFIRST_DIVERSE_PYCLONE_CASES.md)

The analysis covers 4,888 common mutations in ten deliberately selected cases. Exact counts/CN/purity agree between methods. Independent equations reproduced all 493,100 PyClone likelihood entries over the full 4,931 fitted rows within 1.39e-12. Saved-parameter CliPP replay reproduced 340 candidate scores within 1.17e-10 and all ten guard decisions. No optimization, production-source edit, campaign change or CUDA qualification occurred; truth-conditioned grid calculations are diagnostic only.

## Findings to retain

- PyClone-VI 0.1.6 marginalizes genotype scenarios and exports no native multiplicity call. Multiplicity support is 1..major CN. When total CN differs from normal CN, it has a second m1 history and equal scenario priors consequently favor m1 in aggregate. The m>1 scenarios give mutation-negative cancer cells normal CN; the extra m1 scenario gives them tumor CN. This differs from CN-first's fixed tumor CN and uniform integer-m generation.
- That CN-history mismatch materially changes the likelihood of amplified subclonal mutations. At known truth CCF in rep57, changing only reference-cancer CN improves summed log likelihood by 178.930 on 149 subclonal CNA loci, versus another 7.267 from changing to uniform integer-m prior. This is a sequential component diagnostic, not an estimated causal share or predicted refit improvement.
- PyClone's rep57 and rep58 false-clonal assignments include true-m3 loci with true CCF near one third. These have nearly clonal m1 read-count explanations. Reconstructed multiplicity probabilities must distinguish conditioning on the assigned cluster from averaging over uncertain cluster membership; neither is a native PyClone call.
- CliPP's structural guard preserves nine original partitions. Already-computed converged same-K mixtures improve matched ARI in rep122 0.0488→0.6092, rep59 0.6226→0.8195 and rep143 0.3963→0.5544, but are ineligible. The guard reference is the best same-K uniform mixture itself, so changing eligibility alone does not fix the strict reference comparison.
- These alternatives still trail PyClone's ARI in all three cases. In rep59 and rep143, improved ARI/CCF coincides with lower matched CNA multiplicity macro-F1. The guard protects small ARI advantages in rep139 and rep79. Do not infer that removing it preserves full CN-first accuracy.
- Rep2 remains K1 even without the guard: the saved K2 twice-log-likelihood gain 7.4861 is smaller than its 11.9220 parameter penalty. Rep88 passes a K1→K2 birth, but hard sMF 0.0569 differs sharply from fitted low-component mass 0.1699 and truth 0.1860.
- Rep23's PyClone false split is mostly diploid, between adjacent near-one CCF grid positions with weak assignments. Rep58's exact PyClone sMF results from 15 false-clonal and 15 false-subclonal calls cancelling, despite substantially worse ARI. Correct K or sMF alone does not establish correct clustering.

Follow-up selection or emission changes require one truth-blind policy and the existing joint CN-first4K/SimClone1000/PhylogicNDT500 acceptance contract. This selected panel cannot serve as an independent acceptance set. Preserve unconstrained fitting and closest-to-one label0; the investigation provides no basis to restore a clonal fitting constraint.

The full report links three independent audits, per-mutation data, posthoc genotype posteriors, all candidates, model equations and original source bindings. [Machine record](CNFIRST_TEN_CASE_INVESTIGATION.json) binds the local report and completion inventory.
