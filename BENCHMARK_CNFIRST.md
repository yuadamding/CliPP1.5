# CN-first 4K GPU campaign

On September 25, 2026, the user stopped all CliPP1.5 OCCAMS work and reassigned
its ten A100 and four H100 workers to the new CN-first simulation cohort.
OCCAMS outputs and interrupted attempts are preserved. This campaign performs
fresh CliPP1.5 fits for all 4,000 tumors in
`/data/CliPP2/CliPPSim4K_CNfirst_20260924`, importing no prior fits.

On the original workstation, begin with
[`results/CURRENT_CNFIRST.json`](results/CURRENT_CNFIRST.json) and the linked
report/status command. These local receipts are not distributed with Git.
The input preparation and original gated launch evidence lives in
`results/cnfirst-pool14-20260925-v1`. The user subsequently instructed
**"skip qualification"**. The replacement launch and explicit override live in
`results/cnfirst-pool14-20260925-v2`; follow the current pointer for live state.

The two device-family Jobs consume one shared atomic-claim queue, with a total
cap of fourteen single-GPU workers. All inputs have 200–800 mutations. Cases
run in ascending retained-N-squared order. Under the explicit override, both
Jobs start at their full authorized parallelism without separate static,
capacity or real-fit qualification Jobs. Record the gate as **skipped**, not
passed. Each worker still verifies its device, source, environment, admitted
image, input hashes and actual mount access before claiming work. Normal
scientific and output validation remains unchanged. Record actual Ready
workers separately from caps.

Both device families had already passed capacity probes in the first attempt.
H100's real-fit gate completed during the transition and four cohort fits
started; all four were interrupted before publishing validated results and
are restarted in the new queue. The old Jobs, Pods and supervisors were
confirmed stopped before replacement. Their outputs and logs remain intact;
historical qualification evidence does not certify the replacement launch.

The fitting source fingerprint is
`62ec9d1480563cda8ce5c192aff94c6763a9978a2fd73ea53b85c1975ed21e09`;
the underlying base commit is `a8a3ef7aaaeda8dffa8ac5fccef1cc4b77aa71f5`.
The complete dirty source is frozen. Only the added simulation package differs
from the previous OCCAMS package; the fitting implementation is unchanged.
Use unconstrained complete-graph PyTorch CUDA in float64, major-CN cutoff 4,
and closest-to-one public label 0. An 18-hour fit allowance does not change
the numerical budgets, tolerances or score.

Canonical conversion preserves all 2,003,720 generated mutations, normal CN 2,
read counts, purity and major/minor CN. Exact-coordinate truth joins cover
every retained mutation. Truth is used only for evaluation. The validator
reports final-refit metrics, designated-label sMF and CNA-only multiplicity
metrics; incomplete searches remain labeled and failed fits remain failures.
Compare methods using explicitly matched retained populations: each method
can apply a different filter to the same generated cohort.

The existing LSF campaigns continue independently. Do not resume stopped
OCCAMS, Regional-CN or former local CliPP1.5 workers from older pointers.
