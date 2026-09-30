# Repository numerical recovery qualification

The requested improvement is to CliPP1.5 itself. P and R1–R5 are regression
inputs, not algorithm parameters. Implementation is in the shared production
package; no saved six-case partition, failed-QP tensor, case name, mutation ID,
or truth label is consumed by the new solver or search.

Baseline: commit `fc9d349761de68d8c8771fe5d92137eabfc6e784` plus the existing
working-tree changes, preserved before this work. Its 44-file Python package
fingerprint is
`fed985880efc849fe90d10b73cc5b686e5e157b576f36ab46271bb91848dc284`.
The baseline source, test inventory, complete prior diff and file hashes are
under `results/repo-solver-improvement-20260930T163352Z/`.
`results/CURRENT_REPO_SOLVER_IMPROVEMENT.json` is the live operations pointer.

## Changes under qualification

- Bounded terminal QP precision recovery with original objective, gap and KKT
  admission; independent eager and compiled final certificates.
- Tighter scalar interval bounds from conservative posterior-mass limits on
  mixture score variance, retaining the original fallback bound and budget.
- An immutable affine scalar problem for original-box raw precision recovery.
- Optional refit-aware relocation of a mutation between existing groups,
  scored after joint refitting and independently validating the full child.
- Shared solver accounting, ancestry, publication validation and CLI exposure.

The statistical likelihood, CN/multiplicity support, complete-graph recipe,
original boxes, score, extraction tolerance and numerical admission tolerances
are unchanged. Ordinary QP and outer limits remain 20,000 and 150. Additional
precision work has explicit ceilings and is included even when a start fails.
The relocation search is off by default.

## Evidence requirements

1. Freeze the actual changed repository package, tests and worker with full
   file hashes and the dirty diff. Do not substitute a historical run bundle.
2. Run generic numerical oracles and relevant regression/publication tests on
   allocated A100 CUDA using the bound `ml1` interpreter. Distinguish CUDA
   execution from CPU reference tests executed in the same worker.
3. Fit each regression input from its canonical input with fresh baseline and
   candidate processes. Reconcile model/input identities, graph construction,
   actual pilot/graph differences, selected labels, raw and refitted CCFs,
   scores, original certificates, output schema, search failures, work and time.
4. Keep raw-estimator comparisons separate from optional partition estimates.
   No known truth is available for these six inputs: lower scores, smaller K
   or fewer unresolved searches do not demonstrate improved biological accuracy.
5. Retain failures and partial coverage. Captured-state exploratory passes do
   not establish original-start or full-fit qualification of the current source.
   Broader scientific adoption still requires the three-cohort acceptance
   contract in `docs/THREE_COHORT_ACCEPTANCE.md`.

## Qualification snapshot — September 30, 2026, 1:06 PM CDT

The frozen candidate's 48-file production package fingerprint is
`8bd9fc2abc7bcf50296e0c1006c78a231a01aee16e28a70ff2394da2d0e4ec94`.
The publication audit matched every production Python file to that frozen
source. Ruff, Python syntax, JSON parsing and whitespace checks passed locally;
these static checks are separate from numerical qualification.

On an allocated A100, all 830 collected tests passed, with zero failures,
errors or skips. This inventory includes CUDA numerical tests, CPU references
and metadata controls; it is not 830 independent CUDA numerical oracles.
Two fresh synthetic baseline/candidate pairs also passed. Their primary
arrays and scores matched; one optional partition estimate improved its score.

The outer qualification wrapper subsequently failed because a step receipt
collided with the aggregate receipt name. Its original failure is preserved.
Independent recovery revalidated the bound source, environment, complete test
inventory, scientific outputs and their hashes. The recovered scientific gate
is `qualification-recovery/SCIENTIFIC_GATE.json` beneath the evidence root
above, with SHA-256
`21aa8198257bd2515ede7ed760e368978d118a95dba873464c65a6c21d43fc3a`.
This establishes the test and synthetic-pair evidence without relabeling the
failed outer execution as successful.

At the snapshot time, six fresh baseline/candidate comparisons for P and
R1–R5 were running concurrently on six A100 workers, with zero completed
pairs. Each comparison starts from its original input; saved memberships and
solver states are not inputs. Full real-input qualification and the broader
three-cohort adoption gate remain pending. Optional relocation stays off by
default.
