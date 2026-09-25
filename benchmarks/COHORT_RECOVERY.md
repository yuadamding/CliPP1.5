# Cohort failure isolation and recovery

The production model remains the unconstrained, float64 complete-graph CUDA
pipeline. CPU benchmarks use the explicit eager tensor adapter. Operational
recovery does not change the likelihood, graph, search, tolerances, refit, score,
or the closest-to-one public cluster-zero labeling rule.

For the operational sequence, process-identity lessons, scheduler distinctions
and metric caveats, see [operations and reusable lessons](OPERATIONS.md). The
[dated recovery record](../validation/cohort-recovery-20260923/README.md) retains
qualification scope without treating historical allocations as live status.

`cohort_failures.py` separates these execution boundaries:

- Setup/input or output-validation failures stop new admissions. Already
  admitted independent fits drain before cleanup.
- Numerical qualification, resource, timeout, and unexpected fitting failures
  are explicit failed cases. An isolated fitting exception does not cancel
  unrelated cases or become a validated result.
- Three consecutive unexpected fitting failures in one A100 dispatcher stop
  admissions and drain the pool, retaining each traceback. Known scientific
  failures remain distinct from this systemic-error guard.
- A classified CUDA device-loss or initialization failure immediately retires
  the affected dispatcher without setting the pool-wide halt. Its unstarted
  queue entries remain unfinished. The per-case child's hardware probe is
  inside this classification boundary. Initial dispatcher/static/capacity
  probes, plan verification, imports and dispatcher staging can fail outside
  that boundary; diagnose them from phase and scheduler evidence. Kubernetes
  eviction or Job failure can still interrupt independent workers.

The lifecycle records scheduler failure and partial dispatcher completion
explicitly. A log timeout retains the Pod evidence. An unconfirmed delete is
recorded as cleanup pending, never as verified absence or permission to retry.
Recovery must wait for the exact old Job and UID-owned Pods to disappear;
exclude a diagnosed failed node in the next immutable manifest.
Pending cleanup starts no automatic watcher. Later reconciliation or a separate
bound continuation must prove absence; the no-clobber lifecycle is not a
generically repeatable launch/recovery command.

`run_a100_cohort.py` and `supervise_a100_cohort.py` are frozen payload templates.
They run only with a reviewed package-owned `common.py`, plan, inventory,
validator, source tree, exact input ZIP, and institutional manifest. They are
not ad hoc launch commands. They retain suspended creation, UID-bound
activation, one GPU per worker, capacity qualification, and exact cleanup.

Recovery can qualify a preserved successful tumor in a separate output root.
`qualify_cohort_recovery.py` compares graph identity, retained mutation IDs,
labels, pilot/raw/refitted CCFs, multiplicity calls, score, raw objective,
selected penalty, search completeness, and certificate semantics. The two
source identities remain explicit. This repeat is qualification evidence;
it must not be counted as a new cohort completion.

The CPU controller can borrow occupied cores from one identity-bound draining
parent. It refuses new parent admissions and keeps the combined 25-worker cap.
Retries use fresh no-clobber attempt directories and explicit input/source and
resource-budget bindings. A longer timeout does not alter solver iteration
budgets or qualify a failed fit.
The plan must explicitly encode remaining authorization in
`deferred_parent_retry_keys`; omitting that field permits all eligible isolated
parent failures to retry and does not independently count previous retries.
Do not assume CPU and A100 controllers accept identical terminal-state sets.

Preserve all earlier results and failed attempts. A successful result from an
older source retains that source identity; a mixed-source recovery must not be
reported as a homogeneous run. Maintain a single cohort case registry and an
explicit retry map, count one final outcome per case, and keep qualification
repeats and historical failures separate.
