# Remaining Experimental cases and large-seed memory development

This is launch evidence, not a new performance or CUDA qualification claim.
Follow `results/CURRENT_MIXTURE_STUDY.json` for changing job status.

The [complete CN-first4K performance report](CNFIRST_PERFORMANCE.md) evaluates
all 4,000 tumors as of September 29, 2026, 10:28 a.m. CDT. CliPP1.5 leads the
three external methods in mean ARI, sMF CCC and CCF/sMF errors; PhylogicNDT
recovers the exact cluster count more often. This full-cohort result supersedes
partial CN-first accuracy summaries, while SimClone and Phylogic coverage remain
separate. No fitting or scheduling settings changed during this evaluation.

The [September 29 two-cohort comparison](TWOCOHORT_PERFORMANCE_20260929.md)
includes all 400 completed SimClone and 71 completed PhylogicNDT tumors at
10:43 a.m. CDT. CliPP1.5 has higher mean ARI and lower CCF error on both matched
subsets; PyClone-VI still has higher sMF CCC on both and better Phylogic exact
cluster-count recovery. These are partial-cohort development results.

Following the user's request to rush expansion, the September 28, 2026,
**5:06 p.m. CDT** snapshot verified all ten A100 and four H100 workers
Running/Ready, with fifteen LSF jobs submitted and Pending. This includes
fourteen ordinary jobs and memory-development job 77501830. There were
224 new validated cases (3,321 including imports), with no reported case
failures. See [full-expansion evidence](FULL_EXPANSION.json).

The replacement scalar controller preserves the original qualification job
77501680. Additional existing-seed jobs reuse the prior allocated A40 gate
after exact scientific-source and compiler-setting comparison. Fresh-seed
LSF tasks still require the current qualification; the memory experiment
retains its own gate. That expansion did not cancel a healthy GPU worker or
Pending job.

The user's later request to resubmit all fifteen Pending LSF jobs was completed
on September 28, 2026, at **9:46 p.m. CDT**. The transaction preserved job IDs,
source, inputs, worker commands and resource requests. Both verified scalar
controllers were paused during the hold/cancel/requeue transition and resumed
after all fifteen jobs were released. LSF accepted all fifteen `btop` requests
within the user's own same-priority ordering. The final snapshot still showed
fifteen Pending jobs, both controllers alive and ten A100 plus four H100 workers
Running/Ready. Requeue does not bypass scheduler policy or create capacity;
the ordinary jobs had been waiting for A40 resources and the managed test for
its 384-GB host-memory reservation. See [requeue evidence](LSF_REQUEUE.json).

The user authorized all remaining cases from three cohorts, with ten A100,
four H100 and fifteen LSF GPU slots in total. The previous 3,097 validated
outputs remain imported with their original source and result-index ancestry.

| Cohort | Remaining cases | Ordinary GPU queue | Managed-memory development |
| --- | ---: | ---: | ---: |
| CN-first4K | 1,339 | 1,339 | 0 |
| SimClone1000 | 371 | 253 | 118 |
| PhylogicNDT500 | 449 | 449 | 0 |
| Total | 2,159 | 2,041 | 118 |

The ordinary queue uses published `fc9d349` mathematical source. A separate,
fully inventoried execution overlay adds host-backed PyTorch CUDA allocation
for the 118 oversized complete-graph seeds. It preserves likelihood, graph,
search, tolerances, score, structural guard and unconstrained CCF fitting.
It changes storage admission and therefore has its own source fingerprint.

The managed branch reserves one LSF slot and 384 GB host RAM. The ordinary
scalar controller is limited to fourteen. Automatic transfer of all fifteen
scalar slots requires source-bound CUDA parity/capacity qualification and
terminal reconciliation of every ordinary scalar job. Each scalar job still
owns one case and one GPU.

As of September 28, 2026, 4:44 p.m. CDT, the replacement A100 and H100
qualification workers were running, each with verified hardware. Both passed
the three CUDA tests and three paired publication checks; fresh complete-graph
seed qualification was still running. Two LSF jobs were Pending, including
managed-memory job **77501830**. No new cohort cases had been validated.

Local resource, ownership, seeding and API checks passed. The allocator built
with the bound compiler locally and remotely. Neither compilation nor those
CPU checks qualify CUDA managed memory. Job 77501830 must compare ordinary
and managed complete fits, then execute compiled float64 arithmetic beyond
physical VRAM before its first large case. The first large completed fit is
separate evidence; memory oversubscription may impose substantial paging cost.

The first two bootstrap attempts retained missing-fixture/import evidence.
The third payload was complete, but its Kubernetes manifest still referenced
the previous PVC subpath. `mount-recovery-v1` fixes that mount in a new
generation while preserving the valid scalar owner. Failed outputs and
receipts were not overwritten.

See [launch identities](LAUNCH.json) and the
[reusable operations guidance](../../benchmarks/OPERATIONS.md#remaining-three-cohort-campaign-and-large-seeds-september-28).
