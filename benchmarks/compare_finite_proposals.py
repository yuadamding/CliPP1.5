"""Differential finite-proposal audits and bounded-work component measurements.

Compare the unchanged 72ba3ed package with the current source using retained
audits and clipping/plateau workloads. This harness measures finite audits,
not end-to-end fitting throughput. Use a fresh output directory for each run.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import tracemalloc

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"

import numpy as np  # noqa: E402

sys.path[:0] = [str(Path(__file__).resolve().parent),
               str(Path(__file__).resolve().parents[1] / "src")]
from clipp1d import api, solver  # noqa: E402
from clipp1d.policy import Policy  # noqa: E402
from clipp1d.types import CountModel  # noqa: E402
from compare_reuse import array_hash, audit_cases, load_baseline  # noqa: E402


BASELINE_SHA256 = "30a019eba2c99dbfae23b26200164893ab6ac54d82d2ef60a1dec2ce4f8e4055"
SCENARIOS = ("duplicate_lower_clipping", "all_singletons_rejected", "first_singleton_accept",
             "late_singleton_accept", "fused_block_accept", "duplicate_clipped_offsets")


def model_arrays(model):
    return tuple(getattr(model, key) for key in ("alt", "ref", "lower", "upper", "slope", "log_prior", "valid"))


def extended_case(n, scenario):
    """Real clipping contexts with late/absent singleton and fused-block moves."""
    eps = 1e-6
    slopes = np.broadcast_to([.5, .5000005, 2., 2.], (n, 4)).copy()
    alt, ref = np.full(n, 10.), np.full(n, 90.)
    lower, upper = np.full(n, eps), np.ones(n)
    x = np.full(n, .8)
    x[[0, -1]] = 1.
    caps = np.full(n-1, 1e8)
    if scenario == "duplicate_lower_clipping":
        x[1:-1] = eps
    elif scenario == "duplicate_clipped_offsets":
        slopes[:]=[2.,2.000001,2.000002,2.000003]
        lower[1:-1],upper[1:-1]=.499998,.500002
        x[1:-1]=.500002
        caps.fill(1e12)
    elif scenario == "first_singleton_accept":
        caps.fill(0.)
    elif scenario == "late_singleton_accept":
        mid = n // 2
        caps[mid-1:mid+1] = 0.
    elif scenario == "fused_block_accept":
        caps[[0, -1]] = 0.
    elif scenario != "all_singletons_rejected":
        raise ValueError(f"Unknown finite-proposal scenario: {scenario}")
    valid = np.ones_like(slopes, dtype=bool)
    model = CountModel(tuple(f"synthetic-{i}" for i in range(n)), alt, ref, lower, upper,
                       slopes, np.full_like(slopes, -np.log(4)), valid, eps)
    return model, x, np.zeros(n-1), caps


def same_result(first, second):
    return bool(first[0] == second[0] and
                ((first[1] is None and second[1] is None) or
                 (first[1] is not None and second[1] is not None and np.array_equal(first[1], second[1]))))


def proposal_key(start, stop, values):
    array = np.broadcast_to(np.asarray(values, dtype=float), (stop-start,))
    return int(start), int(stop), array.tobytes().hex()


def add_incident_tv(likelihood_delta, proposal, x, caps):
    """Literal old local_interval_delta arithmetic, independent of revised helper."""
    start, stop = proposal.start, proposal.stop
    old = x[start:stop]
    new = np.broadcast_to(np.asarray(proposal.value, dtype=float), old.shape)
    delta = float(likelihood_delta)
    delta += float(np.dot(caps[start:stop-1], np.abs(np.diff(new))-np.abs(np.diff(old))))
    if start:
        delta += caps[start-1]*(abs(new[0]-x[start-1])-abs(old[0]-x[start-1]))
    if stop < len(x):
        delta += caps[stop-1]*(abs(x[stop]-new[-1])-abs(x[stop]-old[-1]))
    return float(delta)


@contextmanager
def observe(module, x, caps):
    """Separate untimed call/allocation instrumentation from performance runs."""
    likelihood = importlib.import_module(module.__package__ + ".model")
    originals = []
    record = dict(evaluate_calls=0, posterior_evaluation_calls=0, logsumexp_calls=0,
                  total_candidate_kernel_elements=0, maximum_candidate_kernel_elements=0,
                  maximum_candidate_kernel_shape=[], local_delta_calls=0, proposals=[],
                  generated_proposals=[], proposal_instrumentation="legacy local deltas",
                  batch_calls=0, maximum_batch_proposals=0, maximum_batch_rows_before_deduplication=0,
                  evaluated_unique_proposals=0, loss_only_calls=0, loss_only_rows=0,
                  maximum_loss_only_rows=0, proposal_cache_hits=0, proposal_cache_misses=0,
                  maximum_proposal_cache_entries=0)

    def patch(owner, name, wrapper):
        original = getattr(owner, name)
        originals.append((owner, name, original))
        setattr(owner, name, wrapper(original))

    def evaluate_wrapper(original):
        def wrapped(*args, **kwargs):
            record["evaluate_calls"] += 1
            record["posterior_evaluation_calls"] += 1
            return original(*args, **kwargs)
        return wrapped

    def kernel_wrapper(original):
        def wrapped(values, *args, **kwargs):
            record["logsumexp_calls"] += 1
            record["total_candidate_kernel_elements"] += values.size
            if values.size > record["maximum_candidate_kernel_elements"]:
                record["maximum_candidate_kernel_elements"] = values.size
                record["maximum_candidate_kernel_shape"] = list(values.shape)
            return original(values, *args, **kwargs)
        return wrapped

    def delta_wrapper(original):
        def wrapped(block, x, caps, start, stop, values, *args, **kwargs):
            result = original(block, x, caps, start, stop, values, *args, **kwargs)
            record["local_delta_calls"] += 1
            record["proposals"].append(dict(key=proposal_key(start, stop, values), delta_hex=float(result).hex()))
            return result
        return wrapped

    def generation_wrapper(original):
        def wrapped(*args, **kwargs):
            for proposal in original(*args, **kwargs):
                record["generated_proposals"].append(proposal_key(proposal.start,proposal.stop,proposal.value))
                yield proposal
        return wrapped

    def consumption_wrapper(original):
        def wrapped(*args, **kwargs):
            for proposal, likelihood_delta in original(*args, **kwargs):
                delta = add_incident_tv(likelihood_delta,proposal,x,caps)
                record["proposals"].append(dict(key=proposal_key(proposal.start,proposal.stop,proposal.value),
                                               delta_hex=delta.hex()))
                yield proposal, likelihood_delta
        return wrapped

    def batch_wrapper(original):
        def wrapped(context,batch):
            record["batch_calls"] += 1
            record["maximum_batch_proposals"] = max(record["maximum_batch_proposals"],len(batch))
            record["maximum_batch_rows_before_deduplication"] = max(
                record["maximum_batch_rows_before_deduplication"],sum(p.stop-p.start for p in batch))
            return original(context,batch)
        return wrapped

    def loss_rows_wrapper(original):
        def wrapped(model,rows,phi):
            record["loss_only_calls"] += 1
            record["loss_only_rows"] += len(rows)
            record["maximum_loss_only_rows"] = max(record["maximum_loss_only_rows"],len(rows))
            return original(model,rows,phi)
        return wrapped

    def cache_get_wrapper(original):
        def wrapped(memo,key):
            result=original(memo,key)
            record["proposal_cache_hits" if result is not None else "proposal_cache_misses"] += 1
            return result
        return wrapped

    def cache_remember_wrapper(original):
        def wrapped(memo,key,value):
            record["evaluated_unique_proposals"] += 1
            result=original(memo,key,value)
            record["maximum_proposal_cache_entries"] = max(record["maximum_proposal_cache_entries"],len(memo._values))
            return result
        return wrapped

    patch(module, "evaluate", evaluate_wrapper)
    patch(likelihood, "evaluate", evaluate_wrapper)
    patch(likelihood, "logsumexp", kernel_wrapper)
    patch(module, "local_interval_delta", delta_wrapper)
    if hasattr(module,"_finite_proposals"):
        proposals=importlib.import_module(module.__package__+".proposals")
        patch(module,"_finite_proposals",generation_wrapper)
        patch(module,"iter_proposal_deltas",consumption_wrapper)
        patch(proposals,"_batch_deltas",batch_wrapper)
        patch(proposals,"loss_at_rows",loss_rows_wrapper)
        patch(proposals.ProposalLossMemo,"get",cache_get_wrapper)
        patch(proposals.ProposalLossMemo,"remember",cache_remember_wrapper)
        record["proposal_instrumentation"] = "ordered generator and consumed batched deltas"
        record["configured_limits"] = dict(batch_proposals=proposals.MAX_BATCH_PROPOSALS,
                                          batch_rows=proposals.MAX_BATCH_ROWS,cache_entries=proposals.MAX_CACHE_ENTRIES)
    try:
        yield record
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def inputs(model, x, caps, anchor):
    lower, upper = model.lower.copy(), model.upper.copy()
    lower[anchor] = upper[anchor] = 1.
    return model, x, caps, lower, upper


def invoke(module, arrays, context, *, finite_only, force=True,direction=None):
    if finite_only:
        return module._finite_interval_check(*arrays, context, direction)
    return module._kink_check(*arrays, Policy(), context=context, force_intervals=force)


def compare_case(baseline, label, index, model, x, dual, caps, anchor, *, finite_only=False, force=True,
                 contexts=None,direction=None):
    arrays = inputs(model, x, caps, anchor)
    if contexts is None:
        contexts = {"baseline": baseline.prepare_audit(model, x), "revised": solver.prepare_audit(model, x)}
    results, counts = {}, {}
    for name, module in (("baseline", baseline), ("revised", solver)):
        with observe(module,x,caps) as count:
            results[name] = invoke(module, arrays, contexts[name], finite_only=finite_only, force=force,direction=direction)
        counts[name] = count
    lower, upper = arrays[-2:]
    first = baseline.stationarity(model, x, dual, lower, upper, caps, context=contexts["baseline"])
    second = solver.stationarity(model, x, dual, lower, upper, caps, context=contexts["revised"])
    old_trace, new_trace = counts["baseline"]["proposals"], counts["revised"]["proposals"]
    expected_order = [row["key"] for row in old_trace]
    generated = counts["revised"]["generated_proposals"]
    same_generation_prefix = (generated[:len(expected_order)] == expected_order if generated else
                              [row["key"] for row in new_trace] == expected_order)
    same_order = [row["key"] for row in new_trace] == expected_order
    same_deltas = old_trace == new_trace
    revised=counts["revised"]
    limits=revised.get("configured_limits")
    bounded=(limits is None or (
        revised["maximum_batch_proposals"] <= limits["batch_proposals"] and
        revised["maximum_batch_rows_before_deduplication"] <= max(len(x),limits["batch_rows"]) and
        revised["maximum_loss_only_rows"] <= max(len(x),limits["batch_rows"]) and
        revised["maximum_proposal_cache_entries"] <= limits["cache_entries"]))
    for count in counts.values():
        trace = count.pop("proposals")
        generated = count.pop("generated_proposals")
        keys = [row["key"] for row in trace]
        count.update(consumed_candidates=len(trace), generated_candidates=len(generated),
                     duplicate_consumed_proposals=len(keys)-len(set(keys)),
                     consumed_order_sha256=hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
                     consumed_delta_sha256=hashlib.sha256(json.dumps(trace).encode()).hexdigest(),
                     last_consumed_proposal=trace[-1] if trace else None,
                     generated_order_sha256=hashlib.sha256(json.dumps(generated).encode()).hexdigest())
    return dict(scenario=label, index=index, mutations=len(x), anchor=anchor, finite_only=finite_only,
                injected_direction=direction,
                force_intervals=force, input_sha256=array_hash((*model_arrays(model), x, dual, caps, lower, upper)),
                same_result=same_result(results["baseline"], results["revised"]),
                same_consumed_candidate_order=same_order, same_generated_prefix=same_generation_prefix,
                same_consumed_candidate_deltas_bitwise=same_deltas,
                bounded_allocation_inputs=bounded,
                same_stationarity=first == second and first[0].hex() == second[0].hex(),
                baseline_kink_okay=results["baseline"][0], revised_kink_okay=results["revised"][0],
                baseline_restart=results["baseline"][1] is not None,
                revised_restart=results["revised"][1] is not None, measurements=counts)


def measured_memory(function):
    tracemalloc.start()
    tracemalloc.reset_peak()
    before, _ = tracemalloc.get_traced_memory()
    result = function()
    live, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, dict(new_peak_bytes=max(0, peak-before), live_at_return_bytes=max(0, live-before),
                        scope="Separate tracemalloc pass; excludes preexisting model/context arrays and timing samples")


def component_times(baseline, arrays, repeats):
    """Contexts are prepared outside each timer; repeats never share memoized work."""
    modules={"baseline":baseline,"revised":solver}
    samples={name:[] for name in modules}
    results={}
    for module in modules.values():
        context=module.prepare_audit(arrays[0],arrays[1])
        invoke(module,arrays,context,finite_only=True)
    for repeat in range(repeats):
        for name in (("baseline","revised") if repeat%2==0 else ("revised","baseline")):
            module=modules[name]
            context=module.prepare_audit(arrays[0],arrays[1])
            started=perf_counter()
            results[name]=invoke(module,arrays,context,finite_only=True)
            samples[name].append(perf_counter()-started)
    summary={name:dict(seconds=values,median_seconds=float(np.median(values)),minimum_seconds=min(values),
                       maximum_seconds=max(values),population_standard_deviation_seconds=float(np.std(values)))
             for name,values in samples.items()}
    summary["median_speed_ratio"]=summary["baseline"]["median_seconds"]/summary["revised"]["median_seconds"]
    summary["context_scope"]="Fresh audit context outside every timer; no memoized proposal values shared across repeats"
    return summary,results


def run(args):
    if args.outdir.exists():
        raise FileExistsError("Output directory must be fresh")
    allowed = sorted(os.sched_getaffinity(0))
    if args.cpu not in allowed:
        raise ValueError("Requested CPU is outside allowed affinity")
    os.sched_setaffinity(0, {args.cpu})
    baseline_api, baseline, baseline_source = load_baseline(args.baseline_package, args.baseline_sha256)
    source = api.source_provenance()
    args.outdir.mkdir(parents=True)
    report = dict(schema="clipp1d.finite_proposal_comparison.v1", started_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Retained and extended audit decisions, finite-proposal component calls, allocation and timing; no full fits",
                  source=source, baseline_source=baseline_source, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), allowed_cpus_before=allowed,
                  thread_environment={key:os.environ[key] for key in THREAD_VARIABLES},
                  repeated_pairs=args.repeats, warmups_per_arm=1,
                  retained_inventory="compare_reuse.audit_cases: seed8821, 100 models, first and last occupied-one anchors",
                  audit_cases=[], component_timings=[])
    for index, label, model, x, dual, caps in audit_cases():
        contexts={"baseline":baseline.prepare_audit(model,x),"revised":solver.prepare_audit(model,x)}
        for anchor in (0, len(x)-1):
            report["audit_cases"].append(compare_case(baseline,label,index,model,x,dual,caps,anchor,
                                                        force=index%3==0,contexts=contexts))
    # Differential boundary probes deliberately inject an interval direction;
    # they test finite acceptance arithmetic, not interval-descent validity.
    model,x,dual,caps=extended_case(8,"first_singleton_accept")
    context=baseline.prepare_audit(model,x)
    start,stop,step=1,7,1e-3
    value=x[start:stop]-step
    delta=baseline.local_interval_delta(model.subset(np.arange(start,stop)),x,caps,start,stop,value,
                                        context.losses[start:stop])
    threshold=delta/(1e-4*step)
    for index,derivative in enumerate((np.nextafter(threshold,-np.inf),threshold,np.nextafter(threshold,np.inf),-1e100)):
        direction=(start,stop,-1,float(derivative),abs(float(derivative)))
        report["audit_cases"].append(compare_case(baseline,"armijo_roundoff_boundary",index,model,x,dual,caps,0,
                                                    finite_only=True,direction=direction))
    for n in args.sizes:
        for label in args.scenarios:
            model,x,dual,caps = extended_case(n,label)
            contexts={"baseline":baseline.prepare_audit(model,x),"revised":solver.prepare_audit(model,x)}
            for anchor in (0,n-1):
                report["audit_cases"].append(compare_case(baseline,label,n,model,x,dual,caps,anchor,
                                                            finite_only=True,contexts=contexts))
            arrays = inputs(model,x,caps,0)
            times, results = component_times(baseline,arrays,args.repeats)
            memory = {}
            for name,module in (("baseline",baseline),("revised",solver)):
                context=module.prepare_audit(model,x)
                memory_result,memory[name] = measured_memory(lambda module=module,context=context:
                                                            invoke(module,arrays,context,finite_only=True))
                assert same_result(memory_result,results[name])
            report["component_timings"].append(dict(scenario=label,mutations=n,**times,memory=memory,
                                                     same_result=same_result(results["baseline"],results["revised"])))
        print(json.dumps(dict(stage="finite_components",mutations=n)),flush=True)
    report["source_unchanged"] = api.source_provenance()["source_sha256"] == source["source_sha256"]
    report["baseline_source_unchanged"] = baseline_api.source_provenance()["source_sha256"] == baseline_source["source_sha256"]
    passed = (report["source_unchanged"] and report["baseline_source_unchanged"] and
              all(row["bounded_allocation_inputs"] and all(value for key,value in row.items() if key.startswith("same_"))
                  for row in report["audit_cases"]) and
              all(row["same_result"] for row in report["component_timings"]))
    report["status"] = "equivalent" if passed else "unresolved"
    with (args.outdir/"comparison.json").open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False,default=lambda value:value.item() if isinstance(value,np.generic) else str(value))
        stream.write("\n")
    print(json.dumps(dict(status=report["status"],audits=len(report["audit_cases"]),
                         mismatches=sum(not(row["same_result"] and row["same_stationarity"]) for row in report["audit_cases"]),
                         source_sha256=source["source_sha256"])),flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-package",type=Path,required=True)
    parser.add_argument("--baseline-sha256",default=BASELINE_SHA256)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--cpu",type=int,default=2)
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--sizes",type=int,nargs="+",default=[100,1000,4000])
    parser.add_argument("--scenarios",choices=SCENARIOS,nargs="+",default=list(SCENARIOS))
    args=parser.parse_args()
    if args.repeats < 3 or any(n < 6 for n in args.sizes):
        parser.error("At least three paired repeats and sizes of at least six are required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
