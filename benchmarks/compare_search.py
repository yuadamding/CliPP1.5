"""Compare two historical constrained searches, not current production inference.

This is a small-instance search-policy comparison, not a global-optimization
proof or a full model-selection benchmark. Each policy has its own continuation
state. Failed penalties retain their diagnostics and do not borrow the other
policy's state. Use a fresh output directory; completed pairs are written as
JSONL as they finish so an externally interrupted run retains partial evidence.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

# Fixed one-CPU/one-thread controls precede NumPy for command-line studies.
# Importing the module has no affinity or thread side effects.
if __name__ == "__main__":
    from benchmark_chain import configure_execution
    EXECUTION = configure_execution(cpu_count=1, threads=1)
else:
    EXECUTION = None

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clipp1d.api import source_provenance  # noqa: E402
from clipp1d.chain import build_chain, extract_blocks  # noqa: E402
from clipp1d.clonal import fit_fixed_lambda  # noqa: E402
from clipp1d.io import read_tumor  # noqa: E402
from clipp1d.model import compile_model  # noqa: E402
from clipp1d.policy import Policy  # noqa: E402
from clipp1d.report import write_json  # noqa: E402
from clipp1d.scalar import compute_pilot  # noqa: E402
from clipp1d.selection import penalty_reference  # noqa: E402
from clipp1d.types import NumericalQualificationError, PrimalWarmState  # noqa: E402
from reference_enumeration import fit_fixed_lambda as fit_enumerated  # noqa: E402
from simulate import simulate  # noqa: E402


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return float(value)
    return value.item() if isinstance(value, np.generic) else value


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_arm(function, model, chain, pilot, penalty, warm, policy):
    started = perf_counter()
    try:
        raw = function(model, chain, pilot, penalty, warm, policy)
    except NumericalQualificationError as exc:
        return dict(status="unresolved", seconds=perf_counter() - started,
                    error_type=type(exc).__name__, message=str(exc), diagnostics=exc.diagnostics), warm
    except Exception as exc:
        return dict(status="error", seconds=perf_counter() - started,
                    error_type=type(exc).__name__, message=str(exc),
                    diagnostics=getattr(exc, "diagnostics", {})), warm
    record = dict(status="qualified" if raw.qualified else "unresolved",
                  seconds=perf_counter() - started, raw_objective=raw.objective,
                  witness_index=int(raw.witness), witness_mutation_id=model.mutation_ids[chain.order[raw.witness]],
                  chain_phi=raw.x.tolist(), partition=list(extract_blocks(raw.x, policy.fusion_tol)),
                  diagnostics=raw.diagnostics)
    return record, PrimalWarmState(raw.x, chain.fingerprint) if raw.qualified else warm


def compare_case(directory, size, seed, scenario, factors, policy, progress):
    source, truth = simulate(directory / "input", size, seed, ambiguous=scenario == "mixture")
    model = compile_model(read_tumor(source), policy)
    pilot = compute_pilot(model, policy)
    chain = build_chain(pilot, model.mutation_ids)
    reference_penalty = penalty_reference(model, chain, pilot)
    setup = dict(mutations=size, seed=seed, scenario=scenario,
                 input_sha256=file_hash(source), truth_sha256=file_hash(truth),
                 chain_sha256=chain.fingerprint, chain_mutation_ids=[model.mutation_ids[i] for i in chain.order],
                 reference_penalty=reference_penalty, pilot_gaps=pilot.gaps.tolist())
    arms = {"common_surrogate": fit_fixed_lambda, "independent_enumeration": fit_enumerated}
    warm = dict.fromkeys(arms)
    records = []
    for factor in factors:
        penalty = reference_penalty * factor
        row = dict(**setup, penalty_factor=factor, lambda_value=penalty)
        for name, function in arms.items():
            row[name], warm[name] = fit_arm(function, model, chain, pilot, penalty, warm[name], policy)
        actual, expected = row["common_surrogate"], row["independent_enumeration"]
        if actual["status"] == expected["status"] == "qualified":
            row["comparison"] = dict(
                objective_difference=actual["raw_objective"] - expected["raw_objective"],
                max_abs_phi_difference=float(np.max(np.abs(np.array(actual["chain_phi"]) - expected["chain_phi"]))),
                identical_partition=actual["partition"] == expected["partition"],
                common_search_complete=actual["diagnostics"].get("search_complete", False),
                reference_witness_search_complete=expected["diagnostics"].get("witness_search_complete", False),
                global_optimality_proven=False,
            )
        else:
            row["comparison"] = None
        records.append(row)
        with progress.open("a") as handle:
            handle.write(json.dumps(clean(row), sort_keys=True, allow_nan=False) + "\n")
        print(f"{scenario} n={size} seed={seed} factor={factor:g}: "
              f"shared={actual['status']}, reference={expected['status']}", flush=True)
    write_json(directory / "comparison.json", clean(dict(setup=setup, results=records)))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23])
    parser.add_argument("--scenarios", nargs="+", choices=("convex", "mixture"), default=["convex", "mixture"])
    parser.add_argument("--penalty-factors", type=float, nargs="+", default=[0., .01, 1., 100.])
    args = parser.parse_args()
    if (any(size < 1 for size in args.sizes) or
            any(not np.isfinite(factor) or factor < 0 for factor in args.penalty_factors)):
        parser.error("sizes must be positive and penalty factors finite and nonnegative")
    for name in ("sizes", "seeds", "scenarios", "penalty_factors"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"duplicate {name} are not allowed")
    if args.outdir.exists():
        raise FileExistsError("Use a new comparison directory")
    args.outdir.mkdir(parents=True)
    policy = Policy()
    metadata = dict(schema="clipp1d.search_policy_comparison.v1",
                    inference_scope="historical constrained policies; not current production inference",
                    clonal_constraint=True,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    provenance=source_provenance(), policy=asdict(policy), execution=EXECUTION,
                    benchmark_sha256=file_hash(__file__),
                    reference_sha256=file_hash(fit_enumerated.__code__.co_filename),
                    simulator_sha256=file_hash(simulate.__code__.co_filename),
                    sizes=args.sizes, seeds=args.seeds, scenarios=args.scenarios,
                    penalty_factors=args.penalty_factors,
                    qualification="small synthetic CPU; attained stationary fits, no global-optimality claim")
    write_json(args.outdir / "setup.json", metadata)
    records = []
    for size in args.sizes:
        for scenario in args.scenarios:
            for seed in args.seeds:
                directory = args.outdir / f"{scenario}-n{size}-seed{seed}"
                directory.mkdir()
                records.extend(compare_case(directory, size, seed, scenario, args.penalty_factors,
                                            policy, args.outdir / "progress.jsonl"))
    paired = [record["comparison"] for record in records if record["comparison"] is not None]
    summary = dict(total_pairs=len(records), both_qualified=len(paired),
                   common_unresolved=sum(record["common_surrogate"]["status"] != "qualified" for record in records),
                   reference_unresolved=sum(record["independent_enumeration"]["status"] != "qualified" for record in records),
                   partitions_differ=sum(not pair["identical_partition"] for pair in paired),
                   largest_absolute_objective_difference=max((abs(pair["objective_difference"]) for pair in paired), default=None),
                   largest_common_objective_excess=max((pair["objective_difference"] for pair in paired), default=None),
                   max_abs_phi_difference=max((pair["max_abs_phi_difference"] for pair in paired), default=None))
    write_json(args.outdir / "comparison.json", clean(dict(**metadata, summary=summary, results=records)))


if __name__ == "__main__":
    main()
