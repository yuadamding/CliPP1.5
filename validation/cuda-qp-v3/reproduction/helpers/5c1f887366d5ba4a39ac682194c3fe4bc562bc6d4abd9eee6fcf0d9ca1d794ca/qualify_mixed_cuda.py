"""One heterogeneous fixture/size per allocated CUDA job; no launch or CPU fallback.

Modes are explicit: full qualifies a complete current path, reference additionally
seals fixed problems under daaf50a, and paired qualifies the current full path and
replays those exact reference problems. Full-only is not paired qualification.
All modes use the unmodified default policy. Numerical timings include journal
instrumentation; this driver is a correctness/coverage study, not a speed claim.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
from time import perf_counter
import traceback

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import mixed_fixtures as fixtures
from benchmarks import qualify_cuda as common
from clipp1d.api import source_provenance
from clipp1d.cuda import selection
from clipp1d.cuda.graph import CompleteGraph, build_graph
from clipp1d.cuda.kernels import Kernels, differences
from clipp1d.cuda.partition import QualifiedPilot, refit
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda.scalar import ScalarBatch
from clipp1d.cuda.selection import DeviceFit
from clipp1d.cuda.solver import fit_lambda
from clipp1d.cuda_api import _export, _publish, _validate_device_fit, _validate_result, require_cuda


SCHEMA = "clipp1d.cuda.mixed_qualification.v1"
BASELINE_COMMIT = "daaf50ad5a2ae7301e9b54b9c31e87d87b24cfa6"
BASELINE_SOURCE = "ea7071788094e0cd3537530fcdc62e4228054fce19ff02ba8f2cf4eacce1dd26"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_bytes(), parse_constant=invalid)


def artifact_file(root, name):
    path = Path(name)
    require(
        not path.is_absolute() and path.parts and ".." not in path.parts,
        "Unsafe relative artifact path",
    )
    output = root / path
    require(output.is_file() and not output.is_symlink(), "Missing regular qualification artifact")
    require(
        all(not parent.is_symlink() for parent in output.parents if parent != root.parent),
        "Symlink in artifact ancestry",
    )
    return output


def load_receipt(path, expected_sha):
    path = Path(path)
    require(
        re.fullmatch(r"[0-9a-f]{64}", expected_sha or "") is not None,
        "An explicit receipt SHA-256 is required",
    )
    require(
        path.is_file() and not path.is_symlink() and common.sha(path) == expected_sha,
        "Qualification receipt hash differs",
    )
    receipt = load_json(path)
    require(
        receipt.get("schema") == SCHEMA and receipt.get("status") == "passed",
        "Qualification predecessor/reference did not pass",
    )
    require(
        receipt.get("cuda_available") is True
        and str(receipt.get("device", "")).startswith("cuda:")
        and receipt.get("policy") == asdict(CudaPolicy()),
        "Qualification receipt is not actual CUDA under the unchanged default policy",
    )
    require(
        receipt["full_path"]["search_status"] == "complete"
        and receipt["full_path"]["final_export_and_publication_qualified"] is True,
        "Qualification receipt lacks complete inference/publication",
    )
    for name, expected in receipt["artifacts"].items():
        require(
            common.sha(artifact_file(path.parent, name)) == expected, "Bound artifact hash differs"
        )
    require(
        common.sha(artifact_file(path.parent, receipt["events_file"])) == receipt["events_sha256"],
        "Qualification event journal differs from its receipt",
    )
    return receipt


def receipt_artifact(receipt_path, receipt, name):
    relative = str(Path(receipt["artifact_directory"]) / name)
    require(relative in receipt["artifacts"], "Required baseline artifact is not hash-bound")
    return artifact_file(Path(receipt_path).parent, relative)


def check_predecessor(args, source):
    if args.nodes == 64:
        require(
            args.predecessor is None and args.predecessor_sha256 is None,
            "64-node admission has no smaller required fixture",
        )
        return None
    require(args.predecessor is not None, "Larger fixtures require a passed smaller-size receipt")
    previous = load_receipt(args.predecessor, args.predecessor_sha256)
    size = {256: 64, 512: 256}[args.nodes]
    require(
        previous["fixture_family"] == args.fixture
        and previous["nodes"] == size
        and previous["source"]["source_sha256"] == source["source_sha256"]
        and previous["fixture_recipe"] == fixtures.RECIPE
        and previous["helpers"] == helper_hashes(),
        "Predecessor must qualify the same family/source/helpers at the immediately smaller size",
    )
    return dict(
        path=str(args.predecessor),
        sha256=args.predecessor_sha256,
        nodes=size,
        elapsed_seconds=previous["elapsed_seconds"],
    )


def helper_hashes():
    return {
        name: common.sha(path)
        for name, path in (
            ("driver", Path(__file__)),
            ("fixtures", Path(fixtures.__file__)),
            ("common_qualifier", Path(common.__file__)),
        )
    }


class Journal:
    def __init__(self, receipt_path):
        self.receipt_path = Path(receipt_path)
        require(self.receipt_path.suffix == ".json", "Output must name a new JSON receipt file")
        self.root = self.receipt_path.with_name(self.receipt_path.stem + ".artifacts")
        self.path = self.receipt_path.with_name(self.receipt_path.stem + ".events.jsonl")
        if any(
            path.exists() or path.is_symlink() for path in (self.receipt_path, self.root, self.path)
        ):
            raise FileExistsError("Qualification receipt, artifacts and events must all be new")
        self.root.mkdir(parents=True, exist_ok=False)
        self.path.touch(exist_ok=False)
        self.started = perf_counter()
        self.artifacts = {}

    def record(self, kind, **detail):
        row = common.clean(
            dict(
                kind=kind,
                utc=common.utc_now(),
                elapsed_seconds=perf_counter() - self.started,
                **detail,
            )
        )
        payload = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        print(payload, end="", flush=True)

    def artifact(self, name, value):
        path = self.root / name
        relative = str(path.relative_to(self.receipt_path.parent))
        require(relative not in self.artifacts, "Qualification artifact must not be overwritten")
        common.write_json(path, value)
        self.artifacts[relative] = common.sha(path)
        return relative

    def bind(self, path):
        path = Path(path)
        relative = str(path.relative_to(self.receipt_path.parent))
        self.artifacts[relative] = common.sha(path)


@contextmanager
def trace_candidates(journal):
    """Persist completed raw starts/refits even if a later stage fails or is killed."""
    original_raw, original_refit = selection.fit_lambda, selection.refit
    original_reference = selection.penalty_reference
    active = dict(index=-1, lambda_value=None)

    def reference(*args, **kwargs):
        value = original_reference(*args, **kwargs)
        policy = CudaPolicy()
        # This observes the exact production reference before the first solve;
        # it does not provide or replace the production path.
        planned = [0.0]
        if args[0].n > 1:
            planned.extend(
                float(torch.ldexp(value, torch.tensor(k, device=value.device)))
                for k in range(policy.path_min_exponent, policy.path_max_exponent + 1)
            )
        journal.artifact(
            "initial-path-plan.json",
            dict(
                lambda_reference=float(value),
                initial_candidates=[
                    dict(index=index, lambda_value=lam, status="not_attempted")
                    for index, lam in enumerate(planned)
                ],
                maximum_adaptive_extensions=policy.path_extensions,
                extension_rule="Double the final penalty only when its candidate is the current best",
                status_scope="Initial states; ordered candidate events and final path records retain subsequent outcomes",
            ),
        )
        journal.record("path_plan_sealed", initial_candidate_count=len(planned))
        return value

    def raw(*args, **kwargs):
        active["index"] += 1
        active["lambda_value"] = float(args[3])
        journal.record("candidate_started", **active)
        try:
            result = original_raw(*args, **kwargs)
        except BaseException as error:
            journal.record(
                "candidate_failed",
                **active,
                error_type=type(error).__name__,
                error=str(error),
                diagnostics=getattr(error, "diagnostics", {}),
            )
            raise
        journal.record(
            "candidate_raw",
            **active,
            qualified=result.qualified,
            diagnostics=result.diagnostics,
            raw_ccf=result.x.detach().cpu().tolist(),
        )
        return result

    def secondary(*args, **kwargs):
        try:
            result = original_refit(*args, **kwargs)
        except BaseException as error:
            journal.record(
                "candidate_refit_failed",
                **active,
                error_type=type(error).__name__,
                error=str(error),
                diagnostics=getattr(error, "diagnostics", {}),
            )
            raise
        journal.record(
            "candidate_refit",
            **active,
            score=float(result.score),
            loss=float(result.loss),
            gap=float(result.gap),
            centers=result.centers.detach().cpu().tolist(),
        )
        return result

    selection.fit_lambda, selection.refit = raw, secondary
    selection.penalty_reference = reference
    try:
        yield
    finally:
        selection.fit_lambda, selection.refit = original_raw, original_refit
        selection.penalty_reference = original_reference


def snapshot(fitted):
    labels, centers = common.canonical_labels(fitted)
    return dict(
        lambda_value=float(fitted.lambda_value),
        raw_ccf=fitted.raw.x.cpu().tolist(),
        refitted_ccf=fitted.refit.phi.cpu().tolist(),
        labels=labels.cpu().tolist(),
        centers=centers.cpu().tolist(),
        score=float(fitted.refit.score),
        raw_objective=float(fitted.raw.objective),
        raw_qualified=fitted.raw.qualified,
        search_status=fitted.search_status,
        raw_diagnostics=fitted.raw.diagnostics,
        raw_multiplicity=(fitted.model.terms(fitted.raw.x)[3].argmax(-1) + 1).cpu().tolist(),
        refitted_multiplicity=(fitted.model.terms(fitted.refit.phi)[3].argmax(-1) + 1)
        .cpu()
        .tolist(),
    )


def compare_snapshots(actual, reference, *, fixed_problem):
    """Full-fit changes are reported; fixed-problem differences use existing gates."""
    tolerances = dict(
        raw_ccf=(2e-5, 1e-8),
        refitted_ccf=(2e-5, 1e-8),
        centers=(2e-5, 1e-8),
        score=(1e-7, 1e-10),
        raw_objective=(1e-7, 1e-10),
    )
    errors, failures = {}, []
    for key, (atol, rtol) in tolerances.items():
        a, b = np.asarray(actual[key]), np.asarray(reference[key])
        finite = np.isfinite(a).all() and np.isfinite(b).all()
        same_shape = a.shape == b.shape
        errors[key] = float(np.max(np.abs(a - b))) if finite and same_shape else None
        if not finite or not same_shape or not np.allclose(a, b, atol=atol, rtol=rtol):
            failures.append(key)
    equal = {
        key: actual[key] == reference[key]
        for key in ("labels", "raw_multiplicity", "refitted_multiplicity")
    }
    failures.extend(key for key, value in equal.items() if not value)
    literal_equal = actual["lambda_value"] == reference["lambda_value"]
    if fixed_problem and not literal_equal:
        failures.append("literal_lambda")
    return dict(
        scope="identical fixed problem" if fixed_problem else "independent adaptive full fits",
        max_absolute_errors=errors,
        equal=equal,
        literal_lambda_equal=literal_equal,
        reference_lambda=reference["lambda_value"],
        current_lambda=actual["lambda_value"],
        within_existing_parity_gates=not failures,
        differing_fields=failures,
        fixed_problem=fixed_problem,
    )


def graph_differences(actual, reference, host):
    """Describe independent adaptive problems without relaxing fixed-problem gates."""
    current, previous = actual["pilot_and_graph"], reference["pilot_and_graph"]
    a, b = np.asarray(current["weights"]), np.asarray(previous["weights"])
    require(a.shape == b.shape == (len(host), len(host)), "Independent graph dimensions differ")
    caps_a, caps_b = a * actual["selected_lambda"], b * reference["selected_lambda"]
    diameter = np.maximum(
        np.abs(host.upper[:, None] - host.lower[None, :]),
        np.abs(host.upper[None, :] - host.lower[:, None]),
    )
    return dict(
        pilot_phi_max_absolute_difference=float(
            np.max(np.abs(np.asarray(current["pilot_phi"]) - previous["pilot_phi"]))
        ),
        weights_max_absolute_difference=float(np.max(np.abs(a - b))),
        graph_weights_identical=bool(np.array_equal(a, b)),
        selected_caps_max_absolute_difference=float(np.max(np.abs(caps_a - caps_b))),
        selected_objective_perturbation_envelope=float(
            0.5 * np.sum(np.abs(caps_a - caps_b) * diameter)
        ),
        reference_lambda_scale=reference["timings"]["lambda_reference"],
        current_lambda_scale=actual["timings"]["lambda_reference"],
        scope="Independent pilot/adaptive graph differences; not the identical fixed-problem comparison",
    )


def shared_problem(fitted, input_sha):
    pilot = fitted.pilot
    positive = [row["lambda_value"] for row in fitted.records if row["lambda_value"] > 0]
    lambdas = sorted(
        set(
            (
                0.0,
                positive[0],
                math.ldexp(fitted.timings["lambda_reference"], -4),
                float(fitted.lambda_value),
            )
        )
    )
    return dict(
        input_sha256=input_sha,
        mutation_ids=list(fitted.model.mutation_ids),
        pilot={
            name: getattr(pilot, name).cpu().tolist()
            for name in ("phi", "loss", "lower_bound", "gap", "alternative", "qualified")
        }
        | dict(subdivisions=pilot.subdivisions),
        graph=dict(
            weights=fitted.graph.weights.cpu().tolist(),
            gap_floor=float(fitted.graph.gap_floor),
            normalization=float(fitted.graph.normalization),
            weight_rule=fitted.graph.weight_rule,
        ),
        lambdas=lambdas,
        scope="Exact serialized baseline pilots/weights/literal penalties; no continuation",
    )


def restore_shared(model, payload, input_sha):
    require(
        payload["input_sha256"] == input_sha
        and tuple(payload["mutation_ids"]) == model.mutation_ids,
        "Fixed-problem input or canonical identities differ",
    )
    tensors = {
        name: torch.tensor(
            value, dtype=torch.bool if name == "qualified" else torch.float64, device=model.device
        )
        for name, value in payload["pilot"].items()
        if name != "subdivisions"
    }
    pilot = ScalarBatch(**tensors, subdivisions=payload["pilot"]["subdivisions"])
    QualifiedPilot(model, pilot, CudaPolicy())
    g = payload["graph"]
    graph = CompleteGraph(
        torch.tensor(g["weights"], dtype=torch.float64, device=model.device),
        pilot.phi.clone(),
        torch.tensor(g["gap_floor"], dtype=torch.float64, device=model.device),
        torch.tensor(g["normalization"], dtype=torch.float64, device=model.device),
        model.mutation_ids,
        g["weight_rule"],
    )
    recipe = build_graph(pilot.phi, model.mutation_ids)
    require(
        torch.equal(graph.weights, recipe.weights)
        and torch.equal(graph.gap_floor, recipe.gap_floor)
        and torch.equal(graph.normalization, recipe.normalization),
        "Fixed graph differs from its exact pilot recipe",
    )
    values = payload["lambdas"]
    require(
        values == sorted(set(values))
        and values[0] == 0
        and len(values) >= 3
        and all(math.isfinite(value) and value >= 0 for value in values),
        "Invalid literal comparison penalties",
    )
    return graph, pilot


def fixed_replays(model, payload, input_sha, journal):
    graph, pilot = restore_shared(model, payload, input_sha)
    outputs = []
    for index, value in enumerate(payload["lambdas"]):
        journal.record("fixed_problem_started", index=index, lambda_value=value)
        began = perf_counter()
        lam = model.lower.new_tensor(value)
        raw = fit_lambda(model, graph, pilot, lam)
        secondary = refit(model, raw.x)
        complete = raw.qualified and raw.diagnostics.get("search_complete", False)
        fitted = DeviceFit(
            model,
            graph,
            pilot,
            raw,
            secondary,
            lam,
            "complete" if complete else "incomplete",
            [],
            {},
            CudaPolicy(),
        )
        detail = snapshot(fitted)
        # Save all raw/start/refit status before independent final admission.
        journal.artifact(f"fixed-{index}-attempt.json", detail)
        require(complete, "A fixed-problem planned start remains unresolved")
        _validate_device_fit(
            fitted
        )  # Fresh observed-objective audit plus scalar/group/score checks.
        torch.cuda.synchronize(model.device)
        detail.update(
            seconds=perf_counter() - began,
            final_device_qualification=True,
            fusion_penalty=float(0.5 * (graph.weights * lam * differences(raw.x).abs()).sum()),
        )
        journal.artifact(f"fixed-{index}-qualified.json", detail)
        journal.record(
            "fixed_problem_qualified", index=index, lambda_value=value, seconds=detail["seconds"]
        )
        outputs.append(detail)
    require(
        any(row["lambda_value"] > 0 and row["fusion_penalty"] > 0 for row in outputs),
        "Fixed replay omitted nonzero nonfused penalty work",
    )
    return outputs


@torch.no_grad()
def execute(args, receipt, journal):
    source = source_provenance()
    receipt["source"] = source
    require(
        source["source_sha256"] == args.expected_source_sha256,
        "Current source differs from the requested seal",
    )
    if args.mode == "reference":
        require(
            source["source_sha256"] == BASELINE_SOURCE,
            "Reference mode requires exact daaf50a production bytes",
        )
    require(
        (args.mode == "paired") == (args.baseline is not None),
        "Only paired mode accepts/requires a baseline receipt",
    )
    require(
        (args.mode == "paired") == (args.baseline_sha256 is not None),
        "Only paired mode accepts/requires an explicit baseline receipt hash",
    )
    predecessor = check_predecessor(args, source)
    receipt["predecessor"] = predecessor
    device = require_cuda(args.device)
    require(
        os.environ.get("CC") and shutil.which(os.environ["CC"]),
        "Set CC to an available CUDA compiler driver",
    )
    receipt.update(
        device=str(device),
        gpu=torch.cuda.get_device_name(device),
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        interpreter=sys.executable,
        compiler=shutil.which(os.environ["CC"]),
        lsf_job_id=os.environ.get("LSB_JOBID"),
        cuda_available=True,
    )
    began = perf_counter()
    operation_started = began
    data, host = fixtures.write_fixture(journal.root / "input.tsv", args.fixture, args.nodes)
    journal.bind(journal.root / "input.tsv")
    difficulty = fixtures.difficulty_summary(host, data)
    fixtures.validate_difficulty(difficulty, args.fixture)
    receipt["input_sha256"] = data.input_sha256
    receipt["difficulty"] = difficulty
    journal.artifact(
        "input-identity.json",
        dict(**common.host_identity(host), difficulty=difficulty, input_sha256=data.input_sha256),
    )
    preparation_seconds = perf_counter() - began
    baseline = None
    if args.mode == "paired":
        baseline = load_receipt(args.baseline, args.baseline_sha256)
        require(
            baseline["mode"] == "reference"
            and baseline["source"]["source_sha256"] == BASELINE_SOURCE
            and baseline["input_sha256"] == data.input_sha256
            and baseline["fixture_family"] == args.fixture
            and baseline["nodes"] == args.nodes
            and baseline["helpers"] == receipt["helpers"],
            "Baseline source/input/fixture/harness does not match this paired job",
        )
        receipt["baseline"] = dict(
            path=str(args.baseline),
            sha256=args.baseline_sha256,
            commit=BASELINE_COMMIT,
            source_sha256=BASELINE_SOURCE,
        )
    torch.cuda.reset_peak_memory_stats(device)
    began = operation_started
    kernels, admission_seconds = common.timed(device, lambda: Kernels(device, compiled=True))
    require(kernels.compiled, "CUDA compiled-kernel admission did not occur")
    model, upload_seconds = common.timed(device, lambda: common.upload(host, device, kernels))
    journal.record("full_path_started", mode=args.mode, nodes=args.nodes)
    with trace_candidates(journal):
        fitted, fit_seconds = common.timed(device, lambda: selection.fit_tensor_model(model))
    summary = common.fit_summary(fitted)
    summary.update(fit_seconds=fit_seconds, path_position=common.path_position(fitted, "default"))
    journal.artifact("full-path.json", summary)
    receipt["full_path"] = dict(
        search_status=fitted.search_status,
        fit_seconds=fit_seconds,
        final_export_and_publication_qualified=False,
    )
    require(
        fitted.search_status == "complete",
        "Complete mixed fixture path contains unresolved candidates or starts",
    )
    phases = dict(
        input_preparation_seconds=preparation_seconds,
        device_upload_and_compile_seconds=upload_seconds + admission_seconds,
    )
    phases.update(
        {
            name: fitted.timings[name]
            for name in (
                "pilot_seconds",
                "graph_build_seconds",
                "path_seconds",
                "refit_seconds",
                "stage_integrity_seconds",
            )
        }
    )
    provenance = dict(
        source,
        backend="cuda",
        input_sha256=data.input_sha256,
        clonal_constraint=False,
        numerical_stages=fitted.timings,
        phase_seconds=phases,
        policy=asdict(CudaPolicy()),
    )
    exported = _export(fitted, provenance, wall_started=began)
    _validate_result(exported, data, fitted.records)
    exported = _publish(
        exported, data, journal.root / "public-output", fitted.records, wall_started=began
    )
    public_receipt = load_json(journal.root / "public-output/run.json")
    common.validate_public_search(exported, public_receipt, device)
    measurements = common.validate_public_measurements(exported, public_receipt)
    for path in sorted((journal.root / "public-output").iterdir()):
        journal.bind(path)
    full_snapshot = snapshot(fitted)
    journal.artifact("full-result.json", full_snapshot)
    receipt["full_path"].update(
        final_export_and_publication_qualified=True,
        total_seconds=perf_counter() - began,
        measurements=measurements,
        initial_compiler_admission_seconds=admission_seconds,
        numerical_stages=fitted.timings,
    )
    journal.record("full_path_qualified", **receipt["full_path"])
    if args.mode in ("reference", "paired"):
        payload = (
            shared_problem(fitted, data.input_sha256)
            if baseline is None
            else load_json(receipt_artifact(args.baseline, baseline, "shared-problem.json"))
        )
        journal.artifact("shared-problem.json", payload)
        outputs = fixed_replays(model, payload, data.input_sha256, journal)
        journal.artifact("fixed-results.json", outputs)
        if baseline is not None:
            old_outputs = load_json(receipt_artifact(args.baseline, baseline, "fixed-results.json"))
            require(
                len(outputs) == len(old_outputs), "Baseline/current fixed-problem coverage differs"
            )
            comparisons = [
                compare_snapshots(a, b, fixed_problem=True) for a, b in zip(outputs, old_outputs)
            ]
            independent = compare_snapshots(
                full_snapshot,
                load_json(receipt_artifact(args.baseline, baseline, "full-result.json")),
                fixed_problem=False,
            )
            adaptive = graph_differences(
                summary,
                load_json(receipt_artifact(args.baseline, baseline, "full-path.json")),
                host,
            )
            journal.artifact(
                "comparison.json",
                dict(
                    fixed_problems=comparisons,
                    independent_full_fit=independent,
                    independent_adaptive_graphs=adaptive,
                    scope="Fixed problem parity and independent full-fit differences are separate claims",
                ),
            )
            receipt["comparison"] = dict(
                fixed_problems=len(comparisons),
                independent_full_fit=independent,
                independent_adaptive_graphs=adaptive,
                all_fixed_problems_within_existing_gates=all(
                    row["within_existing_parity_gates"] for row in comparisons
                ),
            )
            require(
                receipt["comparison"]["all_fixed_problems_within_existing_gates"],
                "Fixed-problem baseline parity failed",
            )
    require(
        source_provenance()["source_sha256"] == source["source_sha256"]
        and helper_hashes() == receipt["helpers"],
        "Source or qualification helpers changed during the job",
    )
    receipt["qualification_scope"] = {
        "full": "complete full-path only; paired baseline comparison not requested",
        "reference": "baseline complete path plus sealed qualified fixed problems",
        "paired": "current complete path plus identical fixed-problem comparison against daaf50a",
    }[args.mode]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--fixture", choices=fixtures.FAMILIES, required=True)
    result.add_argument("--nodes", choices=fixtures.SIZES, type=int, required=True)
    result.add_argument("--mode", choices=("full", "reference", "paired"), required=True)
    result.add_argument("--expected-source-sha256", required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New JSON receipt; sibling <stem>.artifacts/ and <stem>.events.jsonl are created",
    )
    result.add_argument("--predecessor", type=Path)
    result.add_argument("--predecessor-sha256")
    result.add_argument("--baseline", type=Path)
    result.add_argument("--baseline-sha256")
    result.add_argument(
        "--timeout-seconds",
        type=int,
        default=3000,
        help="Wall limit around fixture preparation and numerical/public phases; policy budgets stay unchanged",
    )
    return result


def main():
    args = parser().parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("Qualification timeout must be a positive number of seconds")
    journal = Journal(args.out)
    receipt = dict(
        schema=SCHEMA,
        status="running",
        started_utc=common.utc_now(),
        mode=args.mode,
        fixture_family=args.fixture,
        fixture_recipe=fixtures.RECIPE,
        nodes=args.nodes,
        artifact_directory=journal.root.name,
        events_file=journal.path.name,
        helpers=helper_hashes(),
        source=source_provenance(),
        policy=asdict(CudaPolicy()),
        baseline_commit=BASELINE_COMMIT,
        scope="Synthetic CUDA qualification, not cohort accuracy or GPU speedup; observed timings include journaling",
        command=sys.argv,
        artifacts={},
        timeout_seconds=args.timeout_seconds,
    )

    def stopped(signum, frame):
        raise KeyboardInterrupt(f"Qualification interrupted by signal {signum}")

    def timeout(signum, frame):
        raise TimeoutError(f"{args.timeout_seconds}-second qualification wall limit reached")

    previous_term = signal.signal(signal.SIGTERM, stopped)
    previous_alarm = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    exit_code = 0
    try:
        execute(args, receipt, journal)
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            diagnostics=getattr(error, "diagnostics", {}),
            traceback=traceback.format_exc(),
        )
        journal.record(
            "failure",
            error_type=type(error).__name__,
            error=str(error),
            diagnostics=getattr(error, "diagnostics", {}),
        )
        exit_code = 1
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGALRM, previous_alarm)
        receipt.update(
            finished_utc=common.utc_now(),
            elapsed_seconds=perf_counter() - journal.started,
            artifacts=journal.artifacts,
            events_sha256=common.sha(journal.path),
        )
        common.write_json(journal.receipt_path, receipt)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
