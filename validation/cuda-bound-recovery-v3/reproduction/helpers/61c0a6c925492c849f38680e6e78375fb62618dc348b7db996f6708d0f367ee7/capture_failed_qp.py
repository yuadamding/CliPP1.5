"""Diagnostic-only exact failed-surrogate capture; never a scientific fit pass.

The AST return hooks preserve each explicitly pinned source's arithmetic. Successful QPs keep
only temporary references; failed QPs clone one bounded state per start. JSON
serialization and independent certificates run after the outer start returns,
outside its QP timers. All observed timings remain diagnostic/instrumented.
"""

import argparse
import ast
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import sys
import textwrap
from time import perf_counter
import traceback

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import mixed_fixtures as fixtures
from benchmarks import qualify_cuda as common
from benchmarks import qualify_mixed_cuda as mixed
from clipp1d.api import source_provenance
from clipp1d.cuda import qp, selection, solver
from clipp1d.cuda.kernels import Kernels, gap_kkt
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda_api import require_cuda


SCHEMA = "clipp1d.cuda.failed_qp_capture.v1"
PROFILE_SCHEMA = "clipp1d.cuda.failed_qp_capture.v2"
FROZEN_COMMIT = "430db26cf07466e88e53c6e1a8fbe2be7b7b25e9"
FROZEN_SOURCE = "726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88"
FROZEN_FILES = {
    "cuda/qp.py": "925e1e5ee67f7630f0d229a560f2af8958fffd1b01ac7609f868a85aa01a74b0",
    "cuda/solver.py": "80a0b34cd856698efd7034a523e6d850c21ba21e3805f38928897db1f2d6091b",
}
INPUTS = {
    ("below_one", 64): "ae53764990bc252cb7f6a2a14a8eb21a57e3ec46fef28423c08354575ef8a5db",
    ("mixed_support", 256): "6b5cb4f2cfce4cb6fbd81eff8a6ab8dca4e3c810dcfdf4e1ec75fa76fa6b2428",
}
PROFILES = {
    "baseline430": dict(
        source_sha256=FROZEN_SOURCE,
        source_files=FROZEN_FILES,
        commit=FROZEN_COMMIT,
        failures={("below_one", 64): 2, ("mixed_support", 256): 11},
    ),
    "normalized276da": dict(
        source_sha256="276da27b9a8718f3f06182108ccb0a6b7f04f3937a67f050f9721e031554aea5",
        source_files={
            "cuda/qp.py": "07bacefb982c75418e24936d30de04edd70d5ffc7f577f7dab4befd77171e695",
            "cuda/solver.py": FROZEN_FILES["cuda/solver.py"],
        },
        commit=None,
        failures={("below_one", 64): 1},
        expected_contexts={
            ("below_one", 64): [dict(
                path_index=2, start_index=2, outer_iteration=1,
                backtracks=20, qp_ordinal_within_start=22,
                lambda_value=553.4678435191591,
            )],
        },
    ),
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def profile_for_source(source_sha256):
    matches = [name for name, profile in PROFILES.items()
               if profile["source_sha256"] == source_sha256]
    require(len(matches) == 1, "Capture source is not an explicitly pinned diagnostic profile")
    return matches[0], PROFILES[matches[0]]


def check_profile_source(name, source):
    require(name in PROFILES, "Unknown diagnostic source profile")
    profile = PROFILES[name]
    require(source["source_sha256"] == profile["source_sha256"],
            "Capture source differs from its pinned profile")
    require(all(source["source_files"].get(key) == digest
                for key, digest in profile["source_files"].items()),
            "Profile QP/solver implementation differs")
    return profile


def check_capture_profile(value):
    """Keep archived v1 binding intact; v2 requires an explicit named profile."""
    if value["schema"] == SCHEMA:
        require(value["source"]["source_sha256"] == FROZEN_SOURCE,
                "Capture v1 is not from frozen430db26")
        return "baseline430", PROFILES["baseline430"]
    require(value["schema"] == PROFILE_SCHEMA, "Unknown capture schema")
    name = value["source_profile"]
    profile = check_profile_source(name, value["source"])
    family = (value["fixture_family"], value["nodes"])
    require(family in profile["failures"]
            and value["expected_failures"] == profile["failures"][family]
            and value["captured_failures"] == len(value["captures"]) == value["expected_failures"],
            "Capture profile failure inventory differs")
    require(value["input_sha256"] == INPUTS[family], "Capture profile input differs")
    require(value["policy"] == asdict(CudaPolicy()), "Capture profile numerical policy differs")
    require(all(entry["context"]["source_sha256"] == profile["source_sha256"]
                and entry["context"]["input_sha256"] == value["input_sha256"]
                and entry["context"]["policy"] == value["policy"]
                for entry in value["captures"]), "Capture profile context binding differs")
    expected = profile.get("expected_contexts", {}).get(family)
    if expected is not None:
        actual = [{key: entry["context"].get(key) for key in expected[0]}
                  for entry in value["captures"]]
        require(actual == expected, "Capture profile failure identities differ")
    return name, profile


def number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def helper_hashes():
    return {
        name: common.sha(path)
        for name, path in (
            ("driver", Path(__file__)),
            ("fixtures", Path(fixtures.__file__)),
            ("mixed_qualifier", Path(mixed.__file__)),
            ("common_qualifier", Path(common.__file__)),
        )
    }


def _top(values, count=8):
    indices = torch.argsort(values.abs(), descending=True, stable=True)[:count]
    return [dict(index=int(i), value=number(values[i])) for i in indices]


@torch.no_grad()
def decompose_gap(x, q, h, target, lower, upper, caps):
    """Original stable gap components on the supplied device, no regenerated QP."""
    a = -q.sum(1)
    free = lower < upper
    safe_target = torch.where(free, target, x)
    unconstrained = safe_target - a / h
    minimizer = torch.maximum(lower, torch.minimum(upper, unconstrained))
    delta = torch.where(free, x - minimizer, 0.0)
    normal = torch.where(
        unconstrained < lower,
        h * (lower - safe_target) + a,
        torch.where(unconstrained > upper, h * (upper - safe_target) + a, 0.0),
    )
    quadratic = 0.5 * h * delta.square()
    box = normal * delta
    d = x[None, :] - x[:, None]
    edges = caps * d.abs() - q * d
    triangle = torch.triu_indices(x.numel(), x.numel(), 1, device=x.device)
    edge_values = edges[triangle[0], triangle[1]]
    stats = gap_kkt(x, q, h, target, lower, upper, caps)
    _, sizes = torch.unique(x, sorted=True, return_counts=True)
    top_edges = []
    for item in _top(edge_values):
        index = item.pop("index")
        top_edges.append(dict(left=int(triangle[0, index]), right=int(triangle[1, index]), **item))
    sums = (quadratic.sum(), box.clamp_min(0).sum(), 0.5 * edges.clamp_min(0).sum())
    return dict(
        node_quadratic=number(sums[0]),
        box_normal=number(sums[1]),
        edge=number(sums[2]),
        total=number(sum(sums)),
        raw_box_normal=number(box.sum()),
        raw_edge=number(0.5 * edges.sum()),
        literal_normal_formula_sum=number(((h * (minimizer - safe_target) + a) * delta).sum()),
        certificate=dict(gap=number(stats[0]), scale=number(stats[1]), kkt=number(stats[2])),
        bounds=dict(
            lower_active=int((free & (x == lower)).sum()),
            upper_active=int((free & (x == upper)).sum()),
            fixed=int((~free).sum()),
            interior=int((free & (x > lower) & (x < upper)).sum()),
        ),
        group_sizes=sizes.cpu().tolist(),
        curvature=dict(minimum=number(h.min()), maximum=number(h.max())),
        top_nodes=_top(quadratic + box.clamp_min(0)),
        top_edges=top_edges,
        dual=dict(
            max_capacity_violation=number((q.abs() - caps).clamp_min(0).max()),
            exact_skew=bool(torch.equal(q, -q.T)),
        ),
        scope="Three stable nonnegative gap components use the original safe-target/fixed-coordinate convention. Raw signed terms are separate; this is not a new admission gate.",
    )


def _regular(root, name):
    path = Path(name)
    require(
        path.parts and not path.is_absolute() and ".." not in path.parts,
        "Unsafe capture artifact path",
    )
    full = root / path
    require(full.is_file() and not full.is_symlink(), "Missing regular capture artifact")
    cursor = full.parent
    while cursor != root.parent:
        require(not cursor.is_symlink(), "Symlink capture ancestry")
        cursor = cursor.parent
    return full


def load_capture(path, expected_sha):
    path = Path(path).absolute()
    require(
        path.is_file()
        and not path.is_symlink()
        and all(not part.is_symlink() for part in path.parents),
        "Capture receipt must have regular nonsymlink ancestry",
    )
    path = path.resolve()
    require(
        len(expected_sha) == 64 and common.sha(path) == expected_sha, "Capture receipt hash differs"
    )
    value = mixed.load_json(path)
    require(
        value["status"] == "passed",
        "Capture did not qualify diagnostically",
    )
    check_capture_profile(value)
    require(
        value["cuda_available"] is True and value["device"].startswith("cuda:"),
        "Capture lacks actual CUDA identity",
    )
    for name, digest in value["artifacts"].items():
        require(common.sha(_regular(path.parent, name)) == digest, "Capture artifact hash differs")
    require(
        common.sha(_regular(path.parent, value["events_file"])) == value["events_sha256"],
        "Capture journal differs",
    )
    value["_receipt_path"] = path
    value["_root"] = path.parent / value["artifact_directory"]
    return value


def load_record(receipt, entry):
    path = _regular(receipt["_receipt_path"].parent, entry["record"])
    require(
        receipt["artifacts"].get(entry["record"]) == entry["sha256"] == common.sha(path),
        "Capture record hash differs",
    )
    value = mixed.load_json(path)
    require(value["context"] == entry["context"], "Capture entry context differs")
    require(
        value["context"]["source_sha256"] == receipt["source"]["source_sha256"]
        and value["context"]["policy"] == receipt["policy"],
        "Captured QP source or policy differs from its receipt",
    )
    return value


def load_tensor(receipt, descriptor, device):
    if descriptor is None:
        return None
    path = _regular(receipt["_receipt_path"].parent, descriptor["path"])
    require(
        receipt["artifacts"].get(descriptor["path"]) == descriptor["sha256"] == common.sha(path),
        "Tensor JSON hash differs",
    )
    value = mixed.load_json(path)
    require(
        value["dtype"] == descriptor["dtype"] == "float64"
        and value["shape"] == descriptor["shape"],
        "Tensor metadata differs",
    )
    if value["encoding"] == "json_numbers":
        array = np.asarray(value["values"], dtype="<f8").reshape(value["shape"])
    else:
        require(value["encoding"] == "float64_le_hex", "Unknown lossless tensor encoding")
        array = np.frombuffer(bytes.fromhex(value["values"]), dtype="<f8").reshape(value["shape"])
    require(
        hashlib.sha256(array.tobytes()).hexdigest()
        == descriptor["tensor_sha256"]
        == value["tensor_sha256"],
        "Tensor values did not roundtrip exactly",
    )
    return torch.tensor(array.copy(), dtype=torch.float64, device=device)


class _Returns(ast.NodeTransformer):
    def __init__(self, hook):
        self.hook = hook

    def visit_Return(self, node):
        return ast.copy_location(
            ast.Return(
                ast.Call(
                    ast.Name(self.hook, ast.Load()),
                    [
                        node.value or ast.Constant(None),
                        ast.Call(ast.Name("locals", ast.Load()), [], []),
                    ],
                    [],
                )
            ),
            node,
        )


def hooked_function(function, hook_name, namespace):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    require(
        len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef),
        "Unexpected instrumented function source",
    )
    require(
        not any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            for n in ast.walk(tree.body[0])
            if n is not tree.body[0]
        ),
        "Nested capture target needs explicit review",
    )
    tree.body[0].decorator_list = []
    tree = _Returns(hook_name).visit(tree)
    ast.fix_missing_locations(tree)
    exec(
        compile(tree, inspect.getsourcefile(function) + "::diagnostic_return_hooks", "exec"),
        namespace,
    )
    return namespace[function.__name__]


def _clone(value, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        if id(value) not in memo:
            memo[id(value)] = value.detach().clone()
        return memo[id(value)]
    if isinstance(value, dict):
        return {key: _clone(item, memo) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clone(item, memo) for item in value]
    return value


class Capture:
    def __init__(self, journal, source, maximum=13):
        self.journal, self.source, self.maximum = journal, source, maximum
        self.context = {}
        self.path_index, self.start_index = -1, -1
        self.graph = self.start_vector = self.kernels = None
        self.pending = None
        self.latest_attempt = self.latest_valid = self.latest_refined = None
        self.entries, self.array_cache, self.captured_keys = [], {}, set()
        self.snapshot_seconds = self.export_seconds = 0.0

    def _array(self, tensor):
        if tensor is None:
            return None
        require(
            tensor.dtype == torch.float64,
            "Captured numerical arrays must be literal float64 tensors",
        )
        array = np.asarray(tensor.detach().cpu().numpy(), dtype="<f8")
        bits = array.tobytes()
        digest = hashlib.sha256(bits).hexdigest()
        identity = hashlib.sha256(json.dumps(list(array.shape)).encode() + bits).hexdigest()
        if identity not in self.array_cache:
            finite = np.isfinite(array).all()
            value = dict(
                dtype="float64",
                shape=list(array.shape),
                tensor_sha256=digest,
                encoding="json_numbers" if finite else "float64_le_hex",
                values=array.tolist() if finite else bits.hex(),
            )
            name = "arrays/" + identity + ".json"
            (self.journal.root / "arrays").mkdir(exist_ok=True)
            relative = self.journal.artifact(name, value)
            self.array_cache[identity] = dict(
                path=relative,
                sha256=self.journal.artifacts[relative],
                dtype="float64",
                shape=list(array.shape),
                tensor_sha256=digest,
            )
        return self.array_cache[identity]

    def _encode(self, value):
        if isinstance(value, torch.Tensor):
            return self._array(value)
        if isinstance(value, dict):
            return {key: self._encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self._encode(item) for item in value]
        return value

    def prepared(self, result, local):
        attempt = dict(
            x=local.get("candidate"),
            input_q=local["q"],
            q=local.get("candidate_q"),
            stats=local.get("stats"),
            tolerance=local["tolerance"],
            objective_admitted=result is not None,
            objective_reference=local.get("reference"),
            old_value=local.get("old_value"),
            new_value=local.get("new_value"),
            roundoff=local.get("roundoff"),
        )
        self.latest_attempt = attempt
        if result is not None:
            self.latest_valid = attempt
        return result

    def refined(self, result, local):
        if "candidate_q" in local:
            self.latest_refined = dict(
                x=local["candidate"],
                q=local["candidate_q"],
                input_q=local["q"],
                stats=local["stats"],
                returned_qualified_candidate=result[0] is not None,
                flow_steps=local["steps"],
                tolerance=None if self.latest_valid is None else self.latest_valid["tolerance"],
            )
        return result

    def returned(self, result, local):
        if result.qualified:
            return result
        began = perf_counter()
        key = (self.context.get("path_index"), self.context.get("start_index"))
        require(
            key not in self.captured_keys and self.pending is None,
            "More than one failed QP in a start",
        )
        require(len(self.entries) < self.maximum, "Failed-QP capture admission cap exceeded")
        self.captured_keys.add(key)
        original = dict(
            problem={
                name: local.get(name)
                for name in ("h", "target", "lower", "upper", "caps", "start", "dual")
            },
            terminal_admm={name: local.get(name) for name in ("x", "z", "v", "q", "rho")},
            returned=dict(
                x=result.x,
                q=result.dual,
                gap=result.gap,
                scale=result.scale,
                kkt=result.kkt,
                qualified=result.qualified,
                iterations=result.iterations,
                polish_iterations=result.polish_iterations,
            ),
            last_equality_attempt=self.latest_attempt,
            last_refined_state=self.latest_refined,
            start_vector=self.start_vector,
            graph=None
            if self.graph is None
            else dict(
                weights=self.graph.weights,
                pilot=self.graph.pilot,
                gap_floor=self.graph.gap_floor,
                normalization=self.graph.normalization,
                mutation_ids=list(self.graph.mutation_ids),
                weight_rule=self.graph.weight_rule,
            ),
            context=dict(
                self.context,
                source_sha256=self.source["source_sha256"],
                policy=asdict(local["policy"]),
            ),
            terminal_state_scope="Literal locals at return, including any final rho/v residual balancing. No q=rho*v identity is imposed.",
        )
        self.pending = _clone(original)
        self.snapshot_seconds += perf_counter() - began
        return result

    def call_qp(self, *args, **kwargs):
        self.latest_attempt = self.latest_valid = self.latest_refined = None
        frame = inspect.currentframe().f_back
        outer = frame.f_locals
        self.context.update(
            outer_iteration=outer.get("iteration"),
            backtrack_index=outer.get("_"),
            backtracks=outer.get("backtracks"),
            inflation=outer.get("inflation"),
            qp_ordinal_within_start=outer.get("surrogate_calls", -1) + 1,
        )
        self.kernels = args[5] if len(args) > 5 else kwargs["kernels"]
        try:
            return self.instrumented(*args, **kwargs)
        finally:
            del frame

    def flush(self):
        if self.pending is None:
            return
        began = perf_counter()
        captured, self.pending = self.pending, None
        p, r = captured["problem"], captured["returned"]
        args = [p[name] for name in ("h", "target", "lower", "upper", "caps")]
        original = torch.stack((r["gap"], r["scale"], r["kkt"]))
        reproduced = self.kernels.gap_kkt(r["x"], r["q"], *args)
        # Equal nonfinite bit patterns are represented losslessly too.
        exact = np.array_equal(
            original.detach().cpu().numpy().view(np.uint64),
            reproduced.detach().cpu().numpy().view(np.uint64),
        )
        captured["original_compiled_certificate_replay"] = dict(
            exact_bits_equal=bool(exact), stats=reproduced
        )
        captured["decomposition"] = {"returned": decompose_gap(r["x"], r["q"], *args)}
        for name in ("last_equality_attempt", "last_refined_state"):
            state = captured[name]
            if state is not None and state.get("x") is not None and state.get("q") is not None:
                captured["decomposition"][name] = decompose_gap(state["x"], state["q"], *args)
        name = f"capture-{len(self.entries):03d}.json"
        relative = self.journal.artifact(name, self._encode(captured))
        self.entries.append(
            dict(
                record=relative,
                sha256=self.journal.artifacts[relative],
                context=captured["context"],
            )
        )
        self.journal.record(
            "failed_qp_captured",
            record=relative,
            context=captured["context"],
            original_compiled_certificate_exact=bool(exact),
            decomposition=captured["decomposition"]["returned"],
        )
        self.export_seconds += perf_counter() - began
        require(
            exact,
            "Original compiled terminal certificate did not reproduce from literal saved tensors",
        )

    @contextmanager
    def instrument(self):
        original_fit, original_start, original_solver_qp, original_qp = (
            selection.fit_lambda,
            solver._solve_start,
            solver.solve_qp,
            qp.solve_qp,
        )
        namespace = dict(
            vars(qp),
            _capture_prepared=self.prepared,
            _capture_refined=self.refined,
            _capture_returned=self.returned,
        )
        hooked_function(qp._prepare_equality_proposal, "_capture_prepared", namespace)
        hooked_function(qp._refine_polish, "_capture_refined", namespace)
        self.instrumented = hooked_function(qp.solve_qp, "_capture_returned", namespace)

        def fitting(model, graph, pilot, lam, *args, **kwargs):
            self.path_index += 1
            self.start_index = -1
            self.graph = graph
            self.context = dict(
                path_index=self.path_index,
                lambda_value=float(lam),
                graph_mutation_ids=list(graph.mutation_ids),
                input_sha256=self.source.get("input_sha256"),
            )
            return original_fit(model, graph, pilot, lam, *args, **kwargs)

        def start(model, graph, lam, initial, policy):
            self.start_index += 1
            self.context.update(start_index=self.start_index)
            self.start_vector = initial
            try:
                return original_start(model, graph, lam, initial, policy)
            finally:
                self.flush()

        selection.fit_lambda, solver._solve_start = fitting, start
        solver.solve_qp = qp.solve_qp = self.call_qp
        try:
            yield self
        finally:
            selection.fit_lambda, solver._solve_start = original_fit, original_start
            solver.solve_qp, qp.solve_qp = original_solver_qp, original_qp


@torch.no_grad()
def execute(args, receipt, journal):
    source = source_provenance()
    profile = check_profile_source(args.source_profile, source)
    require(
        args.expected_source_sha256 == source["source_sha256"],
        "Capture requires the exact requested profile production",
    )
    expected = profile["failures"]
    require(
        (args.fixture, args.nodes) in expected
        and args.expected_failures == expected[(args.fixture, args.nodes)],
        "Capture only the declared profile failure families/counts",
    )
    device = require_cuda(args.device)
    receipt.update(
        source=source,
        device=str(device),
        cuda_available=True,
        gpu=torch.cuda.get_device_name(device),
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        lsf_job_id=os.environ.get("LSB_JOBID"),
    )
    data, host = fixtures.write_fixture(journal.root / "input.tsv", args.fixture, args.nodes)
    require(
        data.input_sha256 == INPUTS[(args.fixture, args.nodes)],
        "Canonical archived input bytes changed",
    )
    journal.bind(journal.root / "input.tsv")
    receipt["input_sha256"] = data.input_sha256
    source = dict(source, input_sha256=data.input_sha256)
    kernels = Kernels(device, compiled=True)
    model = common.upload(host, device, kernels)
    capture = Capture(journal, source, maximum=args.expected_failures)
    receipt["captures"] = capture.entries
    began = perf_counter()
    with capture.instrument(), mixed.trace_candidates(journal):
        fitted = selection.fit_tensor_model(model)
    torch.cuda.synchronize(device)
    journal.artifact("full-path.json", common.fit_summary(fitted))
    unresolved = [
        (index, j)
        for index, row in enumerate(fitted.records)
        for j, start in enumerate(row.get("starts", []))
        if not start["qualified"]
    ]
    require(
        set(unresolved) == capture.captured_keys and len(capture.entries) == args.expected_failures,
        "Captured failures differ from exact unresolved-start inventory",
    )
    require(
        fitted.search_status == "incomplete", "Failure reproduction changed scientific path status"
    )
    receipt.update(
        scientific_search_status=fitted.search_status,
        scientific_fit_qualified=False,
        diagnostic_capture_qualified=True,
        captured_failures=len(capture.entries),
        diagnostic_seconds=perf_counter() - began,
        snapshot_dispatch_seconds=capture.snapshot_seconds,
        post_start_export_seconds=capture.export_seconds,
        numerical_stages=fitted.timings,
        qualification_scope="Expected failed surrogate states captured and original compiled certificates reproduced exactly; this is not a complete fit or accuracy qualification.",
    )
    check_capture_profile(receipt)
    check_profile_source(args.source_profile, source_provenance())


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="mode", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--fixture", choices=fixtures.FAMILIES, required=True)
    capture.add_argument("--nodes", type=int, choices=(64, 256), required=True)
    capture.add_argument("--expected-failures", type=int, choices=(1, 2, 11), required=True)
    capture.add_argument("--source-profile", choices=PROFILES, default="baseline430")
    capture.add_argument("--expected-source-sha256", required=True)
    capture.add_argument("--device", default="cuda:0")
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--timeout-seconds", type=int, default=3300)
    return result


def main():
    args = parser().parse_args()
    require(args.timeout_seconds > 0, "Capture timeout must be positive")
    journal = mixed.Journal(args.out)
    receipt = dict(
        schema=SCHEMA if args.source_profile == "baseline430" else PROFILE_SCHEMA,
        status="running",
        source=source_provenance(),
        frozen_commit=PROFILES[args.source_profile]["commit"],
        source_profile=args.source_profile,
        fixture_family=args.fixture,
        nodes=args.nodes,
        expected_failures=args.expected_failures,
        policy=asdict(CudaPolicy()),
        captures=[],
        started_utc=common.utc_now(),
        command=sys.argv,
        artifact_directory=journal.root.name,
        events_file=journal.path.name,
        capture_script_sha256=common.sha(Path(__file__)),
        helpers=helper_hashes(),
        scope="Diagnostic capture with instrumented timing; no scientific fit pass",
    )

    def interrupted(signum, frame):
        raise TimeoutError(f"Capture interrupted by signal{signum}")

    old_alarm, old_term = (
        signal.signal(signal.SIGALRM, interrupted),
        signal.signal(signal.SIGTERM, interrupted),
    )
    signal.alarm(args.timeout_seconds)
    code = 0
    try:
        execute(args, receipt, journal)
        require(
            receipt["helpers"] == helper_hashes()
            and receipt["capture_script_sha256"] == receipt["helpers"]["driver"],
            "Capture or loaded helper files changed during capture",
        )
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        journal.record("capture_failed", error_type=type(error).__name__, error=str(error))
        code = 1
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGTERM, old_term)
        receipt.update(
            finished_utc=common.utc_now(),
            elapsed_seconds=perf_counter() - journal.started,
            artifacts=journal.artifacts,
            events_sha256=common.sha(journal.path),
        )
        common.write_json(journal.receipt_path, receipt)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
