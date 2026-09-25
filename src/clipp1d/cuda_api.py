"""CUDA-only public fit; CPU input canonicalization and final serialization.

Numerical likelihoods, pilots, graph weights, optimization, certificates, path
scores, secondary refits, and multiplicity posteriors remain on the CUDA device.
The existing canonical input/model compiler is reused once before the upload.
"""
from dataclasses import asdict, dataclass, replace
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from .cuda.model import TensorModel
from .cuda.policy import CudaPolicy, QualificationError
from .cuda.selection import fit_tensor_model

SCHEMA = "clipp1d.cuda.run.v3"
PARTITION_SCHEMA = "clipp1d.cuda.run.v4"
BASE_COMMIT = "371003ffcadcc57e23c62df5901d9463085cceea"


@dataclass(frozen=True)
class FitResult:
    mutation_ids: tuple[str, ...]
    pilot_phi: np.ndarray
    raw_phi: np.ndarray
    refitted_phi: np.ndarray
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    multiplicity_calls: np.ndarray
    refitted_multiplicity_calls: np.ndarray
    selected_lambda: float
    selection_score: float
    raw_objective: float
    raw_witness_mutation_id: str | None
    raw_diagnostics: dict
    search_status: str
    provenance: dict
    graph_sha256: str
    original_lower_bounds: np.ndarray
    original_upper_bounds: np.ndarray
    partition_labels: np.ndarray
    candidate_provenance: dict
    operation_metrics: dict | None = None  # Return-only completion; never a self-timed receipt.
    partition_estimate: object = None  # Separately qualified development estimator, never a raw fit.

    def __post_init__(self):
        for name in ("pilot_phi", "raw_phi", "refitted_phi", "cluster_labels", "cluster_centers",
                     "multiplicity_calls", "refitted_multiplicity_calls", "original_lower_bounds",
                     "original_upper_bounds", "partition_labels"):
            value = np.ascontiguousarray(getattr(self, name))
            immutable = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            object.__setattr__(self, name, immutable)
        for name in ("raw_diagnostics", "provenance", "candidate_provenance", "operation_metrics"):
            object.__setattr__(self, name, deepcopy(getattr(self, name)))
        object.__setattr__(self, "_metadata_sha256", self._metadata_hash())

    def _metadata_hash(self):
        payload = dict(raw=self.raw_diagnostics, source=self.provenance, candidate=self.candidate_provenance,
                       operation=self.operation_metrics,
                       partition=None if self.partition_estimate is None else self.partition_estimate.identity())
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def require_cuda(device):
    d = torch.device(device)
    if d.type != "cuda":
        raise ValueError("Production inference requires an explicit CUDA device; CPU is test-reference only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback was performed")
    index = torch.cuda.current_device() if d.index is None else d.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("Requested CUDA device index is unavailable")
    return torch.device("cuda", index)


def _json(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [clean(x) for x in v]
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    payload = json.dumps(clean(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".clipp1d-json-", delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _host(tensor):
    value = tensor.detach().cpu().numpy().copy()
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _tensor_sha256(value):
    """Hash final device state in bounded row chunks after inference completes."""
    digest = hashlib.sha256()
    rows = value.reshape(1) if value.ndim == 0 else value
    for start in range(0, len(rows), 128):
        digest.update(_host(rows[start:start + 128]).tobytes())
    return digest.hexdigest()


def _model_hash(mutation_ids, eps, arrays):
    digest = hashlib.sha256(json.dumps(dict(mutation_ids=mutation_ids, eps=eps), sort_keys=True).encode())
    for key in ("alt", "ref", "lower", "upper", "slope", "log_prior", "valid"):
        value = np.ascontiguousarray(arrays[key])
        digest.update(json.dumps([key, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_metrics(model):
    """Read after ALL final CUDA qualification, posterior and export work."""
    _synchronize(model.device)
    metrics = dict(compilation_statistics=model.kernels.compilation_diagnostics(),
                   integrity_counters=dict(getattr(model, "integrity_counters", {})),
                   scalar_work_counters=dict(getattr(model, "scalar_work_counters", {})),
                   device_measurement_scope="upload through final qualification and completed device export")
    if model.device.type == "cuda":
        metrics.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(model.device))
    return metrics


def _validate_device_fit(d):
    # Contexts own full snapshot checks at entry and exit, including exceptions.
    with d.model.validated_stage(), d.graph.validated_stage():
        _validate_device_fit_owned(d)
        if d.partition_estimate is not None:
            from .partition_output import validate_device_partition
            validate_device_partition(d)


def _validate_device_fit_owned(d, *, separate_exact_one=True):
    """Reconcile the selected device states before the final host boundary."""
    from .cuda.audit import audit_raw
    from .cuda.kernels import differences
    from .cuda.partition import grouping
    m, p = d.model, d.policy
    m.validate()
    d.graph.validate()

    def require(value, message):
        if not bool(value):
            raise QualificationError("Refusing to export " + message)

    require(m.mutation_ids == tuple(sorted(m.mutation_ids)) and len(m.mutation_ids) == m.n and
            d.graph.mutation_ids == m.mutation_ids, "unbound canonical model/graph identities")
    require(m.eps == p.eps, "likelihood clipping different from the declared inference policy")
    require(d.graph.n == m.n and torch.equal(d.graph.pilot, d.pilot.phi), "mismatched frozen graph pilot")
    require(d.raw.qualified and d.raw.witness is None and
            d.lambda_value.ndim == 0 and torch.isfinite(d.lambda_value) and d.lambda_value >= 0,
            "unqualified raw state or invalid selected penalty")
    for name, values in (("raw", d.raw.x), ("pilot", d.pilot.phi), ("refit", d.refit.phi)):
        require(values.shape == m.lower.shape and values.dtype == torch.float64 and values.device == m.device,
                "invalid " + name + " node state")
        require(torch.isfinite(values).all() & (values >= m.lower).all() & (values <= m.upper).all(),
                name + " state outside original bounds")
    require(d.pilot.qualified.all() & torch.isfinite(d.pilot.gap).all() & (d.pilot.gap >= 0).all() &
            torch.isfinite(d.pilot.loss).all() & torch.isfinite(d.pilot.lower_bound).all() &
            torch.equal(d.pilot.gap, (d.pilot.loss - d.pilot.lower_bound).clamp_min(0.)) &
            (d.pilot.gap <= p.scalar_atol + p.scalar_rtol * d.pilot.loss.abs()).all(),
            "unqualified scalar pilot")
    pilot_loss = m.loss(d.pilot.phi)
    require(((pilot_loss - d.pilot.loss).abs() <=
             128 * torch.finfo(torch.float64).eps * (1 + pilot_loss.abs())).all(),
            "pilot loss inconsistent with the model")
    caps = d.graph.weights * d.lambda_value
    observed_objective = m.loss(d.raw.x).sum() + .5 * (caps * differences(d.raw.x).abs()).sum()
    objective_margin = 128 * torch.finfo(torch.float64).eps * (1 + observed_objective.abs())
    require(torch.isfinite(d.raw.objective) & ((d.raw.objective - observed_objective).abs() <= objective_margin),
            "raw objective inconsistent with the exact model/graph state")
    if bool(d.lambda_value == 0):
        require(torch.equal(d.raw.x, d.pilot.phi) and bool((d.raw.dual == 0).all()),
                "separable raw state not bound to the qualified pilot")
    else:
        audit = audit_raw(m, d.raw.x, d.raw.dual, caps, None, p)
        require(audit.qualified, "raw state failing independent final stationarity audit")
    expected_labels = grouping(d.raw.x, p.fusion_tol, separate_clonal=separate_exact_one)[2]
    require(torch.equal(expected_labels, d.refit.labels), "memberships unrelated to the selected raw state")
    _validate_refit(m, d.refit, p)


def _validate_refit(m, fitted, p):
    """Common final qualification for raw-derived and explicit memberships."""
    from .cuda.partition import canonical_labels, partition_score

    def require(value, message):
        if not bool(value):
            raise QualificationError("Refusing to export " + message)

    canonical, _, lengths = canonical_labels(fitted.labels)
    require(torch.equal(canonical, fitted.labels) and fitted.labels.shape == m.lower.shape and
            fitted.labels.device == m.device, "noncanonical explicit memberships")
    scalar = fitted.scalar
    k = fitted.centers.numel()
    require(scalar.phi.shape == (k,) and torch.equal(scalar.phi, fitted.centers) and
            torch.equal(fitted.phi, fitted.centers[fitted.labels]), "mismatched scalar refit centers")
    require(torch.isfinite(fitted.phi).all() & (fitted.phi >= m.lower).all() & (fitted.phi <= m.upper).all(),
            "explicit refit outside original bounds")
    require(scalar.qualified.all() & torch.isfinite(scalar.loss).all() & torch.isfinite(scalar.lower_bound).all() &
            torch.isfinite(scalar.gap).all() & (scalar.gap >= 0).all() &
            torch.equal(scalar.gap, (scalar.loss - scalar.lower_bound).clamp_min(0.)) &
            (scalar.gap <= p.scalar_atol + p.scalar_rtol * scalar.loss.abs()).all(),
            "unqualified secondary scalar refit")
    require(torch.equal(fitted.sizes, lengths) and
            torch.equal(fitted.gap, scalar.gap.sum()) and torch.equal(fitted.loss, scalar.loss.sum()),
            "inconsistent refit membership sizes or scalar arithmetic")
    refit_loss = m.loss(fitted.phi).sum()
    require((refit_loss - fitted.loss).abs() <= 128 * torch.finfo(torch.float64).eps * (1 + refit_loss.abs()),
            "secondary refit loss inconsistent with the model")
    require(torch.equal(fitted.score, partition_score(fitted.loss, fitted.sizes)),
            "secondary score inconsistent with its exact memberships")
    require(fitted.clonal == int((fitted.centers - 1).abs().argmin()),
            "secondary clonal designation inconsistent with closest-to-one rule")


def _export(device_fit, source, *, wall_started=None):
    d, m = device_fit, device_fit.model
    _synchronize(m.device)
    qualification_started = perf_counter()
    _validate_device_fit(d)
    _synchronize(m.device)
    qualification_seconds = perf_counter() - qualification_started
    export_started = perf_counter()
    source = deepcopy(source)
    arrays = {name: _host(getattr(m, name)) for name in ("alt", "ref", "lower", "upper", "slope", "log_prior")}
    arrays["valid"] = _host(torch.isfinite(m.log_prior))
    model_sha = _model_hash(m.mutation_ids, m.eps, arrays)
    if source.get("model_sha256", model_sha) != model_sha:
        raise QualificationError("Uploaded numerical model does not match its canonical source identity")
    source["model_sha256"] = model_sha
    source["policy"] = asdict(d.policy)
    if source.get("backend") == "cuda" and (m.device.type != "cuda" or not m.kernels.compiled):
        raise QualificationError("CPU or eager reference cannot inherit compiled CUDA execution identity")
    # This is the numerical pipeline's final bulk device-to-host boundary.
    raw_calls = m.terms(d.raw.x)[3].argmax(-1) + 1
    refit_calls = m.terms(d.refit.phi)[3].argmax(-1) + 1
    order = torch.argsort(d.refit.centers, descending=True, stable=True)
    # Preserve label zero as the designated SECONDARY-refit clonal group.
    order = torch.cat((order[order == d.refit.clonal], order[order != d.refit.clonal]))
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=m.device)
    pilots, raw, refitted = _host(d.pilot.phi), _host(d.raw.x), _host(d.refit.phi)
    labels, centers = _host(inverse[d.refit.labels]), _host(d.refit.centers[order])
    graph_recipe = dict(rule=d.graph.weight_rule, representation="symmetric_weights_skew_states",
                        edge_count=d.graph.edges, gap_floor=float(d.graph.gap_floor),
                        normalization=float(d.graph.normalization),
                        pilot_sha256=hashlib.sha256(pilots.tobytes()).hexdigest(),
                        mutation_ids=m.mutation_ids,
                        weights_sha256=_tensor_sha256(d.graph.weights))
    graph_sha = hashlib.sha256(json.dumps(graph_recipe, sort_keys=True).encode()).hexdigest()
    source["graph"] = graph_recipe
    source["pilot_maximum_gap"] = float(d.pilot.gap.max())
    source["refit_gap"] = float(d.refit.gap)
    source["primary_estimator"] = "selected_raw_complete_graph_penalized_candidate"
    source["refit_role"] = "secondary summary and path score only"
    partition_labels = _host(d.refit.labels)
    source.update(clonal_constraint=False, clonal_label_rule="nearest_to_one_l2_v1",
                  clonal_designation_basis="secondary_refitted_cluster_centers",
                  numerical_device=str(m.device), compiled_inference=m.kernels.compiled,
                  execution_scope="cuda_inference" if m.device.type == "cuda" else "cpu_numerical_reference",
                  multiplicity_conditioning={"raw_multiplicity_call": "raw_ccf",
                                             "refitted_multiplicity_call": "refitted_ccf"})
    candidate = dict(candidate_family="complete_graph_path", selected_lambda=float(d.lambda_value),
                     graph_sha256=graph_sha, model_sha256=model_sha,
                     partition_sha256=hashlib.sha256(partition_labels.tobytes()).hexdigest(),
                     raw_phi_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                     refitted_phi_sha256=hashlib.sha256(refitted.tobytes()).hexdigest(),
                     public_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
                     refitted_centers_sha256=hashlib.sha256(centers.tobytes()).hexdigest(),
                     original_lower_sha256=hashlib.sha256(arrays['lower'].tobytes()).hexdigest(),
                     original_upper_sha256=hashlib.sha256(arrays['upper'].tobytes()).hexdigest(),
                     raw_multiplicity_sha256=hashlib.sha256(_host(raw_calls).tobytes()).hexdigest(),
                     refitted_multiplicity_sha256=hashlib.sha256(_host(refit_calls).tobytes()).hexdigest(),
                     raw_qualified=bool(d.raw.qualified), refit_qualified=True,
                     refit_gap=float(d.refit.gap), raw_certificate=dict(d.raw.diagnostics),
                     global_optimality_proven=False)
    # Materialize every remaining device scalar/array BEFORE the measurement.
    raw_multiplicity, refit_multiplicity = _host(raw_calls), _host(refit_calls)
    selected_lambda, score, raw_objective = float(d.lambda_value), float(d.refit.score), float(d.raw.objective)
    partition_estimate = None
    if d.partition_estimate is not None:
        from .partition_output import export_partition
        partition_estimate = export_partition(d, model_sha, graph_sha)
        source.update(partition_search=asdict(d.partition_estimate.policy),
                      partition_estimator_role="separate development estimate; primary raw estimator unchanged")
    m.validate(full=True)
    d.graph.validate(full=True)
    metrics = _device_metrics(m)
    exported_at = perf_counter()
    phases = dict(source.get("phase_seconds", {}))
    phases.update(final_device_qualification_seconds=qualification_seconds,
                  device_export_seconds=exported_at - export_started)
    source.update(metrics, graph_integrity_counters=dict(d.graph.integrity_counters), phase_seconds=phases,
                  numerical_completed_utc=datetime.now(timezone.utc).isoformat(),
                  numerical_completion_scope="all CUDA work and device-to-host transfers complete")
    if wall_started is not None:
        source["through_device_export_seconds"] = exported_at - wall_started
    return FitResult(m.mutation_ids, pilots, raw, refitted, labels, centers,
                     raw_multiplicity, refit_multiplicity, selected_lambda,
                     score, raw_objective, None, d.raw.diagnostics, d.search_status,
                     source, graph_sha, arrays["lower"], arrays["upper"], partition_labels, candidate,
                     partition_estimate=partition_estimate)


def _clonality(result):
    distances = np.abs(result.cluster_centers-1)
    count = int(np.count_nonzero(result.cluster_labels == 0))
    return dict(rule="nearest_to_one_l2_v1", clonal_cluster_label=0,
                clonal_ccf=float(result.cluster_centers[0]), distance_to_one=float(distances[0]),
                tie_count=int(np.count_nonzero(distances == distances[0])),
                tied_cluster_labels=np.flatnonzero(distances == distances[0]).tolist(),
                tie_breaker="lowest canonical mutation-ID node among tied memberships",
                total_mutations=len(result.mutation_ids), clonal_mutations=count,
                subclonal_mutations=len(result.mutation_ids)-count,
                subclonal_mutation_fraction=1-count/len(result.mutation_ids),
                mutation_filter="all retained informative mutations after input filtering")


def _validate_result(result, data, records):
    """Reject mismatched arrays, raw certificates and secondary summary semantics."""
    def require(condition, message):
        if not condition:
            raise QualificationError("Refusing to publish " + message)
    require(result._metadata_hash() == result._metadata_sha256, "modified scientific provenance")
    source_files = result.provenance.get('source_files', {})
    source_digest = hashlib.sha256()
    require(isinstance(source_files, dict) and bool(source_files), "missing source file inventory")
    for name, digest in sorted(source_files.items()):
        source_digest.update(f"{name}\0{digest}\n".encode())
    require(source_digest.hexdigest() == result.provenance.get('source_sha256'),
            "inconsistent source inventory fingerprint")
    n = len(result.mutation_ids)
    require(n > 0 and len(set(result.mutation_ids)) == n and
            result.mutation_ids == tuple(sorted(result.mutation_ids)), "noncanonical mutation IDs")
    retained = tuple(sorted(m.mutation_id for m in data.mutations if m.exclusion is None))
    require(retained == result.mutation_ids, "mismatched retained identities")
    for name in ("pilot_phi", "raw_phi", "refitted_phi", "cluster_labels", "multiplicity_calls",
                 "refitted_multiplicity_calls", "original_lower_bounds", "original_upper_bounds", "partition_labels"):
        values = np.asarray(getattr(result, name))
        require(values.shape == (n,) and np.all(np.isfinite(values)), "invalid " + name)
    lo, hi = result.original_lower_bounds, result.original_upper_bounds
    require(np.all((0 <= lo) & (lo <= hi) & (hi <= 1)), "invalid original bounds")
    for name in ("pilot_phi", "raw_phi", "refitted_phi"):
        values = getattr(result, name)
        require(np.all((lo <= values) & (values <= hi)), name + " outside original domains")
    centers, labels = result.cluster_centers, result.cluster_labels
    require(centers.ndim == 1 and len(centers) > 0 and np.all(np.isfinite(centers)) and
            labels.dtype.kind in 'iu' and np.array_equal(np.unique(labels), np.arange(len(centers))) and
            np.array_equal(result.refitted_phi, centers[labels]), "inconsistent occupied cluster summaries")
    distance = np.abs(centers-1)
    require(distance[0] == distance.min(), "a clonal cluster that is not closest to one")
    tied = np.flatnonzero(distance == distance[0])
    require(np.flatnonzero(labels == 0)[0] == min(np.flatnonzero(labels == k)[0] for k in tied),
            "incorrect canonical clonal tie break")
    require(result.partition_labels.dtype.kind in 'iu' and
            np.array_equal(np.unique(result.partition_labels), np.arange(len(centers))), "invalid memberships")
    for group in range(len(centers)):
        member = result.partition_labels == group
        require(len(np.unique(labels[member])) == 1, "labels inconsistent with raw memberships")
    for name in ("multiplicity_calls", "refitted_multiplicity_calls"):
        value = getattr(result, name)
        require(value.dtype.kind in 'iu' and np.all((value >= 1) & (value <= 4)), "invalid multiplicity calls")
    raw = result.raw_diagnostics
    require(raw.get('box_feasible') is True and raw.get('clonal_constraint') is False and
            raw.get('global_optimality_proven') is False and
            (raw.get('raw_branch_stationarity_qualified') is True or raw.get('separable_scalar_gap_qualified') is True),
            "an unqualified raw estimator")
    require(result.raw_witness_mutation_id is None and result.provenance.get('clonal_constraint') is False,
            "a reintroduced clonal fitting constraint")
    require(np.isfinite(result.selected_lambda) and result.selected_lambda >= 0 and
            np.isfinite(result.raw_objective) and np.isfinite(result.selection_score), "invalid score or penalty")
    lineage = result.candidate_provenance
    require(lineage.get('candidate_family') == 'complete_graph_path' and lineage.get('raw_qualified') is True and
            lineage.get('refit_qualified') is True and lineage.get('graph_sha256') == result.graph_sha256 and
            lineage.get('model_sha256') == result.provenance.get('model_sha256') and
            lineage.get('selected_lambda') == result.selected_lambda and
            lineage.get('raw_certificate') == raw and np.isfinite(lineage.get('refit_gap', np.nan)) and
            lineage['refit_gap'] >= 0,
            "mismatched selected raw/refit provenance")
    for field, value in (("raw_phi", result.raw_phi), ("partition", result.partition_labels),
                         ("refitted_phi", result.refitted_phi), ("public_labels", result.cluster_labels),
                         ("refitted_centers", result.cluster_centers), ("original_lower", lo),
                         ("original_upper", hi), ("raw_multiplicity", result.multiplicity_calls),
                         ("refitted_multiplicity", result.refitted_multiplicity_calls)):
        require(lineage.get(field + '_sha256') == hashlib.sha256(value.tobytes()).hexdigest(),
                "mismatched " + field + " state identity")
    graph_recipe = result.provenance.get('graph', {})
    require(hashlib.sha256(json.dumps(graph_recipe, sort_keys=True).encode()).hexdigest() == result.graph_sha256 and
            tuple(graph_recipe.get('mutation_ids', ())) == result.mutation_ids and
            graph_recipe.get('edge_count') == n * (n - 1) // 2 and
            graph_recipe.get('pilot_sha256') == hashlib.sha256(result.pilot_phi.tobytes()).hexdigest(),
            "mismatched complete-graph recipe or node identities")
    require(result.provenance.get('primary_estimator') == 'selected_raw_complete_graph_penalized_candidate' and
            result.provenance.get('refit_role') == 'secondary summary and path score only',
            "ambiguous raw/refit estimator semantics")
    if result.provenance.get('backend') == 'cuda':
        require(result.provenance.get('numerical_device', '').startswith('cuda') and
                result.provenance.get('compiled_inference') is True and
                result.provenance.get('execution_scope') == 'cuda_inference',
                "CPU/reference work described as compiled CUDA inference")
    require(isinstance(records, list) and bool(records), "an empty or invalid searched path")
    qualified_records = [r for r in records if r.get('raw_status') == r.get('refit_status') == 'qualified']
    require(bool(qualified_records) and all(np.isfinite(r.get('lambda_value', np.nan)) and
            r['lambda_value'] >= 0 and np.isfinite(r.get('score', np.nan)) and
            np.isfinite(r.get('raw_objective', np.nan)) and isinstance(r.get('clusters'), int) and
            r['clusters'] >= 1 for r in qualified_records), "invalid qualified path records")
    selected = [r for r in qualified_records if r['lambda_value'] == result.selected_lambda]
    require(any(r['raw_objective'] == result.raw_objective and r['score'] == result.selection_score and
                r['clusters'] == len(centers) for r in selected),
            "a selected state absent from the qualified path")
    winner = min(qualified_records, key=lambda r: (r['score'], r['clusters'], r['lambda_value']))
    require((result.selection_score, len(centers), result.selected_lambda) ==
            (winner['score'], winner['clusters'], winner['lambda_value']),
            "a path state that does not minimize the declared score/tie rule")
    complete = all(r.get('raw_status') == r.get('refit_status') == 'qualified' and r.get('search_complete', False)
                   for r in records)
    require(result.search_status == ('complete' if complete else 'incomplete'), "false search completeness")
    if result.partition_estimate is not None:
        from .partition_output import validate_partition_result
        validate_partition_result(result)


def _prepared_result(result, preparation_started, wall_started, *, tables_written):
    measured_at = perf_counter()
    provenance = deepcopy(result.provenance)
    phases = dict(provenance.get("phase_seconds", {}))
    phases["output_preparation_seconds"] = measured_at - preparation_started
    provenance.update(phase_seconds=phases,
                      output_prepared_utc=datetime.now(timezone.utc).isoformat(),
                      output_preparation_scope="host validation and prepared TSV tables" if tables_written
                                               else "host validation of in-memory result",
                      elapsed_scope="through output preparation; excludes receipt serialization, durable publication and return bookkeeping",
                      publication_completion_scope="return-only operation_metrics; external caller receipt required for durable completion timing")
    if wall_started is not None:
        provenance["elapsed_seconds"] = measured_at - wall_started
    return replace(result, provenance=provenance)


def _write_bundle(result, data, destination, records, *, preparation_started=None, wall_started=None):
    preparation_started = perf_counter() if preparation_started is None else preparation_started
    _validate_result(result, data, records)
    lookup = {mid: i for i, mid in enumerate(result.mutation_ids)}
    prefix = [data.tumor_id, data.sample_id]
    path = destination / "mutation_clusters.tsv"
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["tumor_id", "sample_id", "mutation_id", "status", "node_index", "pilot_ccf",
                    "raw_ccf", "refitted_ccf", "cluster_label", "designated_clonal"])
        for mutation in data.mutations:
            if mutation.exclusion is not None:
                w.writerow(prefix + [mutation.mutation_id, mutation.exclusion] + ["."] * 6)
            else:
                i = lookup[mutation.mutation_id]
                w.writerow(prefix + [mutation.mutation_id, "retained", i, result.pilot_phi[i],
                            result.raw_phi[i], result.refitted_phi[i], result.cluster_labels[i],
                            int(result.cluster_labels[i] == 0)])
    with (destination / "cluster_centers.tsv").open("x", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["tumor_id", "sample_id", "cluster_label", "cluster_size", "refitted_ccf", "designated_clonal"])
        sizes = np.bincount(result.cluster_labels, minlength=len(result.cluster_centers))
        for label, center in enumerate(result.cluster_centers):
            w.writerow(prefix + [label, sizes[label], center, int(label == 0)])
    with (destination / "mutation_multiplicity.tsv").open("x", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["tumor_id", "sample_id", "mutation_id", "raw_ccf", "raw_multiplicity_call",
                    "refitted_ccf", "refitted_multiplicity_call"])
        for i, mid in enumerate(result.mutation_ids):
            w.writerow(prefix + [mid, result.raw_phi[i], result.multiplicity_calls[i],
                                 result.refitted_phi[i], result.refitted_multiplicity_calls[i]])
    if result.partition_estimate is not None:
        from .partition_output import write_partition_tables
        write_partition_tables(result, data, destination)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob("*.tsv")}
    result = _prepared_result(result, preparation_started, wall_started, tables_written=True)
    receipt_sha256 = _json(destination / "run.json", dict(
          schema=SCHEMA if result.partition_estimate is None else PARTITION_SCHEMA,
          partition_estimate=None if result.partition_estimate is None else result.partition_estimate.receipt(),
          status="success", search_status=result.search_status,
          selected_lambda=result.selected_lambda, raw_objective=result.raw_objective,
          selection_score=result.selection_score, raw_witness_mutation_id=result.raw_witness_mutation_id,
          raw_diagnostics=result.raw_diagnostics, graph_sha256=result.graph_sha256, provenance=result.provenance,
          search=records, table_sha256=hashes, global_optimality_proven=False,
          candidate_provenance=result.candidate_provenance,
          partition_labels=result.partition_labels.tolist(),
          clonality=_clonality(result),
          qualification_scope="per-run numerical admission; not hardware/cohort qualification"))
    return result, receipt_sha256


def _publish(result, data, destination, records, *, wall_started=None):
    """Return post-publication timing separately from the self-excluding run receipt."""
    publication_started = perf_counter()
    _validate_result(result, data, records)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or any(destination.iterdir()):
        raise FileExistsError("Output directory must be empty and must not be a symlink")
    stage = Path(tempfile.mkdtemp(prefix=".clipp1d-publish-", dir=destination.parent))
    try:
        result, receipt_sha256 = _write_bundle(result, data, stage, records,
                               preparation_started=publication_started, wall_started=wall_started)
        receipt_bytes = (stage / "run.json").read_bytes()
        if hashlib.sha256(receipt_bytes).hexdigest() != receipt_sha256:
            raise QualificationError("Published receipt readback differs from the intended result")
        receipt = json.loads(receipt_bytes)
        for name, expected in receipt['table_sha256'].items():
            path = stage / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise QualificationError("Published table readback failed", file=name)
        for path in stage.iterdir():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        descriptor = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        # Replacing an empty directory is one filesystem operation. If another
        # writer added evidence, the OS refuses to replace that nonempty target.
        os.replace(stage, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    completed = perf_counter()
    metrics = dict(publication_completed_utc=datetime.now(timezone.utc).isoformat(),
                   publication_seconds=completed - publication_started,
                   scope="completed TSV/receipt readback, fsync, atomic directory publication and parent fsync; excludes return bookkeeping")
    if wall_started is not None:
        metrics["elapsed_seconds"] = completed - wall_started
    return replace(result, operation_metrics=metrics)


@torch.no_grad()
def fit(input_file, outdir=None, *, max_major_cn=4, verbose=False, device="cuda:0",
        partition_search=False, generic_partition_grouping=False):
    """Fit one sample on CUDA, default float64, with strict compiled tensor kernels.

    An unavailable CUDA device or compiler error is fatal. Numerical candidates
    retain separate qualification and search-completeness statuses.
    """
    start = perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    destination = None if outdir is None else Path(outdir)
    if destination is not None:
        if destination.is_symlink():
            raise FileExistsError("Output directory must not be a symlink; existing targets are preserved")
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise FileExistsError("Output directory must be empty; prior evidence is never overwritten")
    from .api import source_provenance
    source = source_provenance()
    source.update(started_utc=started_utc, requested_device=str(device),
                  base_commit=BASE_COMMIT)
    try:
        if not isinstance(partition_search, bool) or not isinstance(generic_partition_grouping, bool):
            raise ValueError("Partition search controls must be boolean")
        if generic_partition_grouping and not partition_search:
            raise ValueError("Generic partition grouping requires partition_search")
        d = require_cuda(device)
        compiler = os.environ.get("CC")
        if not compiler or shutil.which(compiler) is None:
            raise RuntimeError("Set CC to an available C compiler before starting the CUDA worker")
        source["cc"] = compiler
        from .io import read_tumor
        from .model import compile_model
        from .policy import Policy
        input_policy = Policy(max_major_cn=max_major_cn)
        policy = CudaPolicy()
        data = read_tumor(input_file, input_policy)
        canonical = compile_model(data, input_policy)
        canonical = canonical.subset(np.argsort(np.asarray(canonical.mutation_ids), kind="stable"))
        source.update(input_sha256=data.input_sha256, max_major_cn=max_major_cn,
                      policy=asdict(policy), dtype="float64", backend="cuda", device=str(d),
                      clonal_constraint=False, clonal_label_rule="nearest_to_one_l2_v1",
                      torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                      compiled_kernel_policy="torch.compile(fullgraph=True, dynamic=True); bounded_structural_families_v1; no CPU fallback")
        source["model_sha256"] = _model_hash(canonical.mutation_ids, canonical.eps,
            {key: getattr(canonical, key) for key in ("alt", "ref", "lower", "upper", "slope", "log_prior", "valid")})
        props = torch.cuda.get_device_properties(d)
        free, total = torch.cuda.mem_get_info(d)
        # Conservative preflight, not a claimed exact memory bound for compiler allocations.
        estimated = 32 * 8 * len(canonical) ** 2 + 64 * 4096 * 8 * 4
        if partition_search:
            estimated += 4 * 8 * len(canonical) ** 2  # Additional raw reference and worst-case N-by-K costs.
        if estimated > policy.memory_fraction * free:
            raise MemoryError(f"Dense complete-graph preflight needs approximately {estimated} bytes; {free} free")
        source.update(gpu_name=props.name, capability=list(torch.cuda.get_device_capability(d)),
                      total_device_bytes=total, free_device_bytes_at_start=free,
                      estimated_workspace_bytes=estimated, cpu_numeric_fallback=False)
        _synchronize(d)
        phases = dict(input_preparation_seconds=perf_counter() - start)
        torch.cuda.reset_peak_memory_stats(d)
        upload_started = perf_counter()
        model = TensorModel.from_host(canonical, d, compiled=True)
        _synchronize(d)
        phases["device_upload_and_compile_seconds"] = perf_counter() - upload_started
        from .cuda.refinement import PartitionSearchPolicy
        search = PartitionSearchPolicy(separate_exact_one=not generic_partition_grouping) if partition_search else None
        result_device = fit_tensor_model(model, policy, partition_search=search)
        _synchronize(d)
        for key in ("pilot_seconds", "graph_build_seconds", "path_seconds", "refit_seconds", "stage_integrity_seconds"):
            phases[key] = result_device.timings[key]
        phases["partition_search_seconds"] = result_device.timings["partition_search_seconds"]
        source.update(numerical_stages=result_device.timings, phase_seconds=phases,
                      compilation_timing_scope="initial compiler admission in upload phase; lazy specializations charged to the executing phase")
        result = _export(result_device, source, wall_started=start)
        source = deepcopy(result.provenance)
        if destination is not None:
            result = _publish(result, data, destination, result_device.records, wall_started=start)
        else:
            preparation_started = perf_counter()
            _validate_result(result, data, result_device.records)
            result = _prepared_result(result, preparation_started, start, tables_written=False)
            result = replace(result, operation_metrics=dict(
                elapsed_seconds=perf_counter() - start,
                completed_utc=datetime.now(timezone.utc).isoformat(),
                scope="complete in-memory fit; no durable output requested"))
        if verbose:
            print(f"device={d} retained={len(result.mutation_ids)} edges={result_device.graph.edges} "
                  f"search_status={result.search_status}")
        return result
    except Exception as error:
        if destination is not None and not (destination / "run.json").exists():
            _json(destination / "run.json", dict(schema=PARTITION_SCHEMA if partition_search else SCHEMA,
                  status="failure", search_status="not_completed",
                  error_type=type(error).__name__, message=str(error), provenance=source,
                  diagnostics=getattr(error, "diagnostics", {})))
        raise
