"""Prespecified prior-proposal experiment, disabled in the production interface.

Truth is deliberately absent from every proposal/fitting function in this file.
CPU execution is allowed only through explicit component-test calls; the CLI
requires allocated CUDA. Production defaults and continuation remain unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from clipp1d.api import source_provenance
from clipp1d.cuda.ancestry import ancestry_step, membership_hash, refit_identity
from clipp1d.cuda.experimental import solve_extra_start
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.partition import QualifiedPilot, grouping, refit_labels
from clipp1d.cuda.partition_search import PartitionCandidate, PartitionSearch, PartitionSearchResult
from clipp1d.cuda.policy import CudaPolicy, QualificationError
from clipp1d.cuda.refinement import PartitionSearchPolicy, refine_memberships
from clipp1d.cuda.scalar import pilot
from clipp1d.cuda.selection import fit_tensor_model
from clipp1d.cuda_api import (
    _export, _model_hash, _publish, _tensor_sha256, _validate_refit, require_cuda,
)
from clipp1d.io import read_tumor
from clipp1d.model import compile_model

BASE_COMMIT = "da651879c2a33f481957ffae1d1c865d11ad39c8"
MASTER_SEED = 2026092600
MAGNITUDES = (0.01, 0.03, 0.05)
DEDUP_TOLERANCE = 1e-8
MAX_STARTS = 4


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, record):
    with Path(path).open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")


def array_record(path, **arrays):
    """Save exact values, including invalid -inf prior lanes, outside JSON."""
    host = {name: value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
            else np.asarray(value) for name, value in arrays.items()}
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **host)
    return dict(path=Path(path).name, sha256=sha(path), arrays={
        name: dict(shape=list(value.shape), dtype=value.dtype.str,
                   sha256=hashlib.sha256(value.tobytes()).hexdigest())
        for name, value in host.items()})


def synchronized_time(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return perf_counter()


def model_identity(model):
    model.validate(full=True)
    arrays = {name: getattr(model, name).detach().cpu().numpy() for name in
              ("alt", "ref", "lower", "upper", "slope", "log_prior")}
    arrays["valid"] = np.isfinite(arrays["log_prior"])
    return _model_hash(model.mutation_ids, model.eps, arrays)


def keyed_directions(tumor_id, mutation_ids, width, seed=MASTER_SEED):
    """Four ID-keyed independent streams, followed by their paired opposites.

    SHA-256's first 53 bits map to the fixed binary64 uniform grid in [-1,1).
    JSON key encoding avoids delimiter collisions and depends on no row index.
    """
    if not tumor_id or len(set(mutation_ids)) != len(mutation_ids) or width < 1:
        raise ValueError("Unique mutation IDs, tumor identity and support width are required")
    directions = np.empty((8, len(mutation_ids), width), dtype=np.float64)
    for b in range(4):
        for i, mutation_id in enumerate(mutation_ids):
            for m in range(width):
                key = json.dumps([seed, tumor_id, mutation_id, m+1, b],
                                 ensure_ascii=False, separators=(",", ":")).encode()
                u = (int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 11) / 2**53
                directions[2*b, i, m] = 2*u-1
        directions[2*b+1] = -directions[2*b]
    return directions


def perturb_log_prior(log_prior, direction, eta):
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    if (direction.shape != log_prior.shape or direction.dtype != log_prior.dtype or
            direction.device != log_prior.device or not bool(torch.isfinite(direction).all())):
        raise ValueError("Perturbation must be finite and match the prior")
    # Exact no-op: no logsumexp, copy, or rounding at eta=0.
    if eta == 0:
        return log_prior
    valid = torch.isfinite(log_prior)
    if not bool(valid.any(-1).all() & (valid | torch.isneginf(log_prior)).all()):
        raise ValueError("Invalid original prior support")
    # Centering removes row-common offsets before arithmetic. Singleton rows
    # retain their exact original values, including invalid lanes.
    first = valid.to(torch.int64).argmax(-1)
    centered = direction - direction.gather(1, first[:, None])
    tilt = torch.where(valid, log_prior + eta*centered, -torch.inf)
    normalized = tilt - torch.logsumexp(tilt, dim=-1, keepdim=True)
    return torch.where((valid.sum(-1) == 1)[:, None], log_prior, normalized)


def auxiliary_model(model, direction, eta):
    model.validate(full=True)
    if eta == 0:
        return model
    prior = perturb_log_prior(model.log_prior, direction, eta)
    values = {name: getattr(model, name).detach().clone() for name in
              ("alt", "ref", "slope", "lower", "upper")}
    return TensorModel(model.mutation_ids, log_prior=prior.detach().clone(),
                       eps=model.eps, kernels=model.kernels, **values)


def distinct_starts(starts, baseline=None, limit=MAX_STARTS):
    """Deduplicate at max-coordinate 1e-8, then farthest-point RMS selection.

    The original pilot is part of the existing bank, so copies of it add no
    start. Stable numeric direction indices resolve equal-distance ties.
    """
    unique = []
    for index, vector in sorted(starts, key=lambda pair: pair[0]):
        if not bool(torch.isfinite(vector).all()):
            raise ValueError("Nonfinite starting vector")
        old = [v for _, v in unique] + ([] if baseline is None else [baseline])
        if any(bool((vector-v).abs().max() <= DEDUP_TOLERANCE) for v in old):
            continue
        unique.append((index, vector.detach().clone()))
    chosen = []
    while unique and len(chosen) < limit:
        references = ([baseline] if baseline is not None else []) + [v for _, v in chosen]
        def distance(pair):
            return min(float(torch.mean((pair[1]-v).square()).sqrt()) for v in references) if references else 0.
        best = max(range(len(unique)), key=lambda j: (distance(unique[j]), -unique[j][0]))
        chosen.append(unique.pop(best))
    return chosen


def deterministic_starts(model):
    valid = torch.isfinite(model.log_prior)
    multiplicities = torch.arange(1, model.slope.shape[1]+1, device=model.device)
    support = valid.sum(-1)
    if not torch.equal(valid, multiplicities[None, :] <= support[:, None]):
        raise ValueError("Deterministic control requires contiguous integer support starting at one")
    vaf = model.alt / (model.alt+model.ref)
    starts = []
    for k in range(1, 5):
        value = vaf / (model.slope[:, 0] * support.clamp_max(k))
        starts.append((k, value.clamp(model.lower, model.upper)))
    # D deduplicates its four vectors; it does not remove a pilot-equal start.
    return distinct_starts(starts)


def penalty_neighborhood(records, selected):
    positive = [float(row["lambda_value"]) for row in records if row["lambda_value"] > 0]
    if positive != sorted(set(positive)):
        raise ValueError("Expected the recorded increasing, unique positive baseline path")
    if len(positive) <= 3:
        return positive
    middle = 0 if selected == 0 else min(range(len(positive)), key=lambda i: abs(positive[i]-selected))
    first = max(0, min(middle-1, len(positive)-3))
    return positive[first:first+3]


def scalar_record(result):
    return dict(qualified=bool(result.qualified.all()), maximum_gap=float(result.gap.max()),
                total_gap=float(result.gap.sum()), subdivisions=result.subdivisions)


def pilot_proposals(model, original_pilot, tumor_id, directory, eta=0.03, policy=CudaPolicy()):
    directory = Path(directory)
    directory.mkdir()
    noise = keyed_directions(tumor_id, model.mutation_ids, model.slope.shape[1])
    noise_record = array_record(directory / "directions.npz", directions=noise,
                               mutation_ids=np.asarray(model.mutation_ids))
    original_identity = model_identity(model)
    candidates, records = [], []
    support = torch.isfinite(model.log_prior).sum(-1)
    began = synchronized_time(model.device)
    for index, array in enumerate(noise):
        start = synchronized_time(model.device)
        direction = torch.tensor(array, dtype=torch.float64, device=model.device)
        auxiliary = None
        record = dict(direction=index, eta=eta,
                      original_model_sha256=original_identity, status="unresolved")
        arrays = dict(direction=direction)
        try:
            auxiliary = auxiliary_model(model, direction, eta)
            record['model_sha256'] = model_identity(auxiliary)
            arrays['log_prior'] = auxiliary.log_prior
            fitted = pilot(auxiliary, policy)
            QualifiedPilot(auxiliary, fitted, policy)  # Reject invalid auxiliary certificates here.
            shift = (fitted.phi-original_pilot).abs()
            candidates.append((index, fitted.phi.detach().clone()))
            arrays.update(phi=fitted.phi, loss=fitted.loss, gap=fitted.gap,
                          alternative=fitted.alternative,
                          alternative_loss=auxiliary.loss(fitted.alternative),
                          posterior=auxiliary.terms(fitted.phi)[3])
            record.update(status="qualified", scalar=scalar_record(fitted),
                          substantial_shift_fraction=float((shift >= .05).double().mean()),
                          shift_quantiles=torch.quantile(shift, shift.new_tensor([0., .25, .5, .75, .95, 1.])).tolist(),
                          alternative_search_scope="finite solver-reported candidate bank; absence is not a uniqueness proof",
                          support_strata={str(k): dict(count=int((support == k).sum()),
                            substantial_shift_fraction=float((shift[support == k] >= .05).double().mean()))
                            for k in range(1, 5) if bool((support == k).any())})
        except (QualificationError, torch.OutOfMemoryError) as error:
            record.update(error=str(error), error_type=type(error).__name__,
                          diagnostics=getattr(error, 'diagnostics', {}))
        finally:
            record["arrays"] = array_record(directory / f"pilot-{index}.npz", **arrays)
            record["seconds"] = synchronized_time(model.device)-start
            records.append(record)
        del auxiliary
    selected = distinct_starts(candidates, original_pilot)
    receipt = dict(eta=eta, noise=noise_record, records=records,
                   selected_directions=[i for i, _ in selected],
                   status="complete" if all(r["status"] == "qualified" for r in records) else "incomplete",
                   seconds=synchronized_time(model.device)-began, selection_uses_truth=False)
    write_json(directory / "PILOTS.json", receipt)
    if model_identity(model) != original_identity:
        raise ValueError("Auxiliary pilot generation changed the original model")
    return selected, receipt


def _new_candidate(raw, fitted, penalty, origin, separate):
    return PartitionCandidate("raw_fusion_path", fitted, raw, fitted, penalty,
                              origin, separate, {})


def _refined_candidate(parent, fitted, rounds, status, search_policy):
    step = ancestry_step("reassignment", parent.refit, fitted, dict(rounds=rounds), search_policy.minimum_decrease)
    proposal = dict(algorithm="partition_ancestry_v2", parent_origin=parent.origin,
                    root_origin=parent.origin, ancestry=[step], status=status,
                    truth_used=False, raw_certificate_inherited=False)
    return PartitionCandidate("direct_partition", fitted, parent.raw_reference,
        parent.reference_refit, parent.reference_lambda, "refinement:"+parent.origin,
        parent.separate_exact_one, proposal, (parent.refit,))


@torch.no_grad()
def additional_search(baseline, starts, penalties, directory, arm, *, pilot_coverage="complete"):
    """B remains eligible even when all auxiliary solves or refits fail.

    Only a winning extra raw candidate retains a dense dual. All other starts
    are solved and exported sequentially, never stored in a dense workspace bank.
    """
    directory = Path(directory)
    directory.mkdir()
    model, graph, policy = baseline.model, baseline.graph, baseline.policy
    baseline.partition_estimate.validate_identity()
    incumbent = baseline.partition_estimate.candidate
    search_policy = baseline.partition_estimate.policy
    if search_policy.max_rounds != 4:
        raise ValueError("This experiment prespecifies four extra refinement rounds")
    best = incumbent
    original_identity, graph_identity = model_identity(model), _tensor_sha256(graph.weights)
    seen, records = set(), []
    public_records = deepcopy(baseline.partition_estimate.records)
    if pilot_coverage != "complete":
        public_records.append(dict(origin=f'{arm}:auxiliary_pilots', status='unresolved',
                                   coverage_status='unresolved'))
    original = {float(r['lambda_value']): r.get('raw_objective') for r in baseline.records}
    began = synchronized_time(model.device)
    incomplete = pilot_coverage != "complete"
    for penalty in penalties:
        if penalty not in original or penalty <= 0:
            raise ValueError("Extra penalty is not a positive absolute baseline penalty")
        lam = model.alt.new_tensor(penalty)
        for index, vector in starts:
            start = synchronized_time(model.device)
            origin = f"{arm}:lambda={penalty.hex()}:start={index}"
            row = dict(origin=origin, candidate_family="raw_fusion_path", start=index,
                       reference_lambda=penalty, model_sha256=original_identity,
                       graph_weights_sha256=graph_identity, status="unresolved",
                       original_objective_baseline=original[penalty])
            arrays = dict(start=vector)
            raw = result = candidate = fitted = None
            try:
                result = solve_extra_start(model, graph, lam, vector, policy)
                result.validate(model, graph, lam, policy)
                row['raw_seconds'] = synchronized_time(model.device)-start
                raw = result.raw
                arrays["raw_phi"] = raw.x
                row.update(raw_qualified=raw.qualified, raw_diagnostics=raw.diagnostics,
                           raw_objective=float(raw.objective) if bool(torch.isfinite(raw.objective)) else None)
                if not raw.qualified:
                    raise QualificationError("Additional raw start unresolved", **raw.diagnostics)
                row['raw_improvement'] = None if original[penalty] is None else original[penalty]-float(raw.objective)
                labels = grouping(raw.x, policy.fusion_tol, separate_clonal=search_policy.separate_exact_one)[2]
                signature = membership_hash(labels)
                row['membership_sha256'] = signature
                arrays['memberships'] = labels
                if signature in seen:
                    row['status'] = 'duplicate_memberships'
                    continue
                # Fresh scalar qualification under the original prior: no auxiliary cache.
                refit_started = synchronized_time(model.device)
                fitted = refit_labels(model, labels, policy)
                _validate_refit(model, fitted, policy)
                row['refit_seconds'] = synchronized_time(model.device)-refit_started
                seen.add(signature)
                row['initial_refit'] = dict(refit_identity(fitted), loss=float(fitted.loss),
                                            allocation_complexity=float(fitted.score-2*fitted.loss))
                candidate = _new_candidate(raw, fitted, lam, origin, search_policy.separate_exact_one)
                public_records.append(dict(origin=origin, candidate_family='raw_fusion_path',
                    status='qualified', score=float(fitted.score), clusters=fitted.centers.numel(),
                    reference_lambda=penalty, refit_gap=float(fitted.gap)))
                if PartitionSearch.key(candidate) < PartitionSearch.key(best):
                    best = candidate
                refine_started = synchronized_time(model.device)
                refined, rounds, status = refine_memberships(model, fitted, policy, search_policy)
                _validate_refit(model, refined, policy)
                row['refinement_seconds'] = synchronized_time(model.device)-refine_started
                if bool(refined.score < fitted.score):
                    candidate = _refined_candidate(candidate, refined, rounds, status, search_policy)
                    public_records.append(dict(origin=candidate.origin, candidate_family='direct_partition',
                        status='qualified', score=float(refined.score), clusters=refined.centers.numel(),
                        reference_lambda=penalty, refit_gap=float(refined.gap), proposal=candidate.proposal,
                        coverage_status=status))
                else:
                    public_records.append(dict(origin=f'refinement:{origin}', status='no_improvement',
                                               coverage_status=status))
                if PartitionSearch.key(candidate) < PartitionSearch.key(best):
                    best = candidate
                incomplete |= status != "fixed_point"
                arrays.update(refitted_phi=candidate.refit.phi, refined_memberships=candidate.refit.labels,
                              centers=candidate.refit.centers)
                row.update(status="qualified", coverage_status=status, refinement_rounds=rounds,
                           refit=dict(refit_identity(candidate.refit), loss=float(candidate.refit.loss),
                                      allocation_complexity=float(candidate.refit.score-2*candidate.refit.loss)),
                           candidate_family=candidate.family, proposal=candidate.proposal)
            except (QualificationError, torch.OutOfMemoryError) as error:
                incomplete = True
                row.update(status="unresolved", error=str(error), error_type=type(error).__name__,
                           failure_diagnostics=getattr(error, 'diagnostics', {}))
                public_records.append(dict(origin=f'unresolved:{origin}', status='unresolved',
                                           error=str(error), coverage_status='unresolved'))
            finally:
                row['arrays'] = array_record(directory / f"candidate-{len(records):03}.npz", **arrays)
                row['seconds'] = synchronized_time(model.device)-start
                records.append(row)
            del raw, result, fitted, candidate
    best.validate_identity()
    if not bool(best.refit.score <= incumbent.refit.score):
        raise QualificationError("Additional search lost the B incumbent")
    if model_identity(model) != original_identity or _tensor_sha256(graph.weights) != graph_identity:
        raise ValueError("Additional search modified the original model or graph")
    coverage = "incomplete" if incomplete or baseline.partition_estimate.status != "complete" else "complete"
    result = PartitionSearchResult(best, public_records, coverage, search_policy,
                                   synchronized_time(model.device)-began, incumbent)
    write_json(directory / "SEARCH.json", dict(arm=arm, records=records, status=coverage,
        penalties=penalties, starting_vectors=len(starts), attempted_raw_solves=len(records),
        distinct_refitted_memberships=len(seen), selected_origin=best.origin,
        baseline=refit_identity(incumbent.refit), selected=refit_identity(best.refit),
        comparison=ancestry_step("comparison_only", incumbent.refit, best.refit, {}, search_policy.minimum_decrease),
        seconds=result.seconds, extra_birth_generations=0, original_prior_restored=True))
    return result


def prepare_model(input_file, device, compiled=True, *, kernels=None):
    data = read_tumor(input_file)
    host = compile_model(data)
    host = host.subset(np.argsort(np.asarray(host.mutation_ids), kind='stable'))
    if kernels is None:
        return data, TensorModel.from_host(host, device, compiled)
    arrays = {name: torch.tensor(getattr(host, name).copy(), dtype=torch.float64, device=device)
              for name in ('alt', 'ref', 'slope', 'log_prior', 'lower', 'upper')}
    return data, TensorModel(host.mutation_ids, eps=host.eps, kernels=kernels, **arrays)


def _publish_arm(fitted, source, data, folder):
    folder.mkdir()
    public = _export(fitted, source)
    return _publish(public, data, folder, fitted.records)


def run_case(input_file, directory, *, tumor_id, device="cuda:0", compiled=True):
    """Run independent R/B fits, then isolated P/D arms on B's identical graph."""
    device = require_cuda(device)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    # synchronize initializes CUDA in a fresh child. The allocator-stat reset
    # itself does not, even when require_cuda accepts the visible device.
    began = synchronized_time(device)
    torch.cuda.reset_peak_memory_stats(device)
    data, model = prepare_model(input_file, device, compiled)
    source = source_provenance()
    source.update(input_sha256=data.input_sha256, max_major_cn=4,
                  experimental_base_commit=BASE_COMMIT, experimental_policy="prior_proposals_v1",
                  numerical_device=str(device), backend="cuda" if compiled else "eager_cuda_reference")
    preparation = synchronized_time(device)-began
    with torch.no_grad():
        start = synchronized_time(device)
        raw_fit = fit_tensor_model(model)
        raw_public = _publish_arm(raw_fit, source, data, directory / "R")
        r_seconds = preparation+synchronized_time(device)-start
        r_graph = _tensor_sha256(raw_fit.graph.weights)
        r_pilot = raw_fit.pilot.phi.detach().cpu().clone()
        r_phi = raw_fit.raw.x.detach().cpu().clone()
        r_path = [(r['lambda_value'], r.get('raw_objective')) for r in raw_fit.records]
        del raw_fit
        gc.collect()
        start = synchronized_time(device)
        baseline = fit_tensor_model(model, partition_search=PartitionSearchPolicy())
        baseline_public = _publish_arm(baseline, source, data, directory / "B")
        b_seconds = preparation+synchronized_time(device)-start
        if (r_graph != _tensor_sha256(baseline.graph.weights) or
                not torch.equal(r_pilot, baseline.pilot.phi.cpu()) or
                not torch.equal(r_phi, baseline.raw.x.cpu()) or
                r_path != [(r['lambda_value'], r.get('raw_objective')) for r in baseline.records]):
            raise QualificationError("R and B changed raw continuation or the original graph")
        penalties = penalty_neighborhood(baseline.records, float(baseline.lambda_value))
        common = array_record(directory / 'original.npz', mutation_ids=np.asarray(model.mutation_ids),
            alt=model.alt, ref=model.ref, slope=model.slope, log_prior=model.log_prior,
            lower=model.lower, upper=model.upper, pilot=baseline.pilot.phi)
        start = synchronized_time(device)
        starts, pilots = pilot_proposals(model, baseline.pilot.phi, tumor_id, directory / "P-pilots")
        p = additional_search(baseline, starts, penalties, directory / "P-search", "P", pilot_coverage=pilots['status'])
        p_public = _publish_arm(replace(baseline, partition_estimate=p), source, data, directory / "P")
        p_seconds = b_seconds+synchronized_time(device)-start
        del p
        start = synchronized_time(device)
        starts_d = deterministic_starts(model)
        d = additional_search(baseline, starts_d, penalties, directory / "D-search", "D")
        d_public = _publish_arm(replace(baseline, partition_estimate=d), source, data, directory / "D")
        d_seconds = b_seconds+synchronized_time(device)-start
        records = dict(schema="clipp1d.prior_proposals.v1", utc=now(), tumor_id=tumor_id,
            input_sha256=sha(input_file), source=source, original_arrays=common,
            policy=asdict(CudaPolicy()), partition_policy=asdict(PartitionSearchPolicy()),
            model_sha256=model_identity(model), graph_sha256=raw_public.graph_sha256,
            graph_weights_sha256=r_graph, graph_recipe=raw_public.provenance['graph'],
            selected_baseline_lambda=float(baseline.lambda_value), extra_penalties=penalties,
            lambda_reference=baseline.timings['lambda_reference'],
            seconds=dict(R=r_seconds, B=b_seconds, P=p_seconds, D=d_seconds),
            timing_scope="synchronized; P/D charged full B including preparation, final qualification and output publication",
            actual_experiment_seconds=synchronized_time(device)-began,
            peak_device_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_memory_scope="combined sequential R/B/P/D experiment; per-arm cold/warm runs required for acceptance",
            original_raw_preserved=all(np.array_equal(raw_public.raw_phi, v.raw_phi)
                                      for v in (baseline_public, p_public, d_public)),
            result_scores=dict(R=raw_public.selection_score, B=baseline_public.partition_estimate.score,
                               P=p_public.partition_estimate.score, D=d_public.partition_estimate.score),
            truth_used=False)
        if not records['original_raw_preserved']:
            raise QualificationError("An experimental arm altered primary raw output")
        write_json(directory / 'RESULT.json', records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--tumor-id', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run_case(args.input_file, args.outdir, tumor_id=args.tumor_id, device=args.device)


if __name__ == '__main__':
    main()
