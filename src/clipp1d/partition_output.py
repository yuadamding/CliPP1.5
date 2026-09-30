"""Versioned publication of a separate partition estimator and its raw reference."""
from copy import deepcopy
from dataclasses import dataclass, replace, asdict
import csv
import hashlib
import math
import numpy as np
import torch
from .cuda.policy import QualificationError
from .cuda.ancestry import refit_identity


def validate_ancestry(proposal, reference, selected):
    """Validate continuous independently refitted partition ancestry, not raw KKT."""
    def require(test, message):
        if not test:
            raise QualificationError("Invalid partition ancestry: " + message)
    require(proposal.get('algorithm') == 'partition_ancestry_v2' and
            proposal.get('truth_used') is False and proposal.get('raw_certificate_inherited') is False,
            "algorithm or inherited claims")
    steps = proposal.get('ancestry')
    require(isinstance(steps, list) and bool(steps), "empty ancestry")
    expected = reference
    for step in steps:
        require(step.get('operation') in ('birth', 'reassignment', 'refit_relocation'), "unknown operation")
        require(step.get('parent') == expected, "immediate parent identity")
        child = step.get('child', {})
        require(child.get('n') == reference['n'] and isinstance(child.get('k'), int) and
                1 <= child['k'] <= child['n'] and child.get('refit_qualified') is True,
                "child dimensions/qualification")
        require(isinstance(child.get('membership_sha256'), str) and len(child['membership_sha256']) == 64,
                "membership identity")
        require(all(np.isfinite(child.get(key, np.nan)) for key in
                    ('score', 'score_lower', 'score_upper', 'refit_gap')) and child['refit_gap'] >= 0 and
                child['score'] == child['score_upper'] and
                child['score_lower'] == child['score'] - 2*child['refit_gap'], "score interval")
        margin = step.get('comparison_margin', np.nan)
        require(np.isfinite(margin) and margin > 0, "comparison margin")
        require(step.get('published_score_improved') == (child['score'] < expected['score']-margin) and
                step.get('refit_order_certified') == (child['score_upper'] < expected['score_lower']-margin),
                "score ordering claim")
        if step['operation'] == 'birth':
            details = step.get('details', {})
            require(child['k'] == expected['k']+1 and details.get('tumor_n') == reference['n'] and
                    details.get('occupied_k') == expected['k'] and
                    details.get('algorithm') in ('grid_k1_birth_v1', 'conditional_global_split_v1'),
                    "birth dimensions/global score")
        elif step['operation'] == 'refit_relocation':
            details = step.get('details', {})
            move = details.get('best_proposal', {})
            require(details.get('algorithm') == 'refit_aware_existing_group_relocation_v1'
                    and details.get('parent') == expected and details.get('child') == child
                    and details.get('accepted') is True and details.get('scan_complete') is True
                    and details.get('selected_endpoint_complete') is False,
                    "refit relocation parent/child coverage")
            require(details.get('coverage_scope') == 'all_mutations_to_existing_groups'
                    and details.get('planned_candidates') == reference['n'] * (expected['k'] - 1)
                    and type(details.get('evaluated_candidates')) is int
                    and type(details.get('infeasible_candidates')) is int
                    and details['evaluated_candidates'] >= 0 and details['infeasible_candidates'] >= 0
                    and details['evaluated_candidates'] + details['infeasible_candidates'] == details['planned_candidates']
                    and details.get('remaining_candidates') == 0,
                    "refit relocation complete parent accounting")
            require(all(type(move.get(key)) is int for key in ('node', 'source', 'destination'))
                    and 0 <= move['node'] < reference['n']
                    and 0 <= move['source'] < expected['k'] and 0 <= move['destination'] < expected['k']
                    and move['source'] != move['destination'] and type(move.get('source_deleted')) is bool
                    and child['k'] == expected['k'] - int(move['source_deleted'])
                    and step['refit_order_certified'] is True,
                    "refit relocation move and certified ordering")
        else:
            require(child['k'] <= expected['k'] and child['score'] < expected['score'], "refinement ordering")
        expected = child
    require(expected == selected, "selected child identity")


def validate_relocation_summary(summary, policy, selected, records):
    """Reconcile global work and final-endpoint coverage independently at export."""
    def require(value, message):
        if not value:
            raise QualificationError("Invalid refit relocation coverage: " + message)

    phase = [r for r in records if r.get('search_phase') == 'refit_relocation']
    if not policy.get('refit_relocations', False):
        require(summary is None and not phase, "disabled phase has claims")
        return
    require(isinstance(summary, dict) and summary.get('enabled') is True
            and summary.get('neighborhood') == 'all_mutations_to_existing_groups'
            and summary.get('fixed_point_criterion') == 'no_interval_certified_decrease_above_margin'
            and summary.get('global_partition_optimum_proven') is False, "scope")
    scans = summary.get('scans')
    require(isinstance(scans, list) and 1 <= len(scans) <= policy['refit_max_scans'], "scan budget")
    require(scans == [r['details'] for r in phase if 'details' in r], "unbound scan records")
    accepted = [scan for scan in scans if scan.get('accepted') is True]
    direct = [record for record in phase if 'details' not in record]
    require(len(direct) == len(accepted), "accepted scan/direct record count")
    preceding_ancestry = None
    for scan, record in zip(accepted, direct, strict=True):
        ancestry = record.get('proposal', {}).get('ancestry', [])
        require(record.get('status') == 'qualified' and record.get('candidate_family') == 'direct_partition'
                and isinstance(ancestry, list) and bool(ancestry), "accepted scan direct record")
        step = ancestry[-1]
        require(step.get('operation') == 'refit_relocation' and step.get('details') == scan
                and step.get('parent') == scan.get('parent') and step.get('child') == scan.get('child'),
                "accepted scan ancestry binding")
        require(preceding_ancestry is None or ancestry[:-1] == preceding_ancestry,
                "accepted scan ancestry continuity")
        preceding_ancestry = ancestry
    remaining, history, endpoint = policy['refit_max_candidates'], 0, None
    for index, scan in enumerate(scans):
        parent = scan['parent']
        require(type(parent.get('n')) is int and type(parent.get('k')) is int
                and 1 <= parent['k'] <= parent['n'] and parent['n'] == selected['n']
                and (endpoint is None or parent == endpoint), "parent continuity")
        planned = parent['n'] * (parent['k'] - 1)
        names = ('evaluated_candidates', 'infeasible_candidates', 'remaining_candidates',
                 'planned_candidates', 'scan_index', 'candidate_budget_before', 'candidate_budget_after')
        require(all(type(scan.get(name)) is int and scan[name] >= 0 for name in names), "candidate counts")
        charged = scan['evaluated_candidates'] + scan['infeasible_candidates']
        require(scan.get('scan_index') == index and scan.get('planned_candidates') == planned
                and scan.get('candidate_budget_before') == remaining and charged <= remaining
                and charged + scan['remaining_candidates'] == planned
                and scan.get('candidate_budget_after') == remaining-charged,
                "global candidate ledger")
        remaining -= charged
        unresolved = scan.get('unresolved_candidates')
        require(isinstance(unresolved, list) and len(unresolved) <= scan['evaluated_candidates'], "unresolved count")
        history += len(unresolved) + int(scan.get('independent_refit_error') is not None)
        require(type(scan.get('scan_complete')) is bool
                and scan['scan_complete'] == (charged == planned)
                and type(scan.get('accepted')) is bool
                and type(scan.get('selected_endpoint_complete')) is bool, "coverage types")
        if scan['accepted']:
            require(scan['status'] == 'accepted' and scan['scan_complete']
                    and scan['selected_endpoint_complete'] is False, "accepted child is unscanned")
            endpoint = scan['child']
        else:
            require(index == len(scans)-1, "scan continued after termination")
            endpoint = parent
            expected_complete = scan['scan_complete'] and not unresolved and scan['status'] == 'fixed_point'
            require(scan['selected_endpoint_complete'] == expected_complete, "false fixed point")
            require(scan['status'] in ('fixed_point', 'unresolved', 'candidate_budget_exhausted'), "terminal status")
            if scan['status'] == 'candidate_budget_exhausted':
                require(charged == 0 and planned > remaining, "false candidate budget stop")
            if scan['status'] == 'fixed_point':
                require(expected_complete and scan.get('independent_refit_error') is None, "unresolved fixed point")
    last = scans[-1]
    expected_status = 'scan_budget_exhausted' if last['accepted'] else last['status']
    if last['accepted']:
        require(len(scans) == policy['refit_max_scans'], "premature scan budget stop")
    if history:
        expected_status = 'unresolved'
    require(all(type(summary.get(name)) is int and summary[name] >= 0 for name in
                ('historical_unresolved', 'max_scans', 'max_candidates', 'charged_candidates',
                 'remaining_candidate_budget')), "final count types")
    require(endpoint == selected and summary.get('selected_endpoint') == selected,
            "selected partition identity")
    require(summary.get('status') == expected_status
            and summary.get('historical_unresolved') == history
            and summary.get('max_scans') == policy['refit_max_scans']
            and summary.get('max_candidates') == policy['refit_max_candidates']
            and summary.get('charged_candidates') == policy['refit_max_candidates']-remaining
            and summary.get('remaining_candidate_budget') == remaining
            and type(summary.get('selected_endpoint_complete')) is bool
            and summary['selected_endpoint_complete'] == last['selected_endpoint_complete'],
            "final phase ledger")


_ARRAYS = ("ccf", "labels", "centers", "memberships", "multiplicity", "reference_raw_phi",
           "reference_memberships", "reference_refitted_phi")


def partition_estimate_schema(policy):
    return "clipp1d.partition_estimate.v3" if policy.get('refit_relocations', False) else "clipp1d.partition_estimate.v2"


@dataclass(frozen=True)
class PartitionEstimate:
    ccf: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    memberships: np.ndarray
    multiplicity: np.ndarray
    reference_raw_phi: np.ndarray
    reference_memberships: np.ndarray
    reference_refitted_phi: np.ndarray
    score: float
    search_status: str
    provenance: dict
    records: list

    def __post_init__(self):
        for name in _ARRAYS:
            value = np.ascontiguousarray(getattr(self, name))
            object.__setattr__(self, name, np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape))
        object.__setattr__(self, "provenance", deepcopy(self.provenance))
        object.__setattr__(self, "records", deepcopy(self.records))

    def identity(self):
        return dict(provenance=self.provenance, records=self.records, score=self.score,
                    search_status=self.search_status,
                    arrays={name: hashlib.sha256(getattr(self, name).tobytes()).hexdigest() for name in _ARRAYS})

    def receipt(self):
        return dict(schema=partition_estimate_schema(self.provenance['policy']),
                    score=self.score, search_status=self.search_status,
                    provenance=self.provenance, search=self.records,
                    **{name: getattr(self, name).tolist() for name in _ARRAYS})


def validate_device_partition(d):
    from .cuda_api import _validate_device_fit_owned, _validate_refit
    from .cuda.partition import grouping
    result = d.partition_estimate
    candidate = result.candidate
    result.validate_identity()
    validate_relocation_summary(result.refit_relocation, asdict(result.policy),
                                refit_identity(candidate.refit), result.records)
    if candidate.family not in ("raw_fusion_path", "direct_partition"):
        raise QualificationError("Unknown partition candidate family")
    expected_rule = True if candidate.origin == "baseline" or candidate.proposal.get("root_origin") == "baseline" \
        else result.policy.separate_exact_one
    if candidate.separate_exact_one != expected_rule:
        raise QualificationError("Partition reference extraction policy does not match its origin")
    reference = replace(d, raw=candidate.raw_reference, refit=candidate.reference_refit,
                        lambda_value=candidate.reference_lambda, partition_estimate=None)
    # This is an independent audit of the actual parent raw vector/dual under
    # the original model/graph/penalty. The direct partition receives none of it.
    _validate_device_fit_owned(reference, separate_exact_one=candidate.separate_exact_one)
    _validate_refit(d.model, candidate.refit, d.policy)
    for ancestor in candidate.ancestors:
        _validate_refit(d.model, ancestor, d.policy)
    if candidate.family == "raw_fusion_path":
        expected = grouping(candidate.raw_reference.x, d.policy.fusion_tol,
                            separate_clonal=candidate.separate_exact_one)[2]
        if not torch.equal(expected, candidate.refit.labels) or candidate.proposal:
            raise QualificationError("Raw-family partition is unrelated to its raw state")
    else:
        validate_ancestry(candidate.proposal, refit_identity(candidate.reference_refit), refit_identity(candidate.refit))
        if [s['parent'] for s in candidate.proposal['ancestry']] != [refit_identity(r) for r in candidate.ancestors]:
            raise QualificationError("Partition ancestry is not bound to independently qualified parents")
        from .cuda.partition import canonical_labels
        children = (*candidate.ancestors[1:], candidate.refit)
        for step, parent, child in zip(candidate.proposal['ancestry'], candidate.ancestors, children, strict=True):
            if step['operation'] == 'refit_relocation':
                move = step['details']['best_proposal']
                if int(parent.labels[move['node']]) != move['source']:
                    raise QualificationError("Relocation source differs from its exact parent membership")
                proposed = parent.labels.clone()
                proposed[move['node']] = move['destination']
                if not torch.equal(canonical_labels(proposed)[0], child.labels):
                    raise QualificationError("Relocation child is not the declared single membership move")
    _validate_refit(d.model, result.preserved.refit, d.policy)
    if not bool(candidate.refit.score <= result.preserved.refit.score):
        raise QualificationError("Partition estimate is worse than the preserved current result")


def export_partition(d, model_sha, graph_sha):
    from .cuda_api import _host
    result = d.partition_estimate
    c, m = result.candidate, d.model
    r = c.refit
    order = torch.argsort(r.centers, descending=True, stable=True)
    order = torch.cat((order[order == r.clonal], order[order != r.clonal]))
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=m.device)
    values = dict(ccf=_host(r.phi), labels=_host(inverse[r.labels]), centers=_host(r.centers[order]),
                  memberships=_host(r.labels), multiplicity=_host(m.terms(r.phi)[3].argmax(-1) + 1),
                  reference_raw_phi=_host(c.raw_reference.x),
                  reference_memberships=_host(c.reference_refit.labels),
                  reference_refitted_phi=_host(c.reference_refit.phi))
    direct = c.family == "direct_partition"
    provenance = dict(candidate_family=c.family, origin=c.origin, model_sha256=model_sha,
                      graph_sha256=graph_sha, policy=asdict(result.policy),
                      estimator_role="separate development partition estimate; does not replace primary raw output",
                      selected_lambda=None if direct else float(c.reference_lambda),
                      raw_qualified=False if direct else True,
                      raw_certificate=None if direct else deepcopy(c.raw_reference.diagnostics),
                      refit_qualified=True, refit_gap=float(r.gap), loss=float(r.loss),
                      partition_identity=refit_identity(r),
                      preserved_partition=refit_identity(result.preserved.refit),
                      refit_relocation=deepcopy(result.refit_relocation),
                      global_optimality_proven=False, clonal_constraint=False,
                      clonal_label_rule="nearest_to_one_l2_v1", proposal=deepcopy(c.proposal),
                      raw_reference=dict(lambda_value=float(c.reference_lambda),
                                         objective=float(c.raw_reference.objective),
                                         certificate=deepcopy(c.raw_reference.diagnostics),
                                         score=float(c.reference_refit.score),
                                         partition_identity=refit_identity(c.reference_refit),
                                         separate_exact_one=c.separate_exact_one,
                                         role="independently qualified raw reference; not the direct partition's certificate"),
                      array_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in values.items()})
    status = "incomplete" if d.search_status == "incomplete" else result.status
    return PartitionEstimate(**values, score=float(r.score), search_status=status,
                             provenance=provenance, records=result.records)


def validate_partition_result(result):
    p = result.partition_estimate
    meta = p.provenance
    n = len(result.mutation_ids)

    def require(value, message):
        if not value:
            raise QualificationError("Refusing to publish partition estimate: " + message)

    require(isinstance(p, PartitionEstimate), "invalid partition result type")
    for name in _ARRAYS:
        x = getattr(p, name)
        require(x.ndim == 1 and len(x) == (len(p.centers) if name == "centers" else n) and np.isfinite(x).all(),
                "invalid " + name)
        require(meta['array_sha256'].get(name) == hashlib.sha256(x.tobytes()).hexdigest(), "changed " + name)
    require(meta.get("model_sha256") == result.provenance.get("model_sha256") and
            meta.get("graph_sha256") == result.graph_sha256, "unbound model or graph")
    require(meta.get("policy") == result.provenance.get("partition_search"), "unbound partition-search policy")
    require(meta.get("clonal_constraint") is False and meta.get("refit_qualified") is True and
            meta.get("global_optimality_proven") is False, "incorrect qualification semantics")
    require(np.isfinite(meta.get("refit_gap", np.nan)) and meta["refit_gap"] >= 0, "invalid refit gap")
    for name in ("ccf", "reference_raw_phi", "reference_refitted_phi"):
        x = getattr(p, name)
        require(np.all((x >= result.original_lower_bounds) & (x <= result.original_upper_bounds)),
                name + " outside original bounds")
    for name in ("labels", "memberships", "reference_memberships", "multiplicity"):
        require(getattr(p, name).dtype.kind in "iu", "noninteger " + name)
    k = len(p.centers)
    require(k > 0 and np.array_equal(np.unique(p.labels), np.arange(k)) and
            np.array_equal(np.unique(p.memberships), np.arange(k)) and
            np.array_equal(p.ccf, p.centers[p.labels]), "inconsistent occupied memberships/centers")
    require(np.all((p.multiplicity >= 1) & (p.multiplicity <= 4)), "invalid multiplicity")
    first = []
    for group in range(k):
        at = np.flatnonzero(p.memberships == group)
        first.append(int(at[0]))
        require(len(np.unique(p.labels[at])) == 1, "public labels change explicit memberships")
    require(first == sorted(first), "noncanonical explicit membership IDs")
    distance = np.abs(p.centers - 1)
    require(distance[0] == distance.min() and
            np.flatnonzero(p.labels == 0)[0] == min(np.flatnonzero(p.labels == j)[0]
                                                  for j in np.flatnonzero(distance == distance[0])),
            "incorrect closest-to-one designation")
    sizes = np.bincount(p.memberships)
    mass = math.lgamma(k) - math.lgamma(n + k) + sum(math.lgamma(int(s) + 1) for s in sizes) + math.lgamma(k + 1)
    expected = 2 * meta['loss'] + k * math.log(n) - 1.4 * mass
    require(np.isfinite(p.score) and abs(expected - p.score) <= 1e-9 * (1 + abs(p.score)), "score arithmetic mismatch")
    require(p.score <= result.selection_score, "discarded the better baseline")
    identity = meta.get('partition_identity', {})
    require(identity.get('membership_sha256') == hashlib.sha256(p.memberships.astype('<i8').tobytes()).hexdigest()
            and identity.get('score') == p.score and identity.get('n') == n and identity.get('k') == k and
            identity.get('refit_gap') == meta['refit_gap'], "unbound selected partition identity")
    validate_relocation_summary(meta.get('refit_relocation'), meta['policy'], identity, p.records)
    preserved = meta.get('preserved_partition', {})
    require(preserved.get('n') == n and np.isfinite(preserved.get('score', np.nan)) and
            p.score <= preserved['score'] <= result.selection_score,
            "discarded the preserved current partition")
    reference = meta['raw_reference']
    certificate = reference['certificate']
    require(certificate.get('box_feasible') is True and certificate.get('clonal_constraint') is False and
            (certificate.get('raw_branch_stationarity_qualified') is True or
             certificate.get('separable_scalar_gap_qualified') is True) and
            np.isfinite(reference['lambda_value']) and reference['lambda_value'] >= 0 and
            np.isfinite(reference['objective']), "unqualified raw reference")
    family = meta.get("candidate_family")
    if family == "direct_partition":
        require(meta.get("raw_qualified") is False and meta.get("raw_certificate") is None and
                meta.get("selected_lambda") is None and
                meta['proposal'].get("raw_certificate_inherited") is False and
                meta['proposal'].get("truth_used") is False,
                "direct partition inherited a raw certificate or penalty")
        reference_identity = reference.get('partition_identity', {})
        require(reference_identity.get('membership_sha256') ==
                hashlib.sha256(p.reference_memberships.astype('<i8').tobytes()).hexdigest() and
                reference_identity.get('score') == reference['score'], "unbound raw reference partition")
        validate_ancestry(meta['proposal'], reference_identity, identity)
    else:
        require(family == "raw_fusion_path" and meta.get("raw_qualified") is True and
                meta.get("raw_certificate") == certificate and meta.get("selected_lambda") == reference['lambda_value'] and
                np.array_equal(p.memberships, p.reference_memberships), "invalid raw-derived partition")
    qualified = [r for r in p.records if r.get('status') == 'qualified']
    require(bool(qualified) and all(np.isfinite(r.get('score', np.nan)) and
            np.isfinite(r.get('reference_lambda', np.nan)) and r['reference_lambda'] >= 0 and
            isinstance(r.get('clusters'), int) and r['clusters'] > 0 and
            r.get('candidate_family') in ('raw_fusion_path', 'direct_partition') for r in qualified),
            "invalid qualified candidate records")
    selected = [r for r in qualified if r['origin'] == meta['origin']]
    require(len(selected) == 1 and selected[0]['score'] == p.score and selected[0]['clusters'] == k and
            selected[0]['candidate_family'] == family, "winner missing from candidate records")
    if family == "direct_partition":
        require(selected[0].get('proposal') == meta['proposal'], "winner ancestry differs from selected record")
    best = min(qualified, key=lambda r: (r['score'], r['clusters'], r['reference_lambda'], r['origin'] != 'baseline'))
    require(best['origin'] == meta['origin'], "winner violates score/tie ordering")
    require(p.search_status in ('complete', 'incomplete') and
            not (result.search_status == 'incomplete' and p.search_status == 'complete'), "false search coverage")
    ordinary = [r for r in p.records if r.get('search_phase') != 'refit_relocation']
    proposal_statuses = [r.get('coverage_status', 'fixed_point') for r in ordinary]
    relocation = meta.get('refit_relocation')
    complete = (result.search_status == 'complete' and
                all(r.get('status') != 'unresolved' for r in ordinary) and
                all(status == 'fixed_point' for status in proposal_statuses) and
                (relocation is None or relocation['status'] == 'fixed_point'))
    require(p.search_status == ('complete' if complete else 'incomplete'), "false partition search completeness")
    require(result.provenance.get('primary_estimator') == 'selected_raw_complete_graph_penalized_candidate',
            "partition estimate silently replaced the primary raw estimator")


def write_partition_tables(result, data, destination):
    p = result.partition_estimate
    lookup = {mid: i for i, mid in enumerate(result.mutation_ids)}
    prefix = [data.tumor_id, data.sample_id]
    with (destination / "partition_mutation_clusters.tsv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["tumor_id", "sample_id", "mutation_id", "status", "partition_ccf",
                         "cluster_label", "designated_clonal", "multiplicity_call"])
        for mutation in data.mutations:
            if mutation.exclusion is not None:
                writer.writerow(prefix + [mutation.mutation_id, mutation.exclusion] + ["."] * 4)
            else:
                i = lookup[mutation.mutation_id]
                writer.writerow(prefix + [mutation.mutation_id, "retained", p.ccf[i], p.labels[i],
                                          int(p.labels[i] == 0), p.multiplicity[i]])
    with (destination / "partition_cluster_centers.tsv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["tumor_id", "sample_id", "cluster_label", "cluster_size", "partition_ccf", "designated_clonal"])
        sizes = np.bincount(p.labels)
        for label, center in enumerate(p.centers):
            writer.writerow(prefix + [label, sizes[label], center, int(label == 0)])
