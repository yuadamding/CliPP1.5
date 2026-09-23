"""Read-only CPU array diagnosis of hash-bound CUDA captures; no fits/recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import platform

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks import capture_failed_qp as cap


def scalar(v):
    return float(v)


def decode(receipt, descriptor):
    value = cap.load_tensor(receipt, descriptor, 'cpu')
    return None if value is None else value.numpy()


def state_details(x, q, problem, ids):
    h, target, lower, upper, caps = [problem[k] for k in ('h', 'target', 'lower', 'upper', 'caps')]
    same = x[:, None] == x[None, :]
    free = lower < upper
    gradient = np.where(free, h * (x - np.where(free, target, x)), 0.)
    a = -q.sum(1)
    residual = gradient + a
    projected = np.where(x == lower, np.minimum(residual, 0.), residual)
    projected = np.where(x == upper, np.maximum(projected, 0.), projected)
    projected = np.where(free, projected, 0.)
    node_kkt = np.abs(projected) / (1 + np.abs(gradient) + np.abs(a))
    d = x[None, :] - x[:, None]
    edges = caps * np.abs(d) - q * d
    left, right = np.triu_indices(len(x), 1)
    ev = np.maximum(edges[left, right], 0.)
    order = np.argsort(-ev, kind='stable')
    top_edges = [dict(left=ids[left[k]], right=ids[right[k]], gap=scalar(ev[k]),
                      absolute_ccf_difference=scalar(abs(d[left[k], right[k]])),
                      cap=scalar(caps[left[k], right[k]]), physical_dual=scalar(q[left[k], right[k]]),
                      absolute_saturation_deficit=scalar(caps[left[k], right[k]] -
                                                       q[left[k], right[k]] * np.sign(d[left[k], right[k]])))
                 for k in order[:8]]
    external = np.where(same, 0., caps * np.sign(d))
    # Work from the exact float64 input values in extended-precision sums.
    # This is a necessary-cut diagnosis, never a substitute acceptance tolerance.
    ld = np.longdouble
    adjusted = gradient.astype(ld) - external.astype(ld).sum(1)
    groups = []
    for value in np.unique(x):
        ix = np.flatnonzero(x == value)
        fixed = lower[ix] == upper[ix]
        atlo = (~fixed) & (x[ix] == lower[ix])
        atup = (~fixed) & (x[ix] == upper[ix])
        total = adjusted[ix].sum()
        normal = np.zeros(len(ix), dtype=ld)
        if fixed.any():
            normal[fixed] = total / fixed.sum()
        elif total >= 0 and atlo.any():
            normal[atlo] = total / atlo.sum()
        elif total < 0 and atup.any():
            normal[atup] = total / atup.sum()
        demand = adjusted[ix] - normal
        internal_caps = caps[np.ix_(ix, ix)].astype(ld)
        singleton_excess = np.abs(demand) - internal_caps.sum(1)
        candidate_sets = [np.array([k]) for k in range(len(ix))]
        for mask in (fixed, atlo, atup, (~fixed & ~atlo & ~atup), demand > 0, demand < 0):
            selected = np.flatnonzero(mask)
            if 0 < len(selected) < len(ix):
                candidate_sets.append(selected)
        cuts = []
        for subset in candidate_sets:
            complement = np.setdiff1d(np.arange(len(ix)), subset)
            required = np.abs(demand[subset].sum())
            available = internal_caps[np.ix_(subset, complement)].sum()
            bound = ld(512) * np.finfo(ld).eps * (1 + np.abs(demand[subset]).sum() + available)
            excess = required - available
            if excess > bound:
                cuts.append(dict(members=[ids[ix[k]] for k in subset], required=scalar(required),
                                 available=scalar(available), excess=scalar(excess),
                                 longdouble_roundoff_bound=scalar(bound)))
        cuts.sort(key=lambda v: -v['excess'])
        allocatable = int(fixed.sum() if fixed.any() else (atlo.sum() if total >= 0 else atup.sum()))
        groups.append(dict(
            ccf=scalar(value), members=[ids[k] for k in ix], size=len(ix),
            lower_active=int(atlo.sum()), upper_active=int(atup.sum()), fixed=int(fixed.sum()),
            interior=int((~fixed & ~atlo & ~atup).sum()),
            adjusted_gradient_total=scalar(total),
            uniform_normal=[scalar(v) for v in normal],
            uniform_internal_divergence_demand=[scalar(v) for v in demand],
            normal_allocation_degrees_of_freedom=max(0, allocatable - 1),
            max_singleton_capacity_excess=scalar(singleton_excess.max()),
            strongest_tested_cut_obstructions=cuts[:4],
            conservation_residual=scalar(demand.sum()),
            node_kkt_max=scalar(node_kkt[ix].max()),
            node_projected_residual_max=scalar(np.abs(projected[ix]).max()),
        ))
    nodes = np.argsort(-node_kkt, kind='stable')[:8]
    return dict(
        scope='CPU arithmetic over captured exact arrays; original GPU certificates remain authoritative.',
        exact_group_sizes=sorted([g['size'] for g in groups], reverse=True), exact_groups=groups,
        top_node_kkt=[dict(mutation_id=ids[k], kkt=scalar(node_kkt[k]),
                           gradient=scalar(gradient[k]), adjoint=scalar(a[k]),
                           projected_residual=scalar(projected[k]), curvature=scalar(h[k])) for k in nodes],
        edge_gap=scalar(ev.sum()), positive_edge_gap_count=int((ev > 0).sum()),
        top_two_edge_fraction=scalar(ev[order[:2]].sum() / ev.sum()) if ev.sum() else 0.,
        top_eight_edge_fraction=scalar(ev[order[:8]].sum() / ev.sum()) if ev.sum() else 0.,
        top_edges=top_edges,
        groups_with_multiple_normal_allocations=sum(g['normal_allocation_degrees_of_freedom'] > 0 for g in groups),
        cut_scope='Necessary singleton, bound-class, interior-class and demand-sign subset tests only. A positive cut excess proves this uniform divergence target infeasible; no excess does not prove feasibility. Conservation residual and representational error are separate.',
    )


def diagnose(receipt, entry):
    record = cap.load_record(receipt, entry)
    problem = {key: decode(receipt, record['problem'][key]) for key in ('h', 'target', 'lower', 'upper', 'caps')}
    ids = record['graph']['mutation_ids']
    policy = record['context']['policy']
    result = dict(record=entry['record'], record_sha256=entry['sha256'],
                  context={key: val for key, val in record['context'].items() if key not in ('graph_mutation_ids', 'policy')},
                  curvature=dict(minimum=scalar(problem['h'].min()), maximum=scalar(problem['h'].max()),
                                 ratio=scalar(problem['h'].max() / problem['h'].min())),
                  iterations=record['returned']['iterations'], polish_iterations=record['returned']['polish_iterations'],
                  states={})
    weights = decode(receipt, record['graph']['weights'])
    result['literal_graph_caps_binding'] = dict(
        exactly_lambda_times_weights=bool(np.array_equal(problem['caps'], weights * record['context']['lambda_value'])),
        maximum_absolute_difference=scalar(np.abs(problem['caps'] - weights * record['context']['lambda_value']).max()),
        weights_tensor_sha256=record['graph']['weights']['tensor_sha256'],
        pilot_tensor_sha256=record['graph']['pilot']['tensor_sha256'])
    for name in ('returned', 'terminal_admm', 'last_equality_attempt', 'last_refined_state'):
        saved = record.get(name)
        if saved is None or saved.get('x') is None:
            continue
        x = decode(receipt, saved['x'])
        q_descriptor = saved.get('q')
        role = 'saved_output_dual'
        if q_descriptor is None:
            q_descriptor = saved.get('input_q')
            role = 'incoming_dual_only_no_proposal_dual_was_generated'
        if q_descriptor is None:
            continue
        q = decode(receipt, q_descriptor)
        analyzed = state_details(x, q, problem, ids)
        analyzed['dual_role'] = role
        analyzed['x_tensor_sha256'] = saved['x']['tensor_sha256']
        analyzed['q_tensor_sha256'] = q_descriptor['tensor_sha256']
        original = record['decomposition'].get(name)
        if original is not None:
            analyzed['captured_gpu_decomposition'] = original
            certificate = original['certificate']
            allowed = policy['inner_atol'] + policy['inner_rtol'] * certificate['scale']
            analyzed['original_gates'] = dict(allowed_gap=allowed,
                gap_to_allowed_ratio=certificate['gap'] / allowed,
                kkt_to_allowed_ratio=certificate['kkt'] / policy['inner_kkt_tol'])
        if name == 'last_equality_attempt':
            analyzed['objective_admitted'] = saved['objective_admitted']
            for key in ('old_value', 'new_value', 'roundoff'):
                analyzed[key] = scalar(decode(receipt, saved[key])) if saved.get(key) else None
            if analyzed['old_value'] is not None:
                analyzed['objective_increase'] = analyzed['new_value'] - analyzed['old_value']
                analyzed['objective_increase_to_roundoff_ratio'] = analyzed['objective_increase'] / analyzed['roundoff']
            analyzed['tolerance'] = saved['tolerance']
        if name == 'last_refined_state':
            analyzed['objective_admitted'] = True
            analyzed['returned_qualified_candidate'] = saved['returned_qualified_candidate']
            analyzed['tolerance'] = saved['tolerance']
            analyzed['flow_steps'] = saved['flow_steps']
        result['states'][name] = analyzed
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    script_hash = cap.common.sha(Path(__file__))
    receipt = cap.load_capture(args.capture, args.sha256)
    result = dict(schema='clipp1d.cuda.capture_array_diagnosis.v1',
                  capture_path=str(args.capture), capture_sha256=args.sha256,
                  captured_source_sha256=receipt['source']['source_sha256'],
                  input_sha256=receipt['input_sha256'],
                  diagnostic_script_sha256=script_hash,
                  loader_script_sha256=cap.common.sha(Path(cap.__file__)),
                  scope='Read-only local CPU scalar/array diagnosis, no full fit, no recovery and no CUDA qualification.',
                  python=sys.version, platform=platform.platform(), numpy=np.__version__,
                  cases=[diagnose(receipt, entry) for entry in receipt['captures']])
    assert cap.common.sha(Path(__file__)) == script_hash
    cap.common.write_json(args.out, result)
    print(json.dumps(dict(output=str(args.out), sha256=cap.common.sha(args.out), cases=len(result['cases']))))


if __name__ == '__main__':
    main()
