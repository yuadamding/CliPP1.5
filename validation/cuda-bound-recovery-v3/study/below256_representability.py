"""A hash-bound fixed-order singleton representability diagnosis, not a fit."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks import capture_failed_qp as capture
from clipp1d.cuda.kernels import adjoint, differences, gap_kkt


ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT/'capture-first-below256-u/imported/results/capture-first-below256.json'
CAPTURE_SHA = '910a09a7143e27cb4f27e693dec62a4e38db07adb078dc7015b16fa10df2bfc3'


def main():
    receipt = capture.load_capture(CAPTURE, CAPTURE_SHA)
    assert len(receipt['captures']) == 1
    entry = receipt['captures'][0]
    record = capture.load_record(receipt, entry)
    p = {key: capture.load_tensor(receipt, value, 'cpu') for key, value in record['problem'].items()}
    h, target, lower, upper, caps = [p[key] for key in ('h', 'target', 'lower', 'upper', 'caps')]
    saved = record['last_refined_state']
    x, q = [capture.load_tensor(receipt, saved[key], 'cpu') for key in ('x', 'q')]
    g, a = h*(x-target), adjoint(q)
    residual = g+a
    node = residual.abs()/(1+g.abs()+a.abs())
    node = torch.where((x == lower) | (x == upper), 0., node)
    index = int(node.argmax())
    assert index == 139 and int((x == x[index]).sum()) == 1
    assert lower[index] < x[index] < upper[index]
    saturated = caps*differences(x).sign()
    assert torch.equal(q[index], saturated[index])
    adj = adjoint(saturated)[index]
    ld = np.longdouble
    exact_adjoint = -saturated[index].numpy().astype(ld).sum()
    root = ld(float(target[index]))-exact_adjoint/ld(float(h[index]))
    tau = record['context']['policy']['inner_kkt_tol']
    stats = gap_kkt(x, q, h, target, lower, upper, caps)
    allowed = record['context']['policy']['inner_atol']+record['context']['policy']['inner_rtol']*float(stats[1])
    rows = []
    for neighbor in (-1, 0, 1):
        trial = x.clone()
        if neighbor:
            trial[index] = torch.nextafter(x[index], x.new_tensor(float('inf') if neighbor > 0 else -float('inf')))
        gradient = h[index]*(trial[index]-target[index])
        r = gradient+adj
        denominator = 1+gradient.abs()+adj.abs()
        # For a'=a-sign(r)*delta, solve |r|-delta =
        # tau*(1+|g|+|a'|) before any sign change in a.
        required = ((r.abs()-tau*denominator)/(1-tau*r.sign()*adj.sign())).clamp_min(0.)
        appropriate_edges = (saturated[index].sign() == -r.sign()) & (torch.arange(len(x)) != index)
        distance = (trial[index]-trial[appropriate_edges]).abs().min()
        edge_margin = (128*torch.finfo(x.dtype).eps*(1+caps[index])*(trial[index]-trial).abs()).sum()
        cost = required*distance
        rows.append(dict(neighbor=neighbor, ccf=float(trial[index]), gradient=float(gradient),
                         saturated_adjoint=float(adj), stationarity_residual=float(r),
                         normalized_kkt=float(r.abs()/denominator),
                         minimum_required_dual_change=float(required),
                         minimum_edge_distance_for_required_direction=float(distance),
                         lower_bound_incident_edge_gap=float(cost),
                         incident_edge_roundoff_margin=float(edge_margin),
                         gap_cost_to_allowed_ratio=float(cost)/allowed,
                         cost_exceeds_gap_even_after_row_roundoff=bool(cost-edge_margin > allowed)))
    assert all(row['normalized_kkt'] > tau and row['cost_exceeds_gap_even_after_row_roundoff'] for row in rows)
    result = dict(schema='clipp1d.fixed_order_singleton_representability.v1',
                  scope='CPU literal-array proof for the captured singleton and its current neighbor coordinates/edge ordering. This is not a universal infeasibility proof over every float64 primal/dual state, a CPU QP fit, or CUDA qualification.',
                  capture_sha256=CAPTURE_SHA, record_sha256=entry['sha256'],
                  source_sha256=record['context']['source_sha256'], context=record['context'],
                  script_sha256=capture.common.sha(Path(__file__)),
                  problem_tensor_sha256={key: None if descriptor is None else descriptor['tensor_sha256'] for key, descriptor in record['problem'].items()},
                  captured_state_tensor_sha256={key: saved[key]['tensor_sha256'] for key in ('x', 'q')},
                  node_index=index, mutation_id=record['graph']['mutation_ids'][index],
                  curvature=float(h[index]), target=float(target[index]),
                  singleton_is_interior=True, all_incident_edges_exactly_saturated=True,
                  longdouble_saturated_adjoint=str(exact_adjoint), longdouble_stationary_root=str(root),
                  root_displacement_ulps=float((root-ld(float(x[index])))/np.spacing(float(x[index]))),
                  allowed_gap=allowed, kkt_tolerance=tau, neighboring_floats=rows,
                  interpretation='The nearest represented singleton already minimizes the saturated-flow stationarity residual among these neighboring floats. Any incident flow change large enough to pass node KKT, with these neighbor coordinates and edge ordering, costs more than the entire allowed primal-dual gap. Improving the other fused center or continuing fixed-candidate flow cannot resolve this local floor.')
    path = ROOT/'BELOW256_REPRESENTABILITY.json'
    capture.common.write_json(path, result)
    print(json.dumps(dict(output=str(path), sha256=capture.common.sha(path), node=index,
                          nearest_kkt=rows[1]['normalized_kkt'], minimum_gap_cost=rows[1]['lower_bound_incident_edge_gap'],
                          allowed_gap=allowed, cost_to_allowance=rows[1]['gap_cost_to_allowed_ratio'])))


if __name__ == '__main__':
    main()
