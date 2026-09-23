"""Read-only literal-array and single-node-update diagnosis; never a full QP fit."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks import capture_failed_qp as capture
from clipp1d.cuda.kernels import adjoint, boxed_rank_one, differences, edge_update
from clipp1d.cuda.partition import polish_quadratic
from clipp1d.cuda.qp import quadratic_value


def numbers(values):
    return dict(minimum=float(values.min()), median=float(values.median()), maximum=float(values.max()))


def scalar_box_reference(h, b, lower, upper, rho):
    H, B, L, U = [v.numpy().astype(np.longdouble) for v in (h, b, lower, upper)]
    R = np.longdouble(float(rho))
    A = H + R*len(H)
    B = np.where(L < U, B, A*L)
    left, right = L.sum(), U.sum()
    for iteration in range(256):
        middle = (left+right)/2
        point = np.clip((B+R*middle)/A, L, U)
        residual = middle-point.sum()
        if residual < 0:
            left = middle
        elif residual > 0:
            right = middle
        else:
            left = right = middle
        if left == right or (left+right)/2 in (left, right):
            break
    root = (left+right)/2
    unboxed = (B+R*root)/A
    point = np.clip(unboxed, L, U)
    active = (unboxed > L) & (unboxed < U)
    derivative = ((~active).sum()+np.where(active, H/A, 0.).sum())/len(H)
    residual_roundoff = (32*(len(H)+2)*np.finfo(np.longdouble).eps
                         * (1+abs(root)+np.abs(point).sum()))
    root_roundoff = residual_roundoff/derivative
    node_roundoff = root_roundoff*np.where(active, R/A, 0.).max()
    return point, dict(iterations=iteration+1, scalar_root=str(root),
                       bracket_width=str(right-left), stable_scalar_derivative=str(derivative),
                       conservative_scalar_root_roundoff=str(root_roundoff),
                       conservative_node_roundoff=str(node_roundoff))


def boxed_residual(x, h, b, lower, upper, rho):
    a = h+rho*len(h)
    raw = a*x-rho*x.sum()-b
    projected = torch.where(x == lower, raw.clamp_max(0.), raw)
    projected = torch.where(x == upper, projected.clamp_min(0.), projected)
    projected = torch.where(lower < upper, projected, 0.)
    scale = 1+(a*x).abs()+(rho*x.sum()).abs()+b.abs()
    return dict(maximum_absolute=float(projected.abs().max()),
                maximum_componentwise_backward=float((projected.abs()/scale).max()),
                worst_node=int((projected.abs()/scale).argmax()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    receipt = capture.load_capture(args.capture, args.sha256)
    rows = []
    for entry in receipt['captures']:
        record = capture.load_record(receipt, entry)
        p = {key: capture.load_tensor(receipt, value, 'cpu') for key, value in record['problem'].items()}
        h, target, lower, upper, caps = [p[key] for key in ('h', 'target', 'lower', 'upper', 'caps')]
        state = {key: capture.load_tensor(receipt, value, 'cpu') for key, value in record['terminal_admm'].items()}
        x, q, z, v, rho = [state[key] for key in ('x', 'q', 'z', 'v', 'rho')]
        original_args = (h, target, lower, upper, caps)
        reference = torch.clamp(torch.where(lower < upper, target, lower), min=lower, max=upper)
        old_objective = quadratic_value(x, h, target, caps, reference)
        proposals = []
        for divisor in (1., 16., 256.):
            candidate = polish_quadratic(x, q, *original_args, record['context']['policy']['fusion_tol']/divisor)
            value = quadratic_value(candidate, h, target, caps, reference)
            margin = 32*torch.finfo(h.dtype).eps*(1+old_objective.abs()+value.abs())
            proposals.append(dict(divisor=divisor, exact_group_sizes=torch.unique(candidate, return_counts=True)[1].tolist(),
                                  objective=float(value), objective_change=float(value-old_objective),
                                  objective_roundoff=float(margin),
                                  objective_admitted=bool(torch.isfinite(value) & (value <= old_objective+margin))))
        b = h*torch.where(lower < upper, target, lower)+rho*adjoint(z-v)
        next_x = boxed_rank_one(h, b, lower, upper, rho)
        reference_x, scalar = scalar_box_reference(h, b, lower, upper, rho)
        reference_tensor = torch.tensor(reference_x.astype(np.float64), dtype=torch.float64)
        next_z, _ = edge_update(next_x, v, caps, rho)
        dual_residual = rho*adjoint(next_z-z)
        primal_norm = (.5*(differences(next_x)-next_z).square().sum()).sqrt()
        global_dual_norm = dual_residual.square().sum().sqrt()/h.median()
        component_dual_norm = (dual_residual/h).square().sum().sqrt()
        controller = dict(
            scope='One next CPU eager update from literal terminal state; not a captured past controller decision.',
            primal_edge_l2=float(primal_norm),
            current_global_curvature_dual_l2=float(global_dual_norm),
            componentwise_curvature_dual_l2=float(component_dual_norm),
            maximum_raw_dual_residual_node=int(dual_residual.abs().argmax()),
            maximum_scaled_dual_residual_node=int((dual_residual/h).abs().argmax()),
            current_next_rho_ratio=2. if primal_norm > 10*global_dual_norm else (
                .5 if global_dual_norm > 10*primal_norm else 1.),
            componentwise_next_rho_ratio=2. if primal_norm > 10*component_dual_norm else (
                .5 if component_dual_norm > 10*primal_norm else 1.),
        )
        next_update = dict(scope='One CPU eager boxed node subproblem from terminal tensors; not the historical CUDA update or a QP trajectory.',
                           scalar_reference=scalar,
                           maximum_ccf_difference=float(np.max(np.abs(next_x.numpy().astype(np.longdouble)-reference_x))),
                           eager=boxed_residual(next_x, h, b, lower, upper, rho),
                           rounded_longdouble=boxed_residual(reference_tensor, h, b, lower, upper, rho),
                           residual_balance_comparison=controller)
        common_lower, common_upper = lower.max(), upper.min()
        all_fused = dict(common_interval_nonempty=bool(common_lower <= common_upper),
                         lower=float(common_lower), upper=float(common_upper))
        if bool(common_lower <= common_upper):
            H, T = h.numpy().astype(np.longdouble), target.numpy().astype(np.longdouble)
            center = np.clip(np.sum(H*T)/H.sum(), float(common_lower), float(common_upper))
            fused = torch.full_like(x, float(center))
            g = torch.where(lower < upper, h*(fused-target), 0.)
            fixed = lower == upper
            at_lower, at_upper = (fused == lower)&~fixed, (fused == upper)&~fixed
            total = g.sum()
            normal_nodes = fixed if bool(fixed.any()) else (at_lower if bool(total >= 0) else at_upper)
            normal = torch.where(normal_nodes, total/normal_nodes.sum().clamp_min(1), 0.)
            residual = g-normal
            proposed_q = -differences(residual)/len(h)
            capacity_excess = (proposed_q.abs()-caps).clamp_min(0.)
            value = quadratic_value(fused, h, target, caps, reference)
            all_fused.update(center=float(center), center_longdouble=str(center),
                             normal_node_indices=torch.where(normal_nodes)[0].tolist(),
                             aggregate_gradient=float(total),
                             aggregate_flow_demand=float(residual.sum()),
                             direct_dual_capacity_feasible=bool((capacity_excess == 0).all()),
                             direct_dual_maximum_capacity_excess=float(capacity_excess.max()),
                             objective=float(value), objective_change=float(value-old_objective),
                             decomposition=capture.decompose_gap(fused, proposed_q, *original_args))
        rows.append(dict(context=record['context'], record_sha256=entry['sha256'],
                         curvature=numbers(h), curvature_ratio=float(h.max()/h.min()),
                         terminal_rho=float(rho), terminal_rho_over_h_median=float(rho/h.median()),
                         n_rho_over_h_max=float(len(h)*rho/h.max()),
                         terminal_primal_l2=float((.5*(differences(x)-z).square().sum()).sqrt()),
                         terminal_x=numbers(x), terminal_z_absolute_max=float(z.abs().max()),
                         terminal_q_rho_v_max_difference=float((q-rho*v).abs().max()),
                         terminal_certificate=capture.decompose_gap(x, q, *original_args),
                         equality_proposals=proposals, single_boxed_update=next_update,
                         all_fused=all_fused))
    result = dict(scope='CPU read-only literal-array arithmetic and one boxed node-update reference; no full QP fit or CUDA qualification.',
                  capture_sha256=args.sha256, captured_source_sha256=receipt['source']['source_sha256'],
                  script_sha256=capture.common.sha(Path(__file__)), records=rows)
    capture.common.write_json(args.out, result)
    print(json.dumps(dict(output=str(args.out), sha256=capture.common.sha(args.out))))


if __name__ == '__main__':
    main()
