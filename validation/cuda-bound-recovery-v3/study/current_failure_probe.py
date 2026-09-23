"""Literal fixed-candidate CPU references, never a full QP or fit acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks import capture_failed_qp as cap
from clipp1d.api import source_provenance
from clipp1d.cuda import qp
from clipp1d.cuda.kernels import adjoint, differences, gap_kkt
from clipp1d.cuda.partition import polish_quadratic


def certificate(x, q, problem, policy):
    values = gap_kkt(x, q, *(problem[k] for k in ('h', 'target', 'lower', 'upper', 'caps')))
    allowed = policy['inner_atol'] + policy['inner_rtol'] * float(values[1])
    return dict(gap=float(values[0]), allowed_gap=allowed, scale=float(values[1]),
                kkt=float(values[2]), qualified=bool(torch.isfinite(values).all())
                and float(values[0]) <= allowed and float(values[2]) <= policy['inner_kkt_tol'])


def geometry_report(x, q, p):
    h, target, lower, upper, caps = (p[k] for k in ('h', 'target', 'lower', 'upper', 'caps'))
    same = x[:, None] == x[None, :]
    external = adjoint(torch.where(same, 0., caps * differences(x).sign()))
    free = lower < upper
    gradient = torch.where(free, h * (x - target), 0.)
    residual = gradient + external
    capacity = torch.where(same, caps, 0.).sum(1)
    excess = torch.where(x == lower, (-residual-capacity).clamp_min(0),
                        torch.where(x == upper, (residual-capacity).clamp_min(0),
                                    (residual.abs()-capacity).clamp_min(0)))
    excess = torch.where(free, excess, 0.)
    groups = []
    stable = x.clone()
    nearest_longdouble = x.clone()
    for value in torch.unique(x):
        ids = torch.where(x == value)[0]
        H, T, E = (v[ids].numpy().astype(np.longdouble) for v in (h, target, external))
        fixed = (lower[ids] == upper[ids])
        T = np.where(fixed.numpy(), lower[ids].numpy().astype(np.longdouble), T)
        anchor = np.longdouble(float(value))
        root = anchor + (np.sum(H*(T-anchor))-np.sum(E))/np.sum(H)
        root = np.clip(root, float(lower[ids].max()), float(upper[ids].min()))
        stable_center = value + ((h[ids] * (torch.where(fixed, lower[ids], target[ids])-value)).sum()
                                 - external[ids].sum()) / h[ids].sum()
        stable_center = stable_center.clamp(lower[ids].max(), upper[ids].min())
        stable[ids] = stable_center
        nearest_longdouble[ids] = float(root)
        G = (H * (anchor-T) + E)
        groups.append(dict(size=len(ids), ids=ids.tolist(), ccf=float(value),
                           longdouble_root=str(root), nearest_longdouble=float(root),
                           stable_float64_center=float(stable_center),
                           root_displacement_ulps=float((root-anchor)/np.spacing(float(value))),
                           aggregate_gradient_float64=float(residual[ids].sum()),
                           aggregate_gradient_longdouble=str(G.sum()),
                           interior=int(((x[ids] != lower[ids]) & (x[ids] != upper[ids])).sum()),
                           at_lower=int(((x[ids] == lower[ids]) & ~fixed).sum()),
                           at_upper=int(((x[ids] == upper[ids]) & ~fixed).sum()),
                           fixed=int(fixed.sum()),
                           maximum_singleton_cone_excess=float(excess[ids].max())))
    return dict(groups=groups, singleton_cone_obstructions=torch.where(excess > 0)[0].tolist()), stable, nearest_longdouble


def repair(x, q, problem, policy, method):
    h, target, lower, upper, caps = (problem[k] for k in ('h', 'target', 'lower', 'upper', 'caps'))
    context = qp.prepare_flow_polish(x, h, target, lower, upper, caps)
    same = x[:, None] == x[None, :]
    weights = same.sum(1)[:, None]
    external = caps * differences(x).sign()
    q = torch.where(same, q, external).clone()
    y, theta, restarts = q.clone(), 1., 0
    checkpoints = []
    for iteration in range(1, 1025):
        old_q = q
        q = qp.flow_polish_step(y, context)
        next_theta = (1 + math.sqrt(1+4*theta*theta))/2
        if method == 'plain':
            y = q
        elif method == 'restart' and float((weights * (y-q) * (q-old_q)).sum()) > 0:
            y, theta = q, 1.
            restarts += 1
        else:
            y = torch.where(same, q + ((theta-1)/next_theta)*(q-old_q), external)
            theta = next_theta
        if iteration % policy['check_every'] == 0 or iteration == 1024:
            stats = certificate(x, q, problem, policy)
            checkpoints.append(dict(step=iteration, **stats))
            if stats['qualified']:
                break
    return dict(method=method, steps=iteration, restarts=restarts,
                final=checkpoints[-1], checkpoints=checkpoints,
                dual_feasible=bool((q.abs() <= caps).all() & (q == -q.T).all()),
                decomposition=cap.decompose_gap(x, q, h, target, lower, upper, caps))


def split_singleton_obstructions(candidate, raw_x, p):
    """Diagnostic-only conservative split of proven necessary cone-cut failures."""
    h, target, lower, upper, caps = (p[k] for k in ('h', 'target', 'lower', 'upper', 'caps'))
    same = candidate[:, None] == candidate[None, :]
    external_edges = torch.where(same, 0., caps * differences(candidate).sign())
    gradient = torch.where(lower < upper, h * (candidate-target), 0.)
    residual = gradient + adjoint(external_edges)
    capacity = torch.where(same, caps, 0.).sum(1)
    excess = torch.where(candidate == lower, -residual-capacity,
                        torch.where(candidate == upper, residual-capacity,
                                    residual.abs()-capacity))
    margin = 128*torch.finfo(h.dtype).eps*(1+gradient.abs()+external_edges.abs().sum(1)+capacity)
    bad = (lower < upper) & (excess > margin)
    if not bool(bad.any()):
        return None, []
    split = same & ~bad[:, None] & ~bad[None, :]
    split |= torch.eye(len(candidate), dtype=torch.bool)
    external = adjoint(torch.where(split, 0., caps*differences(raw_x).sign()))
    safe_target = torch.where(lower < upper, target, lower)
    numerator = torch.where(split, h[None, :]*(safe_target[None, :]-candidate[:, None])-external[None, :], 0.).sum(1)
    center = candidate + numerator / torch.where(split, h[None, :], 0.).sum(1)
    lo = torch.where(split, lower[None, :], -torch.inf).max(1).values
    hi = torch.where(split, upper[None, :], torch.inf).min(1).values
    return torch.maximum(lo, torch.minimum(hi, center)), torch.where(bad)[0].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    receipt = cap.load_capture(args.capture, args.sha256)
    records = []
    for entry in receipt['captures']:
        saved = cap.load_record(receipt, entry)
        p = {key: cap.load_tensor(receipt, value, 'cpu') for key, value in saved['problem'].items()}
        policy = saved['context']['policy']
        x, q = (cap.load_tensor(receipt, saved['returned'][key], 'cpu') for key in ('x', 'q'))
        candidate_inputs = [('returned', x, q)]
        if saved.get('last_refined_state'):
            s = saved['last_refined_state']
            candidate_inputs.append(('last_refined', cap.load_tensor(receipt, s['x'], 'cpu'),
                                     cap.load_tensor(receipt, s['q'], 'cpu')))
        problem_tuple = tuple(p[k] for k in ('h', 'target', 'lower', 'upper', 'caps'))
        for divisor in (1., 16., 256.):
            candidate_inputs.append((f'polish_{divisor:g}', polish_quadratic(x, q, *problem_tuple,
                                                                          policy['fusion_tol']/divisor), q))
        rows, seen = [], set()
        objective = qp._objective_context(x, *problem_tuple)
        for name, candidate, initial in candidate_inputs:
            identity = hashlib.sha256(candidate.numpy().tobytes() + initial.numpy().tobytes()).hexdigest()
            if identity in seen:
                continue
            seen.add(identity)
            geometry, stable, precise = geometry_report(candidate, initial, p)
            variants = [('literal', candidate)]
            if not torch.equal(stable, candidate):
                variants.append(('stable_float64_center', stable))
            if not any(torch.equal(precise, v) for _, v in variants):
                variants.append(('nearest_longdouble_center', precise))
            split, split_nodes = split_singleton_obstructions(candidate, x, p)
            if split is not None and not any(torch.equal(split, v) for _, v in variants):
                variants.append(('singleton_cone_split', split))
            for variant, value in variants:
                new_value = qp.quadratic_value(value, p['h'], objective[0], p['caps'], objective[1])
                roundoff = 32*torch.finfo(torch.float64).eps*(1+objective[2].abs()+new_value.abs())
                row = dict(name=name, variant=variant, initial=certificate(value, initial, p, policy),
                           objective_admitted=bool(torch.isfinite(new_value) & (new_value <= objective[2]+roundoff)),
                           objective_change=float(new_value-objective[2]),
                           split_nodes=split_nodes if variant == 'singleton_cone_split' else [],
                           geometry=geometry if variant == 'literal' else geometry_report(value, initial, p)[0],
                           repairs=[repair(value, initial, p, policy, method) for method in ('plain', 'fista', 'restart')])
                rows.append(row)
        records.append(dict(context=saved['context'], record_sha256=entry['sha256'], candidates=rows))
    result = dict(scope='Local CPU fixed-candidate diagnostics only, no full QP initialization replay, no full fit, no CUDA acceptance.',
                  capture_sha256=args.sha256, source=source_provenance(),
                  driver_sha256=cap.common.sha(Path(__file__)), records=records)
    cap.common.write_json(args.out, result)
    print(json.dumps(dict(output=str(args.out), sha256=cap.common.sha(args.out),
                          candidates=[[dict(name=v['name'], variant=v['variant'], objective_admitted=v['objective_admitted'],
                                            repairs=[dict(method=r['method'], steps=r['steps'], **r['final']) for r in v['repairs']])
                                       for v in row['candidates']] for row in records])))


if __name__ == '__main__':
    main()
