"""Bounded CPU fixed-candidate diagnostics; no full fit or CUDA acceptance."""
from pathlib import Path
import hashlib
import json
import math
from datetime import datetime, timezone

import numpy as np
import torch

from benchmarks import capture_failed_qp as capture
from clipp1d.api import source_provenance
from clipp1d.cuda import qp
from clipp1d.cuda.kernels import adjoint, differences, gap_kkt


STUDY = Path(__file__).resolve().parent
RECEIPT = STUDY / 'capture-below-a/imported/results/capture-below_one-64.json'
SHA = '4958d20fdaf3f7d047ccd14a75dfb4a5ffadebd2f94c1ee8514a33fecda1b0af'


def stats(x, q, p):
    value = gap_kkt(x, q, *(p[k] for k in ('h', 'target', 'lower', 'upper', 'caps')))
    return dict(gap=float(value[0]), scale=float(value[1]), kkt=float(value[2]),
                qualified=bool(value[0] <= 1e-10 + 1e-11 * value[1]) and bool(value[2] <= 1e-7))


def run():
    source = source_provenance()
    evidence = capture.load_capture(RECEIPT, SHA)
    records = []
    for entry in evidence['captures']:
        saved = capture.load_record(evidence, entry)
        p = {k: capture.load_tensor(evidence, v, 'cpu') for k, v in saved['problem'].items()}
        h, target, lower, upper, caps = (p[k] for k in ('h', 'target', 'lower', 'upper', 'caps'))
        raw_x, raw_q = (capture.load_tensor(evidence, saved['returned'][k], 'cpu') for k in ('x', 'q'))
        x = capture.load_tensor(evidence, saved['last_refined_state']['x'], 'cpu')
        same = x[:, None] == x[None, :]
        external = adjoint(torch.where(same, 0., caps * differences(x).sign()))
        residual = h * (x - target) + external
        capacity = torch.where(same, caps, 0.).sum(1)
        bad = torch.where(x == lower, residual + capacity < 0,
                          torch.where(x == upper, residual - capacity > 0,
                                      residual.abs() > capacity)) & (lower < upper)
        # Diagnostic numerical proposal: retain unaffected exact groups, split
        # only singleton-cut-obstructed coordinates, then recompute block centers
        # with the original terminal ordering and original capacities.
        split = same & ~bad[:, None] & ~bad[None, :]
        split |= torch.eye(x.numel(), dtype=torch.bool)
        adjusted_external = adjoint(torch.where(split, 0., caps * differences(raw_x).sign()))
        weights = torch.where(split, h[None, :], 0.)
        center = x + (torch.where(split, h[None, :] * (target[None, :] - x[:, None])
                                      - adjusted_external[None, :], 0.).sum(1) / weights.sum(1))
        lo = torch.where(split, lower[None, :], -torch.inf).max(1).values
        hi = torch.where(split, upper[None, :], torch.inf).min(1).values
        center = torch.maximum(lo, torch.minimum(hi, center))
        precision = []
        for value in torch.unique(x):
            ids = torch.where(x == value)[0]
            H = h[ids].numpy().astype(np.longdouble)
            T = target[ids].numpy().astype(np.longdouble)
            E = external[ids].numpy().astype(np.longdouble)
            anchor = np.longdouble(float(value))
            root = anchor + (np.sum(H * (T - anchor)) - np.sum(E)) / np.sum(H)
            root = np.clip(root, float(lower[ids].max()), float(upper[ids].min()))
            precision.append(dict(size=len(ids), center=float(value), root=str(root),
                                  displacement_ulps=float((root-anchor)/np.spacing(float(value)))))
        reference = torch.maximum(lower, torch.minimum(upper, target))
        row = dict(context=saved['context'], capture_record_sha256=entry['sha256'],
                   original_stats=stats(raw_x, raw_q, p), saved_groups_precision=precision,
                   split_nodes=torch.where(bad)[0].tolist(),
                   original_objective=float(qp.quadratic_value(raw_x, h, target, caps, reference)),
                   candidate_objective=float(qp.quadratic_value(center, h, target, caps, reference)),
                   candidate_ccf=center.tolist(), probes=[])
        geometry = qp.prepare_flow_polish(center, h, target, lower, upper, caps)
        for method in ('plain_cone', 'fista', 'adaptive_restart'):
            q, y, theta, restarts = raw_q.clone(), raw_q.clone(), 1., 0
            for iteration in range(1, 1025):
                old_q = q
                q = qp.flow_polish_step(y, geometry)
                next_theta = (1 + math.sqrt(1 + 4 * theta * theta)) / 2
                if method == 'plain_cone':
                    y = q
                elif method == 'adaptive_restart' and float(((y-q)*(q-old_q)).sum()) > 0:
                    y, theta = q, 1.
                    restarts += 1
                else:
                    y = q + (theta-1)/next_theta*(q-old_q)
                    theta = next_theta
                if iteration % 16 == 0:
                    checked = stats(center, q, p)
                    if checked['qualified']:
                        break
            row['probes'].append(dict(method=method, steps=iteration, restarts=restarts,
                                      final_stats=checked))
        records.append(row)
    out = dict(scope='CPU fixed-candidate references only; no full QP initialization replay, full fit, or CUDA qualification',
               captured_receipt_sha256=SHA, source_sha256=source['source_sha256'],
               driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               created_utc=datetime.now(timezone.utc).isoformat(), records=records,
               limitation='Group-local splitting and acceleration are deferred proposals, not production changes. All reported qualifications use original QP gates on a fixed candidate.')
    target_path = STUDY / 'BELOW64_RECOVERY_PROBE.json'
    with target_path.open('x') as stream:
        json.dump(out, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact':str(target_path), 'source_sha256':source['source_sha256'],
                      'probes':[row['probes'] for row in records]}))


if __name__ == '__main__':
    run()
