"""Offline observed-gradient dual diagnostic; never edits gradients or fit states.

For fixed x, exact jump complementarity fixes some edge duals. Box and clipping
subgradient stationarity gives an interval of admissible D.T@q at each node for
any requested normalized residual. A forward interval-reachability pass and
backtracking find a feasible chain dual. Bisection seeks the smallest residual.
This is diagnostic code, not an alternate production convergence gate.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def gradient_adjoint_interval(gradient, tolerance):
    """Exact a-interval for |g+a| <= t*(1+|g|+|a|), with 0<=t<1."""
    import numpy as np
    g = np.asarray(gradient, dtype=float)
    if not 0 <= tolerance < 1:
        raise ValueError("Normalized tolerance must be in [0,1)")
    base = tolerance * (1 + np.abs(g))
    numerator = -g - base
    lower = numerator / np.where(numerator < 0, 1 - tolerance, 1 + tolerance)
    numerator = -g + base
    upper = numerator / np.where(numerator > 0, 1 - tolerance, 1 + tolerance)
    return lower, upper


def observed_adjoint_bounds(context, x, lower, upper, tolerance):
    """Use the exact same derivative and box-normal conventions as stationarity."""
    import numpy as np
    left, right = context.left, context.right
    upward = left <= right
    g_low = np.where(upward, left, context.gradient)
    g_high = np.where(upward, right, context.gradient)
    a_low = gradient_adjoint_interval(g_high, tolerance)[0]
    a_high = gradient_adjoint_interval(g_low, tolerance)[1]
    at_lower, at_upper = x == lower, x == upper
    a_low = np.where(at_lower, gradient_adjoint_interval(right, tolerance)[0], a_low)
    a_high = np.where(at_lower, np.inf, a_high)
    a_high = np.where(at_upper, gradient_adjoint_interval(left, tolerance)[1], a_high)
    a_low = np.where(at_upper, -np.inf, a_low)
    fixed = lower == upper
    return np.where(fixed, -np.inf, a_low), np.where(fixed, np.inf, a_high)


def reachable_dual(a_low, a_high, edge_low, edge_high, preferred=None):
    """Find q with a_low<=D.T@q<=a_high and interval edge constraints."""
    import numpy as np
    n = len(a_low)
    if (len(a_high) != n or len(edge_low) != n - 1 or len(edge_high) != n - 1):
        raise ValueError("Mismatched chain dimensions")
    a_low, a_high = np.asarray(a_low, dtype=np.longdouble), np.asarray(a_high, dtype=np.longdouble)
    lows, highs = np.empty(n + 1, dtype=np.longdouble), np.empty(n + 1, dtype=np.longdouble)
    lows[0] = highs[0] = 0
    for i in range(n):
        lo, hi = (edge_low[i], edge_high[i]) if i + 1 < n else (0., 0.)
        lows[i + 1] = max(lo, lows[i] - a_high[i])
        highs[i + 1] = min(hi, highs[i] - a_low[i])
        if lows[i + 1] > highs[i + 1]:
            return None
    q = np.zeros(n + 1, dtype=np.longdouble)
    for i in range(n - 1, 0, -1):
        lo = max(lows[i], q[i + 1] + a_low[i])
        hi = min(highs[i], q[i + 1] + a_high[i])
        if lo > hi:
            # Reverse additions can differ from forward subtractions by guarded
            # arithmetic roundoff. This changes no admission gate: callers must
            # recheck the returned float64 dual with original stationarity.
            guard = 16 * np.finfo(np.longdouble).eps * (1 + abs(lo) + abs(hi))
            if lo - hi > guard:
                return None
            lo = hi = (lo + hi) / 2
        value = 0 if preferred is None else preferred[i - 1]
        q[i] = min(hi, max(lo, value))
    return np.asarray(q[1:n], dtype=float)


def edge_bounds(x, caps, tolerance, mode='exact'):
    """Exact constraints, or the existing validator's declared residual budget."""
    import numpy as np
    if mode not in ('exact', 'edge_tolerant', 'full_gate'):
        raise ValueError('Unknown diagnostic edge-constraint mode')
    allowance = tolerance * (1 + caps)
    box_allowance = allowance if mode == 'full_gate' else 0.
    low, high = -caps - box_allowance, caps + box_allowance
    active = np.diff(x) != 0
    target = caps * np.sign(np.diff(x))
    jump_allowance = allowance if mode != 'exact' else np.zeros_like(caps)
    low[active] = np.maximum(low[active], target[active] - jump_allowance[active])
    high[active] = np.minimum(high[active], target[active] + jump_allowance[active])
    return low, high


def observed_dual(context, x, lower, upper, caps, preferred, upper_tolerance, *, mode='exact'):
    """Minimize node residual while keeping dual boxes/jump complementarity exact."""
    import numpy as np
    jumps = np.diff(x)
    active = jumps != 0

    def at(t):
        al, ah = observed_adjoint_bounds(context, x, lower, upper, t)
        edge_low, edge_high = edge_bounds(x, caps, t, mode)
        return reachable_dual(al, ah, edge_low, edge_high, preferred)

    lo, hi = 0., min(.5, max(upper_tolerance * 1.01, 1e-12))
    while at(hi) is None:
        hi = (1 + hi) / 2
        if hi >= 1 - 1e-12:
            raise ArithmeticError("Could not bracket observed dual feasibility")
    for _ in range(60):
        middle = (lo + hi) / 2
        if at(middle) is None:
            lo = middle
        else:
            hi = middle
    # Roundoff headroom is diagnostic reconstruction only. Returned vectors still
    # undergo the original observed-data stationarity gate, never a relaxed gate.
    target = hi + 1e-12 * (1 + hi)
    q = at(target)
    return q, dict(infeasible_below=lo, feasible_above=hi, reconstruction_target=target,
                   constraint_mode=mode, nonzero_jump_constraints=int(np.count_nonzero(active)))


def independent_lp(context, x, lower, upper, caps, tolerance, *, mode='exact'):
    """Independent sparse linear-programming feasibility check at a fixed gate."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, vstack
    n = len(x)
    al, ah = observed_adjoint_bounds(context, x, lower, upper, tolerance)
    cols = np.repeat(np.arange(n - 1), 2)
    rows = np.column_stack((np.arange(n - 1), np.arange(1, n))).ravel()
    values = np.tile([-1., 1.], n - 1)
    matrix = coo_matrix((values, (rows, cols)), shape=(n, n - 1)).tocsr()
    finite_hi, finite_lo = np.isfinite(ah), np.isfinite(al)
    a = vstack((matrix[finite_hi], -matrix[finite_lo]), format='csr')
    b = np.r_[ah[finite_hi], -al[finite_lo]]
    edge_low, edge_high = edge_bounds(x, caps, tolerance, mode)
    bounds = list(zip(edge_low, edge_high))
    result = linprog(np.zeros(n - 1), A_ub=a, b_ub=b, bounds=bounds,
                     method='highs', options={'dual_feasibility_tolerance': 1e-9,
                                              'primal_feasibility_tolerance': 1e-9})
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-package', type=Path, required=True)
    parser.add_argument('--fixture-directory', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    from benchmark_chain import configure_execution
    controls = configure_execution(cpus=[args.cpu], threads=1)
    import numpy as np
    sys.path.insert(0, str(args.frozen_package.resolve().parent))
    from clipp1d import api, solver
    from clipp1d.io import read_tumor
    from clipp1d.model import compile_model
    from clipp1d.policy import Policy
    if Path(solver.__file__).resolve().parent != args.frozen_package.resolve():
        raise RuntimeError("Wrong package imported")
    source = api.source_provenance()
    if source['source_sha256'] != '30a019eba2c99dbfae23b26200164893ab6ac54d82d2ef60a1dec2ce4f8e4055':
        raise RuntimeError("Expected frozen 72ba3ed source")
    setup = json.loads((args.fixture_directory / 'setup.json').read_text())
    assert sha(setup['input_path']) == setup['input_sha256']
    policy = Policy()
    model = compile_model(read_tumor(setup['input_path'], policy), policy)
    args.outdir.mkdir(parents=True, exist_ok=False)
    records = []
    for fixture in sorted(args.fixture_directory.glob('unresolved-*.npz')):
        tag = fixture.stem.removeprefix('unresolved-')
        detail = args.fixture_directory / ('start-' + tag + '.json')
        receipt = json.loads(detail.read_text())
        assert sha(fixture) == receipt['fixture_sha256']
        with np.load(fixture) as data:
            ordered = model.subset(data['chain_order'])
            x, retained = data['final'], data['final_dual']
            caps = receipt['lambda_value'] * data['chain_weights']
            lower, upper = ordered.lower.copy(), ordered.upper.copy()
            witness = receipt['final_witness']
            lower[witness] = upper[witness] = 1.
            context = solver.prepare_audit(ordered, x)
            before, feasible = solver.stationarity(ordered, x, retained, lower, upper, caps, context=context)
            reconstructed, bounds = observed_dual(context, x, lower, upper, caps, retained, before)
            after, feasible_after = solver.stationarity(ordered, x, reconstructed, lower, upper, caps, context=context)
            lp = independent_lp(context, x, lower, upper, caps, policy.stationarity_tol)
            audits = []
            for anchor in dict.fromkeys((witness, int(np.flatnonzero(x == 1)[0]), int(np.flatnonzero(x == 1)[-1]))):
                al, au = ordered.lower.copy(), ordered.upper.copy()
                al[anchor] = au[anchor] = 1.
                ok, restart = solver._kink_check(ordered, x, caps, al, au, policy,
                                                  force_intervals=anchor != witness, context=context)
                audits.append(dict(anchor=anchor, audit_passed=bool(ok), restart=restart is not None))
            np.savez_compressed(args.outdir / (tag + '-observed-dual.npz'), x=x, retained=retained,
                                reconstructed=reconstructed, caps=caps, lower=lower, upper=upper,
                                gradient=context.gradient, left=context.left, right=context.right)
            record = dict(case=tag, fixture_sha256=sha(fixture), detail_sha256=sha(detail),
                lambda_value=receipt['lambda_value'], witness=witness, retained_stationarity=before,
                observed_dual_stationarity=after, feasible=feasible and feasible_after,
                unchanged_stationarity_gate=policy.stationarity_tol, observed_dual_passes=after <= policy.stationarity_tol,
                minimax_bounds=bounds, lp_at_unchanged_gate=dict(success=lp.success, status=lp.status, message=lp.message),
                interval_audits=audits, x_unchanged=True,
                qualification_after_reconstruction=bool(after <= policy.stationarity_tol and feasible_after and
                   all(r['audit_passed'] and not r['restart'] for r in audits)))
            records.append(record)
            print(json.dumps(record), flush=True)
    report = dict(schema='clipp1d.observed_dual_diagnosis.v1', source=source, controls=controls,
        original_fixture_source=setup['source'], input_sha256=setup['input_sha256'], policy=asdict(policy),
        harness_sha256=sha(__file__), records=records,
        scope='Fixed original primal vectors; original observed-data derivatives without any block correction; exact nonzero-jump complementarity. Minimax residual reconstruction and independent sparse LP are offline diagnostics, not production gates.',
        source_unchanged=api.source_provenance()['source_sha256'] == source['source_sha256'])
    with (args.outdir / 'observed-dual.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
