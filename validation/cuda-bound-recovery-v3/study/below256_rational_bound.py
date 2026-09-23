"""Exact-rational mathematical gate bound for literal captured QP U.

No optimization or source mutation. The proof treats binary64 inputs exactly
and states separately its mathematical gates and floating GPU execution limits.

Use --evidence-root for either the original study directory, an archive root,
or its attempts directory. --repository-root explicitly selects the repository
that supplies the hash-verifying capture loader; by default it is discovered
among this script's ancestors. Receipts are created exclusively, never replaced.
"""
from fractions import Fraction
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CAPTURE_SHA = '910a09a7143e27cb4f27e693dec62a4e38db07adb078dc7015b16fa10df2bfc3'
CAPTURE_RELATIVE = Path('capture-first-below256-u/imported/results/capture-first-below256.json')


def repository_root(explicit):
    candidates = [explicit.resolve()] if explicit else Path(__file__).resolve().parents
    for candidate in candidates:
        if ((candidate/'benchmarks/capture_failed_qp.py').is_file()
                and (candidate/'src/clipp1d').is_dir()):
            return candidate
    raise FileNotFoundError('Pass --repository-root for the CliPP1.5 capture loader')


def capture_path(explicit):
    roots = [explicit.resolve()] if explicit else [ROOT, ROOT.parent]
    for root in roots:
        for candidate in (root/CAPTURE_RELATIVE, root/'attempts'/CAPTURE_RELATIVE):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError('Pass --evidence-root for the original study or archived attempts')


def rational(value):
    return Fraction(float(value))


def packed(value):
    literal = f'{value.numerator}/{value.denominator}'.encode()
    return dict(approximate=float(value), numerator_bits=value.numerator.bit_length(),
                denominator_bits=value.denominator.bit_length(), exact_rational_sha256=hashlib.sha256(literal).hexdigest())


def root_upper(value):
    assert value >= 0
    if not value:
        return Fraction(0)
    upper = math.nextafter(math.sqrt(math.nextafter(float(value), math.inf)), math.inf)
    result = Fraction(upper)
    assert result*result >= value  # Exact verification of the outward rounding.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path)
    parser.add_argument('--repository-root', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'BELOW256_RATIONAL_BOUND_V3.json')
    args = parser.parse_args()
    sys.path.insert(0, str(repository_root(args.repository_root)))
    from benchmarks import capture_failed_qp as capture

    receipt = capture.load_capture(capture_path(args.evidence_root), CAPTURE_SHA)
    assert len(receipt['captures']) == 1
    entry = receipt['captures'][0]
    record = capture.load_record(receipt, entry)
    problem = {key: capture.load_tensor(receipt, value, 'cpu') for key, value in record['problem'].items()}
    saved = record['last_refined_state']
    x, q = [capture.load_tensor(receipt, saved[key], 'cpu') for key in ('x', 'q')]
    h, target, lower, upper, caps = [problem[key] for key in ('h', 'target', 'lower', 'upper', 'caps')]
    n, index = len(x), 139
    assert bool((lower < upper).all() & (x >= lower).all() & (x <= upper).all())
    assert bool((q == -q.T).all() & (q.abs() <= caps).all())
    difference = x[None, :]-x[:, None]
    assert bool(((difference == 0) | (q == caps*difference.sign())).all())
    H, T, L, U, X = [[rational(v) for v in array] for array in (h, target, lower, upper, x)]
    A = [-sum((rational(value) for value in row), Fraction(0)) for row in q]
    gap, scale = Fraction(0), Fraction(0)
    for i in range(n):
        minimizer = min(U[i], max(L[i], T[i]-A[i]/H[i]))
        delta = X[i]-minimizer
        normal = H[i]*(minimizer-T[i])+A[i]
        term = H[i]*delta*delta/2+normal*delta
        assert term >= 0
        gap += term  # Exact edge complementarity above makes every edge term zero.
        reference = min(U[i], max(L[i], T[i]))
        d = X[i]-reference
        energy = H[i]*d*d/2+H[i]*(reference-T[i])*d
        assert energy >= 0
        scale += energy
        for j in range(i+1, n):
            scale += rational(caps[i, j])*abs(X[j]-X[i])
    policy = record['context']['policy']
    atol, rtol, tau = [rational(policy[key]) for key in ('inner_atol', 'inner_rtol', 'inner_kkt_tol')]
    assert 0 <= rtol < 1 and 0 <= tau < 1 and atol >= 0
    # Scale S equals the QP primal objective plus a fixed constant. For any
    # exact mathematical gate pass, G >= S-S* >= S-S0 and G <= atol+rtol*S.
    maximum_scale = (scale+atol)/(1-rtol)
    maximum_gap = atol+rtol*maximum_scale
    # Strong convexity gives coordinate distances from the unique real optimum;
    # this triangle bound is centered at the captured feasible primal instead.
    radii = [root_upper(2*maximum_gap/weight)+root_upper(2*gap/weight) for weight in H]
    assert X[index]-radii[index] > L[index] and X[index]+radii[index] < U[index]
    separations = {j: abs(X[index]-X[j])-radii[index]-radii[j] for j in range(n) if j != index}
    assert all(value > 0 for value in separations.values())
    # Thus m139 remains an interior singleton, with the same incident-edge signs
    # at every primal admitted by the exact mathematical gap gate.
    saturated_a = -sum((rational(caps[index, j])*(1 if X[j] > X[index] else -1)
                        for j in range(n) if j != index), Fraction(0))
    assert saturated_a > 0 and A[index] == saturated_a
    maximum_gradient = H[index]*(X[index]+radii[index]-T[index])
    # At a nonpositive adjoint, |g+a|/(1+|g|+|a|) cannot be <= tau
    # anywhere in this region. This establishes the positive-adjoint branch
    # of the KKT interval for every represented primal, not just its two roots.
    assert maximum_gradient < -tau/(1-tau)
    stationary = T[index]-saturated_a/H[index]
    rounded = float(stationary)
    low = rounded if rational(rounded) <= stationary else math.nextafter(rounded, -math.inf)
    high = rounded if rational(rounded) >= stationary else math.nextafter(rounded, math.inf)
    assert rational(low) <= stationary <= rational(high)
    assert math.nextafter(low, math.inf) == high
    neighbors = []
    for value in (low, high):
        point = rational(value)
        gradient = H[index]*(point-T[index])
        assert gradient < 0
        residual = gradient+saturated_a
        # Passing KKT must use a positive adjoint a', proved across the region.
        # Solve its admissible interval exactly, without changing the gate.
        lo_a = (-tau-gradient*(1-tau))/(1+tau)
        hi_a = (tau-gradient*(1+tau))/(1-tau)
        assert lo_a > 0
        increase = max(lo_a-saturated_a, Fraction(0))
        decrease = max(saturated_a-hi_a, Fraction(0))
        required = max(increase, decrease)
        assert required > 0
        relevant = [j for j in separations if (X[j] > X[index]) == bool(increase)]
        distance = min(separations[j] for j in relevant)
        cost = required*distance
        assert cost > maximum_gap
        neighbors.append(dict(ccf=value, stationarity_residual=packed(residual),
                              normalized_kkt=packed(abs(residual)/(1+abs(gradient)+saturated_a)),
                              required_adjoint_change=packed(required),
                              direction='increase' if increase else 'decrease',
                              minimum_separation_over_entire_admitted_region=packed(distance),
                              minimum_incident_edge_gap=packed(cost),
                              gap_cost_to_global_allowance=packed(cost/maximum_gap)))
    result = dict(schema='clipp1d.literal_qp_rational_gate_bound.v1',
                  scope='Exact rational arithmetic over literal binary64 QP inputs and policy constants. This proves incompatibility for exact mathematical gap/KKT gates with a binary64 primal; it is not a formal interval verification of every CUDA reduction/rounding context.',
                  status='exact_mathematical_gate_incompatibility_proved',
                  capture_sha256=CAPTURE_SHA, record_sha256=entry['sha256'],
                  captured_source_sha256=record['context']['source_sha256'],
                  script_sha256=capture.common.sha(Path(__file__)), node_index=index,
                  captured_exact_primal_dual_gap=packed(gap), captured_exact_objective_scale=packed(scale),
                  global_scale_upper_bound_for_any_mathematical_gate_pass=packed(maximum_scale),
                  global_gap_upper_bound_for_any_mathematical_gate_pass=packed(maximum_gap),
                  node_radius_about_captured_primal=packed(radii[index]),
                  maximum_other_node_radius=packed(max(value for i, value in enumerate(radii) if i != index)),
                  maximum_gradient_over_admitted_region=packed(maximum_gradient),
                  positive_adjoint_necessary_for_kkt_proved=True,
                  all_incident_edge_orderings_proved_unchanged=True,
                  singleton_proved_interior=True, exact_stationary_root=packed(stationary),
                  adjacent_floats_bracketing_stationary_root=neighbors,
                  reason_all_other_floats_are_excluded='Within the proved region g is negative and the saturated adjoint is positive. The distance from the saturated adjoint to the exact KKT-admissible interval increases monotonically as the primal moves away from the stationary root. The two bracketing floats therefore lower-bound the required flow adjustment for every binary64 primal in that region.',
                  implication='More fixed-candidate flow iterations, higher-accuracy center arithmetic, or another ADMM trajectory cannot remove this exact mathematical represented-primal obstruction for this literal surrogate. The existing solver must keep its failed status unless its unchanged actual CUDA certificate passes.',
                  floating_execution_limit='Actual eager/compiled certificates and W replay separately report failure. This rational proof does not authorize a tolerance change, alternative certificate arithmetic, perturbed surrogate, or promotion of the unresolved QP.')
    out = args.output
    capture.common.write_json(out, result)
    print(json.dumps(dict(output=str(out), sha256=capture.common.sha(out),
                          exact_gap=float(gap), global_gap_bound=float(maximum_gap),
                          minimum_edge_gap_bound=min(row['minimum_incident_edge_gap']['approximate'] for row in neighbors),
                          minimum_gap_ratio=min(row['gap_cost_to_global_allowance']['approximate'] for row in neighbors))))


if __name__ == '__main__':
    main()
