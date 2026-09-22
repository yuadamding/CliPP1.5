"""Historical constrained 72ba3ed surrogate, clipping-snap and outer-gate replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-package', type=Path, required=True)
    parser.add_argument('--fixture-directory', type=Path, required=True)
    parser.add_argument('--diagnosis-directory', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    from benchmark_chain import configure_execution
    controls = configure_execution(cpus=[args.cpu], threads=1)
    import numpy as np
    from diagnose_observed_dual import observed_adjoint_bounds, reachable_dual, independent_lp
    from diagnose_rejections import stable_loss_delta, clean
    sys.path.insert(0, str(args.frozen_package.resolve().parent))
    from clipp1d import api, solver
    from clipp1d.io import read_tumor
    from clipp1d.model import compile_model
    from clipp1d.policy import Policy
    if (Path(solver.__file__).resolve().parent != args.frozen_package.resolve() or
            api.source_provenance()['source_sha256'] != '30a019eba2c99dbfae23b26200164893ab6ac54d82d2ef60a1dec2ce4f8e4055'):
        raise RuntimeError('Expected historical constrained frozen 72ba3ed package')
    setup = json.loads((args.fixture_directory / 'setup.json').read_text())
    assert sha(setup['input_path']) == setup['input_sha256']
    model = compile_model(read_tumor(setup['input_path']))
    policy = Policy()
    args.outdir.mkdir(parents=True, exist_ok=False)
    records = []
    for file in sorted((args.diagnosis_directory / 'rejections').glob('*-worst-rejected.npz')):
        tag = file.stem.removesuffix('-worst-rejected')
        with np.load(file) as d, np.load(args.fixture_directory / ('unresolved-' + tag + '.npz')) as original:
            m = model.subset(original['chain_order'])
            x, h, g, caps = (d[k] for k in ('x', 'h', 'gradient', 'caps'))
            target = x - g / h
            p = solver.profile_quadratic_witnesses(h, target, m.lower, m.upper, caps, policy)
            lo, hi = m.lower.copy(), m.upper.copy()
            lo[p.witness] = hi[p.witness] = 1.
            cache = {}
            snap = solver._snap_crossed_breakpoints(m, x, p.fit.x, lo, hi, caps, cache)
            assert np.array_equal(snap, d['trial'])
            context = solver.prepare_audit(m, x)
            stages = []
            for label, y in [('direct_qp', p.fit.x), ('after_clipping_snap', snap)]:
                step = y - x
                tv_delta = float(np.dot(caps, abs(np.diff(y)) - abs(np.diff(x))))
                loss_delta = stable_loss_delta(m, x, y)
                excess = loss_delta - g.astype(np.longdouble) * step - .5 * h.astype(np.longdouble) * step ** 2
                positive = np.maximum(excess, 0)
                stages.append(dict(stage=label, max_step=float(np.max(abs(step))),
                   observed_objective_delta=solver.objective(m, y, caps) - solver.objective(m, x, caps),
                   guarded_observed_delta=float(np.sum(loss_delta) + tv_delta),
                   stable_surrogate_delta=float(np.sum(g * step + .5 * h * step ** 2) + tv_delta),
                   positive_coordinate_excess_count=int(np.count_nonzero(positive > 1e-12)),
                   max_coordinate_excess=float(np.max(positive)), positive_excess_sum=float(np.sum(positive)),
                   positive_excess_node0=float(positive[0]),
                   node0_positive_fraction=float(positive[0] / np.sum(positive)) if np.any(positive) else 0.))
            changed = np.flatnonzero(snap != p.fit.x)
            nodes = [dict(index=int(i), mutation_id=m.mutation_ids[i], alt_count=float(m.alt[i]),
                ref_count=float(m.ref[i]), old=float(x[i]), qp=float(p.fit.x[i]), snap=float(snap[i]),
                ordinary_gradient=float(g[i]), left_derivative=float(context.left[i]),
                right_derivative=float(context.right[i]), h=float(h[i]), slopes=m.slope[i].tolist()) for i in changed]
            old_anchor = int(np.flatnonzero(x == 1)[0])
            al, au = m.lower.copy(), m.upper.copy()
            al[old_anchor] = au[old_anchor] = 1.
            direct = solver.solve_quadratic(h, target, al, au, caps, x, policy)
            step = direct.x - x
            qp_checks = dict(selected_witness=p.witness, previous_anchor=old_anchor,
                profile_qualified=p.qualified, inner_gap=p.fit.gap, inner_gap_scale=p.fit.gap_scale,
                inner_gap_gate=policy.inner_atol + policy.inner_rtol * p.fit.gap_scale,
                inner_kkt=p.fit.kkt_residual, prefix_value_error=p.diagnostics['prefix_value_error'],
                previous_anchor_profile_value_difference=float(p.relative_objectives[old_anchor] - p.relative_objectives[p.witness]),
                previous_anchor_direct_max_phi_difference=float(np.max(abs(direct.x - p.fit.x))),
                previous_anchor_direct_surrogate_delta=float(np.sum(g * step + .5 * h * step ** 2) +
                   np.dot(caps, abs(np.diff(direct.x)) - abs(np.diff(x)))))
            # Gate feasibility certificate in the original final state, independent
            # of all clonal anchors when the obstruction precedes first occupied1.
            saved = np.load(args.diagnosis_directory / 'observed-dual' / (tag + '-observed-dual.npz'))
            final = saved['x']
            ctx = solver.prepare_audit(m, final)
            a_low, a_high = observed_adjoint_bounds(ctx, final, saved['lower'], saved['upper'], policy.stationarity_tol)
            cl, ch = -caps.copy(), caps.copy()
            active = np.diff(final) != 0
            cl[active] = ch[active] = caps[active] * np.sign(np.diff(final)[active])
            obstruction = None
            lower = upper = np.longdouble(0)
            for i in range(len(final)):
                edge_lo, edge_hi = (cl[i], ch[i]) if i < len(caps) else (0., 0.)
                lower = max(edge_lo, lower - np.longdouble(a_high[i]))
                upper = min(edge_hi, upper - np.longdouble(a_low[i]))
                if lower > upper:
                    obstruction = dict(node=int(i), mutation_id=m.mutation_ids[i],
                         reachable_gap=float(lower - upper), phi=float(final[i]),
                         before_any_occupied_witness=bool(i < np.flatnonzero(final == 1)[0]))
                    break
            gate_times = []
            for _ in range(7):
                started = perf_counter()
                low, high = observed_adjoint_bounds(ctx, final, saved['lower'], saved['upper'], policy.stationarity_tol)
                q_gate = reachable_dual(low, high, cl, ch, saved['retained'])
                gate_times.append(perf_counter() - started)
            lp = independent_lp(ctx, final, saved['lower'], saved['upper'], caps, policy.stationarity_tol)
            lp_actual = None if not lp.success else solver.stationarity(m, final, lp.x, saved['lower'], saved['upper'], caps, context=ctx)[0]
            record = dict(case=tag, rejected_fixture_sha256=sha(file), stages=stages, snapped_nodes=nodes,
               qp_checks=qp_checks, first_observed_dual_obstruction=obstruction,
               original_gate_feasibility_pass_seconds=gate_times, feasibility_pass_found_dual=q_gate is not None,
               lp_returned_dual_original_stationarity=lp_actual)
            records.append(record)
            print(json.dumps(clean(record)), flush=True)
    with (args.outdir / 'snap-mechanism.json').open('x') as stream:
        json.dump(clean(dict(source=api.source_provenance(), input_sha256=setup['input_sha256'], controls=controls,
            wrapper_sha256=sha(__file__), records=records,
            scope='Offline exact surrogate replays and fixed-primal certificates; every production numerical gate remains unchanged. Gate-pass timings exclude context construction and are shared-host CPU1 samples. LP shares derivative-interval derivation but independently checks linear feasibility.')), stream, indent=2)


if __name__ == '__main__':
    main()
