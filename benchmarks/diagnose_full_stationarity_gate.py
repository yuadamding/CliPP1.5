"""Historical constrained 72ba3ed dual and stationarity-gate diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-package', type=Path, required=True)
    parser.add_argument('--fixture-directory', type=Path, required=True)
    parser.add_argument('--observed-directory', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    from benchmark_chain import configure_execution
    controls = configure_execution(cpus=[args.cpu], threads=1)
    import numpy as np
    from diagnose_observed_dual import observed_dual, independent_lp
    from diagnose_rejections import clean
    sys.path.insert(0, str(args.frozen_package.resolve().parent))
    from clipp1d import api, solver
    from clipp1d.io import read_tumor
    from clipp1d.model import compile_model
    from clipp1d.policy import Policy
    source = api.source_provenance()
    if (Path(solver.__file__).resolve().parent != args.frozen_package.resolve() or
            source['source_sha256'] != '30a019eba2c99dbfae23b26200164893ab6ac54d82d2ef60a1dec2ce4f8e4055'):
        raise RuntimeError('Expected historical constrained frozen 72ba3ed package')
    setup = json.loads((args.fixture_directory / 'setup.json').read_text())
    assert sha(setup['input_path']) == setup['input_sha256']
    model = compile_model(read_tumor(setup['input_path']))
    policy = Policy()
    args.outdir.mkdir(parents=True, exist_ok=False)
    records = []
    for file in sorted(args.observed_directory.glob('*-observed-dual.npz')):
        tag = file.stem.removesuffix('-observed-dual')
        with np.load(file) as d, np.load(args.fixture_directory / ('unresolved-' + tag + '.npz')) as original:
            m = model.subset(original['chain_order'])
            x, lower, upper, caps = (d[k] for k in ('x', 'lower', 'upper', 'caps'))
            context = solver.prepare_audit(m, x)
            retained = solver.stationarity(m, x, d['retained'], lower, upper, caps, context=context)[0]
            for mode in ('edge_tolerant', 'full_gate'):
                q, bracket = observed_dual(context, x, lower, upper, caps, d['retained'], retained, mode=mode)
                residual, feasible = solver.stationarity(m, x, q, lower, upper, caps, context=context)
                lp = independent_lp(context, x, lower, upper, caps, policy.stationarity_tol, mode=mode)
                lp_residual = None if not lp.success else solver.stationarity(m, x, lp.x, lower, upper, caps, context=context)[0]
                witness = int(np.flatnonzero(lower == upper)[0])
                audits = []
                for anchor in dict.fromkeys((witness, int(np.flatnonzero(x == 1)[0]), int(np.flatnonzero(x == 1)[-1]))):
                    al, au = m.lower.copy(), m.upper.copy()
                    al[anchor] = au[anchor] = 1.
                    ok, restart = solver._kink_check(m, x, caps, al, au, policy,
                         force_intervals=anchor != witness, context=context)
                    audits.append(dict(anchor=anchor, passed=bool(ok), restart=restart is not None))
                passed = feasible and residual <= policy.stationarity_tol and all(r['passed'] and not r['restart'] for r in audits)
                row = dict(case=tag, constraint_mode=mode, retained_stationarity=retained,
                    reconstructed_stationarity=residual, minimax_bracket=bracket,
                    original_gate=policy.stationarity_tol, all_original_audits_passed=passed,
                    exact_dual_box_violation=float(np.max(np.maximum(abs(q) - caps, 0))),
                    normalized_dual_box_violation=float(np.max(np.maximum(abs(q) - caps, 0) / (1 + caps))),
                    lp_gate_success=lp.success, lp_gate_status=lp.status,
                    lp_returned_dual_original_stationarity=lp_residual, interval_audits=audits)
                records.append(row)
                np.savez_compressed(args.outdir / (tag + '-' + mode + '.npz'), x=x, dual=q)
                print(json.dumps(clean(row)), flush=True)
    with (args.outdir / 'full-gate.json').open('x') as stream:
        json.dump(clean(dict(source=source, input_sha256=setup['input_sha256'], controls=controls,
            wrapper_sha256=sha(__file__), dual_helper_sha256=sha(Path(__file__).with_name('diagnose_observed_dual.py')),
            records=records, scope='Original stationarity_tol unchanged. edge_tolerant retains strict dual boxes while allowing the existing normalized nonzero-jump residual. full_gate also includes the existing normalized dual-box residual allowance. Every returned dual is independently passed to the unmodified stationarity implementation and unchanged selected/extreme-anchor audits. This is separate from exact mathematical dual complementarity.')), stream, indent=2)


if __name__ == '__main__':
    main()
