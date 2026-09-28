"""Allocated-CUDA qualification; CPU/eager/compiled comparisons stay distinct."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prior_perturbation import (  # noqa: E402
    auxiliary_model, array_record, keyed_directions, now, prepare_model, run_case,
    sha, write_json,
)
from clipp1d.cuda.experimental import solve_extra_start  # noqa: E402
from clipp1d.cuda.graph import build_graph  # noqa: E402
from clipp1d.cuda.partition import QualifiedPilot, refit_labels  # noqa: E402
from clipp1d.cuda.scalar import pilot  # noqa: E402
from clipp1d.cuda_api import require_cuda  # noqa: E402
from clipp1d.api import source_provenance  # noqa: E402


def qualify_cold_processes(inputs, outdir, case, device):
    """Exercise both measured entry points before any CUDA use in each child."""
    records = []
    source = inputs/case['input_file']
    assert sha(source) == case['input_sha256']
    for mode in ('comparison', 'timing'):
        target = outdir/f'cold-process-{mode}'
        code = '''
import json, sys
from pathlib import Path
import torch
assert not torch.cuda.is_initialized(), "Qualification child was not CUDA-cold"
sys.path.insert(0, sys.argv[1])
mode, source, target, tumor, device = sys.argv[2:]
if mode == 'comparison':
    from prior_perturbation import run_case
    run_case(Path(source), Path(target), tumor_id=tumor, device=device)
else:
    from time_prior_perturbation import measure_once
    measure_once(Path(source), tumor, Path(target), 'B', device=device)
assert torch.cuda.is_initialized()
print(json.dumps(dict(mode=mode, initially_initialized=False, completed=True)))
'''
        with (outdir/f'cold-process-{mode}.out').open('x') as stdout, \
                (outdir/f'cold-process-{mode}.err').open('x') as stderr:
            subprocess.run([sys.executable, '-B', '-c', code, str(Path(__file__).resolve().parent), mode, str(source),
                            str(target), case['case_id'], str(device)],
                           stdout=stdout, stderr=stderr, check=True, timeout=1800)
        name = 'RESULT.json' if mode == 'comparison' else 'MEASUREMENT.json'
        result = json.loads((target/name).read_text())
        assert result['source']['source_sha256'] == source_provenance()['source_sha256']
        if mode == 'comparison':
            assert result['original_raw_preserved']
            assert result['result_scores']['P'] <= result['result_scores']['B']
        else:
            assert result['peak_device_allocated_bytes'] > 0
        records.append(dict(mode=mode, case_id=case['case_id'], initially_initialized=False,
                            result_sha256=sha(target/name), returncode=0))
    write_json(outdir/'COLD_PROCESS_QUALIFICATION.json', dict(status='passed', records=records))
    return records


def alias_diagnostic(original, outdir, mode):
    """Independent bracketing of both known wells; a switch is not accuracy."""
    records = []
    for sign in (-1, 0, 1):
        direction = torch.zeros_like(original.log_prior)
        direction[0, :2] = direction.new_tensor([-sign, sign])
        model = auxiliary_model(original, direction, .01)
        wells = []
        for bounds in ((.25, .5), (.6, .9)):
            result = minimize_scalar(lambda x: independent(model, np.array([x]))[0][0],
                                     bounds=bounds, method='bounded', options={'xatol': 1e-14})
            assert result.success
            wells.append(dict(phi=float(result.x), loss=float(result.fun)))
        fitted = pilot(model)
        QualifiedPilot(model, fitted)
        preferred = min(wells, key=lambda row: row['loss'])
        assert abs(float(fitted.loss[0])-preferred['loss']) <= 1e-7
        if sign:
            assert abs(float(fitted.phi[0])-(.375 if sign == 1 else .75)) < .001
        records.append(dict(sign=sign, eta=.01, wells=wells, pilot=float(fitted.phi[0]),
            arrays=array_record(outdir/f'alias-{mode}-{sign}.npz', phi=fitted.phi,
                loss=fitted.loss, posterior=model.terms(fitted.phi)[3], prior=model.log_prior)))
    return records


def independent(model, points):
    a, r, s, prior = (getattr(model, name).cpu().numpy() for name in ('alt', 'ref', 'slope', 'log_prior'))
    p = np.clip(points[:, None]*s, model.eps, 1-model.eps)
    joint = a[:, None]*np.log(p)+r[:, None]*np.log1p(-p)+prior
    norm = logsumexp(joint, axis=1)
    q = np.exp(joint-norm[:, None])
    active = (points[:, None]*s > model.eps) & (points[:, None]*s < 1-model.eps)
    grad = -(q*s*active*(a[:, None]/p-r[:, None]/(1-p))).sum(-1)
    return -norm, q, grad


def qualify(inputs, outdir, device='cuda:0'):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = json.loads((inputs/'FIT_MANIFEST.json').read_text())
    receipt = dict(utc=now(), status='running',
                   torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   inputs_manifest_sha256=sha(inputs/'FIT_MANIFEST.json'), source=source_provenance(), cases=[])
    try:
        # The parent must not own a CUDA context while an exclusive-GPU child
        # runs. Start the cold checks before even probing the parent's device.
        receipt['cold_processes'] = qualify_cold_processes(inputs, outdir, manifest['cases'][0], device)
        device = require_cuda(device)
        receipt['device'] = torch.cuda.get_device_name(device)
        for case in manifest['cases']:
            path = inputs/case['input_file']
            assert sha(path) == case['input_sha256']
            comparisons = []
            reference = {}
            for mode, execution_device, compiled in [('cpu_reference', 'cpu', False),
                    ('eager_cuda', device, False), ('compiled_cuda', device, True)]:
                _, original = prepare_model(path, execution_device, compiled)
                if case['case_id'] == 'mechanism-02':
                    receipt.setdefault('explicit_alias_directions', {})[mode] = alias_diagnostic(original, outdir, mode)
                original_pilot = pilot(original)
                zero = auxiliary_model(original, torch.zeros_like(original.log_prior), 0.)
                repeated = pilot(zero)
                assert zero is original and torch.equal(repeated.phi, original_pilot.phi)
                noise = keyed_directions(case['case_id'], original.mutation_ids, original.slope.shape[1])
                for index in range(8):
                    direction = torch.tensor(noise[index], device=execution_device, dtype=torch.float64)
                    model = auxiliary_model(original, direction, .03)
                    for point in (model.lower, model.upper, (model.lower+model.upper)/2):
                        expected, posterior, gradient = independent(model, point.cpu().numpy())
                        actual = model.terms(point)
                        np.testing.assert_allclose(actual[0].cpu(), expected, atol=1e-10, rtol=1e-12)
                        np.testing.assert_allclose(actual[3].cpu(), posterior, atol=2e-12, rtol=1e-12)
                        np.testing.assert_allclose(actual[1].cpu(), gradient, atol=2e-7, rtol=2e-10)
                    fitted = pilot(model)
                    QualifiedPilot(model, fitted)
                    labels = torch.arange(model.n, device=model.device)
                    refitted = refit_labels(model, labels)
                    graph = build_graph(original_pilot.phi, model.mutation_ids)
                    extra = solve_extra_start(model, graph, model.alt.new_tensor(.5), fitted.phi)
                    extra.validate(model, graph, model.alt.new_tensor(.5), extra.policy)
                    state = dict(phi=fitted.phi.cpu().numpy(), loss=fitted.loss.cpu().numpy(),
                                 refit_phi=refitted.phi.cpu().numpy(), score=float(refitted.score),
                                 raw_objective=float(extra.raw.objective), raw_qualified=extra.raw.qualified)
                    if mode == 'cpu_reference':
                        reference[index] = state
                    else:
                        other = reference[index]
                        np.testing.assert_allclose(state['loss'], other['loss'], atol=1e-7, rtol=1e-10)
                        # Tied wells can have different coordinates but equivalent objective.
                        assert abs(state['score']-other['score']) <= 1e-6+1e-9*abs(other['score'])
                    artifact = array_record(outdir/f"{case['case_id']}-{mode}-{index}.npz",
                        phi=fitted.phi, loss=fitted.loss, gap=fitted.gap,
                        posterior=model.terms(fitted.phi)[3], refit_phi=refitted.phi,
                        raw_phi=extra.raw.x, log_prior=model.log_prior, direction=direction)
                    comparisons.append(dict(mode=mode, direction=index, raw_qualified=extra.raw.qualified,
                        raw_objective=float(extra.raw.objective), scalar_qualified=True,
                        score=float(refitted.score), arrays=artifact,
                        coordinate_tie_difference=mode != 'cpu_reference' and
                            bool(np.max(np.abs(state['phi']-reference[index]['phi'])) > 1e-6)))
                del original, model, extra, graph
            receipt['cases'].append(dict(case_id=case['case_id'], comparisons=comparisons))
            print('Component qualification:', case['case_id'], flush=True)
        # Complete unmodified paths, baseline preservation, final publication and
        # all candidate qualifications; this is engineering, not accuracy evidence.
        full = []
        for case in manifest['cases']:
            record = run_case(inputs/case['input_file'], outdir/case['case_id'],
                              tumor_id=case['case_id'], device=device)
            assert record['original_raw_preserved']
            assert record['result_scores']['P'] <= record['result_scores']['B']
            full.append(dict(case_id=case['case_id'], result_sha256=sha(outdir/case['case_id']/'RESULT.json')))
            print('Full-path qualification:', case['case_id'], flush=True)
        receipt.update(status='passed', full_path=full, completed_utc=now())
        write_json(outdir/'QUALIFICATION.json', receipt)
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}', completed_utc=now())
        write_json(outdir/'FAILURE.json', receipt)
        raise
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    qualify(args.inputs, args.outdir, args.device)


if __name__ == '__main__':
    main()
