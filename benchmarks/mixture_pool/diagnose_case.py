"""Capture an EM guard failure without changing the estimator or accepting it."""
import json
import math
from pathlib import Path
import sys
import traceback

import torch

from common import read, sha, write


def main(root, key, out):
    sys.path.insert(0, str(root/'payload/source/benchmarks'))
    import run_mixture_experiment as runner
    import clipp1d.cuda.mixture as mixture
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    out.mkdir(parents=True)
    manifest_path = root/'payload/manifests'/(key+'.json')
    manifest = read(manifest_path)
    model, seeds = runner.prepare_model(manifest['cases'][0], 'cuda:0')
    policy = mixture.MixturePolicy(**manifest['policy'])

    def convert(value):
        if isinstance(value, torch.Tensor):
            return convert(value.detach().cpu().tolist())
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value

    passed = []
    for repeat in range(3):
        try:
            fit = mixture.fit_mixture(model, policy, seed_centers=seeds)
            passed.append(dict(repeat=repeat, score=fit.score, status=fit.status))
        except ArithmeticError as exc:
            tb, frame = exc.__traceback__, None
            while tb is not None:
                if tb.tb_frame.f_code.co_name == '_fit':
                    frame = tb.tb_frame
                tb = tb.tb_next
            if frame is None:
                raise
            state = frame.f_locals
            names = ['origin', 'adaptive', 'k', 'iteration', 'centers', 'next_centers',
                'weights', 'next_weights', 'enrichment', 'next_enrichment', 'objective',
                'next_objective', 'margin', 'll', 'next_ll', 'posterior', 'next_posterior',
                'prior', 'valid', 'lower', 'upper']
            snapshot = {name: convert(state[name]) for name in names}
            snapshot.update(alt=convert(model.alt), ref=convert(model.ref), slope=convert(model.slope),
                eps=model.eps, error=repr(exc), traceback=traceback.format_exc())
            write(out/'STATE.json', snapshot)
            args = model.alt, model.ref, model.slope, state['valid']
            ll, posterior, _ = mixture.expectation(*args, state['centers'], state['weights'],
                                                   state['enrichment'], model.eps)
            next_ll, _, _ = mixture.expectation(*args, state['next_centers'], state['next_weights'],
                                                state['next_enrichment'], model.eps)
            observed = state['posterior']
            report = dict(status='reproduced', key=key, repeat=repeat,
                case_id=manifest['cases'][0]['case_id'], manifest_sha256=sha(manifest_path),
                origin=state['origin'], components=state['k'], adaptive=state['adaptive'],
                iteration=state['iteration'], objective=convert(state['objective']),
                next_objective=convert(state['next_objective']), margin=convert(state['margin']),
                compiled_previous_ll=convert(state['ll']), eager_previous_ll=convert(ll),
                compiled_next_ll=convert(state['next_ll']), eager_next_ll=convert(next_ll),
                posterior_max_delta=convert((observed-posterior).abs().max()),
                posterior_max_normalization_error=convert((observed.sum((1, 2))-1).abs().max()),
                state_sha256=sha(out/'STATE.json'), successful_repetitions=passed)
            write(out/'DIAGNOSIS.json', report)
            print(json.dumps(report), flush=True)
            return
    write(out/'DIAGNOSIS.json', dict(status='not_reproduced_in_bounded_replay', key=key,
        manifest_sha256=sha(manifest_path), successful_repetitions=passed))


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
