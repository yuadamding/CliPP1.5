"""Repeated allocated-CUDA regression for the captured A100 posterior failure."""
from pathlib import Path
import json
import sys
import time

import torch

from clipp1d.cuda.kernels import StructuralCompileBank
from clipp1d.cuda.mixture import expectation

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
control = Path(__file__).resolve().parent
state_path = Path(sys.argv[2]) if len(sys.argv) > 2 else control/'A100_FAILURE_STATE.json'
state = json.loads(state_path.read_text())
out = Path(sys.argv[1])
def tensor(name):
    return torch.tensor(state[name], device='cuda', dtype=torch.float64)


alt, ref, slope = [tensor(k) for k in ('alt', 'ref', 'slope')]
valid = torch.tensor(state['valid'], device='cuda', dtype=torch.bool)
results = []
began = time.monotonic()
for k in dict.fromkeys((state['k'], 1, 2, 8)):
    centers = tensor('centers') if k == state['k'] else torch.linspace(.1, .95, k, device='cuda', dtype=torch.float64)
    weights = tensor('weights') if k == state['k'] else torch.full((k,), 1/k, device='cuda', dtype=torch.float64)
    enrichment = tensor('enrichment') if k == state['k'] else torch.zeros_like(weights)
    args = alt, ref, slope, valid, centers, weights, enrichment, state['eps']
    reference = expectation(*args)
    if k == state['k']:
        for repeat in range(2000):
            eager = expectation(*args)
            if not bool((eager[1]-reference[1]).abs().max() <= 1e-10):
                raise ArithmeticError('Independent eager GPU expression is unstable')
    compiled = StructuralCompileBank(expectation)
    maximum = 0.0
    for repeat in range(2000):
        ll, posterior, prior = compiled(*args)
        delta = (posterior-reference[1]).abs().max()
        norm = (posterior.sum((1, 2))-1).abs().max()
        if not bool(torch.isfinite(posterior).all() & (delta <= 1e-10) & (norm <= 1e-10) &
                    ((ll-reference[0]).abs() <= 1e-7)):
            raise ArithmeticError(f'Compiled posterior stress failed: K={k}, repeat={repeat}, delta={float(delta)}, normalization={float(norm)}')
        maximum = max(maximum, float(delta))
    results.append(dict(components=k, repetitions=2000, max_posterior_delta=maximum))
with (out/'STRESS.json').open('x') as stream:
    json.dump(dict(status='passed', cases=results, elapsed_seconds=time.monotonic()-began,
                   reference='independent eager CUDA expression', full_fit_qualification=False), stream, indent=2)
