"""Varying-input CUDA likelihood parity for local compiler correctness guards."""
import pytest
import torch

from clipp1d.cuda import kernels


def test_compilation_options_are_owned_per_bank_without_global_changes():
    options = {'inplace_buffers': False, 'max_fusion_size': 1}
    bank = kernels.StructuralCompileBank(kernels.loss_gradient, options=options)
    options['max_fusion_size'] = 9
    assert bank.options == {'inplace_buffers': False, 'max_fusion_size': 1}
    diagnostics = bank.diagnostics()
    diagnostics['options']['max_fusion_size'] = 8
    assert bank.options['max_fusion_size'] == 1


def test_varying_compiled_likelihood_and_gradient_match_eager_cuda():
    if not torch.cuda.is_available():
        pytest.skip('Allocated CUDA required; no controller numerical execution')
    bank = kernels.Kernels('cuda:0', compiled=True)
    assert bank.likelihood.options == bank.loss_gradient.options == {
        'inplace_buffers': False, 'max_fusion_size': 1}
    assert bank.loss_only.options == {}
    with torch.no_grad():
        for n in (7, 19, 7):
            alt = torch.arange(1, n+1, dtype=torch.float64, device='cuda:0')
            ref = 50-alt
            slope = torch.tensor([.2, .4, .7], dtype=torch.float64, device='cuda:0').expand(n, 3).clone()
            prior = torch.tensor([.2, .3, .5], dtype=torch.float64, device='cuda:0').log().expand(n, 3).clone()
            for offset in (0., .07, .14):
                x = (torch.linspace(1e-6, 1., n, dtype=torch.float64, device='cuda:0') + offset).clamp_max(1.)
                args = (alt, ref, slope, prior, x[:, None], 1e-6)
                for name in ('likelihood', 'loss_gradient'):
                    actual = getattr(bank, name)(*args)
                    expected = getattr(kernels, name)(*args)
                    for measured, reference in zip(actual, expected, strict=True):
                        torch.testing.assert_close(measured, reference, rtol=1e-11, atol=1e-10)
