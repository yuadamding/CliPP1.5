"""Objective-scale covariance of the actual ADMM balance controller."""
from types import SimpleNamespace

import pytest
import torch

from clipp1d.cuda import qp
from clipp1d.cuda.kernels import Kernels, differences
from clipp1d.cuda.policy import CudaPolicy


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


@pytest.mark.parametrize('regime', ['primal', 'dual'])
def test_actual_controller_decisions_are_covariant_under_objective_scaling(monkeypatch, regime):
    # A deterministic injected ADMM trajectory isolates the real balance code.
    # Forced unresolved certificates prevent early exits; no numerical fit is
    # qualified by this bookkeeping test.
    monkeypatch.setattr(qp, '_prepare_equality_proposal', lambda *args: None)
    trajectories = []
    for scale in (2.**-30, 1., 2.**30):
        h, caps = tensor([3., 4.]) * scale, tensor([[0., .2], [.2, 0.]]) * scale
        target, lower, upper = tensor([.25, .75]), tensor([0., 0.]), tensor([1., 1.])
        observed = []

        def step(h, weighted_target, lower, upper, caps, z, v, rho):
            iteration = len(observed) + 1
            observed.append(float(rho / h.median()))
            if regime == 'primal':
                x = target
                z = differences(x) * (.2 + iteration * .0001)
            else:
                x = tensor([.25, .75 + iteration * .0001])
                z = differences(x) * (1 - 1e-8)
            return x, z, torch.zeros_like(v)

        kernels = SimpleNamespace(admm_step=step,
                                  gap_kkt=lambda *args: tensor([float('inf'), 0., float('inf')]))
        fit = qp.solve_qp(h, target, lower, upper, caps, kernels,
                          CudaPolicy(inner_max_iterations=129), start=target)
        assert not fit.qualified and fit.iterations == 129
        trajectories.append(observed)
    assert trajectories[0] == trajectories[1] == trajectories[2]
    ratio = 2. if regime == 'primal' else .5
    assert trajectories[0] == [1.] * 64 + [ratio] * 64 + [ratio * ratio]


@pytest.mark.parametrize('affected,expected_ratio', [('high_nodes', 2.), ('low_node', .5)])
def test_heterogeneous_curvature_uses_each_nodes_ccf_units_and_preserves_covariance(monkeypatch, affected, expected_ratio):
    # Oscillate one edge with a constant primal residual. Its unscaled divergence
    # has two equal/opposite entries. A scalar median cannot distinguish whether
    # those forces act on stiff or compliant coordinates; their CCF effects can.
    # Force unresolved certificates so this tests the real conditioning decision,
    # never qualification of the injected trajectory.
    monkeypatch.setattr(qp, '_prepare_equality_proposal', lambda *args: None)
    trajectories = []
    for objective_scale in (2.**-30, 1., 2.**30):
        h = tensor([1., 2., 1e12, 1e12] if affected == 'high_nodes'
                   else [1., 1e12, 1e12, 1e12]) * objective_scale
        caps = torch.full((4, 4), .1*objective_scale, dtype=h.dtype)
        caps.fill_diagonal_(0.)
        target, lower, upper = tensor([.5]*4), tensor([0.]*4), tensor([1.]*4)
        edge = (2, 3) if affected == 'high_nodes' else (0, 3)
        observed = []

        def step(h, weighted_target, lower, upper, caps, z, v, rho):
            observed.append(float(rho/h.median()))
            next_z = torch.zeros_like(z)
            amount = 1e-4 if len(observed) % 2 else -1e-4
            next_z[edge[0], edge[1]], next_z[edge[1], edge[0]] = amount, -amount
            return target, next_z, torch.zeros_like(v)

        kernels = SimpleNamespace(admm_step=step,
                                  gap_kkt=lambda *args: tensor([float('inf'), 0., float('inf')]))
        fit = qp.solve_qp(h, target, lower, upper, caps, kernels,
                          CudaPolicy(inner_max_iterations=65), start=target)
        assert not fit.qualified and fit.iterations == 65
        trajectories.append(observed)
    assert trajectories[0] == trajectories[1] == trajectories[2]
    assert trajectories[0] == [1.]*64 + [expected_ratio]


@pytest.mark.parametrize('scale', [1e-6, 1., 1e6])
@pytest.mark.parametrize('case', ['separate', 'fused', 'upper_bound'])
def test_scaled_tiny_qps_keep_analytic_solution_and_original_certificates(scale, case):
    h, target = tensor([2., 3.]) * scale, tensor([.2, .8])
    lower, upper = tensor([0., 0.]), tensor([1., 1.])
    cap = 1. if case == 'fused' else .1
    caps = tensor([[0., cap], [cap, 0.]]) * scale
    expected = tensor([.56, .56]) if case == 'fused' else tensor([.25, .8 - .1 / 3])
    if case == 'upper_bound':
        upper[0], expected[0] = .24, .24
    kernels, policy = Kernels('cpu'), CudaPolicy()
    fit = qp.solve_qp(h, target, lower, upper, caps, kernels, policy)
    assert fit.qualified and fit.iterations <= policy.inner_max_iterations
    stats = kernels.gap_kkt(fit.x, fit.dual, h, target, lower, upper, caps)
    assert bool(torch.isfinite(stats).all())
    assert float(stats[0]) <= policy.inner_atol + policy.inner_rtol * float(stats[1])
    assert float(stats[2]) <= policy.inner_kkt_tol
    torch.testing.assert_close(fit.x, expected, atol=2e-8, rtol=0)
