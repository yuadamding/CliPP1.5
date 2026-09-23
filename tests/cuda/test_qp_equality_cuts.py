"""Tiny original-QP references for conservative numerical equality splitting."""
import pytest
import torch

from clipp1d.cuda import qp
from clipp1d.cuda.kernels import Kernels, adjoint, differences
from clipp1d.cuda.partition import polish_quadratic
from clipp1d.cuda.policy import CudaPolicy


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def narrow_pair_problem(extra_group=False):
    x = tensor([.5 - 2e-8, .5 + 2e-8, .3, .3, .3, 0.] if extra_group
               else [.5 - 2e-8, .5 + 2e-8, .3])
    h = torch.full_like(x, 10.)
    h[:2] = 1e8
    caps = torch.full((x.numel(), x.numel()), .005, dtype=x.dtype)
    caps.fill_diagonal_(0.)
    caps[0, 1] = caps[1, 0] = .01
    q = caps * differences(x).sign()
    target = x + adjoint(q) / h
    lower, upper = torch.zeros_like(x), torch.ones_like(x)
    if extra_group:
        target[-1] -= .1  # The bound singleton can carry a positive normal.
    return x, q, h, target, lower, upper, caps


@pytest.mark.parametrize('divisor', [1., 16., 256.])
@pytest.mark.parametrize('extra_group', [False, True])
def test_capacity_obstruction_splits_narrow_pair_without_changing_other_groups(divisor, extra_group):
    x, q, h, target, lower, upper, caps = narrow_pair_problem(extra_group)
    kernels, policy = Kernels('cpu'), CudaPolicy()
    tolerance = policy.fusion_tol / divisor
    merged = polish_quadratic(x, q, h, target, lower, upper, caps, tolerance)
    assert merged[0] == merged[1] and x[0] != x[1]
    objective = qp._objective_context(x, h, target, lower, upper, caps)
    merged_value = qp.quadratic_value(merged, h, objective[0], caps, objective[1])
    roundoff = 32 * torch.finfo(x.dtype).eps * (1 + objective[2].abs() + merged_value.abs())
    assert merged_value > objective[2] + roundoff
    # The required internal divergence exceeds the only available pair edge.
    same = merged[:, None] == merged[None, :]
    external = adjoint(torch.where(same, 0., caps * differences(merged).sign()))
    assert bool((h * (merged-target) + external)[:2].abs().min() > caps[0, 1])
    proposal = qp._prepare_equality_proposal(
        x, torch.zeros_like(q), h, target, lower, upper, caps, kernels, policy,
        tolerance, objective)
    assert proposal is not None and proposal.qualified and proposal.flow_steps <= 1
    torch.testing.assert_close(proposal.candidate, x, atol=1e-15, rtol=0.)
    assert proposal.candidate[0] != proposal.candidate[1]
    if extra_group:
        assert bool((proposal.candidate[2:5] == proposal.candidate[2]).all())
        assert proposal.candidate[-1] == lower[-1]
    stats = kernels.gap_kkt(proposal.candidate, proposal.dual, h, target, lower, upper, caps)
    assert float(stats[0]) <= policy.inner_atol + policy.inner_rtol * float(stats[1])
    assert float(stats[2]) <= policy.inner_kkt_tol


@pytest.mark.parametrize('reflected', [False, True])
def test_bound_normal_signs_and_fixed_nodes_do_not_produce_false_splits(reflected):
    x = tensor([.5] * 5)
    h = torch.ones_like(x)
    lower, upper = tensor([.5, 0., .5, 0., 0.]), tensor([1., .5, .5, 1., 1.])
    target = tensor([-9.5, 10.5, 1e300, .5, .5])
    caps = torch.full((5, 5), .125, dtype=x.dtype)
    caps.fill_diagonal_(0.)
    if reflected:
        x, target, lower, upper = 1-x, 1-target, 1-upper, 1-lower
    assert qp._split_infeasible_equalities(x, x, h, target, lower, upper, caps) is None
    # Reverse one active node's residual: its allowed normal cannot absorb it.
    target[0] = 1-target[0]
    replacement = qp._split_infeasible_equalities(x, x, h, target, lower, upper, caps)
    assert replacement is not None and replacement[0] != x[0]
    torch.testing.assert_close(replacement[1:], x[1:], atol=0, rtol=0)


def test_roundoff_sized_capacity_excess_does_not_split():
    x, h = tensor([.5, .5]), tensor([1., 1.])
    target = tensor([.5-.125-1e-16, .5+.125+1e-16])
    caps = tensor([[0., .125], [.125, 0.]])
    assert qp._split_infeasible_equalities(
        x, x, h, target, torch.zeros_like(x), torch.ones_like(x), caps) is None


def test_qualified_incoming_dual_bypasses_split_proposals(monkeypatch):
    x, q, h, target, lower, upper, caps = narrow_pair_problem()
    # Preserve a literal qualified equality candidate and its exact original dual.
    monkeypatch.setattr(qp, 'polish_quadratic', lambda *args: x)

    def forbidden(*args):
        raise AssertionError('A qualified incoming certificate must be preserved')

    monkeypatch.setattr(qp, '_split_infeasible_equalities', forbidden)
    kernels, policy = Kernels('cpu'), CudaPolicy()
    result = qp._prepare_equality_proposal(
        x, q, h, target, lower, upper, caps, kernels, policy, policy.fusion_tol,
        qp._objective_context(x, h, target, lower, upper, caps))
    assert result.qualified and result.dual is q and result.flow_steps == 0


def test_replacement_cannot_bypass_original_objective_gate(monkeypatch):
    x, q, h, target, lower, upper, caps = narrow_pair_problem()
    monkeypatch.setattr(qp, '_split_infeasible_equalities', lambda *args: torch.ones_like(x))
    kernels, policy = Kernels('cpu'), CudaPolicy()
    result = qp._prepare_equality_proposal(
        x, q, h, target, lower, upper, caps, kernels, policy, policy.fusion_tol,
        qp._objective_context(x, h, target, lower, upper, caps))
    assert result is None


def test_replacement_is_still_independently_certified(monkeypatch):
    x, q, h, target, lower, upper, caps = narrow_pair_problem()
    kernels, policy = Kernels('cpu'), CudaPolicy()
    calls = []

    def reject(*args):
        calls.append(args[0].clone())
        return tensor([1., 1., 1.])

    kernels.gap_kkt = reject
    result = qp._prepare_equality_proposal(
        x, q, h, target, lower, upper, caps, kernels, policy, policy.fusion_tol,
        qp._objective_context(x, h, target, lower, upper, caps))
    assert result is not None and not result.qualified and calls
    torch.testing.assert_close(calls[0], x, atol=1e-15, rtol=0)
