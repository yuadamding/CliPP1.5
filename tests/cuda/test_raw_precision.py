"""Allocated-CUDA terminal recovery: independent raw gates and bounded work."""
from dataclasses import asdict

import pytest


@pytest.fixture(scope="module")
def env():
    import torch
    from clipp1d.cuda import raw_precision
    from clipp1d.cuda.audit import audit_raw
    from clipp1d.cuda.kernels import Kernels, differences
    from clipp1d.cuda.model import TensorModel
    from clipp1d.cuda.policy import CudaPolicy
    if not torch.cuda.is_available():
        pytest.skip("Allocated CUDA required; a skipped test is not numerical qualification")
    kernels = Kernels("cuda:0", compiled=True)

    def make(centers=(.400001,), alts=(40.,), *, fused_capacity=0., external_capacity=1000.):
        x = torch.tensor([.1, *centers, .9], dtype=torch.float64, device="cuda:0")
        alt = x.new_tensor([20., *alts, 80.])
        lower, upper = x.new_full(x.shape, .01), x.new_full(x.shape, .99)
        lower[0] = upper[0] = x[0]
        lower[-1] = upper[-1] = x[-1]
        model = TensorModel((), alt, 100 - alt, x.new_ones((x.numel(), 1)),
                            x.new_zeros((x.numel(), 1)), lower, upper, 1e-6, kernels)
        caps = x.new_zeros((x.numel(), x.numel()))
        caps[1:-1, 0] = caps[0, 1:-1] = external_capacity
        caps[1:-1, -1] = caps[-1, 1:-1] = external_capacity
        caps[1:-1, 1:-1] = fused_capacity
        caps.fill_diagonal_(0.)
        dual = caps * differences(x).sign()
        return model, x, dual, caps

    return torch, raw_precision, audit_raw, differences, CudaPolicy(), make


def energy(model, x, caps, differences):
    return model.loss(x).sum() + .5 * (caps * differences(x).abs()).sum()


def test_fixed_primal_flow_repairs_feasible_fused_dual(env):
    torch, precision, audit, _, policy, make = env
    model, x, dual, caps = make((.4, .4), (39., 41.), fused_capacity=10.)
    original = audit(model, x, dual, caps, policy=policy)
    assert original.status == "componentwise_unresolved"
    result = precision.recover_raw(model, x, dual, caps, policy, allow_qp_precision=False,
                                  limits=precision.RawPrecisionLimits(dual_steps=1))
    assert result.accepted and result.audit.qualified
    assert torch.equal(result.x, x)
    assert result.work["dual_steps"] == 1 and result.work["scalar_calls"] == 0
    assert torch.equal(result.dual, -result.dual.T)
    assert bool((result.dual.abs() <= caps).all())
    assert audit(model, result.x, result.dual, caps, policy=policy).qualified


@pytest.mark.parametrize("rows", [1, 2])
def test_affine_moves_singletons_and_exact_blocks_to_independent_binomial_root(env, rows):
    torch, precision, audit, differences, policy, make = env
    model, x, dual, caps = make((.400001,) * rows, (40.,) * rows, fused_capacity=10.)
    snapshots = tuple(v.clone() for v in (x, dual, caps))
    original = audit(model, x, dual, caps, policy=policy)
    assert original.status == "componentwise_unresolved"
    assert all(original.diagnostics[name]["status"] == "qualified_lower_bound"
               for name in ("positive", "negative"))
    value = energy(model, x, caps, differences)
    before = asdict(policy)
    result = precision.recover_raw(model, x, dual, caps, policy, allow_qp_precision=False,
                                  limits=precision.RawPrecisionLimits(dual_steps=1))
    assert result.accepted and result.audit.qualified and result.qp_work is None
    assert result.work["scalar_calls"] == 1 and result.work["affine_blocks"][0]["rows"] == rows
    assert bool((result.x[1:-1] - .4).abs().max() < 1e-7)
    assert torch.equal(differences(result.x).sign(), differences(x).sign())
    assert result.objective <= value + 32 * torch.finfo(x.dtype).eps * (1 + value.abs())
    assert audit(model, result.x, result.dual, caps, policy=policy).qualified
    assert asdict(policy) == before
    assert all(torch.equal(a, b) for a, b in zip((x, dual, caps), snapshots, strict=True))
    assert result.work["scalar_subdivisions"] <= policy.scalar_max_intervals
    assert result.work["model_work"]["general_groups"] == 1


@pytest.mark.parametrize("budget", [1, 2])
def test_multiple_blocks_require_full_final_audit_and_keep_original_negative_on_exhaustion(env, budget):
    torch, precision, audit, differences, policy, make = env
    model, x, dual, caps = make((.300001, .600001), (30., 60.))
    original = audit(model, x, dual, caps, policy=policy)
    result = precision.recover_raw(model, x, dual, caps, policy, allow_qp_precision=False,
                                  limits=precision.RawPrecisionLimits(dual_steps=1, scalar_calls=budget))
    assert result.work["scalar_calls"] == budget
    assert result.work["dual_steps"] == 1
    assert result.accepted == (budget == 2)
    assert result.audit.qualified == (budget == 2)
    assert result.work["audit_calls"] >= 3
    assert result.work["audit_seconds"] >= 0 and result.work["scalar_seconds"] >= 0
    if budget == 1:
        assert torch.equal(result.x, x) and torch.equal(result.dual, dual)
        assert result.audit.status == original.status and result.audit.residual == original.residual
        assert result.objective == energy(model, x, caps, differences)
        assert result.work["affine_blocks"][0]["accepted"]
    else:
        assert audit(model, result.x, result.dual, caps, policy=policy).qualified


def test_original_qualified_state_gets_no_auxiliary_optimization(env):
    torch, precision, _, _, policy, make = env
    model, x, dual, caps = make((.4,), (40.,))
    result = precision.recover_raw(model, x, dual, caps, policy)
    assert result.audit.qualified and not result.accepted and not result.work["attempted"]
    assert result.qp_work is None
    assert result.work["dual_steps"] == result.work["scalar_calls"] == 0
    assert torch.equal(result.x, x) and torch.equal(result.dual, dual)


def test_signed_descent_cannot_be_hidden_by_componentwise_repair(env):
    torch, precision, audit, _, policy, make = env
    model, x, dual, caps = make((.5,), (40.,), external_capacity=0.)
    original = audit(model, x, dual, caps, policy=policy)
    assert not original.qualified and original.status != "componentwise_unresolved"
    result = precision.recover_raw(model, x, dual, caps, policy, allow_qp_precision=False)
    assert not result.accepted and result.audit.status == original.status
    assert result.work["dual_steps"] == result.work["scalar_calls"] == 0
    assert torch.equal(result.x, x) and torch.equal(result.dual, dual)


def make_literal_context(env):
    torch, precision, _, _, _, make = env
    model, _, _, caps = make((.4001,), (40.,))
    start = model.alt.new_tensor([.1, .4001, .9])
    _, gradient, curvature, _, left, right = model.terms(start)
    gradient = torch.where(start == model.lower, right, gradient)
    gradient = torch.where(start == model.upper, left, gradient)
    h = 2 * curvature.clamp_min(1.)
    target = start - gradient / h
    endpoint = target.clamp(model.lower, model.upper)
    from clipp1d.cuda.kernels import differences
    dual = caps * differences(endpoint).sign()
    return model, endpoint, dual, caps, precision.AcceptedQPContext(h, target, start, endpoint)


def test_exact_accepted_surrogate_chain_counts_qp_once_and_still_requires_raw_audit(env):
    _, precision, audit, _, policy, _ = env
    model, x, dual, caps, context = make_literal_context(env)
    assert audit(model, x, dual, caps, policy=policy).status == "componentwise_unresolved"
    result = precision.recover_raw(model, x, dual, caps, policy, accepted_qp_context=context,
                                  limits=precision.RawPrecisionLimits(dual_steps=1))
    assert result.work["qp_context_status"] == "qualified_literal_context"
    assert result.work["context_eager_certificate_evaluations"] == 1
    assert result.work["context_compiled_certificate_evaluations"] == 1
    assert result.qp_work is not None and result.qp_work["attempted"]
    assert result.qp_work["candidate_accepted"] and result.work["qp_original_mm_pass"]
    assert result.work["scalar_calls"] == 1
    assert result.accepted and audit(model, result.x, result.dual, caps, policy=policy).qualified


def test_spent_qp_allowance_does_not_invoke_another_recovery(env, monkeypatch):
    _, precision, _, _, policy, _ = env
    model, x, dual, caps, context = make_literal_context(env)
    def forbidden(*args, **kwargs):
        raise AssertionError("Spent QP allowance must not call QP precision")
    monkeypatch.setattr(precision, "recover_quadratic", forbidden)
    result = precision.recover_raw(model, x, dual, caps, policy, accepted_qp_context=context,
                                  allow_qp_precision=False, limits=precision.RawPrecisionLimits(dual_steps=1))
    assert result.qp_work is None and result.work["context_eager_certificate_evaluations"] == 0


@pytest.mark.parametrize("change", ["endpoint", "target", "dual"])
def test_stale_or_uncertified_literal_surrogate_cannot_trigger_qp(env, monkeypatch, change):
    _, precision, _, _, policy, _ = env
    model, x, dual, caps, context = make_literal_context(env)
    h, target, start, endpoint = context.values()
    if change == "endpoint":
        endpoint = endpoint.clone()
        endpoint[1] += 1e-8
    elif change == "target":
        target = target + 1e-8
    else:
        dual = dual.clone()
        dual[0, 1] -= 1.
        dual[1, 0] += 1.
    context = precision.AcceptedQPContext(h, target, start, endpoint)
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid accepted context must not call QP precision")
    monkeypatch.setattr(precision, "recover_quadratic", forbidden)
    result = precision.recover_raw(model, x, dual, caps, policy, accepted_qp_context=context,
                                  limits=precision.RawPrecisionLimits(dual_steps=1))
    expected = dict(endpoint="accepted_endpoint_changed", target="literal_target_changed",
                    dual="accepted_surrogate_certificate_rejected")
    assert result.work["qp_context_status"] == expected[change] and result.qp_work is None


@pytest.mark.parametrize("change", ["version", "data"])
def test_accepted_context_rejects_mutation(env, change):
    _, precision, _, _, policy, _ = env
    model, x, dual, caps, context = make_literal_context(env)
    if change == "version":
        context.h[1] += 1.
    else:
        context.h.data[1] += 1.
    with pytest.raises(ValueError, match="context changed"):
        precision.recover_raw(model, x, dual, caps, policy, accepted_qp_context=context)


@pytest.mark.parametrize("proposal", ["same_negative", "higher_objective", "cross_neighbor"])
def test_scalar_flag_cannot_replace_original_objective_sign_and_raw_gates(env, monkeypatch, proposal):
    torch, precision, _, _, policy, make = env
    from clipp1d.cuda.scalar import ScalarBatch
    model, x, dual, caps = make()
    def misleading_scalar(problem, policy):
        phi = dict(same_negative=.400001, higher_objective=.49, cross_neighbor=.95)[proposal]
        value = x.new_tensor([phi])
        zero = value.new_zeros(1)
        return ScalarBatch(value, zero, zero, zero, value, torch.ones(1, device=x.device, dtype=torch.bool), 0)
    monkeypatch.setattr(precision, "solve_affine_scalar", misleading_scalar)
    result = precision.recover_raw(model, x, dual, caps, policy, allow_qp_precision=False,
                                  limits=precision.RawPrecisionLimits(dual_steps=1))
    assert not result.accepted and not result.audit.qualified
    assert torch.equal(result.x, x) and torch.equal(result.dual, dual)
    assert result.work["scalar_calls"] == 1
    if proposal == "higher_objective":
        assert not result.work["affine_blocks"][0]["objective_nonincrease"]
    if proposal == "cross_neighbor":
        assert not result.work["affine_blocks"][0]["original_edge_signs_preserved"]


@pytest.mark.parametrize("phase", ["original", "final"])
def test_nonfinite_audit_diagnostic_stays_negative_and_strict_json(env, monkeypatch, phase):
    import json
    from clipp1d.cuda.audit import Audit
    torch, precision, real_audit, _, policy, make = env
    model, x, dual, caps = make()
    calls = []
    def controlled_audit(*args, **kwargs):
        calls.append(None)
        if (phase == "original" and len(calls) == 1) or (phase == "final" and len(calls) == 3):
            return Audit(False, float("inf"), None, "positive_invalid_cut", 1,
                         {"positive": {"status": "nonfinite_cut_scale"}})
        return real_audit(*args, **kwargs)
    monkeypatch.setattr(precision, "audit_raw", controlled_audit)
    result = precision.recover_raw(model, x, dual, caps, policy, allow_qp_precision=False,
                                  limits=precision.RawPrecisionLimits(dual_steps=1))
    assert not result.accepted and not result.audit.qualified
    assert torch.equal(result.x, x) and torch.equal(result.dual, dual)
    assert result.work[phase + "_residual" if phase == "original" else "proposal_residual"] is None
    json.dumps(result.work, allow_nan=False)


@pytest.mark.parametrize("kwargs", [dict(dual_steps=513), dict(scalar_calls=9), dict(dual_steps=0),
                                   dict(scalar_calls=True)])
def test_auxiliary_limits_reject_budget_expansion(env, kwargs):
    with pytest.raises(ValueError):
        env[1].RawPrecisionLimits(**kwargs)
