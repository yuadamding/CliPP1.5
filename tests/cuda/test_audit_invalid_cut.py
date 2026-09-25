"""Fail-closed raw-audit recovery; malformed caller inputs still raise."""

from dataclasses import replace

import pytest
import torch

import clipp1d.cuda.audit as audit
from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.policy import CudaPolicy


def model(device="cpu"):
    alt = torch.tensor([20.0, 20.0], dtype=torch.float64, device=device)
    slope = torch.full((2, 1), 0.5, dtype=torch.float64, device=device)
    return TensorModel(
        ("a", "b"),
        alt,
        100 - alt,
        slope,
        torch.zeros_like(slope),
        torch.full_like(alt, 1e-6),
        torch.ones_like(alt),
        1e-6,
        Kernels(device),
    )


@pytest.mark.parametrize(
    "corruption,flag",
    [
        ("linear", "linear_finite"),
        ("caps", "capacities_symmetric"),
        ("dual", "dual_finite"),
        ("mask", "mask_binary"),
    ],
)
def test_internal_bad_cut_never_qualifies_or_escapes_raw_audit(monkeypatch, corruption, flag):
    m = model()
    x = torch.full_like(m.alt, 0.4)
    caps = torch.tensor([[0.0, 10.0], [10.0, 0.0]], dtype=x.dtype)
    original = audit.directional_cut

    def bad_cut(a, c, allowed, q, *args, **kwargs):
        a, c, allowed, q = [v.clone() for v in (a, c, allowed, q)]
        if corruption == "linear":
            a[0] = float("nan")
        if corruption == "caps":
            c[0, 1] = torch.nextafter(c[0, 1], c.new_tensor(float("inf")))
        if corruption == "dual":
            q[0, 1] = float("inf")
        if corruption == "mask":
            allowed[0] = 0.5
        return original(a, c, allowed, q, *args, **kwargs)

    monkeypatch.setattr(audit, "directional_cut", bad_cut)
    result = audit.audit_raw(m, x, torch.zeros_like(caps), caps)
    assert not result.qualified and result.direction is None
    assert result.status == "positive_invalid_cut"
    assert result.diagnostics["positive"]["validation_checks"][flag] is False


def test_invalid_caller_graph_is_not_treated_as_a_valid_model():
    m = model()
    caps = torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.float64)
    with pytest.raises(audit.InvalidDirectionalCut) as error:
        audit.audit_raw(m, torch.full_like(m.alt, 0.4), torch.zeros_like(caps), caps)
    assert error.value.checks["capacities_symmetric"] is False


def test_nonfinite_derivatives_cannot_produce_a_certificate(monkeypatch):
    m = model()
    original = m.terms

    def invalid_terms(x):
        values = list(original(x))
        values[-1] = torch.full_like(x, float("nan"))
        return tuple(values)

    monkeypatch.setattr(TensorModel, "terms", lambda self, x: invalid_terms(x))
    caps = torch.zeros((2, 2), dtype=torch.float64)
    result = audit.audit_raw(m, torch.full_like(m.alt, 0.4), caps, caps)
    assert not result.qualified and result.direction is None
    assert result.status == "nonfinite_likelihood_derivatives"


def test_overflowed_cut_normalization_remains_unqualified():
    m = model()
    caps = torch.full((2, 2), torch.finfo(torch.float64).max, dtype=torch.float64)
    caps.fill_diagonal_(0)
    result = audit.audit_raw(m, torch.full_like(m.alt, 0.4), torch.zeros_like(caps), caps)
    assert not result.qualified and result.direction is None
    assert result.status == "positive_invalid_cut"
    assert result.diagnostics["positive"]["status"] == "nonfinite_cut_scale"


def test_valid_stationary_cut_keeps_original_certificate():
    m = model()
    caps = torch.tensor([[0.0, 10.0], [10.0, 0.0]], dtype=torch.float64)
    result = audit.audit_raw(
        m,
        torch.full_like(m.alt, 0.4),
        torch.zeros_like(caps),
        caps,
        policy=replace(CudaPolicy(), inner_max_iterations=1),
    )
    assert result.qualified and result.signed_direction_count == 2


def test_invalid_audit_ends_only_the_affected_start(monkeypatch):
    from clipp1d.cuda import solver
    from clipp1d.cuda.graph import build_graph

    m = model()
    x = torch.full_like(m.alt, 0.4)
    graph = build_graph(x, m.mutation_ids)
    calls = []

    def invalid(*args, **kwargs):
        calls.append(1)
        return audit.Audit(
            False,
            float("inf"),
            None,
            "positive_invalid_cut",
            1,
            dict(positive=dict(status="invalid_cut_inputs")),
        )

    monkeypatch.setattr(solver, "audit_raw", invalid)
    result = solver.solve_start(m, graph, x.new_tensor(1.0), x, CudaPolicy())
    assert not result.qualified and len(calls) == 1
    assert result.diagnostics["status"] == "raw_audit_unresolved"
    assert result.diagnostics["audit_diagnostics"]["positive"]["status"] == "invalid_cut_inputs"
    assert result.diagnostics["outer_iterations"] == 1
