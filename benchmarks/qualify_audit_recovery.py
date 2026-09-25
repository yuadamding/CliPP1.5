"""Allocated-device fault injection for fail-closed directional-audit recovery."""

import argparse
import json
from pathlib import Path

import torch

import clipp1d.cuda.audit as audit
from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.model import TensorModel


def qualify(device="cuda:0"):
    d = torch.device(device)
    alt = torch.tensor([20.0, 20.0], dtype=torch.float64, device=d)
    slope = torch.full((2, 1), 0.5, dtype=torch.float64, device=d)
    model = TensorModel(
        ("a", "b"),
        alt,
        100 - alt,
        slope,
        torch.zeros_like(slope),
        torch.full_like(alt, 1e-6),
        torch.ones_like(alt),
        1e-6,
        Kernels(d),
    )
    x = torch.full_like(alt, 0.4)
    caps = torch.tensor([[0.0, 10.0], [10.0, 0.0]], dtype=torch.float64, device=d)
    q = torch.zeros_like(caps)
    valid = audit.audit_raw(model, x, q, caps)
    assert valid.qualified and valid.signed_direction_count == 2
    original = audit.directional_cut
    observations = []
    for kind, flag in [
        ("linear", "linear_finite"),
        ("caps", "capacities_symmetric"),
        ("dual", "dual_finite"),
        ("mask", "mask_binary"),
    ]:

        def invalid(a, c, allowed, dual, *args, **kwargs):
            a, c, allowed, dual = [v.clone() for v in (a, c, allowed, dual)]
            if kind == "linear":
                a[0] = float("nan")
            elif kind == "caps":
                c[0, 1] = torch.nextafter(c[0, 1], c.new_tensor(float("inf")))
            elif kind == "dual":
                dual[0, 1] = float("inf")
            else:
                allowed[0] = 0.5
            return original(a, c, allowed, dual, *args, **kwargs)

        try:
            audit.directional_cut = invalid
            result = audit.audit_raw(model, x, q, caps)
        finally:
            audit.directional_cut = original
        assert not result.qualified and result.direction is None
        assert result.status == "positive_invalid_cut"
        assert result.diagnostics["positive"]["validation_checks"][flag] is False
        observations.append(
            dict(injected=kind, status=result.status, diagnostics=result.diagnostics)
        )
    assert audit.audit_raw(model, x, q, caps).qualified
    return dict(
        passed=True,
        device=str(d),
        compiled_likelihood=model.kernels.compiled,
        dtype="float64",
        scope="Injected invalid cut inputs are rejected; valid stationary fixture remains certified",
        checks=observations,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    record = qualify(args.device)
    with args.out.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
