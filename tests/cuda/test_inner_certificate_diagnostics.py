"""Individual QP gate reports cannot inherit combined admission or raw auditing."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda.solver import inner_certificate_diagnostics


def fitted(gap, scale, kkt, qualified=False):
    return SimpleNamespace(gap=torch.tensor(gap,dtype=torch.float64),
                           scale=torch.tensor(scale,dtype=torch.float64),
                           kkt=torch.tensor(kkt,dtype=torch.float64),qualified=qualified)


def test_below256_gate_failure_is_kkt_only():
    p=CudaPolicy()
    value=fitted(1.5445872e-10,2298.534098106846,2.4776856e-7)
    d=inner_certificate_diagnostics(value,p)
    assert d['inner_gap_pass'] and d['inner_gap_qualified']
    assert not d['inner_kkt_pass'] and not d['inner_kkt_qualified']
    assert not d['inner_qp_qualified']
    assert d['inner_gap']==float(value.gap)
    assert d['inner_gap_allowed']==float(p.inner_atol+p.inner_rtol*value.scale)
    assert d['inner_kkt_allowed']==p.inner_kkt_tol


def test_separate_gate_passes_do_not_override_combined_invalidity():
    d=inner_certificate_diagnostics(fitted(0.,1.,0.,qualified=False),CudaPolicy())
    assert d['inner_gap_pass'] and d['inner_kkt_pass'] and not d['inner_qp_qualified']


@pytest.mark.parametrize('gap,scale,kkt,gap_pass,kkt_pass',[
    (1.,1.,0.,False,True),
    (float('nan'),1.,0.,False,True),
    (0.,float('inf'),0.,False,True),
    (0.,1.,float('nan'),True,False),
    (0.,1.,float('inf'),True,False),
])
def test_nonfinite_values_are_independently_rejected(gap,scale,kkt,gap_pass,kkt_pass):
    d=inner_certificate_diagnostics(fitted(gap,scale,kkt),CudaPolicy())
    assert d['inner_gap_pass'] is gap_pass and d['inner_kkt_pass'] is kkt_pass


def test_actual_policy_and_exact_threshold_boundary_used():
    p=replace(CudaPolicy(),inner_atol=1e-9,inner_rtol=1e-8,inner_kkt_tol=2e-7)
    d=inner_certificate_diagnostics(fitted(p.inner_atol+p.inner_rtol*2.,2.,p.inner_kkt_tol,True),p)
    assert d['inner_gap_pass'] and d['inner_kkt_pass'] and d['inner_qp_qualified']


def test_missing_qp_does_not_invent_certificate():
    d=inner_certificate_diagnostics(None,CudaPolicy())
    assert not any(d[k] for k in ('inner_gap_pass','inner_kkt_pass','inner_qp_qualified','inner_certificate_present'))
    assert all(d[k] is None for k in ('inner_gap','inner_gap_scale','inner_gap_allowed','inner_kkt_residual'))


@pytest.mark.parametrize('gap,scale,kkt', [(-1.,1.,0.),(0.,-1.,0.),(0.,1.,-1.)])
def test_negative_certificate_measurements_are_not_passes(gap,scale,kkt):
    d=inner_certificate_diagnostics(fitted(gap,scale,kkt),CudaPolicy())
    assert not (d['inner_gap_pass'] and d['inner_kkt_pass'])
