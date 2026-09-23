"""CPU arithmetic checks for the explicit experimental surrogate proposal only."""
import torch

from clipp1d.cuda.solver import coordinate_inflation


def t(x):
    return torch.tensor(x,dtype=torch.float64)


def test_local_backtracking_does_not_inflate_unrelated_stiff_coordinate():
    result=coordinate_inflation(t([1.,4.]),t([100.,200.]),t([-10.,-20.]),t([1.,5.]),t([95.,180.]))
    assert result.tolist()==[2.,4.]


def test_unexplained_rejection_keeps_full_backtrack():
    result=coordinate_inflation(t([1.,4.]),t([100.,200.]),t([-10.,-20.]),t([1.,5.]),t([80.,180.]))
    assert result.tolist()==[2.,8.]


def test_cancellation_margin_does_not_spuriously_select_one_row():
    result=coordinate_inflation(t([1.,4.]),t([1e15,1.]),t([-1e15,0.]),t([0.,0.]),t([1.,2.]))
    assert result.tolist()==[1.,8.]


def test_nonfinite_trial_row_is_not_ignored():
    result=coordinate_inflation(t([1.,4.]),t([1.,1.]),t([0.,0.]),t([0.,0.]),t([float('nan'),0.]))
    assert result.tolist()==[2.,4.]


def test_experimental_backtrack_changes_only_offending_curvature(monkeypatch):
    from types import SimpleNamespace
    from clipp1d.cuda import solver
    from clipp1d.cuda.qp import QuadraticFit
    from clipp1d.cuda.policy import CudaPolicy

    start=t([.5,.5])
    caps=t([[0.,.1],[.1,0.]])
    graph=SimpleNamespace(weights=caps,identity=object(),validate=lambda:None,
                          validate_metadata=lambda:None)
    model=SimpleNamespace(lower=t([0.,0.]),upper=t([1.,1.]),kernels=None,
        loss=lambda x: t([0.,0.]) if torch.equal(x,start) else t([1.,0.]),
        terms=lambda x:(t([0.,0.]),t([-1.,-1.]),t([1.,1e13]),None,t([-1.,-1.]),t([-1.,-1.])))
    monkeypatch.setattr(solver,'bound_majorization_curvature',lambda model,x,curv,posterior:curv)
    histories=[]
    for mode in ('scalar_backtracking_v1','coordinate_backtracking_v1'):
        seen=[]
        def qp(h,target,lower,upper,caps,kernels,policy,**kwargs):
            seen.append(h.clone())
            return QuadraticFit(t([.6,.5]),torch.zeros_like(caps),t(0.),t(1.),
                                t(0. if len(seen)==1 else 1.),len(seen)==1,1)
        monkeypatch.setattr(solver,'solve_qp',qp)
        result=solver._solve_start(model,graph,t(1.),start,CudaPolicy(),surrogate_policy=mode)
        assert result.diagnostics['status']=='surrogate_unresolved'
        assert result.diagnostics['outer_surrogate_policy']==mode
        assert result.diagnostics['inner_gap_pass'] and not result.diagnostics['inner_kkt_pass']
        histories.append(seen)
    assert histories[0][0].tolist()==histories[1][0].tolist()==[1.,1e13]
    assert histories[0][1].tolist()==[2.,2e13]
    assert histories[1][1].tolist()==[2.,1e13]
