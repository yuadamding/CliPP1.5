"""CPU arithmetic checks for the explicit experimental surrogate proposal only."""
import torch
import pytest
import json
import numpy as np

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


@pytest.mark.parametrize('instrument',[False,True])
def test_experimental_backtrack_changes_only_offending_curvature(monkeypatch,tmp_path,instrument):
    from types import SimpleNamespace
    from clipp1d.cuda import solver
    from clipp1d.cuda.qp import QuadraticFit
    from clipp1d.cuda.policy import CudaPolicy
    from benchmarks.qualify_mixed_cuda import Journal
    from benchmarks.surrogate_trace import SurrogateTrace

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
        journal=Journal(tmp_path/(mode+'.json')) if instrument else None
        trace=SurrogateTrace(journal,0,t(1.),CudaPolicy()) if instrument else None
        seen=[]
        def qp(h,target,lower,upper,caps,kernels,policy,**kwargs):
            seen.append(h.clone())
            return QuadraticFit(t([.6,.5]),torch.zeros_like(caps),t(0.),t(1.),
                                t(0. if len(seen)==1 else 1.),len(seen)==1,1)
        monkeypatch.setattr(solver,'solve_qp',qp)
        result=solver._solve_start(model,graph,t(1.),start,CudaPolicy(),surrogate_policy=mode,observer=trace)
        assert result.diagnostics['status']=='surrogate_unresolved'
        assert result.diagnostics['outer_surrogate_policy']==mode
        assert result.diagnostics['inner_gap_pass'] and not result.diagnostics['inner_kkt_pass']
        if trace:
            trace.finish()
            detail=json.loads((journal.root/'start-000-surrogate-trace.json').read_text())
            assert detail['observed_trials']==detail['rejected_trials']==1
            assert detail['trials'][0]['rejected_trial_mask']==[True,False]
            assert not detail['trials'][0]['accepted']
            assert detail['trials'][0]['majorization_slack'] < 0
            failure=detail['unresolved_qps'][0]
            assert not failure['certificate']['inner_qp_qualified']
            with np.load(tmp_path/failure['problem_and_returned_state']['path']) as state:
                np.testing.assert_array_equal(state['h'],seen[-1].numpy())
                np.testing.assert_array_equal(state['start'],start.numpy())
        histories.append(seen)
    assert histories[0][0].tolist()==histories[1][0].tolist()==[1.,1e13]
    assert histories[0][1].tolist()==[2.,2e13]
    assert histories[1][1].tolist()==[2.,1e13]


def test_trace_retains_head_and_recent_tail_with_explicit_omission_count(tmp_path):
    from benchmarks.qualify_mixed_cuda import Journal
    from benchmarks.surrogate_trace import SurrogateTrace
    from clipp1d.cuda.policy import CudaPolicy
    journal=Journal(tmp_path/'trace.json')
    trace=SurrogateTrace(journal,0,t(1.),CudaPolicy(),head_limit=2,tail_limit=2)
    for i in range(10):
        trace('outer_trial',dict(iteration=i,backtrack_index=0,inflation=1.,
            base_h=t([1.,2.]),h=t([1.,2.]),losses=t([1.,1.]),gradient_step=t([0.,0.]),
            quadratic_step=t([0.,0.]),trial_losses=t([1.,1.]),accepted=True,
            finite_gate=True,margin=t(1e-14),majorization_slack=t(1e-14),
            surrogate_slack=t(1e-14),objective_slack=t(1e-14)))
    trace.finish()
    detail=json.loads((journal.root/'start-000-surrogate-trace.json').read_text())
    assert detail['observed_trials']==10 and detail['omitted_trials']==6
    assert [row['sequence'] for row in detail['trials']]==[0,1,8,9]
