"""Research dispatch and qualification admission are independent of CPU fitting."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from benchmarks import qualify_surrogate_cuda as study
from clipp1d.cuda.policy import CudaPolicy


def test_research_dispatch_is_explicit_and_restored_even_on_error(monkeypatch):
    calls=[]
    def original(*args,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(qualified=True,diagnostics=dict(
            outer_surrogate_policy=kwargs['surrogate_policy'],status='qualified'))
    monkeypatch.setattr(study.solver,'solve_start',original)
    previous_source=study.mixed.source_provenance
    previous_pred=study.mixed.check_predecessor
    with pytest.raises(RuntimeError,match='interrupted'):
        with study.strategy(study.CANDIDATE) as starts:
            study.solver.solve_start(None,None,None,None,CudaPolicy())
            assert len(starts)==1
            provenance=study.mixed.source_provenance()
            assert provenance['research_execution'] and provenance['outer_surrogate_policy']==study.CANDIDATE
            assert provenance['production_default_outer_surrogate_policy']==study.CONTROL
            raise RuntimeError('interrupted')
    assert calls==[{'surrogate_policy':study.CANDIDATE}]
    assert study.solver.solve_start is original
    assert study.mixed.source_provenance is previous_source
    assert study.mixed.check_predecessor is previous_pred


def test_unknown_research_strategy_rejected_before_dispatch():
    with pytest.raises(ValueError,match='Unknown'):
        with study.strategy('loosen_tolerance'):
            pass


@pytest.mark.parametrize('failure',['solver_exception','dispatch_exception'])
def test_exception_then_continued_start_reserves_distinct_trace_ids(monkeypatch,tmp_path,failure):
    count=0
    def original(*args,**kwargs):
        nonlocal count
        count+=1
        if count==1 and failure=='solver_exception':
            raise RuntimeError('first start raised')
        return SimpleNamespace(qualified=True,diagnostics=dict(
            outer_surrogate_policy='wrong' if count==1 else study.CANDIDATE,status='qualified'))
    monkeypatch.setattr(study.solver,'solve_start',original)
    journal=study.mixed.Journal(tmp_path/'trial.json')
    with study.strategy(study.CANDIDATE,journal) as calls:
        with pytest.raises((RuntimeError,AssertionError)):
            study.solver.solve_start(None,None,1.,None,CudaPolicy())
        first=journal.root/'start-000-surrogate-trace.json'
        retained=first.read_bytes()
        fit=study.solver.solve_start(None,None,1.,None,CudaPolicy())
        assert fit.qualified
        assert first.read_bytes()==retained
        assert json.loads((journal.root/'start-001-surrogate-trace.json').read_text())['start_index']==1
        assert calls.counts()==dict(starts_attempted=2,starts_returned=1 if failure=='solver_exception' else 2,
                                   starts_qualified=1,starts_raised=1)
        assert len(calls)==1
    assert study.solver.solve_start is original


def test_trial_retains_exception_attempt_counts(monkeypatch,tmp_path):
    def original(*args,**kwargs):
        raise RuntimeError('failed start')
    monkeypatch.setattr(study.solver,'solve_start',original)
    monkeypatch.setattr(study.mixed,'execute',lambda *a:
                        study.solver.solve_start(None,None,1.,None,CudaPolicy()))
    record=study.trial(SimpleNamespace(fixture='below_one',nodes=64),tmp_path/'trial.json',study.CANDIDATE,0)
    assert record['starts_attempted']==record['starts_raised']==1
    assert record['starts_returned']==record['starts_qualified']==0
    assert record['start_attempts'][0]['error']=='failed start'


def predecessor(tmp_path, **changes):
    record=dict(schema=study.SCHEMA,status='passed',fixture_family='below_one',nodes=64,
        source={'source_sha256':'source'},helpers=study.helpers(),policy=asdict(CudaPolicy()),
        candidate_policy=study.CANDIDATE,cuda_available=True,artifacts={},
        trials=[dict(surrogate_policy=study.CANDIDATE,status='passed',
                     full_path=dict(search_status='complete',final_export_and_publication_qualified=True))])
    record.update(changes)
    path=tmp_path/'previous.json'
    path.write_text(json.dumps(record))
    args=SimpleNamespace(nodes=256,fixture='below_one',predecessor=path,
                         predecessor_sha256=study.common.sha(path))
    return args


@pytest.mark.parametrize('changes',[
    {'status':'failed'}, {'nodes':256}, {'source':{'source_sha256':'wrong'}},
    {'fixture_family':'mixed_support'}, {'candidate_policy':study.CONTROL},
    {'cuda_available':False}, {'helpers':{}},
    {'policy':dict(asdict(CudaPolicy()),inner_kkt_tol=1e-6)},
    {'trials':[dict(surrogate_policy=study.CANDIDATE,status='failed',
                   full_path=dict(search_status='incomplete',final_export_and_publication_qualified=False))]},
])
def test_candidate_requires_exact_smaller_complete_cuda_qualification(tmp_path,changes):
    args=predecessor(tmp_path,**changes)
    with pytest.raises(AssertionError):
        study.load_predecessor(args,{'source_sha256':'source'})


def test_candidate_admits_verified_smaller_qualification(tmp_path):
    args=predecessor(tmp_path)
    result=study.load_predecessor(args,{'source_sha256':'source'})
    assert result['nodes']==64


def test_timeout_cannot_be_swallowed_as_a_failed_trial_and_continue(monkeypatch,tmp_path):
    def expire(*args):
        raise TimeoutError('budget')
    monkeypatch.setattr(study.mixed,'execute',expire)
    args=SimpleNamespace(fixture='below_one',nodes=64)
    output=tmp_path/'trial.json'
    with pytest.raises(TimeoutError):
        study.trial(args,output,study.CANDIDATE,0)
    record=json.loads(output.read_bytes())
    assert record['status']=='failed' and record['error_type']=='TimeoutError'


def synthetic_pair(tmp_path, candidate_lambda):
    records=[]
    for name,lam in ((study.CONTROL,1.),(study.CANDIDATE,candidate_lambda)):
        directory=tmp_path/name
        directory.mkdir()
        summary=dict(selected_lambda=lam,raw_objective=1.+lam*.6,raw_ccf=[.3,.7],
            refitted_ccf=[.3,.7],cluster_labels=[0,1],score=2.,search_status='complete',
            fit_seconds=1.,path_records=[{},{}],timings={'qp_admm_iterations':16},
            pilot_and_graph={key:'same' for key in
                ('pilot_sha256','weights_sha256','weight_rule','gap_floor','normalization')})
        (directory/'full-path.json').write_text(json.dumps(summary))
        (directory/'initial-path-plan.json').write_text('{}')
        events=[dict(kind='candidate_raw',lambda_value=value,qualified=True,
                     raw_objective=1.+value*.6,seconds=.1,diagnostics=dict(
                         search_complete=True,qp_admm_iterations=8,starts_attempted=2,
                         starts_unresolved=0)) for value in (0.,1.,lam) if value != 0.]
        events=list({row['lambda_value']:row for row in events}.values())
        events_file=name+'.jsonl'
        (tmp_path/events_file).write_text(''.join(json.dumps(row)+'\n' for row in events))
        records.append(dict(repeat=0,surrogate_policy=name,artifact_directory=name,events_file=events_file))
    return records


@pytest.mark.parametrize('lam',[1.,2.,np.nextafter(1.,2.)])
def test_selected_objectives_only_comparable_at_identical_literal_lambda(tmp_path,lam):
    row,=study.comparisons(synthetic_pair(tmp_path,lam),tmp_path)
    assert row['candidate_selected_lambda']==lam and row['control_selected_lambda']==1.
    same=bool(lam==1.)
    assert row['same_selected_lambda'] is same
    assert row['raw_objectives_comparable_at_selected_lambda'] is same
    assert row['raw_max_difference']==0
    if same:
        assert row['raw_objective_difference']==0
    else:
        assert row['raw_objective_difference'] is None
        assert row['separately_selected_raw_objective_difference']==(1.+lam*.6)-1.6
    fixed=row['fixed_lambda_comparisons']
    assert fixed['common_literal_lambdas'][0]['lambda_value']==1.
    assert fixed['common_literal_lambdas'][0]['raw_objective_difference']==0
    assert fixed['candidate_only_lambdas']==([] if same else [lam])


def test_common_lambda_unqualified_state_is_not_an_objective_comparison(tmp_path):
    records=synthetic_pair(tmp_path,1.)
    path=tmp_path/records[1]['events_file']
    row=json.loads(path.read_text())
    row.update(kind='candidate_failed',qualified=False,error='no qualified starts')
    row.pop('raw_objective')
    path.write_text(json.dumps(row)+'\n')
    result,=study.comparisons(records,tmp_path)
    fixed,=result['fixed_lambda_comparisons']['common_literal_lambdas']
    assert not fixed['qualified_objectives_comparable']
    assert fixed['raw_objective_difference'] is None
    assert fixed['candidate_error']=='no qualified starts'


def main_args(monkeypatch,tmp_path,*extra):
    out=tmp_path/'experiment.json'
    monkeypatch.setattr(study.sys,'argv',['study','--fixture','below_one','--nodes','64',
        '--expected-source-sha256','source','--out',str(out),*extra])
    return out


@pytest.mark.parametrize('timeout',[0,-1,-3600])
def test_nonpositive_deadline_rejected_before_files_or_handlers(monkeypatch,tmp_path,timeout):
    main_args(monkeypatch,tmp_path,'--timeout-seconds',str(timeout))
    monkeypatch.setattr(study.signal,'signal',lambda *a:pytest.fail('installed handler'))
    monkeypatch.setattr(study.signal,'alarm',lambda *a:pytest.fail('installed alarm'))
    with pytest.raises(ValueError,match='timeout must be positive'):
        study.main()
    assert not list(tmp_path.iterdir())


def mock_main(monkeypatch):
    previous={study.signal.SIGALRM:object(),study.signal.SIGTERM:object()}
    current=previous.copy()
    alarms=[]
    def handler(sig,value):
        old=current[sig]
        current[sig]=value
        return old
    monkeypatch.setattr(study.signal,'signal',handler)
    monkeypatch.setattr(study.signal,'alarm',alarms.append)
    monkeypatch.setattr(study,'source_provenance',lambda:dict(source_sha256='source'))
    monkeypatch.setattr(study,'require_cuda',lambda device:'mock-device')
    monkeypatch.setattr(study.torch.cuda,'get_device_name',lambda device:'mock-GPU')
    monkeypatch.setattr(study,'trial',lambda args,out,name,repeat:
                        dict(surrogate_policy=name,repeat=repeat,status='passed'))
    monkeypatch.setattr(study,'comparisons',lambda *a:[])
    return previous,current,alarms


@pytest.mark.parametrize('failure',[None,ValueError,TimeoutError,KeyboardInterrupt])
def test_signal_handlers_restored_after_success_or_failure(monkeypatch,tmp_path,failure):
    out=main_args(monkeypatch,tmp_path,'--timeout-seconds','7')
    previous,current,alarms=mock_main(monkeypatch)
    if failure:
        def fail(*args):
            raise failure('test failure')
        monkeypatch.setattr(study,'require_cuda',fail)
    assert study.main()==int(failure is not None)
    assert current==previous and alarms==[7,0]
    receipt=json.loads(out.read_text())
    assert receipt['status']==('failed' if failure else 'passed')
    assert receipt['timeout_seconds']==7


@pytest.mark.parametrize('qualified',[False,True])
def test_literal_replay_retains_actual_states_and_both_certificates_before_admission(
        monkeypatch,tmp_path,qualified):
    out=main_args(monkeypatch,tmp_path,'--literal-capture','capture.json','--literal-sha256','hash')
    previous,current,_=mock_main(monkeypatch)
    def t(value):
        return torch.tensor(value,dtype=torch.float64)
    problem=dict(h=t([1.,2.]),target=t([.3,.7]),lower=t([0.,0.]),upper=t([1.,1.]),
                 caps=t([[0.,.1],[.1,0.]]),start=t([.2,.8]),dual=None)
    fitted=SimpleNamespace(x=t([.4,.6]),dual=torch.zeros((2,2),dtype=torch.float64),
        iterations=16,qualified=qualified,gap=t(0.),scale=t(1.),kkt=t(0. if qualified else 1.))
    monkeypatch.setattr(study.replay.capture,'load_capture',lambda *a:{'captures':[{}]})
    monkeypatch.setattr(study.replay.capture,'load_record',lambda *a:{})
    monkeypatch.setattr(study.replay,'check_problem',lambda *a:CudaPolicy())
    monkeypatch.setattr(study.replay,'load_problem',lambda *a:problem)
    monkeypatch.setattr(study,'Kernels',lambda device,compiled:compiled)
    monkeypatch.setattr(study.replay.qp,'solve_qp',lambda *a,**k:fitted)
    monkeypatch.setattr(study.replay,'stats_for',lambda *a:t([0.,1.,0. if qualified else 1.]))
    assert study.main()==int(qualified)
    record=json.loads(out.read_text())
    assert current==previous
    detail=record['literal_replay']
    assert detail['fitted_qualified']==qualified
    assert detail['compiled']['independently_qualified']==qualified
    assert detail['eager']['independently_qualified']==qualified
    assert detail['problem_unchanged']
    assert record['literal_replay_artifacts']
    for relative,digest in record['artifacts'].items():
        assert study.common.sha(tmp_path/relative)==digest
    tensors=out.with_suffix('.trials')/detail['returned_state']['path']
    with np.load(tensors,allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['x'],fitted.x.numpy())
        np.testing.assert_array_equal(arrays['q'],fitted.dual.numpy())
    assert (out.with_suffix('.trials')/'literal-replay.json').exists()
    if qualified:
        assert record['trials']==[]
        assert 'unexpectedly promoted' in record['error']
