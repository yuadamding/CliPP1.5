"""Research dispatch and qualification admission are independent of CPU fitting."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

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
