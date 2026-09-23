"""Explicit research comparison of scalar and coordinate outer backtracking.

One fixture/size per allocated CUDA job. The default production strategy is not
changed. Each trial retains the original likelihood, pilots, fixed graph, boxes,
planned starts, QP gates, raw audits, score, refit and publication checks. A
changed outer surrogate is never described as recovery of a literal saved QP.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
from time import perf_counter
import traceback

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import qualify_cuda as common
from benchmarks import qualify_mixed_cuda as mixed
from benchmarks import replay_failed_qp as replay
from benchmarks import surrogate_trace as tracing
from clipp1d.api import source_provenance
from clipp1d.cuda import solver
from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda_api import require_cuda

SCHEMA='clipp1d.cuda.surrogate_experiment.v2'
CONTROL='scalar_backtracking_v1'
CANDIDATE='coordinate_backtracking_v1'
ORIGINAL_HELPERS=mixed.helper_hashes


def helpers():
    return dict(ORIGINAL_HELPERS(),surrogate_driver=common.sha(Path(__file__)),
                surrogate_trace=common.sha(Path(tracing.__file__)),
                literal_replay=common.sha(Path(replay.__file__)),
                capture=common.sha(Path(replay.capture.__file__)))


def load_predecessor(args,source):
    if args.nodes==64:
        mixed.require(args.predecessor is None and args.predecessor_sha256 is None,
                      '64-node experiment has no predecessor')
        return None
    path=args.predecessor
    mixed.require(path is not None and common.sha(path)==args.predecessor_sha256,
                  'Require the exact previous-size candidate qualification')
    prior=mixed.load_json(path)
    mixed.require(prior['schema']==SCHEMA and prior['status']=='passed'
                  and prior['fixture_family']==args.fixture and prior['nodes']=={256:64,512:256}[args.nodes]
                  and prior['source']['source_sha256']==source['source_sha256']
                  and prior['helpers']==helpers() and prior['policy']==asdict(CudaPolicy())
                  and prior['candidate_policy']==CANDIDATE and prior['cuda_available'] is True,
                  'Predecessor source/family/size/policy/helper mismatch')
    candidates=[r for r in prior['trials'] if r['surrogate_policy']==CANDIDATE]
    mixed.require(candidates and all(r['status']=='passed' and
                  r['full_path']['final_export_and_publication_qualified'] is True
                  and r['full_path']['search_status']=='complete' for r in candidates),
                  'Candidate predecessor lacks complete starts/audits/refits/publication')
    for relative,sha in prior['artifacts'].items():
        mixed.require(common.sha(mixed.artifact_file(path.parent,relative))==sha,'Changed predecessor artifact')
    return dict(path=str(path),sha256=args.predecessor_sha256,nodes=prior['nodes'])


@contextmanager
def strategy(name, journal=None):
    """The sole numerical intervention is the declared solve_start keyword."""
    if name not in (CONTROL,CANDIDATE):
        raise ValueError('Unknown experimental strategy')
    original=solver.solve_start
    old_source=mixed.source_provenance
    old_predecessor=mixed.check_predecessor
    old_helpers=mixed.helper_hashes
    calls=[]
    def execute(model,graph,lam,start,policy):
        trace=None if journal is None else tracing.SurrogateTrace(journal,len(calls),lam,policy)
        try:
            kwargs={} if trace is None else dict(observer=trace)
            fit=original(model,graph,lam,start,policy,surrogate_policy=name,**kwargs)
        finally:
            if trace is not None:
                trace.finish()
        mixed.require(fit.diagnostics['outer_surrogate_policy']==name,'Strategy dispatch changed')
        calls.append(dict(qualified=fit.qualified,status=fit.diagnostics['status']))
        return fit
    def source():
        return dict(source_provenance(),outer_surrogate_policy=name,
                    research_execution=True,production_default_outer_surrogate_policy=CONTROL,
                    research_driver_sha256=common.sha(Path(__file__)))
    solver.solve_start=execute
    mixed.source_provenance=source
    mixed.check_predecessor=load_predecessor
    mixed.helper_hashes=helpers
    try:
        yield calls
    finally:
        solver.solve_start=original
        mixed.source_provenance=old_source
        mixed.check_predecessor=old_predecessor
        mixed.helper_hashes=old_helpers


def literal_replay(path,sha,device,journal):
    receipt=replay.capture.load_capture(path,sha)
    mixed.require(len(receipt['captures'])==1,'Expected the single captured below256 obstruction')
    record=replay.capture.load_record(receipt,receipt['captures'][0])
    policy=replay.check_problem(record,receipt['captures'][0])
    problem=replay.load_problem(receipt,record,device)
    before={k:common.tensor_sha(v) if v is not None else None for k,v in problem.items()}
    problem_state=tracing.save_tensors(journal,'literal-problem.npz',problem)
    kernels=Kernels(device,compiled=True)
    fitted=replay.qp.solve_qp(*(problem[k] for k in replay.PROBLEM_KEYS),kernels,policy,
                            start=problem['start'],dual=problem['dual'])
    returned=tracing.save_tensors(journal,'literal-returned.npz',dict(x=fitted.x,q=fitted.dual))
    compiled=replay.certificate(replay.stats_for(fitted.x,fitted.dual,problem,kernels),policy)
    eager=replay.certificate(replay.stats_for(fitted.x,fitted.dual,problem,Kernels(device,compiled=False)),policy)
    after={k:common.tensor_sha(v) if v is not None else None for k,v in problem.items()}
    detail=dict(capture_sha256=sha,problem_sha256=before,iterations=fitted.iterations,
        fitted_qualified=fitted.qualified,compiled=compiled,eager=eager,
        problem_state=problem_state,returned_state=returned,
        problem_sha256_after=after,problem_unchanged=before==after,
        diagnostic_flags=solver.inner_certificate_diagnostics(fitted,policy),
        scope='Unchanged literal binary64 QP and original initialization; no outer-surrogate strategy applies')
    return detail


def admit_literal_replay(detail):
    """Call only after the measurements and binary states are persisted."""
    mixed.require(detail['problem_unchanged'],'Literal replay changed its problem or initialization')
    mixed.require(not detail['fitted_qualified'] and not detail['compiled']['independently_qualified']
                  and not detail['eager']['independently_qualified'],
                  'Literal obstruction unexpectedly promoted: retained evidence requires exact mathematical re-audit')


def trial(args,out,name,repeat):
    journal=mixed.Journal(out)
    record=dict(schema=mixed.SCHEMA,status='running',mode='full',fixture_family=args.fixture,
        nodes=args.nodes,fixture_recipe=mixed.fixtures.RECIPE,source=source_provenance(),
        policy=asdict(CudaPolicy()),helpers=helpers(),artifact_directory=journal.root.name,
        events_file=journal.path.name,surrogate_policy=name,repeat=repeat,
        production_default_changed=False,artifacts={})
    began=perf_counter()
    try:
        with strategy(name,journal) as calls:
            mixed.execute(args,record,journal)
        mixed.require(calls and all(c['qualified'] for c in calls),'Planned starts incompletely qualified')
        record.update(status='passed',starts_attempted=len(calls))
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error),
                      traceback=traceback.format_exc())
        if isinstance(error, (TimeoutError, KeyboardInterrupt, SystemExit)):
            raise
    finally:
        record.update(elapsed_seconds=perf_counter()-began,events_sha256=common.sha(journal.path),
                      artifacts=journal.artifacts)
        common.write_json(out,record)
    return record


def common_lambda_comparisons(pair,root):
    """Literal float equality, never nearest-lambda matching or cross-lambda ranking."""
    paths={}
    for key,record in pair.items():
        rows={}
        for line in (root/record['events_file']).read_text().splitlines():
            event=json.loads(line)
            if event['kind'] not in ('candidate_raw','candidate_failed'):
                continue
            lam=event['lambda_value']
            mixed.require(np.isfinite(lam) and lam >= 0 and lam not in rows,
                          'Invalid or duplicate literal penalty in candidate history')
            rows[lam]=event
        paths[key]=rows
    candidate,control=paths[CANDIDATE],paths[CONTROL]
    matched=[]
    for lam in sorted(candidate.keys() & control.keys()):
        a,b=candidate[lam],control[lam]
        both=bool(a.get('qualified',False) and b.get('qualified',False))
        row=dict(lambda_value=lam,qualified_objectives_comparable=both,
                 raw_objective_difference=a['raw_objective']-b['raw_objective'] if both else None)
        for name,event in (('candidate',a),('control',b)):
            diagnostics=event.get('diagnostics',{})
            row.update({name+'_qualified':event.get('qualified',False),
                        name+'_all_starts_qualified':diagnostics.get('search_complete',False),
                        name+'_raw_objective':event.get('raw_objective'),
                        name+'_seconds':event['seconds'],
                        name+'_qp_iterations':diagnostics.get('qp_admm_iterations'),
                        name+'_starts_attempted':diagnostics.get('starts_attempted'),
                        name+'_starts_unresolved':diagnostics.get('starts_unresolved'),
                        name+'_error':event.get('error')})
        matched.append(row)
    return dict(common_literal_lambdas=matched,
                candidate_only_lambdas=sorted(candidate.keys()-control.keys()),
                control_only_lambdas=sorted(control.keys()-candidate.keys()),
                scope='Same fixed objective at exact literal lambda; trajectories and continuation starts may differ; qualification, coverage and instrumented work are separate')


def comparisons(records,root):
    values=[]
    for index in sorted({r['repeat'] for r in records}):
        pair={r['surrogate_policy']:r for r in records if r['repeat']==index}
        if set(pair)!={CONTROL,CANDIDATE}:
            continue
        summaries={key:mixed.load_json(root/r['artifact_directory']/'full-path.json') for key,r in pair.items()
                   if (root/r['artifact_directory']/'full-path.json').exists()}
        if len(summaries)!=2:
            continue
        a,b=summaries[CANDIDATE],summaries[CONTROL]
        for key in ('pilot_sha256','weights_sha256','weight_rule','gap_floor','normalization'):
            mixed.require(a['pilot_and_graph'][key]==b['pilot_and_graph'][key],
                          'Compared graph/pilots differ')
        plans={key:mixed.load_json(root/r['artifact_directory']/'initial-path-plan.json')
               for key,r in pair.items()}
        mixed.require(plans[CONTROL]==plans[CANDIDATE], 'Initial path policy/penalties differ')
        same_lambda=a['selected_lambda']==b['selected_lambda']
        selected_difference=a['raw_objective']-b['raw_objective']
        values.append(dict(repeat=index,graph_and_pilots_identical=True,
            candidate_search_status=a['search_status'],control_search_status=b['search_status'],
            candidate_fit_seconds=a['fit_seconds'],control_fit_seconds=b['fit_seconds'],
            candidate_penalties=len(a['path_records']),control_penalties=len(b['path_records']),
            candidate_qp_iterations=a['timings']['qp_admm_iterations'],control_qp_iterations=b['timings']['qp_admm_iterations'],
            raw_max_difference=float(np.max(np.abs(np.asarray(a['raw_ccf'])-np.asarray(b['raw_ccf'])))),
            refit_max_difference=float(np.max(np.abs(np.asarray(a['refitted_ccf'])-np.asarray(b['refitted_ccf'])))),
            exact_labels=a['cluster_labels']==b['cluster_labels'],score_difference=a['score']-b['score'],
            candidate_selected_lambda=a['selected_lambda'],control_selected_lambda=b['selected_lambda'],
            same_selected_lambda=same_lambda,
            raw_objectives_comparable_at_selected_lambda=same_lambda,
            separately_selected_raw_objective_difference=selected_difference,
            raw_objective_difference=selected_difference if same_lambda else None,
            selected_objective_scope='Same-lambda objective comparison' if same_lambda else
                'Separately selected objectives at different lambdas; not an optimizer-quality comparison',
            fixed_lambda_comparisons=common_lambda_comparisons(pair,root)))
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',choices=mixed.fixtures.FAMILIES,required=True)
    parser.add_argument('--nodes',type=int,choices=mixed.fixtures.SIZES,required=True)
    parser.add_argument('--expected-source-sha256',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--predecessor',type=Path)
    parser.add_argument('--predecessor-sha256')
    parser.add_argument('--literal-capture',type=Path)
    parser.add_argument('--literal-sha256')
    parser.add_argument('--repeats',type=int,choices=(1,3),default=1)
    parser.add_argument('--timeout-seconds',type=int,default=3000)
    args=parser.parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError('Experiment timeout must be positive')
    args.mode='full'
    args.baseline=args.baseline_sha256=None
    if args.out.exists():
        raise FileExistsError('Never overwrite an experimental receipt')
    work=args.out.with_suffix('.trials')
    work.mkdir()
    record=dict(schema=SCHEMA,status='running',source=source_provenance(),helpers=helpers(),
        policy=asdict(CudaPolicy()),fixture_family=args.fixture,nodes=args.nodes,
        candidate_policy=CANDIDATE,control_policy=CONTROL,trials=[],
        command=sys.argv,started_utc=common.utc_now(),artifacts={},timeout_seconds=args.timeout_seconds,
        scope='Explicit research policy; unchanged binary64 QP gates and observed objective; no production-default promotion',
        timing_scope='Repeated same-fixture fits, alternating execution order; repeat 0 includes cold shape specialization, later repeats reuse process/compiler caches; fit timings include instrumentation')
    began=perf_counter()
    def timeout(signum,frame):
        raise TimeoutError('Bounded experiment wall budget exhausted')
    def terminated(signum,frame):
        raise KeyboardInterrupt('Experimental worker terminated; partial evidence retained')
    previous_alarm=signal.signal(signal.SIGALRM,timeout)
    previous_term=signal.signal(signal.SIGTERM,terminated)
    signal.alarm(args.timeout_seconds)
    try:
        mixed.require(record['source']['source_sha256']==args.expected_source_sha256,'Source differs')
        device=require_cuda(args.device)
        record.update(cuda_available=True,device=str(device),gpu=torch.cuda.get_device_name(device),
                      torch=torch.__version__,cuda_runtime=torch.version.cuda)
        record['predecessor']=load_predecessor(args,record['source'])
        if args.literal_capture is not None:
            journal=mixed.Journal(work/'literal.json')
            record['literal_replay']=literal_replay(args.literal_capture,args.literal_sha256,device,journal)
            common.write_json(work/'literal-replay.json',record['literal_replay'])
            record['literal_replay_artifacts']={str(p.relative_to(args.out.parent)):common.sha(p)
                for p in sorted(work.rglob('*')) if p.is_file()}
            admit_literal_replay(record['literal_replay'])
        for repeat in range(args.repeats):
            order=(CONTROL,CANDIDATE) if repeat%2==0 else (CANDIDATE,CONTROL)
            for name in order:
                out=work/f'{repeat}-{name}.json'
                r=trial(args,out,name,repeat)
                record['trials'].append(r)
                print(json.dumps(dict(repeat=repeat,policy=name,status=r['status'],full_path=r.get('full_path'),error=r.get('error'))),flush=True)
        record['comparisons']=comparisons(record['trials'],work)
        candidate=[r for r in record['trials'] if r['surrogate_policy']==CANDIDATE]
        mixed.require(len(candidate)==args.repeats and all(r['status']=='passed' for r in candidate),
                      'Candidate did not qualify every planned start and final publication')
        mixed.require(all(r['status']=='passed' or
                          r.get('error')=='Complete mixed fixture path contains unresolved candidates or starts'
                          for r in record['trials'] if r['surrogate_policy']==CONTROL),
                      'Control has an unexpected setup/audit/publication failure')
        record['status']='passed'
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM,previous_term)
        signal.signal(signal.SIGALRM,previous_alarm)
        record.update(elapsed_seconds=perf_counter()-began,finished_utc=common.utc_now(),
                      artifacts={str(p.relative_to(args.out.parent)):common.sha(p) for p in sorted(work.rglob('*')) if p.is_file()})
        common.write_json(args.out,record)
    return int(record['status']!='passed')


if __name__=='__main__':
    raise SystemExit(main())
