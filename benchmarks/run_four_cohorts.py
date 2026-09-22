"""Frozen-source local CPU benchmark: one fresh process and one CPU per tumor.

The default raw-path winner is retained in each fit, enabling a paired comparison
of partitions without a second full fusion search. This is an ablation of the
new proposal pool, not a second independent historical-version timing run.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import time
import traceback


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, *, replace=False):
    path = Path(path)
    if replace:
        temporary = path.with_suffix('.tmp')
        with temporary.open('w') as stream:
            json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        os.replace(temporary,path)
    else:
        with path.open('x') as stream:
            json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write('\n')


def load(path):
    return json.loads(Path(path).read_text())


def evaluate(result, case, directory):
    import numpy as np
    from clipp1d.io import read_tumor
    from clipp1d.model import compile_model, posterior_multiplicity
    from compare_clipp2 import read_rows, metrics
    truth = read_rows(case['truth_path'])
    observed = read_rows(directory/'fit/mutation_clusters.tsv')
    mult = read_rows(directory/'fit/mutation_multiplicity.tsv')
    assert set(truth) <= set(observed)
    matched = {k:v for k,v in observed.items() if k in truth}
    assert matched
    model = compile_model(read_tumor(case['input_path']))
    assert set(model.mutation_ids) == set(observed)
    reference = result.candidate_provenance['raw_reference']
    phi = np.empty(len(model))
    labels = np.empty(len(model),dtype=int)
    cuts, centers = reference['partition_cuts'], reference['refit_centers']
    designated = reference['designated_clonal_block']
    order = [designated]+sorted((i for i in range(len(centers)) if i != designated),
                               key=lambda i:(-centers[i],cuts[i]))
    public = {block:label for label,block in enumerate(order)}
    for block,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
        phi[a:b],labels[a:b] = centers[block],public[block]
    phi = phi[result.frozen_chain.inverse_order]
    labels = labels[result.frozen_chain.inverse_order]
    calls = np.argmax(posterior_multiplicity(model,phi),axis=1)+1
    baseline = {mid:dict(cluster_label=str(labels[i]),refitted_ccf=str(phi[i]))
                for i,mid in enumerate(model.mutation_ids) if mid in truth}
    baseline_mult = {mid:dict(multiplicity_call=str(calls[i]))
                     for i,mid in enumerate(model.mutation_ids) if mid in truth}
    comparison = {}
    for name,table,dosage in [('integrated',matched,mult),('fusion_path',baseline,baseline_mult)]:
        item = metrics(table,truth,dosage,'refitted_ccf')
        item['mae'] = float(np.mean([abs(float(table[mid]['refitted_ccf'])-float(truth[mid]['true_ccf'])) for mid in table]))
        item['true_k'] = len({truth[mid]['true_cluster'] for mid in table})
        item['true_smf'] = float(np.mean([float(truth[mid]['true_ccf']) < 1-1e-12 for mid in table]))
        item['smf_error'] = abs(1-item['all_exact_one_fraction']-item['true_smf'])
        item['truth_coverage'] = len(table)/len(observed)
        comparison[name] = item
    write(directory/'metrics.json',comparison)
    return comparison


def worker(args):
    from benchmark_chain import configure_execution
    controls = configure_execution(cpus=[args.cpu],threads=1)
    root = args.root.resolve()
    plan = load(root/'plan.json')
    case = plan['cases'][args.index]
    directory = root/'runs'/case['dataset']/case['case_id']
    directory.mkdir(parents=True)
    sys.path.insert(0,str(root/'frozen/src'))
    from clipp1d import fit
    from clipp1d.api import source_provenance
    source = source_provenance()
    started = time.monotonic()
    try:
        assert source['source_sha256'] == plan['source']['source_sha256'], 'Source identity mismatch'
        assert sha(case['input_path']) == case['input_sha256'], 'Input identity mismatch'
        assert sha(case['truth_path']) == case['truth_sha256'], 'Truth identity mismatch'
        for name,digest in plan['runner_files'].items():
            assert sha(root/'frozen/benchmarks'/name) == digest, 'Runner identity mismatch'
        write(directory/'startup.json',dict(started_utc=now(),pid=os.getpid(),controls=controls,
                                            source=source,source_binding=plan['source_binding'],
                                            input_sha256=case['input_sha256'],truth_sha256=case['truth_sha256'],
                                            plan_sha256=sha(root/'plan.json')))
    except Exception as error:
        write(directory/'terminal.json',dict(status='setup_failure',error=repr(error),finished_utc=now()))
        return 4
    try:
        result = fit(case['input_path'],directory/'fit')
        run = load(directory/'fit/run.json')
        assert run['status']=='success'
        for name,digest in run['table_sha256'].items():
            assert sha(directory/'fit'/name)==digest
        comparison = evaluate(result,case,directory)
        import resource
        write(directory/'terminal.json',dict(status='success',finished_utc=now(),
              elapsed_seconds=time.monotonic()-started,fit_seconds=result.provenance['elapsed_seconds'],
              proposal_seconds=result.search_diagnostics['direct_proposals']['seconds'],
              max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              search_status=result.search_status,selected_origin=result.candidate_provenance['origin'],
              run_sha256=sha(directory/'fit/run.json'),metrics_sha256=sha(directory/'metrics.json'),
              score_gain=result.candidate_provenance['raw_reference']['refit_score']-result.selection_score,
              ari_gain=comparison['integrated']['ari']-comparison['fusion_path']['ari']))
        return 0
    except Exception as error:
        traceback.print_exc()
        fit_run = directory/'fit/run.json'
        status = 'evaluation_failure' if fit_run.exists() and load(fit_run).get('status')=='success' else 'fit_failure'
        write(directory/'terminal.json',dict(status=status,error_type=type(error).__name__,message=str(error),
                                            finished_utc=now(),elapsed_seconds=time.monotonic()-started))
        return 2


def summarize(root, plan, terminals, *, final=False):
    # Derive compact per-cohort numbers from completed, hash-validated worker records.
    import statistics
    cohorts = {}
    for dataset,count in Counter(c['dataset'] for c in plan['cases']).items():
        records = [t for t in terminals if t['dataset']==dataset]
        successful = [t for t in records if t['status']=='success']
        metrics = [load(root/'runs'/dataset/t['case_id']/'metrics.json') for t in successful]
        entry = dict(planned=count,terminal=len(records),status=dict(Counter(t['status'] for t in records)),
                     uncompleted=count-len(records))
        if successful:
            entry['runtime_seconds'] = dict(median=statistics.median(t['fit_seconds'] for t in successful),
                                            maximum=max(t['fit_seconds'] for t in successful),
                                            total=sum(t['fit_seconds'] for t in successful))
            entry['proposal_seconds_median'] = statistics.median(t['proposal_seconds'] for t in successful)
            entry['max_rss_bytes'] = max(t['max_rss_bytes'] for t in successful)
            entry['incomplete_searches'] = sum(t['search_status']!='complete' for t in successful)
            entry['score_improvements'] = sum(t['score_gain']>1e-8 for t in successful)
            entry['ari_improved'] = sum(t['ari_gain']>1e-12 for t in successful)
            entry['ari_worsened'] = sum(t['ari_gain'] < -1e-12 for t in successful)
            for method in ['fusion_path','integrated']:
                values = [m[method] for m in metrics]
                multi = [m for m in values if not m['true_k_one']]
                single = [m for m in values if m['true_k_one']]
                pooled = {}
                for v in values:
                    for k,c in v['cna_only_multiplicity']['per_class'].items():
                        dest = pooled.setdefault(k,dict(tp=0,fp=0,fn=0,support=0))
                        for key in dest:
                            dest[key] += c[key]
                for c in pooled.values():
                    c['f1'] = 2*c['tp']/(2*c['tp']+c['fp']+c['fn']) if 2*c['tp']+c['fp']+c['fn'] else 0
                tp,fp,fn,support = (sum(c[k] for c in pooled.values()) for k in ['tp','fp','fn','support'])
                entry[method] = dict(cases=len(values),multi_cluster_cases=len(multi),single_cluster_cases=len(single),
                    mean_ari_multi=statistics.mean(m['ari'] for m in multi) if multi else None,
                    mean_ari_all=statistics.mean(m['ari'] for m in values),
                    mean_mae=statistics.mean(m['mae'] for m in values),
                    mean_ccc=statistics.mean(m['ccc'] for m in values),
                    mean_smf_error=statistics.mean(m['smf_error'] for m in values),
                    false_splits=sum(m['selected_k']>1 for m in single),
                    exact_k=sum(m['selected_k']==m['true_k'] for m in values),
                    truth_partial_cases=sum(m['truth_coverage']<1 for m in values),
                    cna_multiplicity=dict(eligible=support,per_class=pooled,
                        macro_f1=statistics.mean(c['f1'] for c in pooled.values()) if pooled else None,
                        micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                        weighted_f1=sum(c['f1']*c['support'] for c in pooled.values())/support if support else None,
                        called=sum(v['cna_only_multiplicity']['called'] for v in values)))
        cohorts[dataset] = entry
    output = dict(snapshot_utc=now(),final=final,cohorts=cohorts,
                  scope='All planned tumors; metrics describe successful fits only, with explicit truth coverage. Fusion path is the paired internal ablation.',
                  plan_sha256=sha(root/'plan.json'))
    write(root/('summary-final.json' if final else 'summary-current.json'),output,replace=not final)
    return output


def controller(args):
    root=args.root.resolve()
    plan=load(root/'plan.json')
    if not args.owner.replace('_','').isalnum():
        raise ValueError('Owner must be alphanumeric with optional underscores')
    (root/(args.owner+'.lock')).mkdir()
    owner=dict(pid=os.getpid(),proc_start_ticks=Path('/proc/self/stat').read_text().split()[21],
               started_utc=now(),plan_sha256=sha(root/'plan.json'),command=sys.argv)
    write(root/(args.owner+'.json'),owner)
    admitted_cpus = sorted(os.sched_getaffinity(0))[:args.workers] if args.resume else plan['cpus']
    assert len(admitted_cpus) == (args.workers if args.resume else len(plan['cpus']))
    write(root/(args.owner+'-capacity.json'),dict(cpus=admitted_cpus,workers=len(admitted_cpus),
                                               activated_utc=now(),plan_sha256=sha(root/'plan.json'),
                                               source_and_scientific_settings_unchanged=True))
    cpus=queue.Queue()
    for cpu in admitted_cpus:
        cpus.put(cpu)
    canaries=[]
    for dataset in sorted({c['dataset'] for c in plan['cases']}):
        canaries.append(min((i for i,c in enumerate(plan['cases']) if c['dataset']==dataset),
                            key=lambda i:(plan['cases'][i]['retained_mutations'],i)))
    ordered=canaries+sorted((i for i in range(len(plan['cases'])) if i not in canaries),
                           key=lambda i:(plan['cases'][i]['retained_mutations']**2,i))
    terminals=[]
    completed=set()
    adopted=[]
    if args.resume:
        prior = load(root/'controller.json')
        proc = Path('/proc')/str(prior['pid'])
        assert not proc.exists() or proc.joinpath('stat').read_text().split()[21] != prior['proc_start_ticks'] or proc.joinpath('stat').read_text().split()[2] == 'Z', 'Prior owner must be retired'
        assert (root/'capacity25-retired.json').exists(), 'Drain and retirement receipt required'
        for index,case in enumerate(plan['cases']):
            directory=root/'runs'/case['dataset']/case['case_id']
            terminal=directory/'terminal.json'
            if not terminal.exists():
                if directory.exists():
                    startup=load(directory/'startup.json')
                    registered=load(root/'capacity25-retired.json')['adopted_children']
                    child=next(c for c in registered if c['pid']==startup['pid'])
                    proc=Path('/proc')/str(child['pid'])
                    assert proc.exists() and proc.joinpath('stat').read_text().split()[21]==child['start_ticks']
                    assert startup['plan_sha256']==sha(root/'plan.json')
                    assert startup['input_sha256']==case['input_sha256']
                    cpu=startup['controls']['selected_cpus'][0]
                    assert cpu in admitted_cpus
                    adopted.append(dict(index=index,cpu=cpu,child=child,started_utc=startup['started_utc']))
                    completed.add(index)
                continue
            result=load(terminal)
            assert result['status']=='success', 'Only verified successful imports in this capacity-only recovery'
            assert sha(directory/'fit/run.json')==result['run_sha256']
            run=load(directory/'fit/run.json')
            assert run['provenance']['source_sha256']==plan['source']['source_sha256']
            assert run['provenance']['input_sha256']==case['input_sha256']
            for name,digest in run['table_sha256'].items():
                assert sha(directory/'fit'/name)==digest
            assert sha(directory/'metrics.json')==result['metrics_sha256']
            result.update(dataset=case['dataset'],case_id=case['case_id'],index=index,returncode=0)
            terminals.append(result)
            completed.add(index)
        write(root/(args.owner+'-imports.json'),dict(cases=sorted(c['index'] for c in terminals),count=len(terminals),adopted=adopted,
                                                   validated_utc=now(),plan_sha256=sha(root/'plan.json')))
        ordered=[i for i in ordered if i not in completed]
        canaries=[i for i in canaries if i not in completed]
    # Existing admitted processes retain their CPU and inherited walltime allowance.
    reserved={a['cpu'] for a in adopted}
    cpus=queue.Queue()
    for cpu in admitted_cpus:
        if cpu not in reserved:
            cpus.put(cpu)
    halted=False
    def adopt(record):
        index=record['index']
        case=plan['cases'][index]
        child=record['child']
        proc=Path('/proc')/str(child['pid'])
        expired=False
        try:
            while proc.exists():
                stat=proc.joinpath('stat').read_text().split()
                if stat[21]!=child['start_ticks'] or stat[2]=='Z':
                    break
                elapsed=time.time()-datetime.fromisoformat(record['started_utc']).timestamp()
                if elapsed>plan['timeout_seconds']:
                    os.killpg(child['pid'],signal.SIGTERM)
                    expired=True
                    break
                time.sleep(2)
            directory=root/'runs'/case['dataset']/case['case_id']
            terminal=directory/'terminal.json'
            if terminal.exists():
                result=load(terminal)
            else:
                result=dict(status='timeout' if expired else 'process_failure',finished_utc=now())
                write(terminal,result)
            result.update(dataset=case['dataset'],case_id=case['case_id'],index=index,
                          returncode=0 if result['status']=='success' else 2)
            return result
        finally:
            cpus.put(record['cpu'])
    def run(index):
        case=plan['cases'][index]
        cpu=cpus.get()
        logs=root/'logs'/case['dataset']
        logs.mkdir(parents=True,exist_ok=True)
        directory=root/'runs'/case['dataset']/case['case_id']
        try:
            with (logs/(case['case_id']+'.log')).open('x') as stream:
                command=[sys.executable,'-B',str(root/'frozen/benchmarks/run_four_cohorts.py'),
                         'worker','--root',str(root),'--index',str(index),'--cpu',str(cpu)]
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    code=process.wait(timeout=plan['timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
                    code=-signal.SIGTERM
                terminal=directory/'terminal.json'
                if terminal.exists():
                    result=load(terminal)
                else:
                    directory.mkdir(parents=True,exist_ok=True)
                    result=dict(status='timeout' if code==-signal.SIGTERM else 'process_failure',
                                returncode=code,finished_utc=now())
                    write(terminal,result)
                result.update(dataset=case['dataset'],case_id=case['case_id'],index=index,returncode=code)
                return result
        finally:
            cpus.put(cpu)
    # Qualify a real smallest case from each cohort before admitting the rest.
    for index in canaries:
        result=run(index)
        terminals.append(result)
        print(json.dumps(result),flush=True)
        summarize(root,plan,terminals)
        if result['status']!='success':
            write(root/'HALTED.json',dict(reason='canary failure',case=result,unstarted=len(plan['cases'])-len(terminals)))
            return 3
    with ThreadPoolExecutor(max_workers=len(admitted_cpus)) as pool:
        iterator=iter(ordered[len(canaries):])
        pending={pool.submit(adopt,record):record['index'] for record in adopted}
        def fill():
            while not halted and len(pending)<len(admitted_cpus):
                index=next(iterator,None)
                if index is None:
                    break
                pending[pool.submit(run,index)]=index
        fill()
        while pending:
            future=next(as_completed(pending))
            pending.pop(future)
            result=future.result()
            terminals.append(result)
            print(json.dumps(result),flush=True)
            if result['status'] in ('setup_failure','process_failure','evaluation_failure'):
                halted=True
            if len(terminals)%10==0 or halted:
                summarize(root,plan,terminals)
            write(root/'status.json',dict(snapshot_utc=now(),completed=len(terminals),planned=len(plan['cases']),
                  active=[dict(index=i,dataset=plan['cases'][i]['dataset'],case_id=plan['cases'][i]['case_id']) for i in pending.values()],
                  halted=halted,status=dict(Counter(t['status'] for t in terminals))),replace=True)
            fill()
    if halted:
        write(root/'HALTED.json',dict(reason='infrastructure or validation failure',completed=len(terminals)))
        summarize(root,plan,terminals)
        return 3
    assert len(terminals)==len(plan['cases'])
    summarize(root,plan,terminals,final=True)
    write(root/'COMPLETE.json',dict(finished_utc=now(),cases=len(terminals),status=dict(Counter(t['status'] for t in terminals)),
                                   summary_sha256=sha(root/'summary-final.json')))
    return 0


def freeze(args):
    root=args.root.resolve()
    root.mkdir()
    repo=Path(__file__).resolve().parents[1]
    prepared=load(args.prepared)
    assert len(prepared)==5456 and all(c['status']=='prepared' for c in prepared)
    snapshot=root/'frozen'
    shutil.copytree(repo/'src',snapshot/'src',ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    (snapshot/'benchmarks').mkdir()
    scripts=['run_four_cohorts.py','benchmark_chain.py','compare_clipp2.py']
    for name in scripts:
        shutil.copy2(repo/'benchmarks'/name,snapshot/'benchmarks'/name)
    sys.path.insert(0,str(snapshot/'src'))
    from clipp1d.api import source_provenance
    source=source_provenance()
    base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    patch=subprocess.check_output(['git','diff','HEAD','--','src'],cwd=repo)
    (root/'source.patch').write_bytes(patch)
    untracked=[str(p.relative_to(repo)) for p in (repo/'src/clipp1d').glob('*.py')
               if subprocess.run(['git','ls-files','--error-unmatch',str(p.relative_to(repo))],cwd=repo,capture_output=True).returncode]
    cpus=sorted(os.sched_getaffinity(0))[:args.workers]
    assert len(cpus)==args.workers
    plan=dict(schema='clipp1d.full_cohort.v1',created_utc=now(),cases=prepared,source=source,
              source_binding=dict(base_commit=base,tracked_patch_sha256=sha(root/'source.patch'),
                                  untracked_source_files=untracked,complete_package_files=source['source_files']),
              prepared_manifest=str(args.prepared.resolve()),prepared_sha256=sha(args.prepared),
              runner_files={name:sha(snapshot/'benchmarks'/name) for name in scripts},
              cpus=cpus,timeout_seconds=21600,
              execution='local CPU; one fresh process and one pinned CPU per tumor; shared host',
              comparison='unchanged fusion-path candidate versus full integrated proposal pool in same fit',
              simclone_input='separate normal-CN2 corrected copy; original canonical inputs retained')
    write(root/'plan.json',plan)
    print(json.dumps(dict(plan=str(root/'plan.json'),cases=len(prepared),source_sha256=source['source_sha256'],cpus=cpus)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['freeze','controller','worker'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--index',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--owner',default='controller')
    parser.add_argument('--cpu',type=int)
    args=parser.parse_args()
    if args.mode=='freeze':
        freeze(args)
        return 0
    return controller(args) if args.mode=='controller' else worker(args)


if __name__=='__main__':
    sys.exit(main())
