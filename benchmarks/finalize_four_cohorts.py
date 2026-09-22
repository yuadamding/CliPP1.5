"""Wait for the bound full-cohort run, independently audit it, and publish a report.

No fits, retries, source changes or remote actions are performed. A halted owner
or failed audit produces a separate failure receipt rather than a success report.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def audit(root):
    import numpy as np
    from compare_clipp2 import metrics,read_rows
    plan=load(root/'plan.json')
    plan_digest=sha(root/'plan.json')
    complete=load(root/'COMPLETE.json')
    summary=load(root/'summary-final.json')
    assert sha(root/'summary-final.json')==complete['summary_sha256']
    assert complete['cases']==len(plan['cases'])==5456
    for name,digest in plan['source']['source_files'].items():
        assert sha(root/'frozen/src/clipp1d'/name)==digest
    for name,digest in plan['runner_files'].items():
        assert sha(root/'frozen/benchmarks'/name)==digest
    counts=Counter()
    records=[]
    verified=set()
    for i,case in enumerate(plan['cases']):
        directory=root/'runs'/case['dataset']/case['case_id']
        terminal=load(directory/'terminal.json')
        counts[terminal['status']]+=1
        for path,digest in [(case['input_path'],case['input_sha256']),
                            (case['original_input_path'],case['original_input_sha256']),
                            (case['truth_path'],case['truth_sha256']),*case['truth_sources'].items()]:
            if path not in verified:
                assert sha(path)==digest,path
                verified.add(path)
        record=dict(dataset=case['dataset'],case_id=case['case_id'],
                    retained_mutations=case['retained_mutations'],purity=case['purity'],
                    mean_depth=case['mean_depth'],status=terminal['status'],
                    truth_mixed_cn_mutations=case.get('truth_mixed_cn_mutations',0),
                    truth_primary_cn_mismatches=case.get('truth_primary_cn_mismatches',0))
        if terminal['status']=='success':
            run=load(directory/'fit/run.json')
            startup=load(directory/'startup.json')
            assert startup['plan_sha256']==plan_digest
            assert len(startup['controls']['selected_cpus'])==1
            assert set(startup['controls']['thread_environment'].values())=={'1'}
            assert run['provenance']['source_sha256']==plan['source']['source_sha256']
            assert run['provenance']['input_sha256']==case['input_sha256']
            assert run['schema']=='clipp1d.run.v4' and run['status']=='success'
            assert sha(directory/'fit/run.json')==terminal['run_sha256']
            assert sha(directory/'metrics.json')==terminal['metrics_sha256']
            for name,digest in run['table_sha256'].items():
                assert sha(directory/'fit'/name)==digest
            observed=read_rows(directory/'fit/mutation_clusters.tsv')
            mult=read_rows(directory/'fit/mutation_multiplicity.tsv')
            truth=read_rows(case['truth_path'])
            assert len(observed)==case['retained_mutations']
            assert set(observed)==set(mult)
            assert set(truth)<=set(observed)
            assert set(observed)-set(truth)==set(case.get('truth_unmatched_retained',[]))
            selected=run['candidate_provenance']
            reference=selected['raw_reference']
            assert selected['partition_sha256']==hashlib.sha256(np.asarray(run['partition_cuts'],dtype=np.int64).tobytes()).hexdigest()
            assert selected['chain_sha256']==run['chain_sha256']
            assert selected['model_sha256']==run['provenance']['model_sha256']
            assert run['selection_score']<=reference['refit_score']+1e-8
            if selected['candidate_family']=='direct_chain_partition':
                assert run['selected_lambda'] is None
                assert selected['selected_raw_certificate'] is None
                assert selected['selected_partition_certified'] is False
            raw=run['raw_diagnostics']
            assert raw['clonal_feasible'] and (raw.get('raw_branch_stationarity_qualified') or raw.get('separable_scalar_gap_qualified'))
            assert any(float(r['refitted_ccf'])==1 and r['cluster_label']=='0' for r in observed.values())
            recomputed=metrics({k:v for k,v in observed.items() if k in truth},truth,mult,'refitted_ccf')
            saved=load(directory/'metrics.json')
            for key in ['ari','rmse','ccc','selected_k','all_exact_one_fraction','cna_only_multiplicity']:
                assert recomputed[key]==saved['integrated'][key],(case['case_id'],key)
            record.update(fit_seconds=terminal['fit_seconds'],proposal_seconds=terminal['proposal_seconds'],
                          max_rss_bytes=terminal['max_rss_bytes'],search_status=terminal['search_status'],
                          selected_origin=terminal['selected_origin'],score_gain=terminal['score_gain'])
            for method in ['fusion_path','integrated']:
                for key in ['ari','mae','ccc','smf_error','selected_k','true_k','truth_coverage']:
                    record[method+'_'+key]=saved[method][key]
            record['ari_gain']=record['integrated_ari']-record['fusion_path_ari']
        records.append(record)
        if (i+1)%500==0:
            print('audited',i+1,flush=True)
    assert dict(counts)==complete['status']
    assert sum(c['terminal'] for c in summary['cohorts'].values())==5456
    for dataset,item in summary['cohorts'].items():
        subset=[r for r in records if r['dataset']==dataset]
        success=[r for r in subset if r['status']=='success']
        assert len(subset)==item['planned']
        assert dict(Counter(r['status'] for r in subset))==item['status']
        if success:
            for method in ['fusion_path','integrated']:
                nontrivial=[r for r in success if r[method+'_true_k']>1]
                mean=float(np.mean([r[method+'_ari'] for r in nontrivial])) if nontrivial else None
                assert mean is None or abs(mean-item[method]['mean_ari_multi'])<1e-12
                assert abs(float(np.mean([r[method+'_mae'] for r in success]))-item[method]['mean_mae'])<1e-12
    fields=sorted({key for r in records for key in r})
    with (root/'audited-case-metrics.tsv').open('x') as stream:
        writer=csv.DictWriter(stream,fields,delimiter='\t',lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    write(root/'final-audit.json',dict(status='passed',cases=len(records),terminal_status=dict(counts),
          verified_source_and_input_files=len(verified),finished_utc=datetime.now(timezone.utc).isoformat(),
          plan_sha256=plan_digest,summary_sha256=sha(root/'summary-final.json'),
          case_metrics_sha256=sha(root/'audited-case-metrics.tsv')))
    return plan,summary,records


def report(root,plan,summary,records):
    def fmt(value, digits=4):
        return 'NA' if value is None else f'{value:.{digits}f}'
    finished=datetime.now(ZoneInfo('America/Chicago')).strftime('%B %d, %Y, %I:%M %p %Z')
    lines=['# CliPP1.5 0.3.0: full single-region benchmark','',
           f'Audited {finished}. All {len(records):,} planned tumors have terminal receipts.',
           'The final audit verified source, input/truth/output identities, candidate provenance, public metrics and cohort counts.','',
           '| Cohort | Success / planned | ARI K>1: fusion → integrated | Mean CCF MAE: fusion → integrated | Median fit seconds |',
           '|---|---:|---:|---:|---:|']
    for name,c in summary['cohorts'].items():
        a,b=c.get('fusion_path',{}),c.get('integrated',{})
        lines.append(f"| {name} | {c['status'].get('success',0)} / {c['planned']} | {fmt(a.get('mean_ari_multi'))} → {fmt(b.get('mean_ari_multi'))} | {fmt(a.get('mean_mae'))} → {fmt(b.get('mean_mae'))} | {fmt(c.get('runtime_seconds',{}).get('median'),2)} |")
    lines+=['','## Multiplicity and single-cluster tumors','',
            '| Cohort | CNA macro-F1: fusion → integrated | CNA calls / eligible | False splits: fusion → integrated | True K=1 tumors |',
            '|---|---:|---:|---:|---:|']
    for name,c in summary['cohorts'].items():
        if 'integrated' not in c:
            continue
        a,b=c['fusion_path'],c['integrated']
        f=b['cna_multiplicity']
        lines.append(f"| {name} | {fmt(a['cna_multiplicity']['macro_f1'])} → {fmt(f['macro_f1'])} | {f['called']} / {f['eligible']} | {a['false_splits']} → {b['false_splits']} | {b['single_cluster_cases']} |")
    lines+=['','## Interpretation and limits','',
        'The comparison retains the original fusion-path winner inside each integrated fit. It isolates the added proposals on an identical fixed chain and raw search; it is not a separate historical-version runtime experiment.',
        '', 'The likelihood, score, clonal constraint and qualification tolerances are unchanged. Lower score does not guarantee higher truth ARI. Direct winners carry their own provenance and no inherited raw certificate.',
        '', 'SimClone uses separate corrected normal-CN2 inputs; original files are preserved. Six ambiguous truth-coordinate mutations in three tumors remain fitted but are not accuracy-scored. PhylogicNDT uses the requested TSVs as supplied, including the documented loss of some original CN-state information. These observation-model differences limit algorithm-only interpretations.',
        '', 'CNA means CN other than 1/1. Macro-F1 is pooled over exact dosage classes, with micro/weighted/per-class results in summary-final.json. ARI for true K=1 is separated from nontrivial ARI. CCC is 1 for identical constants and 0 for other constant-vector cases. Accuracy describes successful fits with recorded truth coverage; failures and timeouts remain in the planned denominator.',
        '', 'The pool initially used 24 CPU workers, then 25 at user request; 364 completed fits were validated and imported at the transition. Each fit used one CPU/thread on a shared host. Runtime is observed pool wall time, with a six-hour per-case bound, not exclusive-resource timing.',
        '', '## Evidence','',
        f"Source fingerprint: `{plan['source']['source_sha256']}`. The package is identical to release commit `c4f3d3e54ac94f1b6707463784ed9c85a86f698d`.",
        '', '[Audit](final-audit.json) · [Full summary](summary-final.json) · [Per-tumor metrics](audited-case-metrics.tsv) · [Run plan](plan.json)']
    with (root/'REPORT.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    while not (root/'COMPLETE.json').exists():
        if (root/'HALTED.json').exists():
            raise RuntimeError('Run halted; no final success report will be produced')
        if not args.wait:
            raise RuntimeError('Run is not complete')
        time.sleep(30)
    from benchmark_chain import configure_execution
    configure_execution(cpus=[max(os.sched_getaffinity(0))],threads=1)
    try:
        plan,summary,records=audit(root)
        report(root,plan,summary,records)
    except Exception as error:
        write(root/'finalization-failure.json',dict(error_type=type(error).__name__,message=str(error),
              finished_utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__=='__main__':
    main()
