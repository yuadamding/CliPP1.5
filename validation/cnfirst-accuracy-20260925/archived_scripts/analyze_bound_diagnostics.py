"""Reconcile public multiplicity calls and quantify aliases/search coverage."""
from pathlib import Path
from io import StringIO
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

H=Path(__file__).resolve().parent;B=H.parent
sys.path.insert(0,str(B/'CliPP1.5/results/cnfirst-pool14-20260925-v2/a100/payload/source/src'))
from clipp1d.io import read_tumor
from clipp1d.model import compile_model,posterior_multiplicity

def main():
    out=H/'bound_analysis';out.mkdir(exist_ok=True)
    rs=json.loads((H/'BOUND_DIAGNOSTICS.json').read_bytes())['records']
    assignments=pd.read_csv(H/'reassignment/assignments.tsv',sep='\t').set_index(['case_id','mutation_id'])
    mutations=[];cases=[]
    for r in rs:
        cid=r['case_id'];run=r['content']['run.json'];st=run['provenance']['numerical_stages']
        path=run['search'];qualified=[x for x in path if x.get('score') is not None and np.isfinite(x['score'])]
        assert min(x['score'] for x in qualified)==run['selection_score']
        cases.append(dict(case_id=cid,search_status=run['search_status'],selected_lambda=run['selected_lambda'],
                          qualified_scores=len(qualified),path_points=len(path),selected_k=next(x['clusters'] for x in qualified if x['score']==run['selection_score']),
                          selected_at_upper_boundary=st['selected_at_upper_boundary'],path_truncated=st['path_truncated'],
                          extensions=st['extensions'],raw_unresolved_penalties=st['raw_unresolved_penalties'],
                          refit_unresolved_penalties=st['refit_unresolved_penalties'],
                          path_min_k=min(x['clusters'] for x in qualified),path_max_k=max(x['clusters'] for x in qualified)))
        model=compile_model(read_tumor(B/'CliPP1.5/results/cnfirst-pool14-20260925-v1/inputs-v2'/f'{cid}.tsv'))
        m=pd.read_csv(StringIO(r['content']['mutation_multiplicity.tsv']),sep='\t').set_index('mutation_id').loc[list(model.mutation_ids)]
        truth=pd.read_csv(B/'CliPP1.5/results/cnfirst-pool14-20260925-v1/truth-v2'/f'{cid}.tsv',sep='\t').set_index('mutation_id').loc[list(model.mutation_ids)]
        a=assignments.loc[cid].loc[list(model.mutation_ids)]
        for estimator in ('raw','refitted'):
            call=posterior_multiplicity(model,m[estimator+'_ccf'].to_numpy()).argmax(axis=1)+1
            assert np.array_equal(call,m[estimator+'_multiplicity_call'])
        newcall=posterior_multiplicity(model,a.diagnostic_ccf.to_numpy()).argmax(axis=1)+1
        joined=truth.join(m[['raw_ccf','raw_multiplicity_call','refitted_ccf','refitted_multiplicity_call']])
        joined=joined.join(a[['original_label','diagnostic_label','diagnostic_ccf']])
        joined['diagnostic_multiplicity_call']=newcall;joined['case_id']=cid
        mutations.append(joined.reset_index())
    df=pd.concat(mutations,ignore_index=True)
    df.to_csv(out/'multiplicity_diagnostic.tsv',sep='\t',index=False)
    cs=pd.DataFrame(cases);cs.to_csv(out/'search_diagnostic.tsv',sep='\t',index=False)
    cna=(df.major_cn!=1)|(df.minor_cn!=1)
    clonal=cna&(df.true_ccf==1)
    wrong=clonal&(df.original_label!=0)
    summary=dict(cases=len(rs),public_multiplicity_calls_reproduced=len(df)*2,
                 selected_at_upper_boundary=int(cs.selected_at_upper_boundary.sum()),
                 path_truncated=int(cs.path_truncated.sum()),
                 paths_without_one_cluster_candidate=int((cs.path_min_k>1).sum()),
                 cn_clonal_wrong_label=int(wrong.sum()),
                 cn_clonal_wrong_label_overcalled_m=int((wrong&(df.refitted_multiplicity_call>df.true_multiplicity)).sum()),
                 cn_clonal_wrong_label_correct_m=int((wrong&(df.refitted_multiplicity_call==df.true_multiplicity)).sum()))
    for name,ccf,mc in [('original','refitted_ccf','refitted_multiplicity_call'),('diagnostic','diagnostic_ccf','diagnostic_multiplicity_call')]:
        s=df[cna]
        summary[name]=dict(cna_macro_f1=float(f1_score(s.true_multiplicity,s[mc],labels=[1,2,3,4],average='macro')),
                          cna_micro_f1=float(f1_score(s.true_multiplicity,s[mc],labels=[1,2,3,4],average='micro')),
                          cna_m_overcalled=int((s[mc]>s.true_multiplicity).sum()),
                          cna_m_undercalled=int((s[mc]<s.true_multiplicity).sum()),
                          clonal_cna_ccf_bias=float((df.loc[clonal,ccf]-1).mean()),
                          clonal_cna_dosage_relative_error=float(np.abs(df.loc[clonal,ccf]*df.loc[clonal,mc]/df.loc[clonal,'true_multiplicity']-1).mean()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
