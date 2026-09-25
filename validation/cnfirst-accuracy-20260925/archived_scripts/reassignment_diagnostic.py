"""Offline diagnostic: exact score-descent moves at published, fixed centers.

This neither fits a fusion path nor produces a production result/certificate.
Truth is loaded only to evaluate the resulting partitions.
"""
from pathlib import Path
from io import StringIO
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.metrics import adjusted_rand_score

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
SOURCE=BASE/'CliPP1.5/results/cnfirst-pool14-20260925-v2/a100/payload/source'
sys.path.insert(0,str(SOURCE/'src'))
from clipp1d.io import read_tumor
from clipp1d.model import compile_model,loss

def penalty(sizes):
    sizes=np.asarray(sizes);sizes=sizes[sizes>0]
    n=sizes.sum();k=len(sizes)
    return float(k*np.log(n)-1.4*(gammaln(k)-gammaln(n+k)+gammaln(sizes+1).sum()+gammaln(k+1)))

def evaluate(y,centers,truth):
    occupied=np.unique(y)
    clonal=min(occupied,key=lambda k:(abs(1-centers[k]),k))
    return dict(ari=float(adjusted_rand_score(truth.true_cluster,y)),k=len(occupied),
                ccf_mae=float(np.abs(centers[y]-truth.true_ccf).mean()),
                estimated_smf=float(np.mean(y!=clonal)))

def main():
    out=HERE/'reassignment';out.mkdir(exist_ok=True)
    records=json.loads((BASE/'CNfirst4K_CliPP15_performance_20260925T181236Z/CLIPP15_RESULTS.json').read_bytes())['results']
    diagnostics={r['case_id']:r['content']['run.json'] for r in json.loads((HERE/'BOUND_DIAGNOSTICS.json').read_bytes())['records']}
    table=[];assignments=[]
    for r in records:
        if 'validated' not in r:continue
        cid=r['case']['case_id']
        path=BASE/'CliPP1.5/results/cnfirst-pool14-20260925-v1/inputs-v2'/f'{cid}.tsv'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['case']['input_sha256']
        model=compile_model(read_tumor(path))
        calls=pd.read_csv(StringIO(r['tables']['mutation_clusters.tsv']),sep='\t').set_index('mutation_id').loc[list(model.mutation_ids)]
        centers=pd.read_csv(StringIO(r['tables']['cluster_centers.tsv']),sep='\t').sort_values('cluster_label').refitted_ccf.to_numpy()
        y=calls.cluster_label.to_numpy().copy();before=y.copy();n=len(y)
        assert np.all(centers>=model.lower.max()) and np.all(centers<=model.upper.min())
        costs=np.column_stack([loss(model,np.full(n,c)) for c in centers])
        sizes=np.bincount(y,minlength=len(centers))
        original=2*costs[np.arange(n),y].sum()+penalty(sizes)
        assert abs(original-diagnostics[cid]['selection_score'])<1e-6
        trace=[]
        for step in range(10000):
            occupied=sizes>0;k=int(occupied.sum())
            if k==1:break
            delta=2*(costs-costs[np.arange(n),y,None])+1.4*np.log(sizes[y,None]/(sizes[None,:]+1))
            singleton=sizes[y]==1
            if singleton.any():
                base=lambda kk: kk*np.log(n)-1.4*(gammaln(kk)-gammaln(n+kk)+gammaln(kk+1))
                delta[singleton,:]+=base(k-1)-base(k)
            delta[:,~occupied]=np.inf
            delta[np.arange(n),y]=np.inf
            i,dest=np.unravel_index(np.argmin(delta),delta.shape)
            if delta[i,dest]>=-1e-8:break
            previous=2*costs[np.arange(n),y].sum()+penalty(sizes)
            src=y[i];sizes[src]-=1;sizes[dest]+=1;y[i]=dest
            updated=2*costs[np.arange(n),y].sum()+penalty(sizes)
            assert abs(updated-previous-delta[i,dest])<1e-7
            assert updated<previous
            trace.append(float(updated))
        else:raise AssertionError('move limit reached')
        truth=pd.read_csv(BASE/'CliPP1.5/results/cnfirst-pool14-20260925-v1/truth-v2'/f'{cid}.tsv',sep='\t').set_index('mutation_id').loc[list(model.mutation_ids)]
        final=2*costs[np.arange(n),y].sum()+penalty(sizes)
        row=dict(case_id=cid,search_status=diagnostics[cid]['search_status'],n=n,moves=len(trace),
                 original_score=float(original),diagnostic_score=float(final),score_improvement=float(original-final),
                 changed_mutations=int((y!=before).sum()),true_smf=float((truth.true_ccf<1).mean()))
        for prefix,labels in [('original',before),('diagnostic',y)]:
            row.update({prefix+'_'+k:v for k,v in evaluate(labels,centers,truth).items()})
        table.append(row)
        assignments.extend(dict(case_id=cid,mutation_id=mid,original_label=int(a),diagnostic_label=int(b),diagnostic_ccf=float(centers[b]))
                           for mid,a,b in zip(model.mutation_ids,before,y))
    df=pd.DataFrame(table);df.to_csv(out/'per_case.tsv',sep='\t',index=False)
    pd.DataFrame(assignments).to_csv(out/'assignments.tsv',sep='\t',index=False)
    def ccc(x,y):return float(2*np.mean((x-x.mean())*(y-y.mean()))/(x.var(ddof=0)+y.var(ddof=0)+(x.mean()-y.mean())**2))
    summary=dict(cases=len(df),lower_score_cases=int((df.score_improvement>1e-6).sum()),
                 lower_score_complete=int(((df.score_improvement>1e-6)&(df.search_status=='complete')).sum()),
                 changed_mutations=int(df.changed_mutations.sum()),moves=int(df.moves.sum()),
                 ari_improved=int((df.diagnostic_ari>df.original_ari+1e-10).sum()),
                 ari_worsened=int((df.diagnostic_ari<df.original_ari-1e-10).sum()),
                 mean_score_improvement=float(df.score_improvement.mean()),
                 median_score_improvement=float(df.score_improvement.median()))
    for prefix in ('original','diagnostic'):
        summary[prefix]=dict(mean_ari=float(df[prefix+'_ari'].mean()),mean_k=float(df[prefix+'_k'].mean()),
                             mean_ccf_mae=float(df[prefix+'_ccf_mae'].mean()),
                             smf_ccc=ccc(df.true_smf,df[prefix+'_estimated_smf']))
    summary['limitations']='Fixed published centers; exact score-descent membership diagnostic, no scalar refit, no raw solver certificate, no truth used in moves; selected early193 cases.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
