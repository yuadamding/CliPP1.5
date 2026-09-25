"""Separate exact known-truth scores from approximate oracle-membership refits."""
import json
import math
import numpy as np
import pandas as pd
from scipy.special import gammaln
from diagnose import H, e, INPUT, compile_model, read_tumor, loss, sha


def components(nll, sizes):
    sizes = np.asarray(sizes)
    n, k = sizes.sum(), len(sizes)
    allocation = -1.4*(gammaln(k)-gammaln(n+k)+gammaln(sizes+1).sum()+gammaln(k+1))
    return dict(two_nll=float(2*nll), klogn=float(k*np.log(n)), allocation_penalty=float(allocation),
                score=float(2*nll+k*np.log(n)+allocation))


def main():
    rows=[]
    for path in sorted((H/'all').glob('*.json')):
        if path.name=='SUMMARY.json': continue
        record=json.loads(path.read_text()); cid=record['case_id']
        model=compile_model(read_tumor(INPUT/(cid+'.tsv')))
        t=e.table(e.COHORT/cid/'truth.txt')
        t.index=['chr'+e.mid(ch,pos) for ch,pos in zip(t.chromosome_index,t.position)]
        t=t.loc[list(model.mutation_ids)]
        centers=np.sort(t.ccf.unique()); true_k=len(centers)
        known_nll=float(loss(model,t.ccf.to_numpy()).sum())
        known=components(known_nll,t.groupby('cluster_id').size().to_numpy())
        candidates={r['candidate']:r for r in record['rows']}
        published=candidates['published']; oracle=candidates['truth_partition_approx_refit']
        delta=oracle['score']-published['score']
        row=dict(case_id=cid,depth=int(e.META[cid]['read_depth']),purity=float(e.META[cid]['purity']),
                 cna_rate=float(e.META[cid]['cna_rate']),n=len(t),true_k=true_k,selected_k=published['k'],
                 selected_count='under' if published['k']<true_k else 'over' if published['k']>true_k else 'equal',
                 true_min_gap=float(np.diff(centers).min()) if true_k>1 else math.nan,
                 truth_centers=','.join(map(str,centers)),selected_centers=','.join(map(str,published['centers'])),
                 true_sizes=','.join(map(str,oracle['sizes'])),selected_sizes=','.join(map(str,published['sizes'])),
                 oracle_score_delta=delta,known_truth_score_delta=known['score']-published['score'],
                 oracle_relation='lower' if delta < -1e-4 else 'higher' if delta>1e-4 else 'equal',
                 published_ari=published['ari'],search_status=published['search_status'])
        for name,c in [('published',components(published['nll'],published['sizes'])),
                       ('oracle_approx',components(oracle['nll'],oracle['sizes'])),('known_truth',known)]:
            row.update({name+'_'+k:v for k,v in c.items()})
        rows.append(row)
    frame=pd.DataFrame(rows); frame.to_csv(H/'score_limits_per_case.tsv',sep='\t',index=False)
    multi=frame[frame.true_k>1].copy()
    multi['true_gap_bin']=pd.cut(multi.true_min_gap,[0,.1,.2,.3,.5,1.],include_lowest=True).astype(str)
    strata=[]
    for col in ['depth','true_k','true_gap_bin','selected_count']:
        for val,g in multi.groupby(col):
            strata.append(dict(stratum=col,value=str(val),cases=len(g),
                               oracle_lower=int((g.oracle_relation=='lower').sum()),
                               oracle_equal=int((g.oracle_relation=='equal').sum()),
                               oracle_higher=int((g.oracle_relation=='higher').sum()),
                               mean_published_ari=float(g.published_ari.mean()),
                               mean_true_min_gap=float(g.true_min_gap.mean())))
    pd.DataFrame(strata).to_csv(H/'score_limits_strata.tsv',sep='\t',index=False)
    higher=multi[multi.oracle_relation=='higher']
    summary=dict(multi_cases=len(multi),oracle_relation=multi.oracle_relation.value_counts().to_dict(),
                 higher_selected_count=higher.selected_count.value_counts().to_dict(),
                 higher_gap=dict(minimum=float(higher.true_min_gap.min()),median=float(higher.true_min_gap.median()),
                                 maximum=float(higher.true_min_gap.max())),
                 known_truth_scope='Exact frozen-model likelihood evaluated at known true CCFs; feasible points, not optimized scores.',
                 oracle_scope='1025-grid/local-minima approximate refit of known memberships; not globally certified.',
                 script_sha256=sha(__file__))
    (H/'score_limits_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));print(higher[higher.selected_count=='under'].sort_values('oracle_score_delta',ascending=False).head(5).to_string(index=False))


if __name__=='__main__': main()
