from pathlib import Path
from io import StringIO
import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score

H = Path(__file__).resolve().parent
S = Path('/storage/CliPP2/CNfirst4K_CliPP15_performance_20260925T181236Z')
T = Path('/storage/CliPP2/CliPP1.5/results/cnfirst-pool14-20260925-v1/truth-v2')
D = pd.read_csv(S/'per_case_metrics.tsv', sep='\t')
B = D[(D.method=='CliPP1.5') & (D.population=='native')].set_index('case_id')
records=[]
clusters=[]
mutations=[]
for r in json.loads((S/'CLIPP15_RESULTS.json').read_text())['results']:
    if not r.get('tables'): continue
    cid=r['case']['case_id']
    m=pd.read_csv(StringIO(r['tables']['mutation_clusters.tsv']),sep='\t')
    t=pd.read_csv(T/(cid+'.tsv'),sep='\t')
    m=m.merge(t,on='mutation_id',validate='one_to_one')
    assert len(m)==len(t)==r['case']['retained_mutations']
    c=pd.read_csv(StringIO(r['tables']['cluster_centers.tsv']),sep='\t')
    assert len(c)==m.cluster_label.nunique()
    z=dict(B.loc[cid]); z.update(case_id=cid,family=r['family'])
    z['smf_bias']=z['estimated_smf']-z['true_smf']
    z['k_error']=z['selected_k']-z['true_k']
    z['overclustered']=z['k_error']>0
    z['smallest_cluster']=int(c.cluster_size.min())
    z['largest_cluster_fraction']=float(c.cluster_size.max()/len(m))
    z['singletons']=int((c.cluster_size==1).sum())
    z['clusters_le5']=int((c.cluster_size<=5).sum())
    z['min_ccf_gap']=float(np.diff(np.sort(c.refitted_ccf)).min()) if len(c)>1 else np.nan
    z['true_min_ccf_gap']=float(np.diff(np.sort(m.true_ccf.unique())).min()) if m.true_ccf.nunique()>1 else np.nan
    z['homogeneity']=homogeneity_score(m.true_cluster,m.cluster_label)
    z['completeness']=completeness_score(m.true_cluster,m.cluster_label)
    mat=pd.crosstab(m.true_cluster,m.cluster_label).to_numpy()
    choose2=lambda a: (a*(a-1)/2).sum()
    sameboth=choose2(mat)
    z['false_split_fraction']=float(1-sameboth/choose2(mat.sum(axis=1)))
    z['false_merge_fraction']=float(1-sameboth/choose2(mat.sum(axis=0)))
    for est in ['pilot_ccf','raw_ccf','refitted_ccf']:
        z[est+'_mae']=float((m[est]-m.true_ccf).abs().mean())
        z[est+'_bias']=float((m[est]-m.true_ccf).mean())
    for cn,mask in [('diploid',(m.major_cn==1)&(m.minor_cn==1)),('cna',(m.major_cn!=1)|(m.minor_cn!=1))]:
        z[cn+'_n']=int(mask.sum())
        z[cn+'_mae']=float((m.loc[mask,'refitted_ccf']-m.loc[mask,'true_ccf']).abs().mean())
        z[cn+'_bias']=float((m.loc[mask,'refitted_ccf']-m.loc[mask,'true_ccf']).mean())
        z[cn+'_ari']=adjusted_rand_score(m.loc[mask,'true_cluster'],m.loc[mask,'cluster_label']) if mask.any() else np.nan
    clonal=m.true_ccf>=1-1e-12
    z['true_clonal_n']=int(clonal.sum())
    z['clonal_recall']=float((m.loc[clonal,'cluster_label']==0).mean())
    z['clonal_precision']=float(clonal[m.cluster_label==0].mean())
    z['true_clonal_mae']=float((m.loc[clonal,'refitted_ccf']-1).abs().mean())
    z['true_subclonal_mae']=float((m.loc[~clonal,'refitted_ccf']-m.loc[~clonal,'true_ccf']).abs().mean())
    z['true_clonal_bias']=float((m.loc[clonal,'refitted_ccf']-1).mean())
    z['true_subclonal_bias']=float((m.loc[~clonal,'refitted_ccf']-m.loc[~clonal,'true_ccf']).mean())
    for tol in [1e-6,.01,.02,.05]:
        order=c.sort_values('refitted_ccf')
        group=np.r_[0,np.cumsum(np.diff(order.refitted_ccf)>tol)]
        mp=dict(zip(order.cluster_label,group))
        merged=m.cluster_label.map(mp)
        z[f'merge_{tol}_ari']=adjusted_rand_score(m.true_cluster,merged)
        z[f'merge_{tol}_k']=merged.nunique()
    for _,row in c.iterrows():
        g=m[m.cluster_label==row.cluster_label]
        n=g.true_cluster.value_counts()
        clusters.append(dict(case_id=cid,**row.to_dict(),purity=float(n.iloc[0]/len(g)),dominant_true_cluster=int(n.index[0]),dominant_true_ccf=float(g.loc[g.true_cluster==n.index[0],'true_ccf'].iloc[0]),cna_fraction=float(((g.major_cn!=1)|(g.minor_cn!=1)).mean()),truth_composition=json.dumps(n.to_dict())))
    m['case_id']=cid
    mutations.append(m)
    records.append(z)
df=pd.DataFrame(records).sort_values('case_id')
df.to_csv(H/'case_diagnostics.tsv',sep='\t',index=False)
pd.DataFrame(clusters).to_csv(H/'cluster_diagnostics.tsv',sep='\t',index=False)
md=pd.concat(mutations,ignore_index=True)
md.to_csv(H/'mutation_diagnostics.tsv',sep='\t',index=False)
summary=[]
for col in ['true_k','depth','purity','cna_rate','family','clipp15_search_status','k_error']:
    for val,g in df.groupby(col):
        entry=dict(stratification=col,value=val,cases=len(g))
        for metric in ['ari','selected_k','k_error','overclustered','smf_abs_error','smf_bias','ccf_mae','refitted_ccf_bias','homogeneity','completeness','false_split_fraction','false_merge_fraction','clonal_recall','clonal_precision','diploid_mae','cna_mae','true_clonal_mae','true_subclonal_mae','raw_ccf_mae','pilot_ccf_mae','singletons','clusters_le5']:
            entry[metric]=float(g[metric].mean())
        summary.append(entry)
pd.DataFrame(summary).to_csv(H/'strata.tsv',sep='\t',index=False)
print(pd.DataFrame(summary).to_string(index=False))
print('\nTOTAL',df.select_dtypes(include='number').mean().to_dict())
print('\nWORST',df.nsmallest(15,'ari')[['case_id','ari','true_k','selected_k','min_ccf_gap','true_min_ccf_gap','clonal_recall','clonal_precision','ccf_mae','smf_bias','clipp15_search_status']].to_string(index=False))
