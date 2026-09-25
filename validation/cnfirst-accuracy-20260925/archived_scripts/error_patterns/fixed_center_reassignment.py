"""One-pass assignments to frozen output centers; never refits or uses truth to assign."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

H=Path(__file__).resolve().parent
R=Path('/storage/CliPP2/CliPP1.5/results/cnfirst-pool14-20260925-v1')
sys.path.insert(0,str(R/'source/src'))
from clipp1d.io import read_tumor
from clipp1d.model import compile_model, loss_at_rows

D=pd.read_csv(H/'case_diagnostics.tsv',sep='\t').set_index('case_id')
M=pd.read_csv(H/'mutation_diagnostics.tsv',sep='\t')
C=pd.read_csv(H/'cluster_diagnostics.tsv',sep='\t')
records=[]
exports=[]
for cid,original in M.groupby('case_id',sort=True):
    model=compile_model(read_tumor(R/'inputs-v2'/f'{cid}.tsv'))
    m=original.set_index('mutation_id').loc[list(model.mutation_ids)]
    c=C[C.case_id==cid].sort_values('cluster_label')
    centers=c.refitted_ccf.to_numpy()
    labels=c.cluster_label.to_numpy()
    sizes=c.cluster_size.to_numpy()
    rows=np.repeat(np.arange(len(model)),len(c))
    points=np.tile(centers,len(model))
    ll=-loss_at_rows(model,rows,points).reshape(len(model),len(c))
    feasible=(centers[None,:]>=model.lower[:,None])&(centers[None,:]<=model.upper[:,None])
    ll[~feasible]=-np.inf
    assert np.isfinite(ll).any(axis=1).all()
    original_idx=np.array([np.flatnonzero(labels==v)[0] for v in m.cluster_label])
    for mode,prior in [('original',None),('likelihood_only',np.zeros(len(c))),('fixed_size_prior',np.log(sizes/sizes.sum())),('fixed_size_plus_one_prior',np.log((sizes+1)/(sizes.sum()+len(c))))]:
        idx=original_idx if mode=='original' else np.argmax(ll+prior[None,:],axis=1)
        assigned=labels[idx]
        estimate=centers[idx]
        occupied=np.unique(idx)
        # Closest-to-one among occupied frozen centers. No CCF update occurs.
        clonal=occupied[np.argmin(np.abs(centers[occupied]-1))]
        smf=float(np.mean(idx!=clonal))
        records.append(dict(case_id=cid,mode=mode,n=len(m),ari=adjusted_rand_score(m.true_cluster,assigned),
            selected_k=len(occupied),original_k=len(c),true_k=m.true_cluster.nunique(),
            changed=int((idx!=original_idx).sum()),ccf_mae=float(np.mean(np.abs(estimate-m.true_ccf))),
            true_smf=float(np.mean(m.true_ccf<1-1e-12)),estimated_smf=smf,
            smf_abs_error=abs(smf-float(np.mean(m.true_ccf<1-1e-12))),
            complete=D.loc[cid,'clipp15_search_status']=='complete',
            unpenalized_loglikelihood=float(ll[np.arange(len(m)),idx].sum())))
        if cid=='500_2_0.9_0.7_rep74':
            for j,mid in enumerate(m.index):
                exports.append(dict(case_id=cid,mutation_id=mid,mode=mode,assigned_label=int(assigned[j]),refitted_ccf=float(estimate[j])))
d=pd.DataFrame(records)
d.to_csv(H/'fixed_center_reassignment.tsv',sep='\t',index=False)
pd.DataFrame(exports).to_csv(H/'representative_reassignment.tsv',sep='\t',index=False)
summary=[]
base=d[d['mode']=='original'].set_index('case_id')
for mode,g in d.groupby('mode'):
    g=g.set_index('case_id').loc[base.index]
    delta=g.ari-base.ari
    tx=g.true_smf.to_numpy();py=g.estimated_smf.to_numpy()
    ccc=2*np.mean((tx-tx.mean())*(py-py.mean()))/(tx.var()+py.var()+(tx.mean()-py.mean())**2)
    summary.append(dict(mode=mode,cases=len(g),mean_ari=float(g.ari.mean()),multi_ari=float(g.loc[g.true_k>1,'ari'].mean()),
        mean_k=float(g.selected_k.mean()),mean_ccf_mae=float(g.ccf_mae.mean()),smf_mae=float(g.smf_abs_error.mean()),smf_ccc=float(ccc),
        changed=int(g.changed.sum()),wins=int((delta>1e-10).sum()),losses=int((delta< -1e-10).sum()),ties=int((np.abs(delta)<=1e-10).sum()),
        true_k_one_false_splits=int(((g.true_k==1)&(g.selected_k>1)).sum())))
payload=dict(method='Single assignment pass using existing final-refit centers and frozen exact likelihood. No truth in assignment; no center refits, iteration, score selection, tolerance changes, or new candidates.',
    source=str(R/'source/src/clipp1d/model.py'),model_sha256=hashlib.sha256((R/'source/src/clipp1d/model.py').read_bytes()).hexdigest(),results=summary)
(H/'fixed_center_reassignment_summary.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
print(d[d.case_id=='500_2_0.9_0.7_rep74'].to_string(index=False))
