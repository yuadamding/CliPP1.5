"""Publication-style plots from the fixed 193-case diagnostic export."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

H=Path(__file__).resolve().parent
D=pd.read_csv(H/'case_diagnostics.tsv',sep='\t')
M=pd.read_csv(H/'mutation_diagnostics.tsv',sep='\t')
C=pd.read_csv(H/'cluster_diagnostics.tsv',sep='\t')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
    'axes.labelsize':11,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
    'axes.spines.right':False,'savefig.dpi':220})
colors=['#246B8E','#C37B2F','#883D66']
figures=[]

fig,axes=plt.subplots(2,3,figsize=(11.5,7.2),sharex=True,sharey='row')
rng=np.random.default_rng(9025)
for j,depth in enumerate([100,200,500]):
    dd=D[D.depth==depth]
    for row,(metric,label) in enumerate([('ari','Adjusted Rand index'),('ccf_mae','Mean absolute CCF error')]):
        ax=axes[row,j]
        values=[dd.loc[dd.cna_rate==rate,metric].to_numpy() for rate in [.1,.4,.7]]
        boxes=ax.boxplot(values,positions=range(3),widths=.55,patch_artist=True,
            showfliers=False,medianprops={'color':'black','linewidth':1.5},
            whiskerprops={'color':'#555555'},capprops={'color':'#555555'})
        for p,col in zip(boxes['boxes'],colors): p.set_facecolor(col);p.set_alpha(.30)
        for k,(v,col) in enumerate(zip(values,colors)):
            ax.scatter(k+rng.uniform(-.15,.15,len(v)),v,s=11,color=col,alpha=.52,edgecolors='none')
            if row==0: ax.text(k,1.035,f'n={len(v)}',ha='center',va='bottom',fontsize=8)
        ax.set_xticks(range(3),['0.1','0.4','0.7'])
        ax.set_xlim(-.55,2.55)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        if j==0: ax.set_ylabel(label)
        if row==0: ax.set_title(f'Read depth {depth}');ax.set_ylim(-.05,1.12)
        else: ax.set_xlabel('Simulated CNA fraction');ax.set_ylim(0,.36)
fig.suptitle('Accuracy declines as CNA burden increases, across read depths',fontsize=15,y=.99)
fig.text(.5,.025,'193 finished tumors · 200–232 mutations per tumor · final-refit CCF · boxes: median and IQR; whiskers: 1.5 IQR',ha='center',fontsize=9,color='#444444')
fig.tight_layout(rect=[0,.05,1,.95])
figures.append(('cna_depth_accuracy',fig))

m=M[(M.true_ccf>=1-1e-12)&((M.major_cn!=1)|(M.minor_cn!=1))].copy()
m['correct']=m.cluster_label==0
z=m.groupby(['major_cn','true_multiplicity']).correct.agg(['mean','size'])
matrix=np.full((4,4),np.nan)
for (major,mult),v in z.iterrows():matrix[int(major)-1,int(mult)-1]=v['mean']
fig,ax=plt.subplots(figsize=(8.0,6.4))
cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
im=ax.imshow(matrix,vmin=0,vmax=1,cmap=cmap,aspect='equal')
for i in range(4):
    for j in range(4):
        if np.isfinite(matrix[i,j]):
            v=z.loc[(i+1,j+1)]
            ax.text(j,i,f'{v["mean"]:.1%}\nn={int(v["size"]):,}',ha='center',va='center',
                color='black' if v['mean']>.68 else 'white',fontsize=12)
        else: ax.text(j,i,'—',ha='center',va='center',color='#999999',fontsize=15)
ax.set_xticks(range(4),['1','2','3','4']);ax.set_yticks(range(4),['1','2','3','4'])
ax.set_xlabel('True mutation multiplicity');ax.set_ylabel('Major copy number')
ax.set_title('Clonal CNA loci are lost when multiplicity is below major CN',pad=17)
cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04);cb.set_label('Fraction assigned to clonal label 0')
fig.text(.5,.035,'11,292 truly clonal CNA loci · CCF truth = 1\nCNA includes balanced amplification; unsupported multiplicities are gray.',ha='center',fontsize=9,color='#444444')
fig.tight_layout(rect=[0,.08,1,1])
figures.append(('clonal_multiplicity_recall',fig))

case='500_2_0.9_0.7_rep74'
m=M[M.case_id==case].copy();c=C[C.case_id==case].sort_values('cluster_label')
m['cna']=(m.major_cn!=1)|(m.minor_cn!=1)
labels=c.cluster_label.to_numpy();positions=np.arange(len(labels))
truth=sorted(m.true_cluster.unique())
counts=pd.crosstab(m.true_cluster,m.cluster_label).reindex(index=truth,columns=labels,fill_value=0)
fig=plt.figure(figsize=(13.6,7.1));gs=fig.add_gridspec(2,1,height_ratios=[1.35,1.45],hspace=.14)
ax=fig.add_subplot(gs[0]);im=ax.imshow(counts.to_numpy(),cmap='Blues',vmin=0,vmax=95,aspect='auto')
for i in range(len(truth)):
    for j in range(len(labels)):
        n=counts.iloc[i,j]
        ax.text(j,i,str(n),ha='center',va='center',fontsize=10,color='white' if n>50 else '#18334F')
truthccf=m.groupby('true_cluster').true_ccf.first()
ax.set_yticks(range(len(truth)),[f'True clone {v}\nCCF {truthccf[v]:.3f}' for v in truth])
ax.set_xticks(positions,[]);ax.tick_params(axis='x',length=0)
ax.set_ylabel('Truth membership')
ax.set_title('Counts of mutations: large low-CCF clusters contain truly clonal mutations',fontsize=12,pad=12)
ax2=fig.add_subplot(gs[1],sharex=ax)
cn=m.groupby(['cluster_label','cna']).size().unstack(fill_value=0).reindex(labels,fill_value=0)
dip=cn.get(False,pd.Series(0,index=labels)).to_numpy();cna=cn.get(True,pd.Series(0,index=labels)).to_numpy()
ax2.bar(positions,dip,color='#5B9BB6',label='Diploid (1/1)',width=.76)
ax2.bar(positions,cna,bottom=dip,color='#BC6955',label='CNA',width=.76)
ax2.set_ylim(0,108);ax2.set_ylabel('Mutations per inferred cluster')
ax2.set_xticks(positions,[f'{int(lab)}\n{ccf:.3f}' for lab,ccf in zip(labels,c.refitted_ccf)],fontsize=8)
ax2.set_xlabel('Inferred cluster label / final-refit CCF')
ax2.grid(axis='y',alpha=.18);ax2.set_axisbelow(True);ax2.legend(frameon=False,loc='upper right')
for j,n in enumerate(dip+cna):ax2.text(j,n+1.8,str(n),ha='center',fontsize=8)
fig.suptitle(f'{case}: true K = 2, inferred K = 19, ARI = 0.150',fontsize=15,y=.985)
fig.text(.5,.012,'Complete search · 230 mutations · blue matrix: truth counts · bars: observed copy-number composition',ha='center',fontsize=9,color='#444444')
fig.subplots_adjust(left=.092,right=.99,bottom=.12,top=.87)
figures.append(('representative_membership_contingency',fig))

Q=pd.read_csv(H.parent/'reassignment/per_case.tsv',sep='\t')
fig,axes=plt.subplots(2,2,figsize=(10.2,9.5))
for ax,metric,label,lim,subtitle in zip(axes[0],['ari','ccf_mae'],['ARI','CCF MAE'],[(0,1),(0,.30)],['Mean ARI: 0.485 → 0.640','Mean CCF MAE: 0.0887 → 0.0593']):
    ax.scatter(Q['original_'+metric],Q['diagnostic_'+metric],s=23,alpha=.65,color='#246B8E',linewidths=0)
    ax.plot(lim,lim,linestyle='--',color='#777777',linewidth=1)
    ax.set(xlim=lim,ylim=lim,xlabel=f'Published {label}',ylabel=f'Diagnostic {label}',title=subtitle)
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.16)
for ax,prefix,col,title in zip(axes[1],['original','diagnostic'],['#A86B5A','#246B8E'],['Published sMF · CCC = 0.783','Exact-score diagnostic sMF · CCC = 0.874']):
    ax.scatter(Q.true_smf,Q[prefix+'_estimated_smf'],s=23,alpha=.65,color=col,linewidths=0)
    ax.plot([0,1],[0,1],linestyle='--',color='#777777',linewidth=1)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='True sMF',ylabel='Estimated sMF',title=title)
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.16)
fig.suptitle('Changing membership improves the existing score and accuracy',fontsize=15,y=.985)
fig.text(.5,.047,'193 identical tumors and mutation sets · 152 improve ARI; 0 worsen · mean occupied K: 6.54 → 2.47',ha='center',fontsize=10,color='#333333')
fig.text(.5,.024,'Unchanged production score · frozen published CCF centers · no truth used in moves · no refits or inherited fit certification',ha='center',fontsize=9,color='#444444')
fig.tight_layout(rect=[0,.07,1,.95])
figures.append(('exact_score_reassignment',fig))

with PdfPages(H/'CNfirst4K_CliPP15_error_diagnostics.pdf') as pdf:
    for name,fig in figures:
        pdf.savefig(fig,bbox_inches='tight')
        fig.savefig(H/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(H/f'{name}.png',bbox_inches='tight')
        plt.close(fig)
print(f'Saved {len(figures)} figures and combined PDF under {H}')
