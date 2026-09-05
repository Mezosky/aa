#!/usr/bin/env python
"""Compact appendix panels and tables from the matched final-state analysis."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from geometry_llm.paper_tables import style_table

ROOTS={'Llama':Path('outputs/265b7bb1723b'),'Qwen':Path('outputs/86cc2e353f23')}
OUT=Path('outputs/final_geometry_report'); TEX=Path('paper/generated')
PRIMARY=['frozen','residual','lora','joint']
LABELS={'frozen':'Frozen','residual':'Residual','lora':'LoRA','joint':r'$\Delta$ + LoRA',
        'random':'Random','permuted':'Permuted','joint_residual_off':r'Joint, $\Delta$ off','joint_lora_off':'Joint, LoRA off'}
COLORS={'frozen':'#89969E','residual':'#008C9B','lora':'#C18434','joint':'#7562A5','random':'#AFBAC0','permuted':'#667580'}


def save(fig,name):
    fig.tight_layout(pad=.35)
    for ext in ['pdf','png']: fig.savefig(OUT/f'{name}.{ext}',dpi=260,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#E8EDF0',lw=.6); ax.set_axisbelow(True)


def tabular(name,columns,header,rows,caption,label):
    text=(r'\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}'+'\n'+
          r'\begin{tabular}{'+columns+r'}\toprule'+'\n'+header+r'\\\midrule'+'\n'+
          '\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}'+'\n'+r'\caption{'+caption+'}'+r'\label{'+label+r'}\end{table}'+'\n')
    (TEX/f'{name}.tex').write_text(style_table(text))


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,
                         'ytick.labelsize':7,'axes.edgecolor':'#ADB8BE','axes.linewidth':.6,'pdf.fonttype':42})
    reports={m:json.loads((r/'final_geometry/seed-13/analysis.json').read_text()) for m,r in ROOTS.items()}
    shape_rows=[]; label_rows=[]; probe_rows=[]; norm_rows=[]; norm_reports={}
    chains=[json.loads(l) for l in (ROOTS['Llama']/'selected_chains.jsonl').open()]
    relations=np.array([c['fact_comp_type'] for c in chains]); kinds,counts=np.unique(relations,return_counts=True)
    top=kinds[np.argsort(-counts)[:4]]; palette=['#008C9B','#C18434','#7562A5','#D35D6E']
    point_colors=np.array([palette[list(top).index(r)] if r in top else '#CBD2D6' for r in relations])
    for model,root in ROOTS.items():
        report=reports[model]; stem=model.lower(); path=root/'final_geometry/seed-13'
        norm_report=json.loads((path/'angles_norms.json').read_text())
        norm_reports[model]=norm_report
        for condition in report['conditions']:
            m=report['conditions'][condition]
            shade='llamarow' if model=='Llama' else 'qwenrow'
            shape_rows.append(r'\rowcolor{'+shade+'}'+f"{model} & {LABELS[condition]} & {m['spectrum']['effective_rank']:.1f} & {m['spectrum']['rank90']} & "
                              f"{m['cone']['mean']:.1f} & {m['rotation']['mean']:.1f} & {m['neighbor_retention']['mean']:.2f} & {m['cka_to_frozen']:.3f}"+r' \\')
            z=norm_report['conditions'][condition]['final']; phi=z['update_to_base_angle_deg']['mean']
            phi_text=f'{phi:.1f}' if phi is not None else 'n/a'
            norm_rows.append(f"{model} & {LABELS[condition]} & {z['state_norm']['mean']:.1f} & {z['norm_ratio']['mean']:.3f} & "
                             f"{z['relative_step']['mean']:.3f} & {phi_text} & {z['radial_relative_step']['mean']:+.3f} & "
                             f"{z['tangential_relative_step']['mean']:.3f} & {z['update_to_base_angle_deg']['defined_n']}"+r' \\')
            if condition in PRIMARY+['random','permuted']:
                relation=report['label_tests']['relation'][condition]['cka']
                answer=report['label_tests']['answer'][condition]
                label_rows.append(f"{model} & {LABELS[condition]} & {relation['observed']:.3f} & {answer['cka']['excess']:.4f} & "
                                  f"{answer['cka']['q']:.3f} & {100*answer['purity']['excess']:.2f} & {answer['purity']['q']:.3f}"+r' \\')
        fig,ax=plt.subplots(figsize=(2.65,1.85))
        for c in PRIMARY:
            values=report['conditions'][c]['spectrum']['cumulative']
            ax.plot(np.arange(1,len(values)+1),values,color=COLORS[c],lw=1.3,label=LABELS[c])
        ax.axhline(.9,color='#89969E',lw=.6,ls=':')
        ax.set_xscale('log'); ax.set_xlim(1,533); ax.set_ylim(0,1.02)
        ax.set_xlabel('Number of principal directions'); ax.set_ylabel('Cumulative variance')
        ax.legend(frameon=False,fontsize=6.5,ncol=2,loc='lower right',handlelength=1.1,columnspacing=.7)
        clean(ax); save(fig,f'{stem}_spectrum')
        fig,ax=plt.subplots(figsize=(2.65,1.85))
        for c in ['residual','lora','joint','random','permuted']:
            rows=[r for r in report['conditions'][c]['profiles'] if r['site']=='final']
            x=[r['layer'] for r in rows]; means=[r['rotation']['mean'] for r in rows]
            bounds=np.array([r['rotation']['ci95'] for r in rows])
            ax.plot(x,means,color=COLORS[c],lw=1.15,ls='--' if c=='random' else ':' if c=='permuted' else '-',label=LABELS[c])
            if c in PRIMARY: ax.fill_between(x,bounds[:,0],bounds[:,1],color=COLORS[c],alpha=.13,lw=0)
        ax.set_xlabel('Transformer layer'); ax.set_ylabel('Rotation from frozen (°)'); ax.set_ylim(0,85)
        ax.legend(frameon=False,fontsize=6,ncol=2,loc='upper left',handlelength=1.1,columnspacing=.7)
        clean(ax); save(fig,f'{stem}_rotation')
        fig,ax=plt.subplots(figsize=(2.65,1.85))
        for c in ['residual','lora','joint']:
            metrics=[norm_report['conditions'][c]['final'][f'layer_{l}_norm_ratio'] for l in report['layers']]
            means=np.array([m['mean'] for m in metrics]); bounds=np.array([m['ci95'] for m in metrics])
            ax.plot(report['layers'],means,color=COLORS[c],lw=1.2,label=LABELS[c])
            ax.fill_between(report['layers'],bounds[:,0],bounds[:,1],color=COLORS[c],alpha=.13,lw=0)
        ax.axhline(1,color='#89969E',ls=':',lw=.7); ax.set_xlabel('Transformer layer'); ax.set_ylabel('Adapted / frozen norm')
        ax.legend(frameon=False,fontsize=6.5,ncol=3,loc='lower left',bbox_to_anchor=(0,1.01),handlelength=1,
                  handletextpad=.25,columnspacing=.8,borderaxespad=0)
        clean(ax); save(fig,f'{stem}_norm_layers')
        fig,ax=plt.subplots(figsize=(2.65,1.85))
        for field,offset,label,color in [('radial_relative_step',-.17,'Radial','#788994'),
                                         ('tangential_relative_step',.17,'Tangential','#008C9B')]:
            metrics=[norm_report['conditions'][c]['final'][field] for c in ['residual','lora','joint']]
            means=np.array([m['mean'] for m in metrics]); bounds=np.array([m['ci95'] for m in metrics])
            ax.bar(np.arange(3)+offset,means,width=.3,color=color,label=label)
            ax.errorbar(np.arange(3)+offset,means,yerr=[means-bounds[:,0],bounds[:,1]-means],fmt='none',ecolor='#243D48',lw=.65,capsize=2)
        ax.axhline(0,color='#89969E',lw=.7); ax.set_xticks(range(3),[LABELS[c] for c in ['residual','lora','joint']])
        ax.set_ylabel('Step / frozen state norm'); ax.set_ylim(-.85,1.15)
        ax.legend(frameon=False,fontsize=7,ncol=2,loc='upper left',handlelength=1,columnspacing=.8)
        clean(ax); save(fig,f'{stem}_radial_tangential')
        angles=norm_report['displacement_angles']; matrix=np.array(angles['matrix']); labels=[LABELS[c] for c in angles['conditions']]
        fig,ax=plt.subplots(figsize=(2.65,2.15))
        img=ax.imshow(matrix,vmin=0,vmax=90 if np.max(matrix)<=90 else 180,cmap='YlOrBr')
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                ax.text(j,i,f'{matrix[i,j]:.0f}°',ha='center',va='center',fontsize=7,
                        color='white' if matrix[i,j]>.65*img.norm.vmax else '#233C46')
        ax.set_xticks(range(len(matrix)),labels,rotation=35,ha='right'); ax.set_yticks(range(len(matrix)),labels); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        fig.colorbar(img,ax=ax,fraction=.045,pad=.03,label='Mean update angle (°)')
        save(fig,f'{stem}_update_angles')
        conditions=['residual','lora','joint','random','permuted']
        index=report['delta_cka']['conditions']; rows=[index.index(c) for c in conditions]
        matrix=np.array(report['delta_cka']['matrix'])[np.ix_(rows,rows)]
        fig,ax=plt.subplots(figsize=(2.65,2.15))
        img=ax.imshow(matrix,vmin=0,vmax=1,cmap='GnBu')
        for i in range(len(rows)):
            for j in range(len(rows)):
                ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>.65 else '#233C46')
        ax.set_xticks(range(len(rows)),[LABELS[c] for c in conditions],rotation=35,ha='right')
        ax.set_yticks(range(len(rows)),[LABELS[c] for c in conditions]); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        fig.colorbar(img,ax=ax,fraction=.045,pad=.03,label='Displacement CKA')
        save(fig,f'{stem}_delta_cka')
        # Same joint checkpoint: coarse geometry versus independently recovered facts.
        summary_path=json.loads(Path('outputs/access_report/joint_comparison.json').read_text())[model]['source']
        behavior=json.loads(Path(summary_path).read_text()); ci=report['cka']['conditions']; cm=report['cka']['matrix']
        fig,ax=plt.subplots(figsize=(2.65,1.8))
        settings=[('joint','metrics','Both on','o'),('joint_residual_off','residual_off_metrics','Residual off','s'),
                  ('joint_lora_off','lora_off_metrics','LoRA off','^')]
        for c,key,label,marker in settings:
            x=cm[ci.index('joint')][ci.index(c)]; y=100*behavior[key]['J_1']
            ax.scatter(x,y,s=28,marker=marker,color=COLORS['joint'] if c=='joint' else '#7C8992',zorder=3)
            ax.annotate(label,(x,y),xytext=(-4,6),textcoords='offset points',ha='right' if x>.9 else 'left',fontsize=7)
        ax.set_xlim(.67,1.03); ax.set_ylim(-1,36); ax.set_yticks([0,10,20,30])
        ax.set_xlabel('CKA with full joint adapter'); ax.set_ylabel('Both facts correct (%)')
        clean(ax); save(fig,f'{stem}_ablation')
        maps=np.load(path/'shared_maps.npz'); coords=maps['umap_123']; n=len(chains)
        xlim=(coords[:,0].min()-.3,coords[:,0].max()+.3); ylim=(coords[:,1].min()-.3,coords[:,1].max()+.3)
        manifest=json.loads((path/'manifest.json').read_text())
        predictions={'frozen':root/'predictions/original.jsonl','residual':root/'predictions/correct_delta_seed-13.jsonl',
                     'lora':Path(manifest['checkpoints']['lora']['path']).parent/'predictions.jsonl',
                     'joint':Path(manifest['checkpoints']['joint']['path']).parent/'predictions.jsonl'}
        for i,c in enumerate(PRIMARY):
            points=coords[i*n:(i+1)*n]
            lookup={str(r['chain_id']):r for r in map(json.loads,predictions[c].open())}
            success=np.array([lookup[str(r['chain_id'])]['correct_12'] for r in chains],dtype=bool)
            fig,ax=plt.subplots(figsize=(1.7,1.45))
            ax.scatter(points[:,0],points[:,1],s=5,c=point_colors,alpha=.72,linewidths=0,rasterized=True)
            ax.scatter(points[success,0],points[success,1],s=18,facecolors='none',edgecolors='#182E38',lw=.6)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_axis_off()
            save(fig,f'{stem}_umap_{c}')
        # Overlay all four interfaces in their shared coordinates to avoid four
        # largely empty panels and to make cross-interface displacement visible.
        fig,ax=plt.subplots(figsize=(2.65,2.0))
        order=np.random.default_rng(123).permutation(len(coords))
        interface_colors=np.repeat([COLORS[c] for c in PRIMARY],n)
        ax.scatter(coords[order,0],coords[order,1],s=6,c=interface_colors[order],alpha=.6,linewidths=0,rasterized=True)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_axis_off()
        handles=[Line2D([],[],marker='o',lw=0,ms=4,color=COLORS[c],label=LABELS[c]) for c in PRIMARY]
        ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.14),frameon=False,ncol=4,
                  fontsize=6.5,columnspacing=.7,handletextpad=.15)
        save(fig,f'{stem}_umap_overlay')
        for c in ['residual','lora','joint']:
            for outcome,name in [('direct','Direct'),('both_constituents','Both facts')]:
                r=report['probes'][c][outcome]
                if not r['available']: continue
                delta=r['incremental_auc']; lo,hi=delta['ci95']
                probe_rows.append(f"{model} & {LABELS[c]} & {name} & {r['positive_count']}/{r['n']} & "
                                  f"{r['scores']['frozen_geometry']['auc']:.3f} & {r['scores']['combined']['auc']:.3f} & "
                                  f"{delta['estimate']:+.3f} [{lo:+.3f}, {hi:+.3f}]"+r' \\')
    for outcome in ['direct','both_constituents']:
        fig,ax=plt.subplots(figsize=(2.65,1.85))
        for model,offset,color in [('Llama',-.1,'#D35D6E'),('Qwen',.1,'#008C9B')]:
            for j,c in enumerate(['residual','lora','joint']):
                m=reports[model]['probes'][c][outcome]['incremental_auc']; v=m['estimate']; lo,hi=m['ci95']
                ax.errorbar(v,j+offset,xerr=[[v-lo],[hi-v]],fmt='o',ms=3.5,lw=1,capsize=2,color=color,label=model if j==0 else None)
        ax.axvline(0,color='#89969E',lw=.8,ls=':'); ax.set_yticks(range(3),[LABELS[c] for c in ['residual','lora','joint']])
        ax.invert_yaxis(); ax.set_xlabel('Additional held-out ROC AUC')
        ax.legend(frameon=False,fontsize=7,ncol=2,loc='lower left',bbox_to_anchor=(0,1.01),borderaxespad=0)
        clean(ax); save(fig,f'prediction_{outcome}')
    fig,ax=plt.subplots(figsize=(6,0.25)); ax.axis('off')
    handles=[Line2D([],[],marker='o',lw=0,ms=4,color=c,label=r.replace(' then ',' → ')) for r,c in zip(top,palette)]
    handles.append(Line2D([],[],marker='o',lw=0,ms=4,color='#CBD2D6',label='Other relations'))
    ax.legend(handles=handles,loc='center',frameon=False,ncol=5,fontsize=7,columnspacing=1,handletextpad=.2)
    save(fig,'umap_legend')
    tabular('final_geometry_shape','llrrrrrr',r'Model & Interface & $d_{\rm eff}$ & $d_{90}$ & Cone ($^\circ$) & Turn ($^\circ$) & 10-NN & CKA',shape_rows,
            r'Final-prompt geometry on the same 533 MQuAKE queries. Cone is the mean angle to the cloud mean; turn compares paired frozen and adapted states. Neighborhood retention and CKA use the frozen reference. Joint removals use the same jointly trained checkpoint.',
            'tab:final-shape')
    tabular('final_geometry_norms','llrrrrrrr',r'Model & Interface & $\|h\|$ & $q$ & $s$ & $\phi$ ($^\circ$) & $r$ & $t$ & $n_\phi$',norm_rows,
            r'Mean final-state norms and paired change components. $q=\|h\|/\|h_0\|$, $s=\|h-h_0\|/\|h_0\|$, and $\phi$ is the update-to-frozen-vector angle. Signed radial $r$ and nonnegative tangential $t$ are defined in the text. Zero updates have undefined $\phi$; $n_\phi$ counts defined angles. Absolute norms are not calibrated across models.',
            'tab:final-norms')
    tabular('final_geometry_labels','llrrrrr',r'Model & Interface & Relation CKA & Excess answer CKA & $q$ & Excess purity (pp) & $q$',label_rows,
            r'Answer organization beyond relation-pair-matched label permutations. Excess subtracts the permutation mean, not the frozen score. Purity is the fraction of ten neighbors with the same answer, excluding the same source entity. Adjusted reference-tail probabilities use 999 permutations and the Benjamini and Hochberg correction over 24 tests per model and statistic.',
            'tab:final-labels')
    tabular('final_geometry_probes','lllrrrl',r'Model & Adapter & Outcome & Positives & Frozen control & + Adapted geometry & AUC difference [95\%]',probe_rows,
            r'Outcome prediction with gold-answer-group-held-out folds. The control includes relation pair, prompt length, and frozen geometry. Scores average three fixed fold assignments. Intervals resample answer groups of fixed out-of-fold scores; they do not estimate training-replicate uncertainty.',
            'tab:final-probes')
    (OUT/'summary.json').write_text(json.dumps({m:{'shape':r['conditions'],'labels':r['label_tests'],'probes':r['probes'],
        'angles_norms':norm_reports[m],'umap':r['umap']} for m,r in reports.items()},indent=2))
    print('Generated final-state geometry appendix assets.')


if __name__=='__main__': main()
