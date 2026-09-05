#!/usr/bin/env python
"""Regenerate paper tables and title-free panels from audited row-level outputs.

Categorical comparisons use grouped points, not connected trajectories.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from geometry_llm.commands.audits.audit_composition import read_rows, group_key, paired_audit
from geometry_llm.paper_tables import style_table

ROOTS = {'SL': 'bb53acb4d683', 'SQ': 'c243073e2862', 'ML': '265b7bb1723b', 'MQ': '86cc2e353f23'}
COLORS = {'Llama':'#D35D6E','Qwen':'#008C9B'}
SEEDS = [13,37,71]
OUT = Path('outputs/access_report')
TEX = Path('paper/generated')


def rate_interval(seed_rows, field, samples=5000):
    """Resample seeds and joint bridge/answer groups; paired groups across seeds."""
    groups=sorted({group_key(r) for r in seed_rows[0]}); index={g:i for i,g in enumerate(groups)}
    counts=np.zeros((len(seed_rows),len(groups),2))
    for i,rows in enumerate(seed_rows):
        for r in rows: counts[i,index[group_key(r)]] += [r[field],1]
    rng=np.random.default_rng(123)
    weights=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=samples)
    chosen=rng.integers(len(seed_rows),size=(samples,len(seed_rows)))
    draws=[]
    for j in range(samples):
        c=counts[chosen[j]]
        values=np.einsum('g,sgk->sk',weights[j],c)
        draws.append(np.mean(values[:,0]/values[:,1]))
    vals=np.array([sum(r[field] for r in rows)/len(rows) for rows in seed_rows])
    return {'mean':float(vals.mean()),'sd':float(vals.std(ddof=1)) if len(vals)>1 else 0.,
            'ci95':np.quantile(draws,[.025,.975]).tolist(),'counts':[sum(r[field] for r in rows) for rows in seed_rows],
            'n_per_seed':len(seed_rows[0])}


def pct(x): return 'n/a' if x is None else f'{100*x:.1f}'
def frac(m): return f"{m['numerator']}/{m['denominator']}" if m['denominator'] else '0/0'
def ci(m): return 'n/a' if m['estimate'] is None else f"{pct(m['estimate'])} [{pct(m['ci95'][0])}, {pct(m['ci95'][1])}]"
def save(name):
    plt.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.03)
    plt.savefig(OUT/f'{name}.png',dpi=240,bbox_inches='tight',pad_inches=.03)
    plt.close()


def active_residual_parameters(path, joint=False):
    """Coordinates in nonzero learned rows; exclude reserved/untouched rows."""
    import torch
    saved=torch.load(path,map_location='cpu',weights_only=False)
    delta=saved['residual_state' if joint else 'state_dict']['delta']
    return int(delta.ne(0).any(dim=1).sum())*delta.shape[1]


def main():
    OUT.mkdir(parents=True,exist_ok=True); TEX.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#ADB8BE','axes.linewidth':.6,'pdf.fonttype':42,'savefig.facecolor':'white'})
    reports, conditional, interventions, behavior, access_rows, macros = {}, [], [], [], [], []
    relation_data = {}
    for key,stem in ROOTS.items():
        root=Path('outputs')/stem; model='Llama' if key[-1]=='L' else 'Qwen'
        data='SOCRATES' if key[0]=='S' else 'MQuAKE-CF'
        base=read_rows(root/'predictions/original.jsonl')
        rows=[read_rows(root/'predictions'/f'correct_delta_seed-{s}.jsonl') for s in SEEDS]
        audit=json.loads((root/'analysis/conditional_audit.json').read_text())
        report={k:rate_interval(rows,k) for k in ['correct_1','correct_2','correct_12']}
        for rr in rows:
            for r in rr: r['joint']=r['correct_1'] and r['correct_2']
        report['joint']=rate_interval(rows,'joint')
        report['C_mean']=float(np.mean([audit['seeds'][str(s)]['adapted_C']['estimate'] for s in SEEDS]))
        for label,rr in [('Frozen',base),*[(str(s),r) for s,r in zip(SEEDS,rows)]]:
            both=[r for r in rr if r['correct_1'] and r['correct_2']]
            c=sum(r['correct_12'] for r in both)/len(both) if both else None
            values=[pct(np.mean([r[k] for r in rr])) for k in ['correct_1','correct_2','correct_12']]
            behavior.append(f'{data} & {model} & {label} & '+ ' & '.join(values)+f' & {len(both)}/{len(rr)} & {pct(c)} \\\\')
        for seed in SEEDS:
            a=audit['seeds'][str(seed)]
            conditional.append(f'{data} & {model} & {seed} & {frac(a["adapted_C"])} & {ci(a["adapted_C"])} & '
                               f'{frac(a["common_base"])} & {frac(a["common_adapted"])} & {ci(a["common_change"])} \\\\')
        for name,a in audit['interventions'].items():
            label=name.replace('direction_removal_rank_','Remove rank ').replace('delta_permutation','Permute').replace('delta_removal','Remove').replace('scale_','Scale ')
            interventions.append(f'{model} & '+label+f' & {frac(a["adapted_C"])} & {ci(a["adapted_C"])} & '
                                  f'{frac(a["common_base"])} & {frac(a["common_adapted"])} & {ci(a["common_change"])} \\\\')
        if key[0]=='M':
            access=[read_rows(root/'predictions'/f'access_seed-{s}.jsonl') for s in SEEDS]
            relation_data[model] = access
            for field in ['correct_pipeline','correct_pipeline_stage2_off','correct_pipeline_path','correct_pipeline_path_stage2_off']:
                report[field]=rate_interval(access,field)
            for rr in access:
                for r in rr: r['access_effect']=int(r['correct_pipeline'])-int(r['correct_pipeline_stage2_off'])
            report['access_effect']=rate_interval(access,'access_effect')
            oracle=read_rows(root/'predictions/oracle_explicit_facts.jsonl')
            report['oracle']=rate_interval([oracle],'correct_oracle')
            for seed,rr in zip(SEEDS,access):
                values=[sum(r[k] for r in rr) for k in ['correct_12','correct_pipeline_stage2_off','correct_pipeline','correct_pipeline_path']]
                access_rows.append(f'{model} & {seed} & '+' & '.join(str(v) for v in values)+f' & {sum(r["correct_oracle"] for r in oracle)} \\\\')
            fig,ax=plt.subplots(figsize=(2.8,1.85))
            fields=['correct_12','correct_pipeline_stage2_off','correct_pipeline','oracle']
            for y,f in enumerate(fields):
                m=report[f]; x=100*m['mean']; lo,hi=np.array(m['ci95'])*100
                ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt='o',color=COLORS[model],capsize=2,ms=4,lw=1)
                ax.text(x+3,y-.1,f'{x:.1f}',fontsize=7,color=COLORS[model])
            ax.set_yticks(range(4),['Direct','Stage 2 off','Pipeline','Facts in context']); ax.invert_yaxis()
            ax.set_xlim(-1,103); ax.set_xticks([0,25,50,75,100]); ax.set_xlabel('Answer accuracy (%)')
            ax.grid(axis='x',color='#EDF0F2',lw=.7); fig.tight_layout(pad=.4); save('access_'+model.lower())
        else:
            fig,ax=plt.subplots(figsize=(2.7,1.85))
            for i,s in enumerate(SEEDS):
                m=audit['seeds'][str(s)]['adapted_C']; x=100*m['estimate']; lo,hi=np.array(m['ci95'])*100
                ax.errorbar(i,x,yerr=[[x-lo],[hi-x]],fmt='o',color=COLORS[model],capsize=2,ms=4)
                ax.annotate(frac(m),(i,x),xytext=(4,5),textcoords='offset points',fontsize=7)
            original=audit['seeds']['13']['base_C']['estimate']*100
            ax.axhline(original,color='#758791',ls='--',lw=1)
            ax.set_ylim(-2,65); ax.set_xlim(-.5,2.9); ax.set_xticks(range(3),['13','37','71'])
            ax.set_ylabel('Conditional accuracy (%)'); ax.set_xlabel('Training seed')
            ax.grid(axis='y',color='#EDF0F2',lw=.7); fig.tight_layout(pad=.4); save('conditional_'+model.lower())
        reports[key]=report
        macros.append('\\newcommand{\\'+key+'Direct}{'+pct(report['correct_12']['mean'])+'}')
    for name,rows in [('behavior',behavior),('conditional',conditional),('interventions',interventions),('access',access_rows)]:
        (TEX/f'{name}_rows.tex').write_text(style_table('\n'.join(rows)+'\n\\bottomrule\n'))
    (TEX/'metrics.tex').write_text('\n'.join(macros)+'\n')
    relation_rows=[]
    for relation in sorted({r['fact_comp_type'] for r in relation_data['Llama'][0]}):
        n=sum(r['fact_comp_type']==relation for r in relation_data['Llama'][0])
        if n<10: continue
        values=[]
        for model in ['Llama','Qwen']:
            subset=[r for seedrows in relation_data[model] for r in seedrows if r['fact_comp_type']==relation]
            values.extend(pct(np.mean([r[k] for r in subset])) for k in ['correct_12','correct_pipeline'])
        relation_rows.append(relation.replace(' then ',r'$\rightarrow$')+f' & {n} & '+' & '.join(values)+r' \\')
    (TEX/'relation_rows.tex').write_text(style_table('\n'.join(relation_rows)+'\n\\bottomrule\n'))
    # Compact grouped bars: a shared zero baseline and direct value labels.
    # The final coverage confirmation is primary once its audited report exists.
    # Other panels intentionally remain the reference-fit baseline comparison.
    main_reports={k:dict(v) for k,v in reports.items()}
    confirmation_path=Path('outputs/confirmation_report/summary.json')
    if confirmation_path.exists():
        confirmation=json.loads(confirmation_path.read_text())
        for key,model in [('ML','Llama'),('MQ','Qwen')]:
            main_reports[key].update(confirmation[model]['protocols']['Coverage'])
    fields=['correct_12','correct_pipeline_stage2_off','correct_pipeline','oracle']
    fig,ax=plt.subplots(figsize=(3.25,1.7))
    for key,model,offset in [('ML','Llama',-.18),('MQ','Qwen',.18)]:
        vals=np.array([main_reports[key][f]['mean'] for f in fields])*100
        bounds=np.array([main_reports[key][f]['ci95'] for f in fields])*100
        y=np.arange(4)+offset
        ax.barh(y,vals,height=.31,color=COLORS[model],label=model,zorder=3)
        ax.errorbar(vals,y,xerr=[vals-bounds[:,0],bounds[:,1]-vals],fmt='none',
                    ecolor='#344750',elinewidth=.65,capsize=1.5,zorder=4)
        for yy,v,hi in zip(y,vals,bounds[:,1]):
            ax.text(hi+1.7,yy,f'{v:.1f}',ha='left',va='center',fontsize=7.5,color='#344750')
    ax.set_yticks(range(4),['Direct','Stage 2 off','Stage 2 on','Facts in context']); ax.invert_yaxis()
    ax.set_xlim(0,102); ax.set_xticks([0,25,50,75,100]); ax.set_xlabel('Answer accuracy (%)',labelpad=2)
    ax.tick_params(axis='both',labelsize=8)
    ax.legend(frameon=False,fontsize=8,ncol=2,loc='lower left',bbox_to_anchor=(0,1.01),
              handlelength=1,handletextpad=.4,columnspacing=1.1,borderaxespad=0)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
    ax.grid(axis='x',color='#E8EDF0',lw=.6); ax.set_axisbelow(True)
    fig.tight_layout(pad=.35); save('access_wrap')
    fig,ax=plt.subplots(figsize=(2.25,1.65))
    for key,model,offset in [('SL','Llama',-.08),('SQ','Qwen',.08)]:
        audit=json.loads((Path('outputs')/ROOTS[key]/'analysis/conditional_audit.json').read_text())
        vals=[100*audit['seeds'][str(s)]['adapted_C']['estimate'] for s in SEEDS]
        ax.plot(np.arange(3)+offset,vals,'o-',lw=1,ms=3,color=COLORS[model],label=model)
        ax.axhline(100*audit['seeds']['13']['base_C']['estimate'],color=COLORS[model],lw=.7,ls='--',alpha=.7)
    ax.set_xticks(range(3),['13','37','71']); ax.set_ylim(-2,49); ax.set_yticks([0,20,40])
    ax.set_xlabel('Training seed'); ax.set_ylabel('Conditional accuracy (%)')
    ax.legend(frameon=False,fontsize=6,ncol=2,loc='upper center',handlelength=.8,columnspacing=.6)
    ax.grid(axis='y',color='#EDF0F2',lw=.7); fig.tight_layout(pad=.35); save('conditional_seeds_compact')
    fig,ax=plt.subplots(figsize=(2.25,1.65))
    for key,model,offset in [('SL','Llama',-.1),('SQ','Qwen',.1)]:
        audit=json.loads((Path('outputs')/ROOTS[key]/'analysis/conditional_audit.json').read_text())
        for i,s in enumerate(SEEDS):
            m=audit['seeds'][str(s)]['common_change']; x=m['estimate']*100; lo,hi=np.array(m['ci95'])*100
            ax.errorbar(i+offset,x,yerr=[[x-lo],[hi-x]],fmt='o',ms=3,lw=.85,capsize=2,color=COLORS[model])
    ax.axhline(0,color='#84929A',lw=.7,ls='--'); ax.set_xticks(range(3),['13','37','71'])
    ax.set_ylabel('Common-set change (pp)'); ax.set_xlabel('Training seed'); ax.set_ylim(-28,65)
    ax.grid(axis='y',color='#EDF0F2',lw=.7); fig.tight_layout(pad=.35); save('common_change_compact')
    fig,ax=plt.subplots(figsize=(2.25,1.65))
    for key,model,offset in [('ML','Llama',-.12),('MQ','Qwen',.12)]:
        fields=['joint','correct_pipeline_path','correct_pipeline_path_stage2_off']
        ax.bar(np.arange(3)+offset,[100*reports[key][f]['mean'] for f in fields],width=.23,color=COLORS[model])
    ax.set_xticks(range(3),['Both\nfacts','Correct\npath','Stage 2\noff']); ax.set_ylim(0,26)
    ax.set_ylabel('Chain accuracy (%)'); ax.grid(axis='y',color='#EDF0F2',lw=.7); ax.set_axisbelow(True)
    fig.tight_layout(pad=.35); save('path_success_compact')
    baseline_rows=[]; baseline_wrap_rows=[]; baseline_reports={}; joint_reports={}
    for key,model in [('ML','Llama'),('MQ','Qwen')]:
        root=Path('outputs')/ROOTS[key]
        candidates=[(p,json.loads(p.read_text())) for p in (root/'lora').glob('*/summary.json')]
        if not candidates: continue
        path,chosen=min(candidates,key=lambda item:item[1]['selection']['efficacy_rmse'])
        reference=read_rows(root/'predictions/correct_delta_seed-13.jsonl')
        baseline=read_rows(path.parent/'predictions.jsonl')
        audit=paired_audit(reference,baseline)
        baseline_reports[model]={'source':str(path),'summary':chosen,'paired_with_residual':audit}
        comparisons=[('Residual',reference),('LoRA',baseline)]
        joint_candidates=[(p,json.loads(p.read_text())) for p in (root/'joint_lora').glob('*/summary.json')]
        if joint_candidates:
            joint_path,joint=min(joint_candidates,key=lambda item:item[1]['selection']['efficacy_rmse'])
            joint_rows=read_rows(joint_path.parent/'predictions.jsonl')
            joint_reports[model]={'source':str(joint_path),'summary':joint,
                                  'paired_with_residual':paired_audit(reference,joint_rows),
                                  'paired_with_lora':paired_audit(baseline,joint_rows)}
            comparisons.append((r'$\Delta$ + LoRA',joint_rows))
        for label,rr in comparisons:
            values=[pct(np.mean([r[k] for r in rr])) for k in ['correct_1','correct_2','correct_12']]
            both=[r for r in rr if r['correct_1'] and r['correct_2']]
            c=f'{sum(r["correct_12"] for r in both)}/{len(both)}'
            budget=(chosen['residual_parameter_budget'] if label=='Residual' else
                    chosen['lora_parameters'] if label=='LoRA' else joint['joint_parameters'])
            active=(active_residual_parameters(reference[0]['checkpoint']) if label=='Residual' else
                    chosen['lora_parameters'] if label=='LoRA' else
                    joint['lora_parameters']+active_residual_parameters(joint_path.parent/'selected.pt',joint=True))
            baseline_rows.append(f'{model} & {label} & {budget/1e6:.3f} & {active/1e6:.3f} & '+' & '.join(values)+f' & {c}'+r' \\')
            color='llamarow' if model=='Llama' else 'qwenrow'
            baseline_wrap_rows.append(r'\rowcolor{'+color+'}'+f'{model} & {label} & '+' & '.join(values)+r' \\')
    if baseline_reports:
        table='\n'.join(baseline_rows)
        appendix=(r'\paragraph{LoRA comparison at constituent efficacy.}'+'\n'+
            'This is a seed-13 adaptation comparison. LoRA acts on query and value projections in every layer, '
            'with initial scaling equal to one, zero-initialized output factors, and no dropout. AdamW uses no weight decay, '
            'batch size 8, accumulation 2, and gradient clipping 1. '+
            'Rank is rounded to the nearest allocated residual-parameter budget, including reserved target rows. '
            'Llama uses rank 10 and learning rate $10^{-4}$; Qwen uses rank 12 and searches '+
            '$10^{-4}$ and $3\\cdot10^{-4}$. Candidate epoch counts are 1, 2, 4, and 8, with an additional '+
            'six-epoch Qwen fit after the eight-epoch fit overshot the target efficacy. Qwen also calibrates '+
            'the update magnitude using only constituent outcomes. Selection minimizes the '+
            'root-mean-square distance to the residual\'s first-hop accuracy, second-hop accuracy, and joint '+
            'coverage on installed facts. Composed outcomes do not select either rate or epoch. '+
            'The Qwen expansion is exploratory, and this pilot does not estimate seed variability.\n'+
            r'\begin{table}[H]\centering\small'+'\n'+r'\begin{tabular}{llrrrrrl}\toprule'+'\n'+
            r'Model & Interface & Allocated (M) & Active (M) & $A_{1a}$ & $A_{1b}$ & $A_2$ & $C$ count\\\midrule'+'\n'+table+'\n'+
            r'\bottomrule\end{tabular}'+'\n'+r'\caption{Exploratory reference-fit adaptation comparison. Parameter counts are millions; active counts include coordinates in nonzero residual rows plus all LoRA factors. Reserved and untouched residual rows are excluded, so the allocated-budget match is not an active-budget match. Accuracy is in percent; conditional populations differ by interface.}\end{table}'+'\n')
        for model,report in baseline_reports.items():
            m=report['summary']; a=report['paired_with_residual']
            appendix+=f"{model} selects epoch {m['selection']['epoch']} at learning rate ${m['learning_rate']:g}$ and scale {m['scaling']:g}. "
            appendix+=f"The paired direct change relative to its residual is {ci(a['adapted_direct_change'])} points. "
            appendix+=f"On the common constituent-correct set, counts are {frac(a['common_base'])} and {frac(a['common_adapted'])}; "
            appendix+=f"the paired change is {ci(a['common_change'])} points.\n"
        if joint_reports:
            appendix+='\n'+r'\paragraph{Joint residual and LoRA control.}'+'\n'
            appendix+=('Both components are optimized together from zero effective updates, using the same '
                       'one-hop records, sampling, optimizer, and constituent-only selection rule as LoRA. '
                       'The full residual table and full LoRA rank are retained, so the joint budget is '
                       'approximately twice the single-interface budget; this is a complementary-interface '
                       'control, not a parameter-matched comparison. Candidate epochs are 1, 2, 4, and 8. '
                       'The residual learning rate and anchor come from its dataset-specific local validation. '
                       'The backbone stays frozen. Component removals below use the same selected joint '
                       'checkpoint without retraining. Each joint adapter has one optimization replicate.\n')
            ablation_rows=[]
            for model,report in joint_reports.items():
                m=report['summary']; parent=Path(report['source']).parent
                appendix+=(f"{model}: rank {m['rank']}, LoRA learning rate ${m['learning_rate']:g}$, "
                           f"residual rate ${m['residual_learning_rate']:g}$, anchor ${m['residual_anchor']:g}$, "
                           f"scale {m['scaling']:g}, selected epoch {m['selection']['epoch']}. ")
                for comparison,label in [('paired_with_residual','residual'),('paired_with_lora','LoRA')]:
                    appendix+=f"The paired direct change against {label} is {ci(report[comparison]['adapted_direct_change'])} points. "
                appendix+='\n'
                for filename,label in [('predictions.jsonl','Both on'),('residual_off.jsonl','Residual off'),('lora_off.jsonl','LoRA off')]:
                    if not (parent/filename).exists(): continue
                    rr=read_rows(parent/filename)
                    if len(rr)!=len(reference): continue
                    metrics=[pct(np.mean([r[k] for r in rr])) for k in ['correct_1','correct_2']]
                    direct=rate_interval([rr],'correct_12')
                    both=[r for r in rr if r['correct_1'] and r['correct_2']]
                    count=f'{sum(r["correct_12"] for r in both)}/{len(both)}'
                    ablation_rows.append(f'{model} & {label} & '+' & '.join(metrics)+
                                          f" & {pct(direct['mean'])} [{pct(direct['ci95'][0])}, {pct(direct['ci95'][1])}] & {count}"+r' \\')
            appendix+=(r'\begin{table}[H]\centering\small'+'\n'+r'\begin{tabular}{llrrll}\toprule'+'\n'+
                       r'Model & Joint components & $A_{1a}$ & $A_{1b}$ & $A_2$ [95\% interval] & Conditional count\\\midrule'+'\n'+
                       '\n'.join(ablation_rows)+'\n'+r'\bottomrule\end{tabular}'+'\n'+
                       r'\caption{Joint-adapter component ablations. Accuracy is in percent; intervals resample bridge/answer groups, not optimization replicates.}\end{table}'+'\n')
        (TEX/'lora_appendix.tex').write_text(style_table(appendix))
        wrap=(r'\begin{wraptable}{r}{.51\linewidth}'+'\n'+r'\vspace{-.8em}\centering\scriptsize\setlength{\tabcolsep}{3.5pt}'+'\n'+
              r'\begin{tabular}{llrrr}\toprule'+'\n'+r'Model & Interface & $A_{1a}$ & $A_{1b}$ & $A_2$\\\midrule'+'\n'+
              '\n'.join(baseline_wrap_rows)+'\n'+r'\bottomrule\end{tabular}'+'\n'+
              (r'\caption{Exploratory reference-fit MQuAKE comparison (\%). LoRA approximately matches the allocated, not active, residual budget; joint $\Delta$ + LoRA uses both budgets. Selection and counts: Appendix~\ref{app:behavior}.}' if joint_reports else
               r'\caption{MQuAKE adapters matched by budget and constituent efficacy; accuracy (\%). Selection, replication, and counts: Appendix~\ref{app:behavior}.}')+
              r'\label{tab:lora-main}\vspace{-.6em}\end{wraptable}'+'\n')
        (TEX/'lora_wraptable.tex').write_text(style_table(wrap))
        paragraph=(r'\paragraph{Local efficacy does not determine consequence accuracy.}'+ '\n'+
            'LoRA achieves similar constituent accuracy with higher direct scores '+
            'than the residuals (Table~\\ref{tab:lora-main}). This supports examining where information is installed, '+
            'beyond how often isolated probes succeed. Limited baseline replication does not establish a '+
            'general advantage for weight editing.\n')
        if len(joint_reports)==2:
            paragraph=(r'\paragraph{Better local fitting does not guarantee transfer.}'+'\n'+
                       'LoRA recovers more direct answers than residual-only adaptation at similar local efficacy '+
                       '(Table~\\ref{tab:lora-main}). Joint training further increases both constituent accuracies, '+
                       'yet does not improve direct scores over standalone LoRA in these fits. Adding a weight-level '+
                       'learning route therefore does not automatically make local improvements compositional. '+
                       'Limited replication precludes a general ranking of the methods.\n')
        (TEX/'lora_paragraph.tex').write_text(paragraph)
        (OUT/'lora_comparison.json').write_text(json.dumps(baseline_reports,indent=2,allow_nan=False))
        (OUT/'joint_comparison.json').write_text(json.dumps(joint_reports,indent=2,allow_nan=False))
        # Shading summarizes uncertainty; seeds are replications, not an x axis.
        condition_colors={'Frozen':'#89969E','Residual':'#008C9B','LoRA':'#C18434','Joint':'#7562A5'}
        for key,model in [('ML','Llama'),('MQ','Qwen')]:
            root=Path('outputs')/ROOTS[key]
            conditions={
                'Frozen':[read_rows(root/'predictions/original.jsonl')],
                'Residual':[read_rows(root/'predictions'/f'correct_delta_seed-{s}.jsonl') for s in SEEDS],
                'LoRA':[read_rows(Path(baseline_reports[model]['source']).parent/'predictions.jsonl')]}
            if model in joint_reports:
                conditions['Joint']=[read_rows(Path(joint_reports[model]['source']).parent/'predictions.jsonl')]
            fig,ax=plt.subplots(figsize=(2.15,1.75))
            offsets=np.linspace(-.27,.27,len(conditions))
            for offset,(label,rr) in zip(offsets,conditions.items()):
                metrics=[rate_interval(rr,f) for f in ['correct_1','correct_2','correct_12']]
                means=np.array([m['mean'] for m in metrics])*100
                bounds=np.array([m['ci95'] for m in metrics])*100
                legend_label={'Residual':'Δ','Joint':'Δ + LoRA'}.get(label,label)
                ax.errorbar(np.arange(3)+offset,means,yerr=[means-bounds[:,0],bounds[:,1]-means],
                            fmt='o',linestyle='none',color=condition_colors[label],
                            elinewidth=.8,capsize=1.7,ms=3,label=legend_label)
            ax.set_xticks(range(3),['Hop 1','Hop 2','Direct'])
            ax.set_ylabel('Accuracy (%)'); ax.set_ylim(-2,72); ax.set_yticks([0,20,40,60]); ax.set_xlim(-.43,2.43)
            ax.grid(axis='y',color='#E8EDF0',lw=.6); ax.set_axisbelow(True)
            ax.legend(frameon=False,fontsize=6.3,ncol=len(conditions),loc='lower left',bbox_to_anchor=(0,1.01),
                      handlelength=.7,handletextpad=.25,columnspacing=.55,borderaxespad=0)
            fig.tight_layout(pad=.35); save('efficacy_'+model.lower())
        fig,ax=plt.subplots(figsize=(2.15,1.75))
        for offset,(key,model) in zip([-.12,.12],[('SL','Llama'),('SQ','Qwen')]):
            root=Path('outputs')/ROOTS[key]
            conditions=[[read_rows(root/'predictions/original.jsonl')]]
            conditions += [[read_rows(root/'predictions'/f'{condition}_seed-{s}.jsonl') for s in SEEDS]
                           for condition in ['shuffled_delta','random_delta','correct_delta']]
            metrics=[rate_interval(rr,'correct_12') for rr in conditions]
            means=np.array([m['mean'] for m in metrics])*100
            bounds=np.array([m['ci95'] for m in metrics])*100
            ax.errorbar(np.arange(4)+offset,means,yerr=[means-bounds[:,0],bounds[:,1]-means],
                        fmt='o',linestyle='none',color=COLORS[model],elinewidth=.8,capsize=1.7,ms=3,label=model)
        ax.set_xticks(range(4),['Frozen','Shuffled','Random','Residual'],rotation=20,ha='right')
        ax.set_ylabel('Direct accuracy (%)'); ax.set_ylim(-1,51); ax.set_yticks([0,15,30,45]); ax.set_xlim(-.32,3.32)
        ax.grid(axis='y',color='#E8EDF0',lw=.6); ax.set_axisbelow(True)
        ax.legend(frameon=False,fontsize=6,ncol=2,loc='upper left',handlelength=.8,columnspacing=.8,borderaxespad=.2)
        fig.tight_layout(pad=.35); save('residual_controls')
    (OUT/'summary.json').write_text(json.dumps(reports,indent=2,allow_nan=False))
    print(json.dumps({k:{f:v for f,v in r.items() if f in ['correct_12','correct_pipeline','access_effect','C_mean']} for k,r in reports.items()},indent=2))


if __name__=='__main__':
    main()
