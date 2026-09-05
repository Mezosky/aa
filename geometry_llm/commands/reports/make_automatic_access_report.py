#!/usr/bin/env python
"""Audited paired automatic-access metrics, compact plots and coloured tables."""
from collections import Counter
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from geometry_llm.commands.audits.audit_composition import group_key
from geometry_llm.commands.evaluation.evaluate_access import alias_lookup, read_rows
from geometry_llm.automatic_access import parse_answer, ASSISTANT_PREFIX
from geometry_llm.data import load_saved_chains
from geometry_llm.paper_tables import style_table
from geometry_llm.text import answer_is_correct, normalize_answer
from geometry_llm.commands.reports.make_access_report import rate_interval, COLORS, SEEDS

OUT=Path('outputs/automatic_access_report')
TEX=Path('paper/generated')
ROOTS={'Llama':'265b7bb1723b','Qwen':'86cc2e353f23'}


def conditional_interval(runs,field,denominator,samples=5000):
    groups=sorted({group_key(r) for r in runs[0]}); idx={g:j for j,g in enumerate(groups)}
    counts=np.zeros((len(runs),len(groups),2))
    for j,rows in enumerate(runs):
        for r in rows:
            if r[denominator]: counts[j,idx[group_key(r)]] += [r[field],1]
    raw=counts.sum(axis=1)
    rng=np.random.default_rng(123)
    weights=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=samples)
    selected=rng.integers(len(runs),size=(samples,len(runs)))
    draws=[]
    for w,chosen in zip(weights,selected):
        totals=np.einsum('g,sgk->sk',w,counts[chosen])
        if np.all(totals[:,1]>0): draws.append(float(np.mean(totals[:,0]/totals[:,1])))
    return dict(mean=float(np.mean(raw[:,0]/raw[:,1])) if np.all(raw[:,1]>0) else None,
        ci95=np.quantile(draws,[.025,.975]).tolist() if draws else [None,None],
        counts=raw[:,0].astype(int).tolist(),denominators=raw[:,1].astype(int).tolist(),valid_draws=len(draws))


def relaxed_answer(text):
    """Post-hoc formatting sensitivity; a unique answer-labelled line anywhere."""
    matches=re.findall(r'^\s*Answer:\s*([^\n\r]+?)\s*$',text,flags=re.M|re.I)
    return matches[0] if len(matches)==1 else ''


def interval(m):
    if m['mean'] is None: return 'n/a'
    return f"{100*m['mean']:.1f} [{100*m['ci95'][0]:.1f}, {100*m['ci95'][1]:.1f}]"


def write_table(name,columns,header,rows,caption,label):
    text=(r'\begin{table}[H]\centering\small'+'\n'+r'\setlength{\tabcolsep}{4pt}'+'\n'
        +r'\begin{tabular}{'+columns+r'}\toprule'+'\n'+header+r'\\\midrule'+'\n'
        +'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}'+'\n'
        +r'\caption{'+caption+r'}\label{'+label+r'}\end{table}'+'\n')
    (TEX/(name+'.tex')).write_text(style_table(text))


def main():
    OUT.mkdir(exist_ok=True); TEX.mkdir(exist_ok=True)
    reports={}; seed_rows=[]; aggregate_rows=[]; diagnostics=[]; conditional=[]; macros=[]
    for model,stem in ROOTS.items():
        root=Path('outputs')/stem
        chains=load_saved_chains(root/'selected_chains.jsonl'); lookup=alias_lookup(chains)
        runs=[read_rows(root/'automatic_access_v1'/f'seed-{s}.jsonl') for s in SEEDS]
        for seed,rows in zip(SEEDS,runs):
            assert [r['chain_id'] for r in rows]==[c.chain_id for c in chains]
            for r,c in zip(rows,chains):
                assert r['seed']==seed
                assert r['resolved_entity']==lookup.get(normalize_answer(r['intermediate'])) or r['boundary']!='newline'
                assert r['correct_bridge']==answer_is_correct(r['intermediate'],c.e2_aliases)
                if r['correct_bridge_accessed']: assert r['resolved_entity']==c.e2
                if r['active_nonzero']:
                    assert r['generated_positions'] and min(r['generated_positions'])>=len(r['prefix_ids'])
                else: assert r['continuation_ids_on']==r['continuation_ids_off']
                for condition in ['on','off']:
                    answer=parse_answer(r['response_'+condition][len(ASSISTANT_PREFIX):]) if r['boundary']=='newline' else ''
                    assert answer==r['answer_'+condition]
                    assert r['correct_'+condition]==answer_is_correct(answer,c.e3_aliases)
                    assert r['correct_path_'+condition]==(r['correct_bridge'] and r['correct_'+condition])
                    r['relaxed_'+condition]=answer_is_correct(relaxed_answer(r['response_'+condition]),c.e3_aliases)
                r['relaxed_effect']=int(r['relaxed_on'])-int(r['relaxed_off'])
                r['path_accessed_on']=r['correct_bridge_accessed'] and r['correct_on']
            seed_rows.append(f"{model} & {seed} & "+' & '.join(str(sum(r[k] for r in rows)) for k in
                ['correct_bridge','active_nonzero','correct_off','correct_on','correct_path_off','correct_path_on'])+r'\\')
        fields=['correct_bridge','active_nonzero','correct_bridge_accessed','correct_off','correct_on',
                'correct_path_off','correct_path_on','access_effect','path_effect','format_off','format_on',
                'relaxed_off','relaxed_on','relaxed_effect']
        report={k:rate_interval(runs,k) for k in fields}
        report['conditional_on_accessed_correct_bridge']={k:conditional_interval(runs,k,'correct_bridge_accessed')
            for k in ['correct_on','correct_off','access_effect','format_on','format_off','relaxed_on','relaxed_off']}
        report['resolution_by_seed']=[dict(Counter(r['resolution'] for r in rr)) for rr in runs]
        report['boundaries_by_seed']=[dict(Counter(r['boundary'] for r in rr)) for rr in runs]
        report['changed_answers']=[sum(r['answer_on']!=r['answer_off'] for r in rr) for rr in runs]
        report['paired_discordance']=[dict(gained=sum(r['correct_on'] and not r['correct_off'] for r in rr),
            lost=sum(r['correct_off'] and not r['correct_on'] for r in rr)) for rr in runs]
        report['cache_audits']=[json.loads((root/'automatic_access_v1'/f'cache_audit_seed-{s}.json').read_text()) for s in SEEDS]
        report['strict_scoring_primary']=True; report['relaxed_scoring_posthoc_sensitivity']=True
        reports[model]=report
        for route,final,path in [('Lookup off','correct_off','correct_path_off'),('Lookup on','correct_on','correct_path_on')]:
            aggregate_rows.append(f'{model} & {route} & '+interval(report['correct_bridge'])+' & '+interval(report[final])+' & '+interval(report[path])+r'\\')
        for label,key in [('Nonzero lookup','active_nonzero'),('Correct bridge accessed','correct_bridge_accessed'),
                          ('Valid answer field, off','format_off'),('Valid answer field, on','format_on'),
                          ('Relaxed final, off','relaxed_off'),('Relaxed final, on','relaxed_on')]:
            diagnostics.append(f'{model} & {label} & '+interval(report[key])+r' & '+', '.join(map(str,report[key]['counts']))+r'\\')
        for label,key in [('Lookup off','correct_off'),('Lookup on','correct_on')]:
            m=report['conditional_on_accessed_correct_bridge'][key]
            conditional.append(f'{model} & {label} & '+interval(m)+r' & '+', '.join(f'{n}/{d}' for n,d in zip(m['counts'],m['denominators']))+r'\\')
        for label,key in [('Bridge','correct_bridge'),('Off','correct_off'),('On','correct_on'),
                          ('PathOff','correct_path_off'),('PathOn','correct_path_on'),('Active','active_nonzero')]:
            macros.append(r'\newcommand{\Auto'+model+label+'}{'+f"{100*report[key]['mean']:.1f}"+'}')
        gain=report['access_effect']
        for label,value in [('Gain',gain['mean']),('Low',gain['ci95'][0]),('High',gain['ci95'][1])]:
            macros.append(r'\newcommand{\Auto'+model+label+'}{'+f'{100*value:.1f}'+'}')
    (OUT/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    (TEX/'automatic_metrics.tex').write_text('\n'.join(macros)+'\n')
    write_table('automatic_aggregate','lllll','Model & Generation & Correct bridge & Final answer & Complete path',aggregate_rows,
        'Automatic access on 533 MQuAKE cases: mean percent accuracy [95\\% grouped/fit bootstrap interval]. '
        'The same model-generated intermediate is shared by both branches. No constituent question or second-hop template is supplied.', 'tab:automatic')
    write_table('automatic_counts','lrrrrrrr','Model & Seed & Bridge & Nonzero lookup & Final off & Final on & Path off & Path on',seed_rows,
        'Automatic-access numerators. Every cell has denominator 533; seeds are 13, 37, and 71. Complete path requires the generated bridge and final answer to match their target aliases.','tab:automatic-counts')
    write_table('automatic_diagnostics','llll','Model & Diagnostic & Percent [95\\% interval] & Counts (13, 37, 71)',diagnostics,
        'Lookup and output diagnostics on all 533 cases per fit. Relaxed final scoring is a post-hoc sensitivity check that accepts one answer-labelled line anywhere in the response, not only the second line. It never searches for the gold answer.','tab:automatic-diagnostics')
    write_table('automatic_conditional','llll','Model & Generation & Final accuracy [95\\%] & Correct / accessed correct bridge',conditional,
        'Final accuracy on the common paired set where the model generated the correct bridge and its nonzero residual was activated. Counts expose the limited conditioning populations.','tab:automatic-conditional')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#ADB8BE','axes.linewidth':.6,'pdf.fonttype':42,'savefig.facecolor':'white'})
    for model,report in reports.items():
        fig,ax=plt.subplots(figsize=(2.65,1.65))
        for j,(label,fields,face) in enumerate([
            ('Lookup off',['correct_off','correct_path_off'],'white'),
            ('Lookup on',['correct_on','correct_path_on'],COLORS[model])]):
            x=np.arange(2)+(j-.5)*.30
            mean=np.array([report[f]['mean'] for f in fields])*100
            ci=np.array([report[f]['ci95'] for f in fields]).T*100
            ax.bar(x,mean,width=.27,color=face,edgecolor=COLORS[model],linewidth=.9,label=label,
                yerr=np.maximum(0,np.array([mean-ci[0],ci[1]-mean])),error_kw={'capsize':2,'elinewidth':.8,'ecolor':'#435963'})
        ax.set_xticks([0,1],['Final answer','Complete path'])
        ax.set_ylabel('Accuracy (%)'); ax.set_axisbelow(True); ax.grid(axis='y',color='#EDF0F2',lw=.6)
        # Shared scale across the model panels makes magnitudes comparable.
        ymax=max(r[f]['ci95'][1]*100 for r in reports.values() for f in ['correct_on','correct_off'])*1.15
        ax.set_ylim(0,max(3.,ymax)); ax.set_yticks(np.arange(0,max(3.,ymax),1))
        ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.0),fontsize=8,
                  handlelength=1.,columnspacing=.9,handletextpad=.4)
        fig.tight_layout(pad=.35)
        fig.savefig(OUT/f'automatic_{model.lower()}.pdf',bbox_inches='tight',pad_inches=.025)
        fig.savefig(OUT/f'automatic_{model.lower()}.png',bbox_inches='tight',pad_inches=.025,dpi=240)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(2.65,1.65))
    for j,(model,report) in enumerate(reports.items()):
        fields=['correct_bridge','active_nonzero','correct_bridge_accessed']
        mean=np.array([report[f]['mean'] for f in fields])*100
        ci=np.array([report[f]['ci95'] for f in fields]).T*100
        ax.barh(np.arange(3)+(j-.5)*.29,mean,height=.26,color=COLORS[model],label=model,
            xerr=np.maximum(0,np.array([mean-ci[0],ci[1]-mean])),error_kw={'capsize':2,'elinewidth':.8,'ecolor':'#435963'})
    ax.set_yticks(range(3),['Correct bridge','Any row active','Correct + active']); ax.invert_yaxis()
    ax.set_xlabel('Chains (%)'); ax.set_axisbelow(True); ax.grid(axis='x',color='#EDF0F2',lw=.6)
    ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.35,1.0),fontsize=8,handlelength=1.)
    fig.tight_layout(pad=.35)
    fig.savefig(OUT/'automatic_lookup.pdf',bbox_inches='tight',pad_inches=.025)
    fig.savefig(OUT/'automatic_lookup.png',bbox_inches='tight',pad_inches=.025,dpi=240)
    plt.close(fig)
    print(json.dumps({model:{k:interval(r[k]) for k in ['correct_bridge','correct_off','correct_on','correct_path_on','access_effect']} for model,r in reports.items()},indent=2))


if __name__=='__main__': main()
