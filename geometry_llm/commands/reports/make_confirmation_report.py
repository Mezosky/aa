#!/usr/bin/env python
"""Tables and compact panels for the final, separately identified confirmation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from geometry_llm.commands.audits.audit_composition import paired_audit
from geometry_llm.paper_tables import style_table
from geometry_llm.commands.reports.make_access_report import rate_interval, pct, ci

ROOTS={'Llama':Path('outputs/265b7bb1723b'),'Qwen':Path('outputs/86cc2e353f23')}
SEEDS=[13,37,71]
OUT=Path('outputs/confirmation_report'); TEX=Path('paper/generated')
COLORS={'Llama':'#D35D6E','Qwen':'#008C9B'}


def read(path):
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    for r in rows: r['joint']=bool(r['correct_1'] and r['correct_2'])
    return rows


def interval(m):
    return f"{pct(m['mean'])} [{pct(m['ci95'][0])}, {pct(m['ci95'][1])}]"


def table(name,columns,header,rows,caption):
    tex=(r'\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3pt}'+'\n'+
        r'\begin{tabular}{'+columns+r'}\toprule'+'\n'+header+r'\\\midrule'+'\n'+
        '\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}'+'\n'+
        r'\caption{'+caption+r'}\label{tab:'+name+r'}\end{table}'+'\n')
    (TEX/f'{name}.tex').write_text(style_table(tex))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source',default='/tmp/MQuAKE-CF-3k-v2.json')
    args=parser.parse_args()
    raw={str(r['case_id']):r for r in json.loads(Path(args.source).read_text())}
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#ADB8BE','axes.linewidth':.6,'pdf.fonttype':42})
    summaries={}; aggregate=[]; replicate=[]; robust=[]; edits=[]; macros=[]
    for model,root in ROOTS.items():
        work=root/'confirmation_v1'
        audit=json.loads((work/'final_audit.json').read_text())
        assert len(audit['coverage'])==3
        assert audit['single_edit']['bridge']['direct_string_changes']==0
        reference=[read(root/'predictions'/f'access_seed-{s}.jsonl') for s in SEEDS]
        coverage=[read(work/'both'/f'seed-{s}'/'access.jsonl') for s in SEEDS]
        for rr in reference+coverage:
            for r in rr: r['access_effect']=int(r['correct_pipeline'])-int(r['correct_pipeline_stage2_off'])
        report={'protocols':{},'seeds':{}}
        fields=['correct_1','correct_2','joint','correct_12','correct_pipeline',
                'correct_pipeline_stage2_off','correct_pipeline_path','access_effect']
        for label,rows in [('Reference',reference),('Coverage',coverage)]:
            metrics={f:rate_interval(rows,f) for f in fields}
            report['protocols'][label]=metrics
            aggregate.append(f'{model} & {label} & '+ ' & '.join(pct(metrics[f]['mean']) for f in
                ['correct_1','correct_2','joint'])+' & '+ ' & '.join(interval(metrics[f]) for f in
                ['correct_12','correct_pipeline','access_effect'])+r' \\')
        report['paired_protocol_changes']={}
        for field in fields:
            diffs=[]
            for original,new in zip(reference,coverage):
                assert [r['chain_id'] for r in original]==[r['chain_id'] for r in new]
                diffs.append([r | {'change':float(r[field])-float(b[field])} for b,r in zip(original,new)])
            report['paired_protocol_changes'][field]=rate_interval(diffs,'change')
        for seed,rr in zip(SEEDS,coverage):
            dest=work/'both'/f'seed-{seed}'
            meta=json.loads((dest/'final.json').read_text())
            assert meta['all_eligible_rows_updated'] and all(h['visited']==892 for h in meta['history'])
            audit=paired_audit(read(root/'predictions/original.jsonl'),rr)
            report['seeds'][str(seed)]={'metadata':meta,'conditional':audit}
            vals=[interval(rate_interval([rr],f)) for f in ['correct_12','correct_pipeline_stage2_off','correct_pipeline']]
            c=audit['adapted_C']
            replicate.append(f"{model} & {seed} & {meta['nonzero_rows']}/{meta['eligible_rows']} & "+
                ' & '.join(vals)+f" & {c['numerator']}/{c['denominator']} & {ci(c)}"+r' \\')
        paras=[read(work/'both'/f'seed-{s}'/'paraphrase.jsonl') for s in SEEDS]
        frozen=read(work/'frozen_paraphrase.jsonl'); ids={r['chain_id'] for r in frozen}
        cloze=[[r for r in rr if r['chain_id'] in ids] for rr in coverage]
        frozen_cloze=[r for r in read(root/'predictions/original.jsonl') if r['chain_id'] in ids]
        report['paraphrases']={}
        for label,rows in [('Frozen cloze',[frozen_cloze]),('Frozen questions',[frozen]),('Coverage cloze',cloze),('Coverage questions',paras)]:
            metrics={f:rate_interval(rows,f) for f in ['correct_1','correct_2','joint','correct_12']}
            report['paraphrases'][label]=metrics
            robust.append(f'{model} & {label} & {len(rows[0])} & '+
                ' & '.join(pct(metrics[f]['mean']) for f in ['correct_1','correct_2','joint'])+
                ' & '+interval(metrics['correct_12'])+r' \\')
        report['single_edit']={}
        for scope in ['source','bridge']:
            dest=work/scope/'seed-13'
            pred=read(dest/'consistent.jsonl'); base=read(dest/'consistent_frozen.jsonl'); pipe=read(dest/'consistent_access.jsonl')
            local=read(dest/'predictions.jsonl')
            local_field='correct_1' if scope=='source' else 'correct_2'
            metrics={f:rate_interval([pipe],f) for f in ['correct_12','correct_pipeline','correct_pipeline_stage2_off']}
            metrics['local_edit_hit']=rate_interval([local],local_field)
            metrics['frozen_direct']=rate_interval([base],'correct_12')
            metrics['direct_prediction_changes']=sum(a['prediction_12']!=b['prediction_12'] for a,b in zip(pred,base))
            metrics['second_prediction_changes']=sum(a['prediction_2']!=b['prediction_2'] for a,b in zip(pred,base))
            affected={r['chain_id'] for r in pred if r['e3_id']!=raw[r['chain_id']]['orig']['triples'][1][2]}
            affected_pipe=[r for r in pipe if r['chain_id'] in affected]
            affected_base=[r for r in base if r['chain_id'] in affected]
            metrics['affected_consequences']={'n':len(affected),
                'frozen_direct':rate_interval([affected_base],'correct_12'),
                **{f:rate_interval([affected_pipe],f) for f in ['correct_1','correct_2','correct_12','correct_pipeline','correct_pipeline_stage2_off','access_effect']}}
            metrics['paired_direct_change']=paired_audit(base,pred)['adapted_direct_change']
            metrics['constituent_2']=rate_interval([pred],'correct_2')
            metrics['constituent_1']=rate_interval([pred],'correct_1')
            metrics['n']=len(pred)
            report['single_edit'][scope]=metrics
            edits.append(f"{model} & {scope.capitalize()} only & {len(pred)} & "+
                ' & '.join(pct(metrics[f]['mean']) for f in ['constituent_1','constituent_2'])+' & '+
                ' & '.join(interval(metrics[f]) for f in ['frozen_direct','correct_12','correct_pipeline'])+r' \\')
            if scope=='bridge':
                assert metrics['direct_prediction_changes']==0
                a=metrics['affected_consequences']
                edits.append(f"{model} & Bridge, affected & {a['n']} & "+
                    ' & '.join(pct(a[f]['mean']) for f in ['correct_1','correct_2'])+' & '+
                    ' & '.join(interval(a[f]) for f in ['frozen_direct','correct_12','correct_pipeline'])+r' \\')
        # Both-edits comparison on exactly the same role-separated cases.
        common={r['chain_id'] for r in pred}
        both=[r for r in coverage[0] if r['chain_id'] in common]
        both_base=[r for r in read(root/'predictions/original.jsonl') if r['chain_id'] in common]
        changes=sum(a['prediction_12']!=b['prediction_12'] for a,b in zip(both,both_base))
        edits.append(f'{model} & Both & {len(both)} & '+
            ' & '.join(pct(np.mean([r[f] for r in both])) for f in ['correct_1','correct_2'])+' & '+
            ' & '.join(interval(rate_interval([rr],f)) for rr,f in
                [(both_base,'correct_12'),(both,'correct_12'),(both,'correct_pipeline')])+r' \\')
        summaries[model]=report
        for field,name in [('correct_1','HopA'),('correct_2','HopB'),('joint','Joint'),
                           ('correct_12','Direct'),('correct_pipeline','Pipeline'),
                           ('correct_pipeline_stage2_off','StageOff'),('access_effect','AccessGain')]:
            macros.append(r'\newcommand{\Confirm'+model+name+'}{'+pct(report['protocols']['Coverage'][field]['mean'])+'}')
        bounds=report['protocols']['Coverage']['access_effect']['ci95']
        for label,value in zip(['AccessLow','AccessHigh'],bounds):
            macros.append(r'\newcommand{\Confirm'+model+label+'}{'+pct(value)+'}')
    table('confirmation_aggregate','llrrrrll',r'Model & Protocol & $A_{1a}$ & $A_{1b}$ & $J$ & Direct [95\%] & Pipeline [95\%] & On $-$ off [95\%]',aggregate,
          r'Matched 533-case confirmation, averaging three fits. Coverage visits all 892 unique facts each epoch using answer-balanced loss weights. Intervals resample bridge/answer groups and fits. The last column is a paired access effect in percentage points. These protocols differ in deduplication and weighting as well as coverage.')
    table('confirmation_seeds','llrlllll',r'Model & Seed & Active / eligible & Direct [95\%] & Stage 2 off [95\%] & Pipeline [95\%] & $C$ count & $C$ [95\%]',replicate,
          r'Every coverage-confirmation fit, with group-bootstrap uncertainty. Active counts are entity rows, not chains. Conditional counts retain all 533 evaluated chains. No eligible row remains zero; undefined frozen MQuAKE conditional accuracy is not treated as zero.')
    table('confirmation_paraphrases','llrrrrl',r'Model & Prompt / adapter & Cases & $A_{1a}$ & $A_{1b}$ & $J$ & Direct [95\%]',robust,
          r'Unseen wording: supplied one-hop questions replace training clozes, and the third supplied composed question replaces the first. Comparisons use each model\textquotesingle s identical span-valid cases. Accuracy is percent; coverage conditions average three fits. No paraphrase outcome selects a checkpoint.')
    table('confirmation_edit_scope','llrrrlll',r'Model & Trained edits & Cases & $A_{1a}$ & $A_{1b}$ & Frozen direct & Adapted direct & Pipeline',edits,
          r'Coverage-trained edit-location controls, seed 13. All outcomes in each row use the same cases and condition-consistent facts: edited source/new bridge for source-only, original source/original bridge for bridge-only. The affected subset requires a changed consequence ID. Accuracy is percent; brackets give 95\% group intervals. Target worlds differ across conditions.')
    (TEX/'confirmation_metrics.tex').write_text('\n'.join(macros)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
    for model,report in summaries.items():
        fig,ax=plt.subplots(figsize=(2.7,1.9))
        fields=['correct_1','correct_2','correct_12','correct_pipeline']
        for offset,(label,color) in zip([-.12,.12],[('Reference','#89969E'),('Coverage',COLORS[model])]):
            ms=[report['protocols'][label][f] for f in fields]
            y=np.array([m['mean'] for m in ms])*100; bounds=np.array([m['ci95'] for m in ms])*100
            ax.errorbar(np.arange(4)+offset,y,yerr=[y-bounds[:,0],bounds[:,1]-y],fmt='o',
                linestyle='none',ms=3.5,capsize=2,color=color,label=label)
        ax.set_xticks(range(4),['Hop 1','Hop 2','Direct','Pipeline']); ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(-2,75); ax.grid(axis='y',color='#E8EDF0',lw=.6)
        ax.legend(frameon=False,ncol=2,fontsize=7,loc='lower left',bbox_to_anchor=(0,1.01))
        fig.tight_layout(pad=.35)
        for ext in ['pdf','png']: fig.savefig(OUT/f'coverage_{model.lower()}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.025)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(2.7,1.9))
    for offset,(model,report) in zip([-.12,.12],summaries.items()):
        ms=[report['single_edit'][scope]['paired_direct_change'] for scope in ['source','bridge']]
        y=np.array([m['estimate'] for m in ms])*100; bounds=np.array([m['ci95'] for m in ms])*100
        ax.errorbar(np.arange(2)+offset,y,yerr=[y-bounds[:,0],bounds[:,1]-y],fmt='o',linestyle='none',
                    ms=3.5,capsize=2,color=COLORS[model],label=model)
    ax.axhline(0,color='#89969E',lw=.7,ls=':')
    ax.set_xticks([0,1],['Source only','Bridge only']); ax.set_xlim(-.4,1.4)
    ax.set_ylabel('Direct change (pp)'); ax.set_ylim(-1,15); ax.set_yticks([0,5,10,15])
    ax.grid(axis='y',color='#E8EDF0',lw=.6)
    ax.legend(frameon=False,ncol=2,fontsize=7,loc='lower left',bbox_to_anchor=(0,1.01))
    fig.tight_layout(pad=.35)
    for ext in ['pdf','png']: fig.savefig(OUT/f'edit_location.{ext}',dpi=240,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)
    print(json.dumps({m:{k:v for k,v in r['protocols']['Coverage'].items() if k in ['correct_1','correct_2','correct_12','correct_pipeline','access_effect']} for m,r in summaries.items()},indent=2))


if __name__=='__main__':
    main()
