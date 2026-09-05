#!/usr/bin/env python
"""LoRA comparison selected by constituent efficacy, never by composed answers.

Rank is the nearest integer budget match over q_proj/v_proj. Select the epoch
closest to the residual's two one-hop accuracies and joint constituent coverage
on the installed facts. This is installation-efficacy matching, not a held-out
generalization estimate. Predeclared candidates: 1, 2, 4, 8 epochs at lr=1e-4.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from geometry_llm.config import load_config
from geometry_llm.data import load_saved_chains
from geometry_llm.evaluation import evaluate_chains, accuracy_summary, write_json
from geometry_llm.modeling import (ResidualTable, encode_example, pad_batch, residual_forward,
                                  load_model_and_tokenizer, greedy_generate_batch,
                                  discover_residual_keys, base_row_norms)
from geometry_llm.text import answer_is_correct
from geometry_llm.commands.training.train_delta import records, balanced_epoch


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.scaling = 1.0
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        # LoRA scaling alpha/rank = 1; accumulators remain float32.
        return self.base(x) + self.scaling * F.linear(F.linear(x.float(), self.a), self.b).to(x.dtype)


def install_lora(model, budget):
    targets = [(name,module) for name,module in model.named_modules()
               if name.rsplit('.',1)[-1] in ('q_proj','v_proj') and isinstance(module, nn.Linear)]
    unit = sum(m.in_features + m.out_features for _,m in targets)
    rank = max(1, round(budget/unit))
    modules = {}
    for name,base in targets:
        parent, attr = name.rsplit('.',1)
        layer = LoRALinear(base,rank)
        setattr(model.get_submodule(parent), attr, layer)
        modules[name] = layer
    return modules, rank


def state(modules):
    return {name:{'a':m.a.detach().cpu().clone(), 'b':m.b.detach().cpu().clone()} for name,m in modules.items()}


def restore(modules, saved):
    with torch.no_grad():
        for name,m in modules.items():
            m.a.copy_(saved[name]['a']); m.b.copy_(saved[name]['b'])


def efficacy(model, tokenizer, table, chains, cfg):
    results = []
    batch_size = cfg['model'].get('evaluation_batch_size',8)
    for start in range(0,len(chains),batch_size):
        chunk = chains[start:start+batch_size]
        prompts = [p for c in chunk for p in (c.prompt_1,c.prompt_2)]
        entities = [e for c in chunk for e in (c.e1,c.e2)]
        preds = greedy_generate_batch(model,tokenizer,table,prompts,entities,cfg['model']['max_answer_tokens'])
        results.extend([[answer_is_correct(preds[2*j],c.e2_aliases),
                         answer_is_correct(preds[2*j+1],c.e3_aliases)] for j,c in enumerate(chunk)])
    arr=np.asarray(results)
    return [float(arr[:,0].mean()),float(arr[:,1].mean()),float(arr.all(1).mean())]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True); p.add_argument('--root',required=True)
    p.add_argument('--seed',type=int,default=13)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--joint-residual',action='store_true',
                   help='Jointly train zero-initialized entity residuals and LoRA; combined budget is larger.')
    args=p.parse_args()
    cfg,root=load_config(args.config),Path(args.root)
    suffix='' if args.learning_rate==1e-4 else f'-lr-{args.learning_rate:g}'
    if args.epochs != 8: suffix+=f'-epochs-{args.epochs}'
    out=root/('joint_lora' if args.joint_residual else 'lora')/f'seed-{args.seed}{suffix}'
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed)
    model,tokenizer=load_model_and_tokenizer(cfg)
    chains=load_saved_chains(root/'selected_chains.jsonl')
    residual_rows=[json.loads(l) for l in (root/'predictions'/f'correct_delta_seed-{args.seed}.jsonl').open()]
    checkpoint=torch.load(residual_rows[0]['checkpoint'],map_location='cpu',weights_only=False)
    budget=checkpoint['state_dict']['delta'].numel()
    residual_lr=checkpoint['metadata']['lr']
    residual_anchor=checkpoint['metadata']['anchor']
    del checkpoint
    modules,rank=install_lora(model,budget)
    params=[p for m in modules.values() for p in [m.a,m.b]]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == sum(p.numel() for p in params)
    hidden,device=model.get_input_embeddings().embedding_dim,model.get_input_embeddings().weight.device
    keys,_=discover_residual_keys(chains,tokenizer,cfg['data']['token_mode']) if args.joint_residual else ([],{})
    table=ResidualTable(keys,hidden,alpha=cfg['training']['alpha'] if args.joint_residual else 0,
                        mode=cfg['data']['token_mode']).to(device)
    lora_count=sum(p.numel() for p in params)
    if args.joint_residual:
        assert table.delta.numel()==budget
        params.append(table.delta)
    norms=base_row_norms(model,table) if args.joint_residual else None
    target=[float(np.mean([r[k] for r in residual_rows])) for k in ('correct_1','correct_2')]
    target.append(float(np.mean([r['correct_1'] and r['correct_2'] for r in residual_rows])))
    destination=out/'selected.pt'
    if destination.exists():
        saved=torch.load(destination,map_location='cpu',weights_only=False)
        restore(modules,saved['state'])
        if args.joint_residual: table.load_state_dict(saved['residual_state'])
    else:
        groups=[{'params':[p for m in modules.values() for p in (m.a,m.b)],'lr':args.learning_rate}]
        if args.joint_residual: groups.append({'params':[table.delta],'lr':residual_lr})
        optimizer=torch.optim.AdamW(groups,weight_decay=0)
        rows=records(chains,False,args.seed)
        history,best=[],float('inf')
        for epoch in range(1,args.epochs+1):
            epoch_rows=balanced_epoch(rows,args.seed+epoch-1)
            batch_size=cfg['training']['batch_size']; accumulation=cfg['training']['gradient_accumulation_steps']
            optimizer.zero_grad(set_to_none=True)
            for start in range(0,len(epoch_rows),batch_size):
                chunk=epoch_rows[start:start+batch_size]
                items=[encode_example(tokenizer,r['prompt'],r['entity'],r['answer'],table) for r in chunk]
                batch=pad_batch(items,tokenizer.pad_token_id,device)
                loss=residual_forward(model,table,batch).loss
                if args.joint_residual: loss=loss+residual_anchor*table.anchor_loss(norms)
                loss=loss/accumulation
                loss.backward()
                if args.joint_residual:
                    assert table.delta.grad is not None and torch.isfinite(table.delta.grad).all()
                if ((start//batch_size)+1)%accumulation==0 or start+batch_size>=len(epoch_rows):
                    torch.nn.utils.clip_grad_norm_(params,1.0)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
            print(json.dumps({'epoch':epoch,'train_loss':float(loss.detach())*accumulation}),flush=True)
            if epoch not in (1,2,4,args.epochs):
                continue
            values=efficacy(model,tokenizer,table,chains,cfg)
            distance=float(np.sqrt(np.mean((np.asarray(values)-target)**2)))
            row={'epoch':epoch,'efficacy':values,'target':target,'efficacy_rmse':distance}
            history.append(row); write_json(out/'efficacy_selection.json',history)
            print(json.dumps(row),flush=True)
            if distance < best:
                best=distance
                saved={'state':state(modules),'selection':row,'rank':rank,'scaling':1.0,
                       'residual_parameter_budget':budget,'lora_parameters':lora_count,
                       'learning_rate':args.learning_rate,'selection_uses_two_hop':False,
                       'selection_population':'installed one-hop facts; not held-out generalization',
                       'seed':args.seed,'model':cfg['model']['name']}
                if args.joint_residual:
                    saved.update(residual_state={k:v.detach().cpu().clone() for k,v in table.state_dict().items()},
                                 residual_keys=table.keys,residual_alpha=table.alpha,residual_mode=table.mode,
                                 residual_learning_rate=residual_lr,residual_anchor=residual_anchor,
                                 joint_parameters=budget+lora_count,initialization='both adapters from zero update')
                torch.save(saved,destination)
        restore(modules,saved['state'])
        if args.joint_residual: table.load_state_dict(saved['residual_state'])
    condition='joint_delta_lora' if args.joint_residual else 'lora'
    preds=evaluate_chains(model,tokenizer,table,chains,condition,args.seed,str(destination),
                          cfg['model']['max_answer_tokens'],out/'predictions.jsonl',batch_size=cfg['model']['evaluation_batch_size'])
    report={k:v for k,v in saved.items() if k not in ('state','residual_state','residual_keys')}
    report['metrics']=accuracy_summary(preds)['all']
    write_json(out/'summary.json',report)
    if args.joint_residual:
        # Paired removals of components of the same jointly trained checkpoint.
        for removed in ('residual','lora'):
            table.alpha=0 if removed=='residual' else saved['residual_alpha']
            for module in modules.values(): module.scaling=0 if removed=='lora' else saved['scaling']
            ablated=evaluate_chains(model,tokenizer,table,chains,f'{condition}_{removed}_off',args.seed,
                                    str(destination),cfg['model']['max_answer_tokens'],out/f'{removed}_off.jsonl',
                                    batch_size=cfg['model']['evaluation_batch_size'])
            report[f'{removed}_off_metrics']=accuracy_summary(ablated)['all']
            write_json(out/'summary.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
