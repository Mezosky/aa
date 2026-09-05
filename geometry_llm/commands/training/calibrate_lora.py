#!/usr/bin/env python
"""Match LoRA magnitude to constituent efficacy; composed answers never select scale."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from geometry_llm.config import load_config
from geometry_llm.data import load_saved_chains
from geometry_llm.evaluation import evaluate_chains,accuracy_summary,write_json
from geometry_llm.modeling import load_model_and_tokenizer,ResidualTable
from geometry_llm.commands.training.run_lora_baseline import install_lora,restore,efficacy


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True); p.add_argument('--root',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--scales',type=float,nargs='+',default=[.75,.875,1.0])
    args=p.parse_args(); root=Path(args.root); cfg=load_config(args.config)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    out=root/'lora'/f"seed-{saved['seed']}-scaled"; out.mkdir(parents=True,exist_ok=True)
    model,tok=load_model_and_tokenizer(cfg)
    modules,rank=install_lora(model,saved['residual_parameter_budget']); restore(modules,saved['state'])
    assert rank==saved['rank']
    table=ResidualTable([],model.get_input_embeddings().embedding_dim,alpha=0,mode='entity_span').to(model.get_input_embeddings().weight.device)
    chains=load_saved_chains(root/'selected_chains.jsonl')
    target=np.array(saved['selection']['target']); history=[]
    for scale in args.scales:
        for module in modules.values(): module.scaling=scale
        values=efficacy(model,tok,table,chains,cfg)
        history.append({'scale':scale,'efficacy':values,'efficacy_rmse':float(np.sqrt(np.mean((np.array(values)-target)**2)))})
        print(json.dumps(history[-1]),flush=True)
    selected=min(history,key=lambda row:row['efficacy_rmse'])
    for module in modules.values(): module.scaling=selected['scale']
    saved['selection'] |= selected
    saved['scaling']=selected['scale']; saved['calibration_source']=args.checkpoint
    saved['scale_candidates']=args.scales
    checkpoint=out/'selected.pt'; torch.save(saved,checkpoint)
    rows=evaluate_chains(model,tok,table,chains,'lora',saved['seed'],str(checkpoint),
                         cfg['model']['max_answer_tokens'],out/'predictions.jsonl',
                         batch_size=cfg['model']['evaluation_batch_size'],overwrite=True)
    report={k:v for k,v in saved.items() if k!='state'}
    report['metrics']=accuracy_summary(rows)['all']
    write_json(out/'scale_selection.json',history);write_json(out/'summary.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
