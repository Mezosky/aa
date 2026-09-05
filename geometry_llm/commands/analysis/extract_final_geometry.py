#!/usr/bin/env python
"""Cache matched, answer-free representations for all editing interfaces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from geometry_llm.config import load_config
from geometry_llm.data import load_saved_chains
from geometry_llm.modeling import load_model_and_tokenizer, load_residual, encode_example, pad_batch
from geometry_llm.commands.training.run_lora_baseline import install_lora, restore


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--seed',type=int,default=13)
    args=parser.parse_args()
    cfg=load_config(args.config); root=Path(args.root)
    out=root/'final_geometry'/f'seed-{args.seed}'; out.mkdir(parents=True,exist_ok=True)
    model_name='Llama' if 'llama' in cfg['model']['name'].lower() else 'Qwen'
    standalone=json.loads(Path('outputs/access_report/lora_comparison.json').read_text())[model_name]['source']
    joint=json.loads(Path('outputs/access_report/joint_comparison.json').read_text())[model_name]['source']
    pred_path=root/'predictions'/f'correct_delta_seed-{args.seed}.jsonl'
    residual_path=Path(json.loads(pred_path.open().readline())['checkpoint'])
    paths={'residual':residual_path,'lora':Path(standalone).parent/'selected.pt','joint':Path(joint).parent/'selected.pt'}
    manifest={'model':cfg['model']['name'],'seed':args.seed,'config':cfg,
              'checkpoints':{k:{'path':str(v),'sha256':digest(v)} for k,v in paths.items()},
              'chains_sha256':digest(root/'selected_chains.jsonl'),'protocol':'answer-free-v1',
              'sites':'final prompt token; direct-query designated entity-span mean',
              'conditions':['frozen','residual','random','permuted','lora','joint','joint_residual_off','joint_lora_off'],
              'random_seed':123,'permutation':'nonzero residual rows only; reserved zero rows unchanged'}
    manifest_path=out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise RuntimeError('Cached geometry provenance differs; use a new output directory.')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    model,tok=load_model_and_tokenizer(cfg)
    chains=load_saved_chains(root/'selected_chains.jsonl')
    device=model.get_input_embeddings().weight.device
    table,_=load_residual(residual_path,model.get_input_embeddings().embedding_dim,device)
    original=table.delta.detach().clone(); alpha=table.alpha
    saved_lora=torch.load(paths['lora'],map_location='cpu',weights_only=False)
    saved_joint=torch.load(paths['joint'],map_location='cpu',weights_only=False)
    assert table.keys==saved_joint['residual_keys']
    modules,rank=install_lora(model,saved_lora['residual_parameter_budget'])
    assert rank==saved_lora['rank']==saved_joint['rank']
    rng=torch.Generator(device='cpu').manual_seed(123)
    noise=torch.randn(original.shape,generator=rng).to(device)
    random_delta=noise/noise.norm(dim=1,keepdim=True).clamp_min(1e-12)*original.norm(dim=1,keepdim=True)
    active=torch.where(original.norm(dim=1)>0)[0]
    permutation=active[torch.randperm(len(active),generator=rng).to(device)]
    permuted=original.clone(); permuted[active]=original[permutation]
    layers=cfg['analysis']['layers']; last=model.config.num_hidden_layers
    if last not in layers: layers=sorted(set(layers+[last]))
    batch_size=cfg['model']['evaluation_batch_size']
    for condition in manifest['conditions']:
        destination=out/f'{condition}.npz'
        if destination.exists():
            print(json.dumps({'cached':condition}),flush=True); continue
        with torch.no_grad():
            table.delta.copy_(original); table.alpha=alpha
            for m in modules.values(): m.scaling=0.
            if condition=='frozen': table.alpha=0.
            if condition=='random': table.delta.copy_(random_delta)
            if condition=='permuted': table.delta.copy_(permuted)
            if condition=='lora':
                restore(modules,saved_lora['state']); table.alpha=0.
                for m in modules.values(): m.scaling=saved_lora['scaling']
            if condition.startswith('joint'):
                restore(modules,saved_joint['state']); table.load_state_dict(saved_joint['residual_state'])
                table.alpha=0. if condition=='joint_residual_off' else saved_joint['residual_alpha']
                for m in modules.values(): m.scaling=0. if condition=='joint_lora_off' else saved_joint['scaling']
        arrays={}; lengths={}
        with torch.inference_mode():
            for query in ['direct','hop1','hop2']:
                selected_layers=layers if query=='direct' else [last]
                collected={f'{query}_final_{l}':[] for l in selected_layers}
                if query=='direct': collected.update({f'direct_entity_{l}':[] for l in selected_layers})
                lengths[query]=[]
                for start in range(0,len(chains),batch_size):
                    chunk=chains[start:start+batch_size]
                    prompts=[c.prompt_12 if query=='direct' else c.prompt_1 if query=='hop1' else c.prompt_2 for c in chunk]
                    entities=[c.e2 if query=='hop2' else c.e1 for c in chunk]
                    items=[encode_example(tok,p,e,None,table) for p,e in zip(prompts,entities)]
                    batch=pad_batch(items,tok.pad_token_id,device)
                    embedded=table(model.get_input_embeddings()(batch['input_ids']),batch['delta_indices'])
                    hidden=model.model(inputs_embeds=embedded,attention_mask=batch['attention_mask'],
                                       output_hidden_states=True,use_cache=False,return_dict=True).hidden_states
                    lengths[query].extend(item.prompt_length for item in items)
                    for l in selected_layers:
                        ends=torch.tensor([item.prompt_length-1 for item in items],device=device)
                        collected[f'{query}_final_{l}'].append(hidden[l][torch.arange(len(items),device=device),ends].float().cpu().numpy())
                        if query=='direct':
                            for j,item in enumerate(items):
                                positions=[i for i,k in enumerate(item.delta_indices) if k>=0]
                                assert positions
                                collected[f'direct_entity_{l}'].append(hidden[l][j,positions].float().mean(0).cpu().numpy()[None])
                    del hidden
                arrays.update({k:np.concatenate(v) for k,v in collected.items()})
                print(json.dumps({'condition':condition,'query':query,'n':len(chains)}),flush=True)
        arrays.update(chain_ids=np.array([str(c.chain_id) for c in chains]),layers=np.array(layers),
                      **{f'{q}_lengths':np.array(v) for q,v in lengths.items()})
        np.savez_compressed(destination,**arrays)
    print(f'Complete: {out}',flush=True)


if __name__=='__main__': main()
