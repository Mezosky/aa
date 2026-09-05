#!/usr/bin/env python
"""One no-training experiment: endogenous intermediate entity, paired row lookup."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from geometry_llm.commands.evaluation.evaluate_access import alias_lookup, read_rows
from geometry_llm.automatic_access import (INSTRUCTION, MAX_ENTITY_TOKENS, MAX_FINAL_TOKENS,
    paired_generate, prepare_prompt, resolve_generated, audit_cache)
from geometry_llm.config import load_config
from geometry_llm.data import load_saved_chains
from geometry_llm.modeling import load_model_and_tokenizer, load_residual, FrozenParameterGuard
from geometry_llm.text import answer_is_correct


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True); p.add_argument('--root',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[13,37,71])
    p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    root=Path(args.root); chains=load_saved_chains(root/'selected_chains.jsonl')
    lookup=alias_lookup(chains)
    out=root/'automatic_access_v1'; out.mkdir(exist_ok=True)
    protocol=dict(instruction=INSTRUCTION, assistant_prefill='Intermediate:',
        source_checkpoint='confirmation_v1/both/seed-{seed}/final.pt',
        entity_token_limit=MAX_ENTITY_TOKENS,final_token_limit=MAX_FINAL_TOKENS,
        decoding='greedy', supplied_decomposition=False,gold_bridge_used_for_generation=False,
        lookup='existing unique global alias dictionary; completed first field only; no gold fallback',
        cache='discard entire intermediate cache; replay original IDs with independently normalized source and generated spans',
        inactive_pair_assertion=True, lookup_dictionary_size=len(lookup),
        lookup_sha256=hashlib.sha256(json.dumps(lookup,sort_keys=True).encode()).hexdigest(),
        data_sha256=hashlib.sha256((root/'selected_chains.jsonl').read_bytes()).hexdigest(),
        batch_size=args.batch_size)
    manifest=out/'protocol.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==protocol
    else: manifest.write_text(json.dumps(protocol,indent=2)+'\n')
    model,tokenizer=load_model_and_tokenizer(load_config(args.config))
    guard=FrozenParameterGuard(model)
    for seed in args.seeds:
        dest=out/f'seed-{seed}.jsonl'
        if dest.exists():
            assert len(read_rows(dest))==len(chains)
            continue
        checkpoint=root/'confirmation_v1/both'/f'seed-{seed}/final.pt'
        table,_=load_residual(checkpoint,model.get_input_embeddings().embedding_dim,
                              model.get_input_embeddings().weight.device)
        table.requires_grad_(False)
        partial=dest.with_suffix('.jsonl.partial')
        rows=read_rows(partial) if partial.exists() else []
        assert [r['chain_id'] for r in rows]==[c.chain_id for c in chains[:len(rows)]]
        audit_path=out/f'cache_audit_seed-{seed}.json'
        for start in range(len(rows),len(chains),args.batch_size):
            began=time.time(); chunk=chains[start:start+args.batch_size]
            generated=paired_generate(model,tokenizer,table,[c.prompt_12 for c in chunk],
                                       [c.e1 for c in chunk],lookup)
            batch=[]
            for c,result in zip(chunk,generated):
                bridge=answer_is_correct(result['intermediate'],c.e2_aliases)
                on=answer_is_correct(result['answer_on'],c.e3_aliases)
                off=answer_is_correct(result['answer_off'],c.e3_aliases)
                row=asdict(c)|result|dict(seed=seed,correct_bridge=bridge,correct_on=on,correct_off=off,
                    correct_path_on=bridge and on,correct_path_off=bridge and off,
                    access_effect=int(on)-int(off),path_effect=int(bridge and on)-int(bridge and off),
                    correct_bridge_accessed=bridge and result['active_nonzero'],
                    format_on=bool(result['answer_on']),format_off=bool(result['answer_off']))
                batch.append(row)
                if result['active_nonzero'] and not audit_path.exists():
                    prefix,spans=prepare_prompt(tokenizer,c.prompt_12,c.e1,table)
                    _,span=resolve_generated(tokenizer,prefix,result['intermediate_ids'],lookup,table)
                    audit={cond:audit_cache(model,tokenizer,table,prefix+result['intermediate_ids'],ss)
                           for cond,ss in [('off',spans),('on',spans+[span])]}
                    audit['chain_id']=c.chain_id; audit['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    audit_path.write_text(json.dumps(audit,indent=2)+'\n')
                    # BF16 cached and full-prefix reductions can differ slightly;
                    # retain numerical/greedy audit rather than hide mismatches.
                    print(json.dumps({'cache_audit':audit}),flush=True)
            with partial.open('a') as handle:
                for row in batch: handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            rows.extend(batch); guard.assert_unchanged()
            print(json.dumps(dict(seed=seed,completed=len(rows),n=len(chains),batch_seconds=time.time()-began,
                bridge=sum(r['correct_bridge'] for r in rows),on=sum(r['correct_on'] for r in rows),
                off=sum(r['correct_off'] for r in rows),active=sum(r['active_nonzero'] for r in rows))),flush=True)
        partial.replace(dest)
        summary=dict(seed=seed,n=len(rows),counts={k:sum(r[k] for r in rows) for k in
            ['correct_bridge','correct_on','correct_off','correct_path_on','correct_path_off',
             'active_nonzero','correct_bridge_accessed','format_on','format_off','access_effect','path_effect']})
        (out/f'seed-{seed}.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__': main()
