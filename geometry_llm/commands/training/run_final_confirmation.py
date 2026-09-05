#!/usr/bin/env python
"""Final MQuAKE confirmation; new artifacts, no replacement of prior fits.

Fixed, previously one-hop-selected rates/epochs; coverage-weighted fact sweeps.
Both-edits fits use three seeds; source-only/bridge-only controls use seed 13.
Paraphrases are dataset-supplied question forms, never training or selection data.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from geometry_llm.commands.evaluation.evaluate_access import alias_lookup, stage_two_prompt
from geometry_llm.config import load_config
from geometry_llm.confirmation import unique_weighted_facts, coverage_epoch, weighted_answer_loss, single_edit_targets
from geometry_llm.data import load_saved_chains
from geometry_llm.evaluation import evaluate_chains, write_json, accuracy_summary
from geometry_llm.modeling import (FrozenParameterGuard, ResidualTable, base_row_norms,
    discover_residual_keys, encode_example, pad_batch, save_residual, load_residual,
    load_model_and_tokenizer, greedy_generate_batch, chat_prompt)
from geometry_llm.text import answer_is_correct, token_positions_for_span


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def access(cfg, model, tokenizer, table, chains, predictions, destination):
    if destination.exists():
        return read_rows(destination)
    lookup = alias_lookup(chains)
    old = {r["chain_id"]: r for r in predictions}
    rows = []
    size = cfg["model"]["evaluation_batch_size"]
    for start in range(0, len(chains), size):
        chunk = chains[start:start+size]
        prepared = [stage_two_prompt(c, old[c.chain_id]["prediction_1"], lookup) for c in chunk]
        prompts, entities = map(list, zip(*prepared))
        for j, (prompt, entity) in enumerate(prepared):
            if entity is None or f"entity:{entity}" not in table.key_to_index or len(
                token_positions_for_span(tokenizer, chat_prompt(tokenizer, prompt), entity)) != 1:
                entities[j] = None
        on = greedy_generate_batch(model, tokenizer, table, prompts, entities, cfg["model"]["max_answer_tokens"])
        off = greedy_generate_batch(model, tokenizer, table, prompts, entities, cfg["model"]["max_answer_tokens"], alpha=0)
        for c, prompt, entity, a, b in zip(chunk, prompts, entities, on, off):
            previous = old[c.chain_id]
            rows.append(previous | dict(stage_two_prompt=prompt, resolved_entity=entity,
                prediction_pipeline=a, prediction_pipeline_stage2_off=b,
                correct_pipeline=answer_is_correct(a,c.e3_aliases),
                correct_pipeline_stage2_off=answer_is_correct(b,c.e3_aliases),
                correct_pipeline_path=previous["correct_1"] and answer_is_correct(a,c.e3_aliases),
                access_effect=int(answer_is_correct(a,c.e3_aliases))-int(answer_is_correct(b,c.e3_aliases))))
    destination.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    return rows


def train(cfg, model, tokenizer, chains, run, scope, out):
    seed = run["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = model.get_input_embeddings().weight.device
    hidden = model.get_input_embeddings().embedding_dim
    final = out / "final.pt"
    if final.exists():
        return load_residual(final, hidden, device)[0]
    keys, _ = discover_residual_keys(chains, tokenizer, cfg["data"]["token_mode"])
    table = ResidualTable(keys, hidden, cfg["training"]["alpha"], cfg["data"]["token_mode"]).to(device)
    facts = unique_weighted_facts(chains, scope)
    encoded = [encode_example(tokenizer,r["prompt"],r["entity"],r["answer"],table) for r in facts]
    assert all(any(i >= 0 for i in item.delta_indices) for item in encoded)
    optimizer = torch.optim.AdamW([table.delta],lr=run["lr"],weight_decay=0)
    norms = base_row_norms(model, table)
    guard = FrozenParameterGuard(model)
    size, accumulation = cfg["training"]["batch_size"], cfg["training"]["gradient_accumulation_steps"]
    resume = out / "resume.pt"
    history = []
    if resume.exists():
        saved = torch.load(resume,map_location=device,weights_only=False)
        table.load_state_dict(saved["table"]); optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
    for epoch in range(len(history), run["best_epochs"]):
        began = time.time()
        order = coverage_epoch(facts, seed+epoch)
        visited = set()
        values = []
        optimizer.zero_grad(set_to_none=True)
        chunks = [order[j:j+size] for j in range(0,len(order),size)]
        for j, indices in enumerate(chunks):
            visited.update(indices)
            batch = pad_batch([encoded[i] for i in indices], tokenizer.pad_token_id, device)
            prediction = weighted_answer_loss(model,table,batch,[facts[i]["weight"] for i in indices])
            group_start = (j//accumulation)*accumulation
            group_count = min(accumulation,len(chunks)-group_start)
            loss = (prediction+run["anchor"]*table.anchor_loss(norms))/group_count
            loss.backward()
            assert table.delta.grad is not None and torch.isfinite(table.delta.grad).all()
            if (j+1)%accumulation == 0 or j+1 == len(chunks):
                torch.nn.utils.clip_grad_norm_([table.delta],cfg["training"]["gradient_clip"])
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                guard.assert_unchanged()
            values.append(float(prediction.detach()))
        assert visited == set(range(len(facts)))
        history.append(dict(epoch=epoch+1, unique_facts=len(facts),visited=len(visited),
                            weighted_loss=float(np.mean(values)),seconds=time.time()-began))
        torch.save(dict(table=table.state_dict(),optimizer=optimizer.state_dict(),history=history),resume)
        write_json(out/"history.json",history)
        print(json.dumps(dict(scope=scope,seed=seed,**history[-1])),flush=True)
    active = table.delta.detach().float().norm(dim=1)>0
    eligible = {table.key_to_index[f"entity:{r['entity']}"] for r in facts}
    assert all(active[i] for i in eligible)
    meta = dict(protocol="unique_fact_sweep_weighted_answer_ce_v1", scope=scope,seed=seed,
                learning_rate=run["lr"],anchor=run["anchor"],epochs=run["best_epochs"],
                selection="fixed prior dataset-specific one-hop selection; no new outcome tuning",
                unique_facts=len(facts),eligible_rows=len(eligible),nonzero_rows=int(active.sum()),
                allocated_parameters=table.delta.numel(),active_parameters=int(active.sum())*hidden,
                all_eligible_rows_updated=True,history=history)
    save_residual(final,table,meta)
    write_json(out/"facts.json",facts)
    return table


def paraphrases(chains, raw):
    result=[]
    for c in chains:
        r=raw[c.chain_id]
        # The second listed question sometimes requests both bridge AND answer.
        # Precommit to the third supplied question for answer-only evaluation.
        result.append(replace(c,prompt_1=r["new_single_hops"][0]["question"],
            prompt_2=r["new_single_hops"][1]["question"],prompt_12=r["questions"][2]))
    return result


def controlled_chains(chains,raw,scope):
    source_entities={c.e1 for c in chains}; bridge_entities={c.e2 for c in chains}
    # Role separation prevents a bridge-only row from also editing a source
    # in another case (or the reverse). Keep the exclusion explicit.
    eligible=[c for c in chains if c.e1 not in bridge_entities and c.e2 not in source_entities]
    bridge_edits={}; bridge_edit_ids={}
    for c in chains:
        r=raw[c.chain_id]; t=r["orig"]["new_triples"][1]
        key=(t[0],t[1]); answer=r["new_single_hops"][1]["answer"]
        assert key not in bridge_edits or bridge_edits[key]==answer
        bridge_edits[key]=answer
        bridge_edit_ids[key]=t[2]
    result=[]
    for c in eligible:
        r=raw[c.chain_id]
        bridge,answer=single_edit_targets(r,scope,bridge_edits)
        if scope=="bridge":
            second=r["single_hops"][1]["cloze"]
            ba=list(dict.fromkeys([bridge,*r["single_hops"][0].get("answer_alias",[])]))
            original=r['orig']['triples']
            bridge_id=original[0][2]
            answer_id=bridge_edit_ids.get((original[1][0],original[1][1]),original[1][2])
        else:
            second=c.prompt_2; ba=c.e2_aliases
            bridge_id=r['requested_rewrite'][0]['target_new']['id']
            answer_id=r['requested_rewrite'][1]['target_true']['id'] if scope=='source' else r['requested_rewrite'][1]['target_new']['id']
        aa=[answer]
        if answer==c.e3: aa=c.e3_aliases
        elif answer==r["answer"]: aa=list(dict.fromkeys([answer,*r.get("answer_alias",[])]))
        result.append(replace(c,e2=bridge,e2_aliases=ba,e3=answer,e3_aliases=aa,prompt_2=second,
            e1_id=r['orig']['triples'][0][0],e2_id=bridge_id,e3_id=answer_id))
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--config",required=True); p.add_argument("--root",required=True)
    p.add_argument("--source",default="/tmp/MQuAKE-CF-3k-v2.json")
    p.add_argument("--seeds",nargs="+",type=int,default=[13,37,71])
    p.add_argument("--scopes",nargs="+",choices=["both","source","bridge"],default=["both","source","bridge"])
    args=p.parse_args()
    cfg=load_config(args.config); root=Path(args.root); out=root/"confirmation_v1"
    out.mkdir(exist_ok=True)
    chains=load_saved_chains(root/"selected_chains.jsonl")
    source=Path(args.source); raw={str(r["case_id"]):r for r in json.loads(source.read_text())}
    selection=json.loads((root/"training/correct_delta_selected.json").read_text())
    runs={r["seed"]:r for r in selection["runs"]}
    manifest=dict(protocol="unique_fact_sweep_weighted_answer_ce_v1",model=cfg["model"]["name"],
        chains_sha256=hashlib.sha256((root/"selected_chains.jsonl").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),settings=runs,
        both_seeds=args.seeds,single_edit_seeds=[13],paraphrase_question_index=2)
    if (out/"manifest.json").exists():
        assert json.loads((out/"manifest.json").read_text())==json.loads(json.dumps(manifest))
    write_json(out/"manifest.json",manifest)
    model,tokenizer=load_model_and_tokenizer(cfg)
    paras=paraphrases(chains,raw)
    # Span validity is a pre-outcome check, performed jointly for both interfaces.
    valid=[]; failures=[]
    for c in paras:
        if all(len(token_positions_for_span(tokenizer,chat_prompt(tokenizer,prompt),entity))==1
               for prompt,entity in [(c.prompt_1,c.e1),(c.prompt_2,c.e2),(c.prompt_12,c.e1)]):
            valid.append(c)
        else: failures.append(c.chain_id)
    write_json(out/"paraphrase_selection.json",dict(n=len(valid),excluded=failures,
        policy="third supplied composed question and supplied one-hop questions; unique formatted spans"))
    empty=ResidualTable([],model.get_input_embeddings().embedding_dim,mode="entity_span").to(model.device)
    evaluate_chains(model,tokenizer,empty,valid,"frozen_paraphrase",0,"frozen",
        cfg["model"]["max_answer_tokens"],out/"frozen_paraphrase.jsonl",batch_size=cfg["model"]["evaluation_batch_size"])
    for scope in args.scopes:
        for seed in (args.seeds if scope=="both" else [13]):
            dest=out/scope/f"seed-{seed}"; dest.mkdir(parents=True,exist_ok=True)
            table=train(cfg,model,tokenizer,chains,runs[seed],scope,dest)
            preds=evaluate_chains(model,tokenizer,table,chains,f"coverage_{scope}",seed,str(dest/"final.pt"),
                cfg["model"]["max_answer_tokens"],dest/"predictions.jsonl",batch_size=cfg["model"]["evaluation_batch_size"])
            report=dict(two_edit_target_metrics=accuracy_summary(preds)["all"])
            if scope=="both":
                pipe=access(cfg,model,tokenizer,table,chains,preds,dest/"access.jsonl")
                report["access_counts"]={k:sum(r[k] for r in pipe) for k in
                    ["correct_pipeline","correct_pipeline_stage2_off","correct_pipeline_path"]}
                pp=evaluate_chains(model,tokenizer,table,valid,"coverage_paraphrase",seed,str(dest/"final.pt"),
                    cfg["model"]["max_answer_tokens"],dest/"paraphrase.jsonl",batch_size=cfg["model"]["evaluation_batch_size"])
                report["paraphrase_metrics"]=accuracy_summary(pp)["all"]
            else:
                consistent=controlled_chains(chains,raw,scope)
                pred=evaluate_chains(model,tokenizer,table,consistent,f"{scope}_consistent",seed,str(dest/"final.pt"),
                    cfg["model"]["max_answer_tokens"],dest/"consistent.jsonl",batch_size=cfg["model"]["evaluation_batch_size"])
                frozen=evaluate_chains(model,tokenizer,empty,consistent,f"{scope}_frozen",0,"frozen",
                    cfg["model"]["max_answer_tokens"],dest/"consistent_frozen.jsonl",batch_size=cfg["model"]["evaluation_batch_size"])
                pipe=access(cfg,model,tokenizer,table,consistent,pred,dest/"consistent_access.jsonl")
                report["consistent_metrics"]=accuracy_summary(pred)["all"]
                report["consistent_frozen_metrics"]=accuracy_summary(frozen)["all"]
                report["consistent_n"]=len(consistent)
                report["direct_prediction_changes"]=sum(a["prediction_12"]!=b["prediction_12"] for a,b in zip(pred,frozen))
                report["consistent_access_counts"]={k:sum(r[k] for r in pipe) for k in
                    ["correct_pipeline","correct_pipeline_stage2_off","correct_pipeline_path"]}
            write_json(dest/"summary.json",report)
            print(json.dumps(dict(scope=scope,seed=seed,summary=report)),flush=True)
    print("Confirmation complete",flush=True)


if __name__=="__main__":
    main()
