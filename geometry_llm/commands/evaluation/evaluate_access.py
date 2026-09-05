#!/usr/bin/env python
"""Execute a generated-bridge pipeline, with and without second-stage row access.

The resolver uses one global alias dictionary, never the current gold bridge.
Unknown or ambiguous predictions remain literal and receive no residual.
The second relation/template is supplied, so this is an access control, not a
test of autonomous question decomposition.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from geometry_llm.config import load_config
from geometry_llm.data import load_saved_chains
from geometry_llm.evaluation import write_json
from geometry_llm.modeling import (chat_prompt, encode_example, greedy_generate_batch,
                                  load_model_and_tokenizer, load_residual)
from geometry_llm.text import answer_is_correct, find_entity_spans, normalize_answer, token_positions_for_span


def alias_lookup(chains):
    candidates = defaultdict(set)
    for c in chains:
        for entity, aliases in [(c.e1, [c.e1]), (c.e2, c.e2_aliases), (c.e3, c.e3_aliases)]:
            for alias in [entity, *aliases]:
                candidates[normalize_answer(alias)].add(entity)
    return {key: next(iter(values)) for key, values in candidates.items() if len(values) == 1}


def stage_two_prompt(chain, prediction, lookup):
    spans = find_entity_spans(chain.prompt_2, chain.e2)
    if len(spans) != 1:
        raise ValueError(f"Non-unique bridge slot in chain {chain.chain_id}")
    resolved = lookup.get(normalize_answer(prediction))
    inserted = resolved if resolved is not None else prediction.strip() or "[unknown]"
    start, end = spans[0]
    return chain.prompt_2[:start] + inserted + chain.prompt_2[end:], resolved


def read_rows(path):
    return [json.loads(line) for line in Path(path).open()]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 71])
    args = p.parse_args()
    cfg, root = load_config(args.config), Path(args.root)
    chains = load_saved_chains(root / "selected_chains.jsonl")
    lookup = alias_lookup(chains)
    model, tokenizer = load_model_and_tokenizer(cfg)
    hidden, device = model.get_input_embeddings().embedding_dim, model.get_input_embeddings().weight.device
    selection = json.loads((root / "training/correct_delta_selected.json").read_text())
    runs = {int(r["seed"]): r for r in selection["runs"]}
    for seed in args.seeds:
        destination = root / "predictions" / f"access_seed-{seed}.jsonl"
        if destination.exists():
            continue
        table, _ = load_residual(runs[seed]["checkpoint"], hidden, device)
        predictions = {r["chain_id"]: r for r in read_rows(root / "predictions" / f"correct_delta_seed-{seed}.jsonl")}
        partial = destination.with_suffix(".jsonl.partial")
        rows = read_rows(partial) if partial.exists() else []
        if [r["chain_id"] for r in rows] != [c.chain_id for c in chains[:len(rows)]]:
            raise ValueError("Access partial file does not match selected chains")
        batch_size = cfg["model"].get("evaluation_batch_size", 8)
        for start in tqdm(range(len(rows), len(chains), batch_size), desc=f"access seed {seed}"):
            chunk = chains[start:start + batch_size]
            prepared = [stage_two_prompt(c, predictions[c.chain_id]["prediction_1"], lookup) for c in chunk]
            prompts, entities = map(list, zip(*prepared))
            reasons = []
            for j, (prompt, entity) in enumerate(prepared):
                valid = (entity is not None and f"entity:{entity}" in table.key_to_index
                         and len(token_positions_for_span(tokenizer, chat_prompt(tokenizer, prompt), entity)) == 1)
                reasons.append("resolved_unique_span" if valid else "unresolved_or_ambiguous_span")
                if not valid:
                    entities[j] = None
            on = greedy_generate_batch(model, tokenizer, table, prompts, entities, cfg["model"]["max_answer_tokens"])
            off = greedy_generate_batch(model, tokenizer, table, prompts, entities, cfg["model"]["max_answer_tokens"], alpha=0)
            batch = []
            for c, prompt, entity, reason, a, b in zip(chunk, prompts, entities, reasons, on, off):
                direct = encode_example(tokenizer, c.prompt_12, c.e1, None, table)
                active = sorted({table.keys[i] for i in direct.delta_indices if i >= 0})
                old = predictions[c.chain_id]
                batch.append(old | {
                    "stage_two_prompt": prompt, "resolved_entity": entity, "resolution_status": reason,
                    "direct_active_rows": active,
                    "bridge_row_active_direct": f"entity:{c.e2}" in active,
                    "bridge_surface_present_direct": bool(find_entity_spans(c.prompt_12, c.e2)),
                    "prediction_pipeline": a, "prediction_pipeline_stage2_off": b,
                    "correct_pipeline": answer_is_correct(a, c.e3_aliases),
                    "correct_pipeline_stage2_off": answer_is_correct(b, c.e3_aliases),
                    "correct_pipeline_path": old["correct_1"] and answer_is_correct(a, c.e3_aliases),
                    "correct_pipeline_path_stage2_off": old["correct_1"] and answer_is_correct(b, c.e3_aliases),
                })
            with partial.open("a") as handle:
                for row in batch:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.extend(batch)
        partial.replace(destination)
        fields = ["correct_12", "correct_pipeline", "correct_pipeline_stage2_off", "correct_pipeline_path",
                  "correct_pipeline_path_stage2_off", "bridge_row_active_direct", "bridge_surface_present_direct"]
        report = {"n": len(rows), "seed": seed, "lookup_policy": "unique global alias; no gold fallback",
                  "supplied_decomposition": True,
                  "counts": {k: sum(r[k] for r in rows) for k in fields}}
        write_json(root / "summaries" / f"access_seed-{seed}.json", report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
