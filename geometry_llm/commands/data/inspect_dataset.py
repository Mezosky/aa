#!/usr/bin/env python
from __future__ import annotations

from collections import Counter

from transformers import AutoTokenizer

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import save_chains
from geometry_llm.evaluation import write_json
from geometry_llm.modeling import configure_tokenizer
from geometry_llm.text import token_positions_for_span


def main():
    args = common_parser("Inspect and select a SOCRATES composition").parse_args()
    cfg = load_config(args.config, args.set)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], use_fast=True)
    configure_tokenizer(tokenizer, cfg)
    # Load all types first; selection is reported by load_chains.
    from datasets import load_dataset
    from geometry_llm.data import row_to_chain, select_composition, filter_token_mode
    ds = load_dataset(cfg["data"]["dataset"], split=cfg["data"]["split"])
    all_chains = [row_to_chain(dict(row), i) for i, row in enumerate(ds)]
    counts = Counter(c.fact_comp_type for c in all_chains)
    selected = cfg["data"].get("composition_type", "auto")
    if selected == "auto":
        from pathlib import Path
        selection_file = cfg["data"].get("selection_file")
        if selection_file and Path(selection_file).exists():
            import json
            selected = json.loads(Path(selection_file).read_text())["selected_fact_comp_type"]
        else:
            selected = select_composition(all_chains, cfg["data"].get("min_examples", 40))
        cfg["data"]["resolved_composition_type"] = selected
    subset = [c for c in all_chains if c.fact_comp_type == selected]
    maximum = cfg["data"].get("max_examples")
    if maximum:
        subset = subset[:maximum]
    kept, failures = filter_token_mode(subset, tokenizer, cfg["data"]["token_mode"])
    token_frequency = Counter()
    token_entities = {}
    single = 0
    contextual_single = 0
    for c in subset:
        is_single = True
        for entity in (c.e1, c.e2, c.e3):
            ids = tokenizer(entity, add_special_tokens=False)["input_ids"]
            is_single &= len(ids) == 1
            for token in ids:
                token_frequency[str(token)] += 1
                token_entities.setdefault(str(token), set()).add(entity)
        single += int(is_single)
        contextual_single += int(
            len(token_positions_for_span(tokenizer, c.prompt_1, c.e1)) == 1
            and len(token_positions_for_span(tokenizer, c.prompt_1, c.e1)[0]) == 1
            and len(token_positions_for_span(tokenizer, c.prompt_2, c.e2)) == 1
            and len(token_positions_for_span(tokenizer, c.prompt_2, c.e2)[0]) == 1
            and len(tokenizer(c.e2, add_special_tokens=False)["input_ids"]) == 1
            and len(tokenizer(c.e3, add_special_tokens=False)["input_ids"]) == 1
        )
    report = {
        "model": cfg["model"]["name"], "dataset": cfg["data"]["dataset"],
        "examples_per_fact_comp_type": dict(counts), "selected_fact_comp_type": selected,
        "selected_before_token_filter": len(subset), "selected_after_token_filter": len(kept),
        "span_failures": failures,
        "overall_unique_entities": {f"e{i}": len({getattr(c, f'e{i}') for c in all_chains}) for i in (1, 2, 3)},
        "unique_entities": {f"e{i}": len({getattr(c, f'e{i}') for c in subset}) for i in (1, 2, 3)},
        "standalone_single_token_chains": single,
        "contextual_single_token_entity_and_answer_chains": contextual_single,
        "entity_token_frequency": dict(token_frequency),
        "shared_subword_tokens": {
            token: sorted(entities) for token, entities in token_entities.items() if len(entities) > 1
        },
        "n_shared_subword_tokens": sum(len(v) > 1 for v in token_entities.values()),
    }
    write_json(output_path(cfg, "inspection.json"), report)
    save_chains(kept, output_path(cfg, "selected_chains.jsonl"))
    print(f"Selected {selected}: {len(kept)}/{len(subset)} chains; report in {output_path(cfg)}")


if __name__ == "__main__":
    main()
