#!/usr/bin/env python
"""Targeted source-residual swaps for causal control-code testing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import filter_token_mode, load_chains
from geometry_llm.evaluation import accuracy_summary, evaluate_chains, write_json
from geometry_llm.modeling import load_model_and_tokenizer, load_residual
from geometry_llm.text import answer_is_correct


def different_target_donors(chains):
    representatives = {}
    for chain in chains:
        representatives.setdefault(chain.e1, chain)
    sources = sorted(representatives)
    # A global assignment handles repeated bridge targets, for which no single
    # cyclic shift is guaranteed to be a valid derangement.
    cost = np.asarray([
        [int(representatives[source].e2 == representatives[donor].e2) for donor in sources]
        for source in sources
    ])
    source_indices, donor_indices = linear_sum_assignment(cost)
    if cost[source_indices, donor_indices].sum() != 0:
        raise ValueError("Could not construct source swaps with different bridge targets")
    mapping = {sources[source]: sources[donor]
               for source, donor in zip(source_indices, donor_indices)}
    return mapping, representatives


def main():
    parser = common_parser("Swap source residuals toward different factual targets")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    model, tokenizer = load_model_and_tokenizer(cfg)
    chains, _ = filter_token_mode(load_chains(cfg), tokenizer, cfg["data"]["token_mode"])
    device = model.get_input_embeddings().weight.device
    table, metadata = load_residual(Path(args.checkpoint), model.get_input_embeddings().embedding_dim, device)
    entity_key = metadata.get("entity_key", {})
    mapping, representatives = different_target_donors(chains)
    original = table.delta.detach().clone()
    swapped = 0
    for source, donor in mapping.items():
        source_key, donor_key = entity_key.get(source), entity_key.get(donor)
        if source_key in table.key_to_index and donor_key in table.key_to_index:
            table.delta.data[table.key_to_index[source_key]].copy_(original[table.key_to_index[donor_key]])
            swapped += 1
    rows = evaluate_chains(
        model, tokenizer, table, chains, "targeted_source_swap", args.seed, args.checkpoint,
        cfg["model"]["max_answer_tokens"],
        output_path(cfg, "interventions", "targeted_source_swap.jsonl"),
        overwrite=args.overwrite, batch_size=cfg["model"].get("evaluation_batch_size", 8),
    )
    chain_lookup = {chain.chain_id: chain for chain in chains}
    donor_hop_1 = donor_two_hop = 0
    eligible = 0
    for row in rows:
        source_chain = chain_lookup[row["chain_id"]]
        donor_source = mapping.get(source_chain.e1)
        if donor_source is None:
            continue
        donor = representatives[donor_source]
        eligible += 1
        donor_hop_1 += answer_is_correct(row["prediction_1"], donor.e2_aliases)
        donor_two_hop += answer_is_correct(row["prediction_12"], donor.e3_aliases)
    summary = {
        "n_chains": len(rows), "n_source_rows_swapped": swapped,
        "different_bridge_for_every_swap": True,
        "original_target_accuracy": accuracy_summary(rows)["all"],
        "donor_target_hop_1_accuracy": donor_hop_1 / eligible if eligible else None,
        "donor_target_two_hop_accuracy": donor_two_hop / eligible if eligible else None,
        "interpretation": (
            "Evidence for a control code requires predictions to move toward donor targets, "
            "not only a drop in original-target accuracy."
        ),
    }
    write_json(output_path(cfg, "interventions", "targeted_source_swap_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
