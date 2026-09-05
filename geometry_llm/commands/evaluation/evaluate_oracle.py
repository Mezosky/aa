#!/usr/bin/env python
"""Evaluate whether the frozen model composes two facts stated explicitly in context."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict

import numpy as np
from tqdm import tqdm

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import filter_token_mode, load_chains, load_saved_chains, save_chains
from geometry_llm.evaluation import write_json
from geometry_llm.modeling import ResidualTable, greedy_generate_batch, load_model_and_tokenizer
from geometry_llm.text import answer_is_correct


def oracle_prompt(chain) -> str:
    fact_1 = f"{chain.prompt_1.rstrip().rstrip('.')} {chain.e2}."
    fact_2 = f"{chain.prompt_2.rstrip().rstrip('.')} {chain.e3}."
    return (
        "Use the two stated facts to answer the question.\n"
        f"Fact 1: {fact_1}\n"
        f"Fact 2: {fact_2}\n"
        f"Question: {chain.prompt_12}\n"
        "Answer with only the answer entity."
    )


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["fact_comp_type"]].append(row)
    relation_types = {
        relation: {
            "n": len(items),
            "accuracy": float(np.mean([item["correct_oracle"] for item in items])),
        }
        for relation, items in sorted(grouped.items())
    }
    return {
        "n": len(rows),
        "accuracy": float(np.mean([row["correct_oracle"] for row in rows])) if rows else None,
        "by_relation_type": relation_types,
    }


def main():
    parser = common_parser("Evaluate an explicit-facts composition oracle")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    model, tokenizer = load_model_and_tokenizer(cfg)
    selected_path = output_path(cfg, "selected_chains.jsonl")
    if selected_path.exists():
        chains = load_saved_chains(selected_path)
    else:
        chains, _ = filter_token_mode(load_chains(cfg), tokenizer, cfg["data"]["token_mode"])
        save_chains(chains, selected_path)

    hidden = model.get_input_embeddings().embedding_dim
    device = model.get_input_embeddings().weight.device
    table = ResidualTable([], hidden, alpha=0, mode=cfg["data"]["token_mode"]).to(device)
    destination = output_path(cfg, "predictions", "oracle_explicit_facts.jsonl")
    partial = destination.with_suffix(destination.suffix + ".partial")
    if args.overwrite:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    if destination.exists():
        rows = [json.loads(line) for line in destination.open(encoding="utf-8")]
    else:
        rows = ([json.loads(line) for line in partial.open(encoding="utf-8")]
                if partial.exists() else [])
        observed = [row["chain_id"] for row in rows]
        expected = [chain.chain_id for chain in chains[:len(rows)]]
        if observed != expected:
            raise RuntimeError("Oracle partial file does not match the selected chains")
        batch_size = cfg["model"].get("evaluation_batch_size", 8)
        for start in tqdm(range(len(rows), len(chains), batch_size),
                          desc="explicit-facts oracle"):
            chunk = chains[start:start + batch_size]
            prompts = [oracle_prompt(chain) for chain in chunk]
            predictions = greedy_generate_batch(
                model, tokenizer, table, prompts, [None] * len(chunk),
                cfg["model"]["max_answer_tokens"],
            )
            batch_rows = [
                asdict(chain) | {
                    "oracle_prompt": prompt,
                    "prediction_oracle": prediction,
                    "correct_oracle": answer_is_correct(prediction, chain.e3_aliases),
                }
                for chain, prompt, prediction in zip(chunk, prompts, predictions)
            ]
            partial.parent.mkdir(parents=True, exist_ok=True)
            with partial.open("a", encoding="utf-8") as handle:
                for row in batch_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            rows.extend(batch_rows)
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        partial.unlink(missing_ok=True)

    report = summarize(rows)
    baseline_path = output_path(cfg, "baseline_summary.json")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())["overall"]["all"]["A_2"]
        report["frozen_direct_accuracy"] = baseline
        report["oracle_gain"] = report["accuracy"] - baseline
    write_json(output_path(cfg, "summaries", "oracle_explicit_facts.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
