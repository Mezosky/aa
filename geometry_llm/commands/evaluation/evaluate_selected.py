#!/usr/bin/env python
"""Evaluate every final checkpoint from a validation-selected multi-seed run."""
from __future__ import annotations

import json

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import filter_token_mode, load_chains
from geometry_llm.evaluation import accuracy_summary, evaluate_chains, write_json
from geometry_llm.metrics import summarize_seed_metrics
from geometry_llm.modeling import load_model_and_tokenizer, load_residual


def main():
    parser = common_parser("Evaluate all validation-selected residual checkpoints")
    parser.add_argument("--condition", choices=["correct_delta", "shuffled_delta"],
                        default="correct_delta")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    selection_path = output_path(cfg, "training", f"{args.condition}_selected.json")
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing selected run: {selection_path}")
    selection = json.loads(selection_path.read_text())
    if selection.get("selection_uses_two_hop") is not False:
        raise RuntimeError("Refusing a selection that is not explicitly one-hop-only")

    model, tokenizer = load_model_and_tokenizer(cfg)
    chains, _ = filter_token_mode(load_chains(cfg), tokenizer, cfg["data"]["token_mode"])
    device = model.get_input_embeddings().weight.device
    hidden = model.get_input_embeddings().embedding_dim
    baseline_path = output_path(cfg, "predictions", "original.jsonl")
    baseline = ([json.loads(line) for line in baseline_path.open(encoding="utf-8")]
                if baseline_path.exists() else None)
    summaries = []
    for run in selection["runs"]:
        seed = int(run["seed"])
        table, _ = load_residual(run["checkpoint"], hidden, device)
        rows = evaluate_chains(
            model, tokenizer, table, chains, args.condition, seed, run["checkpoint"],
            cfg["model"]["max_answer_tokens"],
            output_path(cfg, "predictions", f"{args.condition}_seed-{seed}.jsonl"),
            overwrite=args.overwrite,
            batch_size=cfg["model"].get("evaluation_batch_size", 8),
        )
        summary = accuracy_summary(rows, baseline)
        write_json(output_path(cfg, "summaries", f"{args.condition}_seed-{seed}.json"), summary)
        summaries.append(summary)

    aggregate = {}
    metrics = ["A_1a", "A_1b", "A_2", "A_explicit", "J_1",
               "A_1_independent", "C", "A_2_given_adapted_one_hops"]
    for subset in ("all", "knowledge_conditioned"):
        values = [{key: value for key, value in summary[subset].items() if value is not None}
                  for summary in summaries]
        aggregate[subset] = summarize_seed_metrics(
            values, metrics, cfg["analysis"]["bootstrap_samples"])
    destination = output_path(cfg, "summaries", f"{args.condition}_aggregate.json")
    write_json(destination, aggregate)
    print(json.dumps({"runs": len(summaries), "output": str(destination),
                      "aggregate": aggregate["all"]}, indent=2))


if __name__ == "__main__":
    main()
