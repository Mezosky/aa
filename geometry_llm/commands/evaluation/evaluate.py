#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import torch

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import filter_token_mode, load_chains
from geometry_llm.evaluation import accuracy_summary, evaluate_chains, write_json
from geometry_llm.metrics import summarize_seed_metrics
from geometry_llm.modeling import ResidualTable, load_model_and_tokenizer, load_residual


def main():
    parser = common_parser("Evaluate original, trained, shuffled or random residuals")
    parser.add_argument("--condition", choices=["original", "correct_delta", "shuffled_delta", "random_delta"], required=True)
    parser.add_argument("--checkpoint", help="Trained checkpoint (also supplies norms for random_delta)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    model, tokenizer = load_model_and_tokenizer(cfg)
    chains, _ = filter_token_mode(load_chains(cfg), tokenizer, cfg["data"]["token_mode"])
    device = model.get_input_embeddings().weight.device
    hidden = model.get_input_embeddings().embedding_dim
    if args.condition == "original":
        table, metadata = ResidualTable([], hidden, 0, cfg["data"]["token_mode"]).to(device), {}
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required for adapted/random conditions")
        table, metadata = load_residual(Path(args.checkpoint), hidden, device)
        if args.condition == "random_delta":
            generator = torch.Generator(device=device).manual_seed(args.seed)
            random_rows = torch.randn(table.delta.shape, generator=generator, device=device)
            target_norms = table.delta.detach().float().norm(dim=1, keepdim=True)
            random_rows = random_rows / random_rows.float().norm(dim=1, keepdim=True).clamp_min(1e-12) * target_norms
            table.delta.data.copy_(random_rows.to(table.delta.dtype))
    filename = f"{args.condition}_seed-{args.seed}.jsonl"
    rows = evaluate_chains(model, tokenizer, table, chains, args.condition, args.seed,
                           str(args.checkpoint or "original"), cfg["model"]["max_answer_tokens"],
                           output_path(cfg, "predictions", filename), overwrite=args.overwrite,
                           batch_size=cfg["model"].get("evaluation_batch_size", 8))
    baseline_path = output_path(cfg, "predictions", "original.jsonl")
    baseline = [json.loads(line) for line in baseline_path.open()] if baseline_path.exists() else None
    summary = accuracy_summary(rows, baseline)
    write_json(output_path(cfg, "summaries", filename.replace(".jsonl", ".json")), summary)
    # Refresh condition-level statistics as seeds accumulate.
    seed_rows = {"all": [], "knowledge_conditioned": []}
    for path in output_path(cfg, "summaries").glob(f"{args.condition}_seed-*.json"):
        item = json.loads(path.read_text())
        for subset in seed_rows:
            if subset in item:
                seed_rows[subset].append({k: v for k, v in item[subset].items() if v is not None})
    aggregate = {
        subset: summarize_seed_metrics(values, ["A_1a", "A_1b", "A_2", "A_explicit", "J_1",
                                               "A_1_independent", "C", "A_2_given_adapted_one_hops"],
                                       cfg["analysis"]["bootstrap_samples"])
        for subset, values in seed_rows.items()
    }
    write_json(output_path(cfg, "summaries", f"{args.condition}_aggregate.json"), aggregate)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
