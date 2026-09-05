#!/usr/bin/env python
from __future__ import annotations

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import load_chains, filter_token_mode, save_chains
from geometry_llm.evaluation import accuracy_summary, evaluate_chains, write_json
from geometry_llm.modeling import ResidualTable, load_model_and_tokenizer


def main():
    parser = common_parser("Evaluate the frozen baseline")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--all-compositions", action="store_true",
                        help="Report the requested baseline for every composition (expensive)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    model, tokenizer = load_model_and_tokenizer(cfg)
    if args.all_compositions:
        from datasets import load_dataset
        from geometry_llm.data import row_to_chain
        dataset = load_dataset(cfg["data"]["dataset"], split=cfg["data"]["split"])
        raw = [row_to_chain(dict(row), i) for i, row in enumerate(dataset)]
        if not 0 <= args.shard_index < args.num_shards:
            parser.error("--shard-index must be in [0, --num-shards)")
        kinds = sorted({chain.fact_comp_type for chain in raw})
        assigned = set(kinds[args.shard_index::args.num_shards])
        raw = [chain for chain in raw if chain.fact_comp_type in assigned]
        chains, failures = filter_token_mode(raw, tokenizer, cfg["data"]["token_mode"])
    else:
        chains, failures = filter_token_mode(load_chains(cfg), tokenizer, cfg["data"]["token_mode"])
    if not args.all_compositions:
        save_chains(chains, output_path(cfg, "selected_chains.jsonl"))
    hidden = model.get_input_embeddings().embedding_dim
    table = ResidualTable([], hidden, alpha=0, mode=cfg["data"]["token_mode"]).to(
        model.get_input_embeddings().weight.device)
    all_suffix = (f"_shard-{args.shard_index}-of-{args.num_shards}" if args.num_shards > 1 else "")
    rows = evaluate_chains(
        model, tokenizer, table, chains, "original", 0, "original",
        cfg["model"]["max_answer_tokens"], output_path(cfg, "predictions", f"original_all{all_suffix}.jsonl" if args.all_compositions else "original.jsonl"),
        overwrite=args.overwrite, batch_size=cfg["model"].get("evaluation_batch_size", 8),
    )
    # Include per-composition output even when a manual config later combines types.
    summary = {"overall": accuracy_summary(rows)}
    for kind in sorted({r["fact_comp_type"] for r in rows}):
        summary[kind] = accuracy_summary([r for r in rows if r["fact_comp_type"] == kind])
    summary["span_failures"] = failures
    write_json(output_path(cfg, f"baseline_all_summary{all_suffix}.json" if args.all_compositions else "baseline_summary.json"), summary)
    print(summary["overall"])


if __name__ == "__main__":
    main()
