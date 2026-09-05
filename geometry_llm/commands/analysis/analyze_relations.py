#!/usr/bin/env python
"""Report factual-access and composition metrics by relation pair."""
from __future__ import annotations

import json
from collections import defaultdict

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.evaluation import accuracy_summary, write_json


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def relation_summary(rows, key, min_group_size):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return {
        relation: accuracy_summary(items)["all"]
        for relation, items in sorted(grouped.items())
        if len(items) >= min_group_size
    }


def main():
    parser = common_parser("Summarize results by relation pair")
    parser.add_argument("--condition", default="correct_delta")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-group-size", type=int, default=10)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    prediction_dir = output_path(cfg, "predictions")
    baseline = read_jsonl(prediction_dir / "original.jsonl")
    adapted = read_jsonl(prediction_dir / f"{args.condition}_seed-{args.seed}.jsonl")
    report = {
        "condition": args.condition,
        "seed": args.seed,
        "min_group_size": args.min_group_size,
        "baseline": {
            "relation_pair": relation_summary(baseline, "fact_comp_type", args.min_group_size),
            "first_relation": relation_summary(baseline, "r1_type", args.min_group_size),
            "second_relation": relation_summary(baseline, "r2_type", args.min_group_size),
        },
        "adapted": {
            "relation_pair": relation_summary(adapted, "fact_comp_type", args.min_group_size),
            "first_relation": relation_summary(adapted, "r1_type", args.min_group_size),
            "second_relation": relation_summary(adapted, "r2_type", args.min_group_size),
        },
    }
    destination = output_path(cfg, "analysis", f"relations_{args.condition}_seed-{args.seed}.json")
    write_json(destination, report)
    counts = {name: len(groups) for name, groups in report["adapted"].items()}
    print(json.dumps({"output": str(destination), "retained_groups": counts}, indent=2))


if __name__ == "__main__":
    main()
