#!/usr/bin/env python
"""Conditional composition and grouped uncertainty for saved predictions."""
from __future__ import annotations

import json

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.evaluation import write_json
from geometry_llm.metrics import grouped_bootstrap_metrics, grouped_paired_difference


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def add_group(rows):
    for row in rows:
        bridge = row.get("e2_id") or row["e2"]
        answer = row.get("e3_id") or row["e3"]
        row["bridge_answer_group"] = f"{bridge}::{answer}"
    return rows


def main():
    parser = common_parser("Analyze conditional composition with grouped bootstrap intervals")
    parser.add_argument("--conditions", nargs="+", default=["correct_delta", "shuffled_delta", "random_delta"])
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    prediction_dir = output_path(cfg, "predictions")
    baseline = add_group(read_jsonl(prediction_dir / "original.jsonl"))
    report = {
        "group_definition": "joint bridge and answer identity",
        "baseline": grouped_bootstrap_metrics(
            baseline, samples=cfg["analysis"]["bootstrap_samples"], seed=cfg["analysis"]["seed"]),
        "conditions": {},
    }
    for index, condition in enumerate(args.conditions):
        path = prediction_dir / f"{condition}_seed-{args.seed}.jsonl"
        if not path.exists():
            report["conditions"][condition] = {"status": "missing", "path": str(path)}
            continue
        rows = add_group(read_jsonl(path))
        report["conditions"][condition] = {
            "grouped_metrics": grouped_bootstrap_metrics(
                rows, samples=cfg["analysis"]["bootstrap_samples"],
                seed=cfg["analysis"]["seed"] + index + 1),
            "paired_change_from_baseline": grouped_paired_difference(
                baseline, rows, samples=cfg["analysis"]["bootstrap_samples"],
                seed=cfg["analysis"]["seed"] + index + 101),
        }
    destination = output_path(cfg, "analysis", f"behavior_seed-{args.seed}.json")
    write_json(destination, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
