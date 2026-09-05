#!/usr/bin/env python
"""Prepare a CLUTRR short-chain to long-chain generalization split.

This prepares the benchmark records only. CLUTRR requires a shared or
role-conditioned residual, rather than SOCRATES's entity-indexed table, because
the synthetic people change across stories.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from geometry_llm.config import load_config

from geometry_llm.data import load_clutrr_examples


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare CLUTRR length-generalization records")
    parser.add_argument("--config", default="config_clutrr.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]
    train = load_clutrr_examples(data["dataset"], data["train_split"])
    validation = load_clutrr_examples(data["dataset"], data["validation_split"])
    test = load_clutrr_examples(data["dataset"], data["test_split"])
    short = [row for row in train if row.approximate_hops <= data["train_max_hops"]]
    long_validation = [row for row in validation if row.approximate_hops >= data["test_min_hops"]]
    long_test = [row for row in test if row.approximate_hops >= data["test_min_hops"]]
    out = Path(cfg["output_dir"])
    write_jsonl(out / "short_train.jsonl", short)
    write_jsonl(out / "long_validation.jsonl", long_validation)
    write_jsonl(out / "long_test.jsonl", long_test)
    summary = {
        "dataset": data["dataset"],
        "train_max_hops": data["train_max_hops"],
        "test_min_hops": data["test_min_hops"],
        "short_train": len(short),
        "long_validation": len(long_validation),
        "long_test": len(long_test),
        "status": "prepared; requires a shared or role-conditioned residual runner",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
