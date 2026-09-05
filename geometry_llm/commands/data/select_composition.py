#!/usr/bin/env python
"""Select a diverse composition using one-hop baseline knowledge only."""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+", help="JSONL files or glob patterns from --all-compositions")
    parser.add_argument("--min-examples", type=int, default=40)
    parser.add_argument("--min-known", type=int, default=5)
    parser.add_argument("--max-concentration", type=float, default=0.25)
    parser.add_argument("--output", default="outputs/composition_selection.json")
    args = parser.parse_args()
    paths = []
    for pattern in args.predictions:
        matches = glob.glob(pattern)
        paths.extend(matches or [pattern])
    groups = defaultdict(list)
    for path in sorted(set(paths)):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line); groups[row["fact_comp_type"]].append(row)
    candidates = []
    for name, rows in groups.items():
        n = len(rows); e2 = Counter(r["e2"] for r in rows); e3 = Counter(r["e3"] for r in rows)
        known = sum(r["correct_1"] and r["correct_2"] for r in rows)
        concentration = max(max(e2.values()), max(e3.values())) / n
        a1, a2 = (sum(r[key] for r in rows) / n for key in ("correct_1", "correct_2"))
        diversity = math.sqrt(len(e2) * len(e3))
        eligible = n >= args.min_examples and known >= args.min_known and concentration <= args.max_concentration
        # No composed-prompt result appears in this selection score.
        balanced_knowledge = (2 * a1 * a2 / (a1 + a2)) if a1 + a2 else 0.0
        # Known two-edge paths dominate; diversity is a logarithmic tie-breaker
        # so a huge but mostly unknown composition cannot win by size alone.
        score = 2 * known + 10 * balanced_knowledge + math.log1p(diversity)
        candidates.append({"fact_comp_type": name, "n": n, "unique_e2": len(e2), "unique_e3": len(e3),
                           "max_target_concentration": concentration, "baseline_A_1a": a1,
                           "baseline_A_1b": a2, "baseline_known_chains": known,
                           "eligible": eligible, "selection_score": score})
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise RuntimeError("No composition meets diversity and baseline-knowledge thresholds")
    selected = max(eligible, key=lambda row: row["selection_score"])
    report = {"selected_fact_comp_type": selected["fact_comp_type"],
              "selection_uses_two_hop_accuracy": False, "selected": selected,
              "candidates": sorted(candidates, key=lambda row: row["selection_score"], reverse=True)}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["selected"], indent=2))
    print(f"Selection written to {path}")


if __name__ == "__main__":
    main()
