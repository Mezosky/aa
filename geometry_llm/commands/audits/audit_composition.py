#!/usr/bin/env python
"""Counts, cluster intervals, and paired common-population composition metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    return [json.loads(line) for line in Path(path).open()]


def group_key(row):
    return (row.get("e2_id") or row["e2"], row.get("e3_id") or row["e3"])


def metric(numerator, denominator, draws_num, draws_den):
    valid = draws_den > 0
    values = draws_num[valid] / draws_den[valid]
    return {"numerator": int(numerator), "denominator": int(denominator),
            "estimate": float(numerator / denominator) if denominator else None,
            "ci95": np.quantile(values, [.025, .975]).tolist() if len(values) else [None, None],
            "valid_draws": int(valid.sum())}


def paired_audit(base, adapted, samples=5000, seed=123):
    old = {r["chain_id"]: r for r in base}
    new = {r["chain_id"]: r for r in adapted}
    if len(old) != len(base) or len(new) != len(adapted) or old.keys() != new.keys():
        raise ValueError("Expected unique, identical chain IDs in paired predictions")
    groups = sorted({group_key(r) for r in base})
    lookup = {key: i for i, key in enumerate(groups)}
    # n, base direct, adapted direct, old joint, old C numerator,
    # new joint, new C numerator, common n, common old correct, common new correct
    counts = np.zeros((len(groups), 10), dtype=int)
    for chain_id, b in old.items():
        a = new[chain_id]
        if group_key(b) != group_key(a):
            raise ValueError("Pair group identity changed")
        bj, aj = bool(b["correct_1"] and b["correct_2"]), bool(a["correct_1"] and a["correct_2"])
        common = bj and aj
        counts[lookup[group_key(b)]] += [1, b["correct_12"], a["correct_12"], bj,
            bj and b["correct_12"], aj, aj and a["correct_12"], common,
            common and b["correct_12"], common and a["correct_12"]]
    point = counts.sum(0)
    weights = np.random.default_rng(seed).multinomial(len(groups), np.ones(len(groups))/len(groups), size=samples)
    draws = weights @ counts
    pairs = {"base_direct": (1,0), "adapted_direct": (2,0), "base_C": (4,3),
             "adapted_C": (6,5), "common_base": (8,7), "common_adapted": (9,7),
             "common_change": (9,7)}
    out = {"n": len(base), "groups": len(groups), "group_definition": "joint bridge and answer identity"}
    for name, (num, den) in pairs.items():
        pn, dn = point[num], draws[:,num]
        if name == "common_change":
            pn, dn = pn-point[8], dn-draws[:,8]
        out[name] = metric(pn, point[den], dn, draws[:,den])
    valid = (draws[:,3] > 0) & (draws[:,5] > 0)
    diffs = draws[valid,6]/draws[valid,5] - draws[valid,4]/draws[valid,3]
    out["condition_specific_C_change"] = {
        "estimate": float(point[6]/point[5]-point[4]/point[3]) if point[3] and point[5] else None,
        "ci95": np.quantile(diffs,[.025,.975]).tolist() if len(diffs) else [None,None],
        "valid_draws": len(diffs), "interpretation": "different conditioning populations; not a causal effect"}
    out["adapted_direct_change"] = metric(point[2]-point[1], point[0], draws[:,2]-draws[:,1], draws[:,0])
    return out


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--samples", type=int, default=5000)
    args = p.parse_args()
    root = Path(args.root)
    base = read_rows(root / "predictions/original.jsonl")
    report = {"root": str(root), "seeds": {}, "interventions": {}}
    for path in sorted((root / "predictions").glob("correct_delta_seed-*.jsonl")):
        seed = int(path.stem.rsplit("-",1)[1])
        report["seeds"][str(seed)] = paired_audit(base, read_rows(path), args.samples)
    for path in sorted((root / "interventions").glob("*.jsonl")):
        if path.stem.startswith(("scale_", "direction_removal_", "delta_")):
            report["interventions"][path.stem] = paired_audit(base, read_rows(path), args.samples)
    destination = root / "analysis/conditional_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    for seed, row in report["seeds"].items():
        print(root.name, seed, json.dumps({k: row[k] for k in ["base_C","adapted_C","common_base","common_adapted","common_change"]}))


if __name__ == "__main__":
    main()
