from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .modeling import greedy_generate_batch
from .text import answer_is_correct


def evaluate_chains(model, tokenizer, table, chains, condition: str, seed: int,
                    checkpoint: str, max_new_tokens: int, output_file: Path,
                    alpha: float | None = None, overwrite: bool = False,
                    batch_size: int = 8):
    if output_file.exists() and not overwrite:
        with output_file.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    partial_file = output_file.with_suffix(output_file.suffix + ".partial")
    if overwrite and partial_file.exists():
        partial_file.unlink()
    rows = []
    if partial_file.exists():
        with partial_file.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        expected = [chain.chain_id for chain in chains[:len(rows)]]
        observed = [row["chain_id"] for row in rows]
        if observed != expected:
            raise RuntimeError(f"Partial evaluation does not match current chains: {partial_file}")
    start_at = len(rows)
    total_batches = (len(chains) + batch_size - 1) // batch_size
    for start in tqdm(range(start_at, len(chains), batch_size), desc=f"evaluate {condition}",
                      initial=start_at // batch_size, total=total_batches):
        chunk = chains[start:start + batch_size]
        flat_prompts = [p for chain in chunk for p in (chain.prompt_1, chain.prompt_2, chain.prompt_12)]
        flat_entities = [e for chain in chunk for e in (chain.e1, chain.e2, chain.e1)]
        predictions = greedy_generate_batch(model, tokenizer, table, flat_prompts, flat_entities,
                                            max_new_tokens, alpha)
        batch_rows = []
        for offset, chain in enumerate(chunk):
            p1, p2, p12 = predictions[3 * offset:3 * offset + 3]
            correct_1 = answer_is_correct(p1, chain.e2_aliases)
            correct_2 = answer_is_correct(p2, chain.e3_aliases)
            row = asdict(chain) | {
                "prediction_1": p1, "prediction_2": p2, "prediction_12": p12,
                "prediction_explicit": f"{p1} -> {p2}" if correct_1 else p1,
                "correct_1": correct_1,
                "correct_2": correct_2,
                "correct_12": answer_is_correct(p12, chain.e3_aliases),
                # Legacy field: joint success on independently queried gold
                # subjects. This is NOT a generated-bridge pipeline.
                "correct_explicit": correct_1 and correct_2,
                "explicit_metric_protocol": "independent_constituent_coverage",
                "condition": condition, "seed": seed, "checkpoint": checkpoint,
                "alpha": table.alpha if alpha is None else alpha,
            }
            batch_rows.append(row)
        with partial_file.open("a", encoding="utf-8") as handle:
            for row in batch_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        rows.extend(batch_rows)
    with output_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    partial_file.unlink(missing_ok=True)
    return rows


def accuracy_summary(rows, baseline_rows=None):
    if baseline_rows is not None:
        known = {r["chain_id"] for r in baseline_rows if r["correct_1"] and r["correct_2"]}
    else:
        known = {r["chain_id"] for r in rows if r["correct_1"] and r["correct_2"]}
    result = {}
    for label, subset in (("all", rows), ("knowledge_conditioned", [r for r in rows if r["chain_id"] in known])):
        denom = len(subset)
        joint = sum(r["correct_1"] and r["correct_2"] for r in subset)
        result[label] = {
            "n": denom,
            "A_1a": float(np.mean([r["correct_1"] for r in subset])) if denom else None,
            "A_1b": float(np.mean([r["correct_2"] for r in subset])) if denom else None,
            "A_2": float(np.mean([r["correct_12"] for r in subset])) if denom else None,
            "A_explicit": float(np.mean([r.get("correct_explicit", False) for r in subset])) if denom else None,
            "J_1": float(np.mean([r["correct_1"] and r["correct_2"] for r in subset])) if denom else None,
            "A_1_independent": (float(np.mean([r["correct_1"] for r in subset]))
                                  * float(np.mean([r["correct_2"] for r in subset]))) if denom else None,
            "C": (sum(r["correct_1"] and r["correct_2"] and r["correct_12"] for r in subset) / joint)
                 if joint else None,
            "C_numerator": sum(r["correct_1"] and r["correct_2"] and r["correct_12"] for r in subset),
            "C_denominator": joint,
            "A_2_given_adapted_one_hops": (
                sum(r["correct_1"] and r["correct_2"] and r["correct_12"] for r in subset) / joint
            ) if joint else None,
        }
    return result


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))
