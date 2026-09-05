#!/usr/bin/env python
from __future__ import annotations

import itertools
import json
import random
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import filter_token_mode, grouped_split, load_chains
from geometry_llm.modeling import (
    FrozenParameterGuard, ResidualTable, base_row_norms, discover_residual_keys,
    encode_example, load_model_and_tokenizer, pad_batch, residual_forward, save_residual,
)


def records(chains, shuffled: bool, seed: int):
    rows = []
    for c in chains:
        rows += [
            {"chain": c, "hop": 1, "prompt": c.prompt_1, "entity": c.e1, "answer": c.e2},
            {"chain": c, "hop": 2, "prompt": c.prompt_2, "entity": c.e2, "answer": c.e3},
        ]
    if shuffled:
        rng = random.Random(seed)
        # Each (relation composition, hop) is shuffled independently. A cyclic
        # shift is a derangement unless duplicate answer strings coincide.
        for hop in (1, 2):
            indexes = [i for i, r in enumerate(rows) if r["hop"] == hop]
            answers = [rows[i]["answer"] for i in indexes]
            if len(answers) > 1:
                shift = rng.randrange(1, len(answers))
                answers = answers[shift:] + answers[:shift]
            for i, answer in zip(indexes, answers):
                rows[i]["answer"] = answer
    return rows


def balanced_epoch(rows, seed):
    rng = random.Random(seed)
    result = []
    for hop in (1, 2):
        group = [r for r in rows if r["hop"] == hop]
        frequencies = Counter(r["answer"] for r in group)
        weights = [1 / frequencies[r["answer"]] for r in group]
        result.extend(rng.choices(group, weights=weights, k=len(group)))
    rng.shuffle(result)
    return result


@torch.no_grad()
def validation_loss(model, tokenizer, table, rows, batch_size):
    table.eval()
    losses = []
    device = model.get_input_embeddings().weight.device
    for start in range(0, len(rows), batch_size):
        encoded = [encode_example(tokenizer, r["prompt"], r["entity"], r["answer"], table)
                   for r in rows[start:start + batch_size]]
        batch = pad_batch(encoded, tokenizer.pad_token_id, device)
        losses.append(float(residual_forward(model, table, batch).loss))
    table.train()
    return float(np.mean(losses)) if losses else float("nan")


def run_one(cfg, model, tokenizer, chains, condition, seed, lr, anchor):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_chains, valid_chains = grouped_split(chains, cfg["data"]["validation_fraction"], seed)
    train_rows = records(train_chains, condition == "shuffled_delta", seed)
    valid_rows = records(valid_chains, condition == "shuffled_delta", seed)
    tag = f"{condition}/seed-{seed}/lr-{lr:g}_anchor-{anchor:g}"
    existing_checkpoint = output_path(cfg, "checkpoints", tag, "best.pt")
    existing_history = output_path(cfg, "training", tag, "history.json")
    if existing_checkpoint.exists() and existing_history.exists():
        history = json.loads(existing_history.read_text())
        best_row = min(history, key=lambda row: row["validation_loss"])
        return {"condition": condition, "seed": seed, "lr": lr, "anchor": anchor,
                "best_validation_loss": best_row["validation_loss"],
                "best_epochs": best_row["epoch"] + 1,
                "validation_checkpoint": str(existing_checkpoint)}
    keys, entity_key = discover_residual_keys(chains, tokenizer, cfg["data"]["token_mode"])
    hidden = model.get_input_embeddings().embedding_dim
    device = model.get_input_embeddings().weight.device
    table = ResidualTable(keys, hidden, cfg["training"]["alpha"], cfg["data"]["token_mode"]).to(device)
    optimizer = torch.optim.AdamW([table.delta], lr=lr, weight_decay=0)
    norms = base_row_norms(model, table)
    guard = FrozenParameterGuard(model)
    accumulation = cfg["training"]["gradient_accumulation_steps"]
    batch_size = cfg["training"]["batch_size"]
    checkpoint_steps = set(cfg["training"]["checkpoint_steps"])
    history, step, best, best_epochs, stale = [], 0, float("inf"), 1, 0
    if 0 in checkpoint_steps:
        save_residual(output_path(cfg, "checkpoints", tag, "step-0.pt"), table,
                      {"condition": condition, "seed": seed, "lr": lr, "anchor": anchor,
                       "step": 0, "entity_key": entity_key})
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cfg["training"]["epochs"]):
        epoch_rows = balanced_epoch(train_rows, seed + epoch)
        for start in range(0, len(epoch_rows), batch_size):
            chunk = epoch_rows[start:start + batch_size]
            encoded = [encode_example(tokenizer, r["prompt"], r["entity"], r["answer"], table) for r in chunk]
            batch = pad_batch(encoded, tokenizer.pad_token_id, device)
            prediction = residual_forward(model, table, batch).loss
            anchor_loss = table.anchor_loss(norms)
            loss = (prediction + anchor * anchor_loss) / accumulation
            loss.backward()
            assert table.delta.grad is not None and torch.isfinite(table.delta.grad).all()
            if ((start // batch_size) + 1) % accumulation == 0 or start + batch_size >= len(epoch_rows):
                torch.nn.utils.clip_grad_norm_([table.delta], cfg["training"]["gradient_clip"])
                optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
                guard.assert_unchanged()
                if step in checkpoint_steps:
                    save_residual(output_path(cfg, "checkpoints", tag, f"step-{step}.pt"), table,
                                  {"condition": condition, "seed": seed, "lr": lr, "anchor": anchor,
                                   "step": step, "entity_key": entity_key})
        val = validation_loss(model, tokenizer, table, valid_rows, batch_size)
        history.append({"epoch": epoch, "step": step, "train_prediction_loss": float(prediction.detach()),
                        "anchor_loss": float(anchor_loss.detach()), "validation_loss": val,
                        "residual_norm": float(table.delta.detach().float().norm())})
        if val < best:
            best, best_epochs, stale = val, epoch + 1, 0
            save_residual(output_path(cfg, "checkpoints", tag, "best.pt"), table,
                          {"condition": condition, "seed": seed, "lr": lr, "anchor": anchor,
                           "step": step, "best_epochs": best_epochs,
                           "validation_loss": val, "entity_key": entity_key})
        else:
            stale += 1
            if stale >= cfg["training"]["patience"]:
                break
    history_path = output_path(cfg, "training", tag, "history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))
    plt.figure(figsize=(7, 4))
    plt.plot([r["step"] for r in history], [r["validation_loss"] for r in history], marker="o", label="validation")
    plt.plot([r["step"] for r in history], [r["train_prediction_loss"] for r in history], marker="o", label="train")
    plt.xlabel("optimizer step"); plt.ylabel("one-hop loss"); plt.legend(); plt.tight_layout()
    plt.savefig(history_path.with_name("learning_curve.png"), dpi=180); plt.close()
    return {"condition": condition, "seed": seed, "lr": lr, "anchor": anchor,
            "best_validation_loss": best, "best_epochs": best_epochs,
            "validation_checkpoint": str(output_path(cfg, "checkpoints", tag, "best.pt"))}


def retrain_all(cfg, model, tokenizer, chains, selected):
    """Refit from zero on every chain using validation-selected settings."""
    condition, seed = selected["condition"], selected["seed"]
    lr, anchor, epochs = selected["lr"], selected["anchor"], selected["best_epochs"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tag = f"{condition}/seed-{seed}/lr-{lr:g}_anchor-{anchor:g}"
    path = output_path(cfg, "checkpoints", tag, "final.pt")
    if path.exists():
        return str(path)
    rows = records(chains, condition == "shuffled_delta", seed)
    keys, entity_key = discover_residual_keys(chains, tokenizer, cfg["data"]["token_mode"])
    device = model.get_input_embeddings().weight.device
    table = ResidualTable(keys, model.get_input_embeddings().embedding_dim,
                          cfg["training"]["alpha"], cfg["data"]["token_mode"]).to(device)
    optimizer = torch.optim.AdamW([table.delta], lr=lr, weight_decay=0)
    norms, guard = base_row_norms(model, table), FrozenParameterGuard(model)
    batch_size = cfg["training"]["batch_size"]
    accumulation = cfg["training"]["gradient_accumulation_steps"]
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        epoch_rows = balanced_epoch(rows, seed + epoch)
        for start in range(0, len(epoch_rows), batch_size):
            chunk = epoch_rows[start:start + batch_size]
            encoded = [encode_example(tokenizer, r["prompt"], r["entity"], r["answer"], table) for r in chunk]
            batch = pad_batch(encoded, tokenizer.pad_token_id, device)
            prediction = residual_forward(model, table, batch).loss
            loss = (prediction + anchor * table.anchor_loss(norms)) / accumulation
            loss.backward()
            assert table.delta.grad is not None and torch.isfinite(table.delta.grad).all()
            if ((start // batch_size) + 1) % accumulation == 0 or start + batch_size >= len(epoch_rows):
                torch.nn.utils.clip_grad_norm_([table.delta], cfg["training"]["gradient_clip"])
                optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
                guard.assert_unchanged()
    save_residual(path, table, {
        "condition": condition, "seed": seed, "lr": lr, "anchor": anchor,
        "step": step, "epochs": epochs, "entity_key": entity_key,
        "selection_validation_loss": selected["best_validation_loss"],
        "selection_checkpoint": selected["validation_checkpoint"],
        "trained_on_all_chains": True,
    })
    return str(path)


def main():
    parser = common_parser("Train compact embedding residuals")
    parser.add_argument("--condition", choices=["correct_delta", "shuffled_delta"], default="correct_delta")
    parser.add_argument("--grid-shards", type=int, default=1)
    parser.add_argument("--grid-shard-index", type=int, default=0)
    parser.add_argument("--grid-only", action="store_true",
                        help="Run this grid shard without selecting/final retraining")
    parser.add_argument("--select-existing", action="store_true",
                        help="Require all grid cells from cache, then select and retrain")
    parser.add_argument("--control-from",
                        help="For shuffled_delta, reuse correct_delta hyperparameters and epochs")
    parser.add_argument("--fit-all-fixed", action="store_true",
                        help="Skip validation selection and fit all chains once with the first configured settings")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    model, tokenizer = load_model_and_tokenizer(cfg)
    chains, failures = filter_token_mode(load_chains(cfg), tokenizer, cfg["data"]["token_mode"])
    if not chains:
        raise RuntimeError(f"No usable chains; first failures: {failures[:3]}")
    if args.fit_all_fixed:
        if args.control_from or args.grid_only or args.select_existing:
            parser.error("--fit-all-fixed cannot be combined with selection or control options")
        selected = {
            "condition": args.condition,
            "seed": cfg["training"]["seeds"][0],
            "lr": cfg["training"]["learning_rates"][0],
            "anchor": cfg["training"]["anchor_coefficients"][0],
            "best_epochs": cfg["training"]["epochs"],
            "best_validation_loss": None,
            "validation_checkpoint": "not_used_fixed_all",
        }
        selected["checkpoint"] = retrain_all(cfg, model, tokenizer, chains, selected)
        selection = {
            "condition": args.condition,
            "selection": "fixed settings transferred from another dataset",
            "selection_uses_two_hop": False,
            "learning_rate": selected["lr"], "anchor": selected["anchor"],
            "runs": [selected],
        }
        path = output_path(cfg, "training", f"{args.condition}_selected.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(selection, indent=2))
        print(f"Fitted {args.condition} once on all {len(chains)} chains with fixed settings")
        return
    if args.control_from:
        if args.condition != "shuffled_delta":
            parser.error("--control-from is only valid for shuffled_delta")
        reference = json.loads(open(args.control_from, encoding="utf-8").read())
        final_runs = []
        for source in reference["runs"]:
            run = dict(source); run["condition"] = "shuffled_delta"
            run["checkpoint"] = retrain_all(cfg, model, tokenizer, chains, run)
            final_runs.append(run)
        selection = {"condition": "shuffled_delta", "control_hyperparameters_from": args.control_from,
                     "learning_rate": reference["learning_rate"], "anchor": reference["anchor"],
                     "selection_uses_two_hop": False, "runs": final_runs}
        path = output_path(cfg, "training", "shuffled_delta_selected.json")
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(selection, indent=2))
        print(f"Retrained shuffled control for {len(final_runs)} seeds with fixed correct-delta settings")
        return
    results = []
    complete_grid = list(itertools.product(cfg["training"]["seeds"], cfg["training"]["learning_rates"],
                                           cfg["training"]["anchor_coefficients"]))
    if not 0 <= args.grid_shard_index < args.grid_shards:
        parser.error("--grid-shard-index must be in [0, --grid-shards)")
    grid = complete_grid if args.select_existing else complete_grid[args.grid_shard_index::args.grid_shards]
    for seed, lr, anchor in grid:
        if args.select_existing:
            tag = f"{args.condition}/seed-{seed}/lr-{lr:g}_anchor-{anchor:g}"
            if not output_path(cfg, "training", tag, "history.json").exists():
                raise RuntimeError(f"Missing cached grid cell: {tag}")
        results.append(run_one(cfg, model, tokenizer, chains, args.condition, seed, lr, anchor))
    suffix = f"_shard-{args.grid_shard_index}-of-{args.grid_shards}" if args.grid_shards > 1 else ""
    path = output_path(cfg, "training", f"{args.condition}_grid{suffix}.json")
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(results, indent=2))
    if args.grid_only:
        print(f"Finished grid shard {args.grid_shard_index}/{args.grid_shards}: {len(results)} runs")
        return
    grouped = {}
    for lr in cfg["training"]["learning_rates"]:
        for anchor in cfg["training"]["anchor_coefficients"]:
            group = [r for r in results if r["lr"] == lr and r["anchor"] == anchor]
            grouped[(lr, anchor)] = float(np.mean([r["best_validation_loss"] for r in group]))
    selected_lr, selected_anchor = min(grouped, key=grouped.get)
    selected_runs = [r for r in results if r["lr"] == selected_lr and r["anchor"] == selected_anchor]
    final_runs = []
    for run in selected_runs:
        final = dict(run)
        final["checkpoint"] = (retrain_all(cfg, model, tokenizer, chains, run)
                               if cfg["data"].get("final_retrain_on_all_chains", True)
                               else run["validation_checkpoint"])
        final_runs.append(final)
    selection = {"condition": args.condition, "learning_rate": selected_lr,
                 "anchor": selected_anchor, "mean_one_hop_validation_loss": grouped[(selected_lr, selected_anchor)],
                 "selection_uses_two_hop": False, "runs": final_runs}
    path.with_name(f"{args.condition}_selected.json").write_text(json.dumps(selection, indent=2))
    print(f"Finished {len(results)} validation runs; selected lr={selected_lr:g}, anchor={selected_anchor:g}; "
          f"retrained {len(final_runs)} seeds on all chains")


if __name__ == "__main__":
    main()
