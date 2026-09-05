#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import filter_token_mode, load_chains
from geometry_llm.evaluation import accuracy_summary, evaluate_chains
from geometry_llm.modeling import load_model_and_tokenizer, load_residual


def relation_directions(table, chains, entity_key, ranks):
    vectors = []
    for c in chains:
        keys = [entity_key.get(x) for x in (c.e1, c.e2, c.e3)]
        if all(k in table.key_to_index for k in keys):
            rows = [table.delta[table.key_to_index[k]].detach().float().cpu().numpy() for k in keys]
            vectors.extend((rows[1] - rows[0], rows[2] - rows[1]))
    if not vectors:
        raise ValueError("Checkpoint has no entity-to-row metadata for direction removal")
    _, _, vh = np.linalg.svd(np.asarray(vectors), full_matrices=False)
    return {rank: torch.as_tensor(vh[:rank], device=table.delta.device, dtype=table.delta.dtype)
            for rank in ranks if rank <= len(vh)}


def main():
    parser = common_parser("Run post-training causal interventions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    model, tokenizer = load_model_and_tokenizer(cfg)
    chains, _ = filter_token_mode(load_chains(cfg), tokenizer, cfg["data"]["token_mode"])
    device = model.get_input_embeddings().weight.device
    table, metadata = load_residual(Path(args.checkpoint), model.get_input_embeddings().embedding_dim, device)
    original = table.delta.detach().clone()
    baseline_path = output_path(cfg, "predictions", "original.jsonl")
    baseline = [json.loads(line) for line in baseline_path.open()] if baseline_path.exists() else None
    results = []

    def run(name, alpha=None):
        rows = evaluate_chains(model, tokenizer, table, chains, name, args.seed, args.checkpoint,
                               cfg["model"]["max_answer_tokens"],
                               output_path(cfg, "interventions", f"{name}.jsonl"), alpha, args.overwrite,
                               cfg["model"].get("evaluation_batch_size", 8))
        summary = accuracy_summary(rows, baseline)
        results.append({"intervention": name, "alpha": alpha, **summary["all"]})

    for alpha in cfg["analysis"]["alphas"]:
        run(f"scale_{alpha:g}", alpha)
    run("delta_removal", 0.0)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    permutation = torch.randperm(len(original), generator=generator, device=device)
    table.delta.data.copy_(original[permutation]); run("delta_permutation"); table.delta.data.copy_(original)
    for rank, directions in relation_directions(table, chains, metadata.get("entity_key", {}),
                                                 cfg["analysis"]["direction_ranks"]).items():
        projected = original - (original @ directions.T) @ directions
        table.delta.data.copy_(projected); run(f"direction_removal_rank_{rank}")
    table.delta.data.copy_(original)
    out = output_path(cfg, "interventions")
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    scales = [r for r in results if r["intervention"].startswith("scale_")]
    plt.figure(figsize=(7, 4))
    plt.plot([r["alpha"] for r in scales], [r["A_1a"] for r in scales], marker="o", label="hop 1")
    plt.plot([r["alpha"] for r in scales], [r["A_1b"] for r in scales], marker="o", label="hop 2")
    plt.plot([r["alpha"] for r in scales], [r["A_2"] for r in scales], marker="o", label="two hop")
    plt.xlabel("residual scale alpha"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "delta_scaling.png", dpi=180); plt.close()
    print(f"Intervention results: {out}")


if __name__ == "__main__":
    main()
