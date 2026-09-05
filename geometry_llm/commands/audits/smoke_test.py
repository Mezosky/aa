#!/usr/bin/env python
"""A ~20-chain real-model check to run before the grid search."""
from __future__ import annotations

import torch

from geometry_llm.config import common_parser, load_config
from geometry_llm.data import filter_token_mode, load_chains
from geometry_llm.modeling import (
    FrozenParameterGuard, ResidualTable, discover_residual_keys, encode_example,
    load_model_and_tokenizer, pad_batch, residual_forward,
)


def main():
    args = common_parser("Run safety checks on about 20 chains").parse_args()
    cfg = load_config(args.config, args.set)
    cfg["data"]["max_examples"] = min(cfg["data"].get("max_examples") or 20, 20)
    model, tokenizer = load_model_and_tokenizer(cfg)
    chains, failures = filter_token_mode(load_chains(cfg), tokenizer, cfg["data"]["token_mode"])
    if not chains: raise RuntimeError(f"No smoke-test chains: {failures[:3]}")
    keys, _ = discover_residual_keys(chains, tokenizer, cfg["data"]["token_mode"])
    device = model.get_input_embeddings().weight.device
    table = ResidualTable(keys, model.get_input_embeddings().embedding_dim, 1,
                          cfg["data"]["token_mode"]).to(device)
    examples = []
    training_prompts = set()
    for c in chains:
        examples.extend((encode_example(tokenizer, c.prompt_1, c.e1, c.e2, table),
                         encode_example(tokenizer, c.prompt_2, c.e2, c.e3, table)))
        training_prompts.update((c.prompt_1, c.prompt_2))
        assert c.prompt_12 not in training_prompts, "A two-hop prompt entered training"
    for item in examples:
        assert all(label == -100 for label in item.labels[:item.prompt_length])
        assert all(index == -1 for index in item.delta_indices[item.prompt_length:])
        assert any(index >= 0 for index in item.delta_indices[:item.prompt_length])
    guard = FrozenParameterGuard(model)
    optimizer = torch.optim.AdamW([table.delta], lr=1e-3, weight_decay=0)
    batch = pad_batch(examples[:2], tokenizer.pad_token_id, device)
    before_in = model.get_input_embeddings().weight.detach().flatten()[:32].clone()
    before_out = model.get_output_embeddings().weight.detach().flatten()[:32].clone()
    loss = residual_forward(model, table, batch).loss
    loss.backward()
    assert table.delta.grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())
    optimizer.step(); guard.assert_unchanged()
    assert torch.equal(before_in, model.get_input_embeddings().weight.detach().flatten()[:32])
    assert torch.equal(before_out, model.get_output_embeddings().weight.detach().flatten()[:32])
    assert model.config.use_cache is False
    print(f"PASS: {len(chains)} chains; only residual received gradients; masks, prompts, and frozen weights verified")


if __name__ == "__main__":
    main()
