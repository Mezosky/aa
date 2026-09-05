#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import filter_token_mode, load_chains
from geometry_llm.metrics import centered_cka
from geometry_llm.modeling import encode_example, load_model_and_tokenizer, load_residual, pad_batch, residual_forward


def one_hot(values):
    kinds = {v: i for i, v in enumerate(sorted(set(values)))}
    result = np.zeros((len(values), len(kinds)))
    for row, value in enumerate(values): result[row, kinds[value]] = 1
    return result


def main():
    parser = common_parser("Measure layerwise residual propagation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    raw_chains = load_chains(cfg)
    out = output_path(cfg, "layers")
    if (out / "metrics.json").exists() and (out / "representations.npz").exists() and not args.overwrite:
        print(f"Using cached layerwise results: {out}")
        return
    model, tokenizer = load_model_and_tokenizer(cfg)
    chains, _ = filter_token_mode(raw_chains, tokenizer, cfg["data"]["token_mode"])
    chains = chains[:args.max_examples]
    device = model.get_input_embeddings().weight.device
    table, _ = load_residual(Path(args.checkpoint), model.get_input_embeddings().embedding_dim, device)
    requested = cfg["analysis"]["layers"]
    store = {site: {layer: {"h0": [], "hd": []} for layer in requested} for site in ("entity", "final")}
    with torch.no_grad():
        for chain in chains:
            item = encode_example(tokenizer, chain.prompt_12, chain.e1, None, table)
            batch = pad_batch([item], tokenizer.pad_token_id, device)
            original_alpha = table.alpha
            table.alpha = 0; h0 = residual_forward(model, table, batch, True).hidden_states
            table.alpha = original_alpha; hd = residual_forward(model, table, batch, True).hidden_states
            entity_positions = [i for i, row in enumerate(item.delta_indices) if row >= 0]
            for layer in requested:
                if layer >= len(h0): continue
                for site, positions in (("entity", entity_positions), ("final", [item.prompt_length - 1])):
                    store[site][layer]["h0"].append(h0[layer][0, positions].float().mean(0).cpu().numpy())
                    store[site][layer]["hd"].append(hd[layer][0, positions].float().mean(0).cpu().numpy())
    bridge, answer = one_hot([c.e2 for c in chains]), one_hot([c.e3 for c in chains])
    rows = []
    for site in store:
        for layer, values in store[site].items():
            if not values["h0"]: continue
            h0, hd = np.asarray(values["h0"]), np.asarray(values["hd"]); change = hd - h0
            rows.append({
                "site": site, "layer": layer,
                "relative_change_magnitude": float(np.mean(np.linalg.norm(change, axis=1) / np.linalg.norm(h0, axis=1).clip(1e-12))),
                "preservation_cka": centered_cka(h0, hd),
                "bridge_alignment_cka": centered_cka(change, bridge),
                "answer_alignment_cka": centered_cka(change, answer),
            })
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))
    np.savez_compressed(out / "representations.npz", **{
        f"{site}_{layer}_{kind}": np.asarray(vals[kind]) for site, layers in store.items()
        for layer, vals in layers.items() for kind in ("h0", "hd") if vals[kind]
    })
    for metric in ("relative_change_magnitude", "preservation_cka", "bridge_alignment_cka", "answer_alignment_cka"):
        plt.figure(figsize=(7, 4))
        for site in ("entity", "final"):
            selected = [r for r in rows if r["site"] == site]
            plt.plot([r["layer"] for r in selected], [r[metric] for r in selected], marker="o", label=site)
        plt.xlabel("layer"); plt.ylabel(metric); plt.legend(); plt.tight_layout(); plt.savefig(out / f"{metric}.png", dpi=180); plt.close()
    print(f"Layerwise results: {out}. Alignment/decodability does not locate reasoning.")


if __name__ == "__main__":
    main()
