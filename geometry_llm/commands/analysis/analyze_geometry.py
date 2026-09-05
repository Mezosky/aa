#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import filter_token_mode, load_chains
from geometry_llm.evaluation import write_json
from geometry_llm.metrics import cosine_rows, effective_rank, related_auc
from geometry_llm.modeling import load_model_and_tokenizer, load_residual


def entity_matrices(chains, tokenizer, model, table, metadata):
    names = sorted({x for c in chains for x in (c.e1, c.e2, c.e3)})
    entity_key = metadata.get("entity_key", {})
    embedding = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    base, delta = [], []
    for entity in names:
        ids = tokenizer(entity, add_special_tokens=False)["input_ids"]
        base.append(embedding[ids].mean(0))
        key = entity_key.get(entity)
        if key in table.key_to_index:
            delta.append(table.delta[table.key_to_index[key]].detach().float().cpu().numpy())
        else:
            delta.append(np.zeros(embedding.shape[1]))
    return names, np.asarray(base), np.asarray(delta)


def consistency(vectors):
    vectors = np.asarray(vectors)
    centroid = vectors.mean(0, keepdims=True)
    return float(cosine_rows(vectors, np.repeat(centroid, len(vectors), axis=0)).mean())


def matched_unrelated_pairs(chains, index):
    """Match endpoint role/type and approximate entity frequency."""
    role_frequency = {
        role: {name: sum(getattr(c, role) == name for c in chains)
               for name in {getattr(c, role) for c in chains}}
        for role in ("e2", "e3")
    }
    adjacency = set()
    for c in chains:
        adjacency.add((c.e1, c.e2)); adjacency.add((c.e2, c.e3)); adjacency.add((c.e1, c.e3))

    def choose(source, true_target, role, target_type, salt):
        pool = sorted({getattr(c, role) for c in chains
                       if getattr(c, f"{role}_type") == target_type
                       and getattr(c, role) != true_target
                       and (source, getattr(c, role)) not in adjacency})
        if not pool:
            pool = sorted(set(role_frequency[role]) - {true_target})
        wanted = role_frequency[role][true_target]
        best_gap = min(abs(role_frequency[role][candidate] - wanted) for candidate in pool)
        tied = [candidate for candidate in pool if abs(role_frequency[role][candidate] - wanted) == best_gap]
        offset = int(hashlib.sha256(salt.encode()).hexdigest()[:8], 16) % len(tied)
        return tied[offset]

    direct, indirect = [], []
    for c in chains:
        u2 = choose(c.e1, c.e2, "e2", c.e2_type, c.chain_id + ":r1")
        u3 = choose(c.e2, c.e3, "e3", c.e3_type, c.chain_id + ":r2")
        ui = choose(c.e1, c.e3, "e3", c.e3_type, c.chain_id + ":indirect")
        direct.extend(((index[c.e1], index[u2]), (index[c.e2], index[u3])))
        indirect.append((index[c.e1], index[ui]))
    return direct, indirect


def main():
    parser = common_parser("Analyze residual and adapted input geometry")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    model, tokenizer = load_model_and_tokenizer(cfg)
    chains, _ = filter_token_mode(load_chains(cfg), tokenizer, cfg["data"]["token_mode"])
    table, metadata = load_residual(Path(args.checkpoint), model.get_input_embeddings().embedding_dim,
                                    model.get_input_embeddings().weight.device)
    names, base, delta = entity_matrices(chains, tokenizer, model, table, metadata)
    index = {name: i for i, name in enumerate(names)}
    pairs = {
        "direct_r1": [(index[c.e1], index[c.e2]) for c in chains],
        "direct_r2": [(index[c.e2], index[c.e3]) for c in chains],
        "indirect": [(index[c.e1], index[c.e3]) for c in chains],
    }
    pairs["unrelated_direct"], pairs["unrelated_indirect"] = matched_unrelated_pairs(chains, index)
    report, distribution_rows = {}, []
    for space_name, matrix in (("delta", delta), ("base_plus_delta", base + table.alpha * delta)):
        unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-12)
        measurements = {}
        for pair_name, pair in pairs.items():
            left, right = zip(*pair)
            cos = cosine_rows(matrix[list(left)], matrix[list(right)])
            dist = np.linalg.norm(unit[list(left)] - unit[list(right)], axis=1)
            measurements[pair_name] = {"cosine_mean": float(cos.mean()), "cosine_std": float(cos.std()),
                                       "norm_controlled_distance_mean": float(dist.mean()),
                                       "norm_controlled_distance_std": float(dist.std())}
            distribution_rows.extend({"space": space_name, "pair_type": pair_name,
                                      "cosine": float(c), "norm_controlled_distance": float(d)}
                                     for c, d in zip(cos, dist))
        related = pairs["direct_r1"] + pairs["direct_r2"]
        rl, rr = zip(*related); ul, ur = zip(*pairs["unrelated_direct"])
        related_cos = cosine_rows(matrix[list(rl)], matrix[list(rr)])
        unrelated_cos = cosine_rows(matrix[list(ul)], matrix[list(ur)])
        # Any directly connected entity counts as a valid retrieval target.
        adjacency = {i: set() for i in range(len(names))}
        for a, b in related: adjacency[a].add(b); adjacency[b].add(a)
        sim = unit @ unit.T; np.fill_diagonal(sim, -np.inf)
        retrieval = np.mean([int(np.argmax(sim[i]) in adjacency[i]) for i in adjacency if adjacency[i]])
        r1 = [matrix[index[c.e2]] - matrix[index[c.e1]] for c in chains]
        r2 = [matrix[index[c.e3]] - matrix[index[c.e2]] for c in chains]
        singular = np.linalg.svd(matrix - matrix.mean(0), compute_uv=False)
        report[space_name] = {
            "distributions": measurements, "related_vs_unrelated_cosine_auc": related_auc(related_cos, unrelated_cos),
            "effective_rank": effective_rank(matrix), "singular_values": singular.tolist(),
            "nearest_neighbor_related_accuracy": float(retrieval),
            "relation_vector_consistency": {"r1": consistency(r1), "r2": consistency(r2)},
        }
    out = output_path(cfg, "geometry")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", report)
    with (out / "pair_distributions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distribution_rows[0])); writer.writeheader(); writer.writerows(distribution_rows)
    for label, matrix in (("delta", delta), ("base_plus_delta", base + table.alpha * delta)):
        coords = PCA(n_components=2).fit_transform(matrix)
        plt.figure(figsize=(7, 6)); plt.scatter(coords[:, 0], coords[:, 1], s=12, alpha=.7)
        plt.title(f"Supporting PCA: {label}"); plt.tight_layout(); plt.savefig(out / f"pca_{label}.png", dpi=180); plt.close()
        singular = report[label]["singular_values"]
        plt.figure(figsize=(7, 4)); plt.plot(singular); plt.yscale("log"); plt.title(f"Singular spectrum: {label}")
        plt.tight_layout(); plt.savefig(out / f"spectrum_{label}.png", dpi=180); plt.close()
    try:
        import umap
        for label, matrix in (("delta", delta), ("base_plus_delta", base + table.alpha * delta)):
            coords = umap.UMAP(random_state=cfg["analysis"]["seed"]).fit_transform(matrix)
            plt.figure(figsize=(7, 6)); plt.scatter(coords[:, 0], coords[:, 1], s=12, alpha=.7)
            plt.title(f"Supporting UMAP: {label}"); plt.tight_layout(); plt.savefig(out / f"umap_{label}.png", dpi=180); plt.close()
    except (ImportError, ValueError):
        pass
    print(f"Geometry results: {out}")


if __name__ == "__main__":
    main()
