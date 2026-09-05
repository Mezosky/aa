#!/usr/bin/env python
"""Connected-graph calibration for locating residual geometric memory.

Every selected node is presented as an input through bidirectional local-edge
supervision.  No two-hop target or prompt is used in training.  The script then
tests input geometry and the layerwise geometry of the propagated change.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from geometry_llm.config import load_config
from geometry_llm.data import row_to_chain
from geometry_llm.modeling import (
    FrozenParameterGuard, ResidualTable, base_row_norms, encode_example,
    load_model_and_tokenizer, pad_batch, residual_forward, save_residual,
)
from geometry_llm.text import normalize_answer


BG, INK, MUTED, GRID = "#F7F5F2", "#18323F", "#6B7C83", "#D9DFDF"
TEAL, GOLD, CORAL, GREY = "#087E8B", "#D79A22", "#D85B5B", "#78888E"


@dataclass(frozen=True)
class LocalExample:
    source: str
    target: str
    prompt: str


def slug(value: str):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def direct_prompt(entity: str):
    return f"A node directly connected to {entity} in the memorized entity graph is"


def two_hop_prompt(entity: str):
    return f"A node exactly two links from {entity} in the memorized entity graph is"


def all_socrates_edges(dataset: str, split: str):
    rows = load_dataset(dataset, split=split)
    edges = set()
    for i, row in enumerate(rows):
        c = row_to_chain(dict(row), i)
        for left, right in ((c.e1, c.e2), (c.e2, c.e3)):
            if left != right and 1 < len(left) <= 55 and 1 < len(right) <= 55:
                if "\n" not in left and "\n" not in right:
                    edges.add(tuple(sorted((left, right))))
    return sorted(edges)


def connected_tree(edges, n_nodes: int, seed: int, max_children: int = 6):
    full = defaultdict(set)
    for left, right in edges:
        full[left].add(right); full[right].add(left)
    # Work only in the giant component.
    seen, components = set(), []
    for start in sorted(full):
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            node = stack.pop(); component.append(node)
            for neighbor in full[node]:
                if neighbor not in seen:
                    seen.add(neighbor); stack.append(neighbor)
        components.append(component)
    giant = set(max(components, key=len))
    candidates = sorted(giant, key=lambda x: (-len(full[x]), x))
    rng = random.Random(seed)
    for root in candidates[:min(300, len(candidates))]:
        chosen, tree_edges, queue = {root}, [], deque([root])
        local_rng = random.Random(rng.randrange(2**31))
        while queue and len(chosen) < n_nodes:
            node = queue.popleft()
            neighbors = sorted(full[node] & giant)
            local_rng.shuffle(neighbors)
            added = 0
            for neighbor in neighbors:
                if neighbor in chosen:
                    continue
                chosen.add(neighbor); tree_edges.append((node, neighbor)); queue.append(neighbor)
                added += 1
                if len(chosen) == n_nodes or added == max_children:
                    break
        if len(chosen) == n_nodes:
            return sorted(chosen), tree_edges
    raise RuntimeError(f"Could not construct a {n_nodes}-node bounded-degree connected tree")


def adjacency_and_distances(nodes, edges):
    index = {name: i for i, name in enumerate(nodes)}
    adjacency = [set() for _ in nodes]
    for left, right in edges:
        a, b = index[left], index[right]
        adjacency[a].add(b); adjacency[b].add(a)
    distances = np.full((len(nodes), len(nodes)), np.inf)
    for source in range(len(nodes)):
        distances[source, source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if not np.isfinite(distances[source, neighbor]):
                    distances[source, neighbor] = distances[source, node] + 1
                    queue.append(neighbor)
    return adjacency, distances


def graph_spectrum(adjacency):
    n = len(adjacency)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, neighbors in enumerate(adjacency):
        matrix[i, list(neighbors)] = 1
    degree = matrix.sum(1)
    inv = np.diag(1 / np.sqrt(np.maximum(degree, 1)))
    laplacian = np.eye(n) - inv @ matrix @ inv
    values, vectors = np.linalg.eigh(laplacian)
    return matrix, values, vectors


def base_entities(model, tokenizer, nodes):
    embedding = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    return np.asarray([
        embedding[tokenizer(node, add_special_tokens=False)["input_ids"]].mean(0)
        for node in nodes
    ], dtype=np.float32)


def unit_rows(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def kernel_alignment(x, y):
    def center(k):
        return k - k.mean(0, keepdims=True) - k.mean(1, keepdims=True) + k.mean()
    x, y = center(x), center(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.sum(x * y) / denom) if denom else None


def geometry_metrics(matrix, distances, eigenvectors, permutations=200, seed=13):
    unit = unit_rows(matrix)
    similarity = unit @ unit.T
    upper = np.triu_indices(len(matrix), 1)
    d, scores = distances[upper], similarity[upper]
    edge_auc = roc_auc_score(d == 1, scores)
    two_mask = d >= 2
    two_auc = roc_auc_score(d[two_mask] == 2, scores[two_mask])
    distance_rho = spearmanr(scores, -d).statistic if np.std(scores) > 1e-12 else None
    k = min(12, len(matrix) - 1)
    low = eigenvectors[:, 1:k + 1]
    target = low @ low.T
    alignment = kernel_alignment(similarity, target)
    centered = unit - unit.mean(0, keepdims=True)
    total = np.sum(centered * centered)
    energy = float(np.sum((low.T @ centered) ** 2) / total) if total > 1e-12 else None
    rng, null = np.random.default_rng(seed), []
    for _ in range(permutations):
        permuted = centered[rng.permutation(len(centered))]
        denominator = float(np.sum(permuted * permuted))
        if denominator > 1e-12:
            null.append(float(np.sum((low.T @ permuted) ** 2) / denominator))
    return {
        "edge_auc": float(edge_auc), "unseen_two_hop_auc": float(two_auc),
        "distance_spearman": float(distance_rho) if distance_rho is not None else None,
        "low_frequency_cka": alignment,
        "low_frequency_energy": energy,
        "permutation_energy_mean": float(np.mean(null)) if null else None,
        "spectral_enrichment": float(energy / np.mean(null)) if energy is not None and null else None,
        "effective_rank": effective_rank(matrix),
    }


def effective_rank(matrix):
    singular = np.linalg.svd(matrix - matrix.mean(0, keepdims=True), compute_uv=False)
    p = singular / max(singular.sum(), 1e-12)
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def training_examples(nodes, edges):
    result = []
    for left, right in edges:
        result.append(LocalExample(left, right, direct_prompt(left)))
        result.append(LocalExample(right, left, direct_prompt(right)))
    assert not any("two links" in row.prompt for row in result)
    assert {row.source for row in result} == set(nodes)
    return result


def train(model, tokenizer, nodes, edges, cfg, args, out):
    checkpoint = out / "final.pt"
    history_path = out / "history.json"
    if checkpoint.exists() and history_path.exists() and not args.overwrite:
        item = torch.load(checkpoint, map_location="cpu", weights_only=False)
        table = ResidualTable(item["keys"], model.get_input_embeddings().embedding_dim,
                              item["alpha"], item["mode"]).to(model.get_input_embeddings().weight.device)
        table.load_state_dict(item["state_dict"])
        return table, json.loads(history_path.read_text())
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    keys = [f"entity:{node}" for node in nodes]
    table = ResidualTable(keys, model.get_input_embeddings().embedding_dim, 1.0, "entity_span").to(
        model.get_input_embeddings().weight.device)
    rows = training_examples(nodes, edges)
    optimizer = torch.optim.AdamW([table.delta], lr=args.learning_rate, weight_decay=0)
    norms, guard = base_row_norms(model, table), FrozenParameterGuard(model)
    device, history = model.get_input_embeddings().weight.device, []
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(rows)
        losses = []
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            encoded = [encode_example(tokenizer, row.prompt, row.source, row.target, table) for row in chunk]
            batch = pad_batch(encoded, tokenizer.pad_token_id, device)
            prediction = residual_forward(model, table, batch).loss
            loss = prediction + args.anchor * table.anchor_loss(norms)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            assert table.delta.grad is not None and torch.isfinite(table.delta.grad).all()
            torch.nn.utils.clip_grad_norm_([table.delta], 1.0)
            optimizer.step(); guard.assert_unchanged()
            losses.append(float(prediction.detach()))
        history.append({"epoch": epoch + 1, "prediction_loss": float(np.mean(losses)),
                        "residual_norm": float(table.delta.detach().float().norm())})
        print(f"epoch {epoch + 1}/{args.epochs}: loss={history[-1]['prediction_loss']:.4f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    entity_key = {node: f"entity:{node}" for node in nodes}
    save_residual(checkpoint, table, {"model": args.model, "seed": args.seed,
                  "epochs": args.epochs, "learning_rate": args.learning_rate,
                  "anchor": args.anchor, "entity_key": entity_key,
                  "bidirectional_local_edges": True, "two_hop_training_examples": 0})
    history_path.write_text(json.dumps(history, indent=2))
    return table, history


@torch.no_grad()
def generation_accuracy(model, tokenizer, table, nodes, adjacency, distances, args):
    from geometry_llm.modeling import greedy_generate_batch
    local_valid = [[nodes[j] for j in adjacency[i]] for i in range(len(nodes))]
    two_valid = [[nodes[j] for j in np.flatnonzero(distances[i] == 2)] for i in range(len(nodes))]
    batch = args.eval_batch_size
    result = {}
    for label, prompt_fn, valid in (("local", direct_prompt, local_valid),
                                    ("two_hop", two_hop_prompt, two_valid)):
        predictions = []
        for start in range(0, len(nodes), batch):
            selected = nodes[start:start + batch]
            predictions.extend(greedy_generate_batch(model, tokenizer, table,
                               [prompt_fn(x) for x in selected], selected, args.max_answer_tokens))
        correct = [normalize_answer(p) in {normalize_answer(v) for v in answers}
                   for p, answers in zip(predictions, valid)]
        result[label] = {"accuracy": float(np.mean(correct)), "n": len(correct),
                         "predictions": predictions, "correct": correct}
    return result


@torch.no_grad()
def layerwise_geometry(model, tokenizer, table, nodes, distances, eigenvectors, args):
    device = model.get_input_embeddings().weight.device
    layer_ids = sorted(set(np.linspace(0, model.config.num_hidden_layers, 9).round().astype(int).tolist()))
    store = {site: {layer: {kind: [] for kind in ("h0", "hd", "change")}
                    for layer in layer_ids} for site in ("entity", "final")}
    for start in range(0, len(nodes), args.eval_batch_size):
        selected = nodes[start:start + args.eval_batch_size]
        encoded = [encode_example(tokenizer, direct_prompt(node), node, None, table) for node in selected]
        batch = pad_batch(encoded, tokenizer.pad_token_id, device)
        alpha = table.alpha
        table.alpha = 0; h0 = residual_forward(model, table, batch, True).hidden_states
        table.alpha = alpha; hd = residual_forward(model, table, batch, True).hidden_states
        for row, item in enumerate(encoded):
            entity = [i for i, value in enumerate(item.delta_indices) if value >= 0]
            for layer in layer_ids:
                for site, positions in (("entity", entity), ("final", [item.prompt_length - 1])):
                    before = h0[layer][row, positions].float().mean(0).cpu().numpy()
                    after = hd[layer][row, positions].float().mean(0).cpu().numpy()
                    store[site][layer]["h0"].append(before)
                    store[site][layer]["hd"].append(after)
                    store[site][layer]["change"].append(after - before)
    rows = []
    for site, layers in store.items():
        for layer, spaces in layers.items():
            h0, hd = np.asarray(spaces["h0"]), np.asarray(spaces["hd"])
            change = np.asarray(spaces["change"])
            row = {"site": site, "layer": layer,
                   "relative_change": float(np.mean(np.linalg.norm(change, axis=1) /
                                                     np.maximum(np.linalg.norm(h0, axis=1), 1e-12)))}
            for kind, matrix in (("base", h0), ("adapted", hd), ("change", change)):
                metrics = geometry_metrics(matrix, distances, eigenvectors, 50, args.seed + layer)
                for name in ("edge_auc", "unseen_two_hop_auc", "distance_spearman",
                             "low_frequency_cka", "spectral_enrichment"):
                    row[f"{kind}_{name}"] = metrics[name]
            rows.append(row)
    return rows


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.facecolor": BG,
        "figure.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False, "savefig.bbox": "tight"})


def polish(ax, grid="y"):
    for side in ("top", "right", "left", "bottom"): ax.spines[side].set_visible(False)
    ax.tick_params(length=0); ax.grid(axis=grid, color=GRID, linewidth=.8); ax.set_axisbelow(True)


def save_plot(fig, out, stem):
    fig.savefig(out / f"{stem}.png", dpi=260); fig.savefig(out / f"{stem}.pdf"); plt.close(fig)


def plot_input_geometry(model_label, nodes, edges, eigenvectors, distances, metrics, generation, history, out):
    setup_style(); fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    coords = eigenvectors[:, 1:3]
    ax = axes[0, 0]
    index = {n: i for i, n in enumerate(nodes)}
    for left, right in edges:
        a, b = index[left], index[right]
        ax.plot(coords[[a, b], 0], coords[[a, b], 1], color=GRID, linewidth=.65, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], s=18, color=TEAL, alpha=.8, zorder=2)
    ax.set_title("A   Connected calibration graph", loc="left"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    ax = axes[0, 1]
    labels = ["Base E₀", "Raw Δ", "E₀ + Δ"]
    values = [metrics[x]["unseen_two_hop_auc"] for x in ("base", "delta", "adapted")]
    ax.bar(labels, values, color=[GREY, GOLD, TEAL], width=.62)
    ax.axhline(.5, color=GRID, linestyle="--")
    for i, value in enumerate(values): ax.text(i, value + .018, f"{value:.2f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1); ax.set_ylabel("Distance-2 vs farther AUC")
    ax.set_title("B   Unseen two-hop geometry", loc="left"); polish(ax)

    ax = axes[1, 0]
    values = [metrics[x]["spectral_enrichment"] for x in ("base", "delta", "adapted")]
    ax.bar(labels, values, color=[GREY, GOLD, TEAL], width=.62)
    ax.axhline(1, color=GRID, linestyle="--")
    for i, value in enumerate(values): ax.text(i, value + .04, f"{value:.2f}×", ha="center", fontweight="bold")
    ax.set_ylabel("Low-frequency energy / permutation null")
    ax.set_title("C   Graph-spectrum enrichment", loc="left"); polish(ax)

    ax = axes[1, 1]
    ax.plot([x["epoch"] for x in history], [x["prediction_loss"] for x in history],
            color=TEAL, marker="o", linewidth=2.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Local-edge training loss")
    ax.set_title("D   Local-only optimization", loc="left"); polish(ax)
    ax.text(.98, .95, f"local generation: {generation['local']['accuracy']:.1%}\n"
            f"unseen two-hop: {generation['two_hop']['accuracy']:.1%}", transform=ax.transAxes,
            ha="right", va="top", color=MUTED, fontsize=9)
    fig.suptitle(f"Geometric-memory calibration · {model_label}", fontsize=17,
                 fontweight="bold", x=.06, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .95)); save_plot(fig, out, "input_geometry")


def plot_layers(model_label, rows, out):
    setup_style(); fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True)
    panels = (("relative_change", "Propagated change", "Relative magnitude"),
              ("change_low_frequency_cka", "Geometry in the new representation", "Low-frequency CKA"),
              ("change_unseen_two_hop_auc", "Unseen two-hop structure", "Distance-2 AUC"))
    for ax, (field, title, ylabel) in zip(axes, panels):
        for site, color, style in (("entity", TEAL, "-"), ("final", GOLD, "--")):
            selected = sorted((r for r in rows if r["site"] == site), key=lambda x: x["layer"])
            ax.plot([r["layer"] for r in selected], [r[field] for r in selected],
                    color=color, linestyle=style, marker="o", markersize=4, linewidth=2, label=site.capitalize())
        if "auc" in field: ax.axhline(.5, color=GRID, linestyle="--")
        ax.set_title(title, loc="left"); ax.set_xlabel("Layer"); ax.set_ylabel(ylabel); polish(ax)
    axes[-1].legend()
    fig.suptitle(f"Where does residual geometry live? · {model_label}", fontsize=16,
                 fontweight="bold", x=.06, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .93)); save_plot(fig, out, "layerwise_geometry")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--anchor", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-answer-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["model"]["name"] = args.model
    cfg["model"]["device_map"] = "auto"
    run_tag = f"lr-{args.learning_rate:g}_anchor-{args.anchor:g}_epochs-{args.epochs}"
    out = (Path(cfg["output_dir"]) / "geometric_calibration" / slug(args.model) /
           f"seed-{args.seed}" / run_tag)
    out.mkdir(parents=True, exist_ok=True)
    graph_path = Path(cfg["output_dir"]) / "geometric_calibration" / f"graph-{args.nodes}-seed-{args.seed}.json"
    if graph_path.exists():
        graph = json.loads(graph_path.read_text()); nodes, edges = graph["nodes"], [tuple(x) for x in graph["edges"]]
    else:
        nodes, edges = connected_tree(all_socrates_edges(cfg["data"]["dataset"], cfg["data"]["split"]),
                                      args.nodes, args.seed)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2))
    adjacency, distances = adjacency_and_distances(nodes, edges)
    adjacency_matrix, eigenvalues, eigenvectors = graph_spectrum(adjacency)
    model, tokenizer = load_model_and_tokenizer(cfg)
    # Validate every selected entity span before spending compute on training.
    for node in nodes:
        encode_example(tokenizer, direct_prompt(node), node, None,
                       ResidualTable([f"entity:{node}"], 1, 1.0, "entity_span"))
    table, history = train(model, tokenizer, nodes, edges, cfg, args, out)
    base = base_entities(model, tokenizer, nodes)
    delta = table.delta.detach().float().cpu().numpy()
    adapted = base + table.alpha * delta
    metrics = {name: geometry_metrics(matrix, distances, eigenvectors, 200, args.seed)
               for name, matrix in (("base", base), ("delta", delta), ("adapted", adapted))}
    generation = generation_accuracy(model, tokenizer, table, nodes, adjacency, distances, args)
    layers = layerwise_geometry(model, tokenizer, table, nodes, distances, eigenvectors, args)
    diameter = int(np.max(distances))
    summary = {"model": args.model, "seed": args.seed, "nodes": len(nodes), "edges": len(edges),
               "diameter": diameter, "bidirectional_local_training": True,
               "two_hop_training_examples": 0, "metrics": metrics,
               "generation": {k: {"accuracy": v["accuracy"], "n": v["n"]} for k, v in generation.items()},
               "layers": layers}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "predictions.jsonl").open("w") as handle:
        for i, node in enumerate(nodes):
            handle.write(json.dumps({"node": node, "local_prediction": generation["local"]["predictions"][i],
                "local_correct": generation["local"]["correct"][i],
                "two_hop_prediction": generation["two_hop"]["predictions"][i],
                "two_hop_correct": generation["two_hop"]["correct"][i]}) + "\n")
    plot_input_geometry(args.model.split("/")[-1], nodes, edges, eigenvectors, distances,
                        metrics, generation, history, out)
    plot_layers(args.model.split("/")[-1], layers, out)
    print(json.dumps({k: v for k, v in summary.items() if k != "layers"}, indent=2))
    print(f"Calibration artifacts: {out}")


if __name__ == "__main__":
    main()
