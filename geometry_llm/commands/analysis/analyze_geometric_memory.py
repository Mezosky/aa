#!/usr/bin/env python
"""Paper-inspired tests for global geometric rather than merely local memory.

The selected SOCRATES graph is heterogeneous and directed, unlike the paper's
symbolic graph.  Geometry tests therefore use a symmetrized graph only as an
analysis object and keep retrieval candidates role-conditioned.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import filter_token_mode, load_chains
from geometry_llm.modeling import load_model_and_tokenizer


BG = "#F7F5F2"
INK = "#18323F"
MUTED = "#6B7C83"
GRID = "#D9DFDF"
COLORS = {
    "base": "#78888E",
    "correct": "#087E8B",
    "shuffled": "#D85B5B",
    "random": "#D79A22",
}
LABELS = {
    "base": "Base E₀",
    "correct": "True-label\ntrained Δ",
    "shuffled": "Shuffled-label\ntrained Δ",
    "random": "Untrained\nnorm-matched Δ",
}


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open()] if path.exists() else None


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.facecolor": BG, "figure.facecolor": BG,
        "savefig.facecolor": BG, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def polish(ax, grid="y"):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.8, alpha=.8)
    ax.set_axisbelow(True)


def save(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.png", dpi=260)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)


def checkpoint_runs(root: Path, condition: str):
    selected = read_json(root / "training" / f"{condition}_selected.json") or {}
    return selected.get("runs", [])


def entity_base_matrix(chains, tokenizer, model):
    names = sorted({value for chain in chains for value in (chain.e1, chain.e2, chain.e3)})
    embedding = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    rows = []
    for name in names:
        ids = tokenizer(name, add_special_tokens=False)["input_ids"]
        rows.append(embedding[ids].mean(0))
    return names, np.asarray(rows, dtype=np.float32)


def delta_matrix(checkpoint: Path, names: list[str]):
    item = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rows = item["state_dict"]["delta"].float().numpy()
    key_index = {key: i for i, key in enumerate(item["keys"])}
    entity_key = item.get("metadata", {}).get("entity_key", {})
    matrix = np.zeros((len(names), rows.shape[1]), dtype=np.float32)
    for i, name in enumerate(names):
        key = entity_key.get(name)
        if key in key_index:
            matrix[i] = rows[key_index[key]]
    return matrix, float(item.get("alpha", 1.0))


def role_sets(chains):
    return {
        "source": sorted({c.e1 for c in chains}),
        "bridge": sorted({c.e2 for c in chains}),
        "answer": sorted({c.e3 for c in chains}),
    }


def normalized_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12), norms[:, 0]


def retrieval_metrics(matrix, names, chains, left_role, right_role):
    index = {name: i for i, name in enumerate(names)}
    candidates = sorted({getattr(c, right_role) for c in chains})
    targets = defaultdict(set)
    for chain in chains:
        targets[getattr(chain, left_role)].add(getattr(chain, right_role))
    unit, norms = normalized_rows(matrix)
    candidate_ids = [index[name] for name in candidates]
    if max(norms[candidate_ids], default=0.0) < 1e-10:
        return {name: None for name in ("cosine_auc", "cosine_hit1", "cosine_mrr", "cosine_margin",
                                        "dot_auc", "dot_hit1", "dot_mrr", "dot_margin")}

    result = {}
    for metric, scores_all in (("cosine", unit @ unit[candidate_ids].T),
                               ("dot", matrix @ matrix[candidate_ids].T)):
        labels_flat, scores_flat, hit, reciprocal, margins = [], [], [], [], []
        for left, valid in targets.items():
            row = scores_all[index[left]]
            positive = np.array([candidate in valid for candidate in candidates])
            labels_flat.extend(positive.astype(int).tolist())
            scores_flat.extend(row.tolist())
            order = np.argsort(-row, kind="stable")
            ranks = np.flatnonzero(positive[order]) + 1
            hit.append(float(ranks[0] == 1)); reciprocal.append(float(1 / ranks[0]))
            margins.append(float(np.max(row[positive]) - np.max(row[~positive])))
        result[f"{metric}_auc"] = float(roc_auc_score(labels_flat, scores_flat))
        result[f"{metric}_hit1"] = float(np.mean(hit))
        result[f"{metric}_mrr"] = float(np.mean(reciprocal))
        result[f"{metric}_margin"] = float(np.mean(margins))
    return result


def component_basis(names, chains):
    index = {name: i for i, name in enumerate(names)}
    adjacency = [set() for _ in names]
    for c in chains:
        for left, right in ((c.e1, c.e2), (c.e2, c.e3)):
            a, b = index[left], index[right]
            adjacency[a].add(b); adjacency[b].add(a)
    component = np.full(len(names), -1, dtype=int)
    count = 0
    for start in range(len(names)):
        if component[start] >= 0:
            continue
        stack = [start]; component[start] = count
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if component[neighbor] < 0:
                    component[neighbor] = count; stack.append(neighbor)
        count += 1
    indicators = np.eye(count, dtype=np.float64)[component]
    indicators -= indicators.mean(0, keepdims=True)
    q, r = np.linalg.qr(indicators)
    rank = int(np.sum(np.abs(np.diag(r)) > 1e-9))
    return q[:, :rank], component, adjacency


def centered_kernel(matrix):
    unit, _ = normalized_rows(matrix)
    unit = unit - unit.mean(0, keepdims=True)
    kernel = unit @ unit.T
    return kernel, unit


def kernel_alignment(left, right):
    h_left = left - left.mean(0, keepdims=True) - left.mean(1, keepdims=True) + left.mean()
    h_right = right - right.mean(0, keepdims=True) - right.mean(1, keepdims=True) + right.mean()
    denom = np.linalg.norm(h_left) * np.linalg.norm(h_right)
    return float(np.sum(h_left * h_right) / denom) if denom else None


def spectral_metrics(matrix, component_q, permutations, seed):
    kernel, centered = centered_kernel(matrix)
    target = component_q @ component_q.T
    total = float(np.sum(centered * centered))
    energy = float(np.sum((component_q.T @ centered) ** 2) / total) if total else None
    alignment = kernel_alignment(kernel, target)
    rng = np.random.default_rng(seed)
    null_energy, null_alignment = [], []
    for _ in range(permutations):
        order = rng.permutation(len(matrix))
        shuffled = centered[order]
        denom = float(np.sum(shuffled * shuffled))
        null_energy.append(float(np.sum((component_q.T @ shuffled) ** 2) / denom) if denom else 0.0)
        null_alignment.append(kernel_alignment(kernel[np.ix_(order, order)], target) or 0.0)
    return {
        "component_subspace_energy": energy,
        "component_kernel_alignment": alignment,
        "permutation_energy_mean": float(np.mean(null_energy)),
        "permutation_energy_std": float(np.std(null_energy)),
        "permutation_alignment_mean": float(np.mean(null_alignment)),
        "permutation_alignment_std": float(np.std(null_alignment)),
        "energy_enrichment": float(energy / np.mean(null_energy)) if energy is not None and np.mean(null_energy) else None,
    }


def relation_consistency(matrix, names, chains):
    index = {name: i for i, name in enumerate(names)}
    output = {}
    for relation, left, right in (("r1", "e1", "e2"), ("r2", "e2", "e3")):
        vectors = np.asarray([matrix[index[getattr(c, right)]] - matrix[index[getattr(c, left)]] for c in chains])
        centroid = vectors.mean(0)
        denom = np.linalg.norm(vectors, axis=1) * np.linalg.norm(centroid)
        output[relation] = float(np.mean((vectors @ centroid) / np.maximum(denom, 1e-12)))
    return output


def geometry_behavior_association(matrix, names, chains, prediction_rows):
    """Test whether unseen-pair geometry predicts per-chain two-hop success."""
    index = {name: i for i, name in enumerate(names)}
    answers = sorted({c.e3 for c in chains})
    answer_ids = [index[x] for x in answers]
    unit, _ = normalized_rows(matrix)
    correctness = {str(row["chain_id"]): int(row["correct_12"]) for row in prediction_rows}
    margins, labels = [], []
    for chain in chains:
        if str(chain.chain_id) not in correctness:
            continue
        scores = unit[index[chain.e1]] @ unit[answer_ids].T
        target = answers.index(chain.e3)
        wrong = np.delete(scores, target)
        margins.append(float(scores[target] - np.max(wrong)))
        labels.append(correctness[str(chain.chain_id)])
    labels_array, margins_array = np.asarray(labels), np.asarray(margins)
    if not len(labels) or len(np.unique(labels_array)) < 2:
        return {"n": len(labels), "n_correct": int(labels_array.sum()),
                "margin_auc_for_two_hop_correctness": None, "margin_correctness_correlation": None}
    return {
        "n": len(labels), "n_correct": int(labels_array.sum()),
        "margin_auc_for_two_hop_correctness": float(roc_auc_score(labels_array, margins_array)),
        "margin_correctness_correlation": float(np.corrcoef(margins_array, labels_array)[0, 1]),
        "mean_margin_correct": float(margins_array[labels_array == 1].mean()),
        "mean_margin_incorrect": float(margins_array[labels_array == 0].mean()),
    }


def measure(matrix, names, chains, roles, component_q, permutations, seed, prediction_rows=None):
    norms = {role: np.linalg.norm(matrix[[names.index(x) for x in entities]], axis=1) for role, entities in roles.items()}
    result = {
        "role_norms": {role: {"mean": float(v.mean()), "median": float(np.median(v)),
                              "max": float(v.max()), "zero_fraction": float(np.mean(v < 1e-10))}
                       for role, v in norms.items()},
        "r1_source_to_bridge": retrieval_metrics(matrix, names, chains, "e1", "e2"),
        "r2_bridge_to_answer": retrieval_metrics(matrix, names, chains, "e2", "e3"),
        "two_hop_source_to_answer": retrieval_metrics(matrix, names, chains, "e1", "e3"),
        "spectral": spectral_metrics(matrix, component_q, permutations, seed),
        "relation_vector_consistency": relation_consistency(matrix, names, chains),
    }
    if prediction_rows is not None:
        result["geometry_behavior_association"] = geometry_behavior_association(
            matrix, names, chains, prediction_rows)
    return result


def plot_summary(records, out, chance):
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), gridspec_kw={"hspace": .48, "wspace": .25})
    condition_order = ["base", "correct", "shuffled", "random"]

    raw = [r for r in records if r["space"] == "delta" and r["condition"] == "correct"]
    roles = ["source", "bridge", "answer"]
    vals = [[r["metrics"]["role_norms"][role]["mean"] for r in raw] for role in roles]
    ax = axes[0, 0]
    for i, role_vals in enumerate(vals):
        ax.scatter(np.full(len(role_vals), i) + np.linspace(-.06, .06, len(role_vals)), role_vals,
                   color=("#66A9AF", "#D79A22", "#D85B5B")[i], s=48, alpha=.7)
        ax.scatter(i, np.mean(role_vals), marker="D", s=74, color=INK, edgecolor=BG, zorder=4)
    ax.set_xticks(range(3), ["Source\nuniversity", "Bridge\ncountry", "Answer\nanthem"])
    ax.set_ylabel("Mean residual-row L₂ norm")
    ax.set_title("A   The answer residual is structurally unlearnable", loc="left")
    polish(ax)

    def condition_panel(ax, field, title, ylabel, ylim, text_offset, chance_line=None):
        for i, condition in enumerate(condition_order):
            subset = [r for r in records if r["space"] == "adapted" and r["condition"] == condition]
            ys = [r["metrics"]["two_hop_source_to_answer"].get(field) for r in subset]
            ys = [y for y in ys if y is not None]
            if not ys:
                continue
            offsets = np.linspace(-.06, .06, len(ys))
            ax.scatter(i + offsets, ys, s=34, color=COLORS[condition], alpha=.62, zorder=3)
            ax.scatter(i, np.mean(ys), marker="D", s=75, color=COLORS[condition], edgecolor=BG, zorder=4)
            ax.text(i, min(np.mean(ys) + text_offset, ylim[1] - text_offset),
                    f"{np.mean(ys):.2f}", ha="center",
                    fontweight="bold", color=COLORS[condition], fontsize=9)
        if chance_line is not None:
            ax.axhline(chance_line, color=GRID, linestyle="--", linewidth=1.5)
            ax.text(3.4, chance_line + .015, "chance", ha="right", fontsize=8, color=MUTED)
        ax.set_xticks(range(4), [LABELS[x] for x in condition_order])
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left"); ax.set_ylabel(ylabel); polish(ax)

    condition_panel(axes[0, 1], "cosine_auc", "B   Unseen source → answer separation",
                    "Cosine AUC (0.5 = no separation)", (.44, .525), .003, .5)
    condition_panel(axes[1, 0], "cosine_hit1", "C   Unseen answer retrieval",
                    "Role-conditioned Hit@1", (0, .09), .004, chance)

    ax = axes[1, 1]
    for i, condition in enumerate(condition_order):
        subset = [r for r in records if r["space"] == "adapted" and r["condition"] == condition]
        ys = [r["metrics"]["spectral"]["energy_enrichment"] for r in subset]
        offsets = np.linspace(-.06, .06, len(ys))
        ax.scatter(i + offsets, ys, s=34, color=COLORS[condition], alpha=.62)
        ax.scatter(i, np.mean(ys), marker="D", s=75, color=COLORS[condition], edgecolor=BG, zorder=4)
        ax.text(i, np.mean(ys) + .004, f"{np.mean(ys):.2f}×", ha="center", fontweight="bold",
                color=COLORS[condition], fontsize=9)
    ax.axhline(1, color=GRID, linestyle="--", linewidth=1.5)
    ax.set_xticks(range(4), [LABELS[x] for x in condition_order])
    ax.set_ylim(.98, 1.17)
    ax.set_ylabel("Energy / row-permutation null")
    ax.set_title("D   Low-frequency graph spectral enrichment", loc="left")
    polish(ax)

    fig.suptitle("Does the external residual form global geometric memory?", fontsize=17,
                 fontweight="bold", x=.06, ha="left", y=.995)
    fig.text(.06, .958,
             "Paper-inspired tests on the symmetrized university–country–anthem graph · small dots are seeds",
             color=MUTED, fontsize=10)
    save(fig, out, "geometric_memory_tests")


def plot_heatmaps(base, adapted, names, chains, out):
    index = {name: i for i, name in enumerate(names)}
    target = {c.e1: c.e3 for c in chains}
    bridge = {c.e1: c.e2 for c in chains}
    answer_country = {}
    for c in chains:
        answer_country[c.e3] = c.e2
    answers = sorted(answer_country, key=lambda x: answer_country[x])
    sources = sorted(target, key=lambda x: (bridge[x], x))
    source_ids = [index[x] for x in sources]; answer_ids = [index[x] for x in answers]
    base_unit, _ = normalized_rows(base); adapted_unit, _ = normalized_rows(adapted)
    matrices = [base_unit[source_ids] @ base_unit[answer_ids].T,
                adapted_unit[source_ids] @ adapted_unit[answer_ids].T]
    limits = np.quantile(np.concatenate([x.ravel() for x in matrices]), [.01, .99])
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6), sharex=True, sharey=True)
    for ax, matrix, title in zip(axes, matrices, ("A   Base embedding E₀", "B   After true-label training E₀ + Δ")):
        image = ax.imshow(matrix, cmap="mako" if "mako" in plt.colormaps() else "YlGnBu",
                          aspect="auto", vmin=limits[0], vmax=limits[1])
        true_cols = [answers.index(target[source]) for source in sources]
        ax.scatter(true_cols, range(len(sources)), s=4, color="#F7C948", alpha=.9, linewidth=0)
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlabel("Candidate anthem, grouped by country")
        ax.set_xticks(range(len(answers)), [answer_country[x] for x in answers], rotation=65, ha="right", fontsize=7)
        ax.tick_params(axis="y", length=0); ax.set_yticks([])
        for side in ax.spines.values(): side.set_visible(False)
    axes[0].set_ylabel("Source universities, grouped by country")
    cb = fig.colorbar(image, ax=axes, fraction=.025, pad=.025); cb.outline.set_visible(False); cb.set_label("Cosine similarity")
    fig.suptitle("A global test: source and answer entities never co-occurred in training",
                 fontsize=16, fontweight="bold", x=.07, ha="left", y=.99)
    fig.text(.07, .945, "Yellow dots mark each university's correct two-hop anthem.", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=.86, bottom=.22, left=.07, right=.91, wspace=.08)
    save(fig, out, "source_answer_cosine_heatmaps")


def plot_geometry_behavior_link(records, out):
    correct = [r for r in records if r["condition"] == "correct" and r["space"] == "adapted"]
    associations = [r["metrics"].get("geometry_behavior_association", {}) for r in correct]
    associations = [a for a in associations if a.get("margin_auc_for_two_hop_correctness") is not None]
    if not associations:
        return
    setup_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.8, 4.5))
    aucs = [a["margin_auc_for_two_hop_correctness"] for a in associations]
    ax.scatter(np.linspace(-.06, .06, len(aucs)), aucs, color=COLORS["correct"], s=48, alpha=.7)
    ax.scatter(0, np.mean(aucs), marker="D", color=COLORS["correct"], edgecolor=BG, s=82, zorder=4)
    ax.axhline(.5, color=GRID, linestyle="--", linewidth=1.4)
    ax.text(.13, .505, "chance", color=MUTED, fontsize=8)
    ax.set_xlim(-.25, .25); ax.set_ylim(0, 1); ax.set_xticks([0], ["True-label trained Δ"])
    ax.set_ylabel("AUC predicting two-hop correctness")
    ax.set_title("A   Does cosine margin predict success?", loc="left")
    polish(ax)

    for i, a in enumerate(associations):
        ys = [a["mean_margin_incorrect"], a["mean_margin_correct"]]
        bx.plot([0, 1], ys, color=COLORS["correct"], alpha=.42, linewidth=1.8)
        bx.scatter([0, 1], ys, color=COLORS["correct"], alpha=.65, s=38)
    bx.axhline(0, color=GRID, linewidth=1.1)
    bx.set_xticks([0, 1], ["Incorrect chains", "Correct chains"])
    bx.set_ylabel("Mean true-answer cosine margin")
    bx.set_title("B   Successful chains are not geometrically closer", loc="left")
    polish(bx)
    fig.suptitle("Input-space geometry does not explain the two-hop successes", fontsize=15,
                 fontweight="bold", x=.06, ha="left")
    fig.tight_layout()
    save(fig, out, "geometry_behavior_link")


def main():
    parser = common_parser("Test the residual for paper-style global geometric memory")
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    raw_chains = load_chains(cfg)
    root = output_path(cfg)
    model, tokenizer = load_model_and_tokenizer(cfg)
    chains, failures = filter_token_mode(raw_chains, tokenizer, cfg["data"]["token_mode"])
    names, base = entity_base_matrix(chains, tokenizer, model)
    roles = role_sets(chains)
    component_q, components, adjacency = component_basis(names, chains)
    base_predictions = read_jsonl(root / "predictions" / "original.jsonl")
    records = [{
        "condition": "base", "space": "adapted", "seed": None,
        "metrics": measure(base, names, chains, roles, component_q, args.permutations,
                           cfg["analysis"]["seed"], base_predictions),
    }]

    correct_adapted = []
    correct_runs = checkpoint_runs(root, "correct_delta")
    shuffled_by_seed = {int(run["seed"]): run for run in checkpoint_runs(root, "shuffled_delta")}
    for run in correct_runs:
        seed = int(run["seed"])
        delta, alpha = delta_matrix(Path(run["checkpoint"]), names)
        adapted = base + alpha * delta
        correct_predictions = read_jsonl(root / "predictions" / f"correct_delta_seed-{seed}.jsonl")
        correct_adapted.append(adapted)
        records.extend((
            {"condition": "correct", "space": "delta", "seed": seed,
             "metrics": measure(delta, names, chains, roles, component_q, args.permutations, seed)},
            {"condition": "correct", "space": "adapted", "seed": seed,
             "metrics": measure(adapted, names, chains, roles, component_q, args.permutations,
                                seed, correct_predictions)},
        ))
        shuffled_run = shuffled_by_seed.get(seed)
        if shuffled_run:
            shuffled, shuffled_alpha = delta_matrix(Path(shuffled_run["checkpoint"]), names)
            shuffled_predictions = read_jsonl(root / "predictions" / f"shuffled_delta_seed-{seed}.jsonl")
            records.extend((
                {"condition": "shuffled", "space": "delta", "seed": seed,
                 "metrics": measure(shuffled, names, chains, roles, component_q, args.permutations, seed)},
                {"condition": "shuffled", "space": "adapted", "seed": seed,
                 "metrics": measure(base + shuffled_alpha * shuffled, names, chains, roles,
                                    component_q, args.permutations, seed, shuffled_predictions)},
            ))
        rng = np.random.default_rng(seed)
        random = rng.standard_normal(delta.shape).astype(np.float32)
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        random *= norms / np.maximum(np.linalg.norm(random, axis=1, keepdims=True), 1e-12)
        random_predictions = read_jsonl(root / "predictions" / f"random_delta_seed-{seed}.jsonl")
        records.extend((
            {"condition": "random", "space": "delta", "seed": seed,
             "metrics": measure(random, names, chains, roles, component_q, args.permutations, seed)},
            {"condition": "random", "space": "adapted", "seed": seed,
             "metrics": measure(base + alpha * random, names, chains, roles, component_q,
                                args.permutations, seed, random_predictions)},
        ))

    out = output_path(cfg, "geometric_memory")
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": cfg["model"]["name"], "composition_type": cfg["data"]["composition_type"],
        "n_chains": len(chains), "n_nodes": len(names), "n_components": int(components.max() + 1),
        "n_edges": int(sum(len(x) for x in adjacency) // 2), "span_failures": failures,
        "analysis_note": "Graph is symmetrized only for spectral analysis; retrieval remains role-conditioned.",
        "identifiability_note": "Answer entities are never residualized in teacher forcing, so their delta rows receive no gradient.",
        "label_kernel_note": "For this composition, country and anthem are one-to-one, so bridge and answer label kernels are identical.",
    }
    (out / "metrics.json").write_text(json.dumps({"metadata": metadata, "records": records}, indent=2))
    chance = 1 / len(roles["answer"])
    plot_summary(records, out, chance)
    plot_geometry_behavior_link(records, out)
    if correct_adapted:
        plot_heatmaps(base, np.mean(correct_adapted, axis=0), names, chains, out)
    print(json.dumps(metadata, indent=2))
    print(f"Geometric-memory tests: {out}")


if __name__ == "__main__":
    main()
