#!/usr/bin/env python
"""Render publication-style figures from completed experiment artifacts."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import load_chains


BG = "#F7F5F2"
INK = "#18323F"
MUTED = "#6B7C83"
GRID = "#D9DFDF"
COLORS = {
    "original": "#78888E",
    "correct_delta": "#087E8B",
    "shuffled_delta": "#D85B5B",
    "random_delta": "#D79A22",
}
LABELS = {
    "original": "Original",
    "correct_delta": "True-label\ntrained Δ",
    "shuffled_delta": "Shuffled-label\ntrained Δ",
    "random_delta": "Untrained\nnorm-matched Δ",
}
ORDER = tuple(LABELS)
MODEL_LABEL = "Frozen LLM"


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "savefig.facecolor": BG,
        "savefig.bbox": "tight",
    })


def polish(ax, grid="y"):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.8, alpha=.75, zorder=0)
    ax.set_axisbelow(True)


def save(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.png", dpi=260)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    return [out / f"{stem}.png", out / f"{stem}.pdf"]


def selected_rows(root: Path):
    result = read_json(root / "training" / "correct_delta_selected.json") or {}
    return result.get("runs", [])


def result_data(root: Path):
    data = defaultdict(lambda: {"all": [], "knowledge_conditioned": []})
    baseline = read_json(root / "baseline_summary.json") or {}
    overall = baseline.get("overall", baseline)
    for subset in data["original"]:
        if subset in overall:
            data["original"][subset].append(overall[subset])
    for path in sorted((root / "summaries").glob("*_seed-*.json")):
        condition = path.name.split("_seed-", 1)[0]
        item = read_json(path) or {}
        for subset in data[condition]:
            if subset in item:
                data[condition][subset].append(item[subset])
    return data


def values(data, condition, subset, metric):
    return np.array([
        row[metric] for row in data[condition][subset]
        if row.get(metric) is not None
    ], dtype=float)


def dot_summary(ax, data, subset, metric, title, ylabel=None):
    present = [c for c in ORDER if values(data, c, subset, metric).size]
    for x, condition in enumerate(present):
        ys = values(data, condition, subset, metric)
        color = COLORS[condition]
        if len(ys) > 1:
            offsets = np.linspace(-.08, .08, len(ys))
            ax.scatter(x + offsets, ys, s=30, color=color, alpha=.6,
                       edgecolor=BG, linewidth=.6, zorder=3)
            low, high = np.quantile(ys, [.025, .975])
            ax.vlines(x, low, high, color=color, linewidth=2, zorder=2)
        ax.scatter(x, np.mean(ys), marker="D", s=72, color=color,
                   edgecolor=BG, linewidth=1.1, zorder=4)
        ax.text(x, min(1.02, np.mean(ys) + .055), f"{np.mean(ys):.1%}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)
    ax.set_xticks(range(len(present)), [LABELS[c] for c in present])
    ax.set_ylim(-.02, 1.08)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{x:.0%}" for x in np.linspace(0, 1, 6)])
    ax.set_title(title, loc="left", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel)
    polish(ax)


def plot_main_results(root: Path, out: Path):
    data = result_data(root)
    if not data:
        return []
    fig = plt.figure(figsize=(13.3, 8.1))
    gs = fig.add_gridspec(2, 2, hspace=.48, wspace=.22)

    ax = fig.add_subplot(gs[0, 0])
    present = [c for c in ORDER if values(data, c, "all", "A_1a").size]
    for x, condition in enumerate(present):
        a = values(data, condition, "all", "A_1a")
        b = values(data, condition, "all", "A_1b")
        color = COLORS[condition]
        for aa, bb in zip(a, b):
            ax.plot([x - .14, x + .14], [aa, bb], color=color, alpha=.28, linewidth=1.3)
            ax.scatter([x - .14, x + .14], [aa, bb], color=color, alpha=.5, s=19, zorder=3)
        ma, mb = np.mean(a), np.mean(b)
        ax.plot([x - .14, x + .14], [ma, mb], color=color, linewidth=3, zorder=4)
        ax.scatter([x - .14, x + .14], [ma, mb], color=color, s=57,
                   edgecolor=BG, linewidth=1, zorder=5)
        ax.text(x - .14, ma + .045, f"{ma:.0%}", color=color, ha="center", fontweight="bold")
        ax.text(x + .14, mb + .045, f"{mb:.0%}", color=color, ha="center", fontweight="bold")
    ax.set_xticks(range(len(present)), [LABELS[c] for c in present])
    ax.set_ylim(-.02, 1.08)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{x:.0%}" for x in np.linspace(0, 1, 6)])
    ax.set_ylabel("Exact-match accuracy")
    ax.set_title("A   Local facts are learned", loc="left", pad=12)
    ax.text(.01, .97, "left dot: university → country    right dot: country → anthem",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")
    polish(ax)

    dot_summary(fig.add_subplot(gs[0, 1]), data, "all", "A_2",
                "B   Held-out two-hop answer", "Exact-match accuracy")
    dot_summary(fig.add_subplot(gs[1, 0]), data, "knowledge_conditioned", "A_2",
                "C   Two-hop answer when both facts were known", "Exact-match accuracy")
    dot_summary(fig.add_subplot(gs[1, 1]), data, "all", "C",
                "D   Conditional composability", "P(two-hop correct | both hops correct)")

    n_all = int(values(data, "original", "all", "n")[0]) if values(data, "original", "all", "n").size else 0
    n_known = int(values(data, "original", "knowledge_conditioned", "n")[0]) if values(data, "original", "knowledge_conditioned", "n").size else 0
    fig.suptitle("Embedding-only local training transfers to held-out composed prompts",
                 fontsize=17, fontweight="bold", x=.06, ha="left", y=.995)
    fig.text(.06, .955,
             f"{MODEL_LABEL} · university → country → anthem · {n_all} paths · "
             f"{n_known} baseline-known paths · diamonds show means; small dots show seeds",
             fontsize=10, color=MUTED)
    return save(fig, out, "main_results")


def plot_graph(root: Path, out: Path):
    chain_path = root / "selected_chains.jsonl"
    baseline_path = root / "predictions" / "original.jsonl"
    if not chain_path.exists():
        return []
    chains = [json.loads(line) for line in chain_path.open()]
    known = set()
    if baseline_path.exists():
        for line in baseline_path.open():
            row = json.loads(line)
            if row.get("correct_1a") and row.get("correct_1b"):
                known.add(str(row.get("chain_id")))
    example = next((c for c in chains if str(c["chain_id"]) in known), chains[0])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.3, 5.5), gridspec_kw={"width_ratios": [1.55, 1]})
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    xs = [.17, .5, .83]
    roles = [("SOURCE", "University", example["e1"]),
             ("BRIDGE", "Country", example["e2"]),
             ("ANSWER", "Anthem", example["e3"])]
    fills = ["#E6F3F3", "#FFF0D8", "#F9E6E2"]
    for x, (role, kind, value), fill in zip(xs, roles, fills):
        box = FancyBboxPatch((x - .125, .41), .25, .25,
                             boxstyle="round,pad=0.018,rounding_size=.025",
                             facecolor=fill, edgecolor="none")
        ax.add_patch(box)
        ax.text(x, .615, role, ha="center", va="center", fontsize=8, color=MUTED, fontweight="bold")
        ax.text(x, .565, kind, ha="center", va="center", fontsize=10, color=INK)
        shown = value if len(value) <= 27 else value[:25] + "…"
        ax.text(x, .49, shown, ha="center", va="center", fontsize=9, color=INK,
                fontweight="bold", wrap=True)
    for x1, x2, label in ((.295, .375, "r₁"), (.625, .705, "r₂")):
        ax.add_patch(FancyArrowPatch((x1, .535), (x2, .535), arrowstyle="-|>",
                                     mutation_scale=14, color=INK, linewidth=1.8))
        ax.text((x1 + x2) / 2, .575, label, ha="center", fontsize=10, fontweight="bold")
    ax.add_patch(FancyArrowPatch((.18, .37), (.82, .37), connectionstyle="arc3,rad=.23",
                                 arrowstyle="-|>", mutation_scale=15, color=COLORS["correct_delta"],
                                 linewidth=2.2))
    ax.text(.5, .20, "TEST  ·  unseen composed path", ha="center", color=COLORS["correct_delta"],
            fontsize=10, fontweight="bold")
    ax.text(.5, .76, "TRAIN  ·  two local directed edges", ha="center", color=INK,
            fontsize=10, fontweight="bold")
    ax.set_title("A   The experiment as a graph", loc="left", pad=10)

    counts = Counter(c["e2"] for c in chains)
    top = counts.most_common(9)[::-1]
    labels, vals = zip(*top) if top else ([], [])
    y = np.arange(len(labels))
    bx.hlines(y, 0, vals, color=GRID, linewidth=2.5)
    bx.scatter(vals, y, s=75, color=COLORS["correct_delta"], zorder=3)
    for yy, val in zip(y, vals):
        bx.text(val + .5, yy, str(val), va="center", color=INK, fontweight="bold", fontsize=9)
    bx.set_yticks(y, labels)
    bx.set_xlim(0, max(vals) * 1.2 if vals else 1)
    bx.set_xlabel("Number of university paths")
    bx.set_title("B   Bridges create shared subgraphs", loc="left", pad=10)
    polish(bx, "x")
    fig.suptitle("Local edges in the dataset define a compositional graph",
                 fontsize=16, fontweight="bold", x=.06, ha="left", y=1.02)
    fig.tight_layout(w_pad=3)
    return save(fig, out, "graph_experiment")


def plot_hyperparameters(root: Path, out: Path):
    rows = read_json(root / "training" / "correct_delta_grid.json") or []
    chosen = read_json(root / "training" / "correct_delta_selected.json") or {}
    if not rows:
        return []
    lrs = sorted({float(r["lr"]) for r in rows})
    anchors = sorted({float(r["anchor"]) for r in rows})
    matrix = np.full((len(lrs), len(anchors)), np.nan)
    for i, lr in enumerate(lrs):
        for j, anchor in enumerate(anchors):
            v = [r["best_validation_loss"] for r in rows
                 if float(r["lr"]) == lr and float(r["anchor"]) == anchor]
            matrix[i, j] = np.mean(v) if v else np.nan
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(matrix, cmap="YlGnBu_r", aspect="auto")
    threshold = np.nanmean(matrix)
    for i in range(len(lrs)):
        for j in range(len(anchors)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else INK, fontweight="bold")
    ax.set_xticks(range(len(anchors)), [f"{x:g}" for x in anchors])
    ax.set_yticks(range(len(lrs)), [f"{x:g}" for x in lrs])
    ax.set_xlabel("Anchor strength λ"); ax.set_ylabel("Learning rate")
    ax.set_title("One-hop validation selects the residual—not two-hop performance", loc="left", pad=14)
    if chosen:
        i = lrs.index(float(chosen["learning_rate"])); j = anchors.index(float(chosen["anchor"]))
        ax.scatter(j, i, marker="*", s=330, color="#F7C948", edgecolor=INK, linewidth=1.1)
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.04)
    cb.outline.set_visible(False); cb.set_label("Mean best validation loss")
    fig.tight_layout()
    return save(fig, out, "hyperparameter_selection")


def checkpoint_paths(root: Path):
    paths = []
    for run in selected_rows(root):
        path = Path(run.get("checkpoint", ""))
        if path.exists():
            paths.append(path)
    return paths


def plot_training(root: Path, out: Path):
    runs = selected_rows(root)
    series = []
    for run in runs:
        checkpoint = Path(run["checkpoint"])
        relative = checkpoint.parent.relative_to(root / "checkpoints")
        history = root / "training" / relative / "history.json"
        rows = read_json(history) or []
        if rows:
            series.append((run["seed"], rows))
    if not series:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25))
    for seed, rows in series:
        epochs = np.arange(1, len(rows) + 1)
        axes[0].plot(epochs, [r["validation_loss"] for r in rows], color=COLORS["correct_delta"],
                     alpha=.38, linewidth=1.8, marker="o", markersize=3, label=f"seed {seed}")
        axes[1].plot(epochs, [r["residual_norm"] for r in rows], color=COLORS["correct_delta"],
                     alpha=.38, linewidth=1.8, marker="o", markersize=3, label=f"seed {seed}")
    axes[0].set(title="A   One-hop validation loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="B   Residual norm", xlabel="Epoch", ylabel="L₂ norm")
    for ax in axes:
        polish(ax); ax.legend(ncol=3, fontsize=8, loc="best")
    fig.suptitle("Selected runs are stable across random seeds", fontsize=15, fontweight="bold", x=.07, ha="left")
    fig.tight_layout()
    return save(fig, out, "training_dynamics")


def plot_residual_roles(root: Path, out: Path):
    paths = checkpoint_paths(root)
    chains_path = root / "selected_chains.jsonl"
    if not paths or not chains_path.exists():
        return []
    chains = [json.loads(line) for line in chains_path.open()]
    role_entities = {
        "Source\nuniversity": {c["e1"] for c in chains},
        "Bridge\ncountry": {c["e2"] for c in chains},
        "Answer\nanthem": {c["e3"] for c in chains},
    }
    pooled = defaultdict(list)
    for path in paths:
        item = torch.load(path, map_location="cpu", weights_only=False)
        delta = item["state_dict"]["delta"].float().numpy()
        index = {key: i for i, key in enumerate(item["keys"])}
        entity_key = item.get("metadata", {}).get("entity_key", {})
        for role, entities in role_entities.items():
            ids = [index[entity_key[e]] for e in entities if e in entity_key and entity_key[e] in index]
            pooled[role].extend(np.linalg.norm(delta[ids], axis=1).tolist())
    labels = list(role_entities)
    distributions = [np.asarray(pooled[x]) for x in labels]
    if not all(len(x) for x in distributions):
        return []
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    vp = ax.violinplot(distributions, showextrema=False, widths=.72)
    for body, color in zip(vp["bodies"], ("#8BC7CB", "#E5B55F", "#E99B8F")):
        body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(.75)
    rng = np.random.default_rng(9)
    global_top = max(float(np.max(vals)) for vals in distributions)
    for i, vals in enumerate(distributions, 1):
        keep = rng.choice(vals, min(180, len(vals)), replace=False)
        ax.scatter(rng.normal(i, .055, len(keep)), keep, s=8, color=INK, alpha=.18, linewidth=0)
        ax.scatter(i, np.median(vals), marker="D", s=48, color=INK, edgecolor=BG, linewidth=.8, zorder=4)
        label_y = max(float(np.quantile(vals, .99)) * 1.06, global_top * .045)
        ax.text(i, label_y, f"n={len(vals)}", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(range(1, 4), labels); ax.set_ylabel("Residual row L₂ norm")
    ax.set_title("Where does the learned embedding update live?", loc="left", pad=14)
    polish(ax)
    fig.tight_layout()
    return save(fig, out, "residual_by_graph_role")


def plot_geometry(root: Path, out: Path):
    path = root / "geometry" / "pair_distributions.csv"
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open()))
    spaces = [s for s in ("delta", "base_plus_delta") if any(r["space"] == s for r in rows)]
    pair_types = [p for p in ("direct_r1", "direct_r2", "indirect", "unrelated_direct")
                  if any(r["pair_type"] == p for r in rows)]
    if not spaces or not pair_types:
        return []
    fig, axes = plt.subplots(1, len(spaces), figsize=(12, 4.6), squeeze=False)
    pair_colors = ["#8BC7CB", "#4B9DA5", "#E5B55F", "#B7BEC1"]
    for ax, space in zip(axes[0], spaces):
        vals = [np.array([float(r["cosine"]) for r in rows
                         if r["space"] == space and r["pair_type"] == pair]) for pair in pair_types]
        vp = ax.violinplot(vals, showextrema=False, widths=.8)
        for body, color in zip(vp["bodies"], pair_colors):
            body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(.8)
        for i, v in enumerate(vals, 1):
            ax.scatter(i, np.median(v), marker="D", s=35, color=INK, zorder=4)
        ax.axhline(0, color=GRID, linewidth=1)
        ax.set_xticks(range(1, len(pair_types) + 1),
                      [x.replace("direct_", "direct ").replace("unrelated_direct", "unrelated")
                       for x in pair_types])
        ax.set_title("Learned Δ" if space == "delta" else "Base embedding + Δ", loc="left")
        ax.set_ylabel("Cosine similarity"); polish(ax)
    fig.suptitle("Graph structure is reflected mainly in the adapted input geometry",
                 fontsize=15, fontweight="bold", x=.07, ha="left")
    fig.tight_layout()
    return save(fig, out, "geometry_distributions")


def plot_layers(root: Path, out: Path):
    rows = read_json(root / "layers" / "metrics.json") or []
    if not rows:
        return []
    panels = [("relative_change_magnitude", "Change magnitude"),
              ("preservation_cka", "Representation preservation"),
              ("answer_alignment_cka", "Answer alignment")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), sharex=True)
    for ax, (metric, title) in zip(axes, panels):
        for site, color, linestyle in (("entity", COLORS["correct_delta"], "-"),
                                       ("final", COLORS["random_delta"], "--")):
            subset = sorted((r for r in rows if r["site"] == site), key=lambda r: r["layer"])
            ax.plot([r["layer"] for r in subset], [r[metric] for r in subset],
                    color=color, linestyle=linestyle, linewidth=2.1, label=site.capitalize())
        ax.set_title(title, loc="left"); ax.set_xlabel("Transformer layer"); polish(ax)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend()
    fig.suptitle("The input residual propagates through the network", fontsize=15,
                 fontweight="bold", x=.06, ha="left")
    fig.tight_layout()
    return save(fig, out, "layerwise_propagation")


def plot_interventions(root: Path, out: Path):
    rows = read_json(root / "interventions" / "summary.json") or []
    if not rows:
        return []
    rows = [r for r in rows if r.get("A_2") is not None]
    if not rows:
        return []
    scales = sorted((r for r in rows if r["intervention"].startswith("scale_")), key=lambda r: r["alpha"])
    controls = [r for r in rows if not r["intervention"].startswith("scale_")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    styles = (("A_1a", "#66A9AF", "o", "Hop 1"),
              ("A_1b", "#087E8B", "s", "Hop 2"),
              ("A_2", "#D85B5B", "D", "Two hop"))
    for metric, color, marker, label in styles:
        ax.plot([r["alpha"] for r in scales], [r[metric] for r in scales], color=color,
                marker=marker, markersize=6, linewidth=2.3, label=label)
    ax.axvline(1, color=GRID, linestyle="--", linewidth=1.3)
    ax.text(1, .98, "trained scale", transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=8, color=MUTED)
    ax.set_ylim(-.02, 1.02); ax.set_xlabel("Residual scale α"); ax.set_ylabel("Exact-match accuracy")
    ax.set_title("A   Dose–response", loc="left"); ax.legend(ncol=3, fontsize=8)
    polish(ax)

    labels = [r["intervention"].replace("direction_removal_rank_", "remove top ")
              .replace("delta_removal", "remove Δ").replace("delta_permutation", "permute rows")
              for r in controls]
    y = np.arange(len(controls))[::-1]
    for metric, color, marker, label in styles:
        bx.scatter([r[metric] for r in controls], y, color=color, marker=marker,
                   s=54, label=label, alpha=.9)
    bx.set_yticks(y, labels); bx.set_xlim(-.02, 1.02)
    bx.set_xticks(np.linspace(0, 1, 6), [f"{x:.0%}" for x in np.linspace(0, 1, 6)])
    bx.set_xlabel("Exact-match accuracy"); bx.set_title("B   Removal and structure controls", loc="left")
    polish(bx, "x")
    fig.suptitle("Performance depends on the learned residual and its entity assignment",
                 fontsize=15, fontweight="bold", x=.06, ha="left")
    fig.tight_layout()
    return save(fig, out, "causal_interventions")


def main():
    global MODEL_LABEL
    args = common_parser("Plot experiment outcomes and learned geometry").parse_args()
    cfg = load_config(args.config, args.set)
    MODEL_LABEL = cfg["model"]["name"]
    # Resolves composition_type=auto before deriving the configuration hash.
    load_chains(cfg)
    root = output_path(cfg)
    out = root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    made = []
    for fn in (plot_main_results, plot_graph, plot_hyperparameters, plot_training,
               plot_residual_roles, plot_geometry, plot_layers, plot_interventions):
        made.extend(fn(root, out))
    (out / "manifest.json").write_text(json.dumps({"plots": [str(p) for p in made]}, indent=2))
    print(f"Wrote {len(made) // 2} figures (PNG + PDF) to {out}")


if __name__ == "__main__":
    main()
