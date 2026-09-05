#!/usr/bin/env python
"""Publication-style paired comparison of Qwen and Llama SOCRATES runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BG, INK, MUTED, GRID = "#F7F5F2", "#17313B", "#6C7A80", "#D9DFDF"
QWEN, LLAMA = "#087E8B", "#D85B5B"
ORIGINAL, CORRECT, SHUFFLED, RANDOM = "#AAB5B8", "#D79A22", "#8D70B4", "#507DBC"


def read(path: Path):
    return json.loads(path.read_text())


def baseline(root: Path, subset: str, metric: str) -> float:
    return float(read(root / "baseline_summary.json")["overall"][subset][metric])


def seed_values(root: Path, condition: str, subset: str, metric: str) -> list[float]:
    values = []
    for path in sorted((root / "summaries").glob(f"{condition}_seed-*.json")):
        value = read(path).get(subset, {}).get(metric)
        if value is not None:
            values.append(float(value))
    return values


def mean(root: Path, condition: str, subset: str, metric: str) -> float:
    values = seed_values(root, condition, subset, metric)
    return float(np.mean(values)) if values else float("nan")


def report_value(root: Path, condition: str, subset: str, metric: str):
    if condition == "original":
        value = read(root / "baseline_summary.json")["overall"][subset][metric]
        return None if value is None else float(value)
    values = seed_values(root, condition, subset, metric)
    return float(np.mean(values)) if values else None


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.facecolor": BG,
        "figure.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False, "savefig.bbox": "tight",
    })


def polish(ax, grid="y"):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis=grid, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def dumbbell(ax, roots, labels, colors, subset):
    y = np.arange(len(roots))[::-1]
    for yi, root, label, color in zip(y, roots, labels, colors):
        before = baseline(root, subset, "A_2")
        after = mean(root, "correct_delta", subset, "A_2")
        ax.plot([before, after], [yi, yi], color=GRID, linewidth=7, solid_capstyle="round")
        ax.scatter(before, yi, s=90, color=ORIGINAL, edgecolor=BG, zorder=3)
        ax.scatter(after, yi, s=105, color=color, edgecolor=BG, zorder=4)
        ax.text(before, yi + .18, f"{before:.1%}", ha="center", color=MUTED, fontsize=9)
        ax.text(after, yi + .18, f"{after:.1%}", ha="center", color=color,
                fontsize=9, fontweight="bold")
    ax.set_yticks(y, labels); ax.set_xlim(-.02, .36); ax.set_xlabel("Two-hop exact-match accuracy")
    polish(ax, "x")


def grouped_conditions(ax, roots, labels, colors, subset, metric, title):
    conditions = ["original", "correct_delta", "shuffled_delta", "random_delta"]
    condition_labels = ["Original", "True-label\ntrained Δ", "Shuffled-label\ntrained Δ", "Untrained\nnorm-matched Δ"]
    condition_colors = [ORIGINAL, CORRECT, SHUFFLED, RANDOM]
    x = np.arange(len(conditions)); offsets = [-.12, .12]
    for offset, root, model_label, model_color in zip(offsets, roots, labels, colors):
        for j, (condition, ccolor) in enumerate(zip(conditions, condition_colors)):
            if condition == "original":
                vals = [baseline(root, subset, metric)]
            else:
                vals = seed_values(root, condition, subset, metric)
            if not vals:
                continue
            center = x[j] + offset
            ax.scatter(np.full(len(vals), center), vals, s=31, color=model_color,
                       alpha=.48, edgecolor="none", zorder=3)
            ax.scatter(center, np.mean(vals), s=82, marker="D", color=ccolor,
                       edgecolor=model_color, linewidth=1.4, zorder=4)
    ax.set_xticks(x, condition_labels); ax.set_ylim(-.025, .40)
    ax.set_ylabel("Exact-match accuracy"); ax.set_title(title, loc="left")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", linestyle="", markerfacecolor=c,
                              markeredgecolor=c, label=l) for l, c in zip(labels, colors)],
              loc="upper right")
    polish(ax)


def hop_panel(ax, roots, labels, colors):
    x = np.arange(2); width = .32
    for i, (root, label, color) in enumerate(zip(roots, labels, colors)):
        vals = [mean(root, "correct_delta", "all", metric) for metric in ("A_1a", "A_1b")]
        bars = ax.bar(x + (i - .5) * width, vals, width, color=color, alpha=.88, label=label)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, value + .025, f"{value:.0%}",
                    ha="center", color=color, fontweight="bold")
    ax.set_xticks(x, ["Hop 1: source → bridge", "Hop 2: bridge → answer"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("Accuracy after true-label trained Δ")
    ax.set_title("D   The adapter learns both supervised local relations", loc="left")
    ax.legend(loc="lower right"); polish(ax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-root", default="outputs/c243073e2862")
    parser.add_argument("--llama-root", default="outputs/bb53acb4d683")
    parser.add_argument("--output", default="outputs/model_comparison")
    args = parser.parse_args()
    roots = [Path(args.qwen_root), Path(args.llama_root)]
    labels, colors = ["Qwen 2.5 7B", "Llama 3.1 8B"], [QWEN, LLAMA]
    style(); fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))

    dumbbell(axes[0, 0], roots, labels, colors, "all")
    axes[0, 0].set_title("A   Local supervision transfers to unseen two-hop prompts", loc="left")
    grouped_conditions(axes[0, 1], roots, labels, colors, "all", "A_2",
                       "B   Training on the true local answers matters")
    grouped_conditions(axes[1, 0], roots, labels, colors, "knowledge_conditioned", "A_2",
                       "C   The knowledge-conditioned result is model-dependent")
    hop_panel(axes[1, 1], roots, labels, colors)

    fig.suptitle("Embedding-only adaptation in two frozen language models", fontsize=17,
                 fontweight="bold", x=.06, ha="left")
    fig.text(.06, .945,
             "Same 101 university → country → anthem chains · three adaptation seeds · no two-hop training prompts",
             color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .92))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "natural_language_model_comparison.png", dpi=260)
    fig.savefig(out / "natural_language_model_comparison.pdf")
    plt.close(fig)

    report = {"composition": "anthem of university's country", "n": 101, "models": {}}
    for root, label in zip(roots, labels):
        report["models"][label] = {
            subset: {
                condition: {metric: report_value(root, condition, subset, metric)
                            for metric in ("A_1a", "A_1b", "A_2", "C")}
                for condition in ("original", "correct_delta", "shuffled_delta", "random_delta")
                if condition == "original" or seed_values(root, condition, subset, "A_2")
            } for subset in ("all", "knowledge_conditioned")
        }
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Comparison: {out}")


if __name__ == "__main__":
    main()
