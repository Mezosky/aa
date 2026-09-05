#!/usr/bin/env python
"""Create a direct Qwen/Llama comparison from completed calibration runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BG, INK, MUTED, GRID = "#F7F5F2", "#18323F", "#6B7C83", "#D9DFDF"
QWEN, LLAMA = "#087E8B", "#D85B5B"


def latest(root: Path, model_slug: str):
    paths = sorted(root.glob(f"{model_slug}/seed-*/lr-*_anchor-*_epochs-*/summary.json"),
                   key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No calibration summary for {model_slug}")
    return json.loads(paths[-1].read_text()), paths[-1]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.facecolor": BG,
        "figure.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False, "savefig.bbox": "tight"})


def polish(ax, grid="y"):
    for side in ("top", "right", "left", "bottom"): ax.spines[side].set_visible(False)
    ax.tick_params(length=0); ax.grid(axis=grid, color=GRID, linewidth=.8); ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/geometric_calibration")
    args = parser.parse_args()
    root = Path(args.root)
    qwen, qp = latest(root, "qwen-qwen2-5-7b-instruct")
    llama, lp = latest(root, "meta-llama-llama-3-1-8b-instruct")
    models, colors, labels = [qwen, llama], [QWEN, LLAMA], ["Qwen 2.5 7B", "Llama 3.1 8B"]
    style(); fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))

    ax = axes[0, 0]
    x = np.arange(2); width = .32
    for j, task in enumerate(("local", "two_hop")):
        vals = [m["generation"][task]["accuracy"] for m in models]
        bars = ax.bar(x + (j - .5) * width, vals, width, color=["#66A9AF", "#D79A22"][j],
                      label={"local": "Local edge", "two_hop": "Unseen two hop"}[task])
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, value + .025, f"{value:.1%}", ha="center", fontweight="bold")
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.05); ax.set_ylabel("Exact-match accuracy")
    ax.set_title("A   Local fitting is necessary but not sufficient", loc="left"); ax.legend(); polish(ax)

    ax = axes[0, 1]
    spaces = ["base", "delta", "adapted"]
    for i, (m, color, label) in enumerate(zip(models, colors, labels)):
        vals = [m["metrics"][s]["unseen_two_hop_auc"] for s in spaces]
        ax.plot(range(3), vals, marker="o", linewidth=2.3, color=color, label=label)
    ax.axhline(.5, color=GRID, linestyle="--"); ax.set_ylim(.45, .68)
    ax.set_xticks(range(3), ["Base E₀", "Raw Δ", "E₀ + Δ"]); ax.set_ylabel("Distance-2 AUC")
    ax.set_title("B   Weak unseen two-hop geometry appears", loc="left"); ax.legend(); polish(ax)

    ax = axes[1, 0]
    for m, color, label in zip(models, colors, labels):
        vals = [m["metrics"][s]["spectral_enrichment"] for s in spaces]
        ax.plot(range(3), vals, marker="o", linewidth=2.3, color=color, label=label)
    ax.axhline(1, color=GRID, linestyle="--"); ax.set_ylim(.5, 1.15)
    ax.set_xticks(range(3), ["Base E₀", "Raw Δ", "E₀ + Δ"])
    ax.set_ylabel("Low-frequency energy / permutation null")
    ax.set_title("C   No strong Laplacian spectral bias", loc="left"); polish(ax)

    ax = axes[1, 1]
    for m, color, label in zip(models, colors, labels):
        final = sorted((r for r in m["layers"] if r["site"] == "final"), key=lambda r: r["layer"])
        ax.plot([r["layer"] for r in final], [r["change_unseen_two_hop_auc"] for r in final],
                marker="o", linewidth=2.3, color=color, label=label)
    ax.axhline(.5, color=GRID, linestyle="--"); ax.set_ylim(.45, .72)
    ax.set_xlabel("Transformer layer"); ax.set_ylabel("AUC in propagated change at final token")
    ax.set_title("D   Two-hop structure moves to the prediction site", loc="left"); ax.legend(); polish(ax)

    fig.suptitle("Connected-graph geometric calibration across frozen LLMs", fontsize=17,
                 fontweight="bold", x=.06, ha="left")
    fig.text(.06, .945, "64 nodes · diameter 10 · bidirectional local-edge training · zero two-hop training examples",
             color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .92))
    out = root / "comparison"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "model_comparison.png", dpi=260); fig.savefig(out / "model_comparison.pdf"); plt.close(fig)
    (out / "sources.json").write_text(json.dumps({"qwen": str(qp), "llama": str(lp)}, indent=2))
    print(f"Comparison: {out}")


if __name__ == "__main__":
    main()
