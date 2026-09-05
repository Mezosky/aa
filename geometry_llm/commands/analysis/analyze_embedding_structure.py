#!/usr/bin/env python
"""Analyze how an entity embedding cloud is deformed by a learned residual.

This deliberately does not ask whether embeddings reproduce a global graph.  It
measures the structure of E0 -> E0 + Delta itself: displacement orientation and
scale, residual dimensionality, subspace location, neighborhood deformation, and
local organization of source entities by their supervised target.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import filter_token_mode, load_chains
from geometry_llm.metrics import centered_cka
from geometry_llm.modeling import load_model_and_tokenizer


BG, INK, MUTED, GRID = "#F7F5F2", "#18323F", "#6B7C83", "#D9DFDF"
TEAL, CORAL, GOLD, GREY = "#087E8B", "#D85B5B", "#D79A22", "#78888E"
COLORS = {"true_label": TEAL, "shuffled_label": CORAL, "norm_matched_random": GOLD}
LABELS = {
    "true_label": "True-label\ntrained Δ",
    "shuffled_label": "Shuffled-label\ntrained Δ",
    "norm_matched_random": "Untrained\nnorm-matched Δ",
}


def setup_style():
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
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def save(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.png", dpi=260, facecolor=fig.get_facecolor())
    fig.savefig(out / f"{stem}.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def effective_rank(x):
    s = np.linalg.svd(x - x.mean(0, keepdims=True), compute_uv=False)
    p = s / max(float(s.sum()), 1e-12)
    return float(np.exp(-np.sum(p * np.log(p + 1e-12)))), s


def pairwise_scores(x):
    similarity = unit(x) @ unit(x).T
    upper = np.triu_indices(len(x), 1)
    return similarity[upper]


def cone_geometry(x):
    normalized = unit(x)
    axis = unit(normalized.mean(0, keepdims=True))[0]
    axis_cosine = np.clip(normalized @ axis, -1.0, 1.0)
    pairwise = pairwise_scores(x)
    return {
        "axis_cosine_mean": float(axis_cosine.mean()),
        "axis_angle_mean_deg": float(np.degrees(np.arccos(axis_cosine)).mean()),
        "axis_angle_median_deg": float(np.median(np.degrees(np.arccos(axis_cosine)))),
        "pairwise_cosine_mean": float(pairwise.mean()),
    }


def knn_overlap(before, after, k):
    a, b = unit(before) @ unit(before).T, unit(after) @ unit(after).T
    np.fill_diagonal(a, -np.inf); np.fill_diagonal(b, -np.inf)
    total = 0.0
    for i in range(len(a)):
        left = set(np.argpartition(a[i], -k)[-k:])
        right = set(np.argpartition(b[i], -k)[-k:])
        total += len(left & right) / len(left | right)
    return float(total / len(a))


def knn_geodesics(x, initial_k=10):
    """Shortest paths on a connected cosine kNN graph.

    The neighborhood size is increased only when needed for connectivity. This
    makes the metric finite without inventing distances between components.
    """
    similarity = np.clip(unit(x) @ unit(x).T, -1.0, 1.0)
    distances = np.maximum(1.0 - similarity, 1e-8)
    np.fill_diagonal(distances, np.inf)
    n = len(x)
    for k in range(min(initial_k, n - 1), n):
        neighbors = np.argpartition(distances, k - 1, axis=1)[:, :k]
        graph = np.zeros((n, n), dtype=np.float64)
        rows = np.repeat(np.arange(n), k)
        graph[rows, neighbors.reshape(-1)] = distances[rows, neighbors.reshape(-1)]
        graph = np.maximum(graph, graph.T)
        sparse = csr_matrix(graph)
        components, _ = connected_components(sparse, directed=False)
        if components == 1:
            return shortest_path(sparse, directed=False), k
    raise RuntimeError("Could not construct a connected neighborhood graph")


def geodesic_label_geometry(x, labels, initial_k=10):
    distances, k = knn_geodesics(x, initial_k)
    upper = np.triu_indices(len(x), 1)
    labels = np.asarray(labels)
    same = labels[upper[0]] == labels[upper[1]]
    scores = -distances[upper]
    auc = float(roc_auc_score(same, scores)) if len(np.unique(same)) == 2 else None
    return {"same_target_auc": auc, "knn_k": int(k)}


def geodesic_preservation(before, after, initial_k=10):
    left, k_left = knn_geodesics(before, initial_k)
    right, k_right = knn_geodesics(after, initial_k)
    upper = np.triu_indices(len(before), 1)
    correlation = spearmanr(left[upper], right[upper]).statistic
    return {"distance_spearman": float(correlation),
            "base_knn_k": int(k_left), "adapted_knn_k": int(k_right)}


def label_geometry(x, labels, ks=(1, 3, 5)):
    similarity = unit(x) @ unit(x).T
    upper = np.triu_indices(len(x), 1)
    y = np.asarray(labels)[upper[0]] == np.asarray(labels)[upper[1]]
    scores = similarity[upper]
    auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None
    gap = float(scores[y].mean() - scores[~y].mean()) if y.any() and (~y).any() else None
    np.fill_diagonal(similarity, -np.inf)
    purity = {}
    labels = np.asarray(labels)
    for k in ks:
        neighbors = np.argpartition(similarity, -k, axis=1)[:, -k:]
        purity[str(k)] = float(np.mean(labels[neighbors] == labels[:, None]))
    return {"same_target_auc": auc, "within_minus_between_cosine": gap, "knn_target_purity": purity}


def base_rows(model, tokenizer, names):
    weights = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    rows = []
    for name in names:
        ids = tokenizer(name, add_special_tokens=False)["input_ids"]
        rows.append(weights[ids].mean(0))
    return np.asarray(rows, dtype=np.float32)


def checkpoint_rows(path: Path, names):
    item = torch.load(path, map_location="cpu", weights_only=False)
    keys = [str(x).removeprefix("entity:") for x in item["keys"]]
    matrix = item["state_dict"]["delta"].float().numpy()
    lookup = {name: matrix[i] for i, name in enumerate(keys)}
    return np.asarray([lookup.get(name, np.zeros(matrix.shape[1], np.float32)) for name in names])


def random_like(delta, seed):
    rng = np.random.default_rng(seed)
    result = rng.normal(size=delta.shape).astype(np.float32)
    result = unit(result) * np.linalg.norm(delta, axis=1, keepdims=True)
    return result


def roles_for(names, chains):
    roles = defaultdict(set)
    for c in chains:
        roles[c.e1].add("source"); roles[c.e2].add("bridge"); roles[c.e3].add("answer")
    return ["/".join(sorted(roles[name])) for name in names]


def subspace_fractions(base, delta, ks):
    centered_base = base - base.mean(0, keepdims=True)
    _, _, vh = np.linalg.svd(centered_base, full_matrices=False)
    centered_delta = delta - delta.mean(0, keepdims=True)
    total = float(np.sum(centered_delta ** 2))
    return {str(k): (float(np.sum((centered_delta @ vh[:k].T) ** 2) / total)
                     if total > 1e-12 else None) for k in ks}


def analyze(base, delta, names, roles, source_indices, source_labels, ks):
    changed = np.linalg.norm(delta, axis=1) > 1e-10
    before, movement = base[changed], delta[changed]
    after = before + movement
    base_norm = np.linalg.norm(before, axis=1)
    delta_norm = np.linalg.norm(movement, axis=1)
    radial_cos = np.sum(before * movement, axis=1) / np.maximum(base_norm * delta_norm, 1e-12)
    turn_cos = np.sum(unit(before) * unit(after), axis=1).clip(-1.0, 1.0)
    turn_degrees = np.degrees(np.arccos(turn_cos))
    erank, singular = effective_rank(movement)
    energy = singular ** 2 / max(float(np.sum(singular ** 2)), 1e-12)
    role_stats = {}
    changed_roles = np.asarray(roles)[changed]
    for role in ("source", "bridge", "answer"):
        mask = np.asarray([role in value.split("/") for value in changed_roles])
        role_stats[role] = {
            "n_changed": int(mask.sum()),
            "relative_step_mean": float(np.mean(delta_norm[mask] / base_norm[mask])) if mask.any() else None,
            "radial_cosine_mean": float(np.mean(radial_cos[mask])) if mask.any() else None,
            "tangential_fraction_mean": float(np.mean(np.sqrt(np.maximum(0, 1-radial_cos[mask]**2)))) if mask.any() else None,
            "embedding_rotation_mean_deg": float(np.mean(turn_degrees[mask])) if mask.any() else None,
        }
    source_base = base[source_indices]
    source_delta = delta[source_indices]
    source_adapted = source_base + source_delta
    base_pair, adapted_pair = pairwise_scores(before), pairwise_scores(after)
    return {
        "n_entities": len(names), "n_changed": int(changed.sum()),
        "relative_step_mean": float(np.mean(delta_norm / base_norm)),
        "relative_step_median": float(np.median(delta_norm / base_norm)),
        "radial_cosine_mean": float(np.mean(radial_cos)),
        "tangential_fraction_mean": float(np.mean(np.sqrt(np.maximum(0, 1-radial_cos**2)))),
        "embedding_rotation_mean_deg": float(np.mean(turn_degrees)),
        "embedding_rotation_median_deg": float(np.median(turn_degrees)),
        "cone_geometry": {"base": cone_geometry(before), "adapted": cone_geometry(after)},
        "role_stats": role_stats,
        "delta_effective_rank": erank,
        "delta_variance_top_1": float(energy[:1].sum()),
        "delta_variance_top_5": float(energy[:5].sum()),
        "delta_variance_top_10": float(energy[:10].sum()),
        "delta_variance_top_20": float(energy[:20].sum()),
        "base_to_adapted_cka": centered_cka(before, after),
        "pairwise_cosine_spearman": float(spearmanr(base_pair, adapted_pair).statistic),
        "geodesic_preservation": geodesic_preservation(before, after),
        "knn_overlap": {str(k): knn_overlap(before, after, k) for k in (1, 5, 10)},
        "delta_energy_in_base_entity_pcs": subspace_fractions(before, movement, ks),
        "source_geometry": {
            "base": label_geometry(source_base, source_labels),
            "delta": label_geometry(source_delta, source_labels),
            "adapted": label_geometry(source_adapted, source_labels),
            "base_geodesic": geodesic_label_geometry(source_base, source_labels),
            "adapted_geodesic": geodesic_label_geometry(source_adapted, source_labels),
        },
    }


def mean_metric(records, condition, getter):
    vals = [getter(r["metrics"]) for r in records if r["condition"] == condition]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def plot_deformation(records, out):
    setup_style(); fig, axes = plt.subplots(2, 2, figsize=(13, 8.3))
    conditions = list(LABELS)
    ax = axes[0, 0]
    roles = ["source", "bridge", "answer"]; x=np.arange(3); width=.23
    for i, condition in enumerate(conditions):
        vals=[]
        for role in roles:
            value=mean_metric(records,condition,lambda m,role=role:m["role_stats"][role]["relative_step_mean"])
            vals.append(0 if not np.isfinite(value) else value)
        ax.bar(x+(i-1)*width,vals,width,color=COLORS[condition],alpha=.9,
               label=LABELS[condition].replace("\n"," "))
    ax.set_xticks(x,["Source\nuniversity","Bridge\ncountry","Answer\nanthem"]); ax.set_ylabel("Mean ‖Δ‖ / ‖E₀‖")
    ax.set_title("A   The embedding change is role-asymmetric", loc="left")
    ax.text(2,.06,"not updated",ha="center",color=MUTED,fontsize=9); ax.legend(fontsize=7); polish(ax)

    ax = axes[0, 1]
    for i, condition in enumerate(conditions):
        radial = mean_metric(records, condition, lambda m: m["radial_cosine_mean"])
        tangent = mean_metric(records, condition, lambda m: m["tangential_fraction_mean"])
        ax.bar(i-.16, radial, .32, color=COLORS[condition], alpha=.55)
        ax.bar(i+.16, tangent, .32, color=COLORS[condition], alpha=.95)
    ax.axhline(0, color=GRID, linewidth=1); ax.set_xticks(range(3), [LABELS[c] for c in conditions])
    ax.set_ylabel("Mean directional component"); ax.set_title("B   Motion is radial or tangential?", loc="left")
    ax.legend(handles=[plt.Rectangle((0,0),1,1,color=GREY,alpha=.55,label="Radial cosine"),
                       plt.Rectangle((0,0),1,1,color=GREY,alpha=.95,label="Tangential fraction")])
    polish(ax)

    ax = axes[1, 0]
    dimensions = [1, 5, 10, 20]
    for condition in conditions:
        vals = [mean_metric(records, condition, lambda m, k=k: m[f"delta_variance_top_{k}"]) for k in dimensions]
        ax.plot(dimensions, vals, marker="o", linewidth=2.3, color=COLORS[condition], label=LABELS[condition].replace("\n", " "))
    ax.set_xlabel("Leading residual principal components"); ax.set_ylabel("Cumulative variance explained")
    ax.set_title("C   Dimensionality of the learned displacement", loc="left"); ax.legend(); polish(ax)

    ax = axes[1, 1]
    metrics = [("base_to_adapted_cka", "CKA"), ("pairwise_cosine_spearman", "Pairwise\nrank corr."),
               (("knn_overlap", "5"), "5-NN\noverlap")]
    x = np.arange(len(metrics)); width=.23
    for i, condition in enumerate(conditions):
        vals=[]
        for key,_ in metrics:
            if isinstance(key, tuple): vals.append(mean_metric(records, condition, lambda m,k=key: m[k[0]][k[1]]))
            else: vals.append(mean_metric(records, condition, lambda m,k=key: m[k]))
        ax.bar(x+(i-1)*width, vals, width, color=COLORS[condition], alpha=.9,
               label=LABELS[condition].replace("\n", " "))
    ax.set_xticks(x, [label for _,label in metrics]); ax.set_ylim(0,1.05)
    ax.set_ylabel("Structure preserved (1 = unchanged)"); ax.set_title("D   What survives the deformation?", loc="left")
    ax.legend(fontsize=8); polish(ax)
    fig.suptitle("The geometry of the embedding update", fontsize=17, fontweight="bold", x=.06, ha="left")
    fig.text(.06,.948,"Entity-span residuals · diamonds/lines summarize three seeds · no graph-reconstruction assumption",color=MUTED)
    fig.tight_layout(rect=(0,0,1,.92)); save(fig,out,"embedding_deformation")


def plot_local(records, out, base_dim):
    setup_style(); fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    conditions = list(LABELS)
    spaces = [(None,"E₀",GREY), ("true_label","E₀ + true-label Δ",TEAL),
              ("shuffled_label","E₀ + shuffled Δ",CORAL), ("norm_matched_random","E₀ + random Δ",GOLD)]
    ax=axes[0]
    base_auc = records[0]["metrics"]["source_geometry"]["base"]["same_target_auc"]
    vals=[base_auc]+[mean_metric(records,c,lambda m:m["source_geometry"]["adapted"]["same_target_auc"]) for c in conditions]
    bars=ax.bar(range(4),vals,color=[x[2] for x in spaces],alpha=.9)
    for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,v+.015,f"{v:.2f}",ha="center",fontweight="bold")
    ax.axhline(.5,color=GRID,linestyle="--"); ax.set_ylim(.4,max(.75,max(vals)+.08))
    short_labels=["Original\nE₀","True-label\nE₀ + Δ","Shuffled-label\nE₀ + Δ","Random\nE₀ + Δ"]
    ax.set_xticks(range(4),short_labels); ax.set_ylabel("Same-country pair AUC")
    ax.set_title("A   Do universities sharing a target cluster?",loc="left"); polish(ax)

    ax=axes[1]; ks=[1,3,5]; x=np.arange(3); width=.2
    base_p=records[0]["metrics"]["source_geometry"]["base"]["knn_target_purity"]
    for i,(condition,label,color) in enumerate(spaces):
        if condition is None: values=[base_p[str(k)] for k in ks]
        else: values=[mean_metric(records,condition,lambda m,k=k:m["source_geometry"]["adapted"]["knn_target_purity"][str(k)]) for k in ks]
        ax.bar(x+(i-1.5)*width,values,width,color=color,alpha=.9,label=label)
    ax.set_xticks(x,[f"{k}-nearest" for k in ks]); ax.set_ylabel("Neighbors with same country")
    ax.set_title("B   Local target purity",loc="left"); ax.legend(fontsize=7); polish(ax)

    ax=axes[2]; dims=[1,2,5,10,20,40]
    for condition in conditions:
        vals=[mean_metric(records,condition,lambda m,k=k:m["delta_energy_in_base_entity_pcs"][str(k)]) for k in dims]
        ax.plot(dims,vals,marker="o",linewidth=2.2,color=COLORS[condition],label=LABELS[condition].replace("\n"," "))
    ax.plot(dims,np.asarray(dims)/base_dim,color=GREY,linestyle="--",label="Isotropic expectation")
    ax.set_xlabel("Leading PCs of original entity cloud"); ax.set_ylabel("Fraction of Δ energy inside subspace")
    ax.set_title("C   Where Δ lies relative to E₀",loc="left"); ax.legend(fontsize=7); polish(ax)
    fig.suptitle("Local organization of the adapted embedding space",fontsize=16,fontweight="bold",x=.045,ha="left")
    fig.tight_layout(rect=(0,0,1,.92)); save(fig,out,"local_embedding_organization")


def plot_motion_map(base, delta, source_indices, labels, names, out):
    before=base[source_indices]; after=before+delta[source_indices]
    projection=PCA(n_components=2,random_state=13).fit_transform(np.vstack([before,after]))
    p0,p1=projection[:len(before)],projection[len(before):]
    counts=Counter(labels); common={x for x,_ in counts.most_common(7)}
    palette=[TEAL,CORAL,GOLD,"#8D70B4","#507DBC","#3F8F65","#BB6B9B"]
    cmap={label:palette[i] for i,label in enumerate(sorted(common))}
    setup_style(); fig,ax=plt.subplots(figsize=(10.5,7.2))
    for i,label in enumerate(labels):
        color=cmap.get(label,"#BAC2C4"); alpha=.68 if label in common else .22
        ax.annotate("",xy=p1[i],xytext=p0[i],arrowprops=dict(arrowstyle="->",color=color,alpha=alpha,lw=1))
        ax.scatter(*p0[i],s=12,color=color,alpha=alpha); ax.scatter(*p1[i],s=25,color=color,alpha=alpha,marker="D")
    handles=[plt.Line2D([0],[0],color=c,marker="D",linestyle="",label=l) for l,c in cmap.items()]
    handles.append(plt.Line2D([0],[0],color="#BAC2C4",marker="D",linestyle="",label="Other countries"))
    ax.legend(handles=handles,ncol=2,fontsize=8,loc="best"); ax.set_xlabel("Joint PCA axis 1"); ax.set_ylabel("Joint PCA axis 2")
    ax.set_title("University embeddings before → after true-label training",loc="left",fontsize=15,pad=12)
    ax.text(.01,.98,"Circles: E₀   Diamonds: E₀ + Δ   Arrows: learned displacement (seed 13)",transform=ax.transAxes,va="top",color=MUTED)
    polish(ax,"both"); fig.tight_layout(); save(fig,out,"embedding_motion_map")


def plot_umap_map(base, delta, source_indices, labels, out):
    """Supporting view only; quantitative claims use the geodesic metrics."""
    import umap

    before = base[source_indices]
    after = before + delta[source_indices]
    reducer = umap.UMAP(n_neighbors=12, min_dist=.18, metric="cosine",
                        random_state=13, n_jobs=1)
    p0 = reducer.fit_transform(before)
    p1 = reducer.transform(after)
    counts = Counter(labels)
    common = {value for value, _ in counts.most_common(6)}
    palette = [TEAL, CORAL, GOLD, "#8D70B4", "#507DBC", "#3F8F65"]
    cmap = {label: palette[i] for i, label in enumerate(sorted(common))}
    setup_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for i, label in enumerate(labels):
        color = cmap.get(label, "#BAC2C4")
        alpha = .62 if label in common else .16
        ax.plot([p0[i, 0], p1[i, 0]], [p0[i, 1], p1[i, 1]],
                color=color, alpha=alpha, linewidth=.8, zorder=1)
        ax.scatter(*p0[i], s=15, facecolor=BG, edgecolor=color,
                   linewidth=.8, alpha=alpha, zorder=2)
        ax.scatter(*p1[i], s=20, color=color, marker="D", alpha=alpha, zorder=3)
    country_handles = [plt.Line2D([0], [0], color=color, marker="D", linestyle="",
                                  label=label) for label, color in cmap.items()]
    state_handles = [
        plt.Line2D([0], [0], marker="o", markerfacecolor=BG, markeredgecolor=INK,
                   linestyle="", label="Original"),
        plt.Line2D([0], [0], marker="D", color=INK, linestyle="", label="Adapted"),
    ]
    ax.legend(handles=state_handles + country_handles, ncol=2, fontsize=7,
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    polish(ax, "both")
    fig.tight_layout()
    save(fig, out, "embedding_umap")


def main():
    parser=common_parser("Analyze the geometry of E0 -> E0 + Delta")
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args(); cfg=load_config(args.config,args.set)
    model,tokenizer=load_model_and_tokenizer(cfg)
    chains,_=filter_token_mode(load_chains(cfg),tokenizer,cfg["data"]["token_mode"])
    # load_chains resolves composition_type=auto, which is part of the cache key.
    out=output_path(cfg,"embedding_structure"); out.mkdir(parents=True,exist_ok=True)
    metrics_path=out/"metrics.json"
    names=sorted({x for c in chains for x in (c.e1,c.e2,c.e3)})
    base=base_rows(model,tokenizer,names); name_index={x:i for i,x in enumerate(names)}
    # The entity-span adapter applies Delta/sqrt(span length) to every span token.
    # Analyze that effective per-token displacement rather than the stored raw row.
    span_scales=np.asarray([1/np.sqrt(max(1,len(tokenizer(name,add_special_tokens=False)["input_ids"])))
                            for name in names],dtype=np.float32)[:,None]
    roles=roles_for(names,chains); source_indices=np.asarray([name_index[c.e1] for c in chains]); source_labels=[c.e2 for c in chains]
    root=output_path(cfg); correct=json.loads((root/"training/correct_delta_selected.json").read_text())["runs"]
    shuffled={r["seed"]:r for r in json.loads((root/"training/shuffled_delta_selected.json").read_text())["runs"]}
    ks=(1,2,5,10,20,40); records=[]; motion=None
    for run in correct:
        seed=int(run["seed"]); true_delta=checkpoint_rows(Path(run["checkpoint"]),names)*span_scales
        variants={"true_label":true_delta,
                  "shuffled_label":checkpoint_rows(Path(shuffled[seed]["checkpoint"]),names)*span_scales,
                  "norm_matched_random":random_like(true_delta,seed)}
        if seed==13: motion=true_delta
        for condition,delta in variants.items():
            records.append({"condition":condition,"seed":seed,
                            "metrics":analyze(base,delta,names,roles,source_indices,source_labels,ks)})
    payload={"model":cfg["model"]["name"],"composition_type":cfg["data"]["composition_type"],
             "n_chains":len(chains),"n_entities":len(names),
             "question":"How does E0 -> E0 + Delta deform the entity embedding cloud?",
             "span_scaling":"Effective displacement uses Delta/sqrt(number of entity tokens).",
             "records":records}
    metrics_path.write_text(json.dumps(payload,indent=2,allow_nan=False))
    plot_deformation(records,out); plot_local(records,out,base.shape[1])
    if motion is not None:
        plot_motion_map(base,motion,source_indices,source_labels,names,out)
        plot_umap_map(base,motion,source_indices,source_labels,out)
    print(json.dumps({"output":str(out),"n_entities":len(names),"n_records":len(records)},indent=2))


if __name__=="__main__":
    main()
