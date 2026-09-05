#!/usr/bin/env python
"""Compare direct embedding deformations across the paired frozen LLMs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, INK, MUTED, GRID = "#F7F5F2", "#18323F", "#6B7C83", "#D9DFDF"
QWEN, LLAMA, GREY = "#087E8B", "#D85B5B", "#78888E"


def load(path): return json.loads(Path(path).read_text())


def records(data, condition="true_label"):
    return [r["metrics"] for r in data["records"] if r["condition"] == condition]


def avg(rows, getter): return float(np.mean([getter(r) for r in rows]))


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5,"axes.titlesize":12,
        "axes.titleweight":"bold","axes.facecolor":BG,"figure.facecolor":BG,"savefig.facecolor":BG,
        "text.color":INK,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,
        "legend.frameon":False,"savefig.bbox":"tight"})


def polish(ax, grid="y"):
    for side in ("top","right","left","bottom"): ax.spines[side].set_visible(False)
    ax.tick_params(length=0); ax.grid(axis=grid,color=GRID,linewidth=.8); ax.set_axisbelow(True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--qwen",default="outputs/c243073e2862/embedding_structure/metrics.json")
    p.add_argument("--llama",default="outputs/bb53acb4d683/embedding_structure/metrics.json")
    p.add_argument("--output",default="outputs/model_comparison")
    a=p.parse_args(); datasets=[load(a.qwen),load(a.llama)]
    labels=["Qwen 2.5 7B","Llama 3.1 8B"]; colors=[QWEN,LLAMA]; true=[records(d) for d in datasets]
    style(); fig,axes=plt.subplots(2,2,figsize=(12.8,8.2))

    ax=axes[0,0]; roles=["source","bridge","answer"]; x=np.arange(3); width=.32
    for i,(rows,label,color) in enumerate(zip(true,labels,colors)):
        vals=[]
        for role in roles:
            raw=[r["role_stats"][role]["relative_step_mean"] for r in rows]
            vals.append(np.mean([v for v in raw if v is not None]) if any(v is not None for v in raw) else 0)
        bars=ax.bar(x+(i-.5)*width,vals,width,color=color,alpha=.9,label=label)
        for bar,v in zip(bars,vals):
            if v: ax.text(bar.get_x()+bar.get_width()/2,v+.018,f"{v:.2f}×",ha="center",color=color,fontweight="bold",fontsize=9)
    ax.text(2,.035,"not updated",ha="center",color=MUTED,fontsize=8)
    ax.set_xticks(x,["Source\nuniversity","Bridge\ncountry","Answer\nanthem"]); ax.set_ylabel("Mean ‖Δeff‖ / ‖E₀‖")
    ax.set_title("A   The update is role-asymmetric",loc="left"); ax.legend(); polish(ax)

    ax=axes[0,1]; names=["CKA","Pairwise cosine\nrank correlation","5-NN\noverlap"]; x=np.arange(3)
    getters=[lambda r:r["base_to_adapted_cka"],lambda r:r["pairwise_cosine_spearman"],lambda r:r["knn_overlap"]["5"]]
    for i,(rows,label,color) in enumerate(zip(true,labels,colors)):
        vals=[avg(rows,g) for g in getters]
        ax.bar(x+(i-.5)*width,vals,width,color=color,alpha=.9,label=label)
    ax.set_xticks(x,names); ax.set_ylim(0,1.05); ax.set_ylabel("Preservation (1 = unchanged)")
    ax.set_title("B   Coarse geometry survives; neighborhoods turn over",loc="left"); polish(ax)

    ax=axes[1,0]; dims=[1,2,5,10,20,40]
    for data,label,color in zip(datasets,labels,colors):
        t,r=records(data,"true_label"),records(data,"norm_matched_random")
        tv=[avg(t,lambda m,k=k:m["delta_energy_in_base_entity_pcs"][str(k)]) for k in dims]
        rv=[avg(r,lambda m,k=k:m["delta_energy_in_base_entity_pcs"][str(k)]) for k in dims]
        ax.plot(dims,tv,marker="o",color=color,linewidth=2.3,label=f"{label} · trained")
        ax.plot(dims,rv,color=color,linestyle="--",alpha=.55,label=f"{label} · random")
    ax.set_xlabel("Leading PCs of original entity cloud"); ax.set_ylabel("Fraction of Δ energy in subspace")
    ax.set_title("C   Most update energy lies outside dominant E₀ axes",loc="left"); ax.legend(fontsize=8); polish(ax)

    ax=axes[1,1]; conditions=[None,"true_label","shuffled_label","norm_matched_random"]
    names=["Original E₀","True-label\nE₀ + Δ","Shuffled-label\nE₀ + Δ","Random\nE₀ + Δ"]; x=np.arange(4)
    for i,(data,label,color) in enumerate(zip(datasets,labels,colors)):
        vals=[]
        base=records(data)[0]["source_geometry"]["base"]["same_target_auc"]
        for condition in conditions:
            vals.append(base if condition is None else avg(records(data,condition),lambda m:m["source_geometry"]["adapted"]["same_target_auc"]))
        ax.plot(x,vals,marker="o",linewidth=2.3,color=color,label=label)
    ax.axhline(.5,color=GRID,linestyle="--"); ax.set_xticks(x,names); ax.set_ylim(.48,.61)
    ax.set_ylabel("Same-country pair AUC"); ax.set_title("D   Target clustering is weak and non-specific",loc="left")
    ax.legend(); polish(ax)

    fig.suptitle("What geometric structure does the residual add?",fontsize=17,fontweight="bold",x=.06,ha="left")
    fig.text(.06,.948,"Effective entity-span displacement Δeff = Δ / √(span length) · three seeds per model",color=MUTED)
    fig.tight_layout(rect=(0,0,1,.92)); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/"embedding_structure_comparison.png",dpi=260); fig.savefig(out/"embedding_structure_comparison.pdf"); plt.close(fig)
    print(f"Comparison: {out}")


if __name__=="__main__": main()
