#!/usr/bin/env python
"""Export standalone, title-free figures for the workshop paper."""
from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

BG, INK, MUTED, GRID = "#FFFFFF", "#18323F", "#6B7C83", "#D9DFDF"
QWEN, LLAMA, GREY = "#087E8B", "#D85B5B", "#78888E"


def read(path): return json.loads(Path(path).read_text())


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.facecolor":BG,
        "figure.facecolor":BG,"savefig.facecolor":BG,"text.color":INK,"axes.labelcolor":INK,
        "xtick.color":MUTED,"ytick.color":MUTED,"legend.frameon":False,
        "savefig.bbox":"tight","pdf.fonttype":42,"ps.fonttype":42})


def polish(ax, grid="y"):
    for side in ("top","right","left","bottom"): ax.spines[side].set_visible(False)
    ax.tick_params(length=0); ax.grid(axis=grid,color=GRID,linewidth=.7); ax.set_axisbelow(True)


def save(fig, out, stem):
    fig.tight_layout(pad=.8)
    fig.savefig(out/f"{stem}.pdf",bbox_inches="tight",pad_inches=.06)
    fig.savefig(out/f"{stem}.png",dpi=300,bbox_inches="tight",pad_inches=.06)
    plt.close(fig)


def erows(data, condition="true_label"):
    return [r["metrics"] for r in data["records"] if r["condition"]==condition]


def avg(rows, getter): return float(np.mean([getter(r) for r in rows]))


def sample_cka(x, y):
    """Linear CKA through sample Gram matrices; efficient when samples < features."""
    x=x-x.mean(0,keepdims=True); y=y-y.mean(0,keepdims=True)
    left=x@x.T; right=y@y.T
    denominator=np.linalg.norm(left,"fro")*np.linalg.norm(right,"fro")
    return float(np.sum(left*right)/denominator) if denominator else np.nan


def behavior(out):
    qroot=Path("outputs/c243073e2862"); lroot=Path("outputs/bb53acb4d683")
    conditions=["original","correct_delta","shuffled_delta","random_delta"]
    labels=["Original","True-label\ntrained Δ","Shuffled-label\ntrained Δ","Untrained\nnorm-matched Δ"]
    def vals(root):
        base=read(root/"baseline_summary.json")["overall"]["all"]["A_2"]
        result=[base]
        for c in conditions[1:]: result.append(read(root/"summaries"/f"{c}_aggregate.json")["all"]["A_2"]["mean"])
        return result
    fig,ax=plt.subplots(figsize=(5.6,2.8)); x=np.arange(4)
    for model_i,(root,label,color) in enumerate(((qroot,"Qwen 2.5 7B",QWEN),(lroot,"Llama 3.1 8B",LLAMA))):
        y=vals(root); ax.plot(x,y,marker="o",linewidth=2.2,color=color,label=label)
        for xx,yy in zip(x,y):
            dx = (-13 if model_i == 0 else 13) if xx == 2 else 0
            ax.annotate(f"{yy:.1%}",(xx,yy),xytext=(dx,8),textcoords="offset points",
                        ha="center",color=color,fontsize=7.5,fontweight="bold")
    ax.set_xticks(x,labels); ax.set_ylim(-.01,.36); ax.set_ylabel("Held-out two-hop accuracy")
    ax.legend(ncol=2,loc="lower left",bbox_to_anchor=(0,1.01),borderaxespad=0)
    polish(ax); save(fig,out,"behavior_two_hop")


def embedding_figures(out):
    datasets=[read("outputs/c243073e2862/embedding_structure/metrics.json"),
              read("outputs/bb53acb4d683/embedding_structure/metrics.json")]
    labels=["Qwen 2.5 7B","Llama 3.1 8B"]; colors=[QWEN,LLAMA]; true=[erows(d) for d in datasets]
    roles=["source","bridge","answer"]; x=np.arange(3); width=.32
    fig,ax=plt.subplots(figsize=(4.4,2.85))
    for i,(rows,label,color) in enumerate(zip(true,labels,colors)):
        values=[]
        for role in roles:
            raw=[r["role_stats"][role]["relative_step_mean"] for r in rows]
            values.append(np.mean([v for v in raw if v is not None]) if any(v is not None for v in raw) else 0)
        bars=ax.bar(x+(i-.5)*width,values,width,color=color,alpha=.9,label=label)
        for bar,v in zip(bars,values):
            if v: ax.text(bar.get_x()+bar.get_width()/2,v+.025,f"{v:.2f}×",ha="center",color=color,fontsize=7.5,fontweight="bold")
    ax.text(2,.045,"not updated",ha="center",color=MUTED,fontsize=7)
    ax.set_xticks(x,["Source\nuniversity","Bridge\ncountry","Answer\nanthem"]); ax.set_ylabel("Mean ‖Δeff‖ / ‖E₀‖")
    ax.legend(ncol=2,loc="lower left",bbox_to_anchor=(0,1.01),borderaxespad=0)
    polish(ax); save(fig,out,"embedding_role_steps")

    fig,ax=plt.subplots(figsize=(4.6,2.85)); names=["CKA","Pairwise cosine\nrank corr.","5-NN\noverlap"]; x=np.arange(3)
    getters=[lambda r:r["base_to_adapted_cka"],lambda r:r["pairwise_cosine_spearman"],lambda r:r["knn_overlap"]["5"]]
    for i,(rows,label,color) in enumerate(zip(true,labels,colors)):
        ax.bar(x+(i-.5)*width,[avg(rows,g) for g in getters],width,color=color,alpha=.9,label=label)
    ax.set_xticks(x,names); ax.set_ylim(0,1.05); ax.set_ylabel("Preservation (1 = unchanged)")
    ax.legend(ncol=2,loc="lower left",bbox_to_anchor=(0,1.01),borderaxespad=0)
    polish(ax); save(fig,out,"embedding_preservation")

    fig,ax=plt.subplots(figsize=(4.7,3.0)); dims=[1,2,5,10,20,40]
    for data,label,color in zip(datasets,labels,colors):
        trained,random=erows(data,"true_label"),erows(data,"norm_matched_random")
        tv=[avg(trained,lambda m,k=k:m["delta_energy_in_base_entity_pcs"][str(k)]) for k in dims]
        rv=[avg(random,lambda m,k=k:m["delta_energy_in_base_entity_pcs"][str(k)]) for k in dims]
        ax.plot(dims,tv,marker="o",linewidth=2,color=color,label=f"{label} · trained")
        ax.plot(dims,rv,linestyle=":",linewidth=1.8,color=color,alpha=.6,label=f"{label} · random")
    ax.set_xlabel("Leading PCs of original entity cloud"); ax.set_ylabel("Fraction of Δeff energy in subspace")
    ax.legend(fontsize=7,ncol=2,loc="lower left",bbox_to_anchor=(0,1.01),borderaxespad=0)
    polish(ax); save(fig,out,"embedding_subspace")

    fig,ax=plt.subplots(figsize=(5.0,2.9)); conditions=[None,"true_label","shuffled_label","norm_matched_random"]
    names=["Original\nE₀","True-label\nE₀ + Δ","Shuffled-label\nE₀ + Δ","Random\nE₀ + Δ"]; x=np.arange(4)
    for data,label,color in zip(datasets,labels,colors):
        base=erows(data)[0]["source_geometry"]["base"]["same_target_auc"]
        values=[base]+[avg(erows(data,c),lambda m:m["source_geometry"]["adapted"]["same_target_auc"]) for c in conditions[1:]]
        ax.plot(x,values,marker="o",linewidth=2.1,color=color,label=label)
    ax.axhline(.5,color=GRID,linestyle=":"); ax.set_xticks(x,names); ax.set_ylim(.48,.61); ax.set_ylabel("Same-country pair AUC")
    ax.legend(ncol=2,loc="lower left",bbox_to_anchor=(0,1.01),borderaxespad=0)
    polish(ax); save(fig,out,"embedding_target_clustering")


def task_success(out, root, model, color, stem):
    root=Path(root); base=read(root/"baseline_summary.json")["overall"]["all"]
    trained=read(root/"summaries/correct_delta_aggregate.json")["all"]
    keys=["A_1a","A_1b","A_2"]
    labels=["Source to bridge","Bridge to answer","Held-out two hop"]
    original=np.asarray([base[k] for k in keys])
    adapted=np.asarray([trained[k]["mean"] for k in keys])
    y=np.arange(3)[::-1]
    fig,ax=plt.subplots(figsize=(5.0,2.55))
    for yy,left,right in zip(y,original,adapted):
        ax.plot([left,right],[yy,yy],color=GRID,linewidth=5,solid_capstyle="round",zorder=1)
    ax.scatter(original,y,s=55,color=GREY,label="Frozen",zorder=2)
    ax.scatter(adapted,y,s=72,color=color,label="With learned Δ",zorder=3)
    for yy,value in zip(y,original):
        if value < .15:
            ax.text(value+.012,yy+.17,f"{value:.1%}",ha="left",va="center",color=GREY,fontsize=7.5)
        else:
            ax.text(value-.018,yy,f"{value:.1%}",ha="right",va="center",color=GREY,fontsize=7.5)
    for yy,value in zip(y,adapted): ax.text(value+.018,yy,f"{value:.1%}",ha="left",va="center",color=color,fontsize=7.5,fontweight="bold")
    ax.set_yticks(y,labels); ax.set_xlim(0,1.04); ax.set_xlabel("Exact-match accuracy")
    ax.legend(ncol=2,loc="lower left",bbox_to_anchor=(0,1.01),borderaxespad=0)
    polish(ax,"x"); save(fig,out,stem)


def causal_identity(out):
    rows=read("outputs/bb53acb4d683/interventions/summary.json")
    lookup={r["intervention"]:r for r in rows}
    labels=["Residual\nremoved","Rows\npermuted","Correct entity\nassignment"]
    values=[lookup["delta_removal"]["A_2"],lookup["delta_permutation"]["A_2"],lookup["scale_1"]["A_2"]]
    fig,ax=plt.subplots(figsize=(4.5,2.55)); x=np.arange(3)
    bars=ax.bar(x,values,color=[GREY,"#9BA8AD",LLAMA],width=.58)
    for bar,value in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,value+.014,f"{value:.1%}",ha="center",fontsize=8,
                color=LLAMA if value==values[-1] else INK,fontweight="bold")
    ax.set_xticks(x,labels); ax.set_ylim(0,.35); ax.set_ylabel("Held-out two-hop accuracy")
    polish(ax); save(fig,out,"causal_entity_assignment")


def geodesic_structure(out):
    datasets=[read("outputs/c243073e2862/embedding_structure/metrics.json"),
              read("outputs/bb53acb4d683/embedding_structure/metrics.json")]
    if not all("base_geodesic" in erows(data)[0]["source_geometry"] for data in datasets): return
    names=["Original\nE₀","True-label\nE₀ + Δ","Shuffled-label\nE₀ + Δ","Random\nE₀ + Δ"]
    conditions=["true_label","shuffled_label","norm_matched_random"]
    fig,ax=plt.subplots(figsize=(5.0,2.9)); x=np.arange(4)
    for data,label,color in zip(datasets,["Qwen 2.5 7B","Llama 3.1 8B"],[QWEN,LLAMA]):
        base=erows(data)[0]["source_geometry"]["base_geodesic"]["same_target_auc"]
        vals=[base]+[avg(erows(data,c),lambda m:m["source_geometry"]["adapted_geodesic"]["same_target_auc"]) for c in conditions]
        ax.plot(x,vals,marker="o",linewidth=2.1,color=color,label=label)
    ax.axhline(.5,color=GRID,linestyle=":"); ax.set_xticks(x,names); ax.set_ylabel("Same-country geodesic AUC")
    ax.legend(ncol=2,loc="lower left",bbox_to_anchor=(0,1.01),borderaxespad=0)
    polish(ax); save(fig,out,"embedding_geodesic_structure")


def angular_geometry(out):
    datasets=[read("outputs/c243073e2862/embedding_structure/metrics.json"),
              read("outputs/bb53acb4d683/embedding_structure/metrics.json")]
    if not all("embedding_rotation_mean_deg" in erows(data)[0] for data in datasets): return
    labels=["True-label Δ","Shuffled-label Δ","Random Δ"]
    conditions=["true_label","shuffled_label","norm_matched_random"]
    fig,ax=plt.subplots(figsize=(4.8,2.8)); x=np.arange(3)
    for model_i,(data,label,color) in enumerate(zip(datasets,["Qwen 2.5 7B","Llama 3.1 8B"],[QWEN,LLAMA])):
        vals=[avg(erows(data,c),lambda m:m["embedding_rotation_mean_deg"]) for c in conditions]
        ax.plot(x,vals,marker="o",linewidth=2.1,color=color,label=label)
        for xx,value in zip(x,vals):
            ax.annotate(f"{value:.1f}°",(xx,value),xytext=(0,-14 if model_i==0 else 9),
                        textcoords="offset points",ha="center",fontsize=7,color=color,fontweight="bold")
    ax.set_xticks(x,labels); ax.set_ylabel("Mean rotation from E₀ to E₀ + Δ")
    ax.legend(ncol=2,loc="lower left",bbox_to_anchor=(0,1.01),borderaxespad=0)
    polish(ax); save(fig,out,"embedding_angular_rotation")

    names=["Original E₀","True-label","Shuffled-label","Random"]
    fig,ax=plt.subplots(figsize=(4.9,2.8)); x=np.arange(4)
    for data,label,color in zip(datasets,["Qwen 2.5 7B","Llama 3.1 8B"],[QWEN,LLAMA]):
        base=erows(data)[0]["cone_geometry"]["base"]["axis_angle_median_deg"]
        vals=[base]+[avg(erows(data,c),lambda m:m["cone_geometry"]["adapted"]["axis_angle_median_deg"]) for c in conditions]
        ax.plot(x,vals,marker="o",linewidth=2.1,color=color,label=label)
    ax.set_xticks(x,names); ax.set_ylabel("Median angle to cone axis")
    ax.legend(ncol=2,loc="lower left",bbox_to_anchor=(0,1.01),borderaxespad=0)
    polish(ax); save(fig,out,"embedding_cone_aperture")


def feature_cka_heatmaps(out):
    cmap=LinearSegmentedColormap.from_list("geometry_cka",["#F3F0EB","#65AEB4",INK])
    for root,model_stem in (("outputs/c243073e2862","qwen"),("outputs/bb53acb4d683","llama")):
        arrays=np.load(Path(root)/"layers/representations.npz")
        for site in ("entity","final"):
            layers=sorted(int(key.split("_")[1]) for key in arrays.files
                          if key.startswith(f"{site}_") and key.endswith("_h0"))
            # The unmodified final prompt token is constant at layer 0, so its
            # centered Gram matrix has zero norm and CKA is undefined.
            if site == "final": layers=[layer for layer in layers if layer > 0]
            matrix=np.asarray([[sample_cka(arrays[f"{site}_{left}_h0"],arrays[f"{site}_{right}_hd"])
                                for right in layers] for left in layers])
            # A deliberately shallow aspect ratio keeps the two-model CKA row
            # readable at workshop-paper width without wasting vertical space.
            fig,ax=plt.subplots(figsize=(2.25,1.55))
            image=ax.imshow(matrix,vmin=0,vmax=1,cmap=cmap,origin="lower",aspect="auto")
            ax.set_xticks(range(len(layers)),layers); ax.set_yticks(range(len(layers)),layers)
            ax.set_xlabel("Layer with learned Δ"); ax.set_ylabel("Frozen layer")
            for side in ("top","right","left","bottom"): ax.spines[side].set_visible(False)
            ax.tick_params(length=0,labelsize=6.5)
            cbar=fig.colorbar(image,ax=ax,fraction=.045,pad=.035); cbar.set_label("Linear CKA")
            cbar.outline.set_visible(False); cbar.ax.tick_params(length=0,labelsize=6.5)
            save(fig,out,f"{model_stem}_{site}_cross_layer_cka")


def layerwise(out):
    rows=read("outputs/bb53acb4d683/layers/metrics.json")
    fig,ax=plt.subplots(figsize=(4.7,2.65))
    for site,color,style_,label in (("entity",QWEN,"-","Entity span"),("final",LLAMA,"--","Final prompt token")):
        selected=sorted([r for r in rows if r["site"]==site],key=lambda r:r["layer"])
        ax.plot([r["layer"] for r in selected],[r["relative_change_magnitude"] for r in selected],
                marker="o",linewidth=2,color=color,linestyle=style_,label=label)
    ax.set_xlabel("Transformer layer"); ax.set_ylabel("Relative propagated change"); ax.legend(); polish(ax)
    save(fig,out,"layerwise_change")


def interventions(out):
    rows=read("outputs/bb53acb4d683/interventions/summary.json")
    scales=sorted([r for r in rows if r["intervention"].startswith("scale_")],key=lambda r:r["alpha"])
    fig,ax=plt.subplots(figsize=(4.7,2.65))
    for metric,label,color,marker in (("A_1a","Hop 1","#66A9AF","o"),("A_1b","Hop 2",QWEN,"s"),("A_2","Two hop",LLAMA,"D")):
        ax.plot([r["alpha"] for r in scales],[r[metric] for r in scales],marker=marker,linewidth=2,color=color,label=label)
    ax.axvline(1,color=GRID,linestyle=":"); ax.set_xlabel("Residual scale α"); ax.set_ylabel("Exact-match accuracy"); ax.set_ylim(0,1.02); ax.legend(ncol=3)
    polish(ax); save(fig,out,"residual_scaling")


def counterfactual_composition(out):
    root=Path("outputs/a441b75f4844")
    baseline_path=root/"baseline_summary.json"
    adapted_path=root/"summaries/correct_delta_seed-13.json"
    if not baseline_path.exists() or not adapted_path.exists(): return
    base=read(baseline_path)["overall"]["all"]
    adapted=read(adapted_path)["all"]
    original=np.asarray([base["A_1a"],base["A_1b"],0.0,base["A_2"]])
    changed=np.asarray([adapted["A_1a"],adapted["A_1b"],adapted["A_explicit"],adapted["A_2"]])
    labels=["Local fact 1","Local fact 2","Both facts","Direct 2-hop"]
    y=np.arange(len(labels))[::-1]
    fig,ax=plt.subplots(figsize=(3.55,1.72))
    for yy,left,right in zip(y,original,changed):
        ax.plot([left,right],[yy,yy],color=GRID,linewidth=3.5,solid_capstyle="round",zorder=1)
    ax.scatter(original,y,s=34,color=GREY,label="Frozen",zorder=2)
    ax.scatter(changed,y,s=43,color=LLAMA,label="Learned Δ",zorder=3)
    for index,(yy,left,right) in enumerate(zip(y,original,changed)):
        if index == 3:
            ax.annotate(f"{left:.1%}",(left,yy),xytext=(-2,7),textcoords="offset points",
                        ha="right",fontsize=6.5,color=GREY,fontweight="bold")
            ax.annotate(f"{right:.1%}",(right,yy),xytext=(3,-10),textcoords="offset points",
                        ha="left",fontsize=6.5,color=LLAMA,fontweight="bold")
        else:
            ax.text(left+.012,yy+.13,f"{left:.1%}",ha="left",va="center",fontsize=6.5,
                    color=GREY,fontweight="bold")
            ax.text(right+.012,yy,f"{right:.1%}",ha="left",va="center",fontsize=6.5,
                    color=LLAMA,fontweight="bold")
    ax.set_yticks(y,labels,fontsize=7); ax.set_xlim(-.015,.59); ax.set_xlabel("Exact-match accuracy")
    ax.legend(ncol=2,fontsize=6.8,loc="lower left",bbox_to_anchor=(0,1.0),borderaxespad=0,
              handlelength=1.1,columnspacing=.9)
    polish(ax,"x"); save(fig,out,"mquake_counterfactual_composition")


def targeted_swaps(out):
    specs=[("outputs/bb53acb4d683","SOCRATES"),("outputs/a441b75f4844","MQuAKE-CF")]
    values=[]
    for root,label in specs:
        root=Path(root); adapted=root/"summaries/correct_delta_seed-13.json"
        swap=root/"interventions/targeted_source_swap_summary.json"
        if not adapted.exists() or not swap.exists(): return
        before=read(adapted)["all"]["A_1a"]; changed=read(swap)
        values.append((label,before,changed["original_target_accuracy"]["A_1a"],
                       changed["donor_target_hop_1_accuracy"]))
    labels=["Correct residual\noriginal target","Swapped residual\noriginal target",
            "Swapped residual\ndonor target"]
    x=np.arange(3); width=.34
    fig,ax=plt.subplots(figsize=(5.2,2.8))
    for index,(dataset,*row) in enumerate(values):
        bars=ax.bar(x+(index-.5)*width,row,width,
                    color=[LLAMA,QWEN][index],alpha=.9,label=dataset)
        for bar,value in zip(bars,row):
            ax.text(bar.get_x()+bar.get_width()/2,value+.018,f"{value:.1%}",ha="center",
                    fontsize=7.2,color=[LLAMA,QWEN][index],fontweight="bold")
    ax.set_xticks(x,labels); ax.set_ylim(0,1.02); ax.set_ylabel("First-hop exact match")
    ax.legend(ncol=2,loc="lower left",bbox_to_anchor=(0,1.01),borderaxespad=0)
    polish(ax); save(fig,out,"targeted_residual_swaps")


def compact_result_row(out):
    roots=[Path("outputs/c243073e2862"),Path("outputs/bb53acb4d683")]
    if not all((root/"baseline_summary.json").exists() for root in roots): return
    labels=["Frozen","True Δ","Shuffle","Random"]
    fig,ax=plt.subplots(figsize=(2.35,1.82)); x=np.arange(4)
    for model_i,(root,name,color,marker) in enumerate(zip(
            roots,["Qwen","Llama"],[QWEN,LLAMA],["o","s"])):
        base=read(root/"baseline_summary.json")["overall"]["all"]["A_2"]
        values=[base]+[read(root/"summaries"/f"{condition}_aggregate.json")["all"]["A_2"]["mean"]
                           for condition in ("correct_delta","shuffled_delta","random_delta")]
        ax.plot(x,values,color=color,marker=marker,linewidth=1.8,markersize=4.5,label=name)
        for xx,value in zip(x[:2],values[:2]):
            ax.annotate(f"{value:.1%}",(xx,value),xytext=(0,5 if model_i else -10),
                        textcoords="offset points",ha="center",fontsize=6.3,
                        color=color,fontweight="bold")
    ax.set_xticks(x,labels,fontsize=6.5); ax.set_ylim(-.015,.37); ax.set_ylabel("Two-hop EM")
    ax.legend(ncol=2,fontsize=6.4,loc="lower left",bbox_to_anchor=(0,1.0),borderaxespad=0,
              handlelength=1.5,columnspacing=.8)
    polish(ax); save(fig,out,"socrates_transfer_compact")

    specs=[("outputs/bb53acb4d683","SOCRATES",LLAMA),("outputs/a441b75f4844","MQuAKE-CF",QWEN)]
    fig,ax=plt.subplots(figsize=(2.35,1.82)); x=np.arange(3)
    for root,label,color in specs:
        root=Path(root); adapted=root/"summaries/correct_delta_seed-13.json"
        swap=root/"interventions/targeted_source_swap_summary.json"
        if not adapted.exists() or not swap.exists(): plt.close(fig); return
        changed=read(swap); row=[read(adapted)["all"]["A_1a"],
            changed["original_target_accuracy"]["A_1a"],changed["donor_target_hop_1_accuracy"]]
        ax.plot(x,row,marker="o",linewidth=2,color=color,label=label)
    ax.set_xticks(x,["Correct","Swap\nold","Swap\ndonor"],fontsize=6.5)
    ax.set_ylim(-.02,1.02); ax.set_ylabel("First-hop EM")
    ax.legend(ncol=2,fontsize=6.4,loc="lower left",bbox_to_anchor=(0,1.0),borderaxespad=0,
              handlelength=1.5,columnspacing=.8)
    polish(ax); save(fig,out,"targeted_swaps_compact")

    clutrr=Path("outputs/clutrr/results.json")
    if clutrr.exists():
        result=read(clutrr)["conditions"]
        original=[result["original_long_validation"]["accuracy"],
                  result["original_long_test"]["accuracy"]]
        adapted=[result["role_delta_long_validation"]["accuracy"],
                 result["role_delta_long_test"]["accuracy"]]
        x=np.arange(2); width=.34
        fig,ax=plt.subplots(figsize=(2.35,1.82))
        ax.bar(x-width/2,original,width,color=GREY,label="Frozen")
        ax.bar(x+width/2,adapted,width,color=LLAMA,label="Role Δ")
        for xx,left,right in zip(x,original,adapted):
            ax.text(xx-width/2,left+.014,f"{left:.1%}",ha="center",fontsize=7,color=INK)
            ax.text(xx+width/2,right+.014,f"{right:.1%}",ha="center",fontsize=7,
                    color=LLAMA,fontweight="bold")
        ax.set_xticks(x,["Long val.","Long test"],fontsize=6.5)
        ax.set_ylim(0,.48); ax.set_ylabel("Relation EM")
        ax.legend(ncol=2,fontsize=6.4,loc="lower left",bbox_to_anchor=(0,1.0),borderaxespad=0,
                  handlelength=1.5,columnspacing=.8)
        polish(ax); save(fig,out,"clutrr_length_generalization_compact")

    fig,ax=plt.subplots(figsize=(2.35,1.82))
    for root,name,color,marker in zip(roots,["Qwen","Llama"],
                                      [QWEN,LLAMA],["o","s"]):
        rows=read(root/"layers/metrics.json")
        selected=sorted((row for row in rows if row["site"]=="final" and
                         np.isfinite(row["preservation_cka"])),key=lambda row:row["layer"])
        layers=[row["layer"] for row in selected]
        values=[row["preservation_cka"] for row in selected]
        ax.plot(layers,values,color=color,marker=marker,linewidth=1.8,markersize=4,label=name)
    ax.set_xlim(4,max(layers)); ax.set_ylim(.58,1.025)
    ax.set_xticks([4,8,16,24,max(layers)]); ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Final-token CKA")
    ax.legend(ncol=2,fontsize=6.4,loc="lower left",bbox_to_anchor=(0,1.0),borderaxespad=0,
              handlelength=1.5,columnspacing=.8)
    polish(ax); save(fig,out,"layerwise_cka_compact")


def compact_angle_wrap(out):
    datasets=[read("outputs/c243073e2862/embedding_structure/metrics.json"),
              read("outputs/bb53acb4d683/embedding_structure/metrics.json")]
    conditions=["true_label","shuffled_label","norm_matched_random"]
    fig,ax=plt.subplots(figsize=(3.0,1.72)); x=np.arange(3)
    for model_i,(data,label,color,marker) in enumerate(zip(
            datasets,["Qwen","Llama"],[QWEN,LLAMA],["o","s"])):
        values=[avg(erows(data,c),lambda m:m["embedding_rotation_mean_deg"]) for c in conditions]
        ax.plot(x,values,marker=marker,linewidth=1.8,markersize=4.5,color=color,label=label)
        for xx,value in zip(x,values):
            ax.annotate(f"{value:.1f}°",(xx,value),xytext=(0,-10 if model_i==0 else 5),
                        textcoords="offset points",ha="center",fontsize=6.2,
                        color=color,fontweight="bold")
    ax.set_xticks(x,["True Δ","Shuffle","Random"],fontsize=6.7)
    ax.set_ylim(13,31); ax.set_ylabel("Mean embedding rotation")
    ax.legend(ncol=2,fontsize=6.4,loc="lower left",bbox_to_anchor=(0,1.0),
              borderaxespad=0,handlelength=1.4,columnspacing=.8)
    polish(ax); save(fig,out,"geometry_angles_wrap")


def main():
    style(); out=Path("outputs/paper_figures"); out.mkdir(parents=True,exist_ok=True)
    behavior(out); embedding_figures(out); layerwise(out); interventions(out)
    task_success(out,"outputs/bb53acb4d683","Llama 3.1 8B",LLAMA,"llama_task_success")
    task_success(out,"outputs/c243073e2862","Qwen 2.5 7B",QWEN,"qwen_task_success")
    causal_identity(out); geodesic_structure(out); angular_geometry(out); feature_cka_heatmaps(out)
    counterfactual_composition(out); targeted_swaps(out); compact_result_row(out)
    compact_angle_wrap(out)
    for root,stem in (("outputs/bb53acb4d683","llama_embedding_umap"),
                      ("outputs/c243073e2862","qwen_embedding_umap")):
        for suffix in ("pdf","png"):
            source=Path(root)/"embedding_structure"/f"embedding_umap.{suffix}"
            if source.exists(): copyfile(source,out/f"{stem}.{suffix}")
    print(f"Standalone title-free figures: {out}")


if __name__=="__main__": main()
